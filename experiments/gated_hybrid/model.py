"""Probability-level gated fusion with coarse-to-fine and stage-4/5 supervision."""
import torch
from torch import nn
from torch.nn import functional as F

GROUPS=[0,0,0,1,1,2]


def hierarchy_prob(group,early,pair):
    return torch.cat([group[:,:1]*early,group[:,1:2]*pair,group[:,2:3]],1)


def group_prob(p): return torch.stack([p[:,:3].sum(1),p[:,3:5].sum(1),p[:,5]],1)


def relation_loss(p,y,weights):
    primary=F.nll_loss(p.clamp_min(1e-8).log(),y,weight=weights)
    target=torch.as_tensor(GROUPS,device=y.device)[y]
    coarse=F.nll_loss(group_prob(p).clamp_min(1e-8).log(),target)
    select=(y==3)|(y==4)
    pair=p[select,3:5]; pair=pair/pair.sum(1,keepdim=True).clamp_min(1e-8)
    pair_loss=F.nll_loss(pair.clamp_min(1e-8).log(),y[select]-3) if select.any() else p.sum()*0
    thresholds=torch.arange(5,device=y.device)[None]
    cumulative=p.cumsum(1)[:,:5]
    ordinal=F.binary_cross_entropy(cumulative.clamp(1e-6,1-1e-6),(y[:,None]<=thresholds).float())
    return primary+.2*coarse+.4*pair_loss+.1*ordinal


class ImageHead(nn.Module):
    def __init__(self,d=768):
        super().__init__()
        self.trunk=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,256),nn.GELU(),nn.Dropout(.25))
        self.flat=nn.Linear(256,6); self.group=nn.Linear(256,3)
        self.early=nn.Linear(256,3); self.pair=nn.Linear(256,2)

    def forward(self,x):
        z=self.trunk(x); flat=self.flat(z).softmax(1)
        h=hierarchy_prob(self.group(z).softmax(1),self.early(z).softmax(1),self.pair(z).softmax(1))
        return .7*flat+.3*h


class GatedHybrid(nn.Module):
    def __init__(self,sensor_dim,image_dim=768):
        super().__init__()
        self.sensor=nn.Sequential(nn.Linear(sensor_dim,64),nn.LayerNorm(64),nn.GELU())
        self.image=nn.Sequential(nn.LayerNorm(image_dim),nn.Linear(image_dim,64),nn.GELU())
        self.gate=nn.Sequential(nn.Linear(64+64+12+4,64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,1))
        nn.init.zeros_(self.gate[-1].weight); nn.init.zeros_(self.gate[-1].bias)

    def forward(self,sensor,image,ps,pi):
        stats=[]
        for p in [ps,pi]:
            top=p.topk(2,dim=1).values
            stats.extend([-(p*p.clamp_min(1e-8).log()).sum(1,keepdim=True),top[:,:1]-top[:,1:2]])
        alpha=self.gate(torch.cat([self.sensor(sensor),self.image(image),ps,pi,*stats],1)).sigmoid()
        return alpha*ps+(1-alpha)*pi,alpha
