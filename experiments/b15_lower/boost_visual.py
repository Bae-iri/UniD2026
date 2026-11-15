"""Fixed ImageNet encoder: current image only, never fit on this competition."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models import resnet18
from reproduce import sha


class Encoder:
    def __init__(self,weights,device='cuda'):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        self.device=torch.device(device);self.model=resnet18(weights=None).to(self.device).eval()
        self.model.load_state_dict(torch.load(weights,map_location=self.device,weights_only=True))
        self.model.requires_grad_(False)
        self.mean=torch.tensor([.485,.456,.406],device=self.device)[None,:,None,None]
        self.std=torch.tensor([.229,.224,.225],device=self.device)[None,:,None,None]

    @torch.inference_mode()
    def __call__(self,packed):
        n=len(packed);out=[]
        # A fixed single-image position makes numerical results independent of
        # prefix length, row order, neighboring rows and batch placement.
        for start in range(n):
            a=np.asarray(packed[start:start+1]);size=len(a)
            p=np.zeros((1,4,128,128),np.uint8);p[:size]=a
            p=torch.from_numpy(p).to(self.device);x=(p[:,:3].float()/255.-self.mean)/self.std
            m=self.model;x=m.maxpool(m.relu(m.bn1(m.conv1(x))))
            x=m.layer4(m.layer3(m.layer2(m.layer1(x))))
            mask=F.adaptive_avg_pool2d(p[:,3:4].float(),x.shape[-2:])
            out.append(((x*mask).sum((2,3))/mask.sum((2,3)).clamp_min(1.))[:size].cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def main(args):
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    args.output.mkdir(parents=True,exist_ok=False)
    packed=np.load(args.source/'images.npy',mmap_mode='r');encoder=Encoder(args.weights,args.device)
    out=np.lib.format.open_memmap(args.output/'visual.npy',mode='w+',dtype=np.float32,shape=(len(packed),512))
    for start in range(0,len(packed),2048):
        out[start:start+2048]=encoder(packed[start:start+2048]);out.flush()
        print('Encoded',min(start+2048,len(packed)),'/',len(packed),flush=True)
    sample=np.array([0,128,1000,4000])
    np.testing.assert_allclose(encoder(packed[sample]),out[sample],rtol=1e-5,atol=1e-5)
    (args.output/'protocol.json').write_text(json.dumps(dict(weights=str(args.weights),weights_sha256=sha(args.weights),
        encoder='ImageNet ResNet18 frozen; current-image masked average layer4; fixed1 batch; TF32 disabled',
        competition_labels_used_for_encoder=False,source_sha256=sha(Path(__file__))),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','weights','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda');main(p.parse_args())
