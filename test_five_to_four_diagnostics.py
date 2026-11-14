"""Synthetic-only invariants for frozen path interventions."""
from types import SimpleNamespace
import torch
from abc_reference import abc_fusion_experiment as abc
from investigate_five_to_four import head_variant,attention_summary,MODES

torch.set_num_threads(1);torch.manual_seed(42)
m=abc.FusionModel('B',d=16).eval();capture={}
handles=[m.head.register_forward_pre_hook(lambda mod,args:capture.update(f=args[0])),
 m.gas_readout.register_forward_pre_hook(lambda mod,args:capture.update(g=args)),
 m.image_readout.register_forward_pre_hook(lambda mod,args:capture.update(i=args))]
gas=torch.rand(3,11,2);gas[:,:,1]=0
batch=(torch.rand(3,3,32,32),torch.ones(3,1,32,32),torch.rand(3,12),gas)
with torch.no_grad():
 original=m(*batch);f=capture['f'].clone();ga=capture['g'];ia=capture['i'];saved=f.clone()
 assert torch.equal(head_variant(m,f,'normal',ga,ia,gas),original)
 for mode in MODES:
  out=head_variant(m,f,mode,ga,ia,gas)
  assert out.shape==(3,6) and torch.isfinite(out).all() and torch.equal(f,saved)
 for module,args in [(m.gas_readout,ga),(m.image_readout,ia)]:
  a=attention_summary(module,args);assert ((a[:,0]>=-1e-6)&(a[:,0]<=1+1e-6)).all()
 # A linear classifier must have zero gas/image factorial interaction.
 linear=SimpleNamespace(head=torch.nn.Linear(80,6));x=torch.randn(3,80)
 y={mode:head_variant(linear,x,mode,None,None,None) for mode in ['normal','gas_off','image_off','gas_image_off']}
 assert torch.allclose(y['normal']-y['gas_off']-y['image_off']+y['gas_image_off'],torch.zeros(3,6),atol=1e-6)
for h in handles:h.remove()
print('Synthetic tests passed: original equivalence, finite interventions, no mutation, normalized entropy, factorial identity.')
