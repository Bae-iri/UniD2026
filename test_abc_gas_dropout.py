"""Synthetic-only tests: model equivalence, dropout paths and short training I/O."""
import hashlib,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from abc_reference import abc_fusion_experiment as abc
from compare_abc_gas_dropout import GasDropoutB,run

torch.set_num_threads(1)
x=(torch.rand(3,3,32,32),torch.ones(3,1,32,32),torch.randn(3,12),torch.randn(3,11,2))
b=abc.FusionModel('B',d=16).eval();m=GasDropoutB(1.,42,d=16).eval();m.load_state_dict(b.state_dict())
assert torch.equal(m(*x),b(*x))
f=torch.randn(3,80);m.train();z=m.drop_gas(m.head,(f,))[0]
assert torch.equal(z[:,:16],f[:,:16]) and torch.equal(z[:,32:48],f[:,32:48]) and torch.equal(z[:,64:],f[:,64:])
assert z[:,16:32].abs().sum()==0 and z[:,48:64].abs().sum()==0
with tempfile.TemporaryDirectory(prefix='synthetic-abc-gas-') as tmp:
 root=Path(tmp);base=root/'base';base.mkdir();data=root/'data';rng=np.random.default_rng(42)
 for split,groups in [('train',10),('validation',2)]:
  folder=data/split;folder.mkdir(parents=True);n=groups*6
  f=pd.DataFrame({c:rng.uniform(1,2,n) for c in abc.NUMERIC})
  f['sample_id']=[f'{split}_{i}' for i in range(n)]
  f['experiment_id']=[f'{split}_exp_{i//6}' for i in range(n)]
  f['label']=np.tile(np.arange(1,7),groups)
  for c in ['bat_class','tr_method','soc_status']:f[c]=1
  f['bat_cc']=1.;f.to_csv(folder/f'{split}.csv',index=False)
 frames=abc.common.split_frames(data,20261114,.2)
 prep=abc.SensorPreprocessor(frames['fit'])
 (base/'preprocessing.json').write_text(json.dumps(prep.metadata()))
 cfg={'split_seed':20261114,'csv_sha256':{split:hashlib.sha256((data/split/f'{split}.csv').read_bytes()).hexdigest() for split in ['train','validation']}}
 (base/'config.json').write_text(json.dumps(cfg));manifest={}
 checkpoint=base/'seed_42/B';checkpoint.mkdir(parents=True)
 for split,f in frames.items():
  packed=rng.integers(0,256,(len(f),4,128,128),dtype=np.uint8);packed[:,3]=1
  path=base/f'{split}.npy';np.save(path,packed);manifest[split]=str(path)
  if split in ['dev','validation']:f[['sample_id','label']].to_csv(checkpoint/f'{split}_predictions.csv',index=False)
 (base/'image_cache_manifest.json').write_text(json.dumps(manifest))
 torch.save({'state':abc.FusionModel('B').state_dict()},checkpoint/'best.pt')
 args=SimpleNamespace(base_run=base,data_dir=data,output=root/'out',epochs=1,seeds=[42],drop_probability=.2,device='cpu')
 run(args)
 s=pd.read_csv(args.output/'summary.csv');assert len(s)==2
 assert (args.output/'B_gas_dropout_seed42/validation_aggregates.json').exists()
 try:run(args)
 except FileExistsError:pass
 else:raise AssertionError('Output overwritten')
print('Synthetic equivalence, masking, two-arm training, checkpoint, aggregate output and overwrite tests passed.')
