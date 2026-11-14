"""Synthetic tests only. Run: python test_compare_gate_branches.py"""
import sys, tempfile
from pathlib import Path
from types import SimpleNamespace
import compare_gate_branches as m
import numpy as np
import pandas as pd
from PIL import Image
m.torch.set_num_threads(1)
with tempfile.TemporaryDirectory(prefix='b-gate-synthetic-') as t:
 root=Path(t)
 for split in ['train','validation','test']:
  folder=root/split;folder.mkdir()
  f=pd.DataFrame({c:np.arange(6,dtype=float)+1 for c in m.NUMERIC})
  f['sample_id']=[f'{split}_{i}' for i in range(6)]
  f['experiment_id']=split
  for c,_ in m.CATS:f[c]=1
  f['image_path']=[f'{i}.png' for i in range(6)]
  if split!='test':f['label']=np.arange(6)+1
  for i in range(6):Image.new('RGB',(25,30),(i*30,60,90)).save(folder/f'{i}.png')
  f.to_csv(folder/f'{split}.csv',index=False)
 args=SimpleNamespace(output=root/'runs',data_dir=root,image_size=32,dim=16,seeds=[42],epochs=1,lr=3e-4,device='cpu',batch_size=3,workers=0,modes=['B','B_image_gate','B_gas_gate','B_gate'])
 m.train(args)
 assert (args.output/'paired_differences.csv').exists()
 template=root/'template.csv'
 order=[f'test_{i}' for i in reversed(range(6))]
 pd.DataFrame({'sample_id':order,'label':1}).to_csv(template,index=False)
 for mode in args.modes:
  p=SimpleNamespace(checkpoint=args.output/f'{mode}_seed42/best.pt',input=root/'test/test.csv',image_root=root/'test',device='cpu',batch_size=3,workers=0,sample_submission=template,output=root/f'{mode}.csv')
  m.predict(p)
  assert pd.read_csv(p.output).sample_id.tolist()==order
 try:m.train(args)
 except FileExistsError:pass
 else:raise AssertionError('overwrite not rejected')
m.smoke_test()
print('Synthetic integration passed; no competition data read.')
