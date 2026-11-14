"""Synthetic-only causal GRU, masks, feature ablations and selective correction checks."""
import numpy as np,pandas as pd,torch
from select_event_temporal_correction import Correction,groups,raw_features,routing,combine,sensor_mask,batch_sequences
from abc_reference import abc_fusion_experiment as abc
torch.set_num_threads(1);torch.manual_seed(42)
f=pd.DataFrame({c:[1.,2.,3.,4.] for c in abc.NUMERIC})
f['experiment_id']=['A']*4;f['relative_time_s']=[0,5,10,15];f['meas_v']=[4.,3.,4.,4.]
for c in abc.GAS:
 if c!='meas_p':f.loc[[1,2],c]=0.
x=raw_features(f,-.048,.034)
np.testing.assert_allclose(x[:3],raw_features(f.iloc[:3],-.048,.034),equal_nan=True)
shuffled=f.iloc[[3,1,0,2]].reset_index(drop=True);ids=groups(shuffled)[0]
assert shuffled.relative_time_s.to_numpy()[ids].tolist()==[0,5,10,15]
r=routing(f,-.048,.034);assert r['gas_zero'].tolist()==[False,True,True,False]
z=np.ones((4,6),np.float32)/6
p=z.copy();p[:,4]=.6;p=p/p.sum(1,keepdims=True)
a=combine(z,p,r['gas_zero'],.5);np.testing.assert_array_equal(a[[0,3]],z[[0,3]])
assert np.allclose(a.sum(1),1)
features=np.ones((3,x.shape[1]),np.float32);ns=len(abc.NUMERIC)
no_gas=sensor_mask(features,'no_gas');assert not no_gas[:,6:ns].any()
assert no_gas[:,0].all();no_v=sensor_mask(features,'no_voltage');assert not no_v[:,0].any() and no_v[:,6:ns].all()
seq=torch.randn(1,7,12);seq[:,:,0:6]=seq[:,:,0:6].log_softmax(-1)
m=Correction(12).eval()
with torch.no_grad():
 torch.testing.assert_close(m(seq)[:,:4],m(seq[:,:4]),atol=1e-6,rtol=1e-5)
 torch.testing.assert_close(m(seq),seq[:,:,:6])
 # Repeat future invariance after nonzero correction weights.
 m.out.weight.normal_();m.out.bias.normal_()
 torch.testing.assert_close(m(seq)[:,:4],m(seq[:,:4]),atol=1e-6,rtol=1e-5)
xb,yb=batch_sequences([np.arange(2),np.arange(2,4)],np.zeros((4,12),np.float32),np.array([1,-100,5,6]),'cpu')
yb[yb==-101]=-100
assert yb.tolist()==[[0,-100],[4,5]]
print('Synthetic checks passed: causal features/GRU, sorting, event masks, selective blending, input ablations and masked labels.')
