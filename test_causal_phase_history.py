"""Synthetic causality and experiment-separation tests only."""
import numpy as np
import pandas as pd
from probe_causal_phase_history import history,SIGNALS
f=pd.DataFrame({'experiment_id':['A','A','A','A','B'],'relative_time_s':[20,0,10,40,0],
'meas_v':[3.,4.,2.,1.,9.], 'meas_t1':[100.,50.,200.,999.,888.],
'meas_t2':[80.,40.,160.,999.,888.], 'thermal_t':[70.,30.,150.,999.,888.]})
h=history(f,30)
assert np.allclose(h[0,16:20],[200,-100,10,1])
assert np.allclose(h[0,12:16],[2,1,10,1])
prefix=f[(f.experiment_id=='A')&(f.relative_time_s<=20)].reset_index(drop=True)
assert np.allclose(history(prefix,30)[0],h[0])
f.loc[3,SIGNALS]=1e8;assert np.allclose(history(f,30)[0],h[0])
first=h[1];assert np.allclose(first[:12],0)
f.loc[0,'meas_t1']=np.nan;assert np.isfinite(history(f,30)).all()
try:history(pd.concat([f,f.iloc[[0]]],ignore_index=True),30)
except ValueError:pass
else:raise AssertionError('Duplicate time accepted')
print('Synthetic phase history tests passed: exact extrema, past-only slope, future invariance, separation, missing values, duplicate rejection.')
