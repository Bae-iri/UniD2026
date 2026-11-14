"""Synthetic-only sensor event and causality checks."""
import numpy as np
import pandas as pd
from abc_reference import abc_fusion_experiment as abc
from analyze_sensor_events import events
f=pd.DataFrame({'experiment_id':['A']*4,'relative_time_s':[0,5,10,15],'meas_v':[4.,3.,4.,4.]})
for c in abc.GAS:f[c]=1.
for c in abc.GAS:
 if c!='meas_p':f.loc[[1,2],c]=0.
e=events(f,-.048,.034)
assert e['voltage_drop_tail'].tolist()==[False,True,False,False]
assert e['voltage_rebound'].tolist()==[False,False,True,False]
assert e['gas_all_zero'].tolist()==[False,True,True,False]
assert e['gas_zero_onset'].tolist()==[False,True,False,False]
assert e['gas_zero_recovery'].tolist()==[False,False,False,True]
assert e['gas_zero_age'][2]==5
prefix=events(f.iloc[:3],-.048,.034)
for k in prefix:np.testing.assert_allclose(prefix[k],e[k][:3],equal_nan=True)
f.loc[3,'meas_v']=999.
after=events(f,-.048,.034)
np.testing.assert_array_equal(after['voltage_rebound'][:3],e['voltage_rebound'][:3])
print('Synthetic tests passed: pressure exclusion, drop/rebound, zero transitions, observed age and future invariance.')
