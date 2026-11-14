"""Synthetic-only validation of metrics, ID alignment and mismatch rejection."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from compare_abc_experiment_errors import load_aligned
from analyze_experiment_performance import confusion,scores

y=[1,1,2,3,5,6];pred=[1,2,2,3,4,6]
assert np.isclose(scores(confusion(y,pred))['macro_f1_6'],f1_score(y,pred,labels=range(1,7),average='macro',zero_division=0))
with tempfile.TemporaryDirectory() as t:
 p=Path(t)/'synthetic.csv'
 f=pd.DataFrame({'sample_id':['a','b','c'],'experiment_id':['x','x','y'],'label':[1,2,3],'pred':[1,2,3]})
 f.iloc[::-1].to_csv(p,index=False)
 aligned=load_aligned(p,f)
 assert aligned.sample_id.tolist()==f.sample_id.tolist()
 bad=f.copy();bad.loc[0,'label']=2;bad.to_csv(p,index=False)
 try:load_aligned(p,f)
 except ValueError:pass
 else:raise AssertionError('Mismatched targets accepted')
print('Synthetic metric/alignment/mismatch tests passed.')
