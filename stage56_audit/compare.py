from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
R=Path(__file__).resolve().parents[1]
B=Path('/home/ubuntu/leewon/abc_runs/current_query_v1')
d=pd.read_csv(R/'error_audit_full/row_diagnostics.csv')
rows=[]
for split,g in d.groupby('split',sort=False):
 g=g.reset_index(drop=True); base=g.baseline.to_numpy(int); y=g.label.to_numpy(int)
 probs=np.stack([pd.read_csv(B/f'seed_{s}/B/{split}_predictions.csv').set_index('sample_id').loc[g.sample_id,[f'p{i}' for i in range(1,7)]].to_numpy() for s in [42,43,44]])
 variants={'baseline':base.copy()}
 for name,p in [('median',np.median(probs,axis=0)),('geometric',np.exp(np.log(np.clip(probs,1e-12,1)).mean(0)))]:
  q=base.copy();gate=np.isin(base,[5,6]);q[gate]=5+(p[gate,5]>p[gate,4]);variants[name]=q
 for seed,p in zip([42,43,44],probs):
  q=base.copy();gate=np.isin(base,[5,6]);q[gate]=5+(p[gate,5]>p[gate,4]);variants[f'seed{seed}_pair']=q
 for name,q in variants.items():
  for scope,m in [('all',np.ones(len(g),bool)),('noE16',~g.experiment_id.eq('E16').to_numpy())]:
   if scope=='noE16':
    import sys
    sys.path.insert(0,str(R));import final_rules3
    m=~g.experiment_id.eq(final_rules3.E16).to_numpy()
   yy=y[m];pp=q[m];bb=base[m]
   rows.append(dict(split=split,scope=scope,method=name,macro_f1=f1_score(yy,pp,labels=range(1,7),average='macro'),corrected=int(((bb!=yy)&(pp==yy)).sum()),harmed=int(((bb==yy)&(pp!=yy)).sum()),five_to_six=int(((yy==5)&(pp==6)).sum()),six_to_five=int(((yy==6)&(pp==5)).sum())))
pd.DataFrame(rows).to_csv(R/'stage56_audit/scores.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
