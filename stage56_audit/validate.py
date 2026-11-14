from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from rules3_stage23.evaluate import confusion,macro
import final_rules3
B=Path('/home/ubuntu/leewon/abc_runs/current_query_v1');O=R/'stage56_audit'
d=pd.read_csv(R/'error_audit_full/row_diagnostics.csv')
per=[];bounds=[];summary=[];changed=[]
def score(y,p):return f1_score(y,p,labels=range(1,7),average='macro',zero_division=0)
for split,g in d.groupby('split',sort=False):
 g=g.reset_index(drop=True);base=g.baseline.to_numpy(int);y=g.label.to_numpy(int)
 probs=[]
 for seed in [42,43,44]:
  s=pd.read_csv(B/f'seed_{seed}/B/{split}_predictions.csv').set_index('sample_id')
  assert s.index.is_unique
  a=s.loc[g.sample_id,[f'p{i}' for i in range(1,7)]].to_numpy();assert np.isfinite(a).all();probs.append(a)
 lp=np.log(np.clip(np.stack(probs),1e-12,1)).mean(0)
 q=base.copy();gate=np.isin(base,[5,6]);q[gate]=5+(lp[gate,5]>lp[gate,4])
 assert np.array_equal(q[~gate],base[~gate])
 z=g.loc[q!=base,['sample_id','experiment_id','relative_time_s','label','baseline']].copy();z['candidate']=q[q!=base];z['split']=split;changed.append(z)
 for exp,ix in g.groupby('experiment_id').indices.items():
  yy=y[ix];bb=base[ix];pp=q[ix]
  per.append(dict(split=split,experiment_id=exp,rows=len(ix),corrected=int(((bb!=yy)&(pp==yy)).sum()),harmed=int(((bb==yy)&(pp!=yy)).sum()),macro_delta=score(yy,pp)-score(yy,bb)))
  order=np.argsort(g.iloc[ix].relative_time_s.to_numpy(),kind='stable');ii=ix[order];t=g.iloc[ii].relative_time_s.to_numpy();yy=y[ii]
  for j in np.flatnonzero((yy[:-1]==5)&(yy[1:]==6))+1:
   rec=dict(split=split,experiment_id=exp,onset_s=float(t[j]))
   # First 6 after last actual non-5/6: diagnostic only, not inference input.
   prior=np.flatnonzero(~np.isin(yy[:j],[5,6]));start=prior[-1]+1 if len(prior) else 0
   for name,p in [('baseline',base[ii]),('geometric',q[ii])]:
    hits=np.flatnonzero(p[start:]==6)+start
    rec[name+'_first6_offset_s']=float(t[hits[0]]-t[j]) if len(hits) else np.nan
    rec[name+'_early6_60s']=int(((t>=t[j]-60)&(t<t[j])&(yy==5)&(p==6)).sum())
    rec[name+'_miss6_60s']=int(((t>=t[j])&(t<=t[j]+60)&(yy==6)&(p!=6)).sum())
   bounds.append(rec)
 for scope,mask in [('all',np.ones(len(g),bool)),('noE16',~g.experiment_id.eq(final_rules3.E16).to_numpy())]:
  h=g.loc[mask].reset_index(drop=True);yy=y[mask];bb=base[mask];pp=q[mask]
  groups=list(h.groupby('experiment_id').indices.values());a=np.stack([confusion(yy[i],bb[i]) for i in groups]);b=np.stack([confusion(yy[i],pp[i]) for i in groups]);rng=np.random.default_rng(20261115);draw=rng.integers(0,len(groups),(4000,len(groups)));delta=macro(b[draw].sum(1))-macro(a[draw].sum(1))
  loo=[score(np.delete(yy,i),np.delete(pp,i))-score(np.delete(yy,i),np.delete(bb,i)) for i in groups]
  f0=f1_score(yy,bb,labels=range(1,7),average=None);f1=f1_score(yy,pp,labels=range(1,7),average=None)
  summary.append(dict(split=split,scope=scope,macro_base=score(yy,bb),macro_candidate=score(yy,pp),ci_low=np.quantile(delta,.025),ci_high=np.quantile(delta,.975),leave_one_out_min=min(loo),leave_one_out_max=max(loo),F5_base=f0[4],F5_candidate=f1[4],F6_base=f0[5],F6_candidate=f1[5]))
for name,table in [('per_experiment',per),('boundaries',bounds),('validation_summary',summary)]:pd.DataFrame(table).to_csv(O/f'{name}.csv',index=False)
pd.concat(changed).to_csv(O/'changed_rows.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
p=pd.DataFrame(per)
for split,h in p.groupby('split'):
 print(split,'net improved/harmed/unchanged',int((h.corrected>h.harmed).sum()),int((h.corrected<h.harmed).sum()),int((h.corrected==h.harmed).sum()))
 print(h[h.corrected+h.harmed>0].to_string(index=False))
b=pd.DataFrame(bounds)
print(b.groupby('split').sum(numeric_only=True).to_string())
