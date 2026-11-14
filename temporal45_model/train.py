"""Fit-only 4/5 specialist with matched causal time ablations; no Test access."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from compare_abc_phase_history import causal_features
import final_rules3
B=Path('/home/ubuntu/leewon/abc_runs/current_query_v1');D=Path('/home/ubuntu/data/battery_data_light');O=R/'temporal45_model/run_v1'

def main():
 O.mkdir(exist_ok=False)
 manifest=pd.read_csv(B/'split_manifest.csv');train=pd.read_csv(D/'train/train.csv');val=pd.read_csv(D/'validation/validation.csv')
 frames={s:manifest.loc[manifest.split==s,['sample_id']].merge(train,on='sample_id',validate='1:1') for s in ['fit','dev']};frames['validation']=val
 assert not set(frames['fit'].experiment_id)&set(frames['dev'].experiment_id)
 numeric=[c for c in train if c.startswith('meas_') or c=='thermal_t']
 xs={};ys={};gs={}
 for s,f in frames.items():
  h=causal_features(f);x=f[numeric].apply(pd.to_numeric,errors='coerce').to_numpy(float);x[~np.isfinite(x)]=np.nan
  noage=[j for j in range(1,17) if j not in [3,7,11,15]]
  xs[s]={'current':x,'elapsed':np.c_[x,h[:,0]],'extrema':np.c_[x,h[:,noage]],'history_time':np.c_[x,h]};ys[s]=f.label.to_numpy(int);gs[s]=f.experiment_id.to_numpy()
 diag=pd.read_csv(R/'error_audit_full/row_diagnostics.csv');rows=[];cv=[];per=[]
 for mode in xs['fit']:
  mask=np.isin(ys['fit'],[4,5]);x=xs['fit'][mode][mask];y=ys['fit'][mask];groups=gs['fit'][mask]
  def model():return HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=42)
  def weights(a):
   counts=pd.Series(a).value_counts();return np.array([1/counts[z] for z in a])*len(a)/len(counts)
  oof=np.zeros(len(y),int)
  for fold,(a,b) in enumerate(GroupKFold(5).split(x,y,groups)):
   m=model();m.fit(x[a],y[a],sample_weight=weights(groups[a]));oof[b]=m.predict(x[b]);cv.append(dict(mode=mode,fold=fold,macro45=f1_score(y[b],oof[b],labels=[4,5],average='macro')))
  m=model();m.fit(x,y,sample_weight=weights(groups));joblib.dump(dict(model=m,mode=mode,current_columns=numeric,history_source='compare_abc_phase_history.causal_features',fit_experiments=sorted(set(groups))),O/f'{mode}.joblib')
  for split in ['dev','validation']:
   f=frames[split];ref=diag[diag.split==split].set_index('sample_id').loc[f.sample_id];assert np.array_equal(ref.label,ys[split]);base=ref.baseline.to_numpy(int);q=base.copy();gate=np.isin(base,[4,5]);q[gate]=m.predict(xs[split][mode][gate]);yy=ys[split]
   pd.DataFrame(dict(sample_id=f.sample_id,label=yy,baseline=base,pred=q,p5=m.predict_proba(xs[split][mode])[:,list(m.classes_).index(5)])).to_csv(O/f'{mode}_{split}.csv',index=False)
   for scope,sel in [('all',np.ones(len(f),bool)),('noE16',~f.experiment_id.eq(final_rules3.E16).to_numpy())]:
    a=yy[sel];b=base[sel];p=q[sel];e=f.experiment_id.eq(final_rules3.E16).to_numpy()&(yy==5)
    rows.append(dict(mode=mode,split=split,scope=scope,baseline_f1=f1_score(a,b,labels=range(1,7),average='macro'),candidate_f1=f1_score(a,p,labels=range(1,7),average='macro'),corrected=int(((a!=b)&(a==p)).sum()),harmed=int(((a==b)&(a!=p)).sum()),E16_5_correct=int((q[e]==5).sum()),E16_5_to4=int((q[e]==4).sum())))
   for exp,idx in f.groupby('experiment_id').indices.items():per.append(dict(mode=mode,split=split,experiment_id=exp,corrected=int(((base[idx]!=yy[idx])&(q[idx]==yy[idx])).sum()),harmed=int(((base[idx]==yy[idx])&(q[idx]!=yy[idx])).sum())))
  print(mode,rows[-2:],flush=True)
 pd.DataFrame(rows).to_csv(O/'scores.csv',index=False);pd.DataFrame(cv).to_csv(O/'fit_groupcv.csv',index=False);pd.DataFrame(per).to_csv(O/'per_experiment.csv',index=False)
 (O/'config.json').write_text(json.dumps(dict(features=numeric,gate=[4,5],fit_only=True,seed=42,params=model().get_params(),note='Exploratory specialist; base B not retrained; OOF measures specialist only, not full-system OOF'),indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=2):main()
