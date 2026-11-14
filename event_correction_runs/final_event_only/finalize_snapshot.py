"""Choose event-only correction on dev/Validation, then generate one Test CSV."""
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
from select_event_temporal_correction import Correction,aligned_probs,raw_features,transform,sensor_mask,predict,routing,combine
BASE=Path('/home/ubuntu/leewon/abc_runs/current_query_v1')
DATA=Path('/home/ubuntu/data/battery_data_light')
OUT=Path('event_correction_runs/final_event_only')
if OUT.exists():raise FileExistsError('Final output exists')
tables=[];baselines=[]
for run in [Path('event_correction_runs/v1'),Path('event_correction_runs/v2_focused')]:
 t=pd.read_csv(run/'candidate_summary.csv');t['source_run']=str(run);tables.append(t)
 baselines.append(json.loads((run/'selection.json').read_text())['baseline'])
base=baselines[0]
assert base==baselines[1]
table=pd.concat(tables,ignore_index=True)
eligible=table[(table.route!='all')&(table.dev_f1>=base['dev']['macro_f1_6'])&
 (table.val_f1>=base['validation']['macro_f1_6'])&(table.corrected>table.harmed)].copy()
for i in [2,3,5]:eligible=eligible[eligible[f'F1_{i+1}']>=base['validation']['class_f1'][i]-.03]
if eligible.empty:raise RuntimeError('No event-only correction passes guards; retain B')
row=eligible.sort_values(['dev_f1','val_f1'],ascending=False).iloc[0].to_dict()
# All selection above uses no Test information.
ck=torch.load(Path(row['source_run'])/f'{row["variant"]}.pt',map_location='cpu',weights_only=False)
low,high=ck['voltage_thresholds'];norm=ck['normalization']
test=pd.read_csv(DATA/'test/test.csv').sort_values(['experiment_id','relative_time_s'],kind='stable').reset_index(drop=True)
if test.sample_id.isna().any() or not test.sample_id.is_unique:raise ValueError('Bad Test IDs')
p1=aligned_probs(BASE/'ensemble_B_test_predictions.csv',test)
x=np.c_[np.log(np.clip(p1,1e-5,1)),sensor_mask(transform(raw_features(test,low,high),norm),row['variant'])]
m=Correction(x.shape[1]);m.load_state_dict(ck['state']);torch.set_num_threads(2)
p2=predict(m,x,test,'cpu');mask=routing(test,low,high)[row['route']]
p=combine(p1,p2,mask,float(row['alpha']));labels=p.argmax(1)+1;before=p1.argmax(1)+1
assert np.array_equal(labels[~mask],before[~mask])
sample=pd.read_csv(DATA/'sample_submission.csv')
if list(sample.columns)!=['sample_id','label'] or not sample.sample_id.is_unique or set(sample.sample_id)!=set(test.sample_id):raise ValueError('Template mismatch')
byid=pd.Series(labels,index=test.sample_id);sub=sample[['sample_id']].copy();sub['label']=byid.loc[sample.sample_id].to_numpy()
assert len(sub)==3451 and sub.label.isin(range(1,7)).all()
OUT.mkdir(parents=True)
path=OUT/'submission_B_event_temporal_selected.csv';sub.to_csv(path,index=False)
table.to_csv(OUT/'all_candidates.csv',index=False)
result={'selected':row,'baseline_dev_f1':base['dev']['macro_f1_6'],'baseline_val_f1':base['validation']['macro_f1_6'],
 'selection':'event-only, dev >= baseline, Validation >= baseline, more corrected than harmed, class3/4/6 losses <= .03; rank by dev',
 'all_candidates':len(table),'eligible_event_only_candidates':len(eligible),'rows':len(sub),
 'test_eligible_rows':int(mask.sum()),'test_changed_vs_original_B':int((labels!=before).sum()),'changed_outside_event_mask':0,
 'notes':['No Test labels/score used for selection. Existing B fit/dev used; not independent nested OOF.',
 'Candidate screening reuses Validation and is exploratory. CSV creation does not establish Test improvement.',
 'Conditional flag gas_zero is an observation, not a confirmed sensor failure.']}
(OUT/'selection.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2));print('Saved',path)
