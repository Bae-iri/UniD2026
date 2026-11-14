"""Matched causal GRU corrections on frozen original ABC B; aggregate-only reporting.
Original fit trains correction, original dev selects checkpoint/routing. Validation guards
against harm. Test probabilities are read only after final candidate selection.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from abc_reference import abc_fusion_experiment as abc
from diagnose_abc_gas_dependency import restore_preprocessor
from analyze_sensor_events import events
from probe_causal_phase_history import history
from analyze_experiment_performance import confusion,scores

ROUTES=['all','voltage_now','voltage_recent60','gas_zero','intersection','union_now','union_recent60']


def aligned_probs(path,f):
 p=pd.read_csv(path)
 if not p.sample_id.is_unique or set(p.sample_id)!=set(f.sample_id):raise ValueError('Probability IDs differ')
 p=p.set_index('sample_id').loc[f.sample_id]
 if 'label' in f and not np.array_equal(p.label,f.label):raise ValueError('Probability labels differ')
 return p[[f'p{i}' for i in range(1,7)]].to_numpy(np.float32)


@torch.inference_mode()
def fit_probabilities(args,frames):
 prep=restore_preprocessor(args.base_run);manifest=json.loads((args.base_run/'image_cache_manifest.json').read_text())
 f=frames['fit'];q,g=prep.transform(f);images=np.load(manifest['fit'],mmap_mode='c')
 if images.shape!=(len(f),4,128,128):raise ValueError('Cache shape differs')
 data=abc.Data(f,torch.from_numpy(images),torch.from_numpy(q),torch.from_numpy(g),None)
 total=np.zeros((len(f),6),np.float64)
 for seed in [42,43,44]:
  model=abc.FusionModel('B').to(args.device);ck=torch.load(args.base_run/f'seed_{seed}/B/best.pt',weights_only=True,map_location='cpu');model.load_state_dict(ck['state']);model.eval()
  pieces=[]
  for start in range(0,len(f),128):
   idx=torch.arange(start,min(start+128,len(f)));pieces.append(model(*data.batch(idx,args.device)).softmax(-1).cpu().numpy())
  total+=np.concatenate(pieces)/3
  print('Original B fit probabilities ready, seed',seed,flush=True)
 return total.astype(np.float32)


def groups(f):
 t=f.relative_time_s.to_numpy(float);result=[]
 for ids in f.groupby('experiment_id',sort=True).indices.values():
  ids=ids[np.argsort(t[ids],kind='stable')]
  if not np.isfinite(t[ids]).all() or (np.diff(t[ids])<=0).any():raise ValueError('Invalid observation times')
  result.append(ids)
 return result


def routing(f,low,high):
 e=events(f,low,high);v=e['voltage_drop_tail']|e['voltage_rebound']
 return {'all':np.ones(len(f),bool),'voltage_now':v,'voltage_recent60':e['voltage_drop_past60'],
  'gas_zero':e['gas_all_zero'],'intersection':v&e['gas_all_zero'],'union_now':v|e['gas_all_zero'],
  'union_recent60':e['voltage_drop_past60']|e['gas_all_zero']}


def raw_features(f,low,high):
 sensor=f[abc.NUMERIC].apply(pd.to_numeric,errors='coerce').to_numpy(float)
 sensor[~np.isfinite(sensor)]=np.nan
 # Signed log reduces extreme scale without overwriting original observations.
 sensor=np.sign(sensor)*np.log1p(np.abs(sensor))
 hist=history(f,60) # same-experiment causal recent slopes, extrema and observed ages
 e=events(f,low,high)
 times=f.relative_time_s.to_numpy(float);gap=np.zeros(len(f))
 for ids in groups(f):gap[ids[1:]]=np.diff(times[ids])
 flags=np.stack([e['gas_all_zero'],e['gas_zero_onset'],e['voltage_drop_tail'],e['voltage_rebound'],e['voltage_drop_past60']],1).astype(float)
 # Pressure remains separate; all-zero gas is a flag, never interpreted as missing.
 return np.c_[sensor,np.isnan(sensor).astype(float),hist,flags,np.log1p(gap),np.log1p(times)]


def normalize_fit(x):
 med=np.zeros(x.shape[1])
 for j in range(x.shape[1]):
  a=x[:,j][np.isfinite(x[:,j])];med[j]=np.median(a) if len(a) else 0
 filled=np.where(np.isfinite(x),x,med);mean=filled.mean(0);std=filled.std(0);std[std<1e-6]=1
 return {'median':med.tolist(),'mean':mean.tolist(),'std':std.tolist()}


def transform(x,n):
 x=np.where(np.isfinite(x),x,np.asarray(n['median']))
 return np.clip((x-np.asarray(n['mean']))/np.asarray(n['std']),-10,10).astype(np.float32)


def sensor_mask(x,variant):
 x=x.copy();ns=len(abc.NUMERIC)
 if variant=='no_gas':
  x[:,6:ns]=0;x[:,ns+6:2*ns]=0
  # Suppress gas-zero flags; underlying Stage1 still encodes original gas inputs.
  x[:,2*ns+28:2*ns+30]=0
 if variant=='no_voltage':
  x[:,0]=0;x[:,ns]=0
  # Voltage slope block and voltage extrema block from history().
  x[:,2*ns:2*ns+3]=0;x[:,2*ns+12:2*ns+16]=0
  x[:,2*ns+30:2*ns+33]=0
 return x


class Correction(nn.Module):
 def __init__(self,n):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(n,32),nn.ReLU());self.gru=nn.GRU(32,32,batch_first=True)
  self.out=nn.Linear(64,6);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,x):
  z=self.encoder(x);h,_=self.gru(z);delta=self.out(torch.cat([z,h],-1))
  return x[...,:6]+delta


@torch.inference_mode()
def predict(model,x,f,device):
 model.eval();p=np.zeros((len(f),6),np.float32)
 for ids in groups(f):p[ids]=model(torch.tensor(x[ids],device=device)[None])[0].softmax(-1).cpu().numpy()
 return p


def batch_sequences(indices,x,y,device):
 length=max(len(i) for i in indices);a=np.zeros((len(indices),length,x.shape[1]),np.float32);b=np.full((len(indices),length),-100,np.int64)
 for k,ids in enumerate(indices):a[k,:len(ids)]=x[ids];b[k,:len(ids)]=y[ids]-1
 return torch.tensor(a,device=device),torch.tensor(b,device=device)


def combine(p1,p2,mask,alpha):
 p=p1.copy();p[mask]=(1-alpha)*p1[mask]+alpha*p2[mask];return p


def evaluate(y,p):return scores(confusion(y,p.argmax(1)+1))


def run(args):
 if args.output.exists():raise FileExistsError('New output required')
 torch.set_num_threads(2);started=time.monotonic();cfg=json.loads((args.base_run/'config.json').read_text())
 for s in ['train','validation']:
  if hashlib.sha256((args.data_dir/s/f'{s}.csv').read_bytes()).hexdigest()!=cfg['csv_sha256'][s]:raise ValueError('CSV differs')
 frames=abc.common.split_frames(args.data_dir,cfg['split_seed'],.2)
 manifest=pd.read_csv(args.base_run/'split_manifest.csv')
 for split,f in frames.items():
  if not np.array_equal(manifest.loc[manifest.split==split,'sample_id'],f.sample_id):raise ValueError('Split order mismatch')
 p1={'fit':fit_probabilities(args,frames),'dev':aligned_probs(args.base_run/'ensemble_B_dev_predictions.csv',frames['dev']),
     'validation':aligned_probs(args.base_run/'ensemble_B_validation_predictions.csv',frames['validation'])}
 rate=events(frames['fit'])['voltage_rate'];rate=rate[np.isfinite(rate)];low,high=np.quantile(rate,[.01,.99])
 raw={s:raw_features(f,low,high) for s,f in frames.items()};norm=normalize_fit(raw['fit'])
 normed={s:transform(v,norm) for s,v in raw.items()};masks={s:routing(f,low,high) for s,f in frames.items()}
 args.output.mkdir(parents=True)
 settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
 settings.update(voltage_drop_threshold=float(low),voltage_rise_threshold=float(high),selection='dev ranking, nonnegative dev delta; Validation noninferiority .003 and class3/4/6 loss <= .03; else baseline',
  limitations='Stage1 fit predictions are in-sample; dev used in original Stage1 checkpoint selection. Exploratory continuation, not nested OOF. No Test selection.')
 (args.output/'config.json').write_text(json.dumps(settings,indent=2))
 base={s:evaluate(frames[s].label,p1[s]) for s in ['dev','validation']};rows=[];models={};p2s={};normal_y=frames['fit'].label.to_numpy(int)
 weights=abc.SensorPreprocessor(frames['fit']).weights
 for variant in args.variants:
  abc.common.seed_everything(42)
  x={s:np.c_[np.log(np.clip(p1[s],1e-5,1)),sensor_mask(v,variant)] for s,v in normed.items()}
  model=Correction(x['fit'].shape[1]).to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
  criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,device=args.device),ignore_index=-100)
  focus={'gas_focus':'gas_zero','voltage_focus':'voltage_recent60','union_focus':'union_recent60'}.get(variant,'all')
  train_y=normal_y.copy();train_y[~masks['fit'][focus]]=-100
  best=base['dev']['macro_f1_6'];beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bestepoch=0
  rng=np.random.default_rng(42);g=groups(frames['fit']);historylog=[]
  for epoch in range(1,args.epochs+1):
   model.train();order=rng.permutation(len(g))
   for start in range(0,len(order),4):
    xb,yb=batch_sequences([g[i] for i in order[start:start+4]],x['fit'],train_y,args.device)
    # batch_sequences subtracts one from labels, so restore ignore marker.
    yb[yb==-101]=-100
    if not (yb!=-100).any():continue
    optimizer.zero_grad(set_to_none=True);loss=criterion(model(xb).reshape(-1,6),yb.reshape(-1));loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
   pdv=predict(model,x['dev'],frames['dev'],args.device);score=evaluate(frames['dev'].label,combine(p1['dev'],pdv,masks['dev'][focus],1.))['macro_f1_6']
   historylog.append({'epoch':epoch,'dev_macro_f1':score})
   if score>best:best,bestepoch=score,epoch;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   print(f'{variant} epoch={epoch}/{args.epochs} dev={score:.6f} best={best:.6f}',flush=True)
  model.load_state_dict(beststate);models[variant]=model
  torch.save({'state':beststate,'normalization':norm,'variant':variant,'best_epoch':bestepoch,'voltage_thresholds':[float(low),float(high)]},args.output/f'{variant}.pt')
  (args.output/f'{variant}_history.json').write_text(json.dumps(historylog,indent=2))
  p2s[variant]={s:predict(model,x[s],frames[s],args.device) for s in ['dev','validation']}
  for route in ROUTES:
   for alpha in [.5,1.]:
    results={s:evaluate(frames[s].label,combine(p1[s],p2s[variant][s],masks[s][route],alpha)) for s in ['dev','validation']}
    valpred=combine(p1['validation'],p2s[variant]['validation'],masks['validation'][route],alpha).argmax(1)+1
    yv=frames['validation'].label.to_numpy(int);idx=groups(frames['validation'])[15]
    row=dict(variant=variant,route=route,alpha=alpha,dev_f1=results['dev']['macro_f1_6'],val_f1=results['validation']['macro_f1_6'],best_epoch=bestepoch,
       E16_5_to_4=int(((yv[idx]==5)&(valpred[idx]==4)).sum()),E16_5_correct=int(((yv[idx]==5)&(valpred[idx]==5)).sum()),
       corrected=int(((p1['validation'].argmax(1)+1!=yv)&(valpred==yv)).sum()),harmed=int(((p1['validation'].argmax(1)+1==yv)&(valpred!=yv)).sum()),
       **{f'F1_{i+1}':v for i,v in enumerate(results['validation']['class_f1'])})
    rows.append(row)
 table=pd.DataFrame(rows);table.to_csv(args.output/'candidate_summary.csv',index=False)
 eligible=[]
 for row in rows:
  if row['dev_f1']<base['dev']['macro_f1_6']-1e-9 or row['val_f1']<base['validation']['macro_f1_6']-.003:continue
  if any(row[f'F1_{i+1}']<base['validation']['class_f1'][i]-.03 for i in [2,3,5]):continue
  if row['val_f1']>base['validation']['macro_f1_6']+1e-6:eligible.append(row)
 selected=max(eligible,key=lambda r:r['dev_f1']) if eligible else None
 # Candidate fully selected BEFORE Test is read. No Test distribution/score optimization.
 test=pd.read_csv(args.data_dir/'test/test.csv')
 if test.sample_id.isna().any() or not test.sample_id.is_unique:raise ValueError('Invalid Test IDs')
 test=test.sort_values(['experiment_id','relative_time_s'],kind='stable').reset_index(drop=True)
 pt=aligned_probs(args.base_run/'ensemble_B_test_predictions.csv',test)
 if selected:
  xx=transform(raw_features(test,low,high),norm);xx=np.c_[np.log(np.clip(pt,1e-5,1)),sensor_mask(xx,selected['variant'])]
  pc=predict(models[selected['variant']],xx,test,args.device)
  result=combine(pt,pc,routing(test,low,high)[selected['route']],selected['alpha'])
 else:result=pt
 order=pd.read_csv(args.data_dir/'sample_submission.csv')
 if list(order.columns)!=['sample_id','label'] or not order.sample_id.is_unique or set(order.sample_id)!=set(test.sample_id):raise ValueError('Template IDs differ')
 labels=pd.Series(result.argmax(1)+1,index=test.sample_id);submission=order[['sample_id']].copy();submission['label']=labels.loc[order.sample_id].to_numpy()
 if not submission.label.isin(range(1,7)).all():raise ValueError('Invalid labels')
 path=args.output/'submission_selected.csv';submission.to_csv(path,index=False)
 report={'baseline':base,'selected':selected,'candidates':len(rows),'eligible':len(eligible),'test_rows':len(test),
   'test_changed_vs_original_B':int((result.argmax(1)!=pt.argmax(1)).sum()),'elapsed_seconds':time.monotonic()-started,
   'note':'Corrections affect only eligible current rows but consume full same-experiment past history; no future. no_gas/no_voltage remove new Stage2 inputs, not information already encoded by Stage1.'}
 (args.output/'selection.json').write_text(json.dumps(report,indent=2))
 print(table.sort_values('dev_f1',ascending=False).head(12).to_string(index=False),flush=True)
 print('SELECTION',json.dumps(report),flush=True);print('Saved',path,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--base-run',type=Path,default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
 p.add_argument('--data-dir',type=Path,default=Path('/home/ubuntu/data/battery_data_light'))
 p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=8)
 p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
 p.add_argument('--variants',nargs='+',choices=['full','no_gas','no_voltage','gas_focus','voltage_focus','union_focus'],default=['full','no_gas','no_voltage'])
 a=p.parse_args()
 if a.epochs<1:p.error('epochs must be positive')
 run(a)
