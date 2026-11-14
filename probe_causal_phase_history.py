"""CPU current-vs-causal-history probes on original ABC fit/dev split; no Test."""
import argparse,json,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from abc_reference import abc_fusion_experiment as abc
from diagnose_abc_gas_dependency import restore_preprocessor
from analyze_experiment_performance import confusion,scores
SIGNALS=['meas_v','meas_t1','meas_t2','thermal_t']


def history(frame,window):
    times=pd.to_numeric(frame.relative_time_s,errors='raise').to_numpy(float)
    if not np.isfinite(times).all():raise ValueError('Bad times')
    values=frame[SIGNALS].to_numpy(float);out=np.zeros((len(frame),28),np.float64)
    for idx in frame.groupby('experiment_id',sort=False).indices.values():
        order=idx[np.argsort(times[idx],kind='stable')];ts=times[order]
        if (np.diff(ts)<=0).any():raise ValueError('Duplicate/nonincreasing time')
        for k in range(4):
            peak=None;peak_t=None
            for j,row in enumerate(order):
                v=values[row,k]
                if np.isfinite(v):
                    improved=peak is None or (v<peak if k==0 else v>peak)
                    if improved:peak,peak_t=v,times[row]
                if peak is not None:
                    out[row,12+4*k:16+4*k]=[peak,v-peak if np.isfinite(v) else 0,times[row]-peak_t,1]
                start=np.searchsorted(ts,times[row]-window,side='left')
                earlier=order[start:j];a=values[earlier,k];good=np.isfinite(a);earlier=earlier[good];a=a[good]
                if len(a) and np.isfinite(v):
                    t=np.concatenate([times[earlier],[times[row]]]);b=np.concatenate([a,[v]])
                    centered=t-t.mean();den=np.dot(centered,centered)
                    slope=np.dot(centered,b-b.mean())/den if den>0 else 0
                    out[row,3*k:3*k+3]=[slope,v-a.mean(),1]
    return out


def run(args):
    if args.output.exists():raise FileExistsError('Use a new output')
    cfg=json.loads((args.base_run/'config.json').read_text())
    split_seed=cfg['split_seed'] if args.split_seed is None else args.split_seed
    frames=abc.common.split_frames(args.data_dir,split_seed,.2)
    prep=abc.SensorPreprocessor(frames['fit'])
    current={};h30={};h60={};ys={}
    for split,f in frames.items():
        q,g=prep.transform(f);current[split]=np.concatenate([q,g.reshape(len(f),-1)],1)
        h30[split]=history(f,30);h60[split]=history(f,60);ys[split]=f.label.to_numpy(int)
    variants={
      'current':current,
      'current_recent30':{s:np.c_[current[s],h30[s][:,:12]] for s in frames},
      'current_recent60':{s:np.c_[current[s],h60[s][:,:12]] for s in frames},
      'current_past_extrema':{s:np.c_[current[s],h60[s][:,12:]] for s in frames},
      'current_recent60_extrema':{s:np.c_[current[s],h60[s]] for s in frames},
      'current_elapsed':{s:np.c_[current[s],np.log1p(frames[s].relative_time_s.to_numpy(float))] for s in frames},
      'current_extrema_no_age':{s:np.c_[current[s],h60[s][:,[12+i for i in range(16) if i%4!=2]]] for s in frames},
      'current_extrema_age_only':{s:np.c_[current[s],h60[s][:,[14,18,22,26]]] for s in frames}}
    groups=list(frames['validation'].groupby('experiment_id',sort=True).indices.values());result={};rows=[];artifacts={}
    with threadpool_limits(limits=2):
        for name,x in variants.items():
            if name not in args.modes:continue
            scaler=StandardScaler().fit(x['fit']);xx={s:scaler.transform(v) for s,v in x.items()}
            model=LogisticRegression(C=.1,max_iter=400,class_weight={k+1:float(w) for k,w in enumerate(prep.weights)},random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(xx['fit'],ys['fit'])
            entry={'warnings':[str(w.message) for w in caught]}
            for split in ['dev','validation']:
                pred=model.predict(xx[split]);entry[split]=scores(confusion(ys[split],pred))
                if split=='validation':
                    entry['experiments']={f'E{i+1:02}':scores(confusion(ys[split][idx],pred[idx])) for i,idx in enumerate(groups)}
            e=entry['experiments']['E16']
            rows.append(dict(mode=name,dev_f1=entry['dev']['macro_f1_6'],validation_f1=entry['validation']['macro_f1_6'],
                class5_f1=entry['validation']['class_f1'][4],E16_f1=e['macro_f1_true_present'],E16_5_to_4=e['confusion'][4][3],E16_5_correct=e['confusion'][4][4]))
            artifacts[name]={'classifier':model,'scaler':scaler,'mode':name,'preprocessing':prep.metadata(),
                'split_seed':split_seed,'signals':SIGNALS,'history_rule':'same-experiment past/current only; voltage min, temperatures max; no labels or future'}
            result[name]=entry;print(json.dumps(rows[-1]),flush=True)
    args.output.mkdir(parents=True)
    import joblib
    for name,artifact in artifacts.items():joblib.dump(artifact,args.output/f'{name}.joblib')
    (args.output/'report.json').write_text(json.dumps({'split_seed':split_seed,'results':result,'notes':[
      'Raw current sensors/concentrations/pressure; no images, IDs or static conditions. current_elapsed is an explicit elapsed-time control only.',
      'History features are causal within each experiment. Voltage uses past minimum; temperatures use past maximum.',
      'Extrema include current; time since extremum is observed past-only, not experiment end/duration.',
      'Fit-only scaling/classifier; same original fit/dev; C=.1 fixed. Exploratory CPU probes, not final ABC architectures.',
      'No Test access, row-level exports, checkpoint modifications or submission files.']},indent=2,allow_nan=False))
    pd.DataFrame(rows).to_csv(args.output/'history_probe_summary.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run',type=Path,default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    p.add_argument('--data-dir',type=Path,default=Path('/home/ubuntu/data/battery_data_light'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split-seed',type=int)
    p.add_argument('--modes',nargs='+',choices=['current','current_recent30','current_recent60','current_past_extrema','current_recent60_extrema','current_elapsed','current_extrema_no_age','current_extrema_age_only'],default=['current','current_recent30','current_recent60','current_past_extrema','current_recent60_extrema'])
    run(p.parse_args())
