"""Local aggregate-only sensor-event diagnostics; thresholds fitted on original fit.
No Test reads, no row exports, no data cleaning or neural training.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from abc_reference import abc_fusion_experiment as abc
from analyze_experiment_performance import confusion,scores
from probe_causal_phase_history import history

ROOT=Path('/home/ubuntu/data/battery_data_light')
BASE=Path('/home/ubuntu/leewon/abc_runs/current_query_v1')
OUT=Path('experiment_diagnostics/sensor_event_analysis')


def describe(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return {'count':len(a),'min':float(a.min()),'p01':float(np.quantile(a,.01)),'median':float(np.median(a)),
            'p99':float(np.quantile(a,.99)),'max':float(a.max())} if len(a) else {'count':0}


def events(f,low=None,high=None):
    t=pd.to_numeric(f.relative_time_s,errors='raise').to_numpy(float)
    v=pd.to_numeric(f.meas_v,errors='coerce').to_numpy(float)
    gases=f[[c for c in abc.GAS if c!='meas_p']].to_numpy(float)
    zero=np.isfinite(gases).all(1)&(gases==0).all(1)
    rate=np.full(len(f),np.nan);dt=np.full(len(f),np.nan);prev_rate=np.full(len(f),np.nan)
    zero_onset=np.zeros(len(f),bool);zero_recovery=np.zeros(len(f),bool);zero_age=np.full(len(f),np.nan)
    orderings=[]
    for idx in f.groupby('experiment_id',sort=False).indices.values():
        ids=idx[np.argsort(t[idx],kind='stable')];ts=t[ids]
        if not np.isfinite(ts).all() or (np.diff(ts)<=0).any():raise ValueError('Invalid times')
        orderings.append(ids)
        dt[ids[1:]]=np.diff(ts);rate[ids[1:]]=np.diff(v[ids])/np.diff(ts)
        prev_rate[ids[1:]]=rate[ids[:-1]]
        zero_onset[ids[1:]]=zero[ids[1:]]&~zero[ids[:-1]]
        zero_recovery[ids[1:]]=~zero[ids[1:]]&zero[ids[:-1]]
        since=None
        for row in ids:
            if zero[row]:
                if since is None:since=t[row]
                zero_age[row]=t[row]-since
            else:since=None
    result={'voltage_rate':rate,'dt':dt,'gas_all_zero':zero,'gas_zero_onset':zero_onset,'gas_zero_recovery':zero_recovery,
            'gas_zero_age':zero_age,'voltage_nonpositive':np.isfinite(v)&(v<=0),'gas_any_missing':(~np.isfinite(gases)).any(1),
            'gas_any_negative':((gases<0)&np.isfinite(gases)).any(1),'voltage_missing':~np.isfinite(v)}
    if low is not None:
        drop=np.isfinite(rate)&(rate<low);rise=np.isfinite(rate)&(rate>high)
        recent=np.zeros(len(f),bool);ever=np.zeros(len(f),bool)
        for ids in orderings:
            last=None
            for row in ids:
                if drop[row]:last=t[row]
                if last is not None:ever[row]=True;recent[row]=t[row]-last<=60
        result.update(voltage_drop_tail=drop,voltage_rise_tail=rise,
            voltage_rebound=np.isfinite(prev_rate)&(prev_rate<low)&rise,
            voltage_drop_past60=recent,voltage_drop_ever=ever)
    return result


def predict_probe(f,name):
    a=joblib.load(Path('experiment_diagnostics/five_to_four_history_saved')/f'{name}.joblib')
    meta=a['preprocessing'];prep=abc.SensorPreprocessor.__new__(abc.SensorPreprocessor)
    for k in ['median','mean','std']:setattr(prep,k,pd.Series(meta[k],index=abc.NUMERIC))
    q,g=prep.transform(f);x=np.c_[q,g.reshape(len(f),-1)]
    if name=='current_past_extrema':x=np.c_[x,history(f,60)[:,12:]]
    return a['classifier'].predict(a['scaler'].transform(x))


def run():
    if OUT.exists():raise FileExistsError('Output already exists')
    cfg=json.loads((BASE/'config.json').read_text());frames=abc.common.split_frames(ROOT,cfg['split_seed'],.2)
    fit_events=events(frames['fit']);finite=fit_events['voltage_rate'][np.isfinite(fit_events['voltage_rate'])]
    low,high=np.quantile(finite,[.01,.99])
    if low>=0 or high<=0:raise ValueError('Tails do not separate drops/rises')
    allframes={'train_all':pd.read_csv(ROOT/'train/train.csv'),'actual_fit':frames['fit'],'validation':frames['validation']}
    report={'thresholds':{'fit_voltage_rate_p01':float(low),'fit_voltage_rate_p99':float(high)},'splits':{},'validation_buckets':{},'per_experiment':{},
      'notes':['Fit-derived percentile tails describe unusual changes, not proven measurement errors.',
        'All-ten gas-zero requires finite exact zero concentrations; pressure is excluded.',
        'Voltage rebound is past-only: previous downward-tail interval followed by an upward-tail interval.',
        'Gas-zero age measures observed current/past run only; interval sampling does not establish continuous physical duration.',
        'Error associations are conditional summaries, not causal attribution or a license to delete data.',
        'No Test reads, row-level exports or raw-data modifications.']}
    evs={}
    for split,f in allframes.items():
        e=events(f,low,high);evs[split]=e
        summary={'rows':len(f),'voltage':describe(pd.to_numeric(f.meas_v,errors='coerce')),'voltage_rate':describe(e['voltage_rate']),
            'sampling_interval':describe(e['dt']),'gas_zero_observed_age':describe(e['gas_zero_age']),'events':{},'by_class':{}}
        for name,mask in e.items():
            if mask.dtype==bool:summary['events'][name]={'count':int(mask.sum()),'fraction':float(mask.mean())}
        for label in range(1,7):
            mask=f.label.to_numpy(int)==label
            summary['by_class'][str(label)]={'rows':int(mask.sum()),**{name:float(v[mask].mean()) for name,v in e.items() if v.dtype==bool}}
        report['splits'][split]=summary
    f=frames['validation'];e=evs['validation'];y=f.label.to_numpy(int)
    ref=pd.read_csv(BASE/'ensemble_B_validation_predictions.csv')
    if not np.array_equal(ref.sample_id,f.sample_id) or not np.array_equal(ref.label,y):raise ValueError('ABC rows differ')
    pred={'ABC_B':ref.pred.to_numpy(int),'current_probe':predict_probe(f,'current'),'past_extrema_probe':predict_probe(f,'current_past_extrema')}
    masks={'all':np.ones(len(f),bool),'gas_zero':e['gas_all_zero'],'gas_nonzero':~e['gas_all_zero'],
        'voltage_drop_now':e['voltage_drop_tail'],'no_voltage_drop_now':~e['voltage_drop_tail'],
        'voltage_rebound':e['voltage_rebound'],'drop_past60':e['voltage_drop_past60'],
        'drop_ever':e['voltage_drop_ever'],'no_drop_ever':~e['voltage_drop_ever'],
        'gas_zero_onset':e['gas_zero_onset'],'gas_zero_longer60':e['gas_all_zero']&(e['gas_zero_age']>=60),
        'gas_zero_and_drop_ever':e['gas_all_zero']&e['voltage_drop_ever']}
    groupids=list(f.groupby('experiment_id',sort=True).indices.values())
    for i,idx in enumerate(groupids):
        mask=np.zeros(len(f),bool);mask[idx]=True
        if i==15:
            masks['E16']=mask;masks['E16_gas_zero']=mask&e['gas_all_zero'];masks['E16_gas_nonzero']=mask&~e['gas_all_zero']
        report['per_experiment'][f'E{i+1:02}']={'rows':len(idx),'ABC_errors':int((pred['ABC_B'][idx]!=y[idx]).sum()),
            **{name:int(v[idx].sum()) for name,v in e.items() if v.dtype==bool}}
    rows=[]
    for name,mask in masks.items():
        result={'rows':int(mask.sum()),'true5_rows':int((mask&(y==5)).sum()),'models':{}}
        if not mask.any():report['validation_buckets'][name]=result;continue
        for m,p in pred.items():
            five=mask&(y==5);four=mask&(y==4)
            v={'error_rate':float((p[mask]!=y[mask]).mean()),'macro_f1':scores(confusion(y[mask],p[mask]))['macro_f1_6'],
               'five_correct':int((five&(p==5)).sum()),'five_to_four':int((five&(p==4)).sum()),
               'five_error_rate':float((p[five]!=5).mean()) if five.any() else None,
               'four_to_five':int((four&(p==5)).sum())}
            result['models'][m]=v
            rows.append({'bucket':name,'rows':result['rows'],'true5_rows':result['true5_rows'],'model':m,**v})
        report['validation_buckets'][name]=result
    OUT.mkdir(parents=True);(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    pd.DataFrame(rows).to_csv(OUT/'event_error_comparison.csv',index=False)
    print('THRESHOLDS',json.dumps(report['thresholds']))
    for s,v in report['splits'].items():print(s,'events',json.dumps(v['events']),'voltage',v['voltage'],'zero_age',v['gas_zero_observed_age'])
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':run()
