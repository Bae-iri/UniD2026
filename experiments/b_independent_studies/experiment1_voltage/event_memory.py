import numpy as np
import pandas as pd
NAMES=['max_voltage_drop', 'max_drop_rate', 'time_since_drop', 'voltage_after_drop', 'drop_occurred', 'event_not_observed', 'has_observed_voltage_pair']

def ordered_groups(frame):
    if frame[['experiment_id','relative_time_s']].isna().any().any() or frame.duplicated(['experiment_id','relative_time_s']).any():
        raise ValueError('Missing experiment/time or duplicate experiment timestamps')
    time=pd.to_numeric(frame.relative_time_s,errors='raise').to_numpy(float)
    if not np.isfinite(time).all() or (time<0).any(): raise ValueError('Invalid time')
    voltage=pd.to_numeric(frame.meas_v,errors='coerce').to_numpy(float)
    for positions in frame.groupby('experiment_id',sort=False).indices.values():
        ids=positions[np.argsort(time[positions])]
        yield ids,time[ids],voltage[ids]

def fit_detector(train,max_gap_s=30):
    # Fixed rules selected before validation: 10% of fit median positive voltage,
    # plus 90th percentile of positive adjacent drop rates. No labels are used.
    voltage=pd.to_numeric(train.meas_v,errors='coerce').to_numpy(float)
    positive=voltage[np.isfinite(voltage)&(voltage>0)]
    if not len(positive): raise ValueError('No positive fit voltage for detector')
    rates=[]
    for ids,t,v in ordered_groups(train):
        for j in range(1,len(ids)):
            dt=t[j]-t[j-1]
            if 0<dt<=max_gap_s and np.isfinite(v[j-1:j+1]).all() and min(v[j-1:j+1])>=0:
                drop=v[j-1]-v[j]
                if drop>0: rates.append(drop/dt)
    if not rates: raise ValueError('No finite positive fit voltage-drop rate')
    return dict(drop_threshold_v=max(float(np.median(positive))*.1,1e-6),rate_threshold_v_per_s=max(float(np.quantile(rates,.9)),1e-8),max_gap_s=float(max_gap_s),definition='consecutive observed rows; first qualifying drop; fit-only thresholds')

def event_memory(frame,state):
    output=np.zeros((len(frame),7),dtype=np.float64); output[:,5]=1
    for ids,t,v in ordered_groups(frame):
        max_drop=0.; max_rate=0.; event_time=None; after=0.; pair_seen=False
        for j,target in enumerate(ids):
            if j>0:
                dt=t[j]-t[j-1]
                valid=0<dt<=state['max_gap_s'] and np.isfinite(v[j-1:j+1]).all() and min(v[j-1:j+1])>=0
                if valid:
                    pair_seen=True; drop=max(float(v[j-1]-v[j]),0.); rate=drop/dt
                    max_drop=max(max_drop,drop); max_rate=max(max_rate,rate)
                    if event_time is None and drop>=state['drop_threshold_v'] and rate>=state['rate_threshold_v_per_s']:
                        event_time=t[j]; after=v[j]
            happened=event_time is not None
            output[target]=[max_drop,max_rate,t[j]-event_time if happened else 0.,after if happened else 0.,float(happened),float(not happened),float(pair_seen)]
    return output.astype(np.float32)
