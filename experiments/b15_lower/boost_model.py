"""Current-observation multimodal expert for stages2/3/4/5.

All history is causal. Shared lower corrections and every S3 correction remain
unchanged when mixing with the submitted classifier.
"""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

SENSORS=['meas_v','meas_a','meas_t1','meas_t2','atmp_t','thermal_t','indu_t','indu_t.1',
 'meas_p','meas_ch4','meas_co','meas_co2','meas_hcn','meas_hcl','meas_hf','meas_n2o','meas_no','meas_no2','meas_so2','meas_t1_plus_t2']
PARAMS=dict(max_iter=180,max_leaf_nodes=15,max_depth=4,learning_rate=.06,
 min_samples_leaf=64,l2_regularization=10.,max_bins=127,early_stopping=False,random_state=20261115)
SPECS=[dict(name='current',mode='none',alpha=0.)]+[dict(name=f'{mode}_{a}',mode=mode,alpha=a) for mode in ['raw','stack'] for a in [.25,.5,1.]]


def slog(x):return np.sign(x)*np.log1p(abs(x))


def image_grid(packed):
    a=np.asarray(packed);single=a.ndim==3
    if single:a=a[None]
    n=len(a);rgb=a[:,:3].astype(np.float32)/255.;mask=a[:,3:4].astype(np.float32)
    count=mask.reshape(n,1,4,32,4,32).sum((3,5))
    sums=(rgb*mask).reshape(n,3,4,32,4,32).sum((3,5))
    result=np.divide(sums,count,out=np.full_like(sums,np.nan),where=count>0).reshape(n,48)
    return result[0] if single else result


def sensor_features(frame):
    f=frame.reset_index(drop=True);t=f.relative_time_s.to_numpy(float)
    if f.experiment_id.isna().any() or not np.isfinite(t).all() or (t<0).any():raise ValueError('Invalid IDs/times')
    raw=f[SENSORS].apply(pd.to_numeric,errors='coerce').to_numpy(float);raw[~np.isfinite(raw)]=np.nan
    n,d=raw.shape;initial_delta=np.full_like(raw,np.nan);peak_delta=np.full_like(raw,np.nan)
    rolling=np.full((n,d*4),np.nan)
    for ids in f.groupby('experiment_id',sort=False).indices.values():
        ids=ids[np.argsort(t[ids],kind='stable')];ts=t[ids]
        if (np.diff(ts)<=0).any():raise ValueError('Duplicate time within experiment')
        x=raw[ids];valid=np.isfinite(x);values=np.where(valid,x,0.)
        anchor=np.full(d,np.nan);peak=np.full(d,np.nan)
        for j,row in enumerate(ids):
            update=~np.isfinite(anchor)&valid[j];anchor[update]=x[j,update]
            peak=np.fmax(peak,x[j])
            initial_delta[row]=slog(x[j])-slog(anchor)
            peak_delta[row]=slog(peak)-slog(x[j])
        sums=[np.vstack([np.zeros((1,d)),np.cumsum(a,axis=0)]) for a in
              [valid.astype(float),values,valid*ts[:,None],valid*ts[:,None]**2,values*ts[:,None]]]
        end=np.arange(1,len(ids)+1)
        for wi,seconds in enumerate([15.,60.]):
            starts=np.zeros(len(ids),int);left=0
            for j in range(len(ids)):
                while left<j and ts[left]<ts[j]-seconds:left+=1
                starts[j]=left
            count,sx,st,stt,sxt=[a[end]-a[starts] for a in sums]
            mean=np.divide(sx,count,out=np.full_like(sx,np.nan),where=count>0)
            den=count*stt-st**2
            slope=np.divide(count*sxt-st*sx,den,out=np.full_like(den,np.nan),where=(count>=2)&(den>1e-10))
            rolling[ids,wi*2*d:(wi+1)*2*d]=np.c_[slog(mean),slog(slope)]
    conditions=np.c_[np.log1p(pd.to_numeric(f.bat_cc,errors='coerce').to_numpy(float)),
        (f.bat_class.to_numpy()[:,None]==np.arange(1,4)).astype(float),
        (f.soc_status.to_numpy()[:,None]==np.arange(1,3)).astype(float),
        (f.tr_method.to_numpy()[:,None]==np.arange(1,3)).astype(float)]
    return np.c_[slog(raw),~np.isfinite(raw),initial_delta,peak_delta,rolling,conditions].astype(np.float32)


def features(frame,rgb):
    if np.asarray(rgb).shape!=(len(frame),48):raise ValueError('Current-row RGB descriptors required')
    return np.c_[sensor_features(frame),rgb].astype(np.float32)


def design(x,probs,mode):
    if mode=='raw':return x
    p=np.asarray(probs,float)
    if p.shape!=(3,len(x),6):raise ValueError('Aligned B probabilities required')
    return np.c_[x,p.mean(0),p.std(0)].astype(np.float32)


def fit(frame,x,probs,mode):
    use=frame.label.between(2,5).to_numpy();y=frame.label.to_numpy(int)[use]
    groups=frame.experiment_id.to_numpy()[use]
    keys=pd.DataFrame({'e':groups,'y':y});w=1/keys.groupby(['e','y']).y.transform('size').to_numpy(float)
    model=HistGradientBoostingClassifier(**PARAMS).fit(design(x,probs,mode)[use],y,sample_weight=w/w.mean())
    return dict(model=model,mode=mode,fit_experiments=sorted(set(frame.experiment_id)),params=PARAMS)


def probability(artifact,x,probs):
    if 'members' in artifact:
        weights=np.asarray(artifact['weights'],float)
        if len(weights)!=len(artifact['members']) or (weights<=0).any() or not np.isclose(weights.sum(),1.):
            raise ValueError('Invalid group ensemble weights')
        return sum(w*probability(m,x,probs) for w,m in zip(weights,artifact['members']))
    p=artifact['model'].predict_proba(design(x,probs,artifact['mode']))
    if not np.array_equal(artifact['model'].classes_,[2,3,4,5]):raise ValueError('Stage expert schema changed')
    return p


def fit_bagged(frame,x,full):
    """Fixed half-full, half-five-group-subsets expert; no held-label selection."""
    groups=np.array(sorted(set(frame.experiment_id)))
    if set(full['fit_experiments'])!=set(groups):raise ValueError('Full expert groups differ')
    rng=np.random.default_rng(20261115);members=[full]
    for _ in range(5):
        selected=rng.choice(groups,size=max(2,int(np.ceil(.8*len(groups)))),replace=False)
        take=frame.experiment_id.isin(selected).to_numpy()
        members.append(fit(frame.loc[take].reset_index(drop=True),x[take],None,'raw'))
    return dict(members=members,weights=[.5]+[.1]*5,mode='raw',fit_experiments=groups.tolist(),
                protocol='half full expert plus half mean five 80-percent experiment subsets; seed20261115')


def combine(stages,probs,expert,spec):
    current=stages['S3'];out=np.asarray(current,int).copy()
    if spec['alpha']==0:return out
    soft=np.asarray(probs,float).mean(0)[:,1:5].copy()
    den=soft.sum(1,keepdims=True);soft=np.divide(soft,den,out=np.full_like(soft,.25),where=den>1e-12)
    source=soft.argmax(1);target=np.clip(out-2,0,3);rows=np.arange(len(out))
    a=soft[rows,source].copy();b=soft[rows,target].copy();soft[rows,source]=b;soft[rows,target]=a
    mixed=(1-spec['alpha'])*soft+spec['alpha']*expert
    proposed=mixed.argmax(1)+2
    protected=np.isin(current,[1,6])|(stages['J096']!=stages['precision65'])|(current!=stages['J096'])
    out[~protected]=proposed[~protected]
    assert np.array_equal(out[protected],current[protected])
    assert np.array_equal(out==6,np.asarray(current)==6)
    return out


def choose(scores):
    means=scores.mean(0);best=float(means.max())
    eligible=[i for i,v in enumerate(means) if v>=best-.0015]
    selected=min(eligible,key=lambda i:(SPECS[i]['alpha'],SPECS[i]['mode']=='stack',-means[i]))
    return SPECS[selected],dict(means={s['name']:float(v) for s,v in zip(SPECS,means)},
                               selected=SPECS[selected]['name'],allowance=.0015)
