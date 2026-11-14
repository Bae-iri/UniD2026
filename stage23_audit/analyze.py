"""Sensor and causal-state audit of original B's 2<->3 errors. No Test access."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier,export_text

ROOT=Path('/home/ubuntu/JEONGYUN')
DATA=Path('/home/ubuntu/data/battery_data_light')
BASE=Path('/home/ubuntu/leewon/abc_runs/current_query_v1')
OUT=ROOT/'stage23_audit/results'
E16='E_e5bcddc7d77f459ea541cf658ab4346e'
SENS=['meas_t1','meas_t2','thermal_t','meas_v','meas_p','meas_ch4','meas_co','meas_co2','meas_hcn','meas_hcl','meas_hf','meas_n2o','meas_no','meas_no2','meas_so2','indu_t']
STATIC=['bat_class','bat_cc','soc_status','tr_method']


def features(frame):
    if frame.duplicated(['experiment_id','relative_time_s']).any():
        raise ValueError('Duplicate timestamps')
    d=frame.sort_values(['experiment_id','relative_time_s'],kind='stable').copy()
    g=d.groupby('experiment_id',sort=False)
    for c in ['meas_t1','meas_t2','thermal_t','meas_p','meas_v']:
        d[c+'_max']=g[c].cummax().groupby(d.experiment_id).ffill()
        first=g[c].transform('first')
        d[c+'_delta_start']=d[c]-first
        for seconds in [30,60]:
            rate=np.full(len(d),np.nan)
            for ids in g.indices.values():
                t=d.relative_time_s.to_numpy(float)[ids];v=d[c].to_numpy(float)[ids]
                j=np.searchsorted(t,t-seconds,side='right')-1
                valid=j>=0
                rate[ids[valid]]=(v[valid]-v[j[valid]])/(t[valid]-t[j[valid]])*60
            d[f'{c}_rate{seconds}']=rate
    d['t1_minus_t2']=d.meas_t1-d.meas_t2
    for c in ['meas_t1','meas_t2','thermal_t']:
        d[c+'_minus_ambient']=d[c]-d.atmp_t
    # First OBSERVED crossing, not a future-dependent baseline or end time.
    hot_time=d.relative_time_s.where(d.meas_t1>=45)
    crossing=hot_time.groupby(d.experiment_id).cummin().groupby(d.experiment_id).ffill()
    d['since45']=d.relative_time_s-crossing
    d['t1_t2_max']=d[['meas_t1_max','meas_t2_max']].max(axis=1)
    return d.sort_index()


def rule1(d,p):
    p=p.copy();t=d.meas_t1_max.to_numpy(); valid=np.isfinite(t);hot=t>=45
    p[(p==1)&hot]=2;p[(p>=2)&~hot&valid]=1
    return p


def limits(fit):
    out={}
    for cc,d in fit.groupby('bat_cc'):
        out[float(cc)]=[float(d.loc[d.label>=3,'meas_t1_max'].min()),float(d.loc[d.label==2,'meas_t1_max'].max())]
    return out


def envelope(d,base,lim):
    p=base.copy();t=d.meas_t1_max.to_numpy()
    for cc,(lo,hi) in lim.items():
        eligible=(d.bat_cc.to_numpy()==cc)
        p[(base==3)&eligible&(t<lo)]=2
        p[(base==2)&eligible&(t>hi)]=3
    return p


def latch(d,base,seconds=10,confidence=0):
    p=base.copy();n=len(d)
    for ids in d.groupby('experiment_id',sort=False).indices.values():
        ids=ids[np.argsort(d.relative_time_s.to_numpy()[ids],kind='stable')]
        start=None;active=False
        prev=None
        for i in ids:
            t=float(d.relative_time_s.iat[i]);gap=prev is not None and t-prev>15;prev=t
            if base[i]==3 and d.p3.iat[i]>=confidence:
                if start is None or gap:start=t
                if t-start>=seconds:active=True
            else:start=None
            if active and base[i]==2:p[i]=3
    return p


def evaluate(d,p,base):
    y=d.label.to_numpy(int);f=f1_score(y,p,labels=range(1,7),average=None,zero_division=0)
    mask=d.experiment_id.to_numpy()!=E16
    return dict(macro_f1=float(f.mean()),F1_2=float(f[1]),F1_3=float(f[2]),
                noE16_f1=float(f1_score(y[mask],p[mask],labels=range(1,7),average='macro',zero_division=0)),
                true2_pred3=int(((y==2)&(p==3)).sum()),true3_pred2=int(((y==3)&(p==2)).sum()),
                corrected=int(((base!=y)&(p==y)).sum()),harmed=int(((base==y)&(p!=y)).sum()),
                changed=int((base!=p).sum()),other_predictions_changed=int(((~np.isin(base,[2,3]))&(base!=p)).sum()))


def boundary_table(d):
    out=[]
    for exp,g in d.groupby('experiment_id'):
        g=g.sort_values('relative_time_s');y=g.label.to_numpy();k=np.flatnonzero((y[:-1]==2)&(y[1:]==3))
        if not len(k):continue
        i=k[0]+1;r=g.iloc[i];prev=g.iloc[i-1]
        rec=dict(experiment_id=exp,**{c:r[c] for c in STATIC},onset_s=float(r.relative_time_s),nonmonotonic_labels=int((np.diff(y)<0).sum()))
        for c in ['meas_t1_max','meas_t2_max','thermal_t_max','t1_t2_max','meas_v','meas_p','since45']:
            rec[c]=r[c];rec[c+'_before']=prev[c]
        out.append(rec)
    return pd.DataFrame(out)


def main():
    OUT.mkdir(exist_ok=False)
    train=pd.read_csv(DATA/'train/train.csv');val=pd.read_csv(DATA/'validation/validation.csv')
    fit_i,dev_i=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261114).split(train,groups=train.experiment_id))
    fit=features(train.iloc[fit_i].reset_index(drop=True))
    frames={'dev':features(train.iloc[dev_i].reset_index(drop=True)),'validation':features(val)}
    boundaries=boundary_table(features(pd.concat([train,val],ignore_index=True)))
    boundaries.to_csv(OUT/'boundary_values.csv',index=False)
    lim=limits(fit);(OUT/'fit_limits.json').write_text(json.dumps(lim,indent=2))
    # A small, fixed sensor-only tree: no MLP, no B probability, no validation tuning.
    fcols=SENS+[c for c in fit if c.endswith(('_max','_delta_start','_rate30','_rate60','_minus_ambient'))]+['t1_minus_t2','since45']
    fcols=list(dict.fromkeys(fcols)); med=fit[fcols].median().fillna(0)
    pair=fit.label.isin([2,3]);trainpair=fit[pair]
    weights=1/trainpair.groupby(['experiment_id','label']).label.transform('size').to_numpy(float)
    tree=DecisionTreeClassifier(max_depth=3,min_samples_leaf=40,random_state=42)
    tree.fit(trainpair[fcols].fillna(med),trainpair.label,sample_weight=weights)
    (OUT/'sensor_tree.txt').write_text(export_text(tree,feature_names=fcols))
    scores=[];errrows=[];byexp=[];boundary_offsets=[];intervals=[];falseearly=[]
    for split,d in frames.items():
        pred=pd.read_csv(BASE/f'ensemble_B_{split}_predictions.csv')
        assert np.array_equal(d.sample_id,pred.sample_id) and np.array_equal(d.label,pred.label)
        for c in [f'p{i}' for i in range(1,7)]:d[c]=pred[c].to_numpy()
        base=rule1(d,pred.pred.to_numpy(int));d['base']=base
        variants={'rule1':base}
        fixed=base.copy();fixed[(base==3)&(d.meas_t1_max.to_numpy()<50)]=2
        variants['T1_below50']=fixed
        variants['T1_fit_capacity_envelope']=envelope(d,base,lim)
        for sec in [0,10,30]:variants[f'causal_latch_{sec}s']=latch(d,base,sec)
        variants['causal_latch_p3_90_10s']=latch(d,base,10,.9)
        variants['T1below50_latch10s']=latch(d,fixed,10)
        tp=base.copy();gate=np.isin(base,[2,3]);tp[gate]=tree.predict(d.loc[gate,fcols].fillna(med));variants['sensor_tree_depth3']=tp
        for name,p in variants.items():
            assert np.array_equal(p[~np.isin(base,[2,3])],base[~np.isin(base,[2,3])])
            scores.append(dict(split=split,method=name,**evaluate(d,p,base)))
            for e,ids in d.groupby('experiment_id',sort=True).indices.items():
                de=d.iloc[ids];row=evaluate(de,p[ids],base[ids]);byexp.append(dict(split=split,method=name,experiment_id=e,**row))
        for exp,ids in d.groupby('experiment_id',sort=True).indices.items():
            g=d.iloc[ids].sort_values('relative_time_s');y=g.label.to_numpy();p=g.base.to_numpy();t=g.relative_time_s.to_numpy(float)
            k=np.flatnonzero((y[:-1]==2)&(y[1:]==3));onset=t[k[0]+1] if len(k) else np.nan
            prev3=False;err23=((y==2)&(p==3))|((y==3)&(p==2))
            for j in np.flatnonzero(err23):
                errrows.append(dict(split=split,experiment_id=exp,sample_id=g.sample_id.iloc[j],true=int(y[j]),pred=int(p[j]),
                                    time_s=t[j],from_true_onset_s=t[j]-onset,
                                    had_predicted3_before=bool((p[:j]==3).any()),
                                    t1=float(g.meas_t1.iloc[j]),t1_max=float(g.meas_t1_max.iloc[j]),p3=float(g.p3.iloc[j])))
            kpred=np.flatnonzero(p==3)
            if len(kpred):
                boundary_offsets.append(dict(split=split,experiment_id=exp,first_B3_minus_true3_s=t[kpred[0]]-onset,
                                              max_p3_while_true2=float(g.loc[g.label==2,'p3'].max()),
                                              pair_errors=int(err23.sum())))
        # Per-experiment bootstrap for fixed-model changes, with no model refitting.
        ids=list(d.groupby('experiment_id',sort=True).indices.values());y=d.label.to_numpy(int)
        def cms(p):return np.stack([np.bincount((y[i]-1)*6+p[i]-1,minlength=36).reshape(6,6) for i in ids])
        def macro(c):
            den=c.sum(-1)+c.sum(-2)
            return np.divide(2*np.diagonal(c,axis1=-2,axis2=-1),den,out=np.zeros_like(den,dtype=float),where=den>0).mean(-1)
        draws=np.random.default_rng(20261115).integers(0,len(ids),(2000,len(ids)))
        cb=cms(base)
        for name,p in variants.items():
            delta=macro(cms(p)[draws].sum(1))-macro(cb[draws].sum(1));lo,hi=np.quantile(delta,[.025,.975])
            intervals.append(dict(split=split,method=name,lower=lo,upper=hi))
    for name,rows in [('scores',scores),('errors23',errrows),('per_experiment',byexp),('boundary_offsets',boundary_offsets),('bootstrap',intervals)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    # Standalone sensor tree generalization on Train: leave experiments out, never random rows.
    full=features(train);pair=full[full.label.isin([2,3])].reset_index(drop=True)
    cv=[]
    for fold,(a,b) in enumerate(GroupKFold(5).split(pair,groups=pair.experiment_id)):
        tr=pair.iloc[a];te=pair.iloc[b];m=tr[fcols].median().fillna(0)
        w=1/tr.groupby(['experiment_id','label']).label.transform('size').to_numpy(float)
        model=DecisionTreeClassifier(max_depth=3,min_samples_leaf=40,random_state=42)
        model.fit(tr[fcols].fillna(m),tr.label,sample_weight=w);p=model.predict(te[fcols].fillna(m))
        cv.append(dict(fold=fold,macro23=f1_score(te.label,p,labels=[2,3],average='macro'),accuracy=float((p==te.label).mean()),
                       false2to3=int(((te.label==2)&(p==3)).sum()),false3to2=int(((te.label==3)&(p==2)).sum())))
    pd.DataFrame(cv).to_csv(OUT/'sensor_tree_train_groupcv.csv',index=False)
    sources=[Path(__file__),DATA/'train/train.csv',DATA/'validation/validation.csv']+[BASE/f'ensemble_B_{s}_predictions.csv' for s in frames]
    (OUT/'manifest.json').write_text(json.dumps(dict(test_access=False,validation_used_for_fitting=False,
       official_validation='Previously inspected, exploratory; not new independent evidence',
       groupcv='Sensor-only diagnostic; original B was not retrained in outer folds',
       capacity_envelope_training='Original fit only for BOTH dev and validation',
       input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
    print(pd.DataFrame(scores).round(6).to_string(index=False))
    print('LABEL REVERSALS',int(boundaries.nonmonotonic_labels.sum()))
    print('CV',pd.DataFrame(cv).to_string(index=False))

if __name__=='__main__':main()
