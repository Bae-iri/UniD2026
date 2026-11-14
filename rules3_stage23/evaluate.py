"""Audit rules3 residuals and fixed 2/3-only corrections, without MLP duplication.

Test is used only to reproduce the existing submission, never to select variants.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import final_rules3 as rules
from stage23_audit.analyze import latch
DATA=Path('/home/ubuntu/data/battery_data_light')
BASE=Path('/home/ubuntu/leewon/abc_runs/current_query_v1')
OUT=ROOT/'rules3_stage23/results'
EDGES=[45,48,50,52,54,56,58,60,63,66,70,75,80,90,1e9]


def odds_table(train):
    d=rules.prepare(train);d=d[d.label.isin([2,3])&(d.t1max>=45)].copy()
    d['bin']=np.digitize(d.t1max,EDGES)-1
    c=d.groupby(['bat_cc','bin']).label.agg(n3=lambda a:(a==3).sum(),n2=lambda a:(a==2).sum())
    prior=np.log(((d.label==3).sum()+1)/((d.label==2).sum()+1))
    return (np.log((c.n3+1)/(c.n2+1))-prior).to_dict()


def soft23(d,base,table,alpha=.5):
    b=np.digitize(d.t1max,EDGES)-1
    adjust=np.array([table.get((cc,int(k)),0.) for cc,k in zip(d.bat_cc,b)])*alpha
    p=base.copy();mask=np.isin(base,[2,3])
    logodds=np.log(d.p3.to_numpy()+1e-9)-np.log(d.p2.to_numpy()+1e-9)+adjust
    p[mask]=np.where(logodds[mask]>0,3,2)
    return p


def preserve_constraints(d,p,base,floor):
    # Corrections are restricted to the existing 2/3 gate; never promote into 4+.
    p=p.copy();gate=np.isin(base,[2,3]);p[~gate]=base[~gate]
    bound=d.bat_cc.map(floor[0]).fillna(floor[1]).to_numpy()
    p[(p==3)&(d.t1max.to_numpy()<bound)]=2
    assert np.array_equal(p[~gate],base[~gate])
    assert np.isin(p[gate],[2,3]).all()
    return p


def confusion(y,p):
    return np.bincount((np.asarray(y)-1)*6+np.asarray(p)-1,minlength=36).reshape(6,6)


def macro(c):
    den=c.sum(-1)+c.sum(-2)
    return np.divide(2*np.diagonal(c,axis1=-2,axis2=-1),den,out=np.zeros_like(den,dtype=float),where=den>0).mean(-1)


def metrics(d,p,base):
    y=d.label.to_numpy(int);f=f1_score(y,p,labels=range(1,7),average=None,zero_division=0)
    error=y!=p;pair=((y==2)&(p==3))|((y==3)&(p==2))
    return dict(macro_f1=float(f.mean()),rows=len(d),errors=int(error.sum()),pair23_errors=int(pair.sum()),
      pair23_error_share=float(pair.sum()/error.sum()) if error.any() else 0.,
      true2_pred3=int(((y==2)&(p==3)).sum()),true3_pred2=int(((y==3)&(p==2)).sum()),
      corrected=int(((base!=y)&(p==y)).sum()),harmed=int(((base==y)&(p!=y)).sum()),
      changed=int((base!=p).sum()),**{f'F1_{i+1}':float(v) for i,v in enumerate(f)})


def bootstrap(d,p,base):
    groups=list(d.groupby('experiment_id').indices.values());y=d.label.to_numpy(int)
    a=np.stack([confusion(y[i],base[i]) for i in groups]);b=np.stack([confusion(y[i],p[i]) for i in groups])
    draw=np.random.default_rng(20261115).integers(0,len(groups),(2000,len(groups)))
    delta=macro(b[draw].sum(1))-macro(a[draw].sum(1))
    return np.quantile(delta,[.025,.975]).tolist()


def variants(d,base,floor,table,upper):
    hold=latch(d,base,10)
    # Baseline floor may have been computed from more than one sensor; respect it explicitly.
    soft=preserve_constraints(d,soft23(d,base,table),base,floor)
    cap=base.copy();bound=d.bat_cc.map(upper).to_numpy()
    cap[(base==2)&np.isfinite(bound)&(d.t1max.to_numpy()>bound)]=3
    combined=hold.copy()
    combined[cap!=base]=cap[cap!=base]
    return {'rules3':base,'latch10':preserve_constraints(d,hold,base,floor),
            'soft_T1_capacity_a05':soft,'T1_capacity_upper_q995':cap,
            'latch10_plus_upper_q995':preserve_constraints(d,combined,base,floor),
            'soft_T1_a05_latch10':preserve_constraints(d,latch(d,soft,10),base,floor)}


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    OUT=parser.parse_args().output
    OUT.mkdir(exist_ok=False)
    train=pd.read_csv(DATA/'train/train.csv');val=pd.read_csv(DATA/'validation/validation.csv')
    fi,di=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261114).split(train,groups=train.experiment_id))
    fit=train.iloc[fi].reset_index(drop=True);dev=train.iloc[di].reset_index(drop=True)
    manifest=pd.read_csv(BASE/'split_manifest.csv')
    for split,frame in [('fit',fit),('dev',dev),('validation',val)]:
        assert np.array_equal(frame.sample_id,manifest.loc[manifest.split==split,'sample_id'])
    floor=rules.t1_floor(fit);table=odds_table(fit)
    prepared=rules.prepare(fit);upper=prepared[prepared.label<3].groupby('bat_cc').t1max.quantile(.995).to_dict()
    (OUT/'fit_thresholds.json').write_text(json.dumps({'floor':floor,'upper_q995':upper,'edges':EDGES,'alpha':.5},indent=2))
    summary=[];counts=[];byexp=[];changes=[];stages=[];sensitivity=[];probabilities={}
    for split,frame in [('dev',dev),('validation',val)]:
        pr=pd.read_csv(BASE/f'ensemble_B_{split}_predictions.csv')
        assert np.array_equal(pr.sample_id,frame.sample_id) and np.array_equal(pr.label,frame.label)
        d=rules.prepare(frame.merge(pr[['sample_id','pred']+[f'p{i}' for i in range(1,7)]],on='sample_id',validate='1:1'))
        base=rules.apply_rules(d,d.pred.to_numpy(int),floor)
        available=variants(d,base,floor,table,upper)
        row=d[['sample_id','experiment_id','label']].copy()
        for name,p in available.items():row[name]=p
        row.to_csv(OUT/f'{split}_predictions.csv',index=False)
        for name,p in available.items():
            for scope,mask in [('all',np.ones(len(d),bool)),('noE16',(d.experiment_id!=rules.E16).to_numpy())]:
                if split=='dev' and scope=='noE16':continue
                dd=d.loc[mask].reset_index(drop=True);y=dd.label.to_numpy(int);pp=p[mask];bp=base[mask]
                lo,hi=bootstrap(dd,pp,bp)
                summary.append(dict(split=split,scope=scope,method=name,delta_lower=lo,delta_upper=hi,**metrics(dd,pp,bp)))
                for (a,b),n in pd.DataFrame({'true':y,'pred':pp}).groupby(['true','pred']).size().items():
                    counts.append(dict(split=split,scope=scope,method=name,true=int(a),pred=int(b),rows=int(n)))
                for label in range(1,7):
                    true=y==label
                    stages.append(dict(split=split,scope=scope,method=name,label=label,support=int(true.sum()),
                       wrong=int((true&(y!=pp)).sum()),false_positives=int(((~true)&(pp==label)).sum())))
            for exp,ids in d.groupby('experiment_id').indices.items():
                byexp.append(dict(split=split,method=name,experiment_id=exp,**metrics(d.iloc[ids],p[ids],base[ids])))
            for i in np.flatnonzero(p!=base):
                changes.append(dict(split=split,method=name,sample_id=d.sample_id.iat[i],experiment_id=d.experiment_id.iat[i],
                                    true=int(d.label.iat[i]),before=int(base[i]),after=int(p[i]),
                                    time_s=float(d.relative_time_s.iat[i]),t1max=float(d.t1max.iat[i]),
                                    rule2_routed_to3=bool(d.pred.iat[i]>=4 and not d.vdrop.iat[i])))
        # Explicitly expose the previously inconsistent threshold-fit population.
        other=rules.t1_floor(train)
        alternate=rules.apply_rules(d,d.pred.to_numpy(int),other)
        sensitivity.append(dict(split=split,source='train78_including_dev_NOT_independent_for_dev',changed=int((alternate!=base).sum()),
                                **{k:v for k,v in metrics(d,alternate,base).items() if k!='changed'}))
        # Individual seed and sampling-interval checks of the fixed rule, not a fresh training CV.
        seedrows=[]
        for seed in [42,43,44]:
            ref=pd.read_csv(BASE/f'seed_{seed}/B/{split}_predictions.csv').set_index('sample_id').loc[d.sample_id]
            b=rules.apply_rules(d,ref.pred.to_numpy(int),floor)
            v=preserve_constraints(d,latch(d,b,10),b,floor)
            seedrows.append(dict(split=split,seed=seed,**metrics(d,v,b)))
        pd.DataFrame(seedrows).to_csv(OUT/f'{split}_seed_stress.csv',index=False)
    for name,rows in [('scores',summary),('confusion_counts',counts),('per_experiment',byexp),('changes',changes),('per_stage',stages),('threshold_source_sensitivity',sensitivity)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    # Reproduce the existing rules3 submission using its original deployment threshold population.
    labeled=pd.concat([train,val],ignore_index=True);deployment_floor=rules.t1_floor(labeled[labeled.experiment_id!=rules.E16])
    test=pd.read_csv(DATA/'test/test.csv');original=pd.read_csv(BASE/'submission_B.csv').rename(columns={'label':'B'})
    d=rules.prepare(test.merge(original,on='sample_id',validate='1:1'))
    p=rules.apply_rules(d,d.B.to_numpy(),deployment_floor)
    sample=pd.read_csv(DATA/'sample_submission.csv')[['sample_id']]
    sample['label']=sample.sample_id.map(pd.Series(p,index=d.sample_id)).astype(int)
    reference=pd.read_csv(ROOT/'final_submit/submission_rules3.csv')
    pd.testing.assert_frame_equal(sample,reference)
    sample.to_csv(OUT/'reproduced_rules3.csv',index=False)
    assert (OUT/'reproduced_rules3.csv').read_bytes()==(ROOT/'final_submit/submission_rules3.csv').read_bytes()
    metadata=dict(baseline_submission_reproduced_byte_identical=True,test_use='Existing baseline reproduction only; no new candidate generated',
      threshold_fit_for_both_splits='Original fit62 only',validation_selected_hypotheses=True,
      explanation='These are exploratory checks on previously inspected data, not fresh outer CV.',
      variants=list(available),submission_replaced=False,
      source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'final_rules3.py',ROOT/'final_submit/submission_rules3.csv',DATA/'train/train.csv',DATA/'validation/validation.csv']})
    (OUT/'manifest.json').write_text(json.dumps(metadata,indent=2))
    print(pd.DataFrame(summary)[['split','scope','method','macro_f1','errors','pair23_errors','corrected','harmed','delta_lower','delta_upper']].round(6).to_string(index=False))

if __name__=='__main__':main()
