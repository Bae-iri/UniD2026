"""Frozen ImageNet + causal multimodal conditional expert screening."""
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from boost_model import fit,probability,combine,choose,SPECS,features
from pipeline import fit_frozen_recipe,predict_stages
from validate import BackboneAudit,matrix,macro,VIEWS,subsample,summary
from reproduce import sha


def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def select(scores,kind):
    use=[i for i,s in enumerate(SPECS) if kind=='performance' or s['alpha']<=.5]
    selected,detail=choose(scores[:,use]) if len(use)==len(SPECS) else (None,None)
    if selected is not None:return selected,detail
    means=scores.mean(0);best=max(means[i] for i in use)
    eligible=[i for i in use if means[i]>=best-.0015]
    i=min(eligible,key=lambda i:(SPECS[i]['alpha'],SPECS[i]['mode']=='stack',-means[i]))
    return SPECS[i],dict(selected=SPECS[i]['name'],means={SPECS[j]['name']:float(means[j]) for j in use},allowance=.0015)


def ready(folder):
    paths=[folder/f'single_{j}'/f'seed_{s}/complete.json' for j in range(3) for s in [42,43,44]]
    return all(p.exists() and p.stat().st_size for p in paths)


def work(args,k,frame,x,rgb,visual):
    dest=args.output/f'outer_{k}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    folder=args.backbones/f'outer_{k}';sf=args.singles/f'outer_{k}'
    while not ready(sf):
        print('WAIT single teachers',k,flush=True);time.sleep(5)
    with np.load(folder/'partitions.npz') as z:
        training,held=z['training'],z['held'];parts=[z[f'inner_{j}'] for j in range(3)]
    held_exps=set(frame.experiment_id.iloc[held]);audit=BackboneAudit(frame)
    single=[audit.load(sf/f'single_{j}',parts[j],held_exps) for j in range(3)]
    leave=[audit.load(folder/f'leave_{j}',np.setdiff1d(training,parts[j]),held_exps|set(frame.experiment_id.iloc[parts[j]])) for j in range(3)]
    scores=np.zeros((3,len(SPECS)));decisions=[]
    for j in range(3):
        inner_held=parts[j];inner_fit=np.setdiff1d(training,inner_held)
        pfit=np.full((3,len(frame),6),np.nan)
        for m in range(3):
            if m==j:continue
            teacher=next(a for a in range(3) if a not in [j,m])
            assert not set(frame.experiment_id.iloc[parts[teacher]]) & (held_exps|set(frame.experiment_id.iloc[inner_held])|set(frame.experiment_id.iloc[parts[m]]))
            pfit[:,parts[m]]=single[teacher][:,parts[m]]
        ff=frame.iloc[inner_fit].reset_index(drop=True);hf=frame.iloc[inner_held].reset_index(drop=True)
        model=fit_frozen_recipe(ff,pfit[:,inner_fit])
        stages=predict_stages(model,hf,leave[j][:,inner_held])
        preds={}
        for mode in ['raw','stack']:
            expert=fit(ff,x[inner_fit],pfit[:,inner_fit],mode)
            preds[mode]=probability(expert,x[inner_held],leave[j][:,inner_held])
        for i,spec in enumerate(SPECS):
            p=stages['S3'] if spec['alpha']==0 else combine(stages,leave[j][:,inner_held],preds[spec['mode']],spec)
            scores[j,i]=float(macro(matrix(hf.label,p)))
        print('INNER',k,j,dict(zip([s['name'] for s in SPECS],scores[j].round(4).tolist())),flush=True)
    policies={kind:select(scores,kind) for kind in ['conservative','performance']}
    write(dest/'selection.json',{kind:dict(spec=spec,detail=detail) for kind,(spec,detail) in policies.items()})
    pd.DataFrame(scores,columns=[s['name'] for s in SPECS]).to_csv(dest/'inner_scores.csv',index=False)
    pfit=np.full((3,len(frame),6),np.nan)
    for j in range(3):pfit[:,parts[j]]=leave[j][:,parts[j]]
    ff=frame.iloc[training].reset_index(drop=True);hf=frame.iloc[held].reset_index(drop=True)
    pp=audit.load(folder/'full',training,held_exps)
    baseline=joblib.load(args.existing/f'all97_outer_{k}.joblib')
    assert set(baseline['fit_experiments'])==set(ff.experiment_id)
    models={mode:fit(ff,x[training],pfit[:,training],mode) for mode in ['raw','stack']}
    joblib.dump(dict(models=models,policies=policies,fit_experiments=sorted(set(ff.experiment_id))),dest/'models.joblib')
    rows=[]
    for view in VIEWS:
        sub=subsample(hf,view);f=hf.iloc[sub].reset_index(drop=True);global_idx=held[sub]
        xx=x[global_idx] if view=='original' else np.c_[features(f,rgb[global_idx]),visual[global_idx]]
        p=pp[:,global_idx];stages=predict_stages(baseline,f,p)
        preds={mode:probability(models[mode],xx,p) for mode in models}
        table=f[['sample_id','experiment_id','label']].copy();table['fold'],table['view']=k,view;table['current']=stages['S3']
        for kind,(spec,_) in policies.items():
            table[kind]=stages['S3'] if spec['alpha']==0 else combine(stages,p,preds[spec['mode']],spec)
        # Finite-policy diagnostics; do not choose a policy using these outer labels.
        if view=='original':
            for spec in SPECS[1:]:table[spec['name']]=combine(stages,p,preds[spec['mode']],spec)
        rows.append(table)
    table=pd.concat(rows,ignore_index=True);table.to_csv(dest/'predictions.csv',index=False)
    q=table[table.view=='original'];result={kind:summary(q,kind,reference='current') for kind in policies}
    write(dest/'complete.json',dict(fold=k,result=result,backbones_audited=len(audit.records)))
    print('OUTER COMPLETE',k,{kind:(round(v['macro_f1'],6),round(v['delta'],6)) for kind,v in result.items()},flush=True)


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(args.prepared/'frame.csv');x=np.load(args.prepared/'features.npy',mmap_mode='r');rgb=np.load(args.prepared/'rgb.npy',mmap_mode='r')
    visual=np.load(args.visual,mmap_mode='r');assert len(visual)==len(frame)
    x=np.c_[x,visual].astype(np.float32)
    for k in range(5):
        if k%args.workers==args.worker:work(args,k,frame,x,rgb,visual)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','backbones','singles','existing','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--visual',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--worker',type=int,required=True)
    args=p.parse_args()
    with threadpool_limits(limits=2):main(args)
