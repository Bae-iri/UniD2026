"""Confirm frozen raw50 policy on existing alternate experiment splits."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from boost_model import fit,probability,combine,features
from validate import BackboneAudit,summary

SPEC=dict(name='raw_0.5',mode='raw',alpha=.5)


def work(args,suite,scope,data,back,global_frame,xglobal,rgbglobal,cache,visual=None):
    out=args.output/f'{suite}_{scope}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    with np.load(back/scope/'partitions.npz') as z:training,held=z['training'],z['held']
    lookup=pd.Index(global_frame.sample_id);loc=lookup.get_indexer(data.sample_id);assert (loc>=0).all()
    x=xglobal[loc];ff=data.iloc[training].reset_index(drop=True)
    model=fit(ff,x[training],None,'raw');joblib.dump(model,out/'model.joblib')
    audit=BackboneAudit(data);p=audit.load(back/scope/'full',training,set(data.experiment_id.iloc[held]))
    q=cache[(cache.suite==suite)&(cache.scope==scope)]
    idx=pd.Index(data.sample_id);rows=[]
    for view,g in q.groupby('view'):
        sub=idx.get_indexer(g.sample_id);f=data.iloc[sub].reset_index(drop=True)
        gloc=loc[sub];xx=x[sub] if view=='original' else features(f,rgbglobal[gloc])
        if view!='original' and visual is not None:xx=np.c_[xx,visual[gloc]]
        stages={k:g[k].to_numpy(int) for k in ['precision65','J096','S3']}
        prob=probability(model,xx,p[:,sub]);pred=combine(stages,p[:,sub],prob,SPEC)
        table=g[['sample_id','experiment_id','label','view']].copy();table['current']=g.S3.to_numpy(int);table['candidate']=pred
        rows.append(table)
    table=pd.concat(rows,ignore_index=True);table.to_csv(out/'predictions.csv',index=False)
    r={v:summary(g,'candidate',reference='current') for v,g in table.groupby('view')}
    (out/'complete.json').write_text(json.dumps(dict(suite=suite,scope=scope,metrics=r),indent=2)+'\n')
    print('CONFIRM',suite,scope,round(r['original']['delta'],6),flush=True)


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    global_frame=pd.read_csv(args.prepared/'frame.csv');x=np.load(args.prepared/'features.npy',mmap_mode='r');rgb=np.load(args.prepared/'rgb.npy',mmap_mode='r')
    visual=np.load(args.visual,mmap_mode='r') if args.visual else None
    if visual is not None:x=np.c_[x,visual].astype(np.float32)
    cache=pd.read_csv(args.existing/'outer_predictions.csv')
    train=pd.read_csv(args.train_backbones/'row_manifest.csv')
    # Raw official Train has all columns and exactly the same row ordering.
    train=global_frame.set_index('sample_id',drop=False).loc[train.sample_id].reset_index(drop=True)
    dev=pd.read_csv(args.repeated/'development.csv')
    tasks=[('train78',f'outer_{k}',train,args.train_backbones) for k in range(5)]
    tasks += [('repeat77',f'repeat_{r}_fold_{k}',dev,args.repeated/'backbones') for r in range(2) for k in range(5)]
    for i,(suite,scope,data,back) in enumerate(tasks):
        if i%args.workers==args.worker:work(args,suite,scope,data,back,global_frame,x,rgb,cache,visual)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','train-backbones','repeated','existing','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--visual',type=Path)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--worker',type=int,required=True)
    with threadpool_limits(limits=2):main(p.parse_args())
