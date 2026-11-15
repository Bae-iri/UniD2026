"""Fixed experiment-subset ensemble audit; no additional threshold search."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from boost_model import features,probability,fit_bagged
from boost_family import policies
from validate import summary


def main(args):
    root=args.root;global_f=pd.read_csv(root/'prepared/frame.csv');lookup=pd.Index(global_f.sample_id)
    x=np.load(root/'prepared/features.npy',mmap_mode='r');rgb=np.load(root/'prepared/rgb.npy',mmap_mode='r')
    visual=np.load(root/'visual_exact/visual.npy',mmap_mode='r');cache=pd.read_csv(args.existing/'outer_predictions.csv')
    tasks=[]
    for suite in ['all97','train78','repeat77']:
        if suite=='all97':
            data=global_f;back=args.all97
            modelpaths=[(f'outer_{k}',root/'screen'/f'outer_{k}/models.joblib',root/'screen_visual'/f'outer_{k}/models.joblib') for k in range(5)]
        elif suite=='train78':
            data=global_f[global_f._official_split.eq('train')].reset_index(drop=True);back=args.train_backbones
            modelpaths=[(f'outer_{k}',root/'confirm_raw50'/f'train78_outer_{k}/model.joblib',root/'confirm_visual50'/f'train78_outer_{k}/model.joblib') for k in range(5)]
        else:
            data=pd.read_csv(args.repeated/'development.csv');back=args.repeated/'backbones'
            modelpaths=[(f'repeat_{r}_fold_{k}',root/'confirm_raw50'/f'repeat77_repeat_{r}_fold_{k}/model.joblib',root/'confirm_visual50'/f'repeat77_repeat_{r}_fold_{k}/model.joblib') for r in range(2) for k in range(5)]
        for scope,mraw,mvis in modelpaths:tasks.append((suite,scope,data,back,mraw,mvis))
    args.output.mkdir(parents=True,exist_ok=True)
    for i,(suite,scope,data,back,mraw,mvis) in enumerate(tasks):
        if i%args.workers!=args.worker:continue
        out=args.output/f'{suite}_{scope}';out.mkdir(exist_ok=True)
        if (out/'complete.json').exists():continue
        with np.load(back/scope/'partitions.npz') as z:training,held=z['training'],z['held']
        ff=data.iloc[training].reset_index(drop=True);loc=lookup.get_indexer(ff.sample_id);assert (loc>=0).all()
        raw=joblib.load(mraw);vis=joblib.load(mvis)
        if suite=='all97':raw=raw['models']['raw'];vis=vis['models']['raw']
        models=dict(raw=fit_bagged(ff,x[loc],raw),visual=fit_bagged(ff,np.c_[x[loc],visual[loc]],vis))
        for model in models.values():
            for member in model['members']:
                assert not set(member['fit_experiments'])&set(data.experiment_id.iloc[held])
        joblib.dump(models,out/'models.joblib')
        pp=np.stack([np.load(back/scope/f'full/seed_{seed}/probabilities.npy') for seed in [42,43,44]])
        idx=pd.Index(data.sample_id);q=cache[(cache.suite==suite)&(cache.scope==scope)];rows=[]
        for view,g in q.groupby('view'):
            ids=idx.get_indexer(g.sample_id);assert (ids>=0).all();f=data.iloc[ids].reset_index(drop=True)
            gloc=lookup.get_indexer(f.sample_id);xx=x[gloc] if view=='original' else features(f,rgb[gloc]);vx=np.c_[xx,visual[gloc]]
            p=pp[:,ids];stages={name:g[name].to_numpy(int) for name in ['precision65','J096','S3']}
            preds=policies(stages,p,probability(models['raw'],xx,p),probability(models['visual'],vx,p))
            tab=g[['sample_id','experiment_id','label','bat_cc','soc_status','tr_method','view','scope']].copy();tab['current']=stages['S3']
            for name,values in preds.items():tab[name]=values
            rows.append(tab)
        table=pd.concat(rows,ignore_index=True);table.to_csv(out/'predictions.csv',index=False)
        metrics={name:summary(table[table.view=='original'],name,reference='current') for name in preds}
        (out/'complete.json').write_text(json.dumps(dict(suite=suite,scope=scope,metrics=metrics),indent=2)+'\n')
        print('BAGGED',suite,scope,{name:round(v['delta'],6) for name,v in metrics.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','existing','all97','train-backbones','repeated','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--worker',type=int,required=True)
    with threadpool_limits(limits=2):main(p.parse_args())
