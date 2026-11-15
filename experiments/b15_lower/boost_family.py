"""Compare frozen raw50, visual50 and two deterministic combinations by groups.

These are previously explored experiments. Comparing finite policies on outer
predictions is cross-validation model selection, not an untouched test estimate.
"""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from boost_model import features,probability,combine
from pipeline import predict_stages
from validate import summary

SPEC=dict(name='raw_0.5',mode='raw',alpha=.5)


def policies(stages,p,raw,visual):
    a=combine(stages,p,raw,SPEC);b=combine(stages,p,visual,SPEC)
    mean=combine(stages,p,(raw+visual)/2,SPEC)
    agree=np.where(a==b,a,stages['S3'])
    return dict(raw50=a,visual50=b,average50=mean,agreement50=agree)


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(policies=['raw50','visual50','average50','agreement50'],
        alpha=.5,average_weights=[.5,.5],agreement_fallback='current',leaderboard_used=False,
        statistical_scope='cross-validation selection; not blind; reused experiments'),indent=2)+'\n')
    root=args.root;prepared=root/'prepared';global_f=pd.read_csv(prepared/'frame.csv')
    x=np.load(prepared/'features.npy',mmap_mode='r');rgb=np.load(prepared/'rgb.npy',mmap_mode='r')
    visual=np.load(root/'visual_exact/visual.npy',mmap_mode='r');lookup=pd.Index(global_f.sample_id)
    cache=pd.read_csv(args.existing/'outer_predictions.csv');records=[]
    for suite in ['all97','train78','repeat77']:
        rows=[]
        if suite=='all97':
            tasks=[(f'outer_{k}',args.all97,root/'screen'/f'outer_{k}'/'models.joblib',root/'screen_visual'/f'outer_{k}'/'models.joblib') for k in range(5)]
            data=global_f
        elif suite=='train78':
            data=global_f[global_f._official_split.eq('train')].reset_index(drop=True)
            tasks=[(f'outer_{k}',args.train_backbones,root/'confirm_raw50'/f'train78_outer_{k}'/'model.joblib',root/'confirm_visual50'/f'train78_outer_{k}'/'model.joblib') for k in range(5)]
        else:
            data=pd.read_csv(args.repeated/'development.csv')
            tasks=[(f'repeat_{r}_fold_{k}',args.repeated/'backbones',root/'confirm_raw50'/f'repeat77_repeat_{r}_fold_{k}'/'model.joblib',root/'confirm_visual50'/f'repeat77_repeat_{r}_fold_{k}'/'model.joblib') for r in range(2) for k in range(5)]
        index=pd.Index(data.sample_id)
        for scope,back,mraw,mvis in tasks:
            raw=joblib.load(mraw);vis=joblib.load(mvis)
            if suite=='all97':raw=raw['models']['raw'];vis=vis['models']['raw']
            with np.load(back/scope/'partitions.npz') as z:training,held=z['training'],z['held']
            for model in [raw,vis]:
                assert set(model['fit_experiments'])==set(data.experiment_id.iloc[training])
                assert not set(model['fit_experiments'])&set(data.experiment_id.iloc[held])
            pp=np.stack([np.load(back/scope/f'full/seed_{seed}/probabilities.npy') for seed in [42,43,44]])
            groups=cache[(cache.suite==suite)&(cache.scope==scope)]
            for view,g in groups.groupby('view'):
                loc=index.get_indexer(g.sample_id);assert (loc>=0).all();f=data.iloc[loc].reset_index(drop=True)
                gloc=lookup.get_indexer(g.sample_id);assert (gloc>=0).all()
                xx=x[gloc] if view=='original' else features(f,rgb[gloc]);vx=np.c_[xx,visual[gloc]]
                p=pp[:,loc];stages={key:g[key].to_numpy(int) for key in ['precision65','J096','S3']}
                preds=policies(stages,p,probability(raw,xx,p),probability(vis,vx,p))
                tab=g[['sample_id','experiment_id','label','bat_cc','soc_status','tr_method','view','scope']].copy()
                tab['current']=stages['S3']
                for name,values in preds.items():tab[name]=values
                rows.append(tab)
            print('FAMILY',suite,scope,flush=True)
        table=pd.concat(rows,ignore_index=True);table.to_csv(args.output/f'{suite}_predictions.csv',index=False)
        metrics={method:{v:summary(g,method,reference='current') for v,g in table.groupby('view')} for method in preds}
        q=table[table.view=='original']
        folds={method:{s:summary(g,method,reference='current') for s,g in q.groupby('scope')} for method in preds}
        records.append(dict(suite=suite,metrics=metrics,folds=folds))
        print('POOLED',suite,{m:round(v['original']['delta'],6) for m,v in metrics.items()},flush=True)
    (args.output/'report.json').write_text(json.dumps(dict(suites=records),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','existing','all97','train-backbones','repeated','output']:p.add_argument('--'+name,type=Path,required=True)
    with threadpool_limits(limits=2):main(p.parse_args())
