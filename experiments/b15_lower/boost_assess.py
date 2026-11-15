"""Report fixed raw50 policy; outer policy selection makes this CV, not blind."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from boost_model import features,probability,combine
from pipeline import predict_stages
from validate import summary,subsample,VIEWS

SPEC=dict(name='raw_0.5',mode='raw',alpha=.5)


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    frame=pd.read_csv(args.prepared/'frame.csv');x=np.load(args.prepared/'features.npy',mmap_mode='r');rgb=np.load(args.prepared/'rgb.npy',mmap_mode='r')
    visual=np.load(args.visual,mmap_mode='r') if args.visual else None
    if visual is not None:x=np.c_[x,visual].astype(np.float32)
    rows=[]
    for k in range(5):
        folder=args.screen/f'outer_{k}';assert (folder/'complete.json').exists()
        model=joblib.load(folder/'models.joblib')['models']['raw']
        back=args.backbones/f'outer_{k}'
        with np.load(back/'partitions.npz') as z:training,held=z['training'],z['held']
        assert set(model['fit_experiments'])==set(frame.experiment_id.iloc[training])
        assert not set(model['fit_experiments'])&set(frame.experiment_id.iloc[held])
        probs=np.stack([np.load(back/f'full/seed_{seed}/probabilities.npy') for seed in [42,43,44]])
        baseline=joblib.load(args.existing/f'all97_outer_{k}.joblib')
        hf=frame.iloc[held].reset_index(drop=True)
        for view in VIEWS:
            sub=subsample(hf,view);gl=held[sub];f=hf.iloc[sub].reset_index(drop=True)
            xx=x[gl] if view=='original' else features(f,rgb[gl])
            if view!='original' and visual is not None:xx=np.c_[xx,visual[gl]]
            p=probs[:,gl];stages=predict_stages(baseline,f,p)
            pred=combine(stages,p,probability(model,xx,p),SPEC)
            tab=f[['sample_id','experiment_id','label','bat_cc','soc_status','tr_method']].copy()
            tab['fold']=k;tab['view']=view;tab['current']=stages['S3'];tab['candidate']=pred;rows.append(tab)
    table=pd.concat(rows,ignore_index=True);table.to_csv(args.output/'predictions.csv',index=False)
    metrics={v:summary(g,'candidate',reference='current') for v,g in table.groupby('view')}
    q=table[table.view=='original'];folds={str(k):summary(g,'candidate',reference='current') for k,g in q.groupby('fold')}
    strata={}
    for col in ['bat_cc','soc_status','tr_method']:
        strata[col]={str(k):summary(g,'candidate',reference='current') for k,g in q.groupby(col)}
    report=dict(policy=SPEC,outer_used_for_finite_policy_selection=True,unbiased_nested_selector_in_screen_complete_json=True,
                primary=metrics,folds=folds,strata=strata)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('ASSESS',args.screen,[(v,round(m['delta'],6)) for v,m in metrics.items()],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','screen','backbones','existing','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--visual',type=Path)
    with threadpool_limits(limits=2):main(p.parse_args())
