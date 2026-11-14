"""Aggregate saved ABC Validation predictions by experiment. Never exports rows."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def confusion(y, p):
    cm=np.zeros((6,6),dtype=np.int64)
    np.add.at(cm,(np.asarray(y,dtype=int)-1,np.asarray(p,dtype=int)-1),1)
    return cm


def scores(cm):
    support=cm.sum(1); predicted=cm.sum(0); den=support+predicted
    f=np.divide(2*np.diag(cm),den,out=np.zeros(6,float),where=den>0)
    return dict(macro_f1_6=float(f.mean()),macro_f1_true_present=float(f[support>0].mean()),
                accuracy=float(np.trace(cm)/cm.sum()),class_f1=f.tolist(),
                true_counts=support.tolist(),predicted_counts=predicted.tolist(),confusion=cm.tolist())


def run(args):
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    # Reading happens only in this local script; exports below contain aggregates only.
    frame=pd.read_csv(args.run/'ensemble_B_validation_predictions.csv')
    required={'sample_id','experiment_id','label','pred'}
    if not required.issubset(frame) or frame.sample_id.isna().any() or not frame.sample_id.is_unique:
        raise ValueError('Invalid prediction table')
    if frame.experiment_id.isna().any() or not frame.label.isin(range(1,7)).all() or not frame.pred.isin(range(1,7)).all():
        raise ValueError('Invalid group or class')
    probcols=[f'p{i}' for i in range(1,7)]
    if set(probcols).issubset(frame):
        probabilities=frame[probcols].to_numpy(float)
        if not np.isfinite(probabilities).all() or not np.array_equal(probabilities.argmax(1)+1,frame.pred.to_numpy()):
            raise ValueError('Saved probabilities/predictions disagree')
    pooled=confusion(frame.label,frame.pred)
    experiments=[];cms=[];groups=list(frame.groupby('experiment_id',sort=True))
    per_seed={}
    for seed in [42,43,44]:
        p=args.run/f'seed_{seed}/B/validation_predictions.csv'
        if not p.exists():continue
        f=pd.read_csv(p)
        if not f.sample_id.is_unique or set(f.sample_id)!=set(frame.sample_id):raise ValueError('Seed IDs differ')
        f=f.set_index('sample_id').loc[frame.sample_id].reset_index()
        if not np.array_equal(f.label,frame.label) or not np.array_equal(f.experiment_id,frame.experiment_id):
            raise ValueError('Seed target/group alignment differs')
        per_seed[seed]=f
    for index,(experiment,g) in enumerate(groups,1):
        # Fixed aliases avoid exporting actual experiment identifiers.
        alias=f'E{index:02}'
        cm=confusion(g.label,g.pred);cms.append(cm)
        info=scores(cm)
        errors=cm.copy();np.fill_diagonal(errors,0)
        pairs=sorted([(int(errors[i,j]),i+1,j+1) for i in range(6) for j in range(6) if errors[i,j]>0],reverse=True)
        seed_scores={str(seed):scores(confusion(f.loc[g.index,'label'],f.loc[g.index,'pred']))['macro_f1_true_present'] for seed,f in per_seed.items()}
        experiments.append(dict(experiment=alias,rows=len(g),errors=int(cm.sum()-np.trace(cm)),
            true_classes=(np.flatnonzero(cm.sum(1))+1).tolist(),**info,
            pooled_macro_f1_without_this_experiment=scores(pooled-cm)['macro_f1_6'],
            dominant_confusions=[dict(count=n,true_class=i,predicted_class=j) for n,i,j in pairs[:3]],
            seed_macro_f1_true_present=seed_scores))
    stack=np.stack(cms)
    weights=np.random.default_rng(42).multinomial(len(cms),np.full(len(cms),1/len(cms)),size=2000)
    boot=np.einsum('be,eij->bij',weights,stack)
    den=boot.sum(1)+boot.sum(2)
    f=np.divide(2*np.diagonal(boot,axis1=1,axis2=2),den,out=np.zeros_like(den,dtype=float),where=den>0).mean(1)
    overall=scores(pooled)
    ranked=sorted(experiments,key=lambda r:r['macro_f1_true_present'])
    totalerrors=sum(r['errors'] for r in ranked)
    report=dict(source_run=str(args.run),model='B',prediction='three-seed probability ensemble',
        rows=len(frame),experiments=len(experiments),overall=overall,
        fixed_model_experiment_bootstrap_macro_f1_95ci=np.quantile(f,[.025,.975]).tolist(),
        equal_experiment_mean_macro_f1_true_present=float(np.mean([r['macro_f1_true_present'] for r in experiments])),
        worst_five_error_fraction=float(sum(r['errors'] for r in ranked[:5])/totalerrors) if totalerrors else 0,
        groups=ranked,notes=[
          'Aliases E01.. are assigned by sorted experiment_id; no actual IDs or row data exported.',
          'macro_f1_6 includes zero for absent classes; true_present averages only ground-truth-present classes.',
          'Present-class F1 is also affected by class composition and false positives; inspect confusion counts.',
          'Pooled whole-Validation macro F1 differs from an equal-weight mean of experiment F1.',
          'Leave-one-experiment-out scores are descriptive sensitivity, not a reason to drop difficult data.',
          'Bootstrap resamples fixed-model experiments; not training uncertainty or a Test score interval.'])
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    rows=[]
    for r in ranked:
        rows.append(dict(experiment=r['experiment'],rows=r['rows'],errors=r['errors'],true_classes=','.join(map(str,r['true_classes'])),
            macro_f1_6=r['macro_f1_6'],macro_f1_true_present=r['macro_f1_true_present'],accuracy=r['accuracy'],
            pooled_macro_f1_without_this_experiment=r['pooled_macro_f1_without_this_experiment'],
            **{f'F1_{i+1}':r['class_f1'][i] for i in range(6)},
            **{f'true_count_{i+1}':r['true_counts'][i] for i in range(6)},
            **{f'pred_count_{i+1}':r['predicted_counts'][i] for i in range(6)}))
    pd.DataFrame(rows).to_csv(args.output/'experiment_summary.csv',index=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ['groups','notes']},indent=2))
    print(pd.DataFrame(rows)[['experiment','rows','errors','true_classes','macro_f1_true_present','accuracy','pooled_macro_f1_without_this_experiment']].to_string(index=False))
    for r in ranked[:5]:print(r['experiment'],'top_confusions',r['dominant_confusions'],'seed_scores',r['seed_macro_f1_true_present'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
