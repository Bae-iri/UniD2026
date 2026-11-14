"""Aggregate-only A/B/C Validation error comparison. No training or Test access."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_experiment_performance import confusion, scores


def load_aligned(path, reference=None):
    f=pd.read_csv(path)
    required={'sample_id','experiment_id','label','pred'}
    if not required.issubset(f) or f.sample_id.isna().any() or not f.sample_id.is_unique:
        raise ValueError('Invalid prediction schema or IDs')
    if f.experiment_id.isna().any() or not f.label.isin(range(1,7)).all() or not f.pred.isin(range(1,7)).all():
        raise ValueError('Invalid group or labels')
    if reference is not None:
        if set(f.sample_id)!=set(reference.sample_id):raise ValueError('ID sets differ')
        f=f.set_index('sample_id').loc[reference.sample_id].reset_index()
        for col in ['experiment_id','label']:
            if not np.array_equal(f[col],reference[col]):raise ValueError('Targets/groups differ')
    cols=[f'p{i}' for i in range(1,7)]
    if set(cols).issubset(f):
        p=f[cols].to_numpy(float)
        if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5) or not np.array_equal(p.argmax(1)+1,f.pred):
            raise ValueError('Invalid probabilities/predictions')
    return f


def run(args):
    if args.output.exists():raise FileExistsError('Choose a new output directory')
    frames={}
    for m in ['A','B','C']:
        frames[m]=load_aligned(args.run/f'ensemble_{m}_validation_predictions.csv',frames.get('A'))
    ref=frames['A'];groups=list(ref.groupby('experiment_id',sort=True).indices.values())
    names=[f'E{i+1:02}' for i in range(len(groups))]
    cm={m:np.stack([confusion(f.iloc[idx].label,f.iloc[idx].pred) for idx in groups]) for m,f in frames.items()}
    seed_cms={}
    for m in frames:
        for seed in [42,43,44]:
            p=args.run/f'seed_{seed}/{m}/validation_predictions.csv'
            if p.exists():
                f=load_aligned(p,ref)
                seed_cms[m,seed]=np.stack([confusion(f.iloc[idx].label,f.iloc[idx].pred) for idx in groups])
    overall={m:scores(c.sum(0)) for m,c in cm.items()}
    saved=pd.read_csv(args.run/'ensemble_scores.csv')
    for m in frames:
        s=saved.loc[(saved.model==m)&(saved.split=='validation'),'macro_f1']
        if len(s)!=1 or not np.isclose(s.iloc[0],overall[m]['macro_f1_6'],atol=1e-12):
            raise ValueError('Does not match saved ensemble metric')
    rows=[];details={}
    for i,name in enumerate(names):
        row={'experiment':name,'rows':int(cm['A'][i].sum())}
        details[name]={}
        for m in frames:
            c=cm[m][i]; s=scores(c)
            details[name][m]=dict(**s,errors=int(c.sum()-np.trace(c)),
                seed_present_f1={str(seed):scores(v[i])['macro_f1_true_present'] for (model,seed),v in seed_cms.items() if model==m})
            row.update({f'{m}_present_f1':s['macro_f1_true_present'],f'{m}_macro_f1_6':s['macro_f1_6'],
                f'{m}_errors':int(c.sum()-np.trace(c)),f'{m}_5_to_4':int(c[4,3]),f'{m}_F1_5':s['class_f1'][4]})
        row['B_minus_A']=row['B_present_f1']-row['A_present_f1']
        row['B_minus_C']=row['B_present_f1']-row['C_present_f1']
        rows.append(row)
    w=np.random.default_rng(42).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    bootstrap={}
    for m,c in cm.items():
        b=np.einsum('be,eij->bij',w,c); den=b.sum(1)+b.sum(2)
        bootstrap[m]=np.divide(2*np.diagonal(b,axis1=1,axis2=2),den,out=np.zeros_like(den,float),where=den>0).mean(1)
    contrasts={}
    for m in ['A','C']:
        contrasts[f'B-{m}']={'pooled_f1_difference':overall['B']['macro_f1_6']-overall[m]['macro_f1_6'],
            'paired_experiment_bootstrap_95ci':np.quantile(bootstrap['B']-bootstrap[m],[.025,.975]).tolist(),
            'experiments_B_higher':sum(r['B_present_f1']>r[f'{m}_present_f1'] for r in rows),
            'experiments_B_lower':sum(r['B_present_f1']<r[f'{m}_present_f1'] for r in rows)}
    complements={}
    for m in ['A','C']:
        bc=frames['B'].pred.to_numpy()==ref.label.to_numpy(); other=frames[m].pred.to_numpy()==ref.label.to_numpy()
        complements[m]={'both_correct':int((bc&other).sum()),'B_only_correct':int((bc&~other).sum()),
            'other_only_correct':int((~bc&other).sum()),'both_wrong':int((~bc&~other).sum())}
    report=dict(source_run=str(args.run),rows=len(ref),experiments=len(groups),overall=overall,
        contrasts=contrasts,complementarity=complements,experiment_details=details,
        notes=['All models are existing 3-seed probability ensembles on identical Validation rows.',
               'Experiment aliases follow sorted experiment_id and match the prior B report.',
               'Present-class F1 excludes classes absent from ground truth; class composition still affects comparisons.',
               'Bootstrap is paired by experiment for fixed checkpoints; does not include retraining uncertainty.',
               'Complementarity is descriptive, not a realizable oracle or proof an ensemble improves.',
               'Only aggregate outputs; no row data, raw experiment IDs or Test access.'])
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    table=pd.DataFrame(rows).sort_values('B_present_f1')
    table.to_csv(args.output/'experiment_comparison.csv',index=False)
    classrows=[]
    for m,s in overall.items():
        for k in range(6):classrows.append(dict(model=m,label=k+1,f1=s['class_f1'][k],true_count=s['true_counts'][k],predicted_count=s['predicted_counts'][k]))
    pd.DataFrame(classrows).to_csv(args.output/'class_comparison.csv',index=False)
    print('OVERALL',json.dumps({m:{'macro_f1':s['macro_f1_6'],'class_f1':s['class_f1']} for m,s in overall.items()}))
    print('CONTRASTS',json.dumps(contrasts));print('COMPLEMENTARITY',json.dumps(complements))
    print(table.to_string(index=False))
    for name in ['E16','E14','E05']:
        print('DETAIL',name,json.dumps(details[name]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
