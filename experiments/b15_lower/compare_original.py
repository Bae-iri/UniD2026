"""Direct original43 versus current S3 comparison; no fitting or tuning."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from core.precision import matrix,macro
from pipeline import predict_stages
from reproduce import load_artifact,load_probs,sha
from validate import summary


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def metrics(y,p):
    cm=matrix(y,p);tp=int(cm[5,5]);fp=int(cm[:,5].sum()-tp);fn=int(cm[5].sum()-tp)
    return dict(macro_f1=float(macro(cm)),stage6_FP=fp,stage6_FN=fn,
                stage6_precision=tp/max(tp+fp,1),stage6_recall=tp/max(tp+fn,1),
                stage6_f1=2*tp/max(2*tp+fp+fn,1))


def changes(frame):
    rows=[]
    for (before,after),q in frame[frame.J096!=frame.S3].groupby(['J096','S3']):
        rows.append(dict(before=int(before),after=int(after),changed=len(q),
            corrected=int((q.S3==q.label).sum()),harmed=int((q.J096==q.label).sum()),
            experiments=q.experiment_id.nunique()))
    return rows


def condition_match(frame,test):
    # Secondary descriptive sensitivity, based solely on observed static input
    # conditions. Neither test labels nor test predictions set these weights.
    fields=['bat_class','bat_cc','soc_status','tr_method']
    train_keys=list(map(tuple,frame[fields].to_numpy()))
    test_keys=list(map(tuple,test[fields].to_numpy()))
    source=pd.Series(train_keys).value_counts(normalize=True).to_dict()
    target=pd.Series(test_keys).value_counts(normalize=True).to_dict()
    missing=set(target)-set(source)
    if missing:return dict(available=False,unseen_conditions=[list(x) for x in missing])
    weight=np.array([target.get(k,0)/source[k] for k in train_keys])
    groups=list(frame.groupby('experiment_id').indices.values());y=frame.label.to_numpy(int)
    before=np.stack([matrix(y[i],frame.J096.to_numpy(int)[i])*weight[i][0] for i in groups])
    after=np.stack([matrix(y[i],frame.S3.to_numpy(int)[i])*weight[i][0] for i in groups])
    for i in groups:assert np.all(weight[i]==weight[i][0])
    draws=np.random.default_rng(20261115).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=6000)
    delta=macro(np.einsum('be,eij->bij',draws,after))-macro(np.einsum('be,eij->bij',draws,before))
    return dict(available=True,fields=fields,original_f1=float(macro(before.sum(0))),current_f1=float(macro(after.sum(0))),
        delta=float(macro(after.sum(0))-macro(before.sum(0))),delta95=np.quantile(delta,[.025,.975]).tolist(),
        limitation='Descriptive condition-row reweighting only; not a calibrated private-Test score or a new independent benchmark.')


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    artifact=load_artifact(combined_path=args.artifact)
    test=pd.read_csv(args.data/'test/test.csv')
    p=load_probs(args.base_predictions,'test',test.sample_id)
    stages=predict_stages(artifact,test,p)
    original=pd.read_csv(args.original).set_index('sample_id')
    current=pd.read_csv(args.current).set_index('sample_id')
    assert original.index.is_unique and current.index.is_unique
    assert set(test.sample_id)==set(original.index)==set(current.index)
    np.testing.assert_array_equal(stages['J096'],original.loc[test.sample_id].label.to_numpy(int))
    np.testing.assert_array_equal(stages['S3'],current.loc[test.sample_id].label.to_numpy(int))
    # Verify the copied original model code only changes import paths.
    root=Path(__file__).parent
    source=(args.original_root/'targeted_rule1/model.py').read_text().replace('from compare_abc_phase_history import causal_features','from .causal import causal_features')
    assert source==(root/'core/upper.py').read_text()
    source=(args.original_root/'experiments/lower_heads/model.py').read_text().replace('from targeted_rule1 import model as T','from . import upper as T')
    assert source==(root/'core/lower.py').read_text()
    assert artifact['low']['thresholds']['34']>1 and artifact['low']['thresholds']['43']==.5
    table=pd.read_csv(args.audit/'outer_predictions.csv')
    primary=table[(table.suite=='all97')&(table.view=='original')].copy()
    assert primary.sample_id.is_unique and primary.experiment_id.nunique()==97
    assert 'E_e5bcddc7d77f459ea541cf658ab4346e' in set(primary.experiment_id)
    result=summary(primary)
    val=pd.read_csv(args.audit/'validation_predictions.csv');val=val[val.view=='original']
    official_val=pd.read_csv(args.data/'validation/validation.csv')
    assert set(val.sample_id)==set(official_val.sample_id)
    vp=load_probs(args.base_predictions,'validation',official_val.sample_id)
    replay=predict_stages(artifact,official_val,vp)
    aligned=val.set_index('sample_id').loc[official_val.sample_id]
    for name in ['J096','S3']:np.testing.assert_array_equal(replay[name],aligned[name].to_numpy(int))
    comparisons=[]
    for (suite,view),q in table.groupby(['suite','view']):
        comparisons.append(dict(suite=suite,view=view,**summary(q)))
    write(args.output/'cadence_comparison.json',comparisons)
    folds=pd.read_csv(args.audit/'split_scores.csv');folds=folds[folds.view=='original'].pivot(index=['suite','scope'],columns='method',values='macro_f1')
    fold_summary=[]
    for suite,q in folds.groupby(level='suite'):
        fold_summary.append(dict(suite=suite,folds=len(q),positive_folds=int((q.S3>q.J096).sum()),
            original_min=float(q.J096.min()),current_min=float(q.S3.min()),
            original_sd=float(q.J096.std()),current_sd=float(q.S3.std())))
    frame=pd.read_csv(args.backbones/'frame.csv')
    f=frame.merge(primary[['sample_id','J096','S3']],on='sample_id',validate='one_to_one')
    diff=stages['J096']!=stages['S3'];counts=pd.DataFrame({'before':stages['J096'][diff],'after':stages['S3'][diff]}).value_counts()
    directions={}
    for name in ['down_only','up_only']:
        partial=primary.copy();pred=partial.J096.to_numpy(int).copy()
        mask=(partial.J096==6)&(partial.S3==5) if name=='down_only' else partial.J096.isin([4,5])&(partial.S3==6)
        pred[mask]=partial.S3.to_numpy(int)[mask];partial['partial']=pred
        directions[name]=dict(**summary(partial,'partial'),limitation='Diagnostic decomposition only; no new rule selected or deployed.')
    report=dict(recommendation='current_S3',leaderboard_score_used=False,parameter_selection=False,
        original_submission_reproduced_exactly=True,current_submission_reproduced_exactly=True,
        original_source_snapshots_match=True,primary=result,
        primary_original=metrics(primary.label,primary.J096),primary_current=metrics(primary.label,primary.S3),
        official_validation=summary(val),official_original=metrics(val.label,val.J096),official_current=metrics(val.label,val.S3),
        folds=fold_summary,condition_matched=condition_match(f,test),
        correction_directions=changes(primary),direction_decomposition=directions,
        test=dict(rows=len(test),experiments=test.experiment_id.nunique(),changed=int(diff.sum()),
            changed_experiments=test.experiment_id[diff].nunique(),
            transitions=[dict(before=int(a),after=int(b),rows=int(n)) for (a,b),n in counts.items()],
            unknown_changed_row_accuracy=True),
        critical_limitations=['Private Test labels are unavailable; this is an evidence-based recommendation, not a measured private-score comparison.',
            'Official Train/Validation have been explored before, and repeated suites overlap.',
            'OOF independently refits the fixed algorithm; exact archived weights are separately evaluated on complete official Validation.',
            'Both pipelines retain the E16 low-temperature stage5 and stage2/3 weaknesses.',
            'The6->5 branch can lose true6 rows; the complete S3 rule also restores missed6 rows.'],
        inputs_sha256={str(p):sha(p) for p in [args.original,args.current,args.artifact,args.audit/'outer_predictions.csv',args.audit/'validation_predictions.csv',Path(__file__)]})
    write(args.output/'comparison.json',report)
    print('Exact original/current replay PASS; full97 F1:',report['primary_original']['macro_f1'],'->',report['primary_current']['macro_f1'],flush=True)
    print('Condition-matched diagnostic:',json.dumps(report['condition_matched']),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['data','base-predictions','artifact','original','current','original-root','audit','backbones','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    with threadpool_limits(limits=2):main(ap.parse_args())
