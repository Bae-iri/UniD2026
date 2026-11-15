"""Validate one fixed disagreement veto; reject any measured regression.

Unseen-Test non-regression cannot be guaranteed. Current S3 is kept unless a
strict predeclared complete-data audit supports activating the veto.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from core.upper import f1
from fusion import features,fit_gate,apply_gate,predict_fusion,GATE_C,GATE_THRESHOLD,FEATURE_NAMES
from pipeline import fit_frozen_recipe,predict_stages
from reproduce import load_artifact,load_probs,sha
from validate import BackboneAudit,VIEWS,subsample,summary,matrix,macro


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def fit_scope(frame,folder,training,held,parts,audit):
    xx=[];tables=[];old=[];new=[]
    outer_exps=set(frame.experiment_id.iloc[held])
    for j,inner_held in enumerate(parts):
        inner_fit=np.setdiff1d(training,inner_held)
        forbidden=outer_exps|set(frame.experiment_id.iloc[inner_held])
        p=audit.load(folder/f'leave_{j}',inner_fit,forbidden)
        ff=frame.iloc[inner_fit].reset_index(drop=True);hf=frame.iloc[inner_held].reset_index(drop=True)
        model=fit_frozen_recipe(ff,p[:,inner_fit])
        assert not set(model['fit_experiments'])&forbidden
        stages=predict_stages(model,hf,p[:,inner_held])
        xx.append(features(hf,p[:,inner_held],stages['J096'],stages['S3']))
        tables.append(hf);old.append(stages['J096']);new.append(stages['S3'])
    table=pd.concat(tables,ignore_index=True)
    assert table.sample_id.is_unique and set(table.sample_id)==set(frame.sample_id.iloc[training])
    gate=fit_gate(table,np.concatenate(xx),np.concatenate(old),np.concatenate(new))
    assert set(gate['fit_experiments'])==set(frame.experiment_id.iloc[training])
    assert not set(gate['fit_experiments'])&outer_exps
    return gate


def check_prefix(frame,p,old,new,gate):
    full=features(frame,p,old,new);expected=apply_gate(gate,full,old,new);count=0
    for ids in frame.groupby('experiment_id').indices.values():
        ids=ids[np.argsort(frame.relative_time_s.to_numpy()[ids],kind='stable')]
        for end in sorted({1,min(5,len(ids)),max(1,len(ids)//2),len(ids)}):
            sub=ids[:end];f=frame.iloc[sub].reset_index(drop=True)
            x=features(f,p[:,sub],old[sub],new[sub])
            np.testing.assert_allclose(x,full[sub],rtol=0,atol=0,equal_nan=True)
            np.testing.assert_array_equal(apply_gate(gate,x,old[sub],new[sub]),expected[sub])
            count+=1
    return count


def measure(table):
    r=summary(table,'hybrid',reference='current')
    before=matrix(table.label,table.current);after=matrix(table.label,table.hybrid)
    r['current_per_class_f1']=f1(before).tolist()
    r['current_stage6_FP']=int(before[:,5].sum()-before[5,5])
    r['current_stage6_FN']=int(before[5].sum()-before[5,5])
    r['metrics_no_regression']=bool(
        r['delta']>=-1e-12 and np.all(f1(after)>=f1(before)-1e-12)
        and r['stage6']['false_positive']<=r['current_stage6_FP']
        and r['stage6']['false_negative']<=r['current_stage6_FN'])
    return r


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    original_bytes=args.submission.read_bytes()
    write(args.output/'protocol.json',dict(
        candidate='One fixed L2 logistic veto on original/S3 disagreements only',
        C=GATE_C,threshold=GATE_THRESHOLD,features=FEATURE_NAMES,
        feature_range_guard='Fit only on informative training disagreements; nonfinite/out-of-range rows retain S3',
        primary='All97 experiments, E16 included, outer5fold; gate learns from inner experiment-held predictions',
        acceptance=['Positive paired-bootstrap95 lower for primary F1 gain and at least one change',
                    'No F1 regression in any of5folds times8cadence views',
                    'No aggregate per-class F1 or stage6 FP/FN regression in any of8views',
                    'No F1, per-class F1, stage6 FP/FN regression on all19official Validation experiments, in all8views',
                    'Positive full official Validation F1 gain; source and prefix checks pass'],
        fallback='Export current S3 with veto disabled when any criterion fails',
        leaderboard_input=False,parameter_search=False,experiment_exclusions=[],
        no_unseen_score_guarantee=True,
        limitations=['Previously inspected datasets and hypotheses; not a fresh blind benchmark.',
            'Inner stack fit probabilities come from its one teacher in-sample; that teacher excludes inner/outer evaluation groups. Outer baseline uses proper inner OOF training probabilities.',
            'A gate score0.95 is a fixed model threshold, not a95% calibrated guarantee of correctness.'],
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('fusion.py')]}))
    frame=pd.read_csv(args.backbones/'frame.csv')
    assert frame.sample_id.is_unique and frame.experiment_id.nunique()==97
    assert 'E_e5bcddc7d77f459ea541cf658ab4346e' in set(frame.experiment_id)
    cached=pd.read_csv(args.existing/'outer_predictions.csv');cached=cached[cached.suite=='all97']
    index=pd.Index(frame.sample_id);audit=BackboneAudit(frame);rows=[];folds=[];gate_details=[];prefixes=0
    for k in range(5):
        folder=args.backbones/f'outer_{k}'
        with np.load(folder/'partitions.npz') as z:
            training,held=z['training'],z['held'];parts=[z[f'inner_{j}'] for j in range(3)]
        gate=fit_scope(frame,folder,training,held,parts,audit)
        joblib.dump(gate,args.output/f'outer_{k}_gate.joblib')
        p=audit.load(folder/'full',training,set(frame.experiment_id.iloc[held]))
        baseline_model=joblib.load(args.existing/f'all97_outer_{k}.joblib')
        assert set(baseline_model['fit_experiments'])==set(frame.experiment_id.iloc[training])
        expected_exps=set(frame.experiment_id.iloc[held]);fold=cached[cached.scope==f'outer_{k}']
        for view in VIEWS:
            q=fold[fold.view==view].reset_index(drop=True)
            assert set(q.experiment_id)==expected_exps
            ix=index.get_indexer(q.sample_id);assert (ix>=0).all()
            ff=frame.iloc[ix].reset_index(drop=True)
            old=q.J096.to_numpy(int);current=q.S3.to_numpy(int)
            if view=='original':
                stages=predict_stages(baseline_model,ff,p[:,ix])
                np.testing.assert_array_equal(stages['J096'],old);np.testing.assert_array_equal(stages['S3'],current)
                prefixes+=check_prefix(ff,p[:,ix],old,current,gate)
            x=features(ff,p[:,ix],old,current);pred=apply_gate(gate,x,old,current)
            tab=q[['sample_id','experiment_id','label','view']].copy()
            tab['fold'],tab['original'],tab['current'],tab['hybrid']=k,old,current,pred
            rows.append(tab)
            delta=float(macro(matrix(q.label,pred))-macro(matrix(q.label,current)))
            folds.append(dict(fold=k,view=view,delta=delta,changed=int((pred!=current).sum())))
        gate_details.append(dict(fold=k,informative=gate['informative_rows'],original_correct=gate['original_correct_rows'],
                                 current_correct=gate['current_correct_rows'],fitted=gate['model'] is not None))
        print('outer',k,'original-view',folds[-len(VIEWS)],flush=True)
    table=pd.concat(rows,ignore_index=True)
    table.to_csv(args.output/'outer_predictions.csv',index=False)
    metrics={view:measure(q) for view,q in table.groupby('view')}
    write(args.output/'outer_summary.json',metrics);write(args.output/'fold_scores.json',folds)
    write(args.output/'gate_training.json',gate_details)
    primary=metrics['original']
    outer_ok=bool(primary['delta']>0 and primary['bootstrap_delta_95'][0]>0
                  and all(x['delta']>=-1e-12 for x in folds)
                  and all(x['metrics_no_regression'] for x in metrics.values()))
    train=pd.read_csv(args.data/'train/train.csv');train_audit=BackboneAudit(train)
    folder=args.train_backbones/'global'
    with np.load(folder/'partitions.npz') as z:
        training,held=z['training'],z['held'];parts=[z[f'inner_{j}'] for j in range(3)]
    gate=fit_scope(train,folder,training,held,parts,train_audit)
    bundle=load_artifact(combined_path=args.artifact)
    assert not set(gate['fit_experiments'])&set(pd.read_csv(args.data/'validation/validation.csv').experiment_id)
    val=pd.read_csv(args.data/'validation/validation.csv');vp=load_probs(args.base,'validation',val.sample_id)
    validations=[];full_stages=None
    for view in VIEWS:
        sub=subsample(val,view);ff=val.iloc[sub].reset_index(drop=True)
        stages=predict_stages(bundle,ff,vp[:,sub]);old,current=stages['J096'],stages['S3']
        x=features(ff,vp[:,sub],old,current);pred=apply_gate(gate,x,old,current)
        tab=ff[['sample_id','experiment_id','label']].copy();tab['view']=view
        tab['original'],tab['current'],tab['hybrid']=old,current,pred;validations.append(tab)
        if view=='original':
            prefixes+=check_prefix(ff,vp[:,sub],old,current,gate);full_stages=stages
    vt=pd.concat(validations,ignore_index=True);vt.to_csv(args.output/'validation_predictions.csv',index=False)
    vm={view:measure(q) for view,q in vt.groupby('view')};write(args.output/'validation_summary.json',vm)
    validation_ok=bool(vm['original']['delta']>0 and all(v['metrics_no_regression'] for v in vm.values()))
    accepted=bool(outer_ok and validation_ok)
    bundle=dict(bundle,disagreement_gate=gate,fusion_enabled=accepted,
                backbone_sha256={p:sha(args.base/p) for p in ['preprocessing.json']+[f'seed_{s}/B/best.pt' for s in [42,43,44]]})
    joblib.dump(bundle,args.output/'fusion.joblib')
    test=pd.read_csv(args.data/'test/test.csv');tp=load_probs(args.base,'test',test.sample_id)
    stages=predict_stages(bundle,test,tp);prefixes+=check_prefix(test,tp,stages['J096'],stages['S3'],gate)
    deployed=predict_fusion(bundle,test,tp)
    protected=pd.read_csv(args.submission).set_index('sample_id').loc[test.sample_id].label.to_numpy(int)
    np.testing.assert_array_equal(stages['S3'],protected)
    if not accepted:np.testing.assert_array_equal(deployed,protected)
    saved=joblib.load(args.output/'fusion.joblib');np.testing.assert_array_equal(predict_fusion(saved,test,tp),deployed)
    sample=pd.read_csv(args.data/'sample_submission.csv')
    assert sample.sample_id.is_unique and set(sample.sample_id)==set(test.sample_id)
    submission=pd.DataFrame({'sample_id':test.sample_id,'label':deployed}).set_index('sample_id').loc[sample.sample_id].reset_index()
    submission.to_csv(args.output/'submission_fusion.csv',index=False)
    assert args.submission.read_bytes()==original_bytes
    decision=dict(veto_accepted=accepted,deployment='disagreement_veto' if accepted else 'current_S3',
        outer_checks_pass=outer_ok,validation_checks_pass=validation_ok,
        primary=primary,official_validation=vm['original'],
        features_and_gate_prefix_checks=prefixes,backbone_records_audited=len(audit.records)+len(train_audit.records),
        consensus_rows_immutable=True,original_4to3_preserved=True,
        original_submission_unchanged=True,test_prediction_changes=int((deployed!=protected).sum()),
        exported_csv_sha256=sha(args.output/'submission_fusion.csv'),protected_csv_sha256=sha(args.submission),
        gate_threshold=GATE_THRESHOLD,leaderboard_input=False,unseen_non_regression_guaranteed=False,
        caveat='Validation rejection disables the veto and preserves current S3. Identity of Test predictions preserves its score; unknown new predictions have no absolute guarantee.')
    write(args.output/'decision.json',decision)
    print('FINAL',json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['data','backbones','train-backbones','existing','base','artifact','submission','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    with threadpool_limits(limits=2):main(ap.parse_args())
