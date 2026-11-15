"""Nested screening of conservative causal 2->3 / 4->5 residual heads.

The submitted pipeline is protected. No experiment ID or leaderboard score is
an inference input. Dataset/hypotheses have already been explored: not blind.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from core import upper
from pipeline import fit_frozen_recipe, predict_stages, rule1
from reproduce import load_artifact, load_probs, sha
from validate import BackboneAudit, matrix, macro, summary

POLICIES = ['none','23','45','23+45']
PAIRS = {'23':(2,3),'45':(4,5)}
THRESHOLD = .9
PARAMS = dict(max_iter=80,max_leaf_nodes=3,max_depth=2,learning_rate=.05,
              min_samples_leaf=128,l2_regularization=10.,early_stopping=False,random_state=42)
CLOCK = {'elapsed_log1p','observed_elapsed_log1p','observed_count_log1p','observation_gap_log1p'}


def write(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def features(frame):
    # '45' physical features are independent of seed probabilities. Uniform
    # dummy probabilities make that independence explicit for inner selection.
    x,names = upper.features(frame,np.full((3,len(frame),6),1/6))
    keep = np.array([n not in CLOCK for n in names['45']])
    t = frame.relative_time_s.to_numpy(float)
    pressure = pd.to_numeric(frame.meas_p,errors='coerce').to_numpy(float)
    extra = np.full((len(frame),4),np.nan)
    for ids in frame.groupby('experiment_id',sort=False).indices.values():
        ids=ids[np.argsort(t[ids],kind='stable')]
        peak=when=np.nan
        for row in ids:
            value=pressure[row]
            if np.isfinite(value) and (not np.isfinite(peak) or value>peak):
                peak,when=value,t[row]
            if np.isfinite(peak):
                extra[row,0]=np.sign(peak)*np.log1p(abs(peak))
                extra[row,3]=np.log1p(t[row]-when)
                if np.isfinite(value):
                    extra[row,1]=(peak-value)/max(abs(peak),1.)
                    extra[row,2]=np.sign(value)*np.log1p(abs(value))-extra[row,0]
    return np.c_[x['45'][:,keep],extra]


def fit_heads(x,y,groups):
    heads={}
    # No upstream predictions or fitted feature transformations enter the new
    # head fitting. Both target and non-target experiments receive equal weight.
    for pair,(_,target) in PAIRS.items():
        yy=(y==target).astype(int)
        keys=pd.DataFrame({'group':groups,'target':yy})
        weight=1/keys.groupby(['group','target']).target.transform('size').to_numpy(float)
        model=HistGradientBoostingClassifier(**PARAMS)
        model.fit(x,yy,sample_weight=weight/weight.mean())
        heads[pair]=model
    return heads


def apply(base,heads,x,policy):
    p=np.asarray(base,int).copy()
    for pair in PAIRS:
        if pair not in policy.split('+'):
            continue
        source,target=PAIRS[pair]
        score=heads[pair].predict_proba(x)[:,1]
        p[(np.asarray(base)==source)&(score>=THRESHOLD)]=target
    assert np.array_equal(p[np.isin(base,[3,6])],np.asarray(base)[np.isin(base,[3,6])])
    return p


def choose(table):
    groups=list(table.groupby('experiment_id').indices.values())
    y=table.label.to_numpy(int)
    cms={name:np.stack([matrix(y[i],table[name].to_numpy(int)[i]) for i in groups]) for name in POLICIES}
    weights=np.random.default_rng(20261115).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=3000)
    boot_base=macro(np.einsum('be,eij->bij',weights,cms['none']))
    base_score=float(macro(cms['none'].sum(0)))
    choices=[]
    for policy in POLICIES:
        score=float(macro(cms[policy].sum(0)))
        boot=macro(np.einsum('be,eij->bij',weights,cms[policy]))-boot_base
        per=macro(cms[policy])-macro(cms['none'])
        pred=table[policy].to_numpy(int);base=table['none'].to_numpy(int)
        fixed=int(((base!=y)&(pred==y)).sum());harmed=int(((base==y)&(pred!=y)).sum())
        passed=policy=='none' or (np.quantile(boot,.025)>0 and fixed>=max(3,2*harmed) and (per>1e-12).sum()>(per < -1e-12).sum())
        choices.append(dict(policy=policy,macro_f1=score,delta=score-base_score,
            delta95=np.quantile(boot,[.025,.975]).tolist(),corrected=fixed,harmed=harmed,passed=bool(passed)))
    allowed=[c for c in choices if c['passed']]
    best=max(c['macro_f1'] for c in allowed)
    # Prefer fewer changed boundaries if its loss is <=0.002 inside training.
    eligible=[c for c in allowed if c['macro_f1']>=best-.002]
    chosen=min(eligible,key=lambda c:(0 if c['policy']=='none' else len(c['policy'].split('+')),-c['macro_f1']))['policy']
    return chosen,choices


def select_scope(frame,x,folder,audit,training,held,parts):
    rows=[];outer_exps=set(frame.experiment_id.iloc[held])
    for j,inner_held in enumerate(parts):
        inner_fit=np.setdiff1d(training,inner_held)
        forbidden=outer_exps | set(frame.experiment_id.iloc[inner_held])
        pp=audit.load(folder/f'leave_{j}',inner_fit,forbidden)
        ff=frame.iloc[inner_fit].reset_index(drop=True);hf=frame.iloc[inner_held].reset_index(drop=True)
        # A single inner-training teacher supplies both fit/evaluation features.
        # Its training AND checkpoint selection exclude every inner held group.
        # The stack's fit probabilities here are in-sample for that teacher;
        # this calibration difference is disclosed and outer evaluation uses OOF.
        baseline_model=fit_frozen_recipe(ff,pp[:,inner_fit])
        assert not set(baseline_model['fit_experiments']) & forbidden
        base=predict_stages(baseline_model,hf,pp[:,inner_held])['S3']
        heads=fit_heads(x[inner_fit],ff.label.to_numpy(int),ff.experiment_id.to_numpy())
        tab=hf[['sample_id','experiment_id','label']].copy()
        for policy in POLICIES:tab[policy]=apply(base,heads,x[inner_held],policy)
        rows.append(tab)
    table=pd.concat(rows,ignore_index=True)
    assert set(table.sample_id)==set(frame.sample_id.iloc[training])
    policy,choices=choose(table)
    return policy,choices


def feature_prefix_checks(frame,full):
    count=0
    for ids in frame.groupby('experiment_id').indices.values():
        ids=ids[np.argsort(frame.relative_time_s.to_numpy()[ids],kind='stable')]
        for end in sorted({1,min(5,len(ids)),len(ids)//2,len(ids)}):
            if end<1:continue
            sub=ids[:end]
            np.testing.assert_allclose(features(frame.iloc[sub].reset_index(drop=True)),full[sub],rtol=0,atol=0,equal_nan=True)
            count+=1
    return count


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    protected=args.submission.read_bytes()
    write(args.output/'protocol.json',dict(policies=POLICIES,threshold=THRESHOLD,params=PARAMS,
        model_selection='Nested experiment groups; inner paired bootstrap positive and corrected>=2*harmed; prefer fewer heads within0.002F1',
        adoption='outer paired95 lower>0, all5fold deltas>=0, full19validation delta>0; stage6 predictions and existing4->3 protected',
        inference_ids_used=False,leaderboard_input=False,E16_primary_included=True,parameter_search=False,
        limitation='Previously inspected datasets/hypothesis development; not a fresh blind benchmark. Inner stack fit probabilities use the one inner-training teacher in-sample; its entire fit/selection exclude inner held groups.',
        source_sha256=sha(Path(__file__))))
    frame=pd.read_csv(args.backbones/'frame.csv');x=features(frame)
    prefix=feature_prefix_checks(frame,x)
    audit=BackboneAudit(frame);rows=[];fold_scores=[];decisions=[]
    for k in range(5):
        folder=args.backbones/f'outer_{k}'
        with np.load(folder/'partitions.npz') as z:
            training,held=z['training'],z['held'];parts=[z[f'inner_{j}'] for j in range(3)]
        policy,choices=select_scope(frame,x,folder,audit,training,held,parts)
        baseline_model=joblib.load(args.existing/f'all97_outer_{k}.joblib')
        assert set(baseline_model['fit_experiments'])==set(frame.experiment_id.iloc[training])
        assert not set(baseline_model['fit_experiments']) & set(frame.experiment_id.iloc[held])
        pp=audit.load(folder/'full',training,set(frame.experiment_id.iloc[held]))
        hf=frame.iloc[held].reset_index(drop=True)
        base=predict_stages(baseline_model,hf,pp[:,held])['S3']
        heads=fit_heads(x[training],frame.label.to_numpy(int)[training],frame.experiment_id.to_numpy()[training])
        pred=apply(base,heads,x[held],policy)
        tab=hf[['sample_id','experiment_id','label']].copy();tab['baseline'],tab['candidate']=base,pred
        tab['fold'],tab['selected_policy']=k,policy;rows.append(tab)
        delta=float(macro(matrix(hf.label,pred))-macro(matrix(hf.label,base)))
        fold_scores.append(delta);decisions.append(dict(fold=k,selected=policy,choices=choices,outer_delta=delta))
        print('outer',k,'inner-selected',policy,'outer_delta',round(delta,6),flush=True)
    table=pd.concat(rows,ignore_index=True);table.to_csv(args.output/'outer_predictions.csv',index=False)
    assert table.sample_id.is_unique and table.experiment_id.nunique()==97
    assert 'E_e5bcddc7d77f459ea541cf658ab4346e' in set(table.experiment_id)
    outer=summary(table,'candidate',reference='baseline')
    write(args.output/'outer_summary.json',outer);write(args.output/'fold_decisions.json',decisions)
    # Final policy selection is confined to official Train78, before opening Val.
    train=pd.read_csv(args.data/'train/train.csv');tx=features(train)
    global_folder=args.train_backbones/'global';train_audit=BackboneAudit(train)
    with np.load(global_folder/'partitions.npz') as z:
        training=z['training'];held=z['held'];parts=[z[f'inner_{j}'] for j in range(3)]
    policy,choices=select_scope(train,tx,global_folder,train_audit,training,held,parts)
    heads=fit_heads(tx,train.label.to_numpy(int),train.experiment_id.to_numpy())
    joblib.dump(dict(heads=heads,policy=policy,threshold=THRESHOLD,fit_experiments=sorted(set(train.experiment_id))),args.output/'candidate.joblib')
    artifact=load_artifact(combined_path=args.artifact)
    val=pd.read_csv(args.data/'validation/validation.csv');vx=features(val);prefix+=feature_prefix_checks(val,vx)
    vp=load_probs(args.base_predictions,'validation',val.sample_id)
    base=predict_stages(artifact,val,vp)['S3'];pred=apply(base,heads,vx,policy)
    tab=val[['sample_id','experiment_id','label']].copy();tab['baseline'],tab['candidate']=base,pred
    tab.to_csv(args.output/'validation_predictions.csv',index=False)
    result=summary(tab,'candidate',reference='baseline')
    accepted=bool(policy!='none' and outer['bootstrap_delta_95'][0]>0 and min(fold_scores)>=-1e-12 and result['delta']>0)
    decision=dict(accepted=accepted,selected_policy=policy,train_only_choices=choices,outer=outer,validation=result,
        primary_E16_included=True,new_feature_prefix_checks=prefix,
        stage6_predictions_protected=True,existing_4to3_predictions_protected=True,
        backbone_records_audited=len(audit.records)+len(train_audit.records),
        original_submission_unchanged=True,leaderboard_input=False,
        explanation='Only export candidate if prespecified outer/full-validation checks pass; otherwise retain current submission.')
    if accepted:
        test=pd.read_csv(args.data/'test/test.csv');xx=features(test);prefix+=feature_prefix_checks(test,xx)
        pp=load_probs(args.base_predictions,'test',test.sample_id);base=predict_stages(artifact,test,pp)['S3']
        pred=apply(base,heads,xx,policy)
        submission=pd.DataFrame({'sample_id':test.sample_id,'label':pred}).set_index('sample_id')
        sample=pd.read_csv(args.data/'sample_submission.csv');submission=submission.loc[sample.sample_id].reset_index()
        submission.to_csv(args.output/'submission_boundary_candidate.csv',index=False)
        decision['test_changed']=int((pred!=base).sum());decision['new_feature_prefix_checks']=prefix
    assert args.submission.read_bytes()==protected
    write(args.output/'decision.json',decision)
    print('DECISION',json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['data','backbones','train-backbones','existing','base-predictions','artifact','submission','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    with threadpool_limits(limits=2):main(ap.parse_args())
