"""Train/evaluate two fixed specialists on already verified nested B OOF data."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from targeted_stage_model.model import PARAMS, PAIRS, THRESHOLDS, train as train_heads, infer
from temporal45_model.nested_train import DATA, BASE, OUTER, SEEDS, verify_outer
from temporal45_model.causal_rules import predict_rules
from four_rule_cv.run import fit_rules, record_protocol
import final_rules3

NESTED = ROOT/'temporal45_model/past_only_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def target_counts(frame, outputs, protocol):
    rows = []
    y = frame.label.to_numpy(int)
    base = outputs['rules3']
    for method,p in outputs.items():
        for scope,mask in [('all', np.ones(len(frame), bool)),
                           ('noE16', frame.experiment_id.to_numpy() != final_rules3.E16)]:
            a,b,q = y[mask], base[mask], p[mask]
            rows.append(dict(protocol=protocol, scope=scope, method=method,
                true5_pred4=int(((a == 5) & (q == 4)).sum()),
                true6_pred5=int(((a == 6) & (q == 5)).sum()),
                true4_pred5=int(((a == 4) & (q == 5)).sum()),
                true5_pred6=int(((a == 5) & (q == 6)).sum()),
                corrected=int(((a != b) & (a == q)).sum()),
                harmed=int(((a == b) & (a != q)).sum())))
    return rows


def outputs_for(artifact, frame, probs, base):
    selected = infer(artifact, frame, probs, base)
    raw = infer(artifact, frame, probs, base, fixed_threshold=.5)
    return {'rules3': base, **{f'selected_{k}': v for k,v in selected.items()},
            **{f'fixed05_{k}': v for k,v in raw.items()}}


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    previous = json.loads((NESTED/'protocol.json').read_text())
    assert json.loads((NESTED/'audit_checks.json').read_text())['passed']
    for path in [ROOT/'final_rules3.py', ROOT/'compare_abc_phase_history.py', DATA/'train/train.csv']:
        assert sha(path) == previous['source_sha256'][str(path)]
    protocol = dict(targets=['true5/pred4 -> 5', 'true6/pred5 -> 6'],
        heads=PAIRS, params=PARAMS, thresholds=list(THRESHOLDS),
        primary='selected heads, separately assessed; combine only accepted heads',
        diagnostic='fixed 0.5 default reports classifier ability even if selector abstains; not eligible for post-hoc deployment',
        features45='current physical signals, past extremes/differences/ages, trailing60 range/slope, observed elapsed and gap; no B probabilities',
        features56='same physical history plus current three-seed probabilities and trailing30/60 probability history',
        rule='current/past only; only existing4->5 and existing5->6; no same-row cascade',
        threshold_selection='outer training group-crossfit only; corrected>=2*harmed; >=3 improved experiments; more improved than worsened; positive leave-one-out pooled delta',
        acceptance=['positive nested outer OOF delta and bootstrap lower',
                    'more improved than worsened experiments', 'positive leave-one-experiment-out delta',
                    'validation all/noE16 nonnegative delta for both original and fold ensemble B'],
        limitations='Previously inspected data and previously compared outer folds: exploratory replication, not new blind evidence. No capacity/threshold tuning after holdout results.',
        test_access=False, validation_read_after_selection=True,
        source_sha256={str(p): sha(p) for p in [Path(__file__), ROOT/'targeted_stage_model/model.py',
                         ROOT/'final_rules3.py', ROOT/'temporal45_model/causal_rules.py',
                         ROOT/'compare_abc_phase_history.py', DATA/'train/train.csv']})
    write(out/'protocol.json', protocol)
    frame = pd.read_csv(DATA/'train/train.csv')
    original = pd.read_csv(OUTER/'train_oof/predictions.csv')
    verify_outer(frame, original)
    y = frame.label.to_numpy(int)
    all_probs = np.zeros((3, len(frame), 6))
    outputs = {name: np.zeros(len(frame), int) for name in ['rules3', 'selected_45', 'selected_56',
               'selected_both', 'fixed05_45', 'fixed05_56', 'fixed05_both']}
    decisions = []
    for fold in range(5):
        held = np.flatnonzero(original.fold.to_numpy() == fold)
        fitting = np.flatnonzero(original.fold.to_numpy() != fold)
        inner = np.load(NESTED/f'outer_{fold}/inner_oof.npz')
        np.testing.assert_array_equal(inner['sample_id'], frame.sample_id.iloc[fitting].to_numpy(str))
        np.testing.assert_array_equal(inner['label'], y[fitting])
        held_groups = set(frame.experiment_id.iloc[held])
        fitting_groups = set(frame.experiment_id.iloc[fitting])
        assert not held_groups & fitting_groups
        artifact = train_heads(frame.iloc[fitting].reset_index(drop=True), inner['seed_probs'],
                               y[fitting], inner['baseline'])
        assert set(artifact['fit_experiments']) == fitting_groups
        dest = out/f'outer_{fold}'
        dest.mkdir()
        joblib.dump(artifact, dest/'heads.joblib')
        p = np.stack([np.load(OUTER/f'fold_{fold}/seed{s}_probs.npz')['holdout'] for s in SEEDS])
        all_probs[:, held] = p
        hframe = frame.iloc[held].reset_index(drop=True)
        threshold = fit_rules(frame.iloc[fitting])
        base = predict_rules(hframe, p.mean(0).argmax(1)+1, threshold)[0]['rules3']
        np.testing.assert_array_equal(base, original.rules3.to_numpy()[held])
        predicted = outputs_for(artifact, hframe, p, base)
        for name, values in predicted.items():
            outputs[name][held] = values
        decisions.append(dict(fold=fold, thresholds=artifact['thresholds'], choices=artifact['choices']))
        write(out/'fold_decisions.json', decisions)
        print('FOLD COMPLETE', fold, artifact['thresholds'], flush=True)
    f = frame.assign(fold=original.fold.to_numpy())
    summary = record_protocol(out, 'nested_oof', f, outputs)
    counts = target_counts(frame, outputs, 'nested_oof')
    # Thresholds and all model fitting are fixed before opening Validation.
    artifact = train_heads(frame, all_probs, y, outputs['rules3'])
    artifact['production_rule_thresholds'] = json.loads((OUTER/'saved_fit_thresholds.json').read_text())
    joblib.dump(artifact, out/'candidate_heads.joblib')
    write(out/'final_training_choices.json', dict(thresholds=artifact['thresholds'], choices=artifact['choices']))
    val = pd.read_csv(DATA/'validation/validation.csv')
    assert not set(val.experiment_id) & set(frame.experiment_id)
    noE16 = np.flatnonzero(val.experiment_id.to_numpy() != final_rules3.E16)
    validations = {}
    for recipe in ['fold_ensemble', 'original_B']:
        if recipe == 'fold_ensemble':
            probs = np.stack([np.mean([np.load(OUTER/f'fold_{k}/seed{s}_probs.npz')['validation']
                                       for k in range(5)], axis=0) for s in SEEDS])
            thresholds = fit_rules(frame)
        else:
            probs = np.stack([pd.read_csv(BASE/f'seed_{s}/B/validation_predictions.csv').set_index('sample_id').loc[val.sample_id, [f'p{i}' for i in range(1,7)]].to_numpy() for s in SEEDS])
            thresholds = artifact['production_rule_thresholds']
        base = predict_rules(val, probs.mean(0).argmax(1)+1, thresholds)[0]['rules3']
        predicted = outputs_for(artifact, val, probs, base)
        protocol_name = f'validation_{recipe}'
        validations[protocol_name] = record_protocol(out, protocol_name, val, predicted)
        counts += target_counts(val, predicted, protocol_name)
        validations[protocol_name+'_noE16'] = record_protocol(out, protocol_name+'_noE16',
            val.iloc[noE16].reset_index(drop=True), {k: v[noE16] for k,v in predicted.items()})
    pd.DataFrame(counts).to_csv(out/'target_errors.csv', index=False)
    primary = {s['method']: s for s in summary}
    accepted = {}
    checks_by_head = {}
    for pair in PAIRS:
        name = 'selected_'+pair
        s = primary[name]
        checks = dict(positive_oof=s['delta_f1'] > 0, positive_ci_lower=s['ci_lower'] > 0,
                      more_improved=s['better_experiments'] > s['worse_experiments'],
                      positive_loo=s['leave_one_experiment_out_min_delta'] > 0,
                      validation_nonnegative=all(next(a for a in vs if a['method'] == name)['delta_f1'] >= 0
                                                 for vs in validations.values()))
        accepted[pair] = bool(all(checks.values()))
        checks_by_head[pair] = {k: bool(v) for k,v in checks.items()}
    deployed = dict(artifact, thresholds=dict(artifact['thresholds']), accepted=accepted)
    for pair in PAIRS:
        if not accepted[pair]:
            deployed['thresholds'][pair] = 1.01
    # If both independently pass, their joint behavior must also pass the same
    # outer and validation criteria before enabling a combined deployment.
    if all(accepted.values()):
        s = primary['selected_both']
        joint_pass = (s['delta_f1'] > 0 and s['ci_lower'] > 0
                      and s['better_experiments'] > s['worse_experiments']
                      and s['leave_one_experiment_out_min_delta'] > 0
                      and all(next(a for a in vs if a['method'] == 'selected_both')['delta_f1'] >= 0
                              for vs in validations.values()))
        if not joint_pass:
            deployed['thresholds'] = {pair: 1.01 for pair in PAIRS}
            accepted = {pair: False for pair in PAIRS}
    deployed['accepted'] = accepted
    joblib.dump(deployed, out/'deployment_heads.joblib')
    write(out/'decision.json', dict(accepted=accepted, checks=checks_by_head,
          no_posthoc_switch_to_fixed05=True, submission_created=False))
    write(out/'status.json', dict(complete=True, base_retrained=False, nested_models=5*2,
          threshold_crossfit_models=5*3*2, final_models=2))
    print(pd.DataFrame(summary)[['method','macro_f1','delta_f1','corrected','harmed','ci_lower','ci_upper']].to_string(index=False), flush=True)
    print('ACCEPTED', accepted, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.output)
