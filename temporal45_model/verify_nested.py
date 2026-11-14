"""Reconstruct separation, checkpoint selection and predictions independently."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from four_rule_cv.run import abc, fit_rules
from temporal45_model.conservative import MODES, infer
from temporal45_model.nested_train import DATA, OUTER, SEEDS
from temporal45_model.causal_rules import predict_rules


def main(run):
    protocol = json.loads((run/'protocol.json').read_text())
    status = json.loads((run/'status.json').read_text())
    assert status.get('complete') and status['fresh_inner_models'] == 45
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    train = pd.read_csv(DATA/'train/train.csv')
    nested = pd.read_csv(run/'nested_oof/predictions.csv')
    np.testing.assert_array_equal(train.sample_id, nested.sample_id)
    np.testing.assert_array_equal(train.label, nested.label)
    assert train.groupby('experiment_id').apply(lambda g: nested.fold.iloc[g.index].nunique(), include_groups=False).eq(1).all()
    models = 0
    prefix_checks = 0
    for fold in range(5):
        held_ids = np.flatnonzero(nested.fold.to_numpy() == fold)
        outer_ids = np.flatnonzero(nested.fold.to_numpy() != fold)
        held_groups = set(train.experiment_id.iloc[held_ids])
        inner_archive = np.load(run/f'outer_{fold}/inner_oof.npz')
        np.testing.assert_array_equal(inner_archive['sample_id'], train.sample_id.iloc[outer_ids].to_numpy(str))
        cached = pd.Index(inner_archive['sample_id'])
        seen = set()
        for inner in range(3):
            folder = run/f'outer_{fold}/inner_{inner}'
            split = pd.read_csv(folder/'split_manifest.csv')
            assert split.sample_id.is_unique
            assert set(split.sample_id) == set(train.sample_id.iloc[outer_ids])
            roles = ['fit', 'selection', 'inner_holdout']
            sets = [set(split.loc[split.role == r, 'experiment_id']) for r in roles]
            assert not any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3))
            assert not held_groups & set(split.experiment_id)
            assert not seen & sets[2]
            seen |= sets[2]
            fit = train[train.experiment_id.isin(sets[0])]
            pool = train[train.experiment_id.isin(sets[0] | sets[1])]
            expected = abc.SensorPreprocessor(fit).metadata()
            actual = json.loads((folder/'preprocessing.json').read_text())
            for k in ['median', 'mean', 'std', 'class_weights']:
                np.testing.assert_allclose(expected[k], actual[k], rtol=1e-12, atol=1e-12)
            assert json.loads(json.dumps(fit_rules(pool))) == json.loads((folder/'thresholds.json').read_text())
            for si, seed in enumerate(SEEDS):
                hist = pd.read_csv(folder/f'seed{seed}_history.csv')
                assert len(hist) == 15
                ck = torch.load(folder/f'seed{seed}.pt', map_location='cpu', weights_only=True)
                best = hist.loc[hist.selection_f1.idxmax()]
                assert ck['epoch'] == int(best.epoch)
                assert abs(ck['inner_f1'] - float(best.selection_f1)) < 1e-12
                assert all(torch.isfinite(a).all() for a in ck['state'].values())
                hp = np.load(folder/f'seed{seed}_holdout.npz')
                np.testing.assert_array_equal(hp['sample_id'], split.loc[split.role == 'inner_holdout', 'sample_id'].to_numpy(str))
                p = hp['probability']
                assert np.isfinite(p).all() and (p >= 0).all()
                np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)
                index = cached.get_indexer(hp['sample_id'])
                assert (index >= 0).all()
                np.testing.assert_allclose(p, inner_archive['seed_probs'][si, index], atol=0)
                models += 1
        assert seen == set(train.experiment_id.iloc[outer_ids])
        p = np.stack([np.load(OUTER/f'fold_{fold}/seed{s}_probs.npz')['holdout'] for s in SEEDS])
        held_frame = train.iloc[held_ids].reset_index(drop=True)
        thresholds = fit_rules(train.iloc[outer_ids])
        causal_base = predict_rules(held_frame, p.mean(0).argmax(1)+1, thresholds)[0]['rules3']
        np.testing.assert_array_equal(causal_base, nested.rules3.to_numpy()[held_ids])
        for mode in MODES:
            artifact = joblib.load(run/f'outer_{fold}/{mode}.joblib')
            assert set(artifact['fit_experiments']) == set(train.experiment_id.iloc[outer_ids])
            assert not set(artifact['fit_experiments']) & held_groups
            choices = artifact['threshold_selection']
            acceptable = [c for c in choices if c['passed']]
            threshold = max(acceptable, key=lambda c: (c['delta'], c['threshold']))['threshold'] if acceptable else 1.01
            assert artifact['threshold'] == threshold
            pred = infer(artifact, train.iloc[held_ids].reset_index(drop=True), p, nested.rules3.to_numpy()[held_ids])
            np.testing.assert_array_equal(pred, nested[mode].to_numpy()[held_ids])
            if mode == 'history':
                # Truncate real experiments at early rows and halfway through.
                # Recompute both the voltage rule and the gate from that prefix.
                for ids in held_frame.groupby('experiment_id').indices.values():
                    ids = ids[np.argsort(held_frame.relative_time_s.to_numpy()[ids], kind='stable')]
                    for end in sorted(set([1, 2, 3, 4, 5, max(1, len(ids)//2), len(ids)])):
                        if end > len(ids):
                            continue
                        subset = ids[:end]
                        frame_prefix = held_frame.iloc[subset].reset_index(drop=True)
                        prob_prefix = p[:, subset]
                        prefix_base = predict_rules(frame_prefix, prob_prefix.mean(0).argmax(1)+1, thresholds)[0]['rules3']
                        np.testing.assert_array_equal(prefix_base, causal_base[subset])
                        prefix_pred = infer(artifact, frame_prefix, prob_prefix, prefix_base)
                        np.testing.assert_array_equal(prefix_pred, pred[subset])
                        prefix_checks += 1
    summary = pd.read_csv(run/'nested_oof/summary.csv').set_index('method')
    for mode in ['rules3'] + list(MODES):
        expected = f1_score(nested.label, nested[mode], labels=range(1, 7), average='macro', zero_division=0)
        assert abs(expected - summary.loc[mode, 'macro_f1']) < 1e-12
        if mode != 'rules3':
            changed = nested[mode] != nested.rules3
            assert nested.loc[changed, 'rules3'].eq(4).all()
            assert nested.loc[changed, mode].eq(5).all()
    decision = json.loads((run/'decision.json').read_text())
    deployed = joblib.load(run/'deployment_gate.joblib')
    assert deployed['enabled'] == decision['accepted'] == all(decision['checks'].values())
    if not deployed['enabled']:
        assert deployed['threshold'] == 1.01
    report = dict(passed=True, fresh_models_checked=models, epochs_checked=models*15,
                  outer_experiments=train.experiment_id.nunique(), real_prefix_checks=prefix_checks, test_access=False,
                  checks=['Source hashes unchanged', 'Outer groups excluded from all inner fit/selection/OOF',
                          'Each inner OOF experiment predicted once', 'Scalers and rules reconstructed',
                          'Checkpoints selected only by inner selection history',
                          'All 45 saved inner OOF probability arrays reconciled',
                          'Gate training groups and threshold decisions reconciled',
                          'Outer gate predictions and Macro F1 independently reproduced',
                          'Real experiment prefixes reproduce full-sequence earlier rules and gate predictions',
                          'Only 4->5 changes allowed; failed deployment gate disabled'])
    (run/'audit_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
