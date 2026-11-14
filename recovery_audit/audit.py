"""Audit saved predictions without retraining or tuning on Test.

Run from the repository root with /home/ubuntu/venv/bin/python recovery_audit/audit.py.
All recipes are explicit; Test has no role in recipe selection.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/ubuntu/data/battery_data_light')
BASE = Path('/home/ubuntu/leewon/abc_runs/current_query_v1')
OUT = ROOT / 'recovery_audit' / 'results'
E16 = 'E_e5bcddc7d77f459ea541cf658ab4346e'
RECIPES = {
    'B': {'B': 1.},
    'cnx': {'cnx': 1.},
    'r50': {'r50': 1.},
    'equal_pretrained': {'cnx': .5, 'r50': .5},
    'B80_equal20': {'B': .8, 'cnx': .1, 'r50': .1},
    'B80_cnx20': {'B': .8, 'cnx': .2},
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cm(y, pred):
    return np.bincount((np.asarray(y) - 1) * 6 + np.asarray(pred) - 1,
                       minlength=36).reshape(6, 6)


def macro(matrix):
    den = matrix.sum(-1) + matrix.sum(-2)
    return np.divide(2 * np.diagonal(matrix, axis1=-2, axis2=-1), den,
                     out=np.zeros_like(den, dtype=float), where=den != 0).mean(-1)


def duration_rule(frame, prediction, consecutive=False, threshold=575):
    """Reproduce legacy rule; optional true consecutive-run comparator."""
    out = prediction.copy()
    t = frame.relative_time_s.to_numpy(float)
    exp = frame.experiment_id.to_numpy()
    start = None
    previous = None
    for i in np.lexsort((t, exp)):
        if exp[i] != previous:
            start, previous = None, exp[i]
        if prediction[i] == 4:
            if start is None:
                start = t[i]
            elif t[i] - start > threshold:
                out[i] = 5
        elif consecutive:
            start = None
    return out


def bootstrap(frame, a, b):
    """Paired experiment bootstrap, conditional on the already trained models."""
    groups = list(frame.groupby('experiment_id', sort=True).indices.values())
    y = frame.label.to_numpy(int)
    ca = np.stack([cm(y[g], a[g]) for g in groups])
    cb = np.stack([cm(y[g], b[g]) for g in groups])
    draws = np.random.default_rng(20261115).integers(0, len(groups), (2000, len(groups)))
    delta = macro(cb[draws].sum(1)) - macro(ca[draws].sum(1))
    return np.quantile(delta, [.025, .975]).tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    train = pd.read_csv(DATA / 'train/train.csv')
    fit, dev = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20261114)
                    .split(train, groups=train.experiment_id))
    frames = {'dev': train.iloc[dev].reset_index(drop=True),
              'validation': pd.read_csv(DATA / 'validation/validation.csv'),
              'test': pd.read_csv(DATA / 'test/test.csv')}
    assert not set(train.iloc[fit].experiment_id) & set(frames['dev'].experiment_id)
    inputs = {}
    arrays = {s: {} for s in frames}
    paths = {'cnx': ROOT / 'pre_runs/check_cnx/probs.npz',
             'r50': ROOT / 'pre_runs/check_r50_v2/probs.npz'}
    for name, path in paths.items():
        inputs[str(path)] = digest(path)
        with np.load(path) as z:
            for split in frames:
                arrays[split][name] = np.mean([z[f'{split}_{i}'] for i in range(3)], 0)
    for split, frame in frames.items():
        path = BASE / f'ensemble_B_{split}_predictions.csv'
        inputs[str(path)] = digest(path)
        original = pd.read_csv(path)
        assert np.array_equal(original.sample_id, frame.sample_id)
        if split != 'test':
            assert np.array_equal(original.label, frame.label)
        arrays[split]['B'] = original[[f'p{i}' for i in range(1, 7)]].to_numpy()
        for p in arrays[split].values():
            assert p.shape == (len(frame), 6) and np.isfinite(p).all()
            assert (p >= 0).all() and np.allclose(p.sum(1), 1, atol=1e-5)
    # NPZ files have no IDs. Reproduce the saved metrics before trusting reconstructed order.
    for name, directory in [('cnx', 'check_cnx'), ('r50', 'check_r50_v2')]:
        ref = pd.read_csv(ROOT / f'pre_runs/{directory}/check.csv').iloc[0]
        for split, metric in [('dev', 'dev_f1'), ('validation', 'val_f1')]:
            score = f1_score(frames[split].label, arrays[split][name].argmax(1) + 1,
                             labels=range(1, 7), average='macro')
            assert abs(score - ref[metric]) < 1e-12
    summary, experiments, intervals = [], [], []
    predictions = {}
    for split, frame in frames.items():
        predictions[split] = {}
        base = arrays[split]['B'].argmax(1) + 1
        for name, weights in RECIPES.items():
            prob = sum(arrays[split][key] * w for key, w in weights.items())
            raw = prob.argmax(1) + 1
            for variant, pred in [('raw', raw), ('legacy575', duration_rule(frame, raw)),
                                  ('consecutive575', duration_rule(frame, raw, True))]:
                key = f'{name}_{variant}'
                predictions[split][key] = pred
                if split == 'test':
                    continue
                y = frame.label.to_numpy(int)
                m = cm(y, pred)
                assert abs(macro(m) - f1_score(y, pred, labels=range(1, 7), average='macro')) < 1e-12
                keep = frame.experiment_id.to_numpy() != E16
                summary.append(dict(split=split, recipe=name, variant=variant, f1=float(macro(m)),
                                    noE16_f1=float(macro(cm(y[keep], pred[keep]))),
                                    corrected=int(((base != y) & (pred == y)).sum()),
                                    harmed=int(((base == y) & (pred != y)).sum()),
                                    E16_5_correct=int(((~keep) & (y == 5) & (pred == 5)).sum()),
                                    E16_5_to_4=int(((~keep) & (y == 5) & (pred == 4)).sum()),
                                    **{f'F1_{i+1}': float(v) for i, v in enumerate(
                                        f1_score(y, pred, labels=range(1, 7), average=None))}))
                lo, hi = bootstrap(frame, base, pred)
                intervals.append(dict(split=split, candidate=key, lower=lo, upper=hi))
                for exp, ids in frame.groupby('experiment_id', sort=True).indices.items():
                    experiments.append(dict(split=split, candidate=key, experiment_id=exp,
                                            delta_f1=float(macro(cm(y[ids], pred[ids])) - macro(cm(y[ids], base[ids]))),
                                            rows=len(ids), changed=int((pred[ids] != base[ids]).sum())))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'scores.csv', index=False)
    pd.DataFrame(experiments).to_csv(OUT / 'per_experiment.csv', index=False)
    pd.DataFrame(intervals).to_csv(OUT / 'bootstrap.csv', index=False)
    test = frames['test']
    base = predictions['test']['B_raw']
    sample = pd.read_csv(DATA / 'sample_submission.csv')
    def export(name, pred):
        label = pd.Series(pred, index=test.sample_id)
        sub = sample[['sample_id']].copy()
        sub['label'] = sub.sample_id.map(label)
        assert len(sub) == 3451 and sub.sample_id.is_unique
        assert sub.label.notna().all() and sub.label.between(1, 6).all()
        sub.label = sub.label.astype(int)
        path = OUT / name
        sub.to_csv(path, index=False)
        return dict(path=str(path), sha256=digest(path), changes_vs_B=int((pred != base).sum()))
    # Recovery is the evaluated B. The conservative blend is a research candidate only.
    exports = [export('submission_B_original_recovered.csv', base),
               export('submission_B80_equal20_fitonly_raw_candidate.csv', predictions['test']['B80_equal20_raw'])]
    restored = pd.read_csv(exports[0]['path'])
    reference = pd.read_csv(BASE / 'submission_B.csv')
    pd.testing.assert_frame_equal(restored, reference)
    shifts = []
    for name, directory in [('cnx', 'final_cnx'), ('r50', 'final_r50')]:
        path = ROOT / f'pre_runs/{directory}/probs.npz'
        inputs[str(path)] = digest(path)
        with np.load(path) as z:
            final = np.mean([z[f'test_{i}'] for i in range(3)], 0).argmax(1) + 1
        before = predictions['test'][f'{name}_raw']
        shifts.append(dict(model=name, check_vs_final=int((before != final).sum()),
                           check_vs_B=int((before != base).sum()), final_vs_B=int((final != base).sum()),
                           legacy_rule_changes=int((final != duration_rule(test, final)).sum())))
    pd.DataFrame(shifts).to_csv(OUT / 'retraining_shift.csv', index=False)
    failed = ROOT / 'final_submit/submission_convnext_long4rule.csv'
    manifest = dict(recipes=RECIPES, input_sha256=inputs, exports=exports,
                    failed_submission=dict(path=str(failed), sha256=digest(failed), user_reported_score=.77),
                    external_score_of_candidates=None, target=.93, target_achieved=False,
                    rule_threshold=575, rule_threshold_tuned=False,
                    limitations=['Repeatedly inspected dev/validation; exploratory comparisons, not fresh outer CV.',
                                 'NPZ row identity reconstructed from source code and reproduced metrics; no embedded IDs.',
                                 'Bootstrap conditions on frozen models and excludes adaptive selection uncertainty.',
                                 'Test labels unavailable. Changed rows do not imply corrected rows.',
                                 'No GPU device available. No new neural training performed.'])
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(summary[summary.variant == 'raw'][['split', 'recipe', 'f1', 'noE16_f1', 'corrected', 'harmed']].to_string(index=False))
    print(json.dumps(exports, indent=2))


if __name__ == '__main__':
    main()
