"""Reproduce an archived submission from aligned seed predictions and artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pipeline import predict, predict_stages


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_artifact(upper_path=None, lower_path=None, combined_path=None):
    if combined_path is not None:
        if upper_path is not None or lower_path is not None:
            raise ValueError('Use either one combined artifact or the upper/lower pair')
        artifact = joblib.load(combined_path)
        if not {'heads','gate','low'} <= artifact.keys():
            raise ValueError('Combined artifact must include all correction models')
        return artifact
    if upper_path is None or lower_path is None:
        raise ValueError('Both upper and lower artifacts are required')
    source = joblib.load(upper_path)
    return dict(heads=source['baseline_heads'], gate=source['precision_head'],
                low=joblib.load(lower_path))


def load_probs(root, split, ids):
    result = []
    for seed in [42, 43, 44]:
        table = pd.read_csv(Path(root)/f'seed_{seed}/B/{split}_predictions.csv')
        if not table.sample_id.is_unique:
            raise ValueError('Duplicate probability sample ID')
        result.append(table.set_index('sample_id').loc[ids, [f'p{i}' for i in range(1, 7)]].to_numpy())
    return np.stack(result)


def invariants(frame, probs, artifact, full):
    checks = dict(prefixes=0, prefix_mismatches=0, per_experiment=True,
                  shuffled=True, labels_and_ids_ignored=True, feature_prefix_checks=0)
    from core import upper, lower
    x, _ = upper.features(frame, probs)
    lx, _ = lower.features(frame, probs)
    for ids in frame.groupby('experiment_id').indices.values():
        ids = ids[np.argsort(frame.relative_time_s.to_numpy()[ids], kind='stable')]
        one = predict(artifact, frame.iloc[ids].reset_index(drop=True), probs[:, ids])['S3']
        checks['per_experiment'] &= bool(np.array_equal(one, full[ids]))
        for end in sorted(set(np.linspace(1, len(ids), 12).astype(int))):
            sub = ids[:end]
            f = frame.iloc[sub].reset_index(drop=True)
            pr = probs[:, sub]
            checks['prefixes'] += 1
            checks['prefix_mismatches'] += int(not np.array_equal(predict(artifact, f, pr)['S3'], full[sub]))
            # Feature equality also catches future access masked by stable labels.
            xx, _ = upper.features(f, pr)
            ll, _ = lower.features(f, pr)
            for pair in x:
                np.testing.assert_allclose(xx[pair], x[pair][sub], rtol=0, atol=0, equal_nan=True)
            np.testing.assert_allclose(ll, lx[sub], rtol=0, atol=0, equal_nan=True)
            checks['feature_prefix_checks'] += 1
    perm = np.random.default_rng(20261115).permutation(len(frame))
    checks['shuffled'] = bool(np.array_equal(predict(artifact, frame.iloc[perm].reset_index(drop=True), probs[:, perm])['S3'], full[perm]))
    altered = frame.assign(label=1, sample_id=['audit_'+str(i) for i in range(len(frame))])
    checks['labels_and_ids_ignored'] = bool(np.array_equal(predict(artifact, altered, probs)['S3'], full))
    assert checks['prefix_mismatches'] == 0 and all(checks[k] for k in ['per_experiment', 'shuffled', 'labels_and_ids_ignored'])
    return checks


def main(args):
    frame = pd.read_csv(args.input)
    p = load_probs(args.base_predictions, args.split, frame.sample_id)
    artifact = load_artifact(args.upper_artifact, args.lower_artifact)
    output = predict_stages(artifact, frame, p)
    checks = invariants(frame, p, artifact, output['S3']) if args.audit else {}
    table = pd.DataFrame({'sample_id': frame.sample_id, 'label': output['S3']})
    if args.sample_submission:
        sample = pd.read_csv(args.sample_submission)
        assert sample.sample_id.is_unique and set(sample.sample_id) == set(frame.sample_id)
        table = table.set_index('sample_id').loc[sample.sample_id].reset_index()
    assert table.sample_id.is_unique and table.label.isin(range(1, 7)).all()
    reference_mismatches = None
    if args.reference:
        reference = pd.read_csv(args.reference)
        assert reference.sample_id.is_unique and set(reference.sample_id) == set(table.sample_id)
        reference_mismatches = int((table.label.to_numpy() != reference.set_index('sample_id').loc[table.sample_id].label.to_numpy()).sum())
        assert reference_mismatches == 0
    table.to_csv(args.output, index=False, mode='x')
    changed = output['J096'] != output['S3']
    lower_changed = output['precision65'] != output['J096']
    assert np.array_equal(output['S3'][lower_changed], output['J096'][lower_changed])
    report = dict(rows=len(table), experiments=frame.experiment_id.nunique(),
        reference_mismatches=reference_mismatches, checks=checks,
        changed_vs_J096=int(changed.sum()),
        lower43_changed=int(lower_changed.sum()),
        lower43_preserved_after_S3=int((output['S3'][lower_changed] == 3).sum()),
        changes={f'{a}->{b}': int(n) for (a, b), n in pd.DataFrame({'a':output['J096'][changed], 'b':output['S3'][changed]}).value_counts().items()},
        class_counts_J096=np.bincount(output['J096'], minlength=7)[1:].tolist(),
        class_counts_S3=np.bincount(output['S3'], minlength=7)[1:].tolist(),
        sha256={str(path): sha(path) for path in [args.input, args.upper_artifact, args.lower_artifact, args.output]},
        limits='Prefix tests here cover downstream features and labels with fixed aligned B probabilities; raw backbone inference is a separate audit.')
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['input', 'base-predictions', 'upper-artifact', 'lower-artifact', 'output', 'report']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--sample-submission', type=Path)
    parser.add_argument('--audit', action='store_true')
    with threadpool_limits(limits=2):
        main(parser.parse_args())
