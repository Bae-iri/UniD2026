"""Downstream feature/prediction prefix audit using fixed, row-aligned B probabilities.

Raw-image prefixes are checked separately by check_submission.py. This audit never
fits a model or preprocessing transform. Only the evaluator selects the cut points.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

import model as M

HERE = Path(__file__).resolve().parent


def no_fit(*args, **kwargs):
    raise AssertionError('Inference attempted to fit a model or preprocessing')


def features(frame, probs):
    upper, _ = M.T.features(frame, probs)
    lower, _ = M.lower_features(frame, probs)
    peak, decline, dropped = M.stage6_features(frame)
    return {'upper45': upper['45'], 'upper56': upper['56'], 'lower': lower,
            'S3': np.column_stack([peak, decline, dropped])}


def main(args):
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    HistGradientBoostingClassifier.fit = no_fit
    M.abc.SensorPreprocessor.__init__ = no_fit
    frame = pd.read_csv(args.data_dir/'test/test.csv').drop(columns=['label'], errors='ignore')
    bundle = joblib.load(HERE/'weights/pipeline.joblib')
    lower = joblib.load(HERE/'weights/lower_heads.joblib')
    probs = M.BModel(HERE/'weights').probabilities(frame, args.data_dir/'test')
    full = M.predict_labels(frame, probs, bundle, lower)
    expected = pd.read_csv(HERE/'submission_stage6.csv').set_index('sample_id').label
    np.testing.assert_array_equal(full, expected.loc[frame.sample_id].to_numpy())
    full_features = features(frame, probs)
    checks = 0
    for ids in frame.groupby('experiment_id', sort=False).indices.values():
        ids = ids[np.argsort(frame.relative_time_s.to_numpy()[ids], kind='stable')]
        cuts = {min(n, len(ids)) for n in [1, 2, 4, 5, 6]}
        cuts.update(np.linspace(1, len(ids), 12).astype(int).tolist())
        for cut in sorted(cuts):
            sub = ids[:cut]
            prefix = frame.iloc[sub].reset_index(drop=True)
            part_features = features(prefix, probs[:, sub])
            for name, values in part_features.items():
                np.testing.assert_allclose(values, full_features[name][sub], rtol=0, atol=0, equal_nan=True)
            np.testing.assert_array_equal(M.predict_labels(prefix, probs[:, sub], bundle, lower), full[sub])
            checks += 1
    permutation = np.random.default_rng(20261115).permutation(len(frame))
    shuffled = frame.iloc[permutation].reset_index(drop=True)
    shuffled_features = features(shuffled, probs[:, permutation])
    for name, values in shuffled_features.items():
        np.testing.assert_allclose(values, full_features[name][permutation], rtol=0, atol=0, equal_nan=True)
    renamed = frame.assign(sample_id=['renamed_'+str(i) for i in range(len(frame))], label=999)
    np.testing.assert_array_equal(M.predict_labels(renamed, probs, bundle, lower), full)
    result = {'passed': True, 'experiments': int(frame.experiment_id.nunique()), 'rows': len(frame),
              'downstream_feature_and_prediction_prefix_checks': checks,
              'prefix_mismatches': 0, 'feature_groups': list(full_features),
              'shuffled_features_exact': True, 'labels_and_sample_ids_ignored': True,
              'model_and_preprocessor_fit_blocked_during_audit': True,
              'scope': 'Fixed row-aligned B probabilities; check_submission.py separately checks raw-image prefixes'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE/'verification/causality.json')
    with threadpool_limits(limits=4):
        main(parser.parse_args())
