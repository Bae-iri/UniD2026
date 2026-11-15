"""Check archived fit lineage, raw inference and raw prefix invariance."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from infer import Runtime, abc
from pipeline import predict
from reproduce import load_probs, sha


def main(args):
    torch.set_num_threads(4)
    train = pd.read_csv(args.data/'train/train.csv')
    val = pd.read_csv(args.data/'validation/validation.csv')
    test = pd.read_csv(args.data/'test/test.csv')
    assert not set(train.experiment_id) & set(val.experiment_id)
    assert not (set(train.experiment_id) | set(val.experiment_id)) & set(test.experiment_id)
    manifest = pd.read_csv(args.base/'split_manifest.csv')
    fit_ids = manifest.loc[manifest.split == 'fit', 'sample_id']
    dev_ids = manifest.loc[manifest.split == 'dev', 'sample_id']
    indexed = train.set_index('sample_id', drop=False)
    fit, dev = indexed.loc[fit_ids], indexed.loc[dev_ids]
    assert set(fit.sample_id) | set(dev.sample_id) == set(train.sample_id)
    assert not set(fit.experiment_id) & set(dev.experiment_id)
    actual = abc.SensorPreprocessor(fit).metadata()
    saved = json.loads((args.base/'preprocessing.json').read_text())
    assert actual == saved
    for seed in [42, 43, 44]:
        folder = args.base/f'seed_{seed}/B'
        history = pd.read_csv(folder/'history.csv')
        checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        assert int(history.iloc[history.dev_macro_f1.to_numpy().argmax()].epoch) == checkpoint['epoch']
    runtime = Runtime(args.base, args.upper_artifact, args.lower_artifact, args.device)
    assert set(runtime.artifact['heads']['fit_experiments']) == set(train.experiment_id)
    assert set(runtime.artifact['gate']['fit_experiments']) == set(train.experiment_id)
    original_probs = load_probs(args.base, 'test', test.sample_id)
    labels, pp = runtime.labels(test, args.data/'test')
    reference = pd.read_csv(args.reference).set_index('sample_id').loc[test.sample_id].label.to_numpy(int)
    mismatches = int((labels != reference).sum())
    assert mismatches == 0
    probability_error = float(abs(pp-original_probs).max())
    # Archived probabilities came from a GPU; this audit may run on a CPU.
    # Record cross-device numeric drift, while requiring exact submitted labels.
    # Same-device raw prefix probabilities still have a strict check below.
    print('Raw inference reproduces all', len(test), 'labels; maximum probability error', probability_error, flush=True)
    prefix_count, max_prefix_error = 0, 0.
    for ids in test.groupby('experiment_id').indices.values():
        ids = ids[np.argsort(test.relative_time_s.to_numpy()[ids], kind='stable')]
        for end in [min(5, len(ids)), max(1, len(ids)//2)]:
            ix = ids[:end]
            sub = test.iloc[ix].reset_index(drop=True)
            label, prob = runtime.labels(sub, args.data/'test')
            np.testing.assert_array_equal(label, labels[ix])
            np.testing.assert_allclose(prob, pp[:, ix], atol=2e-5, rtol=1e-4)
            max_prefix_error = max(max_prefix_error, float(abs(prob-pp[:, ix]).max()))
            prefix_count += 1
        print('Raw prefixes PASS', ids.size, 'rows', flush=True)
    shift = {}
    for name, f in [('train', train), ('validation', val), ('test', test)]:
        times = f.sort_values(['experiment_id', 'relative_time_s']).groupby('experiment_id').relative_time_s.diff()
        shift[name] = dict(rows=len(f), experiments=f.experiment_id.nunique(),
            observation_gaps_s={str(k):int(v) for k,v in times.value_counts().items()},
            capacity_experiments={str(k):int(v) for k,v in f.groupby('experiment_id').first().bat_cc.value_counts().items()})
    result = dict(passed=True, raw_test_label_mismatches=mismatches,
        raw_test_rows=len(test), raw_test_experiments=test.experiment_id.nunique(),
        raw_prefix_checks=prefix_count, raw_prefix_label_mismatches=0,
        max_probability_difference_vs_archived=probability_error,
        archived_probability_device='GPU; cross-device probability differences are descriptive',
        max_probability_prefix_difference=max_prefix_error,
        device=args.device, original_fit_experiments=fit.experiment_id.nunique(),
        original_checkpoint_selection_experiments=dev.experiment_id.nunique(),
        original_preprocessing_recomputed_exactly=True,
        official_splits_disjoint=True, original_checkpoint_selection_matches_history=True,
        original_upper_heads_and_gate_fit_only_official_train=True,
        low_model_lineage_limit='Archived lower artifact has no fit_experiments field; raw/feature invariance is verified, full historical lower training lineage cannot be proven from the artifact alone.',
        dataset_shift=shift,
        artifacts={str(p):sha(p) for p in [args.upper_artifact,args.lower_artifact,args.base/'preprocessing.json']+
                   [args.base/f'seed_{s}/B/best.pt' for s in [42,43,44]]})
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'base', 'upper-artifact', 'lower-artifact', 'reference', 'report']:
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    with threadpool_limits(limits=2):
        main(ap.parse_args())
