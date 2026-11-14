"""Strict nested, experiment-held-out training for conservative 4->5 correction.

Reuses existing outer B models, trains fresh inner B models from scratch. No
outer-held-out labels enter any inner model, scaler, gate or threshold choice.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from temporal45_model.conservative import MODES, fit_artifact, infer
from four_rule_cv.run import (abc, data_for, train_seed, View, fit_rules,
                              record_protocol)
from temporal45_model.causal_rules import predict_rules
import final_rules3

DATA = Path('/home/ubuntu/data/battery_data_light')
BASE = Path('/home/ubuntu/leewon/abc_runs/current_query_v1')
OUTER = ROOT/'four_rule_cv/runs/group5_seed42_43_44_v1'
SEEDS = [42, 43, 44]


def write(path, value):
    abc.common.write_json(path, value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_images(frame, device):
    manifest = pd.read_csv(BASE/'split_manifest.csv')
    paths = json.loads((BASE/'image_cache_manifest.json').read_text())
    images = torch.empty((len(frame), 4, 128, 128), dtype=torch.uint8, device=device)
    location = pd.Index(frame.sample_id)
    seen = np.zeros(len(frame), bool)
    for split in ['fit', 'dev']:
        ids = location.get_indexer(manifest.loc[manifest.split == split, 'sample_id'])
        if (ids < 0).any() or seen[ids].any():
            raise ValueError('Cache IDs inconsistent')
        seen[ids] = True
        a = np.array(np.load(paths[split], mmap_mode='r'), copy=True)
        images.index_copy_(0, torch.tensor(ids, device=device), torch.from_numpy(a).to(device))
    assert seen.all()
    return images


def verify_outer(train, oof):
    config = json.loads((OUTER/'config.json').read_text())
    verification = json.loads((OUTER/'verification.json').read_text())
    assert config['seeds'] == SEEDS and config['epochs'] == 15 and config['folds'] == 5
    assert sha(DATA/'train/train.csv') == verification['source_sha256'][str(DATA/'train/train.csv')]
    assert np.array_equal(train.sample_id, oof.sample_id)
    assert np.array_equal(train.label, oof.label)
    seen = set()
    for fold in range(5):
        split = pd.read_csv(OUTER/f'fold_{fold}/split_manifest.csv')
        assert split.sample_id.is_unique and set(split.sample_id) == set(train.sample_id)
        sets = [set(split.loc[split.role == r, 'experiment_id']) for r in ['fit', 'selection', 'holdout']]
        assert not any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3))
        assert not seen & sets[2]
        seen |= sets[2]
        assert set(train.loc[oof.fold == fold, 'experiment_id']) == sets[2]
    assert seen == set(train.experiment_id)


def main(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    protocol = dict(
        task='conservative 4->5 gate', outer_folds=5, inner_folds=3, seeds=SEEDS,
        base_epochs=15, modes=list(MODES), primary_mode='history',
        ablations=['probability', 'history_time'],
        thresholds=[.90, .95, .99], no_pass_threshold=1.01,
        threshold_selection='outer training only, 3-group meta CV; positive delta; corrections >= 2*harm; at least 3 improved experiments; improved>worsened; positive leave-one-experiment-out delta',
        regularization='linear logistic C=1, sum weights per experiment=1, no class balancing',
        base_checkpoint_selection='inner fit/selection split; no outer/inner heldout use',
        acceptance=['primary outer OOF delta>0', 'paired experiment bootstrap lower>0',
                    'more improved than worsened experiments', 'positive leave-one-experiment-out delta',
                    'official validation all and noE16 nonnegative delta'],
        deployment='disabled unless all acceptance checks pass; no submission produced',
        limitation='Previously inspected data; outer CV is strict computational separation, not a new blind dataset. Inner meta-CV is selection only.',
        validation_read_after_training=True, test_access=False,
        causal_baseline='first five voltage observations use prefix maximum; later frozen; never backfill',
        reuse_inner_base=str(args.reuse_inner) if args.reuse_inner else None,
        source_sha256={str(p): sha(p) for p in [Path(__file__), ROOT/'temporal45_model/conservative.py',
                                               ROOT/'compare_abc_phase_history.py', ROOT/'four_rule_cv/run.py',
                                               ROOT/'temporal45_model/causal_rules.py',
                                               ROOT/'final_rules3.py',
                                               DATA/'train/train.csv']})
    write(out/'protocol.json', protocol)
    if args.reuse_inner:
        previous_protocol = json.loads((args.reuse_inner/'protocol.json').read_text())
        # Only current-observation B models are reusable. The gate/rules are
        # always rebuilt under this run's strictly causal protocol.
        assert previous_protocol['seeds'] == SEEDS and previous_protocol['base_epochs'] == 15
        for path in [ROOT/'four_rule_cv/run.py', DATA/'train/train.csv']:
            assert previous_protocol['source_sha256'][str(path)] == sha(path)
    train = pd.read_csv(DATA/'train/train.csv')
    outer_rows = pd.read_csv(OUTER/'train_oof/predictions.csv')
    verify_outer(train, outer_rows)
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the planned nested training')
    images = train_images(train, device)
    y = train.label.to_numpy(int)
    n = len(train)
    oof_probs = np.zeros((3, n, 6), float)
    outputs = {name: np.zeros(n, int) for name in ['rules3'] + list(MODES)}
    details = []
    started = time.monotonic()
    for fold in range(5):
        folder = out/f'outer_{fold}'
        folder.mkdir()
        holdout = np.flatnonzero(outer_rows.fold.to_numpy() == fold)
        outer = np.flatnonzero(outer_rows.fold.to_numpy() != fold)
        held_groups = set(train.experiment_id.iloc[holdout])
        assert not held_groups & set(train.experiment_id.iloc[outer])
        threshold = fit_rules(train.iloc[outer])
        original_threshold = json.loads((OUTER/f'fold_{fold}/thresholds.json').read_text())
        assert json.loads(json.dumps(threshold)) == original_threshold
        outer_p = np.stack([np.load(OUTER/f'fold_{fold}/seed{s}_probs.npz')['holdout'] for s in SEEDS])
        np.testing.assert_array_equal(train.sample_id.iloc[holdout],
                                      pd.read_csv(OUTER/f'fold_{fold}/split_manifest.csv').query("role == 'holdout'").sample_id)
        oof_probs[:, holdout] = outer_p
        reproduced = predict_rules(train.iloc[holdout].reset_index(drop=True), outer_p.mean(0).argmax(1)+1, threshold)[0]['rules3']
        outputs['rules3'][holdout] = reproduced
        for mode in MODES:
            outputs[mode][holdout] = reproduced
        inner_probs = np.full((3, n, 6), np.nan)
        inner_base = np.zeros(n, int)
        for inner, (a, b) in enumerate(GroupKFold(3, shuffle=True, random_state=20261115+fold).split(outer, groups=train.experiment_id.iloc[outer])):
            pool, held = outer[a], outer[b]
            ia, ib = next(GroupShuffleSplit(1, test_size=.2, random_state=20261115+10*fold+inner).split(pool, groups=train.experiment_id.iloc[pool]))
            fit_ids, selection_ids = pool[ia], pool[ib]
            partitions = [set(train.experiment_id.iloc[i]) for i in [fit_ids, selection_ids, held, holdout]]
            assert not any(partitions[i] & partitions[j] for i in range(4) for j in range(i+1, 4))
            dest = folder/f'inner_{inner}'
            dest.mkdir()
            pd.concat([train.iloc[i][['sample_id', 'experiment_id']].assign(role=r)
                       for i, r in [(fit_ids, 'fit'), (selection_ids, 'selection'), (held, 'inner_holdout')]]).to_csv(dest/'split_manifest.csv', index=False)
            prep = abc.SensorPreprocessor(train.iloc[fit_ids])
            write(dest/'preprocessing.json', prep.metadata())
            inner_thresholds = fit_rules(train.iloc[pool])
            write(dest/'thresholds.json', inner_thresholds)
            data = data_for(train, images, prep, device)
            view = View(data, held)
            for si, seed in enumerate(SEEDS):
                # Original B architecture, initialized from scratch for every inner fold.
                prior = args.reuse_inner/f'outer_{fold}/inner_{inner}' if args.reuse_inner else None
                saved_files = [f'seed{seed}.pt', f'seed{seed}_history.csv', f'seed{seed}_holdout.npz']
                reuse = prior is not None and all((prior/name).exists() for name in saved_files)
                if reuse:
                    previous = pd.read_csv(prior/'split_manifest.csv')
                    pd.testing.assert_frame_equal(previous, pd.read_csv(dest/'split_manifest.csv'))
                    assert json.loads((prior/'preprocessing.json').read_text()) == prep.metadata()
                    z = np.load(prior/f'seed{seed}_holdout.npz')
                    np.testing.assert_array_equal(z['sample_id'], train.sample_id.iloc[held].to_numpy(str))
                    hp = z['probability']
                    for name in saved_files:
                        shutil.copy2(prior/name, dest/name)
                    print(f'REUSED CAUSAL CURRENT-ONLY B outer={fold} inner={inner} seed={seed}', flush=True)
                else:
                    base_args = argparse.Namespace(epochs=15)
                    m, epoch = train_seed(data, fit_ids, selection_ids, prep, base_args,
                                          f'{fold}/{inner}', seed, dest, device)
                    hp = abc.predict(m, view, None, device)
                    del m
                inner_probs[si, held] = hp
                np.savez_compressed(dest/f'seed{seed}_holdout.npz', sample_id=train.sample_id.iloc[held].to_numpy(str), probability=hp)
                write(out/'status.json', dict(phase='inner_base_training', outer_fold=fold,
                      inner_fold=inner, completed_seed=seed, elapsed_seconds=time.monotonic()-started))
            inner_base[held] = predict_rules(train.iloc[held].reset_index(drop=True),
                                            inner_probs[:, held].mean(0).argmax(1)+1, inner_thresholds)[0]['rules3']
            del data, view
        assert np.isfinite(inner_probs[:, outer]).all()
        assert np.isnan(inner_probs[:, holdout]).all()
        trainframe = train.iloc[outer].reset_index(drop=True)
        heldframe = train.iloc[holdout].reset_index(drop=True)
        for mode in MODES:
            with threadpool_limits(limits=2):
                artifact = fit_artifact(trainframe, inner_probs[:, outer], y[outer], inner_base[outer], mode)
            assert not set(artifact['fit_experiments']) & held_groups
            joblib.dump(artifact, folder/f'{mode}.joblib')
            outputs[mode][holdout] = infer(artifact, heldframe, outer_p, outputs['rules3'][holdout])
            details.append(dict(outer_fold=fold, mode=mode, threshold=artifact['threshold'],
                                inner_selection=artifact['threshold_selection']))
        np.savez_compressed(folder/'inner_oof.npz', sample_id=train.sample_id.iloc[outer].to_numpy(str),
                            seed_probs=inner_probs[:, outer], baseline=inner_base[outer], label=y[outer])
        write(out/'fold_decisions.json', details)
        print('OUTER COMPLETE', fold, [{k:v for k,v in z.items() if k != 'inner_selection'} for z in details if z['outer_fold'] == fold], flush=True)
    # The primary mode was fixed before outer predictions were produced. No mode
    # switching based on their scores is allowed.
    frame = train.copy()
    frame['fold'] = outer_rows.fold
    summary = record_protocol(out, 'nested_oof', frame, outputs)
    # Fit final gates on the independent-per-experiment original B OOF records.
    artifacts = {}
    for mode in MODES:
        with threadpool_limits(limits=2):
            artifacts[mode] = fit_artifact(train, oof_probs, y, outputs['rules3'], mode)
        joblib.dump(artifacts[mode], out/f'{mode}_candidate.joblib')
    del images
    torch.cuda.empty_cache()
    # Official validation is first read after all gate training/selection ends.
    validation = pd.read_csv(DATA/'validation/validation.csv')
    assert not set(validation.experiment_id) & set(train.experiment_id)
    vprobs = np.stack([np.mean([np.load(OUTER/f'fold_{fold}/seed{s}_probs.npz')['validation']
                              for fold in range(5)], axis=0) for s in SEEDS])
    vthreshold = fit_rules(train)
    vbase = predict_rules(validation, vprobs.mean(0).argmax(1)+1, vthreshold)[0]['rules3']
    vp = {'rules3': vbase, **{mode: infer(artifacts[mode], validation, vprobs, vbase) for mode in MODES}}
    vs = record_protocol(out, 'validation_fold_ensemble', validation, vp)
    ids = np.flatnonzero(validation.experiment_id.to_numpy() != final_rules3.E16)
    vn = record_protocol(out, 'validation_fold_ensemble_noE16', validation.iloc[ids].reset_index(drop=True),
                         {name: p[ids] for name, p in vp.items()})
    # Compatibility check against the actual original B+rules3 is descriptive:
    # this is a different base training recipe than the nested primary protocol.
    saved_p = np.stack([pd.read_csv(BASE/f'seed_{s}/B/validation_predictions.csv').set_index('sample_id').loc[validation.sample_id, [f'p{i}' for i in range(1, 7)]].to_numpy() for s in SEEDS])
    # Recompute the original B reference causally as well. Stored legacy rules3
    # predictions may have used future rows in the initial voltage window.
    saved_floor = json.loads((OUTER/'saved_fit_thresholds.json').read_text())
    saved_base = predict_rules(validation, saved_p.mean(0).argmax(1)+1, saved_floor)[0]['rules3']
    sp = {'rules3': saved_base, **{mode: infer(artifacts[mode], validation, saved_p, saved_base) for mode in MODES}}
    ss = record_protocol(out, 'validation_original_B', validation, sp)
    sn = record_protocol(out, 'validation_original_B_noE16', validation.iloc[ids].reset_index(drop=True),
                         {name: p[ids] for name, p in sp.items()})
    primary = next(s for s in summary if s['method'] == 'history')
    checks = dict(positive_oof=primary['delta_f1'] > 0,
                  positive_ci_lower=primary['ci_lower'] > 0,
                  more_improved_experiments=primary['better_experiments'] > primary['worse_experiments'],
                  positive_leave_one_out=primary['leave_one_experiment_out_min_delta'] > 0,
                  validation_all=next(s for s in vs if s['method'] == 'history')['delta_f1'] >= 0,
                  validation_noE16=next(s for s in vn if s['method'] == 'history')['delta_f1'] >= 0,
                  original_B_validation_all=next(s for s in ss if s['method'] == 'history')['delta_f1'] >= 0,
                  original_B_validation_noE16=next(s for s in sn if s['method'] == 'history')['delta_f1'] >= 0)
    accepted = all(checks.values())
    write(out/'decision.json', dict(primary_mode='history', checks={k: bool(v) for k,v in checks.items()},
          accepted=bool(accepted), deployment_enabled=bool(accepted), submission_created=False,
          note='Failed primary is rejected; ablation modes cannot replace it based on outer/validation results.'))
    deployed = dict(artifacts['history'], enabled=bool(accepted))
    if not accepted:
        deployed['threshold'] = 1.01
    joblib.dump(deployed, out/'deployment_gate.joblib')
    write(out/'status.json', dict(complete=True, fresh_inner_models=45,
                                 elapsed_seconds=time.monotonic()-started))
    print('NESTED OOF', pd.DataFrame(summary).to_string(index=False), flush=True)
    print('DECISION', checks, 'accepted=', accepted, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-inner', type=Path)
    main(parser.parse_args())
