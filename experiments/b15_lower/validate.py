"""Audit fixed J096/S3 on independent outer-group backbone predictions.

Backbone training/selection/preprocessing and specialist fitting exclude every
outer evaluation experiment. Thresholds (.7, .7, .5, H=300, d=5) stay frozen.
Previously explored datasets make this a robustness audit, not a new blind test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pipeline import fit_frozen_recipe, predict, s3
from reproduce import load_artifact, load_probs, sha
from core.precision import matrix, macro

SEEDS = [42, 43, 44]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


class BackboneAudit:
    def __init__(self, frame):
        self.frame = frame
        self.records = []

    def load(self, folder, allowed, forbidden):
        out = []
        frame = self.frame
        allowed = set(frame.experiment_id.iloc[allowed])
        for seed in SEEDS:
            dest = folder/f'seed_{seed}'
            record = json.loads((dest/'complete.json').read_text())
            manifest = pd.read_csv(dest/'split_manifest.csv')
            np.testing.assert_array_equal(manifest.sample_id, frame.sample_id)
            np.testing.assert_array_equal(manifest.experiment_id, frame.experiment_id)
            fit = manifest.role.to_numpy() == 'fit'
            selection = manifest.role.to_numpy() == 'selection'
            fit_exps, select_exps = set(frame.experiment_id[fit]), set(frame.experiment_id[selection])
            assert not fit_exps & select_exps
            assert fit_exps | select_exps == allowed
            assert not (fit_exps | select_exps) & forbidden
            assert fit_exps == set(record['fit_experiments'])
            assert select_exps == set(record['selection_experiments'])
            for filename, key in [('best.pt', 'checkpoint_sha256'), ('probabilities.npy', 'probability_sha256'),
                                  ('split_manifest.csv', 'split_manifest_sha256'), ('preprocessing.json', 'preprocessing_sha256')]:
                assert sha(dest/filename) == record[key], str(dest/filename)
            meta = json.loads((dest/'preprocessing.json').read_text())
            cols = meta['electro']+meta['gas']
            numeric = frame.loc[fit, cols].astype(float).replace([np.inf, -np.inf], np.nan)
            median = numeric.median().fillna(0)
            filled = numeric.fillna(median)
            for key, actual in [('median', median), ('mean', filled.mean()),
                                ('std', filled.std(ddof=0).clip(lower=1e-6))]:
                np.testing.assert_allclose(meta[key], actual.to_numpy(), rtol=1e-12, atol=1e-12)
            history = pd.read_csv(dest/'history.csv')
            assert int(history.iloc[history.selection_macro_f1.to_numpy().argmax()].epoch) == record['best_epoch']
            out.append(np.load(dest/'probabilities.npy'))
            self.records.append(dict(path=str(dest), seed=seed, fit_experiments=len(fit_exps),
                selection_experiments=len(select_exps), excluded_evaluation_experiments=len(forbidden),
                checkpoint_sha256=record['checkpoint_sha256'], probability_sha256=record['probability_sha256']))
        p = np.stack(out)
        assert p.shape == (3, len(frame), 6) and np.isfinite(p).all()
        np.testing.assert_allclose(p.sum(2), 1, atol=1e-5)
        return p


def subsample(frame, view):
    chosen = []
    rng = np.random.default_rng(20261115)
    for ids in frame.groupby('experiment_id', sort=False).indices.values():
        ids = ids[np.argsort(frame.relative_time_s.to_numpy()[ids], kind='stable')]
        if view == 'original':
            selected = ids
        elif view.startswith('stride'):
            step, phase = map(int, view.removeprefix('stride').split('_'))
            selected = ids[phase::step]
        elif view == 'alternating_2_3':
            positions, i, j = [], 0, 0
            while i < len(ids):
                positions.append(i)
                i += 2 if j % 2 == 0 else 3
                j += 1
            selected = ids[positions]
        elif view == 'missing30':
            keep = rng.random(len(ids)) >= .3
            keep[0] = True
            selected = ids[keep]
        else:
            raise ValueError(view)
        chosen.extend(selected)
    return np.sort(np.array(chosen, int))


VIEWS = ['original', 'stride2_0', 'stride2_1', 'stride3_0', 'stride3_1', 'stride3_2',
         'alternating_2_3', 'missing30']


def scope_predictions(frame, root, scope, audit, dest, suite):
    folder = root/scope
    with np.load(folder/'partitions.npz') as z:
        training, held = z['training'], z['held']
        parts = [z[f'inner_{k}'] for k in range(3)]
    outer_exps = set(frame.experiment_id.iloc[held])
    assert not set(frame.experiment_id.iloc[training]) & outer_exps
    assert sorted(np.concatenate(parts).tolist()) == sorted(training.tolist())
    oof = np.full((3, len(frame), 6), np.nan)
    for j, part in enumerate(parts):
        allowed = np.setdiff1d(training, part)
        forbidden = outer_exps | set(frame.experiment_id.iloc[part])
        inner = audit.load(folder/f'leave_{j}', allowed, forbidden)
        oof[:, part] = inner[:, part]
    assert np.isfinite(oof[:, training]).all()
    model = fit_frozen_recipe(frame.iloc[training].reset_index(drop=True), oof[:, training])
    assert not set(model['fit_experiments']) & outer_exps
    joblib.dump(model, dest/f'{suite}_{scope}.joblib')
    pp = audit.load(folder/'full', training, outer_exps)
    hf = frame.iloc[held].reset_index(drop=True)
    rows = []
    for view in VIEWS:
        sub = subsample(hf, view)
        f = hf.iloc[sub].reset_index(drop=True)
        p = pp[:, held][:, sub]
        output = predict(model, f, p)
        tab = f[['sample_id', 'experiment_id', 'label', 'bat_cc', 'soc_status', 'tr_method']].copy()
        tab['suite'], tab['scope'], tab['view'] = suite, scope, view
        for name, values in output.items():
            tab[name] = values
        # The small original grid is diagnostic only; no new winner is selected.
        if view == 'original':
            for height in [300, 400]:
                for decline in [3, 5, 10]:
                    tab[f'S3_H{height}_d{decline}'] = s3(f, output['J096'], height, decline)
        rows.append(tab)
    print(suite, scope, 'fit', len(model['fit_experiments']), 'held', len(outer_exps),
          'S3', round(float(macro(matrix(hf.label, rows[0].S3))), 5), flush=True)
    return rows


def summary(table, method='S3', draws=6000):
    y, base, p = table.label.to_numpy(int), table.J096.to_numpy(int), table[method].to_numpy(int)
    groups = list(table.groupby('experiment_id').indices.values())
    before = np.stack([matrix(y[i], base[i]) for i in groups])
    after = np.stack([matrix(y[i], p[i]) for i in groups])
    # Repeated appearances of the same experiment always stay together.
    weights = np.random.default_rng(20261115).multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=draws)
    boot_before = macro(np.einsum('be,eij->bij', weights, before))
    boot_after = macro(np.einsum('be,eij->bij', weights, after))
    delta = macro(after)-macro(before)
    cm = after.sum(0)
    f1 = np.divide(2*np.diag(cm), cm.sum(0)+cm.sum(1), out=np.zeros(6), where=cm.sum(0)+cm.sum(1) > 0)
    return dict(method=method, rows=len(table), experiments=len(groups),
        base_f1=float(macro(before.sum(0))), macro_f1=float(macro(cm)),
        delta=float(macro(cm)-macro(before.sum(0))),
        bootstrap_f1_95=np.quantile(boot_after, [.025, .975]).tolist(),
        bootstrap_delta_95=np.quantile(boot_after-boot_before, [.025, .975]).tolist(),
        better_experiments=int((delta > 1e-12).sum()), worse_experiments=int((delta < -1e-12).sum()),
        corrected=int(((base != y) & (p == y)).sum()), harmed=int(((base == y) & (p != y)).sum()),
        changed=int((base != p).sum()), per_class_f1=f1.tolist(), confusion_matrix=cm.tolist())


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    records, tables = [], []
    train = pd.read_csv(args.data/'train/train.csv')
    frame = pd.read_csv(args.repeated/'development.csv')
    np.testing.assert_array_equal(pd.read_csv(args.nested/'row_manifest.csv').sample_id, train.sample_id)
    for suite, data, root, scopes in [
        ('train78', train, args.nested, [f'outer_{k}' for k in range(5)]),
        ('repeat77', frame, args.repeated/'backbones', [f'repeat_{r}_fold_{k}' for r in range(2) for k in range(5)])]:
        audit = BackboneAudit(data)
        for scope in scopes:
            tables += scope_predictions(data, root, scope, audit, args.output, suite)
        records += audit.records
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(args.output/'outer_predictions.csv', index=False)
    results = {}
    for (suite, view), sub in table.groupby(['suite', 'view']):
        results[suite+'_'+view] = summary(sub)
    write(args.output/'outer_summary.json', results)
    split = []
    for (suite, scope, view), sub in table.groupby(['suite', 'scope', 'view']):
        for method in ['rule1', 'J096', 'S3']:
            split.append(dict(suite=suite, scope=scope, view=view, method=method,
                              macro_f1=float(macro(matrix(sub.label, sub[method])))))
    pd.DataFrame(split).to_csv(args.output/'split_scores.csv', index=False)
    conditions = []
    original = table[table.view == 'original']
    for suite, sub in original.groupby('suite'):
        for field in ['bat_cc', 'soc_status', 'tr_method']:
            for value, group in sub.groupby(field):
                conditions.append(dict(suite=suite, field=field, value=str(value), **summary(group)))
    write(args.output/'condition_summary.json', conditions)
    sensitivity = []
    for suite, sub in original.groupby('suite'):
        for height in [300, 400]:
            for decline in [3, 5, 10]:
                sensitivity.append(dict(suite=suite, height=height, decline=decline,
                    **summary(sub, f'S3_H{height}_d{decline}')))
    write(args.output/'parameter_sensitivity.json', sensitivity)
    # Exact archived weights: all official validation experiments, with E16 separately.
    val = pd.read_csv(args.data/'validation/validation.csv')
    pp = load_probs(args.base_predictions, 'validation', val.sample_id)
    artifact = load_artifact(args.upper_artifact, args.lower_artifact)
    val_rows = []
    for view in VIEWS:
        ix = subsample(val, view)
        f = val.iloc[ix].reset_index(drop=True)
        output = predict(artifact, f, pp[:, ix])
        t = f[['sample_id', 'experiment_id', 'label']].copy()
        t['view'] = view
        for name, values in output.items():
            t[name] = values
        val_rows.append(t)
    validation = pd.concat(val_rows, ignore_index=True)
    validation.to_csv(args.output/'validation_predictions.csv', index=False)
    val_result = {v: summary(s) for v, s in validation.groupby('view')}
    e16 = 'E_e5bcddc7d77f459ea541cf658ab4346e'
    val_result['original_without_E16'] = summary(validation[(validation.view == 'original') & (validation.experiment_id != e16)])
    write(args.output/'validation_summary.json', val_result)
    # Descriptive 30/70 split experiment; never extrapolate a calibrated private score.
    simulation = {}
    for suite, sub in original.groupby('suite'):
        groups = list(sub.groupby('experiment_id').indices.values())
        y, p = sub.label.to_numpy(int), sub.S3.to_numpy(int)
        cms = np.stack([matrix(y[i], p[i]) for i in groups])
        rng = np.random.default_rng(20261115)
        public, private = [], []
        for _ in range(5000):
            perm = rng.permutation(len(groups))
            count = max(1, round(len(groups)*.3))
            public.append(float(macro(cms[perm[:count]].sum(0))))
            private.append(float(macro(cms[perm[count:]].sum(0))))
        public, private = np.array(public), np.array(private)
        simulation[suite] = dict(unit='experiment_id; 30/70 experiments, not necessarily 30/70 rows',
            simulations=5000, public_95=np.quantile(public, [.025,.975]).tolist(),
            private_95=np.quantile(private, [.025,.975]).tolist(),
            private_minus_public_95=np.quantile(private-public, [.025,.975]).tolist(),
            private_below_public_fraction=float((private < public).mean()),
            limitation='Descriptive partition sensitivity; not the hidden score confidence interval and not conditioned on public 0.9719.')
    write(args.output/'public_private_simulation.json', simulation)
    write(args.output/'audit.json', dict(passed=True, backbone_models_checked=len(records),
        unique_backbone_models=len(set(r['path'] for r in records)), outer_splits=15,
        frozen_thresholds=dict(head56=.7, gate65=.7, lower43=.5, lower34=1.01, S3_H=300, S3_d=5),
        outer_experiments_excluded_from_all_fit_preprocessing_checkpoint_and_head_fitting=True,
        training_head_probabilities_cross_fitted_inside_each_outer_scope=True,
        preprocessing_independently_recomputed=True, checkpoint_selection_and_file_hashes_checked=True,
        public_score_used_for_selection=False, test_labels_used=False,
        train_sha256=sha(args.data/'train/train.csv'), validation_sha256=sha(args.data/'validation/validation.csv'),
        sources={str(p.relative_to(Path(__file__).parent)):sha(p) for p in Path(__file__).parent.rglob('*.py')},
        limits=['Archived submitted weights only reproduce on Test and evaluate on already explored Validation.',
                'Outer CV independently refits the fixed submitted structure; it does not certify archived checkpoint accuracy.',
                'The gate is refit directly with its eligible baseline; archived training used OOF baseline in threshold selection. Thresholds stay frozen here.',
                'Repeated splits and train78/repeat77 suites overlap; do not treat them as independent datasets.',
                'Frozen hypotheses and thresholds originated from previously inspected data; no fresh blind benchmark.',
                'Capacity summaries are stratified evaluation, not leave-capacity-out training.',
                'Previously used final confirmation is deliberately not reused.']))
    write(args.output/'backbone_lineage.json', records)
    print('RESULT', json.dumps({k:v for k,v in results.items() if k.endswith('_original')}, ensure_ascii=False), flush=True)
    print('VALIDATION', json.dumps(val_result['original'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'nested', 'repeated', 'base-predictions', 'upper-artifact', 'lower-artifact', 'output']:
        ap.add_argument('--'+name, type=Path, required=True)
    with threadpool_limits(limits=2):
        main(ap.parse_args())
