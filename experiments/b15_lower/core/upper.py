"""Two small past-only specialists for true5/pred4 and true6/pred5 errors."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .causal import causal_features


def matrix(y, p):
    return np.bincount((np.asarray(y)-1)*6+np.asarray(p)-1, minlength=36).reshape(6, 6)


def f1(c):
    den = c.sum(-1)+c.sum(-2)
    return np.divide(2*np.diagonal(c, axis1=-2, axis2=-1), den, out=np.zeros_like(den, dtype=float), where=den > 0)

PAIRS = {'45': (4, 5), '56': (5, 6)}
THRESHOLDS = (.5, .7, .9)
PARAMS = dict(max_iter=120, max_leaf_nodes=5, max_depth=3,
              learning_rate=.05, min_samples_leaf=64, l2_regularization=5.,
              early_stopping=False, random_state=42)
SIGNALS = ['meas_v', 'meas_t1', 'meas_t2', 'thermal_t', 'meas_p']


def features(frame, seed_probs):
    p = np.asarray(seed_probs, float)
    if p.shape != (3, len(frame), 6) or not np.isfinite(p).all():
        raise ValueError('Three aligned seed probability arrays required')
    if (p < 0).any() or not np.allclose(p.sum(2), 1, atol=1e-5):
        raise ValueError('Invalid probabilities')
    t = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    h = causal_features(frame)
    sensor_values = frame[SIGNALS].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    sensor_values[~np.isfinite(sensor_values)] = np.nan
    # Fixed transforms do not fit anything on an evaluation experiment.
    raw = np.sign(sensor_values) * np.log1p(np.abs(sensor_values))
    cols = [raw, h]
    names = SIGNALS + ['elapsed_log1p'] + [f'{s}_{v}' for s in SIGNALS[:4]
                                        for v in ['extreme', 'difference', 'age_log1p', 'valid']]
    physical_history = np.full((len(frame), 15), np.nan)
    probability_history = np.zeros((len(frame), 24), float)
    mean = p.mean(0)
    pred = mean.argmax(1)+1
    for ids in frame.groupby('experiment_id', sort=False).indices.values():
        ids = ids[np.argsort(t[ids], kind='stable')]
        if (np.diff(t[ids]) <= 0).any():
            raise ValueError('Duplicate/nonincreasing observation times')
        ts = t[ids]
        peak_p = np.zeros(3)
        last6 = None
        run6_start = None
        for j, row in enumerate(ids):
            peak_p = np.maximum(peak_p, mean[row, 3:6])
            if pred[row] == 6:
                last6 = t[row]
                if run6_start is None:
                    run6_start = t[row]
            else:
                run6_start = None
            probability_history[row, :6] = [*peak_p, float(last6 is not None),
                np.log1p(t[row]-last6) if last6 is not None else 0.,
                np.log1p(t[row]-run6_start) if run6_start is not None else 0.]
            physical_history[row, 0] = np.log1p(t[row]-t[ids[j-1]]) if j else 0.
            # All lookback slices stop at j+1, inclusive of the current row.
            for wi, seconds in enumerate([30., 60.]):
                start = np.searchsorted(ts[:j+1], t[row]-seconds, side='left')
                past = ids[start:j+1]
                for k in range(3):
                    a = mean[past, 3+k]
                    probability_history[row, 6+wi*9+k*3:9+wi*9+k*3] = [a.mean(), a.max(), mean[row, 3+k]-a[0]]
            past = ids[np.searchsorted(ts[:j+1], t[row]-60., side='left'):j+1]
            for k in range(4):
                values = sensor_values[past, k]
                good = np.isfinite(values)
                if good.any():
                    a = values[good]
                    tt = t[past][good]
                    physical_history[row, 1+3*k:4+3*k] = [a.max()-a.min(),
                        (a[-1]-a[0])/max(tt[-1]-tt[0], 1.),
                        raw[row, k]-np.sign(a[0])*np.log1p(abs(a[0]))]
            physical_history[row, 13:15] = [np.log1p(j+1), np.log1p(t[row]-t[ids[0]])]
    cols.append(physical_history)
    names += ['observation_gap_log1p'] + [f'{s}_recent60_{v}' for s in SIGNALS[:4]
                                       for v in ['range', 'slope', 'log_difference']]
    names += ['observed_count_log1p', 'observed_elapsed_log1p']
    physical = np.column_stack(cols)
    phys_names = names.copy()
    odds = np.log(np.clip(p[:, :, 5], 1e-7, 1))-np.log(np.clip(p[:, :, 4], 1e-7, 1))
    probability = np.c_[mean[:, 3:6], odds.mean(0), odds.std(0),
                        (p.argmax(2) == 5).mean(0), probability_history]
    probability_names = ['B_p4', 'B_p5', 'B_p6', 'seed_logodds65_mean',
                         'seed_logodds65_std', 'seed_vote6']
    probability_names += ['past_peak_p4', 'past_peak_p5', 'past_peak_p6', 'observed6',
                          'last6_age_log1p', 'current6_run_log1p']
    probability_names += [f'p{k}_past{w}_{v}' for w in [30, 60]
                          for k in [4, 5, 6] for v in ['mean', 'max', 'difference']]
    return {'45': physical, '56': np.c_[physical, probability]}, {
        '45': phys_names, '56': phys_names+probability_names}


def fit_pair(x, y, base, groups, pair):
    source, target = PAIRS[pair]
    keep = np.isin(y, [source, target]) | (base == source)
    target_y = (y[keep] == target).astype(int)
    if len(np.unique(target_y)) != 2:
        return None
    # Equal weight for each experiment/target class; normalize for HGB's fixed L2.
    keys = pd.DataFrame({'experiment': groups[keep], 'target': target_y})
    sizes = keys.groupby(['experiment', 'target']).target.transform('size').to_numpy(float)
    weights = 1/sizes
    weights /= weights.mean()
    model = HistGradientBoostingClassifier(**PARAMS)
    model.fit(x[keep], target_y, sample_weight=weights)
    return model


def probability(model, x):
    return np.zeros(len(x)) if model is None else model.predict_proba(x)[:, 1]


def apply_pair(base, score, pair, threshold):
    source, target = PAIRS[pair]
    p = np.asarray(base, int).copy()
    p[(p == source) & (score >= threshold)] = target
    return p


def select_threshold(score, y, base, groups, pair):
    # Selection scores are group-cross-fitted within the outer training set.
    ids = list(pd.DataFrame({'group': groups}).groupby('group').indices.values())
    cb = np.stack([matrix(y[i], base[i]) for i in ids])
    choices = []
    for threshold in THRESHOLDS:
        p = apply_pair(base, score, pair, threshold)
        cp = np.stack([matrix(y[i], p[i]) for i in ids])
        delta = float(f1(cp.sum(0)).mean()-f1(cb.sum(0)).mean())
        per = f1(cp).mean(1)-f1(cb).mean(1)
        corrected = int(((base != y) & (p == y)).sum())
        harmed = int(((base == y) & (p != y)).sum())
        loo = [float(f1(cp.sum(0)-a).mean()-f1(cb.sum(0)-b).mean()) for a,b in zip(cp, cb)]
        passed = (delta > 0 and corrected >= max(3, 2*harmed)
                  and (per > 1e-12).sum() >= 3
                  and (per > 1e-12).sum() > (per < -1e-12).sum()
                  and min(loo) > 0)
        choices.append(dict(threshold=threshold, delta=delta, corrected=corrected,
                            harmed=harmed, improved=int((per > 1e-12).sum()),
                            worsened=int((per < -1e-12).sum()), loo_min=min(loo),
                            passed=bool(passed)))
    valid = [a for a in choices if a['passed']]
    selected = max(valid, key=lambda a: (a['delta'], a['threshold']))['threshold'] if valid else 1.01
    return selected, choices


def train(frame, probs, y, base):
    x, names = features(frame, probs)
    groups = frame.experiment_id.to_numpy()
    artifact = dict(models={}, thresholds={}, names=names,
                    fit_experiments=sorted(set(groups)), choices={}, params=PARAMS)
    folds = list(GroupKFold(3).split(frame, groups=groups))
    for pair in PAIRS:
        scores = np.zeros(len(frame))
        for a,b in folds:
            assert not set(groups[a]) & set(groups[b])
            model = fit_pair(x[pair][a], y[a], base[a], groups[a], pair)
            scores[b] = probability(model, x[pair][b])
        threshold, choices = select_threshold(scores, y, base, groups, pair)
        artifact['models'][pair] = fit_pair(x[pair], y, base, groups, pair)
        artifact['thresholds'][pair] = threshold
        artifact['choices'][pair] = choices
    return artifact


def infer(artifact, frame, probs, base, fixed_threshold=None):
    x, names = features(frame, probs)
    if names != artifact['names']:
        raise ValueError('Feature schema changed')
    outputs = {}
    for pair in PAIRS:
        score = probability(artifact['models'][pair], x[pair])
        threshold = artifact['thresholds'][pair] if fixed_threshold is None else fixed_threshold
        outputs[pair] = apply_pair(base, score, pair, threshold)
    # Both decisions are made against the ORIGINAL baseline. A new 4->5 never
    # cascades into 6 on the same observation.
    both = np.asarray(base, int).copy()
    both[np.asarray(base) == 4] = outputs['45'][np.asarray(base) == 4]
    both[np.asarray(base) == 5] = outputs['56'][np.asarray(base) == 5]
    outputs['both'] = both
    return outputs
