"""Small, one-sided 4->5 gate. All selection stays inside training experiments."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from compare_abc_phase_history import causal_features
from four_rule_cv.run import matrix, f1

THRESHOLDS = (.90, .95, .99)
MODES = ('probability', 'history', 'history_time')


def features(frame, seed_probs, mode):
    if mode not in MODES:
        raise ValueError(mode)
    p = np.asarray(seed_probs, float)
    if p.shape != (3, len(frame), 6) or not np.isfinite(p).all():
        raise ValueError('Expected three aligned seed probability arrays')
    if (p < 0).any() or not np.allclose(p.sum(2), 1, atol=1e-5):
        raise ValueError('Invalid probability distribution')
    mean = p.mean(0)
    odds = np.log(np.clip(p[:, :, 4], 1e-7, 1)) - np.log(np.clip(p[:, :, 3], 1e-7, 1))
    cols = [mean[:, 3], mean[:, 4], odds.mean(0), odds.std(0),
            (p.argmax(2) == 4).mean(0)]
    names = ['B_p4', 'B_p5', 'seed_logodds54_mean', 'seed_logodds54_std', 'seed_vote5']
    if mode != 'probability':
        h = causal_features(frame)
        for k, signal in enumerate(['meas_v', 'meas_t1', 'meas_t2', 'thermal_t']):
            j = 1 + 4*k
            cols.extend([h[:, j], h[:, j+1], h[:, j+3]])
            names.extend([signal+'_past_extreme', signal+'_difference', signal+'_valid'])
        if mode == 'history_time':
            cols.extend([h[:, 0]] + [h[:, j] for j in [3, 7, 11, 15]])
            names.extend(['elapsed_log1p'] + [f'extreme_age{k}_log1p' for k in range(4)])
    return np.column_stack(cols), names


def fit_gate(x, y, base, groups):
    gate = np.asarray(base) == 4
    target = (np.asarray(y)[gate] == 5).astype(int)
    if len(np.unique(target)) < 2:
        return None
    # Each experiment contributes total weight one; adjacent rows do not count
    # as thousands of independent experiments for regularization strength.
    g = np.asarray(groups)[gate]
    counts = pd.Series(g).value_counts()
    weights = np.array([1. / counts[z] for z in g])
    model = make_pipeline(SimpleImputer(strategy='median', add_indicator=True,
                                       keep_empty_features=True),
                          StandardScaler(),
                          LogisticRegression(C=1., max_iter=1000, random_state=42))
    model.fit(np.asarray(x)[gate], target, logisticregression__sample_weight=weights)
    return model


def predict_gate(model, x):
    return np.zeros(len(x)) if model is None else model.predict_proba(x)[:, 1]


def apply_gate(base, probability, threshold):
    p = np.asarray(base, int).copy()
    p[(p == 4) & (np.asarray(probability) >= threshold)] = 5
    return p


def select_threshold(x, y, base, groups):
    """Three prespecified thresholds, evaluated inside outer training only.

    Base OOF is already cross-fitted within outer training. Meta-CV here is a
    selector, not an independent full-system estimate. Only the untouched outer
    fold provides the full-system generalization estimate.
    """
    groups = np.asarray(groups)
    probability = np.zeros(len(y))
    for train, hold in GroupKFold(3).split(x, groups=groups):
        assert not set(groups[train]) & set(groups[hold])
        m = fit_gate(x[train], y[train], base[train], groups[train])
        probability[hold] = predict_gate(m, x[hold])
    group_ids = list(pd.DataFrame({'g': groups}).groupby('g').indices.values())
    cb = np.stack([matrix(y[i], base[i]) for i in group_ids])
    candidates = []
    for threshold in THRESHOLDS:
        p = apply_gate(base, probability, threshold)
        cp = np.stack([matrix(y[i], p[i]) for i in group_ids])
        corrected = int(((base != y) & (p == y)).sum())
        harmed = int(((base == y) & (p != y)).sum())
        delta = float(f1(cp.sum(0)).mean() - f1(cb.sum(0)).mean())
        per = f1(cp).mean(1) - f1(cb).mean(1)
        loo = [float(f1(cp.sum(0)-a).mean()-f1(cb.sum(0)-b).mean())
               for a, b in zip(cp, cb)]
        passed = (delta > 0 and corrected >= max(3, 2*harmed)
                  and int((per > 1e-12).sum()) >= 3
                  and int((per > 1e-12).sum()) > int((per < -1e-12).sum())
                  and min(loo) > 0)
        candidates.append(dict(threshold=threshold, delta=delta, corrected=corrected,
                               harmed=harmed, improved=int((per > 1e-12).sum()),
                               worsened=int((per < -1e-12).sum()),
                               loo_min=min(loo), passed=bool(passed)))
    acceptable = [c for c in candidates if c['passed']]
    selected = max(acceptable, key=lambda c: (c['delta'], c['threshold']))['threshold'] if acceptable else 1.01
    return selected, candidates


def fit_artifact(frame, probs, y, base, mode):
    x, names = features(frame, probs, mode)
    groups = frame.experiment_id.to_numpy()
    threshold, selection = select_threshold(x, np.asarray(y), np.asarray(base), groups)
    model = fit_gate(x, y, base, groups)
    return dict(model=model, mode=mode, feature_names=names, threshold=threshold,
                threshold_selection=selection, fit_experiments=sorted(set(groups)),
                correction='only baseline 4 -> 5', regularization='C=1; unit total weight per experiment')


def infer(artifact, frame, probs, base):
    x, names = features(frame, probs, artifact['mode'])
    if names != artifact['feature_names']:
        raise ValueError('Feature schema mismatch')
    return apply_gate(base, predict_gate(artifact['model'], x), artifact['threshold'])
