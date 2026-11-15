"""New 6->5 correction; trained independently of any old targeted56 artifact."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

PARAMS = dict(max_iter=120, max_leaf_nodes=5, max_depth=3, learning_rate=.05,
              min_samples_leaf=64, l2_regularization=5., early_stopping=False,
              random_state=42)
THRESHOLDS = [.7, .9, .95, .99]


def matrix(y, p):
    return np.bincount((np.asarray(y, int)-1)*6+np.asarray(p, int)-1, minlength=36).reshape(6, 6)


def macro(c):
    den = c.sum(-1)+c.sum(-2)
    return np.divide(2*np.diagonal(c, axis1=-2, axis2=-1), den,
                     out=np.zeros_like(den, dtype=float), where=den > 0).mean(-1)


def apply(base, eligible, score, threshold):
    result = np.asarray(base, int).copy()
    result[(np.asarray(eligible) == 6) & (result == 6) & (score >= threshold)] = 5
    return result


def score(model, x):
    return np.zeros(len(x)) if model is None else model.predict_proba(x)[:, 1]


def fit(x, y, eligible, groups):
    keep = np.isin(y, [5, 6]) | (eligible == 6)
    binary = (y[keep] == 5).astype(int)
    if len(np.unique(binary)) < 2:
        return None
    table = pd.DataFrame({'group': groups[keep], 'target': binary})
    w = 1/table.groupby(['group', 'target']).target.transform('size').to_numpy(float)
    model = HistGradientBoostingClassifier(**PARAMS)
    model.fit(x[keep], binary, sample_weight=w/w.mean())
    return model


def train(x, y, baseline, eligible, groups):
    s = np.zeros(len(y))
    for a, b in GroupKFold(3).split(x, groups=groups):
        assert not set(groups[a]) & set(groups[b])
        s[b] = score(fit(x[a], y[a], eligible[a], groups[a]), x[b])
    ids = list(pd.DataFrame({'group': groups}).groupby('group').indices.values())
    cb = np.stack([matrix(y[i], baseline[i]) for i in ids])
    choices = []
    for threshold in THRESHOLDS:
        p = apply(baseline, eligible, s, threshold)
        cp = np.stack([matrix(y[i], p[i]) for i in ids])
        delta = float(macro(cp.sum(0))-macro(cb.sum(0)))
        per = macro(cp)-macro(cb)
        fixed = int(((baseline != y) & (p == y)).sum())
        harmed = int(((baseline == y) & (p != y)).sum())
        loo = [float(macro(cp.sum(0)-a)-macro(cb.sum(0)-b)) for a, b in zip(cp, cb)]
        valid = delta > 0 and fixed >= max(3, 3*harmed) and (per > 1e-12).sum() >= 3 and (per > 1e-12).sum() > (per < -1e-12).sum() and min(loo) > 0
        choices.append(dict(threshold=threshold, delta=delta, corrected=fixed, harmed=harmed,
                            improved=int((per > 1e-12).sum()), worsened=int((per < -1e-12).sum()),
                            loo_min=min(loo), passed=bool(valid)))
    valid = [c for c in choices if c['passed']]
    threshold = max(valid, key=lambda c: (c['delta'], c['threshold']))['threshold'] if valid else 1.01
    return dict(model=fit(x, y, eligible, groups), threshold=threshold, choices=choices,
                fit_experiments=sorted(set(groups)), params=PARAMS)


def evaluate(frame, baseline, pred):
    y = frame.label.to_numpy(int)
    ids = list(frame.groupby('experiment_id').indices.values())
    a = np.stack([matrix(y[i], baseline[i]) for i in ids])
    b = np.stack([matrix(y[i], pred[i]) for i in ids])
    per = macro(b)-macro(a)
    draws = np.random.default_rng(20261115).integers(0, len(ids), (4000, len(ids)))
    boot = macro(b[draws].sum(1))-macro(a[draws].sum(1))
    loo = [float(macro(b.sum(0)-bb)-macro(a.sum(0)-aa)) for aa, bb in zip(a, b)]
    return dict(base_f1=float(macro(a.sum(0))), new_f1=float(macro(b.sum(0))),
                delta=float(macro(b.sum(0))-macro(a.sum(0))), ci_lower=float(np.quantile(boot, .025)),
                ci_upper=float(np.quantile(boot, .975)), corrected=int(((baseline != y) & (pred == y)).sum()),
                harmed=int(((baseline == y) & (pred != y)).sum()), changed=int((baseline != pred).sum()),
                better_experiments=int((per > 1e-12).sum()), worse_experiments=int((per < -1e-12).sum()),
                loo_min=min(loo))
