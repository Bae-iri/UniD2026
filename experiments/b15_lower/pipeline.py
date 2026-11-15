"""Frozen J096 + S3 recipe. Inference reads current and past observations only."""
import numpy as np
import pandas as pd
from core import upper, lower, precision

H, DROP = 300., 5.


def history(frame):
    t = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    if frame.experiment_id.isna().any() or not np.isfinite(t).all() or (t < 0).any():
        raise ValueError('Experiment IDs and finite nonnegative times are required')
    t1 = pd.to_numeric(frame.meas_t1, errors='coerce').to_numpy(float)
    total = pd.to_numeric(frame.meas_t1_plus_t2, errors='coerce').to_numpy(float)
    voltage = pd.to_numeric(frame.meas_v, errors='coerce').to_numpy(float)
    hot = np.zeros(len(frame), bool)
    peak, drop = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    vdrop = np.zeros(len(frame), bool)
    for ids in frame.groupby('experiment_id', sort=False).indices.values():
        ids = ids[np.argsort(t[ids], kind='stable')]
        if (np.diff(t[ids]) <= 0).any():
            raise ValueError('Duplicate observation time within an experiment')
        max_t1 = max_total = reference = np.nan
        dropped = False
        for j, i in enumerate(ids):
            if np.isfinite(t1[i]):
                max_t1 = t1[i] if not np.isfinite(max_t1) else max(max_t1, t1[i])
            if np.isfinite(total[i]):
                max_total = total[i] if not np.isfinite(max_total) else max(max_total, total[i])
            # The initial reference evolves as observations arrive; no backfill.
            if j < 5 and np.isfinite(voltage[i]):
                reference = voltage[i] if not np.isfinite(reference) else max(reference, voltage[i])
            if np.isfinite(reference) and np.isfinite(voltage[i]) and voltage[i] < .8*reference:
                dropped = True
            hot[i], peak[i], vdrop[i] = max_t1 >= 45, max_total, dropped
            if np.isfinite(total[i]):
                drop[i] = max_total-total[i]
    return hot, peak, drop, vdrop


def rule1(frame, raw):
    hot = history(frame)[0]
    out = np.asarray(raw, int).copy()
    out[(out == 1) & hot] = 2
    out[(out >= 2) & ~hot] = 1
    return out


def s3(frame, predictions, height=H, decline=DROP):
    _, peak, drop, vdrop = history(frame)
    p = np.asarray(predictions, int).copy()
    valid = np.isfinite(drop)
    down = (p == 6) & (peak >= height) & valid & (drop <= 0)
    up = np.isin(p, [4, 5]) & (peak >= height) & valid & (drop >= decline) & vdrop
    p[down], p[up] = 5, 6
    return p


def predict(artifact, frame, probabilities):
    base = rule1(frame, probabilities.mean(0).argmax(1)+1)
    x, names = upper.features(frame, probabilities)
    if names != artifact['heads']['names']:
        raise ValueError('Feature schema changed')
    # 4->5 is disabled in the submitted J096 recipe.
    both = upper.apply_pair(base, upper.probability(artifact['heads']['models']['56'], x['56']),
                            '56', artifact['heads']['thresholds']['56'])
    gate = artifact['gate']
    j = precision.apply(both, base, precision.score(gate['model'], x['56']), gate['threshold'])
    lo = lower.decide(artifact['low'], frame, probabilities, base, ['34', '43'])
    j[lo != base] = lo[lo != base]
    return dict(rule1=base, J096=j, S3=s3(frame, j))


def fit_frozen_recipe(frame, probabilities):
    """Refit the submitted structure with its ALREADY frozen thresholds.

    The caller supplies backbone probabilities cross-fitted strictly inside the
    outer training experiments. No new head or S3 threshold search is performed.
    This assesses the fixed algorithm, not the archived checkpoint's exact score.
    """
    y, groups = frame.label.to_numpy(int), frame.experiment_id.to_numpy()
    base = rule1(frame, probabilities.mean(0).argmax(1)+1)
    x, names = upper.features(frame, probabilities)
    lx, lnames = lower.features(frame, probabilities)
    return dict(
        heads=dict(models={'56': upper.fit_pair(x['56'], y, base, groups, '56')},
                   thresholds={'45': 1.01, '56': .7}, names=names),
        gate=dict(model=precision.fit(x['56'], y, base, groups), threshold=.7),
        low=dict(models={'34': None, '43': lower.fit(lx, y, base, groups, '43')},
                 thresholds={'34': 1.01, '43': .5}, names=lnames),
        fit_experiments=sorted(set(groups)))
