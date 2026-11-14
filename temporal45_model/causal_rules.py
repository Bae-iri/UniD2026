"""rules3 with an initial voltage reference that never reads ahead."""
import numpy as np
import pandas as pd

import final_rules3 as rules


def prepare(frame):
    d = frame.copy()
    if '_causal_position' in d:
        raise ValueError('Reserved column')
    d['_causal_position'] = np.arange(len(d))
    d = d.sort_values(['experiment_id', 'relative_time_s'], kind='stable').reset_index(drop=True)
    times = pd.to_numeric(d.relative_time_s, errors='raise').to_numpy(float)
    if not np.isfinite(times).all() or d.experiment_id.isna().any():
        raise ValueError('Invalid experiment or time')
    if d.duplicated(['experiment_id', 'relative_time_s']).any():
        raise ValueError('Duplicate observation time')
    # The production rules now implement the same strict current/past policy.
    # Reuse that implementation so the two inference paths cannot diverge.
    d = rules.prepare(d)
    return d.sort_values('_causal_position').drop(columns='_causal_position').reset_index(drop=True)


def predict_rules(frame, raw, thresholds):
    d = prepare(frame)
    t = thresholds['floor3']
    by_capacity = {float(k): v for k, v in t['by_capacity'].items()}
    floor = (by_capacity, t['fallback'])
    return {'rules3': rules.apply_rules(d, np.asarray(raw, int), floor)}, d
