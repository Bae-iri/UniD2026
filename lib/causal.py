"""Causal feature implementation copied without training entry points."""
import numpy as np
import pandas as pd
SIGNALS = ["meas_v", "meas_t1", "meas_t2", "thermal_t"]

def causal_features(frame):
    """All observations up to current; never uses labels, future or duration."""
    if frame.experiment_id.isna().any():
        raise ValueError('Missing experiment_id')
    ts = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    if not np.isfinite(ts).all() or (ts < 0).any():
        raise ValueError('Time must be finite and nonnegative')
    raw = frame[SIGNALS].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    out = np.zeros((len(frame), 17), dtype=np.float64)
    out[:, 0] = np.log1p(ts)
    for indices in frame.groupby('experiment_id', sort=False).indices.values():
        ordered = indices[np.argsort(ts[indices], kind='stable')]
        if (np.diff(ts[ordered]) <= 0).any():
            raise ValueError('Duplicate timestamps within experiment')
        extrema = np.full(4, np.nan)
        when = np.zeros(4)
        for row in ordered:
            for k in range(4):
                value = raw[row, k]
                if np.isfinite(value):
                    if not np.isfinite(extrema[k]) or (value < extrema[k] if k == 0 else value > extrema[k]):
                        extrema[k], when[k] = value, ts[row]
                j = 1+4*k
                if np.isfinite(extrema[k]):
                    out[row, j] = extrema[k]
                    # Difference is unavailable when current observation is missing.
                    out[row, j+1] = value-extrema[k] if np.isfinite(value) else np.nan
                    out[row, j+2] = np.log1p(ts[row]-when[k])
                    out[row, j+3] = 1
                else:
                    out[row, j:j+3] = np.nan
    return out
