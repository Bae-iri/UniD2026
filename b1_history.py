"""Causal history features shared verbatim by B1 training and inference."""
import numpy as np
import pandas as pd

SLOPE_COLUMNS = ['meas_v', 'meas_t1', 'meas_t2', 'thermal_t']
FEATURE_NAMES = ([f'{c}_slope_per_s' for c in SLOPE_COLUMNS]
                 + [f'{c}_slope_valid' for c in SLOPE_COLUMNS]
                 + ['history_span_s', 'previous_gap_s', 'previous_gap_valid'])
HISTORY_STEPS = (5, 10, 15)
WINDOW_SECONDS = 30.0


def history_features(frame, step_seconds=5):
    """Return one row per input target, in its ORIGINAL order, without imputation.

    Keep the target, then walk backward selecting the most recent observation at
    least step_seconds before the last selected time, bounded by [t-30,t).
    Equal-time peer rows and future rows never enter a target's history. Fit an
    OLS slope per channel on finite measurements at >=2 distinct selected times.
    A missing current value is allowed if >=2 real past observations suffice.
    history_span/gap describe selected timestamps, not a nominal 30-second span.
    """
    if step_seconds not in HISTORY_STEPS:
        raise ValueError(f'step_seconds must be one of {HISTORY_STEPS}')
    required = ['experiment_id', 'relative_time_s'] + SLOPE_COLUMNS
    if not set(required).issubset(frame.columns):
        raise ValueError(f'Missing history columns: {set(required)-set(frame.columns)}')
    if frame.experiment_id.isna().any():
        raise ValueError('experiment_id cannot be missing')
    times = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    if not np.isfinite(times).all():
        raise ValueError('All timestamps must be finite; target rows are never silently dropped')
    observed = frame[SLOPE_COLUMNS].astype(float).to_numpy()
    output = np.zeros((len(frame), len(FEATURE_NAMES)), dtype=np.float64)
    # groupby.indices contains positional indices even when the DataFrame index is arbitrary.
    for positions in frame.groupby('experiment_id', sort=False).indices.values():
        positions = np.asarray(positions)
        order = positions[np.argsort(times[positions], kind='stable')]
        group_times = times[order]
        for target in order:
            current = times[target]
            selected = [int(target)]
            cutoff = current - step_seconds
            while True:
                location = np.searchsorted(group_times, cutoff, side='right') - 1
                if location < 0 or group_times[location] < current - WINDOW_SECONDS:
                    break
                selected.append(int(order[location]))
                cutoff = group_times[location] - step_seconds
            selected = np.asarray(selected)
            relative = times[selected] - current
            if len(selected) > 1:
                output[target, 8:] = [-relative[-1], -relative[1], 1.0]
            for column in range(4):
                values = observed[selected, column]
                valid = np.isfinite(values)
                x, y = relative[valid], values[valid]
                if len(x) < 2:
                    continue
                centered = x - x.mean()
                denominator = np.dot(centered, centered)
                if denominator > 0:
                    output[target, column] = np.dot(centered, y-y.mean()) / denominator
                    output[target, column+4] = 1.0
    if not np.isfinite(output).all():
        raise ValueError('History features overflowed; check raw sensor units/values')
    return output


class HistoryNormalizer:
    """Fit on raw Train features for ALL augmentation cadences, never validation."""
    continuous = [0, 1, 2, 3, 8, 9]

    @classmethod
    def fit(cls, training_features):
        arrays = list(training_features)
        if not arrays:
            raise ValueError('No training history supplied')
        values = np.concatenate(arrays, axis=0)
        mean, std = np.zeros(11), np.ones(11)
        for column in cls.continuous:
            valid = values[:, column+4] > 0 if column < 4 else (
                values[:, 10] > 0 if column == 9 else np.ones(len(values), dtype=bool))
            if valid.any():
                mean[column] = values[valid, column].mean()
                std[column] = max(values[valid, column].std(), 1e-6)
        return cls(mean, std)

    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean, float), np.asarray(std, float)
        if self.mean.shape != (11,) or self.std.shape != (11,) or not (
            np.isfinite(self.mean).all() and np.isfinite(self.std).all() and (self.std > 0).all()
        ):
            raise ValueError('Invalid history normalization statistics')

    def transform(self, raw):
        if raw.ndim != 2 or raw.shape[1] != 11 or not np.isfinite(raw).all():
            raise ValueError('Expected finite N x 11 history features')
        result = raw.copy()
        for column in self.continuous:
            result[:, column] = (raw[:, column]-self.mean[column])/self.std[column]
        # Zero is a neutral placeholder only AFTER calculating slopes from real values.
        for column in range(4):
            result[raw[:, column+4] == 0, column] = 0
        result[raw[:, 10] == 0, 9] = 0
        return result.astype(np.float32)

    def metadata(self):
        return dict(feature_names=FEATURE_NAMES, mean=self.mean.tolist(), std=self.std.tolist(),
                    fit_source='official Train only, pooled 5/10/15-second history views',
                    slope_method='OLS on finite raw observations; units per second',
                    window_seconds=WINDOW_SECONDS, training_steps_seconds=list(HISTORY_STEPS))
