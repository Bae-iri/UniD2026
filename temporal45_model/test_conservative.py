import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from temporal45_model import conservative as gate
from temporal45_model.causal_rules import prepare as causal_prepare, predict_rules


class GateTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(experiment_id=['a']*3+['b']*3,
                                 relative_time_s=[0, 5, 10]*2,
                                 meas_v=[4, 2, 3, 4, 3, 2],
                                 meas_t1=[20, 90, 50, 20, 30, 40],
                                 meas_t2=[20, 80, 40, 20, 30, 40],
                                 thermal_t=[20, 100, 50, 20, 30, 40]))

    def test_future_and_labels_do_not_change_earlier_features(self):
        f = self.frame()
        p = np.full((3, len(f), 6), 1/6)
        a, names = gate.features(f, p, 'history_time')
        future = f.iloc[[0]].assign(relative_time_s=1000, meas_t1=9999)
        extended = pd.concat([f, future], ignore_index=True).assign(label=6)
        b, other = gate.features(extended, np.full((3, len(extended), 6), 1/6), 'history_time')
        np.testing.assert_allclose(a, b[:len(f)], equal_nan=True)
        self.assertEqual(names, other)
        # Different experiment cannot change a's extrema.
        changed = f.copy()
        changed.loc[changed.experiment_id == 'b', 'meas_t1'] = 9999
        c, _ = gate.features(changed, p, 'history_time')
        np.testing.assert_allclose(a[:3], c[:3], equal_nan=True)

    def test_history_without_time_is_time_scale_invariant(self):
        f = self.frame()
        p = np.full((3, len(f), 6), 1/6)
        a, _ = gate.features(f, p, 'history')
        f.relative_time_s *= 13
        b, _ = gate.features(f, p, 'history')
        np.testing.assert_allclose(a, b, equal_nan=True)

    def test_only_four_can_be_promoted_and_disabled_gate_is_identity(self):
        base = np.arange(1, 7)
        np.testing.assert_array_equal(gate.apply_gate(base, np.ones(6), .9), [1, 2, 3, 5, 5, 6])
        np.testing.assert_array_equal(gate.apply_gate(base, np.ones(6), 1.01), base)
        np.testing.assert_array_equal(base, np.arange(1, 7))

    def test_selection_separates_groups_and_abstains_without_evidence(self):
        groups = np.repeat(np.arange(9), 4)
        x = np.column_stack([groups, np.arange(len(groups))])
        y = np.full(len(groups), 4)
        base = y.copy()
        seen = []
        original = gate.fit_gate

        def fit(a, yy, bb, gg):
            seen.append(set(gg))
            return original(a, yy, bb, gg)

        def predict(model, a):
            self.assertFalse(seen[-1] & set(a[:, 0]))
            return np.zeros(len(a))

        with patch.object(gate, 'fit_gate', side_effect=fit), patch.object(gate, 'predict_gate', side_effect=predict):
            threshold, candidates = gate.select_threshold(x, y, base, groups)
        self.assertEqual(threshold, 1.01)
        self.assertFalse(any(c['passed'] for c in candidates))

    def test_duplicate_times_rejected(self):
        f = self.frame()
        f.loc[1, 'relative_time_s'] = 0
        with self.assertRaises(ValueError):
            gate.features(f, np.full((3, len(f), 6), 1/6), 'history')

    def test_voltage_reference_and_whole_pipeline_never_read_future(self):
        # A future high voltage must not make an earlier row appear to have
        # already dropped below 80% of that future value.
        f = pd.DataFrame(dict(experiment_id=['a']*7, relative_time_s=np.arange(7)*5,
                              meas_v=[1., 10., 9., 8., 7., 6., 5.],
                              meas_t1=[100.]*7, meas_t2=[100.]*7,
                              thermal_t=[100.]*7, bat_cc=[100.]*7))
        p = np.tile(np.array([.01, .01, .01, .90, .06, .01]), (3, len(f), 1))
        thresholds = {'floor3': {'by_capacity': {100.: 50.}, 'fallback': 50.}}
        full_base = predict_rules(f, p.mean(0).argmax(1)+1, thresholds)[0]['rules3']
        self.assertEqual(full_base[0], 3)
        np.testing.assert_allclose(causal_prepare(f).voltage_reference, [1, 10, 10, 10, 10, 10, 10])
        x, names = gate.features(f, p, 'history_time')
        model = gate.fit_gate(x, np.array([4, 4, 4, 5, 5, 5, 5]),
                              np.full(7, 4), np.array(['train']*7))
        artifact = dict(model=model, mode='history_time', feature_names=names, threshold=.5)
        whole = gate.infer(artifact, f, p, full_base)
        for end in range(1, len(f)+1):
            prefix = f.iloc[:end].copy()
            prefix_p = p[:, :end]
            prefix_base = predict_rules(prefix, prefix_p.mean(0).argmax(1)+1, thresholds)[0]['rules3']
            np.testing.assert_array_equal(prefix_base, full_base[:end])
            np.testing.assert_array_equal(gate.infer(artifact, prefix, prefix_p, prefix_base), whole[:end])

    def test_deployment_path_aligns_ids_and_rebuilds_past_only_baseline(self):
        from temporal45_model.predict_conservative import predict
        f = self.frame().iloc[:3].copy().assign(sample_id=['a', 'b', 'c'],
                                               meas_t1=100., bat_cc=100.)
        probabilities = np.full((3, 6), .02)
        probabilities[np.arange(3), [5, 3, 4]] = .90
        p = np.stack([probabilities]*3)
        _, names = gate.features(f, p, 'history')
        artifact = dict(model=None, mode='history', threshold=1.01, feature_names=names)
        s = pd.DataFrame(probabilities, columns=[f'p{i}' for i in range(1, 7)])
        s['sample_id'] = f.sample_id.to_numpy()
        s = s.iloc[[2, 0, 1]]
        thresholds = {'floor3': {'by_capacity': {'100.0': 50.}, 'fallback': 50.}}
        result = predict(artifact, f, thresholds, [s]*3)
        np.testing.assert_array_equal(result.pred, [3, 4, 5])
        with self.assertRaises(ValueError):
            predict(artifact, f, thresholds, [s.iloc[:2]]*3)


if __name__ == '__main__':
    unittest.main()
