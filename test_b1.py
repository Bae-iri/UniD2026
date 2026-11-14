"""User-invoked unit checks. No training or external data access."""
import unittest
import numpy as np
import pandas as pd
import torch

import abc_fusion_experiment as abc
from b1_history import history_features, HistoryNormalizer
from b1_model import B1Model


def frame(times, experiment='e'):
    times = np.asarray(times, float)
    return pd.DataFrame(dict(experiment_id=experiment, relative_time_s=times,
                             meas_v=2*times+1, meas_t1=3*times+2,
                             meas_t2=-times+100, thermal_t=.5*times+20))


class HistoryChecks(unittest.TestCase):
    def test_real_units_all_targets_and_cadences(self):
        data = frame(np.arange(0, 41, 5))
        data.index = np.arange(len(data))+100
        for step in [5, 10, 15]:
            result = history_features(data, step)
            self.assertEqual(result.shape, (len(data), 11))
            np.testing.assert_allclose(result[6, :4], [2, 3, -1, .5])
            np.testing.assert_array_equal(result[6, 4:8], 1)
            np.testing.assert_allclose(result[6, 8:], [30, step, 1])
            np.testing.assert_array_equal(result[0], 0)

    def test_irregular_times_and_short_history(self):
        result = history_features(frame([0, 7, 18, 30]), 5)
        np.testing.assert_allclose(result[-1, :4], [2, 3, -1, .5])
        np.testing.assert_allclose(result[-1, 8:], [30, 12, 1])
        np.testing.assert_allclose(result[1, 8:], [7, 7, 1])
        sparse = history_features(frame([0, 31]), 5)
        np.testing.assert_array_equal(sparse[-1], 0)

    def test_no_future_or_cross_experiment_leakage(self):
        original = frame([0,5,10,15,20,25,30,35])
        before = history_features(original, 5)
        altered = original.copy()
        altered.loc[7, ['meas_v','meas_t1','meas_t2','thermal_t']] = 1e9
        other = frame([0,5,10,15,20,25,30], 'other')
        other['meas_v'] = -999
        after = history_features(pd.concat([altered, other], ignore_index=True), 5)
        np.testing.assert_array_equal(before[:7], after[:7])
        # A row older than the trailing 30-second window cannot affect the last target.
        altered = original.copy()
        altered.loc[0,'meas_v'] = 1e9
        np.testing.assert_array_equal(before[7], history_features(altered,5)[7])

    def test_equal_time_peer_is_not_history(self):
        data = frame([0,5,5])
        data.loc[2,'meas_v'] = 1001
        result = history_features(data,5)
        self.assertEqual(result[1,0], 2)
        self.assertEqual(result[2,0], 200)

    def test_unsorted_input_output_alignment(self):
        ordered = frame([0,5,10,15,20,25,30])
        expected = history_features(ordered,5)
        permutation = [6,0,3,2,5,1,4]
        actual = history_features(ordered.iloc[permutation],5)
        np.testing.assert_array_equal(actual, expected[permutation])

    def test_missing_values_are_not_imputed_for_slopes(self):
        data = frame([0,5,10])
        data.loc[1,'meas_v'] = np.nan
        data['meas_t1'] = np.nan
        result = history_features(data,5)
        self.assertEqual(result[2,0], 2)
        self.assertEqual(result[2,4], 1)
        self.assertEqual(result[2,5], 0)
        self.assertEqual(result[2,1], 0)
        # Missing current value: actual past measurements can still define a valid slope.
        data.loc[1,'meas_v'], data.loc[2,'meas_v'] = 11, np.inf
        self.assertEqual(history_features(data,5)[2,0], 2)

    def test_normalization_is_frozen_and_invalid_slopes_stay_zero(self):
        train = frame([0,5,10,15,20,25,30])
        raw = [history_features(train,s) for s in [5,10,15]]
        norm = HistoryNormalizer.fit(raw)
        before = norm.metadata()
        val = frame([0,5,10,15,20,25,30])
        val['meas_v'] *= 1000
        transformed = norm.transform(history_features(val,5))
        self.assertEqual(before, norm.metadata())
        np.testing.assert_array_equal(transformed[0,:8], 0)
        self.assertEqual(transformed[0,9], 0)
        self.assertTrue(np.isfinite(transformed).all())
        restored = HistoryNormalizer(before['mean'], before['std'])
        np.testing.assert_array_equal(restored.transform(raw[0]), norm.transform(raw[0]))

    def test_bad_time_is_not_silently_dropped(self):
        data = frame([0,5,10])
        data.loc[1,'relative_time_s'] = np.nan
        with self.assertRaises(ValueError): history_features(data)


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_zero_initialized_b1_exactly_reproduces_b(self):
        torch.manual_seed(42)
        base = abc.FusionModel('B', d=16).eval()
        model = B1Model(d=16).eval()
        model.load_b_weights(base.state_dict())
        image, mask = torch.rand(3,3,32,32), torch.ones(3,1,32,32)
        q = torch.rand(3,12)
        gas = torch.stack([torch.rand(3,11), torch.zeros(3,11)],-1)
        with torch.no_grad():
            expected = base(image,mask,q,gas)
            for history in [torch.zeros(3,11), torch.randn(3,11)*100]:
                torch.testing.assert_close(model(image,mask,q,gas,history), expected, rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(model.history_mlp[-1].weight).item(), 0)
        self.assertEqual(torch.count_nonzero(model.history_mlp[-1].bias).item(), 0)

    def test_checkpoint_key_mismatch_rejected(self):
        state = abc.FusionModel('B',d=16).state_dict()
        state.pop('head.3.bias')
        with self.assertRaises(ValueError): B1Model(d=16).load_b_weights(state)


if __name__ == '__main__': unittest.main(verbosity=2)
