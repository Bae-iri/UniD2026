"""Adversarial cases for causal rules, independent of competition scores."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pipeline import history, s3, predict_stages


def frame(total, voltage):
    return pd.DataFrame(dict(experiment_id=['a']*len(total), relative_time_s=np.arange(len(total))*5.,
                             meas_t1=[50.]*len(total), meas_t1_plus_t2=total, meas_v=voltage))


class CausalityTests(unittest.TestCase):
    def test_initial_reference_never_backfills(self):
        f = frame([310., 320., 330., 325., 340.], [1., 1., 1., 1., 4.])
        # A high fifth voltage cannot retrospectively declare the fourth low.
        np.testing.assert_array_equal(history(f)[3], [False]*5)
        np.testing.assert_array_equal(s3(f, [5]*5), [5]*5)

    def test_falling_temperature_needs_past_voltage_drop(self):
        f = frame([310., 320., 314., 310.], [4., 4., 4., 2.])
        np.testing.assert_array_equal(s3(f, [6, 6, 5, 5]), [5, 5, 5, 6])

    def test_missing_sum_disables_both_rules(self):
        f = frame([320., np.nan, 310.], [4., 2., 2.])
        np.testing.assert_array_equal(s3(f, [6, 6, 4]), [5, 6, 6])

    def test_shuffling_and_duplicate_time(self):
        f = frame([310., 320., 310.], [4., 4., 2.])
        p = np.array([6, 6, 5])
        perm = np.array([2, 0, 1])
        np.testing.assert_array_equal(s3(f.iloc[perm].reset_index(drop=True), p[perm]), s3(f, p)[perm])
        f.loc[2, 'relative_time_s'] = f.relative_time_s.iloc[1]
        with self.assertRaises(ValueError):
            s3(f, p)

    def test_stage6_never_overwrites_lower43(self):
        f = frame([320., 315.], [4., 2.])
        # Both base predictions are 4. The original lower head corrects the
        # second row to 3 even though the thermal rule could otherwise raise 4 to 6.
        p = np.zeros((3,2,6)); p[:,:,3] = 1.
        artifact = dict(heads=dict(names={},models={'56':None},thresholds={'56':.7}),
                        gate=dict(model=None,threshold=.7),low={})
        with patch('pipeline.upper.features', return_value=({'56':np.zeros((2,1))},{})), \
             patch('pipeline.upper.probability', return_value=np.zeros(2)), \
             patch('pipeline.precision.score', return_value=np.zeros(2)), \
             patch('pipeline.lower.decide', return_value=np.array([4,3])) as lower_decide:
            out = predict_stages(artifact,f,p)
        self.assertEqual(lower_decide.call_args.args[-1], ['43'])
        np.testing.assert_array_equal(out['S3'], [4,3])


if __name__ == '__main__':
    unittest.main()
