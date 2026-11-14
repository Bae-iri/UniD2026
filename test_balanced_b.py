import unittest

import numpy as np
import pandas as pd
import torch

from balanced_b import sampling_weights


class BalancedSamplingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([(label, exp) for label in range(1, 7)
                                   for exp, count in [('a', 1), ('b', label + 3)]
                                   for _ in range(count)], columns=['label', 'experiment_id'])

    def test_stage_and_experiment_masses(self):
        f = self.frame.copy()
        f['p'] = sampling_weights(f, 'experiment_balanced')
        np.testing.assert_allclose(f.groupby('label').p.sum(), np.repeat(1 / 6, 6))
        np.testing.assert_allclose(f.groupby(['label', 'experiment_id']).p.sum(), np.repeat(1 / 12, 12))

    def test_longer_recording_does_not_gain_group_mass(self):
        extra = self.frame[(self.frame.label == 5) & (self.frame.experiment_id == 'b')]
        larger = pd.concat([self.frame, extra, extra], ignore_index=True)
        for frame in [self.frame, larger]:
            p = sampling_weights(frame, 'experiment_balanced')
            self.assertAlmostEqual(p[(frame.label == 5) & (frame.experiment_id == 'b')].sum(), 1 / 12)

    def test_controls_and_reproducibility(self):
        f = self.frame.copy()
        f['p'] = sampling_weights(f, 'stage_balanced')
        np.testing.assert_allclose(f.groupby('label').p.sum(), np.repeat(1 / 6, 6))
        np.testing.assert_allclose(sampling_weights(f, 'control'), np.repeat(1 / len(f), len(f)))
        w = torch.from_numpy(sampling_weights(f, 'experiment_balanced'))
        a = torch.multinomial(w, 200, True, generator=torch.Generator().manual_seed(42))
        b = torch.multinomial(w, 200, True, generator=torch.Generator().manual_seed(42))
        torch.testing.assert_close(a, b)

    def test_missing_stage_rejected(self):
        with self.assertRaises(ValueError):
            sampling_weights(self.frame[self.frame.label != 6], 'experiment_balanced')


if __name__ == '__main__':
    unittest.main()
