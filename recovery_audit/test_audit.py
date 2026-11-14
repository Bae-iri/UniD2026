import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from recovery_audit.audit import cm, duration_rule, macro


class AuditTests(unittest.TestCase):
    def test_rule_interruption_and_experiment_boundary(self):
        f = pd.DataFrame({'experiment_id': ['A', 'A', 'A', 'B'],
                          'relative_time_s': [0, 300, 600, 700]})
        p = np.array([4, 3, 4, 4])
        np.testing.assert_array_equal(duration_rule(f, p), [4, 3, 5, 4])
        np.testing.assert_array_equal(duration_rule(f, p, True), p)
        np.testing.assert_array_equal(p, [4, 3, 4, 4])

    def test_rule_threshold_and_unsorted_rows(self):
        f = pd.DataFrame({'experiment_id': ['A'] * 3, 'relative_time_s': [576, 0, 575]})
        p = np.array([4, 4, 4])
        np.testing.assert_array_equal(duration_rule(f, p), [5, 4, 4])

    def test_vectorized_macro_matches_sklearn_with_missing_classes(self):
        y = np.array([1, 1, 2, 4, 5, 5])
        p = np.array([1, 2, 2, 5, 5, 4])
        m = cm(y, p)
        expected = f1_score(y, p, labels=range(1, 7), average='macro')
        np.testing.assert_allclose(macro(np.stack([m, m])), [expected, expected])


if __name__ == '__main__':
    unittest.main()
