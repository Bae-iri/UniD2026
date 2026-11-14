import unittest
import numpy as np
import pandas as pd

from targeted_stage_model.model import features, infer, select_threshold


class ConstantModel:
    def predict_proba(self, x):
        return np.tile([.1, .9], (len(x), 1))


class TargetTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(experiment_id=['a']*6+['b']*6,
            relative_time_s=[0,5,10,15,20,25]*2,
            meas_v=[4,4,2,1,1,1]*2, meas_t1=[20,80,100,200,150,100]*2,
            meas_t2=[20,70,100,180,140,80]*2, thermal_t=[20,80,120,220,150,100]*2,
            meas_p=np.arange(12, dtype=float)))

    def probs(self, n):
        p = np.full((3,n,6), .02)
        for j in range(n):
            p[:,j,3+j%3] = .90
        return p

    def test_every_prefix_is_identical_and_labels_are_not_features(self):
        f = self.frame()
        p = self.probs(len(f))
        full, names = features(f, p)
        alternate, _ = features(f.assign(label=6), p)
        for pair in ['45','56']:
            self.assertEqual(len(names[pair]), full[pair].shape[1])
            np.testing.assert_allclose(full[pair], alternate[pair], equal_nan=True)
        for end in range(1,7):
            partial, _ = features(f.iloc[:end].copy(), p[:,:end])
            for pair in ['45','56']:
                np.testing.assert_allclose(full[pair][:end], partial[pair], equal_nan=True)
        changed = f.copy()
        changed.loc[changed.experiment_id == 'b', 'meas_t1'] = 99999
        other, _ = features(changed, p)
        for pair in ['45','56']:
            np.testing.assert_allclose(full[pair][:6], other[pair][:6], equal_nan=True)

    def test_one_sided_predictions_do_not_cascade(self):
        f = self.frame()
        p = self.probs(len(f))
        _, names = features(f, p)
        artifact = dict(names=names, models={'45':ConstantModel(),'56':ConstantModel()},
                        thresholds={'45':.7,'56':.7})
        base = np.tile(np.arange(1,7),2)
        out = infer(artifact,f,p,base)
        np.testing.assert_array_equal(out['both'], np.tile([1,2,3,5,6,6],2))
        artifact['thresholds'] = {'45':1.01,'56':1.01}
        np.testing.assert_array_equal(infer(artifact,f,p,base)['both'],base)

    def test_training_selector_rejects_harm(self):
        groups=np.repeat(np.arange(9),4)
        y=np.full(len(groups),4);base=y.copy()
        threshold, choices=select_threshold(np.ones(len(y)),y,base,groups,'45')
        self.assertEqual(threshold,1.01)
        self.assertFalse(any(c['passed'] for c in choices))

    def test_bad_times_and_distributions_rejected(self):
        f=self.frame();p=self.probs(len(f))
        f.loc[1,'relative_time_s']=0
        with self.assertRaises(ValueError):features(f,p)
        with self.assertRaises(ValueError):features(self.frame(),p*2)


if __name__ == '__main__':
    unittest.main()
