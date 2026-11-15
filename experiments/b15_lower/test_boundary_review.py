"""Causal pressure history and correction precedence regression checks."""
import unittest
import numpy as np
import pandas as pd
from boundary_review import features,apply


class ConstantHead:
    def predict_proba(self,x):
        return np.tile([.01,.99],(len(x),1))


class BoundaryChecks(unittest.TestCase):
    def test_pressure_peak_never_backfills(self):
        f=pd.DataFrame(dict(experiment_id=['x']*3,relative_time_s=[0.,5.,10.],
            meas_v=[4.,4.,2.],meas_t1=[50.]*3,meas_t2=[60.]*3,thermal_t=[55.]*3,meas_p=[100.,50.,200.]))
        full=features(f)
        np.testing.assert_allclose(features(f.iloc[:2]),full[:2],rtol=0,atol=0,equal_nan=True)
        self.assertEqual(full[1,-3],.5)
        shuffled=f.iloc[[2,0,1]].reset_index(drop=True)
        np.testing.assert_allclose(features(shuffled),full[[2,0,1]],rtol=0,atol=0,equal_nan=True)

    def test_corrections_protect_3_and_6(self):
        p=np.array([1,2,3,4,5,6]);heads={'23':ConstantHead(),'45':ConstantHead()}
        np.testing.assert_array_equal(apply(p,heads,np.zeros((6,1)),'23+45'),[1,3,3,5,5,6])
        np.testing.assert_array_equal(apply(p,heads,np.zeros((6,1)),'none'),p)


if __name__=='__main__':unittest.main()
