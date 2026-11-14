import unittest
import numpy as np
import pandas as pd
from stage23_audit.analyze import features,rule1,latch


class CausalRuleTests(unittest.TestCase):
    def frame(self):
        d=pd.DataFrame({'experiment_id':['A']*6+['B']*3,
                        'relative_time_s':[0,5,10,15,20,25,0,5,10],
                        'meas_t1':[40,44,45,46,47,48,30,31,32]})
        for c in ['meas_t2','thermal_t','meas_p','meas_v','atmp_t']:d[c]=d.meas_t1
        d['p3']=.6
        return d

    def test_confirmation_delay_and_experiment_isolation(self):
        d=self.frame();p=np.array([3,3,3,2,2,2,2,2,2])
        actual=latch(d,p,10)
        np.testing.assert_array_equal(actual,[3,3,3,3,3,3,2,2,2])
        np.testing.assert_array_equal(p,[3,3,3,2,2,2,2,2,2])

    def test_one_spike_does_not_latch(self):
        d=self.frame();p=np.array([3,2,2,2,2,2,2,2,2])
        np.testing.assert_array_equal(latch(d,p,10),p)

    def test_large_gap_cannot_confirm(self):
        d=self.frame();d.loc[:2,'relative_time_s']=[0,5,30];d.loc[3:5,'relative_time_s']=[35,40,45]
        p=np.array([3,3,3,2,2,2,2,2,2])
        np.testing.assert_array_equal(latch(d,p,10),p)

    def test_features_and_repair_are_prefix_invariant(self):
        d=self.frame();prefix=d.iloc[:4].copy()
        a=features(d).iloc[:4];b=features(prefix)
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(np.isnan(a.since45.iloc[0]))
        self.assertEqual(a.since45.iloc[2],0)
        p=np.array([3,3,3,2,2,2,2,2,2])
        np.testing.assert_array_equal(latch(d,p,10)[:4],latch(prefix,p[:4],10))

    def test_rule1_does_not_reassign_unknown_temperature(self):
        d=self.frame();d.loc[0,'meas_t1']=np.nan;d=features(d)
        p=np.array([2,2,1,3,3,3,1,1,1])
        actual=rule1(d,p)
        self.assertEqual(actual[0],2)
        self.assertEqual(actual[1],1)
        self.assertEqual(actual[2],2)

    def test_duplicate_timestamps_rejected(self):
        d=self.frame();d.loc[1,'relative_time_s']=0
        with self.assertRaises(ValueError):features(d)


if __name__=='__main__':unittest.main()
