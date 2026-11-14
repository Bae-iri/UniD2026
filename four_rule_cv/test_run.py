import unittest
import numpy as np
import pandas as pd
from four_rule_cv.run import fit_rules,predict_rules,prepare,NAMES


class FourRuleTests(unittest.TestCase):
    def train_frame(self):
        return pd.DataFrame(dict(sample_id=[f's{i}' for i in range(12)],experiment_id=['A']*6+['B']*6,
            relative_time_s=list(range(6))*2,meas_t1=[30,46,60,90,120,150]*2,
            thermal_t=[20,30,40,60,80,100,20,30,40,60,80,110],meas_v=[4,4,4,4,2,0]*2,
            bat_cc=[4.8]*12,label=[1,2,3,4,5,6]*2))

    def test_fitting_does_not_need_holdout(self):
        tr=self.train_frame();thr=fit_rules(tr)
        self.assertEqual(thr['experiments'],['A','B'])
        self.assertEqual(thr['floor6']['support'][4.8],2)
        self.assertNotIn('HOLDOUT',thr['experiments'])

    def test_disjoint_rules_and_union(self):
        tr=self.train_frame();thr=fit_rules(tr)
        d=tr.iloc[:3].copy();d.experiment_id='C';d.relative_time_s=[0,1,2]
        d.meas_t1=[60,70,80];d.thermal_t=[30,40,50];d.meas_v=[4,0,0]
        raw=np.array([2,6,2]);p,_=predict_rules(d,raw,thr)
        np.testing.assert_array_equal(p['plus_thermal_floor'],[2,5,2])
        np.testing.assert_array_equal(p['plus_t1_upper'],[3,6,3])
        np.testing.assert_array_equal(p['plus_both'],[3,5,3])
        self.assertEqual(list(p),NAMES)

    def test_output_matches_unsorted_input_ids(self):
        tr=self.train_frame();thr=fit_rules(tr);d=tr.sample(frac=1,random_state=7).reset_index(drop=True)
        outputs,state=predict_rules(d,d.label.to_numpy(),thr)
        np.testing.assert_array_equal(state.sample_id,d.sample_id)
        self.assertEqual(len(outputs['rules3']),len(d))

    def test_fallback_for_unseen_capacity(self):
        tr=self.train_frame();thr=fit_rules(tr);d=tr.iloc[-2:].copy();d.bat_cc=999
        out,_=predict_rules(d,np.array([5,6]),thr)
        self.assertTrue(all(np.isin(p,range(1,7)).all() for p in out.values()))

    def test_missing_temperature_does_not_force_demotion(self):
        tr=self.train_frame();thr=fit_rules(tr);d=tr.iloc[-1:].copy();d.meas_v=0;d.thermal_t=np.nan
        p,_=predict_rules(d,np.array([6]),thr)
        self.assertEqual(p['plus_thermal_floor'][0],p['rules3'][0])

if __name__=='__main__':unittest.main()
