import unittest
import numpy as np
import pandas as pd
from rules3_stage23.evaluate import preserve_constraints,soft23,variants,metrics


class PairRepairTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(experiment_id=['A']*6+['B']*2,relative_time_s=[0,5,10,15,20,25,0,5],
                                 t1max=[55,56,57,58,59,90,55,56],bat_cc=[4.8]*8,
                                 p2=[.3]*8,p3=[.6]*8,label=[3,3,3,3,3,3,1,6]))

    def test_early_and_late_classes_are_preserved(self):
        d=self.frame();base=np.array([3,3,3,2,2,2,1,6]);fl=({4.8:52.},52.)
        outputs=variants(d,base,fl,{}, {4.8:85.})
        for p in outputs.values():
            np.testing.assert_array_equal(p[[6,7]],base[[6,7]])
            self.assertTrue(np.isin(p[:6],[2,3]).all())

    def test_floor_remains_a_hard_constraint(self):
        d=self.frame();d.loc[0,'t1max']=49
        base=np.array([2,3,3,2,2,2,1,6]);candidate=np.repeat(3,8)
        p=preserve_constraints(d,candidate,base,({4.8:52.},52.))
        self.assertEqual(p[0],2)
        self.assertEqual(p[6],1)
        self.assertEqual(p[7],6)

    def test_combined_upper_rule_does_not_invent_history_evidence(self):
        d=self.frame();base=np.array([2,2,2,2,2,2,1,6]);fl=({4.8:52.},52.)
        result=variants(d,base,fl,{}, {4.8:85.})['latch10_plus_upper_q995']
        np.testing.assert_array_equal(result,[2,2,2,2,2,3,1,6])

    def test_pair_metric_directions(self):
        d=self.frame();base=np.array([3,3,3,2,2,2,1,6]);p=np.array([3,3,3,3,3,3,1,6])
        m=metrics(d,p,base)
        self.assertEqual(m['corrected'],3)
        self.assertEqual(m['harmed'],0)
        self.assertEqual(m['pair23_errors'],0)


if __name__=='__main__':unittest.main()
