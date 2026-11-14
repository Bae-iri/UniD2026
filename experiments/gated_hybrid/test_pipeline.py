import unittest
import numpy as np
import pandas as pd
import torch
from features import CURRENT,temporal_features,Normalizer
from model import GatedHybrid,ImageHead,hierarchy_prob,relation_loss
from train import make_submission

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def frame(self):
        f=pd.DataFrame({c:np.arange(8,dtype=float) for c in CURRENT})
        f['relative_time_s']=[0,15,30,60,0,15,30,60]; f['experiment_id']=['a']*4+['b']*4
        return f

    def test_history_is_causal_grouped_and_order_invariant(self):
        f=self.frame(); x=temporal_features(f)
        changed=f.copy(); changed.loc[3,'meas_v']=999; changed.loc[4:,'meas_v']=999
        np.testing.assert_equal(temporal_features(changed).iloc[:3].to_numpy(),x.iloc[:3].to_numpy())
        f=f.sample(frac=1,random_state=4)
        np.testing.assert_allclose(temporal_features(f).to_numpy(),x.loc[f.index].to_numpy(),equal_nan=True)
        self.assertAlmostEqual(x.loc[2,'meas_v_ols_slope_30s'],1/15)
        self.assertEqual(x.loc[4,'history_count_30s'],1)
        self.assertTrue(np.isnan(x.loc[4,'previous_gap_30s']))
        pd.testing.assert_frame_equal(temporal_features(f.assign(label=6,bat_class=999,bat_cc=999,soc_status=999,tr_method=999)),temporal_features(f))

    def test_acceleration_and_normalization(self):
        f=self.frame().iloc[:4].copy(); f['relative_time_s']=[0,30,60,120]
        f['meas_t1']=.5*f.relative_time_s**2
        x=temporal_features(f)
        self.assertAlmostEqual(x.loc[2,'meas_t1_acceleration_30s'],1)
        self.assertAlmostEqual(x.loc[3,'meas_t1_acceleration_60s'],1)
        normalizer=Normalizer().fit(x.to_numpy()); state=normalizer.state()
        self.assertTrue(np.isfinite(normalizer.transform(x)).all())
        normalizer.transform(x*999); self.assertEqual(state,normalizer.state())

    def test_gate_is_exact_convex_probability_fusion(self):
        torch.manual_seed(4); m=GatedHybrid(8,16).eval()
        s=torch.randn(5,8); image=torch.randn(5,16); ps=torch.rand(5,6).softmax(1); pi=torch.rand(5,6).softmax(1)
        p,a=m(s,image,ps,pi)
        torch.testing.assert_close(p,.5*(ps+pi))
        self.assertTrue(((a>=0)&(a<=1)).all()); torch.testing.assert_close(p.sum(1),torch.ones(5))
        with torch.no_grad(): m.gate[-1].weight.normal_()
        p,a=m(s,image,ps,pi); torch.testing.assert_close(p,a*ps+(1-a)*pi)
        loss=relation_loss(p,torch.tensor([0,3,4,5,2]),torch.ones(6)); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertGreater(m.gate[-1].weight.grad.abs().sum(),0)

    def test_hierarchy_normalizes_and_pair_loss_handles_no45(self):
        p=hierarchy_prob(torch.rand(4,3).softmax(1),torch.rand(4,3).softmax(1),torch.rand(4,2).softmax(1))
        torch.testing.assert_close(p.sum(1),torch.ones(4))
        head=ImageHead(16); q=head(torch.randn(4,16)); loss=relation_loss(q,torch.tensor([0,1,2,5]),torch.ones(6)); loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_submission_matches_template_ids(self):
        sample=pd.DataFrame({'sample_id':['b','a'],'label':[1,1]}); test=pd.DataFrame({'sample_id':['a','b']})
        prob=np.eye(6)[[3,4]]
        out=make_submission(sample,test,prob)
        self.assertEqual(out.label.tolist(),[5,4])
        with self.assertRaises(ValueError): make_submission(sample.iloc[:1],test,prob)

if __name__=='__main__': unittest.main()
