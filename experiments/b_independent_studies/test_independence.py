import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'common'))
import numpy as np
import pandas as pd
import torch
import abc_fusion_experiment as original
from experiment1_voltage.model import VoltageBoostB
from experiment1_voltage.event_memory import event_memory
from experiment2_gas_temperature.model import GasTemperatureB,all_observed_zero,CHEMICAL

class IndependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def inputs(self):
        return [torch.rand(2,3,32,32),torch.ones(2,1,32,32),torch.randn(2,12),torch.stack([torch.randn(2,11),torch.zeros(2,11)],-1)]

    def test_original_B_exact_for_voltage_zero_init_and_nonzero_gas(self):
        torch.manual_seed(42);b=original.FusionModel('B').eval();x=self.inputs();ref=b(*x)
        e1=VoltageBoostB().eval();e1.load_state_dict(b.state_dict(),strict=False)
        torch.testing.assert_close(e1(x[0],x[1],torch.cat([x[2],torch.randn(2,7)],1),x[3]),ref,atol=0,rtol=0)
        e2=GasTemperatureB().eval();e2.load_state_dict(b.state_dict(),strict=False)
        torch.testing.assert_close(e2(x[0],x[1],torch.cat([x[2],torch.zeros(2,1)],1),x[3]),ref,atol=0,rtol=0)
        e2.gas_gain=1
        torch.testing.assert_close(e2(x[0],x[1],torch.cat([x[2],torch.ones(2,1)],1),x[3]),ref,atol=0,rtol=0)

    def test_gas_zero_missing_pressure_independent_and_no_event_parameters(self):
        f=pd.DataFrame({c:[0.,0.] for c in CHEMICAL});f['meas_p']=[10,100]
        np.testing.assert_equal(all_observed_zero(f),[1,1]);f.loc[1,CHEMICAL[0]]=np.nan
        np.testing.assert_equal(all_observed_zero(f),[1,0])
        self.assertFalse(any('event' in k or 'voltage' in k for k in GasTemperatureB().state_dict()))
        self.assertFalse(any('temperature' in k or 'chemical' in k for k in VoltageBoostB().state_dict()))
        # Experiment2 rejects the experiment1 feature shape, preventing accidental stacking.
        x=self.inputs()
        with self.assertRaises(ValueError):GasTemperatureB()(x[0],x[1],torch.randn(2,19),x[3])

    def test_event_is_causal_and_retained(self):
        f=pd.DataFrame(dict(experiment_id=['a']*4,relative_time_s=[0,5,10,1900],meas_v=[3.3,3.2,.1,.1]))
        cfg=dict(drop_threshold_v=.5,rate_threshold_v_per_s=.05,max_gap_s=30)
        e=event_memory(f,cfg);self.assertEqual(e[-1,2],1890);self.assertEqual(e[-1,4],1)
        f.loc[3,'meas_v']=99;np.testing.assert_equal(event_memory(f,cfg)[:3],e[:3])

    def test_temperature_boost_only_affects_zero_gas_rows(self):
        torch.manual_seed(1);b=original.FusionModel('B').eval();m=GasTemperatureB().eval();m.load_state_dict(b.state_dict(),strict=False)
        with torch.no_grad():m.temperature[-1].bias.fill_(1)
        x=self.inputs();q=torch.cat([x[2],torch.tensor([[0.],[1.]])],1)
        before=b(*x);after=m(x[0],x[1],q,x[3])
        torch.testing.assert_close(before[0],after[0],rtol=0,atol=0)
        self.assertFalse(torch.allclose(before[1],after[1]))
        self.assertTrue(torch.isfinite(after).all())

if __name__=='__main__':unittest.main()
