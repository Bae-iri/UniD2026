import unittest
import numpy as np
import pandas as pd
import torch
import abc_fusion_experiment as a


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_static_columns_not_read(self):
        n=12
        frame=pd.DataFrame({name:np.arange(n,dtype=float) for name in a.NUMERIC})
        frame['label']=np.tile(np.arange(1,7),2)
        prep=a.SensorPreprocessor(frame)
        original=prep.transform(frame)
        altered=frame.assign(bat_class=999,bat_cc=float('nan'),soc_status=-1,tr_method='excluded')
        actual=prep.transform(altered)
        for x,y in zip(original,actual): np.testing.assert_array_equal(x,y)
        self.assertEqual(actual[0].shape,(n,12))

    def test_fixed_query_independent_of_current_state(self):
        torch.manual_seed(1)
        read=a.FusionReadout(16)
        tokens=torch.randn(3,8,16); valid=torch.ones(3,8,dtype=torch.bool)
        q=torch.randn(3,16)
        torch.testing.assert_close(read(tokens,q,valid,'C'),read(tokens,q+50,valid,'C'),rtol=0,atol=0)
        self.assertFalse(torch.allclose(read(tokens,q,valid,'B'),read(tokens,q+50,valid,'B')))
        read(tokens,q,valid,'C').sum().backward()
        self.assertIsNotNone(read.fixed_query.grad)

    def test_same_direct_paths_and_initial_weights(self):
        image=torch.rand(3,3,32,32); mask=torch.ones(3,1,32,32)
        q=torch.rand(3,12); gas=torch.stack([torch.rand(3,11),torch.zeros(3,11)],-1)
        a.common.seed_everything(42); initial=a.FusionModel('A',d=16).state_dict()
        direct=[]
        for mode in a.MODES:
            model=a.FusionModel(mode,d=16).eval(); model.load_state_dict(initial)
            values=[]
            hook=model.head.register_forward_pre_hook(lambda module,args:values.append(args[0].detach()))
            out=model(image,mask,q,gas)
            hook.remove()
            self.assertEqual(out.shape,(3,6))
            direct.append(values[0][:,:48])
            self.assertEqual(set(model.state_dict()),set(initial))
        for x in direct[1:]: torch.testing.assert_close(x,direct[0],rtol=0,atol=0)

    def test_all_missing_readouts_are_finite(self):
        r=a.FusionReadout(16)
        for mode in a.MODES:
            out=r(torch.randn(3,8,16),torch.randn(3,16),torch.zeros(3,8,dtype=torch.bool),mode)
            self.assertTrue(torch.isfinite(out).all())

    def test_submission_joins_ids_not_row_positions(self):
        sample=pd.DataFrame({'sample_id':['b','a'],'label':[1,1]})
        test=pd.DataFrame({'sample_id':['a','b']})
        prob=np.zeros((2,6)); prob[0,2]=1; prob[1,4]=1
        out=a.make_submission(sample,test,prob)
        self.assertEqual(out.sample_id.tolist(),['b','a'])
        self.assertEqual(out.label.tolist(),[5,3])
        with self.assertRaises(ValueError): a.make_submission(sample.iloc[:1],test,prob)


if __name__=='__main__': unittest.main(verbosity=2)
