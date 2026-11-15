import unittest
import numpy as np,pandas as pd
from boost_model import SENSORS,sensor_features,image_grid,combine


class BoostChecks(unittest.TestCase):
    def test_causal_features_missing_anchor_and_shuffle(self):
        f=pd.DataFrame({s:[2.,3.,4.,100.] for s in SENSORS})
        f['meas_ch4']=[np.nan,np.nan,4.,100.]
        f['experiment_id']='x';f['relative_time_s']=[0.,5.,10.,15.]
        f['bat_cc']=4.8;f['bat_class']=1;f['soc_status']=1;f['tr_method']=1
        full=sensor_features(f)
        np.testing.assert_allclose(sensor_features(f.iloc[:3]),full[:3],rtol=0,atol=0,equal_nan=True)
        perm=[3,0,2,1]
        np.testing.assert_allclose(sensor_features(f.iloc[perm].reset_index(drop=True)),full[perm],rtol=0,atol=0,equal_nan=True)

    def test_existing_corrections_and_stage6_are_protected(self):
        stages=dict(precision65=np.array([4,6,5,1,3]),J096=np.array([3,6,5,1,3]),S3=np.array([3,5,6,1,3]))
        p=np.full((3,5,6),1/6);expert=np.tile([.99,.003,.003,.004],(5,1))
        np.testing.assert_array_equal(combine(stages,p,expert,dict(alpha=1.)),[3,5,6,1,2])

    def test_masked_raw_image_grid(self):
        packed=np.zeros((4,128,128),np.uint8);packed[:3,:64]=255;packed[3,:64]=1
        grid=image_grid(packed).reshape(3,4,4)
        np.testing.assert_array_equal(grid[:,:2],1.)
        self.assertTrue(np.isnan(grid[:,2:]).all())


if __name__=='__main__':unittest.main()
