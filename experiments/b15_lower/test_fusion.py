"""Disagreement-only fusion and causal history regression checks."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from fusion import features,apply_gate,FEATURE_NAMES,Runtime


class IdentityScaler:
    def transform(self,x):return x


class CertainOriginal:
    def predict_proba(self,x):return np.tile([.001,.999],(len(x),1))


class FusionTests(unittest.TestCase):
    def test_agreements_missing_and_unknown_ranges_stay_current(self):
        gate=dict(model=CertainOriginal(),scaler=IdentityScaler(),feature_names=FEATURE_NAMES,
                  threshold=.95,minimum=np.zeros(10),maximum=np.ones(10))
        x=np.full((5,10),.5);x[2,0]=np.nan;x[3,0]=2.
        old=np.array([3,6,6,5,4]);new=np.array([3,5,5,6,6])
        np.testing.assert_array_equal(apply_gate(gate,x,old,new),[3,6,5,6,4])

    def test_disabled_gate_is_identity(self):
        np.testing.assert_array_equal(apply_gate({'model':None},np.zeros((2,10)),[6,5],[5,6]),[5,6])

    def test_future_peak_and_prediction_do_not_backfill(self):
        f=pd.DataFrame(dict(experiment_id=['x']*4,relative_time_s=[0.,5.,10.,15.],
            meas_t1=[50.]*4,meas_t1_plus_t2=[310.,320.,320.,400.],meas_v=[4.,4.,4.,2.]))
        p=np.full((3,4,6),1/6);old=np.array([6,6,5,6]);new=np.array([5,5,5,5])
        full=features(f,p,old,new)
        np.testing.assert_allclose(features(f.iloc[:3],p[:,:3],old[:3],new[:3]),full[:3],rtol=0,atol=0,equal_nan=True)
        perm=np.array([3,0,2,1])
        np.testing.assert_allclose(features(f.iloc[perm].reset_index(drop=True),p[:,perm],old[perm],new[perm]),full[perm],rtol=0,atol=0,equal_nan=True)

    def test_wrong_preprocessing_is_rejected_before_prediction(self):
        with tempfile.TemporaryDirectory(prefix='fusion_hash_') as folder:
            Path(folder,'preprocessing.json').write_text('{}')
            fake=SimpleNamespace(artifact={'backbone_sha256':{'preprocessing.json':'incorrect'}})
            with patch('infer.Runtime',return_value=fake):
                with self.assertRaisesRegex(ValueError,'differs from validated bundle'):
                    Runtime(folder,'unused.joblib')


if __name__=='__main__':unittest.main()
