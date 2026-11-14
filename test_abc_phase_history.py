"""Synthetic-only causal features, matching initialization and full comparison tests."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import compare_abc_phase_history as phase
from abc_reference import abc_fusion_experiment as abc


class PhaseTests(unittest.TestCase):
    def test_causality_and_modes(self):
        f=pd.DataFrame({'experiment_id':['A','A','A','A','B'], 'relative_time_s':[20,0,10,40,0],
            'meas_v':[3.,4.,2.,1.,9.], 'meas_t1':[100.,50.,200.,999.,888.],
            'meas_t2':[80.,40.,160.,999.,888.], 'thermal_t':[70.,30.,150.,999.,888.]})
        h=phase.causal_features(f)
        np.testing.assert_allclose(h[0,5:9],[200,-100,np.log1p(10),1])
        np.testing.assert_allclose(h[0,1:5],[2,1,np.log1p(10),1])
        prefix=f[(f.experiment_id=='A')&(f.relative_time_s<=20)].reset_index(drop=True)
        np.testing.assert_allclose(phase.causal_features(prefix)[0],h[0])
        f.loc[3,phase.SIGNALS]=1e9
        np.testing.assert_allclose(phase.causal_features(f)[0],h[0])
        f.loc[f.index,'meas_t1']=np.nan
        raw=phase.causal_features(f);prep=phase.fit_extra(raw)
        z=phase.transform_extra(raw,prep)
        self.assertTrue(np.isfinite(z).all())
        x=torch.ones(2,17)
        self.assertEqual(phase.mode_features(x,'B').sum().item(),0)
        self.assertEqual(phase.mode_features(x,'B_time').sum().item(),2)
        self.assertEqual(phase.mode_features(x,'B_extrema').sum().item(),24)
        self.assertEqual(phase.mode_features(x,'B_extrema_time').sum().item(),34)
        with self.assertRaises(ValueError):phase.causal_features(pd.concat([f,f.iloc[[0]]],ignore_index=True))

    def test_initial_output_and_gradients(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        base=abc.FusionModel('B',d=16).eval()
        f=torch.randn(4,80);extra=torch.randn(4,17)
        state=None
        for mode in phase.MODES:
            torch.manual_seed(123);head=phase.PhaseHead(base.head,mode).eval()
            if state is None:state={k:v.clone() for k,v in head.state_dict().items()}
            else:self.assertTrue(all(torch.equal(v,state[k]) for k,v in head.state_dict().items()))
            torch.testing.assert_close(head(f,extra),base.head(f),rtol=1e-5,atol=1e-6)
            head(f,extra).sum().backward()
            grad=head.net[0].weight.grad[:,80:]
            self.assertEqual(bool(grad.abs().sum()>0),mode!='B')
            if mode=='B_time':self.assertEqual(grad[:,1:].abs().sum().item(),0)
            if mode=='B_extrema':self.assertEqual(grad[:,[0,3,7,11,15]].abs().sum().item(),0)

    def test_four_arm_training(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='phase-synthetic-') as tmp:
            root=Path(tmp);data=root/'data';run=root/'base';run.mkdir();rng=np.random.default_rng(42)
            for split,ngroups in [('train',10),('validation',2)]:
                folder=data/split;folder.mkdir(parents=True);n=6*ngroups
                f=pd.DataFrame({c:rng.uniform(1,2,n) for c in abc.NUMERIC})
                f['sample_id']=[f'{split}_{i}' for i in range(n)]
                f['experiment_id']=[f'{split}_exp_{i//6}' for i in range(n)]
                f['relative_time_s']=np.tile(np.arange(6)*5,ngroups)
                f['label']=np.tile(np.arange(1,7),ngroups)
                for c in ['bat_class','tr_method','soc_status']:f[c]=1
                f['bat_cc']=1.;f.to_csv(folder/f'{split}.csv',index=False)
            frames=abc.common.split_frames(data,20261114,.2)
            prep=abc.SensorPreprocessor(frames['fit'])
            (run/'preprocessing.json').write_text(json.dumps(prep.metadata()))
            cfg={'split_seed':20261114,'csv_sha256':{s:hashlib.sha256((data/s/f'{s}.csv').read_bytes()).hexdigest() for s in ['train','validation']}}
            (run/'config.json').write_text(json.dumps(cfg));manifest={}
            folder=run/'seed_42/B';folder.mkdir(parents=True)
            for split,f in frames.items():
                pixels=rng.integers(0,256,(len(f),4,128,128),dtype=np.uint8);pixels[:,3]=1
                p=run/f'{split}.npy';np.save(p,pixels);manifest[split]=str(p)
                if split in ['dev','validation']:f[['sample_id','label']].to_csv(folder/f'{split}_predictions.csv',index=False)
            (run/'image_cache_manifest.json').write_text(json.dumps(manifest))
            pd.concat([f[['sample_id','experiment_id']].assign(split=s) for s,f in frames.items()]).to_csv(run/'split_manifest.csv',index=False)
            checkpoint=folder/'best.pt';torch.save({'state':abc.FusionModel('B').state_dict()},checkpoint)
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            args=SimpleNamespace(base_run=run,data_dir=data,output=root/'out',epochs=1,seeds=[42],modes=phase.MODES,lr=3e-4,device='cpu')
            phase.run(args)
            summary=pd.read_csv(args.output/'summary.csv')
            self.assertEqual(summary['mode'].tolist(),phase.MODES)
            epoch0=[]
            for mode in phase.MODES:
                path=args.output/f'{mode}_seed42'
                epoch0.append(json.loads((path/'history.json').read_text())[0]['dev_macro_f1'])
                ck=torch.load(path/'best.pt',weights_only=False,map_location='cpu')
                restored=phase.PhaseHead(abc.FusionModel('B').head,mode)
                restored.load_state_dict(ck['head_state'])
                self.assertTrue((path/'validation_aggregates.json').exists())
            self.assertEqual(len(set(epoch0)),1)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),digest)
            with self.assertRaises(FileExistsError):phase.run(args)


if __name__=='__main__':unittest.main()
