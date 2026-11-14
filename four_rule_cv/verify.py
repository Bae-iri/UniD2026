"""Independent checks on saved fold separation, checkpoint selection and OOF metrics."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from four_rule_cv.run import abc,fit_rules,predict_rules,NAMES


def canonical(value):return json.loads(json.dumps(value))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args();r=args.run
    config=json.loads((r/'config.json').read_text());data=Path(config['data_root']);tr=pd.read_csv(data/'train/train.csv')
    if not json.loads((r/'status.json').read_text()).get('complete'):raise RuntimeError('Training incomplete')
    oof=pd.read_csv(r/'train_oof/predictions.csv');np.testing.assert_array_equal(oof.sample_id,tr.sample_id)
    probs=np.load(r/'probs.npz');seen=set();epochs=0;models=0
    for fold in range(config['folds']):
        folder=r/f'fold_{fold}';split=pd.read_csv(folder/'split_manifest.csv')
        assert split.sample_id.is_unique and set(split.sample_id)==set(tr.sample_id)
        group_sets=[set(split.loc[split.role==role,'experiment_id']) for role in ['fit','selection','holdout']]
        assert not any(group_sets[i]&group_sets[j] for i in range(3) for j in range(i+1,3))
        assert not seen&group_sets[2];seen.update(group_sets[2])
        f=tr[tr.sample_id.isin(set(split.loc[split.role=='fit','sample_id']))]
        expected=abc.SensorPreprocessor(f).metadata();actual=json.loads((folder/'preprocessing.json').read_text())
        for key in ['median','mean','std','class_weights']:np.testing.assert_allclose(expected[key],actual[key],rtol=1e-12,atol=1e-12)
        outer=tr[tr.experiment_id.isin(group_sets[0]|group_sets[1])]
        threshold=fit_rules(outer)
        assert canonical(threshold)==json.loads((folder/'thresholds.json').read_text())
        ids=np.flatnonzero(oof.fold.to_numpy()==fold)
        assert set(tr.experiment_id.iloc[ids])==group_sets[2]
        np.testing.assert_array_equal(tr.sample_id.iloc[ids],split.loc[split.role=='holdout','sample_id'])
        avg=np.zeros((len(ids),6))
        for seed in config['seeds']:
            history=pd.read_csv(folder/f'seed{seed}_history.csv');assert len(history)==config['epochs'];epochs+=len(history)
            ck=torch.load(folder/f'seed{seed}.pt',map_location='cpu',weights_only=True)
            best=history.loc[history.selection_f1.idxmax()]
            assert ck['epoch']==int(best.epoch) and abs(ck['inner_f1']-best.selection_f1)<1e-12
            assert all(torch.isfinite(v).all() for v in ck['state'].values())
            z=np.load(folder/f'seed{seed}_probs.npz');p=z['holdout']
            assert p.shape==(len(ids),6) and np.isfinite(p).all() and (p>=0).all()
            np.testing.assert_allclose(p.sum(1),1,atol=1e-5)
            avg+=p/len(config['seeds']);models+=1
        np.testing.assert_allclose(avg,probs['train_oof'][ids],atol=1e-12)
        pred,_=predict_rules(tr.iloc[ids].reset_index(drop=True),avg.argmax(1)+1,threshold)
        for name in NAMES:np.testing.assert_array_equal(pred[name],oof[name].to_numpy()[ids])
    assert seen==set(tr.experiment_id)
    summary=pd.read_csv(r/'train_oof/summary.csv')
    for name in NAMES:
        score=f1_score(oof.label,oof[name],labels=range(1,7),average='macro',zero_division=0)
        assert abs(score-summary.loc[summary.method==name,'macro_f1'].iloc[0])<1e-12
    report=dict(passed=True,models_checked=models,epochs_checked=epochs,experiments_checked=len(seen),
        checks=['No overlap of fit/selection/holdout experiments','All official Train rows covered once in OOF',
                'Scalers reconstructed from inner fit only','All rule thresholds reconstructed from outer train only',
                'Best checkpoint reconstructed from inner selection history','All model weights finite',
                'All OOF probabilities and four rule predictions reproduced','Pooled Macro F1 matches sklearn'],
        test_access=False)
    (r/'audit_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
