"""Independent reconstruction of saved decisions and past-only model inputs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from targeted_stage_model.model import PARAMS, PAIRS, features, infer, apply_pair, probability
from targeted_stage_model.run import DATA, OUTER, SEEDS, NESTED
from temporal45_model.causal_rules import predict_rules
from four_rule_cv.run import fit_rules


def verify(run):
    assert json.loads((run/'status.json').read_text())['complete']
    protocol = json.loads((run/'protocol.json').read_text())
    for path,digest in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert json.loads((NESTED/'audit_checks.json').read_text())['passed']
    tr = pd.read_csv(DATA/'train/train.csv')
    saved = pd.read_csv(run/'nested_oof/predictions.csv')
    np.testing.assert_array_equal(tr.sample_id, saved.sample_id)
    np.testing.assert_array_equal(tr.label, saved.label)
    prefix_checks = 0
    for fold in range(5):
        held = np.flatnonzero(saved.fold.to_numpy() == fold)
        fit = np.flatnonzero(saved.fold.to_numpy() != fold)
        artifact = joblib.load(run/f'outer_{fold}/heads.joblib')
        assert set(artifact['fit_experiments']) == set(tr.experiment_id.iloc[fit])
        assert not set(artifact['fit_experiments']) & set(tr.experiment_id.iloc[held])
        for pair in PAIRS:
            m = artifact['models'][pair]
            assert m is not None and m.n_iter_ == PARAMS['max_iter']
            for k,v in PARAMS.items():
                assert m.get_params()[k] == v
            for choice in artifact['choices'][pair]:
                expected = (choice['delta'] > 0 and choice['corrected'] >= max(3,2*choice['harmed'])
                            and choice['improved'] >= 3 and choice['improved'] > choice['worsened']
                            and choice['loo_min'] > 0)
                assert choice['passed'] == expected
            valid = [c for c in artifact['choices'][pair] if c['passed']]
            threshold = max(valid,key=lambda c:(c['delta'],c['threshold']))['threshold'] if valid else 1.01
            assert artifact['thresholds'][pair] == threshold
        f = tr.iloc[held].reset_index(drop=True)
        p = np.stack([np.load(OUTER/f'fold_{fold}/seed{s}_probs.npz')['holdout'] for s in SEEDS])
        thresholds = fit_rules(tr.iloc[fit])
        base = predict_rules(f,p.mean(0).argmax(1)+1,thresholds)[0]['rules3']
        np.testing.assert_array_equal(base,saved.rules3.to_numpy()[held])
        selected = infer(artifact,f,p,base)
        raw = infer(artifact,f,p,base,fixed_threshold=.5)
        for pair in ['45','56','both']:
            np.testing.assert_array_equal(selected[pair],saved['selected_'+pair].to_numpy()[held])
            np.testing.assert_array_equal(raw[pair],saved['fixed05_'+pair].to_numpy()[held])
        full_x, names = features(f,p)
        for ids in f.groupby('experiment_id').indices.values():
            ids = ids[np.argsort(f.relative_time_s.to_numpy()[ids],kind='stable')]
            for end in sorted(set([1,5,max(1,len(ids)//2),len(ids)])):
                if end > len(ids):continue
                subset = ids[:end]
                partial = f.iloc[subset].reset_index(drop=True)
                pp = p[:,subset]
                px, partial_names = features(partial,pp)
                assert partial_names == names
                pb = predict_rules(partial,pp.mean(0).argmax(1)+1,thresholds)[0]['rules3']
                np.testing.assert_array_equal(pb,base[subset])
                for pair in PAIRS:
                    np.testing.assert_allclose(px[pair],full_x[pair][subset],equal_nan=True)
                    ps = probability(artifact['models'][pair],px[pair])
                    for threshold, expected in [(artifact['thresholds'][pair],selected[pair]),(.5,raw[pair])]:
                        np.testing.assert_array_equal(apply_pair(pb,ps,pair,threshold),expected[subset])
                prefix_checks += 1
    for folder in [run/'nested_oof']+sorted(run.glob('validation_*')):
        s=pd.read_csv(folder/'summary.csv').set_index('method')
        p=pd.read_csv(folder/'predictions.csv')
        for method in s.index:
            score=f1_score(p.label,p[method],labels=range(1,7),average='macro',zero_division=0)
            assert abs(score-s.loc[method,'macro_f1']) < 1e-12
            if method != 'rules3':
                changed=p[method]!=p.rules3
                allowed=((p.rules3==4)&(p[method]==5))|((p.rules3==5)&(p[method]==6))
                assert (~changed|allowed).all()
    decision=json.loads((run/'decision.json').read_text())
    deployed=joblib.load(run/'deployment_heads.joblib')
    assert deployed['accepted']==decision['accepted']
    for pair in PAIRS:
        if not decision['accepted'][pair]:assert deployed['thresholds'][pair]==1.01
    result=dict(passed=True,outer_experiments=tr.experiment_id.nunique(),heads_checked=10,
                actual_experiment_prefix_checks=prefix_checks,test_access=False,
                checks=['Source hashes unchanged','Nested base OOF audit passed',
                        'All held-out experiments excluded from specialist training',
                        'Model capacities and inner threshold decisions reproduced',
                        'All selected and default outer predictions reproduced',
                        'Both heads inputs and decisions unchanged for every checked prefix',
                        'No same-row cascade or changes to other baseline stages',
                        'OOF and all validation Macro F1 independently reproduced',
                        'Deployment flags reconcile with acceptance decisions'])
    (run/'audit_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    with threadpool_limits(limits=2):verify(parser.parse_args().run)
