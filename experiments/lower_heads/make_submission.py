"""제출: 최종 모델(B + t1 + 5→6 + 6→5, submission_rule1_56_65_past와 같은 헤드) + 개발 검증에서 채택된 하위 경계 보정.

하위 보정은 train 78실험·targeted_rule1 B OOF 확률로 학습한다(5→6·6→5 헤드와 같은 데이터·절차).
기준값은 학습 내부 3-fold로 자동 선택. Test는 추론만 하고 실험별·행 셔플·prefix 불변성을 검증한다.
  python experiments/lower_heads/make_submission.py
"""
import copy, hashlib, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
G = '/home/ubuntu/gahyun/experiments/rule1_precision65'
sys.path.insert(0, G+'/vendor'); sys.path.insert(0, G)
from experiments.b_cadence_robust import run as C
from experiments.lower_heads import model as L
import pipeline as PL

OUT = ROOT/'experiments/lower_heads/submission_v1'
KEPT = json.loads((ROOT/'experiments/lower_heads/eval_v1/selection.json').read_text())['kept']
bundle = joblib.load(G+'/artifacts/pipeline.joblib')
heads = copy.deepcopy(bundle['baseline_heads']); heads['thresholds']['45'] = np.inf


def run(art, f, p, pairs):
    base, _, upper, _ = PL.from_probabilities(f, p, heads, bundle['precision_head'])
    lower = L.decide(art, f, p, base, pairs)
    out = upper.copy(); m = lower != base; out[m] = lower[m]
    return out, base


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    tr = pd.read_csv(C.DATA/'train/train.csv'); y = tr.label.to_numpy(int)
    with np.load(ROOT/'targeted_rule1/b_oof_v1/b_oof.npz') as z:
        assert np.array_equal(z['sample_id'], tr.sample_id.to_numpy(str)); p = z['oof']
    art = L.train(tr, p, y, PL.rule1(tr, p.mean(0).argmax(1)+1))
    joblib.dump(art, OUT/'lower_heads.joblib')
    print('thresholds', art['thresholds'], 'kept by dev rule', KEPT)
    test = pd.read_csv(C.DATA/'test/test.csv'); assert 'label' not in test
    tp = C.b_probs('test', test.sample_id)
    full, base = run(art, test, tp, KEPT)
    checks = dict(per_experiment=True, shuffled=True, prefixes=0, prefix_mismatch=0); t = test.relative_time_s.to_numpy()
    for ids in test.groupby('experiment_id').indices.values():
        checks['per_experiment'] &= bool(np.array_equal(run(art, test.iloc[ids].reset_index(drop=True), tp[:, ids], KEPT)[0], full[ids]))
        ids = ids[np.argsort(t[ids], kind='stable')]
        for end in sorted(set(np.linspace(1, len(ids), 12).astype(int))):
            s = ids[:end]; checks['prefixes'] += 1
            checks['prefix_mismatch'] += int(not np.array_equal(run(art, test.iloc[s].reset_index(drop=True), tp[:, s], KEPT)[0], full[s]))
    perm = np.random.default_rng(0).permutation(len(test))
    checks['shuffled'] = bool(np.array_equal(run(art, test.iloc[perm].reset_index(drop=True), tp[:, perm], KEPT)[0], full[perm]))
    sample = pd.read_csv(C.DATA/'sample_submission.csv')
    sub = sample[['sample_id']].merge(pd.DataFrame({'sample_id': test.sample_id, 'label': full}), on='sample_id', how='left')
    assert len(sub) == len(sample) and sub.label.between(1, 6).all()
    path = OUT/'submission_rule1_56_65_lower_past.csv'; sub.to_csv(path, index=False)
    ref = test[['sample_id']].merge(pd.read_csv(ROOT/'final_submit/submission_rule1_56_65_past.csv'), on='sample_id').label.to_numpy()
    d = full != ref
    man = dict(kept=KEPT, thresholds=art['thresholds'], choices=art['choices'], checks=checks,
               sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               diff_vs_56_65=int(d.sum()), changes_vs_56_65={f'{a}->{b}': int(n) for (a, b), n in pd.DataFrame({'a': ref[d], 'b': full[d]}).value_counts().items()},
               diff_by_experiment=pd.Series(test.experiment_id[d]).str[:8].value_counts().to_dict(),
               changed_vs_base=int((full != base).sum()))
    (OUT/'manifest.json').write_text(json.dumps(man, indent=1, default=float))
    print(json.dumps({k: v for k, v in man.items() if k != 'choices'}, indent=1, default=float))
    for pair in L.PAIRS: print(pair, [(c['threshold'], round(c['delta']*100, 2), c['fixed'], c['harmed'], c['passed']) for c in art['choices'][pair]])


if __name__ == '__main__':
    main()
