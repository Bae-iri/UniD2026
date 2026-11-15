"""최종 모델(5→6 + 6→5) + 6→5 기준 후보 확장(0.5·0.6 추가, 학습 내부 자동 선택).

gahyun train.py와 같은 데이터·절차(train 78실험, targeted_rule1 B OOF, fold 헤드로 만든 OOF 기준 예측)로
6→5 기준만 다시 고른다. 4→5 헤드는 끈다. 6→5 게이트 모델 자체는 기준 후보와 무관하므로 기존 학습본과 같다(검증).
  python experiments/holdout_12_8/make_ext_submission.py
"""
import copy, hashlib, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
G = '/home/ubuntu/gahyun/experiments/rule1_precision65'
sys.path.insert(0, G+'/vendor'); sys.path.insert(0, G)
from experiments.b_cadence_robust import run as C
import pipeline as PL, precision as P

FRESH = ROOT/'targeted_rule1'; OUT = ROOT/'experiments/holdout_12_8/final_ext_v1'
EXT = [.5, .6, .7, .9, .95, .99]


def off45(h):
    h = copy.deepcopy(h); h['thresholds']['45'] = np.inf; return h


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    frame = pd.read_csv(C.DATA/'train/train.csv'); y = frame.label.to_numpy(int)
    with np.load(FRESH/'b_oof_v1/b_oof.npz') as z:
        assert np.array_equal(z['sample_id'], frame.sample_id.to_numpy(str)); p, folds = z['oof'], z['fold']
    baseline = np.zeros(len(frame), int)
    for k in range(5):
        held = np.flatnonzero(folds == k); hf = frame.iloc[held].reset_index(drop=True)
        heads = joblib.load(FRESH/f'run_v1/fold_{k}_heads.joblib'); assert not set(heads['fit_experiments']) & set(hf.experiment_id)
        baseline[held] = PL.from_probabilities(hf, p[:, held], off45(heads))[1]
    bundle = joblib.load(G+'/artifacts/pipeline.joblib'); heads = off45(bundle['baseline_heads'])
    eligible = PL.rule1(frame, p.mean(0).argmax(1)+1); x = PL.targeted.features(frame, p)[0]['56']
    P.THRESHOLDS = EXT
    gate = P.train(x, y, baseline, eligible, frame.experiment_id.to_numpy())
    same = np.allclose(P.score(gate['model'], x), P.score(bundle['precision_head']['model'], x))
    print('threshold', gate['threshold'], 'same gate model', same); print(pd.DataFrame(gate['choices']).round(4).to_string())
    test = pd.read_csv(C.DATA/'test/test.csv'); tp = C.b_probs('test', test.sample_id)
    run = lambda f, pp: PL.from_probabilities(f, pp, heads, gate)[2]
    full = run(test, tp); base = PL.rule1(test, tp.mean(0).argmax(1)+1)
    checks = dict(per_experiment=True, shuffled=True, prefixes=0, prefix_mismatch=0); t = test.relative_time_s.to_numpy()
    for ids in test.groupby('experiment_id').indices.values():
        checks['per_experiment'] &= bool(np.array_equal(run(test.iloc[ids].reset_index(drop=True), tp[:, ids]), full[ids]))
        ids = ids[np.argsort(t[ids], kind='stable')]
        for end in sorted(set(np.linspace(1, len(ids), 12).astype(int))):
            s = ids[:end]; checks['prefixes'] += 1
            checks['prefix_mismatch'] += int(not np.array_equal(run(test.iloc[s].reset_index(drop=True), tp[:, s]), full[s]))
    perm = np.random.default_rng(0).permutation(len(test))
    checks['shuffled'] = bool(np.array_equal(run(test.iloc[perm].reset_index(drop=True), tp[:, perm]), full[perm]))
    sample = pd.read_csv(C.DATA/'sample_submission.csv')
    sub = sample[['sample_id']].merge(pd.DataFrame({'sample_id': test.sample_id, 'label': full}), on='sample_id', how='left')
    assert len(sub) == len(sample) and sub.label.between(1, 6).all()
    path = OUT/'submission_rule1_56_65ext_past.csv'; sub.to_csv(path, index=False)
    s40 = test[['sample_id']].merge(pd.read_csv(ROOT/'final_submit/submission_rule1_targeted_precision65_past.csv'), on='sample_id').label.to_numpy()
    d = full != s40; m = full != base
    man = dict(threshold65=gate['threshold'], choices=gate['choices'], same_gate_model=bool(same), checks=checks,
               sha256=hashlib.sha256(path.read_bytes()).hexdigest(), changed_vs_base=int(m.sum()),
               changes={f'{a}->{b}': int(n) for (a, b), n in pd.DataFrame({'a': base[m], 'b': full[m]}).value_counts().items()},
               diff_vs_S40=int(d.sum()), diff_by_experiment=pd.Series(test.experiment_id[d]).str[:8].value_counts().to_dict(),
               diff_rows=pd.DataFrame({'exp': test.experiment_id[d].str[:8], 't': test.relative_time_s[d], 'S40': s40[d], 'new': full[d]}).to_dict('records'))
    (OUT/'manifest.json').write_text(json.dumps(man, indent=1, default=float)); print(json.dumps({k: v for k, v in man.items() if k != 'choices'}, indent=1, default=float))


if __name__ == '__main__':
    main()
