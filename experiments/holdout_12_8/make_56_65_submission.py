"""최종 단순화 모델: B + t1 45°C 규칙 + 5→6 보정 + 6→5 보정(기준 0.7, 학습 내부 자동 선택). 4→5 보정 제거.

학습된 헤드는 gahyun rule1_precision65/artifacts/pipeline.joblib(S40 제출 파일을 만든 것)을 읽기 전용으로 쓰고,
4→5 헤드의 기준만 무한대로 둬서 끈다. Test는 추론만 하고, 실험별·행 셔플·prefix 불변성을 검증한다.
  python experiments/holdout_12_8/make_56_65_submission.py
"""
import copy, hashlib, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
G = '/home/ubuntu/gahyun/experiments/rule1_precision65'
sys.path.insert(0, G+'/vendor'); sys.path.insert(0, G)
from experiments.b_cadence_robust import run as C
import pipeline as PL

OUT = ROOT/'experiments/holdout_12_8/final_56_65_v1'
bundle = joblib.load(G+'/artifacts/pipeline.joblib')
heads = copy.deepcopy(bundle['baseline_heads']); heads['thresholds']['45'] = np.inf


def run(f, p):
    return PL.from_probabilities(f, p, heads, bundle['precision_head'])[2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(C.DATA/'test/test.csv'); assert 'label' not in test
    tp = C.b_probs('test', test.sample_id)
    full = run(test, tp)
    base = PL.rule1(test, tp.mean(0).argmax(1)+1)
    checks = dict(per_experiment=True, shuffled=True, prefixes=0, prefix_mismatch=0)
    t = test.relative_time_s.to_numpy()
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
    path = OUT/'submission_rule1_56_65_past.csv'; sub.to_csv(path, index=False)
    s40 = test[['sample_id']].merge(pd.read_csv(ROOT/'final_submit/submission_rule1_targeted_precision65_past.csv'), on='sample_id').label.to_numpy()
    m = full != base
    man = dict(checks=checks, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), changed_vs_base=int(m.sum()),
               changes={f'{a}->{b}': int(n) for (a, b), n in pd.DataFrame({'a': base[m], 'b': full[m]}).value_counts().items()},
               diff_vs_S40=int((full != s40).sum()), thresholds=dict(t56=heads['thresholds']['56'], t65=bundle['precision_head']['threshold']))
    (OUT/'manifest.json').write_text(json.dumps(man, indent=2, default=str)); print(json.dumps(man, indent=1, default=str))


if __name__ == '__main__':
    main()
