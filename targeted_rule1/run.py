"""5→4·6→5 전용 보정 모델을 '원본 B + t1 45°C 규칙' 위에서 다시 학습·평가·제출.

기준 예측(base) = 원본 ABC B 3-seed 평균 argmax + t1 45°C 규칙(같은 실험의 현재까지 meas_t1 최고 ≥45 이면 ≥2, 아니면 1).
외부 규칙 모듈 없이 이 폴더의 코드와 ABC 원본 코드(abc_reference)만 사용한다.

평가 구조:
  - B 확률: targeted_rule1/b_oof.py로 새로 학습한 실험 단위 5-fold OOF(각 행은 그 실험을 보지 않은 B의 예측).
  - 보정 모델 성능: 같은 5-fold로 보정 모델을 나머지 fold에서 학습하고 해당 fold에서 평가.
    보정 기준(.5/.7/.9)은 학습 fold 안 실험 단위 3-fold로만 선택.
  - 채택 조건: 바깥 OOF F1 개선, 실험 bootstrap 95% 하한 > 0, 개선 실험 > 악화 실험, 한 실험 제외 최소 개선 > 0,
    validation(원본 B·fold 앙상블 B, 전체·E16 제외) 모두 비악화. validation은 모든 학습·선택 후 읽는다.
  - Test는 고정된 모델로 추론만 한다.

  python -u targeted_rule1/run.py --output targeted_rule1/run_v1
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from targeted_rule1.model import PAIRS, PARAMS, THRESHOLDS, train as train_heads, infer, matrix, f1

DATA = Path('/home/ubuntu/data/battery_data_light')
BASE = Path('/home/ubuntu/leewon/abc_runs/current_query_v1')        # 원본 ABC B (public 0.9121)
BOOF = ROOT/'targeted_rule1/b_oof_v1/b_oof.npz'                    # b_oof.py 결과(실험 단위 5-fold B OOF)
SEEDS = [42, 43, 44]
E16 = 'E_e5bcddc7d77f459ea541cf658ab4346e'                          # validation의 3→5 점프 실험(별도 집계용)
T1_STAGE2 = 45.0


def rule1(frame, pred):
    """t1 45°C 규칙. 같은 실험의 현재까지 관측만 사용(시간순 누적 최고). 입력 행 순서대로 반환."""
    t = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    v = pd.to_numeric(frame.meas_t1, errors='coerce').to_numpy(float)
    t1max = np.full(len(frame), np.nan)
    for ids in frame.groupby('experiment_id', sort=False).indices.values():
        ids = ids[np.argsort(t[ids], kind='stable')]
        best = np.nan
        for i in ids:
            if np.isfinite(v[i]) and not (best >= v[i]):
                best = v[i]
            t1max[i] = best
    p = np.asarray(pred, int).copy()
    hot = t1max >= T1_STAGE2
    p[(p == 1) & hot] = 2
    p[(p >= 2) & ~hot] = 1
    return p


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def macro(y, p):
    return float(f1(matrix(y, p)).mean())


def summarize(frame, base, outputs):
    y = frame.label.to_numpy(int)
    groups = list(frame.groupby('experiment_id', sort=True).indices.values())
    cb = np.stack([matrix(y[i], base[i]) for i in groups])
    rng = np.random.default_rng(20261115)
    draws = rng.integers(0, len(groups), (2000, len(groups)))
    rows = []
    for name, p in outputs.items():
        cp = np.stack([matrix(y[i], p[i]) for i in groups])
        boot = f1(cp[draws].sum(1)).mean(-1)-f1(cb[draws].sum(1)).mean(-1)
        per = f1(cp).mean(1)-f1(cb).mean(1)
        loo = [float(f1(cp.sum(0)-a).mean()-f1(cb.sum(0)-b).mean()) for a, b in zip(cp, cb)] if len(groups) > 1 else [0.]
        rows.append(dict(method=name, macro_f1=macro(y, p), delta_f1=macro(y, p)-macro(y, base),
                         corrected=int(((base != y) & (p == y)).sum()), harmed=int(((base == y) & (p != y)).sum()),
                         ci_lower=float(np.quantile(boot, .025)), ci_upper=float(np.quantile(boot, .975)),
                         better_experiments=int((per > 1e-12).sum()), worse_experiments=int((per < -1e-12).sum()),
                         loo_min=float(min(loo))))
    return rows


def b_probs(split, sample_ids):
    return np.stack([pd.read_csv(BASE/f'seed_{s}/B/{split}_predictions.csv').set_index('sample_id')
                     .loc[sample_ids, [f'p{i}' for i in range(1, 7)]].to_numpy() for s in SEEDS])


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', dict(base='original ABC B 3-seed mean + t1 45C rule (current/past only)',
        heads=PAIRS, params=PARAMS, thresholds=list(THRESHOLDS), test_access_during_training=False,
        validation_read_after_selection=True,
        source_sha256={str(p): sha(p) for p in [Path(__file__), ROOT/'targeted_rule1/model.py',
                       ROOT/'compare_abc_phase_history.py', DATA/'train/train.csv']}))
    frame = pd.read_csv(DATA/'train/train.csv')
    z = np.load(BOOF)
    assert np.array_equal(z['sample_id'], frame.sample_id.to_numpy(str))
    y = frame.label.to_numpy(int); fold_of = z['fold']; all_probs = z['oof']
    base_all = rule1(frame, all_probs.mean(0).argmax(1)+1)
    outs = {k: np.zeros(len(frame), int) for k in ['45', '56', 'both']}
    decisions = []
    for fold in range(5):
        held = np.flatnonzero(fold_of == fold); fitting = np.flatnonzero(fold_of != fold)
        assert not set(frame.experiment_id.iloc[held]) & set(frame.experiment_id.iloc[fitting])
        ffit = frame.iloc[fitting].reset_index(drop=True)
        art = train_heads(ffit, all_probs[:, fitting], y[fitting], base_all[fitting])
        joblib.dump(art, out/f'fold_{fold}_heads.joblib')
        hframe = frame.iloc[held].reset_index(drop=True)
        o = infer(art, hframe, all_probs[:, held], base_all[held])
        for k in outs: outs[k][held] = o[k]
        decisions.append(dict(fold=fold, thresholds=art['thresholds']))
        print('FOLD', fold, art['thresholds'], flush=True)
    write(out/'fold_decisions.json', decisions)
    oof_summary = summarize(frame, base_all, {f'selected_{k}': v for k, v in outs.items()})
    artifact = train_heads(frame, all_probs, y, base_all)                          # validation 읽기 전에 고정
    joblib.dump(artifact, out/'candidate_heads.joblib')
    val = pd.read_csv(DATA/'validation/validation.csv')
    keep = val.experiment_id.to_numpy() != E16
    vsum = {}
    for recipe in ['original_B', 'fold_ensemble']:
        if recipe == 'original_B':
            probs = b_probs('validation', val.sample_id)
        else:
            assert np.array_equal(z['val_sample_id'], val.sample_id.to_numpy(str)); probs = z['validation']
        vb = rule1(val, probs.mean(0).argmax(1)+1)
        o = infer(artifact, val, probs, vb)
        vsum[f'val_{recipe}'] = summarize(val, vb, {f'selected_{k}': v for k, v in o.items()})
        vsum[f'val_{recipe}_noE16'] = summarize(val.iloc[np.flatnonzero(keep)].reset_index(drop=True), vb[keep],
                                                {f'selected_{k}': v[keep] for k, v in o.items()})
    oofd = {r['method']: r for r in oof_summary}
    accepted = {}; checks = {}
    for pair in ['45', '56']:
        s = oofd[f'selected_{pair}']
        c = dict(positive_oof=s['delta_f1'] > 0, positive_ci_lower=s['ci_lower'] > 0,
                 more_improved=s['better_experiments'] > s['worse_experiments'], positive_loo=s['loo_min'] > 0,
                 validation_nonnegative=all(next(r for r in v if r['method'] == f'selected_{pair}')['delta_f1'] >= 0
                                            for v in vsum.values()))
        checks[pair] = {k: bool(v) for k, v in c.items()}; accepted[pair] = all(c.values())
    deployed = dict(artifact, thresholds={k: (v if accepted[k] else 1.01) for k, v in artifact['thresholds'].items()},
                    accepted=accepted)
    joblib.dump(deployed, out/'deployment_heads.joblib')
    write(out/'results.json', dict(oof=oof_summary, validation=vsum, accepted=accepted, checks=checks,
                                   final_thresholds=artifact['thresholds']))
    print(pd.DataFrame(oof_summary)[['method', 'macro_f1', 'delta_f1', 'corrected', 'harmed', 'ci_lower',
                                     'better_experiments', 'worse_experiments', 'loo_min']].round(4).to_string(index=False))
    for k, v in vsum.items():
        print(k, {r['method']: (round(r['delta_f1'], 4), r['corrected'], r['harmed']) for r in v})
    print('ACCEPTED', accepted, 'thresholds', artifact['thresholds'], flush=True)
    # ---- Test: 고정 모델로 추론만 ----
    test = pd.read_csv(DATA/'test/test.csv'); assert 'label' not in test
    tp = b_probs('test', test.sample_id)
    raw = tp.mean(0).argmax(1)+1
    assert np.array_equal(raw, pd.read_csv(BASE/'submission_B.csv').set_index('sample_id').label.loc[test.sample_id].to_numpy())
    tb = rule1(test, raw)

    def run_all(f, pp):
        return infer(deployed, f, pp, rule1(f, pp.mean(0).argmax(1)+1))

    full = run_all(test, tp)
    head = 'both' if all(accepted.values()) else ('56' if accepted['56'] else ('45' if accepted['45'] else None))
    checks_t = dict(per_experiment=True, shuffled=True, prefixes=0, prefix_mismatch=0)
    for ids in test.groupby('experiment_id').indices.values():
        o = run_all(test.iloc[ids].reset_index(drop=True), tp[:, ids])
        checks_t['per_experiment'] &= all(np.array_equal(o[k], full[k][ids]) for k in full)
        ids = ids[np.argsort(test.relative_time_s.to_numpy()[ids], kind='stable')]
        for end in sorted(set(np.linspace(1, len(ids), 12).astype(int))):
            s = ids[:end]; o = run_all(test.iloc[s].reset_index(drop=True), tp[:, s]); checks_t['prefixes'] += 1
            checks_t['prefix_mismatch'] += int(any(not np.array_equal(o[k], full[k][s]) for k in full))
    perm = np.random.default_rng(0).permutation(len(test))
    o = run_all(test.iloc[perm].reset_index(drop=True), tp[:, perm])
    checks_t['shuffled'] = all(np.array_equal(o[k], full[k][perm]) for k in full)
    sample = pd.read_csv(DATA/'sample_submission.csv')
    files = []
    for name, labels in [('base_rule1', tb)] + ([(f'targeted_{head}', full[head])] if head else []):
        sub = sample[['sample_id']].merge(pd.DataFrame({'sample_id': test.sample_id, 'label': labels}), on='sample_id', how='left')
        assert len(sub) == len(sample) and sub.label.between(1, 6).all() and (sub.sample_id.values == sample.sample_id.values).all()
        sub['label'] = sub.label.astype(int)
        path = out/f'submission_{name}.csv'; sub.to_csv(path, index=False)
        files.append(dict(file=str(path), changed_vs_base=int((sub.label.values != sample[['sample_id']].merge(
            pd.DataFrame({'sample_id': test.sample_id, 'label': tb}), on='sample_id').label.values).sum()), sha256=sha(path)))
    write(out/'test_manifest.json', dict(head=head, checks=checks_t, files=files))
    print('TEST', json.dumps(dict(head=head, checks=checks_t, files=files), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    with threadpool_limits(limits=4):
        main(ap.parse_args().output)
