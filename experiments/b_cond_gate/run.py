"""원본 B(동결) + MLP456 + 배터리 조건 Feature Gating 실험 (V0~V3).

기준 예측 = 원본 ABC B 3-seed 평균 argmax + t1 45°C 규칙(현재까지 meas_t1 최고). 기준이 4·5·6인 행만
MLP456 변형이 4/5/6을 다시 결정한다. 원본 B 체크포인트·확률은 읽기만 한다.

평가:
  - train 78실험: targeted_rule1/b_oof_v1의 실험 단위 5-fold. B 확률도 같은 fold의 OOF.
    MLP456·조건 모듈은 나머지 fold로 학습하고 해당 fold를 예측한다. 하이퍼파라미터는 사전 고정(선택 없음).
  - validation 19실험: train 전체로 학습을 마친 뒤 1회 평가(원본 B 확률 / fold 앙상블 B 확률).
  - test: 고정 모델로 추론만. 실험별·셔플·prefix 불변성 검증.

  python -u experiments/b_cond_gate/run.py --output experiments/b_cond_gate/run_v1
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.b_cond_gate import model as M
from targeted_rule1.model import matrix, f1
from targeted_rule1.run import rule1

DATA = Path('/home/ubuntu/data/battery_data_light')
BASE = Path('/home/ubuntu/leewon/abc_runs/current_query_v1')
BOOF = ROOT/'targeted_rule1/b_oof_v1/b_oof.npz'
SEEDS = [42, 43, 44]
E16 = 'E_e5bcddc7d77f459ea541cf658ab4346e'
GROUPS = ['voltage', 'temp_t1', 'temp_t2', 'thermal_t', 'pressure', 'gas', 'B_prob', 'time']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def b_probs(split, sample_ids):
    return np.stack([pd.read_csv(BASE/f'seed_{s}/B/{split}_predictions.csv').set_index('sample_id')
                     .loc[sample_ids, [f'p{i}' for i in range(1, 7)]].to_numpy() for s in SEEDS])


def macro(y, p):
    return float(f1(matrix(y, p)).mean())


def combos(cond):
    keys = sorted({(int(a), round(float(b), 6)) for a, b in cond})
    return keys


KEYS = []   # train 조합 목록(main에서 설정)


def shift_condition(cond, shift):
    """각 행의 배터리 조합을 train의 6개 조합 목록에서 shift만큼 다른 조합으로 바꾼다(제거 실험)."""
    keys = KEYS
    idx = {k: i for i, k in enumerate(keys)}
    out = cond.copy()
    for r in range(len(cond)):
        k = keys[(idx[(int(cond[r, 0]), round(float(cond[r, 1]), 6))]+shift) % len(keys)]
        out[r] = k
    return out


class Fitted:
    """seed별 모델 + 표준화."""

    def __init__(self, x, b, cond, y, experiments):
        keep = np.isin(y, [4, 5, 6])
        self.prep = M.Prep().fit(x[keep], cond[keep], experiments[keep])
        xs, cs = self.prep.x(x[keep]), self.prep.c(cond[keep])
        self.models = [M.train_all(xs, b[keep], cs, y[keep]-4, experiments[keep], s) for s in SEEDS]

    def probs(self, name, x, b, cond, gate_fn=None):
        xs, cs = self.prep.x(x), self.prep.c(cond)
        out = []
        for m in self.models:
            go = None if gate_fn is None else gate_fn(M.gates(m, name, cs))
            out.append(M.predict(m, name, xs, b, cs, go))
        return np.mean(out, 0)

    def gate_values(self, name, cond):
        cs = self.prep.c(cond)
        return np.mean([M.gates(m, name, cs) for m in self.models], 0)


def all_predictions(fitted, x, b, cond, base, group_of):
    """변형별 예측 + 제거 실험 예측."""
    out = {v: M.apply456(base, fitted.probs(v, x, b, cond)) for v in M.VARIANTS}
    for v in ('V1', 'V2', 'V3'):
        for s in range(1, 6):
            out[f'{v}|cond_shift{s}'] = M.apply456(base, fitted.probs(v, x, b, shift_condition(cond, s)))
    for v in ('V2', 'V3'):
        for g in GROUPS:
            cols = np.asarray(group_of) == g
            def reset(gv, cols=cols):
                gv = gv.copy(); gv[:, cols] = 1; return gv
            out[f'{v}|reset_{g}'] = M.apply456(base, fitted.probs(v, x, b, cond, reset))
            def only(gv, cols=cols):
                gv = gv.copy(); gv[:, ~cols] = 1; return gv
            out[f'{v}|only_{g}'] = M.apply456(base, fitted.probs(v, x, b, cond, only))
    return out


def summarize(frame, base, preds):
    y = frame.label.to_numpy(int)
    exps = frame.experiment_id.to_numpy()
    ids = list(pd.Series(range(len(frame))).groupby(exps).indices.values())
    ref = preds['V0']
    c0 = np.stack([matrix(y[i], ref[i]) for i in ids])
    rng = np.random.default_rng(20261114)
    draws = rng.integers(0, len(ids), (2000, len(ids)))
    rows = []
    for name, p in {'B_base': base, **preds}.items():
        c = np.stack([matrix(y[i], p[i]) for i in ids])
        boot = f1(c[draws].sum(1)).mean(-1)-f1(c0[draws].sum(1)).mean(-1)
        cm = c.sum(0)
        acc_e = np.array([(p[i] == y[i]).mean() for i in ids]); acc_0 = np.array([(ref[i] == y[i]).mean() for i in ids])
        rows.append(dict(method=name, macro_f1=macro(y, p), d_vs_base=macro(y, p)-macro(y, base),
                         d_vs_V0=macro(y, p)-macro(y, ref),
                         ci_vs_V0_lo=float(np.quantile(boot, .025)), ci_vs_V0_hi=float(np.quantile(boot, .975)),
                         true4_pred5=int(cm[3, 4]), true5_pred4=int(cm[4, 3]), true5_pred6=int(cm[4, 5]),
                         true6_pred5=int(cm[5, 4]), true4_pred6=int(cm[3, 5]), true6_pred4=int(cm[5, 3]),
                         changed_vs_V0=int((p != ref).sum()),
                         wrong_to_right_vs_V0=int(((ref != y) & (p == y)).sum()),
                         right_to_wrong_vs_V0=int(((ref == y) & (p != y)).sum()),
                         better_exps_vs_V0=int((acc_e > acc_0+1e-12).sum()), worse_exps_vs_V0=int((acc_e < acc_0-1e-12).sum()),
                         f1_stage4=float(f1(cm)[3]), f1_stage5=float(f1(cm)[4]), f1_stage6=float(f1(cm)[5])))
    return rows


def per_experiment(frame, base, preds, split):
    y = frame.label.to_numpy(int)
    rows = []
    for e, i in frame.groupby('experiment_id').indices.items():
        r = dict(split=split, experiment_id=e, bat_class=int(frame.bat_class.iloc[i[0]]), bat_cc=float(frame.bat_cc.iloc[i[0]]),
                 n=len(i), n_true456=int((y[i] >= 4).sum()), acc_B_base=float((base[i] == y[i]).mean()))
        for v in M.VARIANTS:
            p = preds[v]
            r[f'acc_{v}'] = float((p[i] == y[i]).mean())
            if v != 'V0':
                r[f'{v}_fixed_vs_V0'] = int(((preds['V0'][i] != y[i]) & (p[i] == y[i])).sum())
                r[f'{v}_broken_vs_V0'] = int(((preds['V0'][i] == y[i]) & (p[i] != y[i])).sum())
        rows.append(r)
    return rows


def changed_rows(frame, preds, split):
    y = frame.label.to_numpy(int)
    out = []
    for v in ('V1', 'V2', 'V3'):
        m = preds[v] != preds['V0']
        d = frame.loc[m, ['sample_id', 'experiment_id', 'relative_time_s', 'bat_class', 'bat_cc']].copy()
        d.insert(0, 'variant', v); d.insert(0, 'split', split)
        d['label'] = y[m]; d['V0'] = preds['V0'][m]; d[v] = preds[v][m]
        d['change'] = np.where(d[v] == d.label, 'wrong_to_right', np.where(d.V0 == d.label, 'right_to_wrong', 'wrong_to_wrong'))
        out.append(d.rename(columns={v: 'new'}))
    return pd.concat(out)


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(6)
    src = [Path(__file__), Path(M.__file__), ROOT/'targeted_rule1/model.py', ROOT/'targeted_rule1/run.py',
           ROOT/'compare_abc_phase_history.py']
    protocol = dict(base='original ABC B 3-seed mean + t1 45C rule; B frozen', variants=M.VARIANTS,
                    gate_alpha=M.GATE_ALPHA, gate_penalty=M.GATE_PENALTY, hidden=M.HIDDEN,
                    base_epochs=M.BASE_EPOCHS, head_epochs=M.HEAD_EPOCHS, seeds=SEEDS,
                    hyperparameters_selected_on='none (prefixed)', validation_read_after_training=True,
                    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in src})
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))

    frame = pd.read_csv(DATA/'train/train.csv')
    z = np.load(BOOF)
    assert np.array_equal(z['sample_id'], frame.sample_id.to_numpy(str))
    y = frame.label.to_numpy(int); folds = z['fold']; probs = z['oof']
    exps = frame.experiment_id.to_numpy()
    base = rule1(frame, probs.mean(0).argmax(1)+1)
    x, names, b, cond = M.build_features(frame, probs)
    group_of = M.feature_groups(names)
    KEYS[:] = combos(cond)
    print('features', x.shape, dict(pd.Series(group_of).value_counts()), flush=True)

    oof = {}
    for k in range(5):
        held = np.flatnonzero(folds == k); fit = np.flatnonzero(folds != k)
        assert not set(exps[held]) & set(exps[fit])
        fitted = Fitted(x[fit], b[fit], cond[fit], y[fit], exps[fit])
        pr = all_predictions(fitted, x[held], b[held], cond[held], base[held], group_of)
        for name, p in pr.items():
            oof.setdefault(name, np.zeros(len(frame), int))[held] = p
        print('fold', k, {v: round(macro(y[held], pr[v]), 4) for v in M.VARIANTS}, flush=True)

    results = {'train_oof': summarize(frame, base, oof)}
    per = per_experiment(frame, base, oof, 'train_oof')
    changes = [changed_rows(frame, oof, 'train_oof')]

    final = Fitted(x, b, cond, y, exps)   # validation·test 읽기 전에 고정
    torch.save({'prep': final.prep.__dict__, 'states': [{k: m.state_dict() for k, m in ms.items()} for ms in final.models],
                'names': names}, out/'final_models.pt')

    # 학습된 gate 값: 조합 × 특징
    keys = combos(cond)
    kc = np.array(keys, float)
    gate_rows = []
    for v in ('V2', 'V3'):
        g = final.gate_values(v, kc)
        for (cl, cc), row in zip(keys, g):
            for n, grp, val in zip(names, group_of, row):
                gate_rows.append(dict(variant=v, bat_class=cl, bat_cc=float(np.expm1(cc)), feature=n, group=grp, gate=float(val)))
    pd.DataFrame(gate_rows).to_csv(out/'gates_by_combo.csv', index=False)

    val = pd.read_csv(DATA/'validation/validation.csv')
    keep = val.experiment_id.to_numpy() != E16
    for recipe in ['original_B', 'fold_ensemble_B']:
        vp = b_probs('validation', val.sample_id) if recipe == 'original_B' else z['validation']
        if recipe != 'original_B':
            assert np.array_equal(z['val_sample_id'], val.sample_id.to_numpy(str))
        vb = rule1(val, vp.mean(0).argmax(1)+1)
        vx, vn, vbb, vc = M.build_features(val, vp)
        assert vn == names
        pr = all_predictions(final, vx, vbb, vc, vb, group_of)
        results[f'val_{recipe}'] = summarize(val, vb, pr)
        sub = val.iloc[np.flatnonzero(keep)].reset_index(drop=True)
        results[f'val_{recipe}_noE16'] = summarize(sub, vb[keep], {k: p[keep] for k, p in pr.items()})
        per += per_experiment(val, vb, pr, f'val_{recipe}')
        changes.append(changed_rows(val, pr, f'val_{recipe}'))

    rows = [dict(split=s, **r) for s, rs in results.items() for r in rs]
    table = pd.DataFrame(rows)
    table.to_csv(out/'summary.csv', index=False)
    pd.DataFrame(per).to_csv(out/'per_experiment.csv', index=False)
    pd.concat(changes).to_csv(out/'changed_rows_vs_V0.csv', index=False)
    cols = ['split', 'method', 'macro_f1', 'd_vs_base', 'd_vs_V0', 'ci_vs_V0_lo', 'true4_pred5', 'true5_pred4',
            'wrong_to_right_vs_V0', 'right_to_wrong_vs_V0', 'better_exps_vs_V0', 'worse_exps_vs_V0']
    print(table[table.method.isin(['B_base', *M.VARIANTS])][cols].round(4).to_string(index=False), flush=True)

    # ---- Test: 고정 모델로 추론만 ----
    test = pd.read_csv(DATA/'test/test.csv'); assert 'label' not in test
    tp = b_probs('test', test.sample_id)

    def run_all(f, pp):
        bb = rule1(f, pp.mean(0).argmax(1)+1)
        fx, fn, fb, fc = M.build_features(f, pp)
        assert fn == names
        return {v: M.apply456(bb, final.probs(v, fx, fb, fc)) for v in M.VARIANTS}, bb

    full, tb = run_all(test, tp)
    checks = dict(per_experiment=True, shuffled=True, prefixes=0, prefix_mismatch=0)
    t = test.relative_time_s.to_numpy()
    for ids in test.groupby('experiment_id').indices.values():
        o, _ = run_all(test.iloc[ids].reset_index(drop=True), tp[:, ids])
        checks['per_experiment'] &= all(np.array_equal(o[v], full[v][ids]) for v in full)
        ids = ids[np.argsort(t[ids], kind='stable')]
        for end in sorted(set(np.linspace(1, len(ids), 12).astype(int))):
            s = ids[:end]
            o, _ = run_all(test.iloc[s].reset_index(drop=True), tp[:, s])
            checks['prefixes'] += 1
            checks['prefix_mismatch'] += int(any(not np.array_equal(o[v], full[v][s]) for v in full))
    perm = np.random.default_rng(0).permutation(len(test))
    o, _ = run_all(test.iloc[perm].reset_index(drop=True), tp[:, perm])
    checks['shuffled'] = all(np.array_equal(o[v], full[v][perm]) for v in full)
    sample = pd.read_csv(DATA/'sample_submission.csv')
    files = []
    for v in M.VARIANTS:
        sub = sample[['sample_id']].merge(pd.DataFrame({'sample_id': test.sample_id, 'label': full[v]}), on='sample_id', how='left')
        assert len(sub) == len(sample) and sub.label.between(1, 6).all()
        sub['label'] = sub.label.astype(int)
        path = out/f'submission_condgate_{v}.csv'
        sub.to_csv(path, index=False)
        files.append(dict(file=path.name, changed_vs_base=int((full[v] != tb).sum()),
                          changed_vs_V0=int((full[v] != full['V0']).sum()), sha256=sha(path)))
    (out/'test_manifest.json').write_text(json.dumps(dict(checks=checks, files=files), indent=2))
    print('TEST', json.dumps(dict(checks=checks, files=files), indent=1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    main(ap.parse_args().output)
