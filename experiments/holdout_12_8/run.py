"""고정 holdout(원래 Train 12 + Validation 8실험)으로 보정 후보를 한 번만 확인한다.

역할 분리
  - 개발 77실험: 후보 학습과 선택(실험 단위 5-fold, b_holdout.py와 같은 fold, B 확률도 그 fold의 OOF).
  - holdout 20실험: 선택이 끝난 뒤 한 번 평가. 어떤 B·정규화·보정기·기준 선택에도 쓰지 않는다.
  - Test: 읽지 않는다.

기준(base) = 원본 B 레시피 3-seed 평균 argmax + t1 45°C 규칙(현재까지 meas_t1 최고).
후보
  - targeted_rule1: targeted_rule1.model.train(4→5·5→6 HGB, 기준은 학습 실험 내부 3-fold로 선택)
  - V0_nocount: MLP456, 관측 개수 특징 제거, 간격 증강 없음
  - V0_nocount_56only: 위와 같고 5↔6 변경만 허용
  - V0_aug_ref: 간격 증강(비교용, 선택 대상 아님)

선택 규칙(holdout을 보기 전 고정, 개발 OOF만 사용). private 구성을 모르므로 탈락 기준은 최소로 둔다.
  탈락 기준: 4개 간격(5s/10s/15s/10·15s) 평균 Macro F1 이득 > 0 이고 5s 실험 bootstrap 95% 하한 > 0
  선택: 통과 후보 중 4개 간격 평균 이득이 가장 큰 것. 없으면 기준(B+t1)만.
  경고(선택에 영향 없음): 간격별 이득 음수, 최악 간격 개선<악화 실험, 위험 과소 판정 증가,
    배터리 조합 통째 제외에서 1점 넘게 하락
holdout 채택 확인(보고용, 선택을 바꾸지 않음): 전체·train 출신·validation 출신 모두 개선, 한 실험 제외 최소 개선 > 0,
  개선 실험 > 악화 실험.

조건 통째 제외: 배터리 조합 하나를 통째로 뺀 MLP/HGB 학습 → 그 조합의 개발 실험에서 평가(B는 다른 fold의 같은 조합 실험을 봄).
t1 45°C: 개발 실험 라벨만으로 1단계 최고 t1 < 45 ≤ 2단계 진입 시 누적 최고 t1 인지 확인(holdout 라벨 미사용).

  python -u experiments/holdout_12_8/run.py --output experiments/holdout_12_8/run_v1
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
from experiments.b_cadence_robust import run as C
from experiments.holdout_12_8.b_holdout import pooled
from targeted_rule1.model import train as targeted_train, infer as targeted_infer, matrix, f1

BPROBS = ROOT/'experiments/holdout_12_8/b_v1/b_probs.npz'
E16 = C.E16
SELECTABLE = ['targeted_rule1', 'V0_nocount', 'V0_nocount_56only']
CONDITION_MAX_DROP = 0.01
TEST_COMBO_SHARE = {(1, 4.8): 4, (3, 280.0): 2, (3, 28.0): 2, (1, 6.0): 1, (2, 73.0): 1, (2, 39.0): 1}  # test 11실험 구성(라벨 없음)
MLP = {'V0_nocount': (False, 'all'), 'V0_nocount_56only': (False, '56only'), 'V0_aug_ref': (True, 'all')}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def macro(y, p):
    return float(f1(matrix(y, p)).mean())


def loo_min(frame, y, base, p):
    ids = list(frame.groupby('experiment_id').indices.values())
    cb = np.stack([matrix(y[i], base[i]) for i in ids]); cp = np.stack([matrix(y[i], p[i]) for i in ids])
    return float(min(f1(cp.sum(0)-a).mean()-f1(cb.sum(0)-b).mean() for a, b in zip(cp, cb)))


class Candidates:
    """한 학습 실험 집합에서 모든 후보를 학습."""

    def __init__(self, views, fit_exps, robust_cols, names):
        self.models = {}
        for aug in (False, True):
            keys = C.TRAIN_VIEWS if aug else [(1, 0)]
            parts = [views[k] for k in keys]
            m = [np.isin(v['exps'], list(fit_exps)) for v in parts]
            x, b, y, e = (np.concatenate([v[a][i] for v, i in zip(parts, m)]) for a in ('x', 'b', 'y', 'exps'))
            self.models[aug] = C.Model(x, b, y, e, robust_cols)
        v = views[(1, 0)]; m = np.isin(v['exps'], list(fit_exps))
        self.targeted = targeted_train(v['frame'][m].reset_index(drop=True), v['probs'][:, m], v['y'][m], v['base'][m])

    def predict(self, v, mask=None):
        m = np.ones(len(v['base']), bool) if mask is None else mask
        out = {'B_rule1': v['base'][m]}
        f = v['frame'][m].reset_index(drop=True)
        out['targeted_rule1'] = targeted_infer(self.targeted, f, v['probs'][:, m], v['base'][m])['both']
        for name, (aug, mode) in MLP.items():
            out[name] = C.decode(v['base'][m], self.models[aug].p456(v['x'][m], v['b'][m]), mode)
        return out


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    tr, va, frame = pooled()
    z = np.load(BPROBS)
    assert np.array_equal(z['sample_id'], frame.sample_id.to_numpy(str))
    dev_i, hold_i = z['dev_index'], z['holdout_index']
    dev = frame.iloc[dev_i].reset_index(drop=True); hold = frame.iloc[hold_i].reset_index(drop=True)
    dev_p, hold_p = z['dev_oof'], z['holdout']
    fold_of_exp = dict(zip(dev.experiment_id, z['fold'][dev_i]))
    assert min(fold_of_exp.values()) == 0 and not set(dev.experiment_id) & set(hold.experiment_id)
    src = [Path(__file__), Path(C.__file__), ROOT/'experiments/b_cond_gate/model.py', ROOT/'targeted_rule1/model.py',
           ROOT/'targeted_rule1/run.py', ROOT/'compare_abc_phase_history.py', ROOT/'experiments/holdout_12_8/b_holdout.py']
    protocol = dict(selectable=SELECTABLE, reference_only=['V0_aug_ref'], cadences=C.EVAL_CADENCES,
                    selection='dev OOF only: all cadences > base, worst-cadence better_exps>worse_exps, '
                              'danger_under<=base at all cadences; max worst-cadence gain; else base',
                    holdout_read_after_selection=True, test_read=False,
                    holdout_sources=hold.drop_duplicates('experiment_id').original_split.value_counts().to_dict(),
                    E16_in_holdout=bool((hold.experiment_id == E16).any()),
                    b_probs_sha256=sha(BPROBS), source_sha256={str(p.relative_to(ROOT)): sha(p) for p in src})
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2, default=str))

    # t1 45°C 임계값을 개발 라벨만으로 확인
    t1 = []
    for e, g in dev.sort_values('relative_time_s').groupby('experiment_id'):
        cm = g.meas_t1.cummax(); s2 = g.label.to_numpy() >= 2
        t1.append((g.meas_t1[g.label == 1].max(), cm[s2].iloc[0] if s2.any() else np.nan))
    t1 = np.array(t1, float)
    t1_check = dict(dev_stage1_max_t1=float(np.nanmax(t1[:, 0])), dev_stage2_entry_min_cummax_t1=float(np.nanmin(t1[:, 1])))
    t1_check['threshold_45_separates_dev'] = bool(t1_check['dev_stage1_max_t1'] < 45 <= t1_check['dev_stage2_entry_min_cummax_t1'])
    (out/'t1_threshold_dev_check.json').write_text(json.dumps(t1_check, indent=2))
    print('t1 check', t1_check, flush=True)

    dviews = {v: C.view_data(dev, dev_p, v) for v in C.TRAIN_VIEWS}
    names = dviews[(1, 0)]['names']
    robust = np.array([i for i, n in enumerate(names) if n not in C.COUNT_FEATURES])
    exps = np.array(sorted(fold_of_exp))
    evals = {c: dviews[v] for c, v in C.EVAL_CADENCES.items()}
    oof = {c: {} for c in evals}
    for k in range(5):
        fit = set(exps[[fold_of_exp[e] != k for e in exps]]); held = set(exps) - fit
        cand = Candidates(dviews, fit, robust, names)
        for c, v in evals.items():
            m = np.isin(v['exps'], list(held))
            for name, p in cand.predict(v, m).items():
                oof[c].setdefault(name, np.zeros(len(v['base']), int))[m] = p
        print('fold', k, 'done', flush=True)

    rows = []
    for c, v in evals.items():
        for name, p in oof[c].items():
            rows.append(dict(split='dev_oof', cadence=c, method=name, **C.metrics(v['frame'], v['y'], v['base'], p)))
    dev_table = pd.DataFrame(rows)

    # ---- 배터리 조합 통째 제외 (개발 실험, MLP/HGB 단계) — 선택 규칙 4에 사용 ----
    combo = dev.drop_duplicates('experiment_id').set_index('experiment_id')[['bat_class', 'bat_cc']].apply(tuple, axis=1)
    lco = []
    v5 = evals['5s']; v15 = evals['10/15s']
    for key in sorted(set(combo)):
        held = set(combo.index[combo == key]); fit = set(exps) - held
        cand = Candidates(dviews, fit, robust, names)
        for c, v in [('5s', v5), ('10/15s', v15)]:
            m = np.isin(v['exps'], list(held))
            f = v['frame'][m].reset_index(drop=True)
            for name, p in cand.predict(v, m).items():
                lco.append(dict(bat_class=key[0], bat_cc=key[1], experiments=len(held), cadence=c, method=name,
                                **C.metrics(f, v['y'][m], v['base'][m], p)))
        print('leave-combo-out', key, flush=True)
    pd.DataFrame(lco).to_csv(out/'leave_condition_out.csv', index=False)

    lco = pd.DataFrame(lco)
    choice = {}
    base_rows = dev_table[dev_table.method == 'B_rule1'].set_index('cadence')
    for name in SELECTABLE:
        t = dev_table[dev_table.method == name].set_index('cadence')
        worst = t.d_vs_base.idxmin()
        choice[name] = dict(all_positive=bool((t.d_vs_base > 0).all()), worst_cadence=worst, worst_gain=float(t.d_vs_base.min()),
                            worst_better_gt_worse=bool(t.loc[worst, 'better_exps'] > t.loc[worst, 'worse_exps']),
                            danger_not_worse=bool((t.danger_under <= base_rows.loc[t.index, 'danger_under']).all()))
        lc = lco[lco.method == name]
        worst_lc = lc.loc[lc.d_vs_base.idxmin()]
        choice[name].update(condition_worst_drop=float(worst_lc.d_vs_base),
                            condition_worst=f"{worst_lc.bat_class}/{worst_lc.bat_cc}@{worst_lc.cadence}",
                            condition_ok=bool(lc.d_vs_base.min() >= -CONDITION_MAX_DROP))
        choice[name].update(mean_gain=float(t.d_vs_base.mean()), ci_lo_5s=float(t.loc['5s', 'ci_lo']),
                            passed=bool(t.d_vs_base.mean() > 0 and t.loc['5s', 'ci_lo'] > 0))
    # 탈락 기준은 passed 하나(평균 이득 > 0, 5s bootstrap 하한 > 0). 나머지 항목은 경고로만 보고한다.
    ok = [n for n, r in choice.items() if r['passed']]
    selected = max(ok, key=lambda n: choice[n]['mean_gain']) if ok else 'B_rule1'
    (out/'selection.json').write_text(json.dumps(dict(choice=choice, selected=selected), indent=2))
    print('SELECTED', selected, json.dumps(choice, indent=1), flush=True)

    # ---- 최종 학습(개발 77실험) → holdout 1회 평가 ----
    final = Candidates(dviews, set(exps), robust, names)
    per_exp = []
    for c, view in C.EVAL_CADENCES.items():
        v = C.view_data(hold, hold_p, view)
        preds = final.predict(v)
        for name, p in preds.items():
            for scope, m in [('all', np.ones(len(p), bool)), ('from_train12', v['frame'].original_split.eq('train').to_numpy()),
                             ('from_validation8', v['frame'].original_split.eq('validation').to_numpy()),
                             ('all_noE16', v['exps'] != E16)]:
                if not m.any():
                    continue
                f = v['frame'][m].reset_index(drop=True)
                r = C.metrics(f, v['y'][m], v['base'][m], p[m])
                r['loo_min'] = loo_min(f, v['y'][m], v['base'][m], p[m]) if f.experiment_id.nunique() > 1 else np.nan
                rows.append(dict(split=f'holdout_{scope}', cadence=c, method=name, **r))
            for e, i in v['frame'].groupby('experiment_id').indices.items():
                y = v['y'][i]
                per_exp.append(dict(cadence=c, method=name, experiment_id=e, source=v['frame'].original_split.iloc[i[0]],
                                    bat_class=int(v['frame'].bat_class.iloc[i[0]]), bat_cc=float(v['frame'].bat_cc.iloc[i[0]]),
                                    rows=len(i), true456=int((y >= 4).sum()),
                                    f1_base=macro(y, v['base'][i]), f1=macro(y, p[i]),
                                    acc_base=float((v['base'][i] == y).mean()), acc=float((p[i] == y).mean()),
                                    fixed=int(((v['base'][i] != y) & (p[i] == y)).sum()),
                                    broken=int(((v['base'][i] == y) & (p[i] != y)).sum())))
    table = pd.DataFrame(rows)
    table.to_csv(out/'summary.csv', index=False)
    pe = pd.DataFrame(per_exp)
    pe.to_csv(out/'holdout_per_experiment.csv', index=False)
    pe['d'] = pe.f1-pe.f1_base
    ew = pe.groupby(['cadence', 'method']).agg(exp_weighted_f1=('f1', 'mean'), exp_weighted_d=('d', 'mean'),
                                               better=('d', lambda d: int((d > 1e-12).sum())), worse=('d', lambda d: int((d < -1e-12).sum())))
    combo_mean = pe.groupby(['cadence', 'method', 'bat_class', 'bat_cc']).d.mean().reset_index()
    combo_mean['w'] = [TEST_COMBO_SHARE.get((int(a), float(b)), 0) for a, b in zip(combo_mean.bat_class, combo_mean.bat_cc)]
    tw = combo_mean.groupby(['cadence', 'method']).apply(lambda g: float((g.d*g.w).sum()/g.w.sum()), include_groups=False)
    ew['test_composition_weighted_d'] = tw
    ew.reset_index().to_csv(out/'holdout_experiment_weighted.csv', index=False)

    pd.set_option('display.width', 250)
    cols = ['split', 'cadence', 'method', 'macro_f1', 'd_vs_base', 'ci_lo', 'fixed', 'broken', 'better_exps', 'worse_exps',
            'danger_under', 'onset6_late_exps', 'loo_min']
    print(table[table.split.str.startswith('holdout')][cols].round(4).to_string(index=False), flush=True)
    print(ew.round(4).to_string(), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    main(ap.parse_args().output)
