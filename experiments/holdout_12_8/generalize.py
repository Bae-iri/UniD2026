"""보정을 줄이면서 범용성을 높이는 두 방법: 단조 하한선(floor)과 두 모델 동의 게이트(agree).

근거: train+validation 97실험 모두에서 단계 라벨은 시간에 따라 내려가지 않는다(2단계 건너뛰기 11회).

floor@H  : 같은 실험에서 예측이 s 이상인 상태가 H초 이상 끊김 없이 이어지면 s를 확정하고, 이후 예측을 s 미만으로
           내리지 않는다. 현재 행까지의 예측만 쓴다(원 예측으로 확정, 하한선이 스스로를 강화하지 않음).
agree    : precision65(S40) 레시피가 기준을 바꾼 행 중, 독립적으로 학습한 MLP(V0_nocount)도 같은 단계를 낼 때만 바꾼다.
           S40이 바꾸는 행의 부분집합이므로 보정 양은 항상 S40 이하.

역할 분리·데이터: correction_ablation.py와 같다(개발 77실험 5-fold OOF로 선택, 12+8 holdout 1회 확인, Test 미사용).
선택 규칙(holdout 보기 전 고정): 후보 중 개발 OOF 4개 간격 평균 이득 ≥ S40 평균 이득 − 0.3점, 5s bootstrap 하한 > 0
  을 만족하는 것 가운데 개발 5s에서 기준 대비 바뀐 행이 가장 적은 것.

  python -u experiments/holdout_12_8/generalize.py --output experiments/holdout_12_8/generalize_v1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.b_cadence_robust import run as C
from experiments.holdout_12_8 import correction_ablation as A
from experiments.holdout_12_8.b_holdout import pooled

HOLDS = (15, 30, 60, 120)
TOLERANCE = 0.003
E16 = C.E16


def floor(frame, pred, hold):
    t = frame.relative_time_s.to_numpy(float)
    pred = np.asarray(pred, int)
    out = pred.copy()
    for ids in frame.groupby('experiment_id', sort=False).indices.values():
        ids = ids[np.argsort(t[ids], kind='stable')]
        start = {s: None for s in range(2, 7)}
        confirmed = 1
        for i in ids:
            for s in range(2, 7):
                if pred[i] >= s:
                    start[s] = t[i] if start[s] is None else start[s]
                    if t[i]-start[s] >= hold:
                        confirmed = max(confirmed, s)
                else:
                    start[s] = None
            out[i] = max(pred[i], confirmed)
    return out


def agree(base, s40, v0):
    out = np.asarray(base, int).copy()
    m = (s40 != base) & (v0 == s40)
    out[m] = s40[m]
    return out


class Both:
    def __init__(self, view, fit_mask, robust):
        f = view['frame'][fit_mask].reset_index(drop=True)
        self.hgb = A.Fitted(f, view['probs'][:, fit_mask], view['y'][fit_mask], view['base'][fit_mask])
        self.mlp = C.Model(view['x'][fit_mask], view['b'][fit_mask], view['y'][fit_mask], view['exps'][fit_mask], robust)

    def predict(self, v, m):
        f = v['frame'][m].reset_index(drop=True)
        base = v['base'][m]
        pr, _ = self.hgb.variants(f, v['probs'][:, m], base)
        v0 = C.decode(base, self.mlp.p456(v['x'][m], v['b'][m]), 'all')
        out = {'base': base, 'S41': pr['S41'], 'S40': pr['S40'], 'V0': v0, 'agree': agree(base, pr['S40'], v0)}
        for h in HOLDS:
            out[f'floor{h}'] = floor(f, base, h)
            out[f'S40+floor{h}'] = floor(f, pr['S40'], h)
            out[f'agree+floor{h}'] = floor(f, out['agree'], h)
        return out


def evaluate(frame, y, base, preds, split, cadence):
    rows = []
    for name, p in preds.items():
        r = C.metrics(frame, y, base, p)
        r['loo_min'] = A.loo_min(frame, y, base, p)
        rows.append(dict(split=split, cadence=cadence, method=name, **r))
    return rows


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    tr, va, frame = pooled()
    z = np.load(A.BPROBS)
    assert np.array_equal(z['sample_id'], frame.sample_id.to_numpy(str))
    dev = frame.iloc[z['dev_index']].reset_index(drop=True); hold = frame.iloc[z['holdout_index']].reset_index(drop=True)
    dev_p, hold_p = z['dev_oof'], z['holdout']
    fold = z['fold'][z['dev_index']]
    dv = {c: C.view_data(dev, dev_p, v) for c, v in C.EVAL_CADENCES.items()}
    full = dv['5s']
    robust = np.array([i for i, n in enumerate(full['names']) if n not in C.COUNT_FEATURES])
    (out/'protocol.json').write_text(json.dumps(dict(holds=HOLDS, tolerance=TOLERANCE, selection=__doc__.split('선택 규칙')[1].strip(),
                                                     test_read=False), indent=2, ensure_ascii=False))

    oof = {c: {} for c in dv}
    for k in range(5):
        model = Both(full, fold != k, robust)
        held = set(dev.experiment_id[fold == k])
        for c, v in dv.items():
            m = np.isin(v['exps'], list(held))
            for name, p in model.predict(v, m).items():
                oof[c].setdefault(name, np.zeros(len(v['base']), int))[m] = p
        print('fold', k, flush=True)
    rows = []
    for c, v in dv.items():
        rows += evaluate(v['frame'], v['y'], v['base'], oof[c], 'dev_oof', c)
    dev_t = pd.DataFrame(rows)

    # ---- 선택 (개발 OOF만) ----
    g = dev_t.groupby('method')
    summary = pd.DataFrame(dict(mean_gain=g.d_vs_base.mean(),
                                ci_lo_5s=dev_t[dev_t.cadence == '5s'].set_index('method').ci_lo,
                                changed_5s=dev_t[dev_t.cadence == '5s'].set_index('method').changed))
    ref = summary.loc['S40', 'mean_gain']
    cand = summary.drop(index=['base', 'S41', 'V0'])
    ok = cand[(cand.mean_gain >= ref-TOLERANCE) & (cand.ci_lo_5s > 0)]
    selected = ok.changed_5s.idxmin() if len(ok) else 'S40'
    (out/'selection.json').write_text(json.dumps(dict(selected=selected, reference_S40_mean_gain=float(ref),
        table=summary.round(5).reset_index().to_dict('records')), indent=2))
    print('SELECTED', selected, flush=True)
    print(summary.sort_values('mean_gain', ascending=False).round(4).to_string(), flush=True)

    # ---- 배터리 조합 통째 제외 (개발, 5s·10/15s) ----
    combo = dev.bat_class.astype(int).astype(str)+'/'+dev.bat_cc.astype(str)
    lco = []
    for key in sorted(combo.unique()):
        held = (combo == key).to_numpy()
        model = Both(full, ~held, robust)
        hexps = set(dev.experiment_id[held])
        for c in ('5s', '10/15s'):
            v = dv[c]; m = np.isin(v['exps'], list(hexps))
            f = v['frame'][m].reset_index(drop=True)
            for r in evaluate(f, v['y'][m], v['base'][m], model.predict(v, m), 'leave_condition_out', c):
                lco.append(dict(combo=key, experiments=len(hexps), **r))
        print('lco', key, flush=True)
    pd.DataFrame(lco).to_csv(out/'leave_condition_out.csv', index=False)

    # ---- holdout 1회 ----
    model = Both(full, np.ones(len(dev), bool), robust)
    for c, view in C.EVAL_CADENCES.items():
        v = C.view_data(hold, hold_p, view)
        preds = model.predict(v, np.ones(len(v['base']), bool))
        for scope, m in [('all', np.ones(len(v['base']), bool)), ('from_train12', v['frame'].original_split.eq('train').to_numpy()),
                         ('from_validation8', v['frame'].original_split.eq('validation').to_numpy()),
                         ('all_noE16', v['exps'] != E16)]:
            f = v['frame'][m].reset_index(drop=True)
            rows += evaluate(f, v['y'][m], v['base'][m], {k: p[m] for k, p in preds.items()}, f'holdout_{scope}', c)
    t = pd.DataFrame(rows)
    t.to_csv(out/'summary.csv', index=False)
    t['d'] = (t.d_vs_base*100).round(2)
    pd.set_option('display.width', 250)
    print(t[t.cadence == '5s'].pivot_table(index='method', columns='split', values='d', sort=False).to_string(), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    with threadpool_limits(limits=8):
        main(ap.parse_args().output)
