"""최종 모델(B + t1 + 5→6 + 6→5)에 2↔3·3↔4 보정을 더했을 때의 검증. Test 미사용.

데이터·역할: holdout_12_8과 같다(개발 77실험 실험 단위 5-fold OOF, 12+8 holdout 1회, B 확률도 그 분할의 OOF).
채택 규칙(결과 보기 전 고정):
  보정 하나를 최종 모델에 단독으로 더했을 때
   (1) 개발 OOF 이득 > 0 : 4개 간격(5s·10s·15s·10/15s) 모두
   (2) 배터리 조합 통째 제외 평균 이득 > 0 : 5s와 10/15s 모두
  를 만족하는 보정만 남기고, 남은 것을 모두 더한 모델을 후보로 한다. holdout은 그 뒤 확인만 한다.
  python -u experiments/lower_heads/evaluate.py
"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from experiments.b_cadence_robust import run as C
from experiments.holdout_12_8 import correction_ablation as A
from experiments.holdout_12_8.b_holdout import pooled
from experiments.lower_heads import model as L
OUT = ROOT/'experiments/lower_heads/eval_v1'
PAIRS = list(L.PAIRS)


class Fit:
    def __init__(self, frame, probs, y, base):
        self.upper = A.Fitted(frame, probs, y, base)
        self.lower = L.train(frame, probs, y, base)

    def predict(self, frame, probs, base):
        final = self.upper.variants(frame, probs, base)[0]['56+65']
        x, _ = L.features(frame, probs)
        ch = {p: L.apply(base, A.probability(self.lower['models'][p], x), p, self.lower['thresholds'][p]) for p in PAIRS}
        out = {'base': base.copy(), 'final': final}
        for r in range(1, 5):
            for combo in itertools.combinations(PAIRS, r):
                o = final.copy(); fired = np.zeros(len(o), int)
                for p in combo:
                    m = ch[p] != base; o[m] = ch[p][m]; fired += m
                o[fired > 1] = base[fired > 1]
                out['final+'+'+'.join(combo)] = o
        return out, dict(self.lower['thresholds'])


def ev(f, y, base, preds, **kw):
    rows = []
    for n, p in preds.items():
        r = C.metrics(f, y, base, p); r['loo_min'] = A.loo_min(f, y, base, p); rows.append(dict(kw, method=n, **r))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    tr, va, frame = pooled(); z = np.load(A.BPROBS)
    dev = frame.iloc[z['dev_index']].reset_index(drop=True); hold = frame.iloc[z['holdout_index']].reset_index(drop=True)
    dev_p, hold_p, fold = z['dev_oof'], z['holdout'], z['fold'][z['dev_index']]
    dv = {c: C.view_data(dev, dev_p, v) for c, v in C.EVAL_CADENCES.items()}; full = dv['5s']
    rows, ths = [], []
    oof = {c: {} for c in dv}
    for k in range(5):
        fit = fold != k; f = Fit(dev[fit].reset_index(drop=True), dev_p[:, fit], full['y'][fit], full['base'][fit])
        held = set(dev.experiment_id[fold == k])
        for c, v in dv.items():
            m = np.isin(v['exps'], list(held)); pr, th = f.predict(v['frame'][m].reset_index(drop=True), v['probs'][:, m], v['base'][m])
            for n, p in pr.items(): oof[c].setdefault(n, np.zeros(len(v['base']), int))[m] = p
        ths.append(dict(where=f'fold{k}', **th)); print('fold', k, th, flush=True)
    for c, v in dv.items(): rows += ev(v['frame'], v['y'], v['base'], oof[c], split='dev_oof', cadence=c)
    combo = dev.bat_class.astype(int).astype(str)+'/'+dev.bat_cc.astype(str)
    for key in sorted(combo.unique()):
        held = (combo == key).to_numpy(); f = Fit(dev[~held].reset_index(drop=True), dev_p[:, ~held], full['y'][~held], full['base'][~held])
        for c in ('5s', '10/15s'):
            v = dv[c]; m = np.isin(v['exps'], list(set(dev.experiment_id[held]))); fr = v['frame'][m].reset_index(drop=True)
            pr, th = f.predict(fr, v['probs'][:, m], v['base'][m]); rows += ev(fr, v['y'][m], v['base'][m], pr, split=f'lco_{key}', cadence=c)
        ths.append(dict(where=f'lco_{key}', **th)); print('lco', key, th, flush=True)
    t = pd.DataFrame(rows); t['d'] = t.d_vs_base*100
    # ---- 선택 (개발만) ----
    dev_t = t[t.split == 'dev_oof'].pivot_table(index='method', columns='cadence', values='d')
    lco = t[t.split.str.startswith('lco')].groupby(['method', 'cadence']).d.mean().unstack()
    fin_dev, fin_lco = dev_t.loc['final'], lco.loc['final']
    kept = [p for p in PAIRS if (dev_t.loc[f'final+{p}'] > fin_dev).all() and (lco.loc[f'final+{p}'] > fin_lco).all()]
    selected = 'final+'+'+'.join(kept) if kept else 'final'
    (OUT/'selection.json').write_text(json.dumps(dict(kept=kept, selected=selected,
        dev_gain_vs_final=(dev_t.sub(fin_dev, axis=1)).round(3).to_dict('index'),
        lco_gain_vs_final=(lco.sub(fin_lco, axis=1)).round(3).to_dict('index')), indent=1))
    print('KEPT', kept, 'SELECTED', selected, flush=True)
    # ---- holdout 1회 ----
    f = Fit(dev, dev_p, full['y'], full['base'])
    for c, view in C.EVAL_CADENCES.items():
        v = C.view_data(hold, hold_p, view); pr, th = f.predict(v['frame'], v['probs'], v['base'])
        for scope, m in [('all', np.ones(len(v['base']), bool)), ('noE16', v['exps'] != C.E16)]:
            fr = v['frame'][m].reset_index(drop=True)
            rows += ev(fr, v['y'][m], v['base'][m], {k: p[m] for k, p in pr.items()}, split=f'holdout_{scope}', cadence=c)
    ths.append(dict(where='final_dev', **th))
    t = pd.DataFrame(rows); t['d'] = (t.d_vs_base*100).round(2); t.to_csv(OUT/'summary.csv', index=False)
    (OUT/'thresholds.json').write_text(json.dumps(ths, indent=1))
    pd.set_option('display.width', 300); pd.set_option('display.max_rows', 100)
    show = ['final', 'final+23', 'final+32', 'final+34', 'final+43', selected, 'final+23+32+34+43']
    show = list(dict.fromkeys(show))
    main_t = t[~t.split.str.startswith('lco') & t.method.isin(show)]
    print(main_t.pivot_table(index='method', columns=['split', 'cadence'], values='d', sort=False).loc[show].to_string())
    l = t[t.split.str.startswith('lco') & t.method.isin(show)]
    print(l.pivot_table(index='method', columns=['cadence', 'split'], values='d', sort=False).loc[show].to_string())
    print(l.groupby(['cadence', 'method'])[['d', 'changed', 'broken', 'danger_under', 'onset6_late_exps']].mean().round(2).to_string())
    cols = ['split', 'cadence', 'method', 'macro_f1', 'd', 'ci_lo', 'loo_min', 'changed', 'fixed', 'broken', 'better_exps', 'worse_exps', 'danger_under']
    print(main_t[main_t.cadence.isin(['5s', '10/15s'])][cols].round(4).to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=8): main()
