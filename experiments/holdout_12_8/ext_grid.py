"""최종 모델(5→6 + 6→5)에서 6→5 기준 후보를 넓혀(0.5·0.6 추가) 학습 내부 자동 선택에 맡기는 변형 검증.

선택 규칙은 gahyun precision.train과 같다(학습 실험 3-fold OOF, delta>0, 고친 행≥3×망친 행, 개선 실험 수 조건, LOO>0).
4→5 헤드는 끈다. 개발 OOF / holdout 1회 / 배터리 조합 통째 제외. Test 미사용.
  python -u experiments/holdout_12_8/ext_grid.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from experiments.b_cadence_robust import run as C
from experiments.holdout_12_8 import correction_ablation as A
from experiments.holdout_12_8.b_holdout import pooled
P = A.P
ORIG = (.7, .9, .95, .99); EXT = (.5, .6, .7, .9, .95, .99)
OUT = ROOT/'experiments/holdout_12_8/ext_grid_v1'


class Fit:
    def __init__(self, frame, probs, y, base):
        self.art = A.targeted_train(frame, probs, y, base)
        x = A.targeted_features(frame, probs)[0]
        s56 = A.probability(self.art['models']['56'], x['56'])
        bl = A.correct(base, s56*0, s56, 9, self.art['thresholds']['56'])
        g = frame.experiment_id.to_numpy(); self.gates = {}
        for name, grid in [('orig', ORIG), ('ext', EXT)]:
            P.THRESHOLDS = list(grid); self.gates[name] = P.train(x['56'], y, bl, base, g)
        P.THRESHOLDS = list(ORIG)

    def predict(self, frame, probs, base):
        x = A.targeted_features(frame, probs)[0]
        s45 = A.probability(self.art['models']['45'], x['45']); s56 = A.probability(self.art['models']['56'], x['56'])
        th = self.art['thresholds']; only56 = A.correct(base, s45, s56, 9, th['56'])
        s41 = A.correct(base, s45, s56, th['45'], th['56'])
        out = {'base': base.copy(), 'S40': P.apply(s41, base, P.score(self.gates['orig']['model'], x['56']), self.gates['orig']['threshold'])}
        for n, gt in self.gates.items():
            out[f'56+65_{n}'] = P.apply(only56, base, P.score(gt['model'], x['56']), gt['threshold'])
        return out, {n: gt['threshold'] for n, gt in self.gates.items()}


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
    f = Fit(dev, dev_p, full['y'], full['base'])
    for c, view in C.EVAL_CADENCES.items():
        v = C.view_data(hold, hold_p, view); pr, th = f.predict(v['frame'], v['probs'], v['base'])
        for scope, m in [('all', np.ones(len(v['base']), bool)), ('noE16', v['exps'] != C.E16)]:
            fr = v['frame'][m].reset_index(drop=True)
            rows += ev(fr, v['y'][m], v['base'][m], {k: p[m] for k, p in pr.items()}, split=f'holdout_{scope}', cadence=c)
    ths.append(dict(where='final_dev', **th))
    t = pd.DataFrame(rows); t.to_csv(OUT/'summary.csv', index=False); (OUT/'thresholds.json').write_text(json.dumps(ths, indent=1))
    t['d'] = (t.d_vs_base*100).round(2); pd.set_option('display.width', 250)
    main_s = t[~t.split.str.startswith('lco')]
    print(main_s.pivot_table(index='method', columns=['split', 'cadence'], values='d', sort=False).to_string())
    l = t[t.split.str.startswith('lco')]
    print(l.pivot_table(index='method', columns=['cadence', 'split'], values='d', sort=False).to_string())
    print(l.groupby(['cadence', 'method'])[['d', 'changed', 'broken', 'danger_under', 'onset6_late_exps']].mean().round(2).to_string())
    cols = ['split', 'cadence', 'method', 'd', 'ci_lo', 'loo_min', 'changed', 'fixed', 'broken', 'better_exps', 'worse_exps', 'danger_under', 'onset6_late_exps']
    print(main_s[main_s.cadence.isin(['5s', '10/15s'])][cols].round(4).to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=8): main()
