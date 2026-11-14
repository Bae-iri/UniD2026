"""제출 레시피 두 개(rule1base_targeted, rule1_targeted_precision65)의 과적합 점검과 보정 풀기.

같은 12+8 holdout 분할, 같은 B 재학습 확률(b_v1)을 쓴다.
  - 개발 77실험: 실험 단위 5-fold OOF(B도 그 fold의 OOF). 보정기는 나머지 fold로 학습, 기준값 자동 선택도 그 안에서만.
  - holdout 20실험: 개발 77실험 전체로 학습한 보정기로 1회 평가. Test는 읽지 않는다.

변형(모두 기반 = B + t1 45°C 위에서)
  자동 선택 기준(원래 레시피):
    S41      = 4→5 + 5→6                (submission_rule1base_targeted 레시피)
    S40      = 4→5 + 5→6 + 6→5 정밀 게이트 (submission_rule1_targeted_precision65 레시피)
    only56 / only45 / 56+65            보정 하나씩 빼기
  고정 기준(자동 선택 없이 풀기/조이기):
    56@t (t=.5,.6,.7,.8,.9)            5→6만, 기준 고정
    S41+65@t (t=.5,.6,.7,.9,.95,.99)          6→5 게이트 기준 고정
6→5 게이트는 gahyun rule1_precision65/precision.py를 읽기 전용으로 불러 같은 방식으로 다시 학습한다.

  python -u experiments/holdout_12_8/correction_ablation.py --output experiments/holdout_12_8/ablation_v1
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.b_cadence_robust import run as C
from experiments.holdout_12_8.b_holdout import pooled
from targeted_rule1.model import train as targeted_train, features as targeted_features, probability

spec = importlib.util.spec_from_file_location(
    'precision65', '/home/ubuntu/gahyun/experiments/rule1_precision65/precision.py')
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

BPROBS = ROOT/'experiments/holdout_12_8/b_v1/b_probs.npz'
E16 = C.E16
T56 = (.5, .6, .7, .8, .9)
T65 = (.5, .6, .7, .9, .95, .99)


def correct(base, s45, s56, t45, t56):
    p = np.asarray(base, int).copy()
    p[(base == 4) & (s45 >= t45)] = 5
    p[(base == 5) & (s56 >= t56)] = 6
    return p


class Fitted:
    def __init__(self, frame, probs, y, base):
        self.art = targeted_train(frame, probs, y, base)
        x = targeted_features(frame, probs)[0]
        s45 = probability(self.art['models']['45'], x['45']); s56 = probability(self.art['models']['56'], x['56'])
        th = self.art['thresholds']
        baseline = correct(base, s45, s56, th['45'], th['56'])   # gahyun 방식: 학습 행에 학습된 헤드를 적용한 기준
        self.gate = P.train(x['56'], y, baseline, base, frame.experiment_id.to_numpy())

    def variants(self, frame, probs, base):
        x = targeted_features(frame, probs)[0]
        s45 = probability(self.art['models']['45'], x['45']); s56 = probability(self.art['models']['56'], x['56'])
        g = P.score(self.gate['model'], x['56'])
        th = self.art['thresholds']; tg = self.gate['threshold']
        s41 = correct(base, s45, s56, th['45'], th['56'])
        out = {'base': base.copy(), 'S41': s41, 'S40': P.apply(s41, base, g, tg),
               'only56': correct(base, s45, s56, 9, th['56']), 'only45': correct(base, s45, s56, th['45'], 9)}
        out['56+65'] = P.apply(out['only56'], base, g, tg)
        for t in T56:
            out[f'56@{t}'] = correct(base, s45, s56, 9, t)
        for t in T65:
            out[f'S41+65@{t}'] = P.apply(s41, base, g, t)
        return out, dict(t45=th['45'], t56=th['56'], t65=tg)


def loo_min(frame, y, base, p):
    from targeted_rule1.model import matrix, f1
    ids = list(frame.groupby('experiment_id').indices.values())
    cb = np.stack([matrix(y[i], base[i]) for i in ids]); cp = np.stack([matrix(y[i], p[i]) for i in ids])
    return float(min(f1(cp.sum(0)-a).mean()-f1(cb.sum(0)-b).mean() for a, b in zip(cp, cb)))


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    tr, va, frame = pooled()
    z = np.load(BPROBS)
    assert np.array_equal(z['sample_id'], frame.sample_id.to_numpy(str))
    dev = frame.iloc[z['dev_index']].reset_index(drop=True); hold = frame.iloc[z['holdout_index']].reset_index(drop=True)
    dev_p, hold_p = z['dev_oof'], z['holdout']
    fold = z['fold'][z['dev_index']]
    dv = {c: C.view_data(dev, dev_p, v) for c, v in C.EVAL_CADENCES.items()}
    full = dv['5s']
    assert np.array_equal(full['idx'], np.arange(len(dev)))
    rows, thresholds = [], []
    oof = {c: {} for c in dv}
    for k in range(5):
        fit = fold != k
        held_exps = set(dev.experiment_id[fold == k])
        fitted = Fitted(dev[fit].reset_index(drop=True), dev_p[:, fit], full['y'][fit], full['base'][fit])
        for c, v in dv.items():
            m = np.isin(v['exps'], list(held_exps))
            pr, th = fitted.variants(v['frame'][m].reset_index(drop=True), v['probs'][:, m], v['base'][m])
            for name, p in pr.items():
                oof[c].setdefault(name, np.zeros(len(v['base']), int))[m] = p
        thresholds.append(dict(fold=k, **th))
        print('fold', k, th, flush=True)
    for c, v in dv.items():
        for name, p in oof[c].items():
            r = C.metrics(v['frame'], v['y'], v['base'], p)
            r['loo_min'] = loo_min(v['frame'], v['y'], v['base'], p)
            rows.append(dict(split='dev_oof', cadence=c, method=name, **r))

    final = Fitted(dev, dev_p, full['y'], full['base'])
    per_exp = []
    for c, view in C.EVAL_CADENCES.items():
        v = C.view_data(hold, hold_p, view)
        pr, th = final.variants(v['frame'], v['probs'], v['base'])
        for name, p in pr.items():
            for scope, m in [('all', np.ones(len(p), bool)), ('from_train12', v['frame'].original_split.eq('train').to_numpy()),
                             ('from_validation8', v['frame'].original_split.eq('validation').to_numpy()),
                             ('all_noE16', v['exps'] != E16)]:
                f = v['frame'][m].reset_index(drop=True)
                r = C.metrics(f, v['y'][m], v['base'][m], p[m])
                r['loo_min'] = loo_min(f, v['y'][m], v['base'][m], p[m])
                rows.append(dict(split=f'holdout_{scope}', cadence=c, method=name, **r))
            for e, i in v['frame'].groupby('experiment_id').indices.items():
                y = v['y'][i]
                per_exp.append(dict(cadence=c, method=name, experiment_id=e, source=v['frame'].original_split.iloc[i[0]],
                                    bat_cc=float(v['frame'].bat_cc.iloc[i[0]]), rows=len(i),
                                    f1_base=C.macro(y, v['base'][i]), f1=C.macro(y, p[i]),
                                    fixed=int(((v['base'][i] != y) & (p[i] == y)).sum()),
                                    broken=int(((v['base'][i] == y) & (p[i] != y)).sum())))
    thresholds.append(dict(fold='final', **th))
    pd.DataFrame(rows).to_csv(out/'summary.csv', index=False)
    pd.DataFrame(per_exp).to_csv(out/'holdout_per_experiment.csv', index=False)
    (out/'thresholds.json').write_text(json.dumps(thresholds, indent=2, default=float))
    t = pd.DataFrame(rows); t['d'] = (t.d_vs_base*100).round(2)
    pd.set_option('display.width', 250)
    print(t[t.cadence == '5s'].pivot_table(index='method', columns='split', values='d', sort=False).to_string())
    print(t[t.split == 'holdout_all'].pivot_table(index='method', columns='cadence', values='d', sort=False).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    with threadpool_limits(limits=8):
        main(ap.parse_args().output)
