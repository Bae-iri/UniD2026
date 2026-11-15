"""Nested CV 평가. 바깥 fold마다
  1) 개발 실험(나머지)만으로: 보정 학습·기준값 자동 선택·하위 보정 채택 규칙(lower_heads/evaluate.py와 같은 규칙:
     안쪽 OOF 4개 간격 모두 이득 + 안쪽 배터리 조합 제외 평균 5s·10/15s 이득)
  2) 개발 전체로 최종 적합 → 바깥 fold 실험 예측
을 하고, 97실험의 바깥 fold 예측을 모아 평가한다. 바깥 fold는 학습·선택에 한 번도 쓰이지 않는다. Test 미사용.
비교 방법(모두 같은 절차로 fold마다 다시 학습): base(B+t1), S40(4→5+5→6+6→5), 56_65(5→6+6→5),
  final_fixed(56_65 + 4→3, 제출 구성), final_rule(56_65 + fold 안 채택 규칙이 고른 하위 보정).
  python -u experiments/nested_cv/evaluate.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from experiments.b_cadence_robust import run as C
from experiments.holdout_12_8 import correction_ablation as A
from experiments.holdout_12_8.b_holdout import pooled
from experiments.lower_heads import evaluate as E
B = ROOT/'experiments/nested_cv/b_v1'; OUT = ROOT/'experiments/nested_cv/eval_v1'
PAIRS = E.PAIRS


def predict(fit, f, p, base, kept):
    pr, _ = fit.predict(f, p, base)
    s40 = fit.upper.variants(f, p, base)[0]['S40']
    return {'base': pr['base'], 'S40': s40, '56_65': pr['final'], 'final_fixed': pr['final+43'],
            'final_rule': pr['final+'+'+'.join(kept)] if kept else pr['final']}


def select(dev, dev_p, fold):
    dv = {c: C.view_data(dev, dev_p, v) for c, v in C.EVAL_CADENCES.items()}; full = dv['5s']
    oof = {c: {} for c in dv}; rows = []
    for k in range(5):
        fit = fold != k; m0 = E.Fit(dev[fit].reset_index(drop=True), dev_p[:, fit], full['y'][fit], full['base'][fit])
        held = set(dev.experiment_id[fold == k])
        for c, v in dv.items():
            m = np.isin(v['exps'], list(held)); pr, _ = m0.predict(v['frame'][m].reset_index(drop=True), v['probs'][:, m], v['base'][m])
            for n, p in pr.items(): oof[c].setdefault(n, np.zeros(len(v['base']), int))[m] = p
    for c, v in dv.items():
        for n, p in oof[c].items(): rows.append(dict(split='dev', cadence=c, method=n, d=C.metrics(v['frame'], v['y'], v['base'], p)['d_vs_base']))
    combo = dev.bat_class.astype(int).astype(str)+'/'+dev.bat_cc.astype(str)
    for key in sorted(combo.unique()):
        held = (combo == key).to_numpy()
        if held.all() or (~held).sum() == 0: continue
        m0 = E.Fit(dev[~held].reset_index(drop=True), dev_p[:, ~held], full['y'][~held], full['base'][~held])
        for c in ('5s', '10/15s'):
            v = dv[c]; m = np.isin(v['exps'], list(set(dev.experiment_id[held]))); fr = v['frame'][m].reset_index(drop=True)
            pr, _ = m0.predict(fr, v['probs'][:, m], v['base'][m])
            for n, p in pr.items(): rows.append(dict(split='lco', cadence=c, method=n, d=C.metrics(fr, v['y'][m], v['base'][m], p)['d_vs_base']))
    t = pd.DataFrame(rows)
    d = t[t.split == 'dev'].pivot_table(index='method', columns='cadence', values='d')
    l = t[t.split == 'lco'].groupby(['method', 'cadence']).d.mean().unstack()
    kept = [p for p in PAIRS if (d.loc[f'final+{p}'] > d.loc['final']).all() and (l.loc[f'final+{p}'] > l.loc['final']).all()]
    return kept, (d.sub(d.loc['final'], axis=1)*100).round(3), (l.sub(l.loc['final'], axis=1)*100).round(3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, frame = pooled()
    preds = {c: {} for c in C.EVAL_CADENCES}; views = {}; info = []
    for o in range(5):
        z = np.load(B/f'outer{o}.npz'); assert np.array_equal(z['sample_id'], frame.sample_id.to_numpy(str))
        dev = frame.iloc[z['dev_index']].reset_index(drop=True); hold = frame.iloc[z['hold_index']].reset_index(drop=True)
        assert not set(dev.experiment_id) & set(hold.experiment_id)
        kept, dgain, lgain = select(dev, z['dev_oof'], z['dev_fold'])
        full = C.view_data(dev, z['dev_oof'], C.EVAL_CADENCES['5s'])
        fit = E.Fit(dev, z['dev_oof'], full['y'], full['base'])
        info.append(dict(outer=o, hold_exps=int(hold.experiment_id.nunique()), kept=kept, thresholds=fit.lower['thresholds'],
                         upper_thresholds=dict(fit.upper.art['thresholds'], t65=fit.upper.gate['threshold']),
                         inner_dev_gain=dgain.loc[[f'final+{p}' for p in PAIRS]].to_dict('index'),
                         inner_lco_gain=lgain.loc[[f'final+{p}' for p in PAIRS]].to_dict('index')))
        print(json.dumps({k: info[-1][k] for k in ('outer', 'kept', 'thresholds', 'upper_thresholds')}, default=float), flush=True)
        for c, view in C.EVAL_CADENCES.items():
            v = C.view_data(hold, z['hold'], view)
            views.setdefault(c, []).append(dict(frame=v['frame'].assign(outer=o), y=v['y'], base=v['base']))
            for n, p in predict(fit, v['frame'], v['probs'], v['base'], kept).items():
                preds[c].setdefault(n, []).append(p)
    (OUT/'folds.json').write_text(json.dumps(info, indent=1, default=float))
    rows, per = [], []
    for c in C.EVAL_CADENCES:
        f = pd.concat([x['frame'] for x in views[c]], ignore_index=True)
        y = np.concatenate([x['y'] for x in views[c]]); base = np.concatenate([x['base'] for x in views[c]])
        for n, ps in preds[c].items():
            p = np.concatenate(ps)
            for scope, m in [('all97', np.ones(len(y), bool)), ('noE16', f.experiment_id.to_numpy() != C.E16),
                             ('from_train', f.original_split.eq('train').to_numpy()), ('from_validation', f.original_split.eq('validation').to_numpy())]:
                ff = f[m].reset_index(drop=True); r = C.metrics(ff, y[m], base[m], p[m]); r['loo_min'] = A.loo_min(ff, y[m], base[m], p[m])
                rows.append(dict(scope=scope, cadence=c, method=n, **r))
            for o in range(5):
                m = f.outer.to_numpy() == o
                per.append(dict(outer=o, cadence=c, method=n, macro_f1=C.macro(y[m], p[m]), base_f1=C.macro(y[m], base[m])))
    t = pd.DataFrame(rows); t.to_csv(OUT/'summary.csv', index=False); pd.DataFrame(per).to_csv(OUT/'per_outer_fold.csv', index=False)
    pd.set_option('display.width', 250)
    order = ['base', 'S40', '56_65', 'final_fixed', 'final_rule']
    print(t.pivot_table(index='method', columns=['scope', 'cadence'], values='macro_f1').loc[order].round(4).to_string())
    cols = ['scope', 'cadence', 'method', 'macro_f1', 'd_vs_base', 'ci_lo', 'loo_min', 'fixed', 'broken', 'better_exps', 'worse_exps', 'danger_under', 'onset6_late_exps']
    print(t[t.cadence.isin(['5s', '10/15s']) & t.scope.isin(['all97', 'noE16'])][cols].round(4).to_string(index=False))
    pf = pd.DataFrame(per); pf = pf[pf.cadence == '5s'].pivot_table(index='method', columns='outer', values='macro_f1').loc[order]
    print(pf.round(4).to_string())


if __name__ == '__main__':
    with threadpool_limits(limits=8): main()
