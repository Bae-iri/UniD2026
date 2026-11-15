"""마지막 후보(4→5 제거, 6→5 기준 0.6/0.5)의 배터리 조합 통째 제외 평가. correction_ablation.Fitted 재사용, Test 미사용."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from experiments.b_cadence_robust import run as C
from experiments.holdout_12_8 import correction_ablation as A
from experiments.holdout_12_8.b_holdout import pooled
P = A.P
out = ROOT/'experiments/holdout_12_8/final_candidates_v1'
with threadpool_limits(limits=8):
    tr, va, frame = pooled(); z = np.load(A.BPROBS)
    dev = frame.iloc[z['dev_index']].reset_index(drop=True); dev_p = z['dev_oof']
    dv = {c: C.view_data(dev, dev_p, C.EVAL_CADENCES[c]) for c in ('5s', '10/15s')}; full = dv['5s']
    combo = dev.bat_class.astype(int).astype(str)+'/'+dev.bat_cc.astype(str)
    rows = []
    for key in sorted(combo.unique()):
        held = (combo == key).to_numpy()
        f = A.Fitted(dev[~held].reset_index(drop=True), dev_p[:, ~held], full['y'][~held], full['base'][~held])
        for c, v in dv.items():
            m = np.isin(v['exps'], list(set(dev.experiment_id[held])))
            fr = v['frame'][m].reset_index(drop=True); base = v['base'][m]
            pr, _ = f.variants(fr, v['probs'][:, m], base)
            x = A.targeted_features(fr, v['probs'][:, m])[0]; g = P.score(f.gate['model'], x['56'])
            cands = {'S40': pr['S40'], '56+65': pr['56+65'], 'S40@0.6': pr['S41+65@0.6'], 'S40@0.5': pr['S41+65@0.5'],
                     '56+65@0.6': P.apply(pr['only56'], base, g, .6)}
            for n, p in cands.items():
                r = C.metrics(fr, v['y'][m], base, p)
                rows.append(dict(combo=key, cadence=c, method=n, d=round(r['d_vs_base']*100, 2), changed=r['changed'],
                                 fixed=r['fixed'], broken=r['broken'], danger_under=r['danger_under'], onset6_late=r['onset6_late_exps']))
        print(key, flush=True)
    t = pd.DataFrame(rows); t.to_csv(out/'leave_condition_out.csv', index=False)
    pd.set_option('display.width', 250)
    print(t.pivot_table(index='method', columns=['cadence', 'combo'], values='d', sort=False).to_string())
    print(t.groupby(['cadence', 'method'])[['d', 'changed', 'broken', 'danger_under', 'onset6_late']].agg(['mean', 'min']).round(2).to_string())
