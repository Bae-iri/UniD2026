"""4/5/6 specialist variants: increase reliance on time-since-drop / trends, reduce reliance on current absolute temperature.
Adoption rule (fixed before running): dev and val_noE5 must not drop more than 0.3 vs V0."""
import json
import numpy as np, pandas as pd
from tune import F, FEATS, MODELS, SENS, GAS, EVC, specialist, spec_proba, combine, PB, BR
from pair_experts import report
HIST = EVC + ['tsd_log', 't1_since_peak'] + [f'{c}_{k}' for c in ['meas_t1', 'meas_t2', 'thermal_t', 'meas_v', 'meas_p'] for k in ['cummax', 'd30', 'd120', 'd300']]
C = SENS + GAS + HIST
V = {'V0 robust (temp_dyn)': FEATS['temp_dyn'], 'V1 C (+pressure history)': C,
     'V2 C - indu_t,meas_t1': [c for c in C if c not in ('indu_t', 'meas_t1')],
     'V3 C - all current temps': [c for c in C if c not in ('indu_t', 'meas_t1', 'meas_t2', 'thermal_t', 'atmp_t')]}
rows = []; cache = {}
for name, cols in V.items():
    for split, frames in [('dev', [F['fit']]), ('validation', [F['fit'], F['dev']])]:
        out = combine(PB[split], spec_proba(specialist(frames, cols, MODELS['mid']), F[split][cols]), .7)
        r = report(split, out, name, '-'); r['n_feat'] = len(cols); rows.append(r)
r = pd.DataFrame(rows); r.to_csv('variants_results.csv', index=False)
for s in ['dev', 'validation']: print(r[r.split == s].drop(columns=['split', 'tau']).to_string(index=False), '\n')
base = r[r.exp.str.startswith('V0')].set_index('split')
ok = []
for name in V:
    x = r[r.exp == name].set_index('split')
    good = x.loc['dev', 'macro'] >= base.loc['dev', 'macro'] - .3 and x.loc['validation', 'macro_noE5'] >= base.loc['validation', 'macro_noE5'] - .3
    print(f'{name:28} adopt={good}  dev {x.loc["dev","macro"]:.2f} val {x.loc["validation","macro"]:.2f} noE5 {x.loc["validation","macro_noE5"]:.2f}'); ok.append((name, good))
# CSVs for every passing variant (trained on all labeled data)
o = pd.read_csv(BR + 'submission_B.csv'); allf = [F['fit'], F['dev'], F['validation']]
ref = None
for name, good in ok:
    if not good: continue
    cols = V[name]; p = combine(PB['test'], spec_proba(specialist(allf, cols, MODELS['mid']), F['test'][cols]), .7)
    out = o.copy(); out['label'] = pd.Series(p, index=F['test'].sample_id).loc[o.sample_id].values.astype(int)
    assert len(out) == 3451 and out.label.between(1, 6).all()
    fn = 'submission_B_spec456_' + name.split()[0] + '.csv'; out.to_csv(fn, index=False)
    if ref is None: ref = out.label.values
    print(fn, 'changed vs B', int((out.label != o.label).sum()), '| vs V0', int((out.label.values != ref).sum()))
