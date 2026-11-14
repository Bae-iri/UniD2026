"""V3 + battery-specific stage-4 duration rule.
For each bat_cc, T4[bat_cc] = max post-drop seconds labelled 4 in the TRAINING frames (+ one 15 s step margin).
If final pred is 4 and time_since_drop > T4[bat_cc] -> 5. Unknown capacity -> no rule."""
import numpy as np, pandas as pd
from tune import F, MODELS, specialist, spec_proba, combine, PB, BR
from pair_experts import report
from variants import V
cols = V['V3 C - all current temps']

def t4(frames):
    d = pd.concat(frames); d = d[(d.label == 4) & (d.drop_occurred > 0)]
    return (d.groupby('bat_cc').time_since_drop.max() + 15).to_dict()

def rule(split, pred, T):
    f = F[split]; lim = f.bat_cc.map(T).values
    out = pred.copy(); m = (pred == 4) & (f.drop_occurred.values > 0) & ~np.isnan(lim) & (f.time_since_drop.values > lim)
    out[m] = 5; return out

rows = []
for split, frames in [('dev', [F['fit']]), ('validation', [F['fit'], F['dev']])]:
    T = t4(frames); print(split, 'T4 by capacity:', {k: int(v) for k, v in T.items()})
    v3 = combine(PB[split], spec_proba(specialist(frames, cols, MODELS['mid']), F[split][cols]), .7)
    rows += [report(split, PB[split].argmax(1) + 1, 'S0 B', '-'), report(split, v3, 'V3', '-'), report(split, rule(split, v3, T), 'V3 + battery 4-duration', '-')]
r = pd.DataFrame(rows).drop(columns='tau'); r.to_csv('battery_duration_results.csv', index=False)
for s in ['dev', 'validation']: print(r[r.split == s].drop(columns='split').to_string(index=False), '\n')
allf = [F['fit'], F['dev'], F['validation']]; T = t4(allf); print('final T4:', {k: int(v) for k, v in T.items()})
p = combine(PB['test'], spec_proba(specialist(allf, cols, MODELS['mid']), F['test'][cols]), .7); q = rule('test', p, T)
o = pd.read_csv(BR + 'submission_B.csv'); s = pd.Series(q, index=F['test'].sample_id).loc[o.sample_id].values.astype(int)
out = o.copy(); out['label'] = s; assert len(out) == 3451 and out.label.between(1, 6).all(); out.to_csv('submission_B_spec456_V3_battery4dur.csv', index=False)
v3 = pd.read_csv('submission_B_spec456_V3.csv').label.values; ex = F['test'].set_index('sample_id').loc[o.sample_id].experiment_id.str[:8].values
print('test: vs B', int((s != o.label.values).sum()), '| vs V3', int((s != v3).sum()), '| changed rows by experiment', pd.Series(ex[s != v3]).value_counts().to_dict())
