"""V3 specialist: reduce B's grip on the 4/5/6 decision.
mix = w * specialist + (1-w) * normalize(B456 ** t)   (t<1 flattens B's confidence). Rule: dev & val_noE5 >= V3(w=.7,t=1) - 0.3."""
import numpy as np, pandas as pd
from tune import F, MODELS, specialist, spec_proba, PB, BR
from pair_experts import report
from variants import V
cols = V['V3 C - all current temps']

def mix(Pb, ps, w, t):
    pred = Pb.argmax(1) + 1; m = pred >= 4
    b = Pb[m, 3:] ** t; b /= b.sum(1, keepdims=True)
    out = pred.copy(); out[m] = (w * ps[m] + (1 - w) * b).argmax(1) + 4; return out

PS = {'dev': spec_proba(specialist([F['fit']], cols, MODELS['mid']), F['dev'][cols]),
      'validation': spec_proba(specialist([F['fit'], F['dev']], cols, MODELS['mid']), F['validation'][cols])}
rows = []
for w in [.7, .8, .9, 1.]:
    for t in [1., .5, .25]:
        if w == 1. and t != 1.: continue
        for s in ['dev', 'validation']:
            r = report(s, mix(PB[s], PS[s], w, t), f'w={w} t={t}', '-'); rows.append(r)
r = pd.DataFrame(rows).drop(columns='tau')
d = r[r.split == 'dev'].set_index('exp'); v = r[r.split == 'validation'].set_index('exp')
t = pd.DataFrame(dict(dev=d.macro, val=v.macro, val_noE5=v.macro_noE5, val_5to4=v.t5p4, val_4to5=v.t4p5, dev_5to4=d.t5p4, dev_4to5=d.t4p5))
ref = t.loc['w=0.7 t=1.0']; t['adopt'] = (t.dev >= ref.dev - .3) & (t.val_noE5 >= ref.val_noE5 - .3)
print(t.to_string()); t.to_csv('soften_b_results.csv')
# test CSVs for adopted settings (specialist trained on all labeled data)
allf = [F['fit'], F['dev'], F['validation']]; pst = spec_proba(specialist(allf, cols, MODELS['mid']), F['test'][cols])
o = pd.read_csv(BR + 'submission_B.csv'); v3 = pd.read_csv('submission_B_spec456_V3.csv').label.values
for name, row in t[t.adopt].iterrows():
    w, tt = float(name.split()[0][2:]), float(name.split()[1][2:])
    p = pd.Series(mix(PB['test'], pst, w, tt), index=F['test'].sample_id).loc[o.sample_id].values.astype(int)
    fn = f'submission_B_spec456_V3_w{w}_t{tt}.csv'; out = o.copy(); out['label'] = p; out.to_csv(fn, index=False)
    print(fn, 'changed vs B', int((p != o.label.values).sum()), '| vs V3', int((p != v3).sum()))
