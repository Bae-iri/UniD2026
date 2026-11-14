"""S4: 4/5/6 tree specialist (robust: temp_dyn, mid, w=0.7) + 3<->5 pairwise expert (tau=0.9)."""
import json, sys
import numpy as np, pandas as pd
from tune import F, FEATS, MODELS, specialist, spec_proba, combine, PB, BR, te
from pair_experts import expert, apply, report, COLS
cols = FEATS['temp_dyn']
def s4(split, spec_frames, exp_frames):
    base = PB[split].argmax(1) + 1
    s456 = combine(PB[split], spec_proba(specialist(spec_frames, cols, MODELS['mid']), F[split][cols]), .7)
    out = apply(base, F[split][COLS], {(3, 5): expert(exp_frames, 3, 5)}, .9)   # 3<->5 decided on B's own prediction
    flip35 = out != base
    final = s456.copy(); final[flip35] = out[flip35]                               # 3<->5 flips override; rest from 4/5/6 specialist
    return final, s456
rows = []
for split, frames in [('dev', [F['fit']]), ('validation', [F['fit'], F['dev']])]:
    fin, s456 = s4(split, frames, frames)
    rows += [report(split, PB[split].argmax(1) + 1, 'S0 B', '-'), report(split, s456, '4/5/6 specialist', '-'), report(split, fin, 'S4 456spec + 3v5', .9)]
r = pd.DataFrame(rows); print(r.to_string(index=False)); r.to_csv('s4_results.csv', index=False)
# final test CSV: everything trained on fit+dev+validation
allf = [F['fit'], F['dev'], F['validation']]; fin, _ = s4('test', allf, allf)
o = pd.read_csv(BR + 'submission_B.csv'); out = o.copy()
out['label'] = pd.Series(fin, index=F['test'].sample_id).loc[o.sample_id].values.astype(int)
assert len(out) == 3451 and out.sample_id.is_unique and out.label.between(1, 6).all()
out.to_csv('submission_B_spec456_plus_3v5.csv', index=False); ch = (out.label != o.label).values
print('test changed', ch.sum(), dict(pd.Series([f'{a}>{b}' for a, b in zip(o.label[ch], out.label[ch])]).value_counts()))
