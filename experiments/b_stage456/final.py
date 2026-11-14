"""Final: specialist trained on ALL labeled 4/5/6 rows (fit+dev+validation), applied on top of original B test probabilities."""
import json
import numpy as np, pandas as pd
from tune import F, FEATS, MODELS, specialist, spec_proba, combine, PB, BR, te
o = pd.read_csv(BR + 'submission_B.csv'); ex = te.set_index('sample_id').loc[o.sample_id].experiment_id.str[:8].values
for tag, fname, mname, w in [('maxval', 'temp_dyn', 'mid', 1.0), ('robust', 'temp_dyn', 'mid', 0.7)]:
    cols = FEATS[fname]; S = specialist([F['fit'], F['dev'], F['validation']], cols, MODELS[mname])
    pred = pd.Series(combine(PB['test'], spec_proba(S, F['test'][cols]), w), index=F['test'].sample_id).loc[o.sample_id].values
    out = o.copy(); out['label'] = pred.astype(int)
    assert len(out) == 3451 and out.sample_id.is_unique and out.label.between(1, 6).all() and (out.sample_id == o.sample_id).all()
    fn = f'submission_B_spec456_{tag}.csv'; out.to_csv(fn, index=False); ch = (out.label != o.label).values
    json.dump(dict(features=fname, model=mname, specialist_weight=w, bags=5, trained_on='fit+dev+validation rows with label 4-6',
                   base='original leewon B 3-seed ensemble probabilities (unchanged)', changed_vs_original_B=int(ch.sum())), open(fn.replace('.csv', '.json'), 'w'), indent=1)
    print(fn, 'changed', ch.sum(), dict(pd.Series([f'{a}>{b}' for a, b in zip(o.label[ch], out.label[ch])]).value_counts()), dict(pd.Series(ex[ch]).value_counts()))
