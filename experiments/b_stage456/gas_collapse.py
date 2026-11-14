"""Is 'chemical gases all exactly 0 now, after having been non-zero earlier' a stage-5 signal outside E_e5bc too?"""
import numpy as np, pandas as pd
from tune import F, PB
CHEM = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
def add(d):
    d = d.copy(); z = (d[CHEM].fillna(-1) == 0).all(1)
    d['gas_zero'] = z.astype(int)
    d['gas_seen'] = d.sort_values('relative_time_s').groupby('experiment_id')[CHEM].cummax().reindex(d.index).sum(1) > 0
    d['gas_collapse'] = (z & d.gas_seen).astype(int)        # zero now, non-zero before (causal)
    return d
for s in ['fit', 'dev', 'validation', 'test']:
    d = add(F[s]); F[s] = d
    if 'label' in d:
        e5 = d.experiment_id.str.startswith('E_e5bc')
        for tag, m in [('all', np.ones(len(d), bool)), ('excl E5', ~e5.values)] if s == 'validation' else [('all', np.ones(len(d), bool))]:
            t = pd.crosstab(d.label[m], d.gas_collapse[m]).reindex(columns=[0, 1], fill_value=0)
            print(f'{s:10} {tag:8} gas_collapse rows by label:', t[1].to_dict(), '| experiments with collapse:', d[m & (d.gas_collapse == 1).values].experiment_id.nunique())
    else:
        print('test      gas_collapse rows:', int(d.gas_collapse.sum()), 'experiments', d[d.gas_collapse == 1].experiment_id.str[:8].value_counts().to_dict(),
              '| B pred on them', pd.Series(PB['test'].argmax(1)[d.gas_collapse.values == 1] + 1).value_counts().to_dict())
v = F['validation']; pb = PB['validation'].argmax(1) + 1
w = v.experiment_id.str.startswith('E_e5bc').values & (v.label.values == 5) & (pb == 4)
print('\nE5 misread 5->4 rows covered by gas_collapse:', int(v.gas_collapse.values[w].sum()), '/', int(w.sum()))
