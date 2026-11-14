"""Rule 1: stage>=2 iff running-max T1 >= 45 (1->2 boundary is T1 = 45.0-45.4 C).
Rule 2: no voltage-drop event yet -> stage <= 3.  Applied to ORIGINAL B predictions only."""
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from b_errors import D, E5, ROOT
r = pd.read_csv('transition_values.csv')
print('== 2->3 boundary T1 by battery (median [p25-p75]):')
print(r[r.trans == '2>3'].groupby('bat_cc').meas_t1.describe()[['count', '25%', '50%', '75%']].round(1).to_string())

def add_max(d):
    d = d.sort_values(['experiment_id', 'relative_time_s']); d['t1max'] = d.groupby('experiment_id').meas_t1.cummax(); return d.sort_index()
def apply(d, r1=True, r2=True):
    p = d.B.values.copy()
    if r1:
        hot = d.t1max.values >= 45
        p[(p == 1) & hot] = 2; p[(p >= 2) & ~hot & ~np.isnan(d.t1max.values)] = 1
    if r2: p[(p >= 4) & (d.drop_occurred.values == 0)] = 3
    return p
tr = add_max(pd.read_csv(ROOT + 'train/train.csv').merge(pd.read_csv('/home/ubuntu/gahyun/experiments/b_independent_studies/runs/independent_seed42_43_44_v1/experiment1_voltage/fit_voltage_event_columns.csv')[['sample_id', 'drop_occurred']], on='sample_id'))
print('\n== rule truth on train (fit rows): label>=2 when t1max>=45:', round(((tr.label >= 2) == (tr.t1max >= 45)).mean(), 4),
      '| label 1 with t1max>=45:', int(((tr.label == 1) & (tr.t1max >= 45)).sum()), '| label>=2 with t1max<45:', int(((tr.label >= 2) & (tr.t1max < 45)).sum()),
      '| label>=4 without drop:', int(((tr.label >= 4) & (tr.drop_occurred == 0)).sum()), '/', int((tr.label >= 4).sum()))
rows = []
for s in ['dev', 'validation']:
    d = add_max(D[s]); 
    for scope, m in [('all', np.ones(len(d), bool)), ('excl E_e5bc', (d.experiment_id != E5).values)]:
        if s == 'dev' and scope != 'all': continue
        y = d.label.values[m]
        for tag, p in [('B', d.B.values), ('B + rule1 (T1 45)', apply(d, True, False)), ('B + rule2 (drop)', apply(d, False, True)), ('B + both', apply(d))]:
            p = p[m]; f = f1_score(y, p, labels=range(1, 7), average=None)
            rows.append(dict(split=s, scope=scope, model=tag, macro=round(100 * f.mean(), 2), F1_1=round(100 * f[0], 1), F1_2=round(100 * f[1], 1), F1_3=round(100 * f[2], 1), F1_4=round(100 * f[3], 1),
                             changed=int((p != d.B.values[m]).sum()), fixed=int(((p != d.B.values[m]) & (p == y)).sum()), broken=int(((p != d.B.values[m]) & (d.B.values[m] == y)).sum())))
print(); print(pd.DataFrame(rows).to_string(index=False))
te = add_max(D['test'])
for tag, p in [('rule1', apply(te, True, False)), ('rule2', apply(te, False, True)), ('both', apply(te))]:
    ch = p != te.B.values; print(f'test {tag}: changed {ch.sum()}', dict(pd.Series([f"{a}>{b}" for a, b in zip(te.B.values[ch], p[ch])]).value_counts()), dict(te.experiment_id[ch].str[:8].value_counts()))
