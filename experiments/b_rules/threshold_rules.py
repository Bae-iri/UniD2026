"""Are stage boundaries defined by a temperature crossing a fixed value? For each transition, spread of each temperature
(and its running max) at the exact boundary row across experiments. Tight spread => threshold definition."""
import numpy as np, pandas as pd
ROOT = '/home/ubuntu/data/battery_data_light/'; E5 = 'E_e5bcddc7d77f459ea541cf658ab4346e'
x = pd.concat([pd.read_csv(ROOT + 'train/train.csv'), pd.read_csv(ROOT + 'validation/validation.csv')])
x = x.sort_values(['experiment_id', 'relative_time_s'])
T = ['meas_t1', 'meas_t2', 'thermal_t', 'indu_t', 'atmp_t', 'meas_t1_plus_t2']
for c in T: x[c + '_max'] = x.groupby('experiment_id')[c].cummax()
rows = []
for e, g in x.groupby('experiment_id'):
    g = g.reset_index(drop=True); L = g.label.values
    for a in range(1, 6):
        k = np.where((L[:-1] <= a) & (L[1:] > a))[0]
        if not len(k): continue
        k = k[0] + 1
        rows.append(dict(exp=e, trans=f'{a}>{L[k]}', bat_cc=g.bat_cc[0], tr=g.tr_method[0], **{c: g[c][k] for c in T + [t + '_max' for t in T]}, **{c + '_prev': g[c][k - 1] for c in T}))
r = pd.DataFrame(rows)
out = []
for tr_, d in r.groupby('trans'):
    for c in T + [t + '_max' for t in T]:
        q = d[c].quantile([.1, .25, .5, .75, .9]).values
        out.append(dict(trans=tr_, n=len(d), feature=c, p10=q[0], p25=q[1], median=q[2], p75=q[3], p90=q[4], rel_IQR=(q[3] - q[1]) / (abs(q[2]) + 1e-9)))
o = pd.DataFrame(out).round(2); o.to_csv('threshold_spread.csv', index=False)
for tr_, d in o.groupby('trans'):
    print(f'\n{tr_} (n={d.n.iloc[0]}) - tightest features:'); print(d.sort_values('rel_IQR').head(4)[['feature', 'p10', 'p25', 'median', 'p75', 'p90', 'rel_IQR']].to_string(index=False))
r.to_csv('transition_values.csv', index=False)
