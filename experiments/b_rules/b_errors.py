"""Where does ORIGINAL B err most on data it never trained on (dev, validation excl E_e5bc)? Which boundaries have a sharp observable marker?"""
import numpy as np, pandas as pd
ROOT = '/home/ubuntu/data/battery_data_light/'; BR = '/home/ubuntu/leewon/abc_runs/current_query_v1/'
EV = '/home/ubuntu/gahyun/experiments/b_independent_studies/runs/independent_seed42_43_44_v1/experiment1_voltage/'
E5 = 'E_e5bcddc7d77f459ea541cf658ab4346e'
src = {'dev': pd.read_csv(ROOT + 'train/train.csv'), 'validation': pd.read_csv(ROOT + 'validation/validation.csv'), 'test': pd.read_csv(ROOT + 'test/test.csv')}
def load(s):
    p = pd.read_csv(BR + f'ensemble_B_{s}_predictions.csv')[['sample_id'] + [f'p{i}' for i in range(1, 7)]]
    d = p.merge(src[s], on='sample_id').merge(pd.read_csv(EV + f'{s}_voltage_event_columns.csv')[['sample_id', 'drop_occurred', 'time_since_drop']], on='sample_id')
    d['B'] = d[[f'p{i}' for i in range(1, 7)]].values.argmax(1) + 1; return d
D = {s: load(s) for s in ['dev', 'validation', 'test']}
print('== 1. original B errors by (true -> pred), excluding E_e5bc')
tab = []
for s in ['dev', 'validation']:
    d = D[s]; d = d[d.experiment_id != E5]; w = d[d.label != d.B]
    c = w.groupby(['label', 'B']).size().rename(s); tab.append(c)
t = pd.concat(tab, axis=1).fillna(0).astype(int); t['total'] = t.sum(1); print(t.sort_values('total', ascending=False).head(10).to_string())
print('\n== 2. is the 3|4 boundary explained by the voltage-drop event? (dev + validation excl E_e5bc)')
d = pd.concat([D['dev'], D['validation'][D['validation'].experiment_id != E5]])
for (a, b) in [(3, 4), (4, 3), (3, 5), (2, 3), (3, 2)]:
    w = d[(d.label == a) & (d.B == b)]
    print(f'  true{a}->pred{b}: {len(w):4d} rows | drop already happened: {int(w.drop_occurred.sum()):4d} | no drop yet: {int((w.drop_occurred == 0).sum()):4d}')
print('  (reference) all true4 rows with drop:', round(d[d.label == 4].drop_occurred.mean(), 3), '| all true3 rows with drop:', round(d[d.label == 3].drop_occurred.mean(), 3))
print('\n== 3. 1->2 and 2->3 transitions: which sensor jumps exactly at the boundary? (+-120 s window, share of experiments)')
FE = ['meas_v', 'meas_t1', 'meas_t2', 'thermal_t', 'indu_t', 'meas_p', 'meas_co', 'meas_co2', 'meas_hf', 'meas_ch4']
x = pd.concat([pd.read_csv(ROOT + 'train/train.csv'), src['validation']]); x = x[x.experiment_id != E5].sort_values(['experiment_id', 'relative_time_s'])
rows = []
for e, g in x.groupby('experiment_id'):
    g = g.reset_index(drop=True); L = g.label.values; tt = g.relative_time_s.values
    for a in [1, 2]:
        k = np.where((L[:-1] == a) & (L[1:] == a + 1))[0]
        if not len(k): continue
        k = k[0] + 1; w = np.where(np.abs(tt - tt[k]) <= 120)[0]; rec = dict(trans=f'{a}>{a+1}')
        for c in FE:
            v = g[c].values.astype(float); dv = np.abs(np.diff(v, prepend=v[0]))
            rec[c] = int(abs(w[np.nanargmax(dv[w])] - k) <= 1) if np.nanmax(dv[w]) > 0 else np.nan
        rec['t1_at'] = g.meas_t1[k]; rec['t1_before'] = g.meas_t1[k - 1]; rows.append(rec)
r = pd.DataFrame(rows); print(r.groupby('trans')[FE].mean().round(2).T.to_string())
print('  T1 at transition (median / IQR):', r.groupby('trans').t1_at.describe()[['25%', '50%', '75%']].round(1).to_string())
print('\n== 4. test: B predictions that contradict the drop event')
te = D['test']
print('  pred>=4 with NO drop yet:', int(((te.B >= 4) & (te.drop_occurred == 0)).sum()), '| pred<=3 AFTER drop:', int(((te.B <= 3) & (te.drop_occurred > 0)).sum()))
