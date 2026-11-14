"""Find features where E_e5bc's true-5 rows (B says 4) look like stage 5, and that also separate 4 vs 5 elsewhere.
Per feature: best single threshold on TRAIN 4 vs 5 (balanced acc), then check it on dev, validation-excl-E5, and E5's misread rows."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from tune import F, PB
skip = {'sample_id', 'experiment_id', 'relative_time_s', 'label', 'image_path', 'bat_class', 'bat_cc', 'soc_status', 'tr_method', 'meas_a', 'indu_t.1', 'meas_t1_plus_t2'}
cols = [c for c in F['fit'].columns if c not in skip and pd.api.types.is_numeric_dtype(F['fit'][c])]
tr = F['fit'][F['fit'].label.isin([4, 5])]
dv = F['dev'][F['dev'].label.isin([4, 5])]
va = F['validation']; pb = PB['validation'].argmax(1) + 1
e5 = va.experiment_id.str.startswith('E_e5bc').values
wrong = va[e5 & (va.label.values == 5) & (pb == 4)]                     # the rows we must fix
vn = va[~e5 & va.label.isin([4, 5]).values]
rows = []
for c in cols:
    x = tr[c].replace([np.inf, -np.inf], np.nan); y = (tr.label == 5).values; ok = x.notna().values
    if ok.sum() < 100 or x[ok].nunique() < 3: continue
    auc = roc_auc_score(y[ok], x[ok]); sign = 1 if auc >= .5 else -1
    qs = np.unique(np.nanpercentile(x[ok], np.linspace(1, 99, 99)))
    best = max(qs, key=lambda q: ((sign * x[ok] > sign * q)[y[ok]].mean() + (sign * x[ok] <= sign * q)[~y[ok]].mean()))
    rule = lambda d: (sign * d[c] > sign * best).values                 # True = "looks like 5"
    def bal(d): yy = (d.label == 5).values; r = rule(d); return (r[yy].mean() + (~r[~yy]).mean()) / 2
    rows.append(dict(feature=c, train_AUC=round(max(auc, 1 - auc), 3), rule=f'{c} {">" if sign > 0 else "<="} {best:.4g}',
                     train_balacc=round(bal(tr), 3), dev_balacc=round(bal(dv), 3), val_noE5_balacc=round(bal(vn), 3),
                     E5_wrong_says5=round(rule(wrong).mean(), 3),
                     med4=round(tr[tr.label == 4][c].median(), 3), med5=round(tr[tr.label == 5][c].median(), 3), medE5wrong=round(wrong[c].median(), 3)))
r = pd.DataFrame(rows)
r['score'] = r[['dev_balacc', 'val_noE5_balacc', 'E5_wrong_says5']].min(1)   # must work everywhere
r = r.sort_values('score', ascending=False); r.to_csv('find_diff_4v5.csv', index=False)
print('rows to fix (E_e5bc true5, B says 4):', len(wrong))
print(r.head(15).to_string(index=False))
print('\nE_e5bc misread rows - what do they look like vs train 4 / train 5 (medians):')
for c in ['meas_v', 'meas_v_cummax', 'time_since_drop', 'meas_t1', 'meas_t1_cummax', 'meas_t1_d300', 'thermal_t', 'meas_p', 'meas_co', 'meas_co2', 'meas_co_cummax', 'meas_co2_cummax', 'meas_hf', 'meas_hcl']:
    if c in tr: print(f'  {c:18} train4 {tr[tr.label==4][c].median():9.3f}  train5 {tr[tr.label==5][c].median():9.3f}  E5wrong {wrong[c].median():9.3f}')
