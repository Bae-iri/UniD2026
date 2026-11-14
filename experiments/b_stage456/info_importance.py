"""Permutation importance (AUC drop) for set C on validation 4/5 rows: all, excl E_e5bc, and recall on E_e5bc 5-rows."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from tune import F, SENS, GAS, EVC
HIST = EVC + ['tsd_log', 't1_since_peak'] + [f'{c}_{k}' for c in ['meas_t1', 'meas_t2', 'thermal_t', 'meas_v', 'meas_p'] for k in ['cummax', 'd30', 'd120', 'd300']]
cols = SENS + GAS + HIST
tr = pd.concat([F['fit'], F['dev']]); tr = tr[tr.label.isin([4, 5])]
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1., class_weight='balanced', random_state=0).fit(tr[cols], tr.label == 5)
v = F['validation']; v = v[v.label.isin([4, 5])]; y = (v.label == 5).values; e5 = v.experiment_id.str.startswith('E_e5bc').values
X = v[cols].copy(); base = m.predict_proba(X)[:, 1]
b_all, b_ne, b_e5 = roc_auc_score(y, base), roc_auc_score(y[~e5], base[~e5]), (base[e5] > .5).mean()
rng = np.random.default_rng(0); rows = []
for c in cols:
    d = []
    for _ in range(3):
        Xp = X.copy(); Xp[c] = rng.permutation(Xp[c].values); p = m.predict_proba(Xp)[:, 1]
        d.append((b_all - roc_auc_score(y, p), b_ne - roc_auc_score(y[~e5], p[~e5]), b_e5 - (p[e5] > .5).mean()))
    d = np.mean(d, 0); rows.append(dict(feature=c, AUC_drop_all=round(d[0], 4), AUC_drop_noE5=round(d[1], 4), E5_recall_drop=round(d[2], 3)))
r = pd.DataFrame(rows).sort_values('AUC_drop_all', ascending=False); r.to_csv('info_importance_4v5.csv', index=False)
print(f'base: AUC all {b_all:.3f} noE5 {b_ne:.3f} E5 5-row recall {b_e5:.3f}'); print(r.head(12).to_string(index=False))
