"""Before building 3<->5 / 4<->5 experts: is there anything to fix outside E_e5bc, and are the classes separable?"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from tune import F, FEATS, PB
BAD = 'E_e5bc'
print('== 1. original B confusions (B never trained on dev/validation)')
for s in ['dev', 'validation']:
    y = F[s].label.values; p = PB[s].argmax(1) + 1; e5 = F[s].experiment_id.str.startswith(BAD).values
    for a, b in [(5, 3), (5, 4), (3, 5), (4, 5), (3, 4), (4, 3)]:
        m = (y == a) & (p == b); print(f'  {s:10} true{a}->pred{b}: {m.sum():4d}  (in E_e5bc {(m & e5).sum():4d}, elsewhere {(m & ~e5).sum():3d})')
    print(f'  {s:10} support: true3 {(y==3).sum()} true4 {(y==4).sum()} true5 {(y==5).sum()} | elsewhere true5 {((y==5)&~e5).sum()}')
print('\n== 2. separability, tabular causal features (temp_dyn)')
cols = FEATS['temp_dyn']; tr = pd.concat([F['fit'], F['dev']])
for a, b in [(3, 5), (4, 5)]:
    d = tr[tr.label.isin([a, b])]; yy = (d.label == b).values; oof = np.zeros(len(d))
    for i, j in GroupKFold(5).split(d, groups=d.experiment_id):
        oof[j] = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, random_state=0, class_weight='balanced').fit(d[cols].iloc[i], yy[i]).predict_proba(d[cols].iloc[j])[:, 1]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, random_state=0, class_weight='balanced').fit(d[cols], yy)
    v = F['validation']; vm = v.label.isin([a, b]).values; ve = v.experiment_id.str.startswith(BAD).values
    pv = m.predict_proba(v[cols])[:, 1]; yv = (v.label == b).values
    print(f'  {a} vs {b}: train group-CV AUC {roc_auc_score(yy, oof):.3f} | validation AUC {roc_auc_score(yv[vm], pv[vm]):.3f} | '
          f'val excl E_e5bc {roc_auc_score(yv[vm & ~ve], pv[vm & ~ve]):.3f} | E_e5bc only ' +
          (f'{roc_auc_score(yv[vm & ve], pv[vm & ve]):.3f}' if len(set(yv[vm & ve])) > 1 else 'n/a'))
    # B's own separability on the same rows (out-of-sample)
    Pb = PB['validation']; sb = Pb[:, b - 1] / (Pb[:, a - 1] + Pb[:, b - 1] + 1e-9)
    print(f'         original B validation AUC {roc_auc_score(yv[vm], sb[vm]):.3f} | excl E_e5bc {roc_auc_score(yv[vm & ~ve], sb[vm & ~ve]):.3f}')
