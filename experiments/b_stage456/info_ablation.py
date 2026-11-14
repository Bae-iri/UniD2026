"""Which information separates true stage 4 from true stage 5?  Same small boosted tree, different feature groups.
A image only | B current sensors+gas | C B + history (temp/pressure/voltage dynamics, drop memory) | D A + C"""
import json
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from tune import F, SENS, GAS, EVC
MAN = json.load(open('/home/ubuntu/gahyun/experiments/b_more_data/runs/pool_fit/image_cache_manifest.json'))

def image_feats(split):
    im = np.load(MAN[split], mmap_mode='r'); out = []
    for i in range(0, len(im), 2048):
        x = im[i:i + 2048, :3].astype(np.float32) / 255; n = len(x)
        pooled = x.reshape(n, 3, 8, 16, 8, 16).mean((3, 5)).reshape(n, -1)
        flat = x.reshape(n, 3, -1)
        stats = np.concatenate([flat.mean(2), flat.std(2), np.percentile(flat, 90, axis=2), flat.max(2)], 1)
        out.append(np.concatenate([pooled, stats], 1))
    return pd.DataFrame(np.concatenate(out), columns=[f'img{i}' for i in range(204)])

for s in ['fit', 'dev', 'validation']:
    F[s] = pd.concat([F[s].reset_index(drop=True), image_feats(s)], axis=1)
IMG = [f'img{i}' for i in range(204)]
HIST = EVC + ['tsd_log', 't1_since_peak'] + [f'{c}_{k}' for c in ['meas_t1', 'meas_t2', 'thermal_t', 'meas_v', 'meas_p'] for k in ['cummax', 'd30', 'd120', 'd300']]
SETS = {'A image only': IMG, 'B current sensor+gas': SENS + GAS, 'C B + history': SENS + GAS + HIST, 'D image + sensor + history': IMG + SENS + GAS + HIST}
P = dict(max_iter=300, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1., class_weight='balanced', random_state=0)

tr = pd.concat([F['fit'], F['dev']], ignore_index=True); tr = tr[tr.label.isin([4, 5])]; ytr = (tr.label == 5).values
va = F['validation']; vm = va.label.isin([4, 5]).values; v = va[vm]; yv = (v.label == 5).values; e5 = v.experiment_id.str.startswith('E_e5bc').values
rows = []
for name, cols in SETS.items():
    oof = np.zeros(len(tr))
    for i, j in GroupKFold(5).split(tr, groups=tr.experiment_id):
        oof[j] = HistGradientBoostingClassifier(**P).fit(tr[cols].iloc[i], ytr[i]).predict_proba(tr[cols].iloc[j])[:, 1]
    pv = HistGradientBoostingClassifier(**P).fit(tr[cols], ytr).predict_proba(v[cols])[:, 1]
    k = ~e5
    rows.append({'set': name, 'n_feat': len(cols),
                 'trainCV_AUC': round(roc_auc_score(ytr, oof), 3), 'trainCV_balacc': round(balanced_accuracy_score(ytr, oof > .5), 3),
                 'trainCV_5to4': int(((ytr) & (oof <= .5)).sum()), 'trainCV_4to5': int(((~ytr) & (oof > .5)).sum()),
                 'val_AUC': round(roc_auc_score(yv, pv), 3), 'val_balacc': round(balanced_accuracy_score(yv, pv > .5), 3),
                 'val_5to4': int((yv & (pv <= .5)).sum()), 'val_4to5': int((~yv & (pv > .5)).sum()),
                 'val_noE5_AUC': round(roc_auc_score(yv[k], pv[k]), 3), 'val_noE5_5to4': int((yv[k] & (pv[k] <= .5)).sum()),
                 'E5_5rows_recall': round(float((pv[e5] > .5).mean()), 3)})
    print(rows[-1], flush=True)
r = pd.DataFrame(rows); r.to_csv('info_ablation_4v5.csv', index=False)
print(); print(f'train 4/5 rows: 4={(~ytr).sum()} 5={ytr.sum()} | validation 4={(~yv).sum()} 5={yv.sum()} (E_e5bc 5-rows {e5.sum()})')
print(r.to_string(index=False))
