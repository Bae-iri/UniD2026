"""Feasibility: original B decides 1-3 vs 4-6; a separate 4/5/6 specialist decides inside. Causal features only."""
import sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
sys.path.insert(0, '/home/ubuntu/gahyun/experiments/b_independent_studies/common')
from sklearn.model_selection import GroupShuffleSplit
ROOT = '/home/ubuntu/data/battery_data_light/'; BR = '/home/ubuntu/leewon/abc_runs/current_query_v1/'
EV = '/home/ubuntu/gahyun/experiments/b_independent_studies/runs/independent_seed42_43_44_v1/experiment1_voltage/'
SENS = ['meas_v', 'meas_t1', 'meas_t2', 'atmp_t', 'thermal_t', 'indu_t']
GAS = ['meas_p', 'meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']

def feats(df, split):
    df = df.copy(); ev = pd.read_csv(EV + f'{split}_voltage_event_columns.csv').set_index('sample_id').loc[df.sample_id]
    for c in ['time_since_drop', 'drop_occurred', 'max_voltage_drop', 'voltage_after_drop']: df[c] = ev[c].values
    df = df.sort_values(['experiment_id', 'relative_time_s']); g = df.groupby('experiment_id')
    for c in ['meas_t1', 'meas_t2', 'thermal_t', 'meas_co', 'meas_co2', 'meas_p']:
        df[c + '_cummax'] = g[c].cummax()
        for w in [6, 24]:  # rows back (~30s / ~2min at 5s)
            df[f'{c}_d{w}'] = (df[c] - g[c].shift(w)) / (df.relative_time_s - g.relative_time_s.shift(w))
    return df.sort_index()

tr = pd.read_csv(ROOT + 'train/train.csv'); va = pd.read_csv(ROOT + 'validation/validation.csv'); te = pd.read_csv(ROOT + 'test/test.csv')
fi, di = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20261114).split(tr, groups=tr.experiment_id))
fit, dev = tr.iloc[fi].reset_index(drop=True), tr.iloc[di].reset_index(drop=True)
F = {'fit': feats(fit, 'fit'), 'dev': feats(dev, 'dev'), 'validation': feats(va, 'validation'), 'test': feats(te, 'test')}
X = [c for c in F['fit'].columns if c in SENS + GAS or any(c.startswith(s) for s in ['time_since', 'drop_occ', 'max_volt', 'voltage_after']) or '_cummax' in c or '_d6' in c or '_d24' in c]

def specialist(train_frames):
    d = pd.concat(train_frames); d = d[d.label >= 4]
    w = d.label.map(1 / d.label.value_counts()).values
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1., random_state=0).fit(d[X], d.label, sample_weight=w)

def bprob(split):
    p = pd.read_csv(BR + f'ensemble_B_{split}_predictions.csv').set_index('sample_id').loc[F[split].sample_id]
    return p[[f'p{i}' for i in range(1, 7)]].values

def combine(Pb, S, x, conservative=False):
    pred = Pb.argmax(1) + 1; m = pred >= 4
    ps = S.predict_proba(x[m])  # classes 4,5,6
    new = ps.argmax(1) + 4
    if conservative: new = np.where(ps.max(1) >= .8, new, pred[m])
    out = pred.copy(); out[m] = new; return out

for name, trainF, evals in [('fit -> dev,val', [F['fit']], ['dev', 'validation']), ('fit+dev -> val', [F['fit'], F['dev']], ['validation'])]:
    S = specialist(trainF)
    for sp in evals:
        y = F[sp].label.values; Pb = bprob(sp); b = Pb.argmax(1) + 1; keep = ~F[sp].experiment_id.str.startswith('E_e5bc').values
        for tag, p in [('B only', b), ('B+spec', combine(Pb, S, F[sp][X])), ('B+spec conservative', combine(Pb, S, F[sp][X], True))]:
            f = f1_score(y, p, labels=range(1, 7), average=None)
            print(f'{name:16} {sp:10} {tag:20} macro {100*f.mean():.2f} noE5 {100*f1_score(y[keep], p[keep], average="macro"):.2f}  F1_4/5/6 {np.round(100*f[3:],1)}  changed {(p!=b).sum()}')
S = specialist([F['fit'], F['dev']]); Pb = bprob('test'); b = Pb.argmax(1) + 1
print('test changed (full / conservative):', (combine(Pb, S, F['test'][X]) != b).sum(), (combine(Pb, S, F['test'][X], True) != b).sum())

if __name__ == '__main__':
    o = pd.read_csv(BR + 'submission_B.csv')
    for cons, fn in [(False, 'submission_B_spec456_full.csv'), (True, 'submission_B_spec456_conservative.csv')]:
        p = pd.Series(combine(Pb, S, F['test'][X], cons), index=F['test'].sample_id).loc[o.sample_id]
        out = o.copy(); out['label'] = p.values.astype(int)
        assert len(out) == 3451 and out.label.between(1, 6).all() and out.sample_id.is_unique
        out.to_csv(fn, index=False); ch = out.label != o.label
        print(fn, 'changed', ch.sum()); print(pd.crosstab(o.label[ch], out.label[ch]))
        ex = te.set_index('sample_id').loc[o.sample_id].experiment_id.str[:8].values
        print(pd.Series(ex[ch.values]).value_counts().to_dict())
