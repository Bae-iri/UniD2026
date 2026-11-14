"""Tune the 4/5/6 specialist on top of frozen original B.
Selection metric = mean(dev macro F1, validation macro F1), both out-of-sample for B and for the specialist."""
import itertools, json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
ROOT = '/home/ubuntu/data/battery_data_light/'; BR = '/home/ubuntu/leewon/abc_runs/current_query_v1/'
EV = '/home/ubuntu/gahyun/experiments/b_independent_studies/runs/independent_seed42_43_44_v1/experiment1_voltage/'
SENS = ['meas_v', 'meas_t1', 'meas_t2', 'atmp_t', 'thermal_t', 'indu_t']
GAS = ['meas_p', 'meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
EVC = ['time_since_drop', 'drop_occurred', 'max_voltage_drop', 'voltage_after_drop']

def feats(df, split):
    df = df.copy(); ev = pd.read_csv(EV + f'{split}_voltage_event_columns.csv').set_index('sample_id').loc[df.sample_id]
    for c in EVC: df[c] = ev[c].values
    df = df.sort_values(['experiment_id', 'relative_time_s']); g = df.groupby('experiment_id')
    for c in ['meas_t1', 'meas_t2', 'thermal_t', 'meas_v'] + GAS:
        df[c + '_cummax'] = g[c].cummax()
        for sec in [30, 120, 300]:
            # time-based lag: value `sec` seconds ago (works for 5/10/15 s spacing)
            lag = df[['experiment_id', 'relative_time_s', c]].copy(); lag['relative_time_s'] += sec
            prev = df[['experiment_id', 'relative_time_s']].merge(lag, on=['experiment_id', 'relative_time_s'], how='left')[c].values
            df[f'{c}_d{sec}'] = (df[c].values - prev) / sec
    df['t1_since_peak'] = df.meas_t1 - df.meas_t1_cummax
    df['tsd_log'] = np.log1p(df.time_since_drop)
    return df.sort_index()

tr = pd.read_csv(ROOT + 'train/train.csv'); va = pd.read_csv(ROOT + 'validation/validation.csv'); te = pd.read_csv(ROOT + 'test/test.csv')
fi, di = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20261114).split(tr, groups=tr.experiment_id))
F = {'fit': feats(tr.iloc[fi].reset_index(drop=True), 'fit'), 'dev': feats(tr.iloc[di].reset_index(drop=True), 'dev'),
     'validation': feats(va, 'validation'), 'test': feats(te, 'test')}
T = ['meas_t1', 'meas_t2', 'thermal_t', 'meas_v']
FEATS = {
    'base': SENS + GAS + EVC,
    'temp_dyn': SENS + GAS + EVC + ['tsd_log', 't1_since_peak'] + [f'{c}_{s}' for c in T for s in ['cummax', 'd30', 'd120', 'd300']],
    'all_dyn': SENS + GAS + EVC + ['tsd_log', 't1_since_peak'] + [f'{c}_{s}' for c in T + GAS for s in ['cummax', 'd30', 'd120', 'd300']],
}
MODELS = {
    'small': dict(max_iter=200, learning_rate=.05, max_leaf_nodes=7, min_samples_leaf=40, l2_regularization=1.),
    'mid': dict(max_iter=300, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.),
    'big': dict(max_iter=500, learning_rate=.03, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.),
}
WEIGHTS = [0.5, 0.7, 1.0]   # share of specialist in 4/5/6 probability
BAGS = 5

def specialist(frames, cols, params):
    d = pd.concat(frames); d = d[d.label >= 4]; w = d.label.map(1 / d.label.value_counts()).values
    return [HistGradientBoostingClassifier(random_state=s, **params).fit(d[cols], d.label, sample_weight=w) for s in range(BAGS)]

def spec_proba(models, x): return np.mean([m.predict_proba(x) for m in models], 0)

def bprob(split):
    p = pd.read_csv(BR + f'ensemble_B_{split}_predictions.csv').set_index('sample_id').loc[F[split].sample_id]
    return p[[f'p{i}' for i in range(1, 7)]].values

def combine(Pb, ps, w):
    pred = Pb.argmax(1) + 1; m = pred >= 4
    b456 = Pb[m, 3:] / Pb[m, 3:].sum(1, keepdims=True)
    out = pred.copy(); out[m] = (w * ps[m] + (1 - w) * b456).argmax(1) + 4; return out

PB = {s: bprob(s) for s in ['dev', 'validation', 'test']}
def mf(y, p): return 100 * f1_score(y, p, labels=range(1, 7), average='macro')

if __name__ == '__main__':
    rows = []
    for fname, mname in itertools.product(FEATS, MODELS):
        cols = FEATS[fname]
        S1 = specialist([F['fit']], cols, MODELS[mname])                 # -> dev
        S2 = specialist([F['fit'], F['dev']], cols, MODELS[mname])       # -> validation
        p_dev, p_val = spec_proba(S1, F['dev'][cols]), spec_proba(S2, F['validation'][cols])
        yd, yv = F['dev'].label.values, F['validation'].label.values; keep = ~F['validation'].experiment_id.str.startswith('E_e5bc').values
        for w in WEIGHTS:
            d, v = combine(PB['dev'], p_dev, w), combine(PB['validation'], p_val, w)
            rows.append(dict(features=fname, model=mname, w=w, dev=round(mf(yd, d), 2), val=round(mf(yv, v), 2),
                             val_noE5=round(mf(yv[keep], v[keep]), 2), mean=round((mf(yd, d) + mf(yv, v)) / 2, 2)))
            print(rows[-1], flush=True)
    r = pd.DataFrame(rows).sort_values('mean', ascending=False); r.to_csv('tuning_results.csv', index=False)
    print(r.head(10).to_string(index=False))
    print('B only: dev %.2f val %.2f' % (mf(F['dev'].label, PB['dev'].argmax(1) + 1), mf(F['validation'].label, PB['validation'].argmax(1) + 1)))
