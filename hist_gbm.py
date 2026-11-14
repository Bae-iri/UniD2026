"""센서 + 인과(과거 방향) 이력 특징 + HistGradientBoosting.

- 같은 experiment_id 안에서 relative_time_s <= 현재 시점 관측만 사용 (누적 최대/최소, 30·120초 시간 기반 창 평균·변화율,
  가스 최초 검출 후 경과시간, 온도 최고점 이후 경과시간 등). 미래 관측·실험 전체 통계 미사용.
- 창은 행 수가 아니라 초 단위라 test의 15초 간격 구간에도 동일 정의.
- 이미지는 Validation에서 성능을 떨어뜨려(실험별 과적합) 제외.

  python hist_gbm.py eval    --data-dir D            # Train→Validation, Train+Val GroupKFold(5)
  python hist_gbm.py fit     --data-dir D --model M  # Train+Validation 전체 학습, seed 3개
  python hist_gbm.py predict --data-dir D --model M --input test.csv --output submission.csv
"""
import argparse, warnings
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings('ignore')

GAS = ['meas_ch4','meas_co','meas_co2','meas_hcn','meas_hcl','meas_hf','meas_n2o','meas_no','meas_no2','meas_so2']
TEMP = ['meas_t1','meas_t2','thermal_t','meas_t1_plus_t2','indu_t','indu_t.1','atmp_t']
CUR = ['meas_v','meas_p'] + TEMP + GAS
STATIC = ['bat_class','bat_cc','soc_status','tr_method']
WINS = [30, 120]
SEEDS = [0, 1, 2]


def _window(t, x, W):
    s = np.searchsorted(t, t - W, side='right'); lag = s - 1
    cs = np.concatenate([[0], np.nancumsum(np.nan_to_num(x))]); cn = np.concatenate([[0], np.cumsum(~np.isnan(x))])
    e = np.arange(1, len(t) + 1)
    mean = (cs[e] - cs[s]) / np.maximum(cn[e] - cn[s], 1)
    j = np.maximum(lag, 0)
    rate = np.where(lag >= 0, (x - x[j]) / np.maximum(t - t[j], 1) * W, np.nan)  # W초당 변화량
    return mean, rate


def build(frame, static_codes):
    df = frame.copy()
    for c in CUR + ['relative_time_s']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in STATIC:
        df[c] = df[c].astype(str).map(static_codes[c]).fillna(-1)
    df['gas_sum'] = df[GAS].clip(lower=0).sum(1); df['gas_nz'] = (df[GAS] > 0).sum(1)
    df = df.sort_values(['experiment_id', 'relative_time_s'], kind='mergesort').reset_index(drop=True)
    sig = ['meas_v','meas_p','gas_sum','gas_nz'] + TEMP + GAS
    feats = {}
    def put(k, idx, v):
        feats.setdefault(k, np.full(len(df), np.nan))[idx] = v
    T = df.relative_time_s.to_numpy(float)
    for _, idx in df.groupby('experiment_id', sort=False).indices.items():
        t = T[idx]
        for c in sig:
            x = df[c].to_numpy(float)[idx]
            cm = np.fmax.accumulate(np.nan_to_num(x, nan=-np.inf)); cmin = np.fmin.accumulate(np.nan_to_num(x, nan=np.inf))
            put(f'{c}_cmax', idx, cm); put(f'{c}_cmin', idx, cmin); put(f'{c}_frommax', idx, x - cm)
            for W in WINS:
                m, r = _window(t, x, W); put(f'{c}_m{W}', idx, m); put(f'{c}_d{W}', idx, r)
        for c in ['meas_v', 'meas_p', 'gas_sum'] + TEMP:  # 실험 첫 관측 대비 변화량(실험별 기준값 차이 상쇄)
            x = df[c].to_numpy(float)[idx]; put(f'{c}_rel0', idx, x - x[0])
        gs = df.gas_sum.to_numpy(float)[idx]
        fm = np.minimum.accumulate(np.where(gs > 0, t, np.inf))
        put('t_since_gas', idx, np.where(np.isinf(fm), -1, t - fm))
        put('gas_frac_past', idx, np.cumsum(gs > 0) / np.arange(1, len(t) + 1))
        for c in ['meas_t1', 'thermal_t']:
            x = np.nan_to_num(df[c].to_numpy(float)[idx], nan=-np.inf); cm = np.maximum.accumulate(x)
            put(f't_since_{c}_peak', idx, t - pd.Series(np.where(x >= cm, t, np.nan)).ffill().to_numpy())
    F = pd.DataFrame(feats); F = F.where(~np.isinf(F))
    X = pd.concat([df[CUR + ['relative_time_s', 'gas_sum', 'gas_nz'] + STATIC], F], axis=1)
    return df, X


def model(seed):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed, early_stopping=False)


def fit_many(X, y):
    w = compute_sample_weight('balanced', y) ** 0.5
    return [model(s).fit(X, y, sample_weight=w) for s in SEEDS]


def proba(models, X):
    return np.mean([m.predict_proba(X) for m in models], axis=0)


def codes(frames):
    return {c: {v: i for i, v in enumerate(sorted(pd.concat([f[c] for f in frames]).astype(str).unique()))} for c in STATIC}


def read(d, s):
    return pd.read_csv(f'{d}/{s}/{s}.csv')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['eval', 'fit', 'predict'])
    ap.add_argument('--data-dir', required=True); ap.add_argument('--model', default='hist_gbm.joblib')
    ap.add_argument('--input'); ap.add_argument('--output', default='submission_hist_gbm.csv')
    ap.add_argument('--sample-submission'); a = ap.parse_args()
    if a.cmd == 'eval':
        tr, va = read(a.data_dir, 'train'), read(a.data_dir, 'validation'); sc = codes([tr])
        dtr, Xtr = build(tr, sc); dva, Xva = build(va, sc)
        p = proba(fit_many(Xtr, dtr.label), Xva).argmax(1) + 1
        print('Train->Validation macro F1 %.4f' % f1_score(dva.label, p, average='macro'),
              'per class', f1_score(dva.label, p, average=None).round(3))
        al = pd.concat([tr, va], ignore_index=True); sc = codes([al]); d, X = build(al, sc)
        oof = np.zeros(len(d), int)
        for k, (i, j) in enumerate(GroupKFold(5).split(X, groups=d.experiment_id)):
            w = compute_sample_weight('balanced', d.label[i]) ** 0.5
            oof[j] = model(k).fit(X.loc[i], d.label[i], sample_weight=w).predict(X.loc[j])
        print('Train+Val GroupKFold(5) pooled macro F1 %.4f' % f1_score(d.label, oof, average='macro'))
    elif a.cmd == 'fit':
        al = pd.concat([read(a.data_dir, 'train'), read(a.data_dir, 'validation')], ignore_index=True)
        sc = codes([al]); d, X = build(al, sc)
        joblib.dump({'models': fit_many(X, d.label), 'codes': sc, 'columns': list(X.columns)}, a.model)
        print('saved', a.model, 'rows', len(d), 'experiments', d.experiment_id.nunique())
    else:
        b = joblib.load(a.model); te = pd.read_csv(a.input); d, X = build(te, b['codes'])
        pred = pd.Series(proba(b['models'], X[b['columns']]).argmax(1) + 1, index=d.sample_id.values)
        ss = pd.read_csv(a.sample_submission) if a.sample_submission else te[['sample_id']]
        out = pd.DataFrame({'sample_id': ss.sample_id, 'label': pred.loc[ss.sample_id].values})
        out.to_csv(a.output, index=False); print('wrote', a.output, len(out), 'rows; label counts', out.label.value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
