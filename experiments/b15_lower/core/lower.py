"""2↔3, 3↔4 경계 보정(현재·과거 관측만).

각 보정은 원래 기준 예측(B + t1 45°C)이 source인 행에서 target으로 바꿀지 판단한다.
구조·하이퍼파라미터·기준값 자동 선택은 targeted_rule1과 같다(얕은 HGB, 학습 실험 3-fold, 0.5/0.7/0.9).
관측 개수에 의존하는 특징(observation_gap, observed_count)은 쓰지 않는다(간격 강건성).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import upper as T

PAIRS = {'23': (2, 3), '32': (3, 2), '34': (3, 4), '43': (4, 3)}
DROP = ('observation_gap_log1p', 'observed_count_log1p')


def _first_time(cond, t):
    """조건이 처음 참이 된 시각부터 지금까지 경과(log1p). 아직이면 -1. cond는 시간순·누적형."""
    out = np.full(len(t), -1.0)
    hit = np.flatnonzero(cond)
    if len(hit):
        out[hit[0]:] = np.log1p(t[hit[0]:]-t[hit[0]])
    return out


def features(frame, probs):
    x, names = T.features(frame, probs)
    keep = [i for i, n in enumerate(names['45']) if n not in DROP]
    phys = x['45'][:, keep]; pn = [names['45'][i] for i in keep]
    p = np.asarray(probs, float); mean = p.mean(0); pred = mean.argmax(1)+1
    t = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    V = pd.to_numeric(frame.meas_v, errors='coerce').to_numpy(float)
    T1 = pd.to_numeric(frame.meas_t1, errors='coerce').to_numpy(float)
    T2 = pd.to_numeric(frame.meas_t2, errors='coerce').to_numpy(float)
    en = ['v_ratio', 'v_min_ratio', 'v_drop', 't1_cummax', 't2_cummax', 'since_t1_45', 'since_v95', 'since_v90',
          'B_p2', 'B_p3', 'B_p4', 'peak_p3', 'peak_p4', 'since_pred3', 'since_pred4',
          'p2_past60_mean', 'p3_past60_mean', 'p4_past60_mean']
    ex = np.full((len(frame), len(en)), np.nan)
    for ids in frame.groupby('experiment_id', sort=False).indices.values():
        ids = ids[np.argsort(t[ids], kind='stable')]; ts = t[ids]
        v = pd.Series(V[ids]).ffill().to_numpy()
        vmax = np.fmax.accumulate(v); vmin = np.fmin.accumulate(v)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = v/vmax; mratio = vmin/vmax
        t1m = np.fmax.accumulate(T1[ids]); t2m = np.fmax.accumulate(T2[ids])
        lowest = np.fmin.accumulate(np.where(np.isfinite(ratio), ratio, np.inf))
        pm = np.maximum.accumulate(mean[ids][:, 2:4], axis=0)
        reached = np.maximum.accumulate(pred[ids])
        past = np.zeros((len(ids), 3))
        for j in range(len(ids)):
            s = np.searchsorted(ts[:j+1], ts[j]-60., side='left')
            past[j] = mean[ids[s:j+1], 1:4].mean(0)
        ex[ids] = np.column_stack([ratio, mratio, vmax-v, t1m, t2m,
            _first_time(t1m >= 45, ts), _first_time(lowest < .95, ts), _first_time(lowest < .90, ts),
            mean[ids][:, 1:4], pm, _first_time(reached >= 3, ts), _first_time(reached >= 4, ts), past])
    ex[~np.isfinite(ex)] = np.nan
    return np.c_[phys, ex], pn+en


def fit(x, y, base, groups, pair):
    s, tg = PAIRS[pair]
    keep = np.isin(y, [s, tg]) | (base == s)
    yy = (y[keep] == tg).astype(int)
    if len(np.unique(yy)) < 2:
        return None
    keys = pd.DataFrame({'e': groups[keep], 'y': yy})
    w = 1/keys.groupby(['e', 'y']).y.transform('size').to_numpy(float)
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(**T.PARAMS); m.fit(x[keep], yy, sample_weight=w/w.mean())
    return m


def apply(base, score, pair, th):
    s, tg = PAIRS[pair]
    out = np.asarray(base, int).copy(); out[(np.asarray(base) == s) & (score >= th)] = tg
    return out


def select(score, y, base, groups, pair):
    ids = list(pd.DataFrame({'g': groups}).groupby('g').indices.values())
    cb = np.stack([T.matrix(y[i], base[i]) for i in ids]); choices = []
    for th in T.THRESHOLDS:
        p = apply(base, score, pair, th)
        cp = np.stack([T.matrix(y[i], p[i]) for i in ids])
        delta = float(T.f1(cp.sum(0)).mean()-T.f1(cb.sum(0)).mean()); per = T.f1(cp).mean(1)-T.f1(cb).mean(1)
        fixed = int(((base != y) & (p == y)).sum()); harmed = int(((base == y) & (p != y)).sum())
        loo = [float(T.f1(cp.sum(0)-a).mean()-T.f1(cb.sum(0)-b).mean()) for a, b in zip(cp, cb)]
        ok = (delta > 0 and fixed >= max(3, 2*harmed) and (per > 1e-12).sum() >= 3
              and (per > 1e-12).sum() > (per < -1e-12).sum() and min(loo) > 0)
        choices.append(dict(threshold=th, delta=delta, fixed=fixed, harmed=harmed, passed=bool(ok)))
    valid = [c for c in choices if c['passed']]
    return (max(valid, key=lambda c: (c['delta'], c['threshold']))['threshold'] if valid else 1.01), choices


def train(frame, probs, y, base):
    x, names = features(frame, probs); g = frame.experiment_id.to_numpy()
    art = dict(models={}, thresholds={}, choices={}, names=names)
    folds = list(GroupKFold(3).split(x, groups=g))
    for pair in PAIRS:
        sc = np.zeros(len(y))
        for a, b in folds:
            sc[b] = T.probability(fit(x[a], y[a], base[a], g[a], pair), x[b])
        art['thresholds'][pair], art['choices'][pair] = select(sc, y, base, g, pair)
        art['models'][pair] = fit(x, y, base, g, pair)
    return art


def decide(art, frame, probs, base, pairs):
    """선택된 보정들을 원래 기준 예측에 대해 각각 판단. 같은 행(기준 3)에서 3→2와 3→4가 둘 다 켜지면 3을 유지."""
    x, names = features(frame, probs)
    assert names == art['names']
    out = np.asarray(base, int).copy(); fired = np.zeros(len(out), int)
    for pair in pairs:
        p = apply(base, T.probability(art['models'][pair], x), pair, art['thresholds'][pair])
        m = p != base; out[m] = p[m]; fired += m
    out[fired > 1] = np.asarray(base)[fired > 1]
    return out
