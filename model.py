"""모델 정의와 피처 함수. test.py와 train.py가 같이 쓴다.

최종 모델 = B 융합 모델(seed 42/43/44 평균) + t1 45°C 규칙 + 경계 보정 3개(5→6, 6→5, 4→3)

모든 피처는 같은 실험(experiment_id)에서 현재 행 시각(relative_time_s)까지 관측한 값만 쓴다.
실험 안에서는 시각으로 정렬해서 계산하므로 입력 행 순서와 무관하고, 다른 실험의 행은 쓰지 않는다.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'lib'))
from abc_reference import abc_fusion_experiment as abc          # noqa: E402  B 융합 모델(FusionModel)
from targeted_rule1 import model as T                            # noqa: E402  5→6 보정 피처·HGB
import precision as P                                            # noqa: E402  6→5 보정

SEEDS = (42, 43, 44)
T1_STAGE2 = 45.0
LOWER_PAIRS = {'23': (2, 3), '32': (3, 2), '34': (3, 4), '43': (4, 3)}
LOWER_USED = ('34', '43')          # 사전 채택 규칙이 남긴 보정. 3→4는 학습 내부 기준값 선택에서 꺼짐(1.01)
LOWER_DROP = ('observation_gap_log1p', 'observed_count_log1p')   # 관측 개수 의존 특징은 쓰지 않음


# ----------------------------------------------------------------------------- B 모델
class BModel:
    """저장된 B 체크포인트 3개와 학습 데이터로 적합한 센서 전처리 통계로 확률만 계산한다(다시 적합하지 않음)."""

    def __init__(self, weights):
        weights = Path(weights)
        meta = json.loads((weights/'preprocessing.json').read_text())
        self.prep = abc.SensorPreprocessor.__new__(abc.SensorPreprocessor)
        for key in ('median', 'mean', 'std'):
            setattr(self.prep, key, pd.Series(meta[key], index=abc.NUMERIC))
        self.models = []
        for seed in SEEDS:
            m = abc.FusionModel('B')
            m.load_state_dict(torch.load(weights/f'B_seed{seed}.pt', map_location='cpu', weights_only=True)['state'])
            self.models.append(m.eval())

    @torch.inference_mode()
    def probabilities(self, frame, image_root):
        """(3, N, 6) seed별 확률. 행마다 독립 계산(GroupNorm, eval 모드)."""
        root = Path(image_root).resolve()
        paths = []
        for name in frame.image_path:
            rel = Path(str(name).replace('\\', '/'))
            if rel.is_absolute():
                raise ValueError('image_path must be relative to image_root')
            path = (root/rel).resolve()
            if not path.is_relative_to(root):
                raise ValueError('image path leaves image_root')
            paths.append(path)
        q, gas = self.prep.transform(frame)
        out = [[] for _ in self.models]
        for s in range(0, len(frame), 128):
            e = min(s+128, len(frame))
            packed = torch.from_numpy(np.stack([abc.common.read_image(p, 128, 4)[0] for p in paths[s:e]]))
            args = (packed[:, :3].float()/255., packed[:, 3:4], torch.from_numpy(q[s:e]), torch.from_numpy(gas[s:e]))
            for i, m in enumerate(self.models):
                out[i].append(m(*args).softmax(-1).numpy())
        return np.stack([np.concatenate(o) for o in out])


# ----------------------------------------------------------------------------- t1 45°C 규칙
def rule1(frame, raw):
    """현재까지 t1 누적 최대 ≥ 45°C면 2단계 이상, 아니면 1단계로 확정(양방향)."""
    t = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    v = pd.to_numeric(frame.meas_t1, errors='coerce').to_numpy(float)
    maxima = np.full(len(frame), np.nan)
    for ids in frame.groupby('experiment_id', sort=False).indices.values():
        ids = ids[np.argsort(t[ids], kind='stable')]
        best = np.nan
        for row in ids:
            if np.isfinite(v[row]) and not (best >= v[row]):
                best = v[row]
            maxima[row] = best
    out = np.asarray(raw, int).copy()
    hot = maxima >= T1_STAGE2
    out[(out == 1) & hot] = 2
    out[(out >= 2) & ~hot] = 1
    return out


# ----------------------------------------------------------------------------- 하위 경계 보정(4→3) 피처
def _first_time(cond, t):
    out = np.full(len(t), -1.0)
    hit = np.flatnonzero(cond)
    if len(hit):
        out[hit[0]:] = np.log1p(t[hit[0]:]-t[hit[0]])
    return out


def lower_features(frame, probs):
    """센서 이력(관측 개수 특징 제외) + 전압 상대 하락·온도 누적·B 3/4단계 확률 이력. 모두 현재까지의 관측."""
    x, names = T.features(frame, probs)
    keep = [i for i, n in enumerate(names['45']) if n not in LOWER_DROP]
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
        v = pd.Series(V[ids]).ffill().to_numpy()                    # 앞 방향 채움만(과거 값)
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


def lower_apply(base, score, pair, th):
    s, tg = LOWER_PAIRS[pair]
    out = np.asarray(base, int).copy(); out[(np.asarray(base) == s) & (score >= th)] = tg
    return out


def lower_decide(art, frame, probs, base, pairs=LOWER_USED):
    x, names = lower_features(frame, probs)
    if names != art['names']:
        raise ValueError('lower feature schema changed')
    out = np.asarray(base, int).copy(); fired = np.zeros(len(out), int)
    for pair in pairs:
        p = lower_apply(base, T.probability(art['models'][pair], x), pair, art['thresholds'][pair])
        m = p != base; out[m] = p[m]; fired += m
    out[fired > 1] = np.asarray(base)[fired > 1]
    return out


# ----------------------------------------------------------------------------- 전체 예측
def upper_heads(bundle):
    """5→6 헤드는 그대로, 4→5 헤드는 끈다(기준값 ∞). 처음 보는 배터리 평가에서 손해라 제거."""
    heads = copy.deepcopy(bundle['baseline_heads']); heads['thresholds']['45'] = np.inf
    return heads


def predict_labels(frame, probs, bundle, lower):
    """B 확률 → 기준(B+t1) → 5→6·6→5·4→3 보정. 세 보정은 모두 기준 예측에 대해 따로 판단한다."""
    base = rule1(frame, np.asarray(probs).mean(0).argmax(1)+1)
    heads = upper_heads(bundle)
    x, names = T.features(frame, probs)
    if names != heads['names']:
        raise ValueError('upper feature schema changed')
    upper = base.copy()
    s56 = T.probability(heads['models']['56'], x['56'])
    upper[(base == 5) & (s56 >= heads['thresholds']['56'])] = 6
    gate = bundle['precision_head']
    upper = P.apply(upper, base, P.score(gate['model'], x['56']), gate['threshold'])
    low = lower_decide(lower, frame, probs, base)
    out = upper.copy(); m = low != base; out[m] = low[m]
    return out
