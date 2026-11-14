"""원본 B(동결) 위의 4·5·6단계 MLP456과 배터리 조건(bat_class, bat_cc) 모듈.

모든 입력은 같은 실험에서 현재 행까지 관측한 값이다.
- 센서/B 확률 이력 특징: targeted_rule1.model.features (현재·과거만, prefix 검증됨)
- 현재 가스 농도: 현재 행 값의 고정 변환(log1p)
- 배터리 조건: 현재 행의 bat_class, bat_cc (실험 내 상수, 관측된 행 값 그대로 사용)
정규화 통계는 학습 실험의 행으로만 적합한다.

출력 logit(4,5,6) = log(B의 4·5·6 재정규화 확률) + MLP 보정. 보정 출력층을 0으로 시작하므로
학습 전 MLP456은 원본 B와 같은 결정을 낸다.

변형:
  V0  MLP456 (센서 특징만)
  V1  V0 동결 + 조건을 첫 은닉층에 단순 결합(0 초기화, 결합 가중치만 학습)
  V2  V0 동결 + 조건별 입력 특징 가중치 x * (1 + a*tanh(g(c))), a=1, g 마지막 층 0 초기화
  V3  V2와 같으나 a=0.25, tanh(g)^2 페널티로 변화량 제한
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from targeted_rule1.model import features as history_features

GASES = ['meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf',
         'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
BAT_CLASSES = (1, 2, 3)
VARIANTS = ('V0', 'V1', 'V2', 'V3')
GATE_ALPHA = {'V2': 1.0, 'V3': 0.25}
GATE_PENALTY = {'V2': 0.0, 'V3': 1e-2}
HIDDEN = 64
BASE_EPOCHS = 40
HEAD_EPOCHS = 20


def feature_groups(names):
    """해석용 센서 그룹 이름."""
    out = []
    for n in names:
        if n.startswith('meas_v'): g = 'voltage'
        elif n.startswith('meas_t1'): g = 'temp_t1'
        elif n.startswith('meas_t2'): g = 'temp_t2'
        elif n.startswith('thermal_t'): g = 'thermal_t'
        elif n.startswith('meas_p'): g = 'pressure'
        elif n.startswith('gas_'): g = 'gas'
        elif n.startswith(('B_', 'seed_', 'past_peak', 'observed6', 'last6', 'current6', 'p4_', 'p5_', 'p6_')): g = 'B_prob'
        else: g = 'time'
        out.append(g)
    return out


def build_features(frame, probs):
    """(raw features, names, B456 log-prob, condition raw). probs: (3, N, 6) seed probabilities."""
    x, names = history_features(frame, probs)
    gas = frame[GASES].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    gas = np.sign(gas)*np.log1p(np.abs(gas))
    x = np.c_[x['56'], gas]
    names = names['56']+[f'gas_{g[5:]}' for g in GASES]
    mean = np.asarray(probs, float).mean(0)
    b456 = np.clip(mean[:, 3:6], 1e-6, 1)
    b456 = np.log(b456/b456.sum(1, keepdims=True))
    cls = pd.to_numeric(frame.bat_class, errors='raise').to_numpy(int)
    if not np.isin(cls, BAT_CLASSES).all():
        raise ValueError('Unknown bat_class')
    cc = pd.to_numeric(frame.bat_cc, errors='raise').to_numpy(float)
    return x.astype(np.float32), names, b456.astype(np.float32), np.c_[cls, np.log1p(cc)]


class Prep:
    """학습 행으로만 적합하는 표준화."""

    def fit(self, x, cond, experiments):
        x = np.asarray(x, np.float64)   # float32 누적 오차로 상수 특징의 std가 0이 아니게 되는 것을 막는다
        self.mean = np.nanmean(x, 0)
        self.scale = np.nanstd(x, 0)
        self.mean[~np.isfinite(self.mean)] = 0
        # 학습 행에서 사실상 상수인 특징은 정보가 없으므로 추론 시 항상 0으로 둔다(분포 밖 값이 증폭되지 않게).
        self.constant = ~np.isfinite(self.scale) | (self.scale <= 1e-6*np.maximum(1, np.abs(self.mean)))
        self.scale[self.constant] = 1
        first = pd.Series(cond[:, 1]).groupby(np.asarray(experiments)).first()
        self.cc_mean, self.cc_std = float(first.mean()), float(max(first.std(), 1e-6))
        return self

    def x(self, x):
        z = (np.asarray(x, np.float64)-self.mean)/self.scale
        z[:, getattr(self, 'constant', np.zeros(z.shape[1], bool))] = 0
        z[~np.isfinite(z)] = 0
        return np.clip(z, -10, 10).astype(np.float32)

    def c(self, cond):
        onehot = (cond[:, :1] == np.asarray(BAT_CLASSES)[None]).astype(np.float32)
        return np.c_[onehot, (cond[:, 1:2]-self.cc_mean)/self.cc_std].astype(np.float32)


class MLP456(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1 = nn.Linear(d, HIDDEN)
        self.l2 = nn.Linear(HIDDEN, HIDDEN)
        self.out = nn.Linear(HIDDEN, 3)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.drop = nn.Dropout(.2)

    def forward(self, x, b456, extra=None):
        h = self.l1(x)
        if extra is not None:
            h = h+extra
        h = self.drop(F.gelu(h))
        h = self.drop(F.gelu(self.l2(h)))
        return b456+self.out(h)


class Concat(nn.Module):
    """V1: 조건 c를 첫 은닉층에 결합. 0 초기화로 V0와 같은 출력에서 시작."""

    def __init__(self, c):
        super().__init__()
        self.w = nn.Linear(c, HIDDEN, bias=False)
        nn.init.zeros_(self.w.weight)

    def forward(self, base, x, b456, c):
        return base(x, b456, self.w(c))


class Gate(nn.Module):
    """V2/V3: x' = x * (1 + alpha*tanh(g(c))). g 마지막 층 0 초기화 → 시작 시 gate=1."""

    def __init__(self, c, d, alpha):
        super().__init__()
        self.alpha = alpha
        self.g1 = nn.Linear(c, 16)
        self.g2 = nn.Linear(16, d)
        nn.init.zeros_(self.g2.weight)
        nn.init.zeros_(self.g2.bias)

    def raw(self, c):
        return torch.tanh(self.g2(torch.tanh(self.g1(c))))

    def gate(self, c):
        return 1+self.alpha*self.raw(c)

    def forward(self, base, x, b456, c, gate_override=None):
        g = self.gate(c) if gate_override is None else gate_override
        return base(x*g, b456)


def sample_weights(y, experiments):
    keys = pd.DataFrame({'e': experiments, 'y': y})
    w = 1/keys.groupby(['e', 'y']).y.transform('size').to_numpy(float)
    return (w/w.mean()).astype(np.float32)


def _loop(params, forward, x, b, c, y, w, epochs, seed, penalty=None):
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-3)
    tx, tb, tc = map(torch.from_numpy, (x, b, c))
    ty = torch.from_numpy(y.astype(np.int64)); tw = torch.from_numpy(w)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=gen)
        for s in range(0, len(order), 256):
            i = order[s:s+256]
            loss = (F.cross_entropy(forward(tx[i], tb[i], tc[i]), ty[i], reduction='none')*tw[i]).mean()
            if penalty is not None:
                loss = loss+penalty(tc[i])
            opt.zero_grad(); loss.backward(); opt.step()


def train_all(x, b, c, y456, experiments, seed):
    """y456 ∈ {0,1,2}. V0을 학습한 뒤 동결하고 V1~V3 조건 모듈만 학습."""
    torch.manual_seed(seed)
    w = sample_weights(y456, experiments)
    base = MLP456(x.shape[1])
    base.train()
    _loop(base.parameters(), lambda xx, bb, cc: base(xx, bb), x, b, c, y456, w, BASE_EPOCHS, seed)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    models = {'V0': base}
    torch.manual_seed(seed+1)
    v1 = Concat(c.shape[1])
    _loop(v1.parameters(), lambda xx, bb, cc: v1(base, xx, bb, cc), x, b, c, y456, w, HEAD_EPOCHS, seed+1)
    models['V1'] = v1
    for name in ('V2', 'V3'):
        torch.manual_seed(seed+2)
        gate = Gate(c.shape[1], x.shape[1], GATE_ALPHA[name])
        lam = GATE_PENALTY[name]
        pen = (lambda cc, g=gate, l=lam: l*g.raw(cc).pow(2).mean()) if lam > 0 else None
        _loop(gate.parameters(), lambda xx, bb, cc, g=gate: g(base, xx, bb, cc), x, b, c, y456, w, HEAD_EPOCHS, seed+2, pen)
        models[name] = gate
    for m in models.values():
        m.eval()
    return models


@torch.no_grad()
def predict(models, name, x, b, c, gate_override=None):
    tx, tb, tc = map(torch.from_numpy, (x, b, c))
    base = models['V0']
    if name == 'V0':
        out = base(tx, tb)
    elif name == 'V1':
        out = models['V1'](base, tx, tb, tc)
    else:
        go = None if gate_override is None else torch.as_tensor(gate_override, dtype=torch.float32)
        out = models[name](base, tx, tb, tc, go)
    return F.softmax(out, 1).numpy()


@torch.no_grad()
def gates(models, name, c):
    return models[name].gate(torch.from_numpy(c)).numpy()


def apply456(base_pred, p456):
    """기준 예측이 4·5·6인 행만 MLP456의 argmax로 바꾼다."""
    out = np.asarray(base_pred, int).copy()
    m = out >= 4
    out[m] = p456[m].argmax(1)+4
    return out
