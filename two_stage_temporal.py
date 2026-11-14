"""투스테이지 시계열: Stage1 ABC B(현재 관측) → Stage2 인과 GRU(같은 실험의 과거 Stage1 확률 + 센서).

비교 모드
  S1          : ABC B 단독(fold 재학습)
  S2_full     : Stage1 확률 + 센서 시계열을 단방향 GRU로 보정. 초기 출력 = Stage1과 동일
  S2_branch45 : Stage1이 4/5로 예측한 행만 S2_full의 p4/p5로 4↔5 재판정, 나머지는 Stage1 그대로
  S2_gasoff   : 이상치(가스 10종 동시 0 = 측정 끊김) 행만 S2_full로 교체, 나머지는 Stage1 그대로
  S2_gasoff45 : 이상치 행 중 Stage1이 4/5로 예측한 행만 4↔5 재판정
  S2_anomaly  : 이상치(가스 끊김 | 전압 급락 | 급락 직후 반등) 행만 S2_full로 교체
  S2_noS1     : Stage1 없이 센서 시계열 GRU만(대조군)

평가: train 실험 단위 GroupKFold(주 지표, Stage1도 fold마다 재학습해 OOF 확률 사용),
      validation(전체/E16 제외), 각 모드별 test 제출 파일.
GRU는 단방향이며 같은 experiment_id의 relative_time_s ≤ 현재 관측만 사용한다(미래 미사용).
학습 시 일부 구간을 15초 간격으로 솎는 증강으로 test의 간격 차이를 반영한다.

  python two_stage_temporal.py --smoke --device cpu          # 작은 부분집합으로 동작 확인
  python -u two_stage_temporal.py --data-dir /home/ubuntu/data/battery_data_light \
      --output two_stage_runs/v1 --device cuda
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from abc_reference import abc_fusion_experiment as abc

common = abc.common
ROOT = Path(__file__).resolve().parent
SENS = abc.ELECTRO + abc.GAS            # 전기·열 6 + 압력 + 가스 10
CONC = abc.GAS[1:]                      # 압력 제외 가스 농도 10종


# ---------------------------------------------------------------- data
def load(root, smoke):
    fr = {s: pd.read_csv(root / s / f'{s}.csv') for s in ['train', 'validation', 'test']}
    if smoke:  # 실험 몇 개, 전 구간을 8행 간격으로 솎음(모든 단계 유지)
        for s, n in [('train', 12), ('validation', 3), ('test', 2)]:
            f = fr[s]; keep = sorted(f.experiment_id.unique())[:n]
            f = f[f.experiment_id.isin(keep)].sort_values(['experiment_id', 'relative_time_s'])
            fr[s] = f[f.groupby('experiment_id').cumcount() % 8 == 0]
    for s in fr:
        fr[s] = fr[s].sort_values(['experiment_id', 'relative_time_s'], kind='mergesort').reset_index(drop=True)
    return fr


def blocks(frame):
    e = frame.experiment_id.to_numpy()
    cut = np.r_[0, np.flatnonzero(e[1:] != e[:-1]) + 1, len(e)]
    return [np.arange(cut[i], cut[i + 1]) for i in range(len(cut) - 1)]


def class_weights(y):
    c = np.bincount(np.asarray(y) - 1, minlength=6).astype(np.float32)
    w = np.where(c > 0, c, 1) ** -0.5
    return w / w.mean()


def macro(y, p):
    return float(f1_score(y, p, labels=range(1, 7), average='macro', zero_division=0))


class View:
    """abc.Data의 부분집합(이미지 텐서 복사 없음)."""
    def __init__(self, base, idx):
        self.base, self.idx = base, torch.as_tensor(np.asarray(idx), dtype=torch.long)
        self.frame = base.frame.iloc[np.asarray(idx)].reset_index(drop=True)
        self.y = None if base.y is None else base.y.index_select(0, self.idx.to(base.y.device))

    def batch(self, ids, device):
        return self.base.batch(self.idx[ids.cpu()], device)


def make_data(frame, images, prep):
    q, gas = prep.transform(frame)
    dev = images.device
    y = torch.tensor(frame.label.to_numpy(int) - 1, device=dev) if 'label' in frame else None
    return abc.Data(frame, images, torch.from_numpy(q).to(dev), torch.from_numpy(gas).to(dev), y)


# ---------------------------------------------------------------- stage 1 (ABC B)
def train_stage1(fit, sel, weights, seed, epochs, device):
    common.seed_everything(seed)
    model = abc.FusionModel('B').to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed); best, state = -1, None
    for ep in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(fit.frame), generator=gen)
        for s in range(0, len(order), 64):
            ids = order[s:s + 64]
            y = fit.y.index_select(0, ids.to(fit.y.device)).to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(*fit.batch(ids, device)), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
        f = macro(sel.frame.label, abc.predict(model, sel, None, device).argmax(1) + 1)
        if f > best:
            best, state = f, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f'  S1 seed{seed} ep{ep:02d} sel {f:.4f} best {best:.4f}', flush=True)
    model.load_state_dict(state)
    return model


def run_stage1(fr, imgs, folds, args, device):
    tr = fr['train']; n = len(tr)
    oof = np.zeros((n, 6)); pv = np.zeros((len(fr['validation']), 6)); pt = np.zeros((len(fr['test']), 6))
    for seed in args.seeds:
        for k, (a, b) in enumerate(folds):
            t0 = time.monotonic()
            i, j = next(GroupShuffleSplit(1, test_size=.15, random_state=seed + k).split(a, groups=tr.experiment_id.values[a]))
            fit_i, sel_i = a[i], a[j]
            prep = abc.SensorPreprocessor(tr.iloc[fit_i])
            base = {s: make_data(fr[s], imgs[s], prep) for s in fr}
            model = train_stage1(View(base['train'], fit_i), View(base['train'], sel_i), prep.weights, seed, args.s1_epochs, device)
            oof[b] += abc.predict(model, View(base['train'], b), None, device) / len(args.seeds)
            pv += abc.predict(model, base['validation'], None, device) / (len(folds) * len(args.seeds))
            pt += abc.predict(model, base['test'], None, device) / (len(folds) * len(args.seeds))
            print(f'S1 seed{seed} fold{k} OOF {macro(tr.label.values[b], oof[b].argmax(1) + 1):.4f} '
                  f'({(time.monotonic() - t0) / 60:.1f} min)', flush=True)
            del model, base
            if device.type == 'cuda': torch.cuda.empty_cache()
    return oof, pv, pt


# ---------------------------------------------------------------- stage 2 (causal GRU)
class SensorStats:
    def __init__(self, frame):
        x = frame[SENS].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        self.med = x.median().fillna(0); x = x.fillna(self.med)
        self.mu, self.sd = x.mean(), x.std(ddof=0).clip(lower=1e-6)

    def __call__(self, frame):
        x = frame[SENS].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
        z = ((x.fillna(self.med) - self.mu) / self.sd).clip(-10, 10).to_numpy(np.float32)
        gas_off = (x[CONC].fillna(0) == 0).all(axis=1).to_numpy(np.float32)[:, None]  # 가스 10종 동시 0 = 측정 끊김
        return np.concatenate([z, gas_off], 1)


def s2_inputs(frame, p1, stats, use_s1):
    x = stats(frame)
    if use_s1:
        x = np.concatenate([np.log(np.clip(p1, 1e-4, 1)).astype(np.float32), x], 1)  # 앞 6열 = log Stage1 확률
    return x


class Stage2(nn.Module):
    def __init__(self, nin, use_s1, h=64):
        super().__init__()
        self.use_s1 = use_s1
        self.inp = nn.Sequential(nn.Linear(nin, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.gru = nn.GRU(h, h, num_layers=2, batch_first=True, dropout=0.1)  # 단방향 = 인과
        self.out = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 6))
        if use_s1:  # 시작 시 출력 = Stage1 그대로, 학습된 만큼만 보정
            nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x):
        z = self.inp(x); h, _ = self.gru(z)
        logit = self.out(torch.cat([z, h], -1))
        return logit + x[..., :6] if self.use_s1 else logit


def collate(bl, X, t, y, rng, augment, device):
    seqs = []
    for b in bl:
        if augment and len(b) > 10 and rng.random() < 0.5:  # 30% 구간을 15초 간격으로 솎음(test 모사)
            L = int(len(b) * 0.3); s = rng.integers(0, len(b) - L + 1)
            keep = np.ones(len(b), bool); seg = np.arange(s, s + L); keep[seg[(seg - s) % 3 != 0]] = False
            b = b[keep]
        seqs.append(b)
    T = max(len(b) for b in seqs)
    xb = np.zeros((len(seqs), T, X.shape[1] + 2), np.float32); yb = np.full((len(seqs), T), -100, np.int64)
    for i, b in enumerate(seqs):
        tt = t[b]; dt = np.diff(tt, prepend=tt[0])
        xb[i, :len(b), :X.shape[1]] = X[b]
        xb[i, :len(b), -2] = np.log1p(np.clip(dt, 0, None)) / 3
        xb[i, :len(b), -1] = np.log1p(np.clip(tt, 0, None)) / 8
        if y is not None: yb[i, :len(b)] = y[b] - 1
    return torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device)


@torch.no_grad()
def predict2(model, bl, X, t, n, device):
    model.eval(); out = np.zeros((n, 6), np.float32)
    for b in bl:
        xb, _ = collate([b], X, t, None, None, False, device)
        out[b] = model(xb)[0].softmax(-1).cpu().numpy()
    return out


def train_stage2(bl_fit, bl_sel, X, t, y, use_s1, seed, epochs, device):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Stage2(X.shape[1] + 2, use_s1).to(device)
    fit_rows = np.concatenate(bl_fit)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y[fit_rows]), device=device), ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sel_rows = np.concatenate(bl_sel)

    def score():
        return macro(y[sel_rows], predict2(model, bl_sel, X, t, len(y), device)[sel_rows].argmax(1) + 1)
    best = score() if use_s1 else -1  # use_s1: epoch0(=Stage1)도 후보 → 보정이 해로우면 Stage1로 복귀
    state = {k: v.clone() for k, v in model.state_dict().items()}
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(bl_fit))
        for s in range(0, len(order), 8):
            xb, yb = collate([bl_fit[i] for i in order[s:s + 8]], X, t, y, rng, True, device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb).reshape(-1, 6), yb.reshape(-1)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        f = score()
        if f > best:
            best, state = f, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, best


def fit_predict_s2(tr, p1_tr, y, rows_fit, targets, use_s1, seed, args, device):
    """rows_fit(train 행)으로 Stage2 학습 → targets [(frame, p1)] 예측."""
    sub = tr.iloc[rows_fit]
    exps = sub.experiment_id.unique()
    rng = np.random.default_rng(seed); rng.shuffle(exps)
    sel_exps = set(exps[:max(1, int(round(len(exps) * .15)))])
    stats = SensorStats(sub)
    X = s2_inputs(tr, p1_tr, stats, use_s1); t = tr.relative_time_s.to_numpy(float)
    rows_set = set(rows_fit.tolist())
    bl = [b for b in blocks(tr) if b[0] in rows_set]
    bl_fit = [b for b in bl if tr.experiment_id.iat[b[0]] not in sel_exps]
    bl_sel = [b for b in bl if tr.experiment_id.iat[b[0]] in sel_exps]
    model, _ = train_stage2(bl_fit, bl_sel, X, t, y, use_s1, seed, args.s2_epochs, device)
    outs = []
    for frame, p1 in targets:
        Xf = s2_inputs(frame, p1, stats, use_s1)
        outs.append(predict2(model, blocks(frame), Xf, frame.relative_time_s.to_numpy(float), len(frame), device))
    return outs


def branch45(p1, p2, only=None):
    pred = p1.argmax(1) + 1
    m = np.isin(pred, [4, 5]) if only is None else np.isin(pred, [4, 5]) & only
    pred[m] = np.where(p2[m, 4] > p2[m, 3], 5, 4)
    return pred


def gas_off(frame):
    """이상치: 가스 농도 10종이 동시에 정확히 0(측정 끊김). 이 행에서 ABC B 오류율이 크게 높았다."""
    return (frame[CONC].apply(pd.to_numeric, errors='coerce').fillna(0) == 0).all(axis=1).to_numpy()


V_DROP, V_SPIKE = -0.048, 0.034  # meas_v 변화율(/초). sensor_event_analysis가 fit에서 정한 하위/상위 1%


def anomaly(frame):
    """이상치: 가스 끊김 OR 전압 급락(현재 구간) OR 급락 직후 반등. 같은 실험의 직전 관측만 사용."""
    v = pd.to_numeric(frame.meas_v, errors='coerce').to_numpy(float)
    t = frame.relative_time_s.to_numpy(float)
    same = np.r_[False, frame.experiment_id.to_numpy()[1:] == frame.experiment_id.to_numpy()[:-1]]
    rate = np.full(len(v), np.nan)
    rate[1:] = (v[1:] - v[:-1]) / np.maximum(t[1:] - t[:-1], 1e-6)
    rate[~same] = np.nan
    drop = np.nan_to_num(rate, nan=0) < V_DROP
    spike = np.nan_to_num(rate, nan=0) > V_SPIKE
    prev_drop = np.r_[False, drop[:-1]] & same
    return gas_off(frame) | drop | (spike & prev_drop)


def only_where(p1, p2, mask):
    pred = p1.argmax(1) + 1
    pred[mask] = p2[mask].argmax(1) + 1
    return pred


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', type=Path, default=Path('/home/ubuntu/data/battery_data_light'))
    ap.add_argument('--output', type=Path, default=ROOT / 'two_stage_runs/v1')
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--s1-epochs', type=int, default=15)
    ap.add_argument('--s2-epochs', type=int, default=40)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--stage1-npz', type=Path, help='이전 실행의 stage1_probs.npz 재사용(같은 데이터·fold 수일 때). Stage1 학습 생략')
    args = ap.parse_args()
    if args.smoke:
        args.folds, args.s1_epochs, args.s2_epochs = 3, 1, 2
        if args.output == ROOT / 'two_stage_runs/v1': args.output = ROOT / 'two_stage_runs/smoke'
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f'{args.output} 가 이미 있음. 새 --output 경로를 지정하세요.')
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device); torch.set_num_threads(4)
    (args.output / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    t_start = time.monotonic()

    fr = load(args.data_dir, args.smoke)
    for s, f in fr.items(): print(s, len(f), 'rows', f.experiment_id.nunique(), 'experiments', flush=True)
    tr, va, te = fr['train'], fr['validation'], fr['test']
    y = tr.label.to_numpy(int)
    folds = list(GroupKFold(args.folds).split(tr, groups=tr.experiment_id))
    if args.stage1_npz:
        z = np.load(args.stage1_npz)
        s1_oof, s1_va, s1_te = z['oof'], z['validation'], z['test']
        if (len(s1_oof), len(s1_va), len(s1_te)) != (len(tr), len(va), len(te)):
            raise SystemExit('stage1_npz 행 수가 현재 데이터와 다름')
        print(f'Stage1 재사용: {args.stage1_npz}', flush=True)
    else:
        s1_oof, s1_va, s1_te = stage1_all(fr, args, device)
    np.savez(args.output / 'stage1_probs.npz', oof=s1_oof, validation=s1_va, test=s1_te)
    print(f'S1 train OOF macro F1 {macro(y, s1_oof.argmax(1) + 1):.4f}', flush=True)
    stage2_and_report(fr, folds, y, s1_oof, s1_va, s1_te, args, device, t_start)


def stage1_all(fr, args, device):
    tr = fr['train']
    folds = list(GroupKFold(args.folds).split(tr, groups=tr.experiment_id))
    imgs = {}
    for s in fr:
        arr = common.cache_images(args.data_dir, fr[s], s, ROOT / '.abc_cache', 128, 4, 8)[0]
        imgs[s] = torch.from_numpy(np.ascontiguousarray(arr))
    need = sum(v.numel() for v in imgs.values())
    if device.type == 'cuda' and torch.cuda.mem_get_info()[0] > need + 4 * 1024 ** 3:
        imgs = {s: v.to(device) for s, v in imgs.items()}
    print(f'image cache {need / 1024 ** 3:.2f} GiB on {next(iter(imgs.values())).device}', flush=True)
    out = run_stage1(fr, imgs, folds, args, device)
    del imgs
    if device.type == 'cuda': torch.cuda.empty_cache()
    return out


def stage2_and_report(fr, folds, y, s1_oof, s1_va, s1_te, args, device, t_start):
    tr, va, te = fr['train'], fr['validation'], fr['test']
    # Stage 2: train OOF (fold 동일) + 전체 train 학습 → validation/test
    preds = {'S1': {'oof': s1_oof.argmax(1) + 1, 'validation': s1_va.argmax(1) + 1, 'test': s1_te.argmax(1) + 1}}
    for use_s1, name in [(True, 'S2_full'), (False, 'S2_noS1')]:
        oof = np.zeros((len(tr), 6)); pv = np.zeros((len(va), 6)); pt = np.zeros((len(te), 6))
        for seed in args.seeds:
            for k, (a, b) in enumerate(folds):
                (o,) = fit_predict_s2(tr, s1_oof, y, a, [(tr, s1_oof)], use_s1, seed + k, args, device)
                oof[b] += o[b] / len(args.seeds)
            v, t_ = fit_predict_s2(tr, s1_oof, y, np.arange(len(tr)), [(va, s1_va), (te, s1_te)], use_s1, seed, args, device)
            pv += v / len(args.seeds); pt += t_ / len(args.seeds)
        print(f'{name} train OOF macro F1 {macro(y, oof.argmax(1) + 1):.4f}', flush=True)
        preds[name] = {'oof': oof.argmax(1) + 1, 'validation': pv.argmax(1) + 1, 'test': pt.argmax(1) + 1}
        if use_s1:
            preds['S2_branch45'] = {'oof': branch45(s1_oof, oof), 'validation': branch45(s1_va, pv), 'test': branch45(s1_te, pt)}
            # 이상치(가스 끊김) 행만 보정, 나머지는 ABC B 그대로
            go = {'oof': gas_off(tr), 'validation': gas_off(va), 'test': gas_off(te)}
            P1 = {'oof': s1_oof, 'validation': s1_va, 'test': s1_te}; P2 = {'oof': oof, 'validation': pv, 'test': pt}
            preds['S2_gasoff'] = {s: only_where(P1[s], P2[s], go[s]) for s in go}
            preds['S2_gasoff45'] = {s: branch45(P1[s], P2[s], go[s]) for s in go}
            an = {'oof': anomaly(tr), 'validation': anomaly(va), 'test': anomaly(te)}
            preds['S2_anomaly'] = {s: only_where(P1[s], P2[s], an[s]) for s in an}

    # 집계(행별 정보는 저장/출력하지 않음, 제출 파일 제외)
    yv = va.label.to_numpy(int)
    vexp = sorted(va.experiment_id.unique())
    e16 = vexp[15] if len(vexp) >= 16 else None  # 기존 보고서 별칭 E16 = 정렬 16번째
    not16 = (va.experiment_id != e16).to_numpy()
    is16_5 = ((va.experiment_id == e16) & (va.label == 5)).to_numpy()
    go_tr, go_va, go_te = gas_off(tr), gas_off(va), gas_off(te)
    gas_off_hi = go_tr & (y >= 4)
    print(f'가스 끊김 행: train {go_tr.mean():.1%}, validation {go_va.mean():.1%}, test {go_te.mean():.1%}', flush=True)
    print(f'이상치(가스끊김|전압급락|반등) 행: train {anomaly(tr).mean():.1%}, validation {anomaly(va).mean():.1%}, '
          f'test {anomaly(te).mean():.1%}', flush=True)
    rows, classes = [], []
    sample = pd.read_csv(args.data_dir / 'sample_submission.csv')
    for name in ['S1', 'S2_full', 'S2_branch45', 'S2_gasoff', 'S2_gasoff45', 'S2_anomaly', 'S2_noS1']:
        p = preds[name]
        r = dict(mode=name, train_cv_f1=macro(y, p['oof']), val_f1=macro(yv, p['validation']),
                 val_noE16_f1=macro(yv[not16], p['validation'][not16]),
                 E16_5to4=int((p['validation'][is16_5] == 4).sum()),
                 train_gasoff_4to6_err=float((p['oof'][gas_off_hi] != y[gas_off_hi]).mean()) if gas_off_hi.any() else np.nan,
                 train_gasoff_acc=float((p['oof'][go_tr] == y[go_tr]).mean()) if go_tr.any() else np.nan,
                 val_gasoff_acc=float((p['validation'][go_va] == yv[go_va]).mean()) if go_va.any() else np.nan,
                 test_changed_vs_S1=int((p['test'] != preds['S1']['test']).sum()))
        rows.append(r)
        cf = f1_score(y, p['oof'], labels=range(1, 7), average=None, zero_division=0)
        classes.append(dict(mode=name, **{f'trainCV_F1_{i + 1}': float(c) for i, c in enumerate(cf)}))
        pred = pd.Series(p['test'], index=te.sample_id)
        if not args.smoke:
            sub = sample[['sample_id']].copy(); sub['label'] = sub.sample_id.map(pred).astype(int)
            sub.to_csv(args.output / f'submission_{name}.csv', index=False)
    summ = pd.DataFrame(rows); summ.to_csv(args.output / 'summary.csv', index=False)
    pd.DataFrame(classes).to_csv(args.output / 'per_class.csv', index=False)
    print('\n' + summ.round(4).to_string(index=False))
    print('\n' + pd.DataFrame(classes).round(3).to_string(index=False))
    print(f'\n총 {(time.monotonic() - t_start) / 60:.1f} 분. 결과: {args.output}', flush=True)


if __name__ == '__main__':
    main()
