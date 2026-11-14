"""ABC B 전체 재학습 4조건 비교: 이미지 색 보정 × 가스 상태 입력.

진단 근거: E16의 실제 5단계를 4로 보는 원인은 센서가 아니라 열화상 이미지 색조였다
(센서를 train 5단계 값으로 바꿔도 4 유지, 이미지만 train 5단계로 바꾸면 5 예측 1%→56%).

  B           : 원래 ABC B (대조군, 같은 코드로 재학습)
  B_color     : 학습 시 이미지 채널별 색/밝기 증강 + 같은 실험 첫 프레임 대비 변화 이미지 3채널 추가
  B_gas       : 가스 상태 4개를 query 입력에 추가
                (농도 10종 동시 0 여부, 0 지속 시간, 이전에 가스가 측정된 적 있는지, 과거 중 0 비율)
  B_color_gas : 둘 다
  B_time      : 현재 경과시간 log1p(relative_time_s)/8 을 query 입력에 추가
  B_gas_time  : B_gas + 경과시간

모든 추가 정보는 같은 experiment_id의 현재·과거 관측만 사용한다(첫 프레임 = 실험 시작 시점).
fit/dev 분할, 전처리, weighted CE, batch64, AdamW lr3e-4 wd1e-4, 15epoch, dev 최고 epoch 선택은 원래 ABC와 같다.
Validation은 선택 후 평가만, Test는 조건별 3-seed 확률 평균으로 제출 파일 생성.

  python abc_color_gas.py --smoke --device cpu
  python -u abc_color_gas.py --output color_gas_runs/v1 --device cuda
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score

from abc_reference import abc_fusion_experiment as abc

common = abc.common
ROOT = Path(__file__).resolve().parent
CONC = [c for c in abc.GAS if c != 'meas_p']
ARMS = {'B': (False, False, False), 'B_color': (True, False, False), 'B_gas': (False, True, False),
        'B_color_gas': (True, True, False), 'B_time': (False, False, True), 'B_gas_time': (False, True, True)}
N_GAS = 4


def macro(y, p):
    return float(f1_score(y, p, labels=range(1, 7), average='macro', zero_division=0))


def gas_status(frame):
    """(n,4): 가스 0 여부, log1p(0 지속 초)/8, 과거에 가스 측정된 적 있음, 과거 중 0 비율. 현재·과거만."""
    f = frame[['experiment_id', 'relative_time_s']].copy()
    f['off'] = (frame[CONC].apply(pd.to_numeric, errors='coerce').fillna(0) == 0).all(axis=1).astype(float)
    order = f.sort_values(['experiment_id', 'relative_time_s'], kind='mergesort').index
    s = f.loc[order]; g = s.groupby('experiment_id')
    t = s.relative_time_s.astype(float)
    onset = s.off.astype(bool) & ~g.off.shift(1, fill_value=0).astype(bool)
    start = t.where(onset).groupby(s.experiment_id).ffill()
    dur = np.where(s.off > 0, (t - start).fillna(0).clip(lower=0), 0.0)
    had = (1 - s.off).groupby(s.experiment_id).cummax()
    frac = s.off.groupby(s.experiment_id).cumsum() / (g.cumcount() + 1)
    out = pd.DataFrame({'off': s.off, 'dur': np.log1p(dur) / 8, 'had': had, 'frac': frac}, index=order)
    return out.loc[frame.index].to_numpy(np.float32)


def first_frame_index(frame):
    """각 행과 같은 실험에서 가장 이른 관측의 행 번호(실험 시작 프레임)."""
    pos = pd.Series(np.arange(len(frame)), index=frame.index)
    t = frame.relative_time_s.astype(float)
    first = t.groupby(frame.experiment_id).transform('idxmin')
    return pos.loc[first.values].to_numpy()


class Split:
    def __init__(self, frame, images, prep, device):
        q, gas = prep.transform(frame)
        dev = images.device
        self.frame = frame
        self.images = images
        self.q = torch.from_numpy(q).to(dev)
        self.gs = torch.from_numpy(gas_status(frame)).to(dev)
        self.gas = torch.from_numpy(gas).to(dev)
        self.first = torch.from_numpy(first_frame_index(frame)).to(dev)
        self.time = torch.from_numpy((np.log1p(frame.relative_time_s.astype(float).clip(lower=0).to_numpy()) / 8)
                                     .astype(np.float32)[:, None]).to(dev)
        self.y = torch.tensor(frame.label.to_numpy(int) - 1, device=dev) if 'label' in frame else None

    def batch(self, ids, device, color, use_gas, augment, gen=None, use_time=False):
        ids = ids.to(self.images.device)
        cur = self.images.index_select(0, ids)
        img = cur[:, :3].to(device, dtype=torch.float32).div_(255)
        mask = cur[:, 3:4].to(device)
        if color:
            first = self.images.index_select(0, self.first.index_select(0, ids))[:, :3].to(device, dtype=torch.float32).div_(255)
            if augment:  # 같은 실험·같은 카메라 가정: 현재/첫 프레임에 같은 색 변환
                n = len(ids)
                gain = (torch.rand(n, 3, 1, 1, generator=gen) * 0.5 + 0.75).to(device)
                bias = (torch.rand(n, 3, 1, 1, generator=gen) * 0.2 - 0.1).to(device)
                img = (img * gain + bias).clamp_(0, 1); first = (first * gain + bias).clamp_(0, 1)
            m = (mask > 0).float()
            img = torch.cat([img, (img - first) * m], 1)
        q = self.q.index_select(0, ids).to(device)
        if use_gas:
            q = torch.cat([q, self.gs.index_select(0, ids).to(device)], 1)
        if use_time:
            q = torch.cat([q, self.time.index_select(0, ids).to(device)], 1)
        return img, mask, q, self.gas.index_select(0, ids).to(device)


class Model(abc.FusionModel):
    def __init__(self, color, use_gas, use_time=False, d=64):
        super().__init__('B', d)
        if color:
            self.cnn[0] = nn.Conv2d(6, 16, 3, stride=2, padding=1, bias=False)
        extra = N_GAS * use_gas + use_time
        if extra:
            self.q_encoder[0] = nn.Linear(2 * len(abc.ELECTRO) + extra, d)


@torch.no_grad()
def predict(model, split, color, use_gas, device, use_time=False):
    model.eval(); out = []
    for s in range(0, len(split.frame), 256):
        ids = torch.arange(s, min(s + 256, len(split.frame)))
        out.append(model(*split.batch(ids, device, color, use_gas, False, use_time=use_time)).softmax(-1).cpu().numpy())
    return np.concatenate(out)


def train(arm, seed, data, prep, args, device):
    color, use_gas, use_time = ARMS[arm]
    common.seed_everything(seed)
    model = Model(color, use_gas, use_time).to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(prep.weights, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed); aug = torch.Generator().manual_seed(seed + 1000)
    fit = data['fit']; best, state, best_ep = -1, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(fit.frame), generator=gen)
        for s in range(0, len(order), 64):
            ids = order[s:s + 64]
            y = fit.y.index_select(0, ids.to(fit.y.device)).to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(*fit.batch(ids, device, color, use_gas, True, aug, use_time)), y)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
        f = macro(data['dev'].frame.label, predict(model, data['dev'], color, use_gas, device, use_time).argmax(1) + 1)
        if f > best:
            best, best_ep, state = f, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f'  {arm} seed{seed} ep{ep:02d} dev {f:.4f} best {best:.4f}', flush=True)
    model.load_state_dict(state)
    probs = {s: predict(model, data[s], color, use_gas, device, use_time) for s in ['dev', 'validation', 'test']}
    return probs, best_ep


def load_images(args, frames):
    manifest = args.base_run / 'image_cache_manifest.json'
    if not args.smoke and manifest.exists():
        sm = pd.read_csv(args.base_run / 'split_manifest.csv')
        paths = json.loads(manifest.read_text())
        ok = all(np.array_equal(frames[s].sample_id.values, sm.sample_id[sm.split == s].values) for s in frames)
        if ok:
            imgs = {s: np.load(paths[s], mmap_mode='c') for s in frames}
            if all(imgs[s].shape == (len(frames[s]), 4, 128, 128) for s in frames):
                print('이미지 캐시: 0.91 run 캐시 재사용(행 순서 검증 완료)', flush=True)
                return {s: torch.from_numpy(np.ascontiguousarray(v)) for s, v in imgs.items()}
    return {s: torch.from_numpy(np.ascontiguousarray(common.cache_images(
        args.data_root, f, 'train' if s in ['fit', 'dev'] else s, ROOT / '.abc_cache', 128, 4, 8)[0]))
        for s, f in frames.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', type=Path, default=Path('/home/ubuntu/data/battery_data_light'))
    ap.add_argument('--base-run', type=Path, default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    ap.add_argument('--output', type=Path, default=ROOT / 'color_gas_runs/v1')
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    ap.add_argument('--split-seed', type=int, default=20261114)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = args.seeds[:1], 1
        if args.output == ROOT / 'color_gas_runs/v1': args.output = ROOT / 'color_gas_runs/smoke'
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f'{args.output} 가 이미 있음. 새 --output 경로를 지정하세요.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    device = torch.device(args.device); torch.set_num_threads(4); t0 = time.monotonic()

    frames = abc.load_frames(argparse.Namespace(data_root=args.data_root, split_seed=args.split_seed, smoke=args.smoke))
    for s, f in frames.items(): print(s, len(f), 'rows', f.experiment_id.nunique(), 'experiments', flush=True)
    imgs = load_images(args, frames)
    need = sum(v.numel() for v in imgs.values())
    if device.type == 'cuda' and torch.cuda.mem_get_info()[0] > need + 4 * 1024 ** 3:
        imgs = {s: v.to(device) for s, v in imgs.items()}
    print(f'image cache {need / 1024 ** 3:.2f} GiB on {next(iter(imgs.values())).device}', flush=True)
    prep = abc.SensorPreprocessor(frames['fit'])
    data = {s: Split(f, imgs[s], prep, device) for s, f in frames.items()}

    va = frames['validation']; yv = va.label.to_numpy(int); yd = frames['dev'].label.to_numpy(int)
    vexp = sorted(va.experiment_id.unique()); e16 = vexp[15] if len(vexp) >= 16 else None
    not16 = (va.experiment_id != e16).to_numpy(); t5 = ((va.experiment_id == e16) & (va.label == 5)).to_numpy()

    def scores(pd_, pv):
        a, b = pd_.argmax(1) + 1, pv.argmax(1) + 1
        f = f1_score(yv, b, labels=range(1, 7), average=None, zero_division=0)
        return dict(dev_f1=macro(yd, a), val_f1=macro(yv, b), val_noE16_f1=macro(yv[not16], b[not16]),
                    E16_5_correct=int((b[t5] == 5).sum()), E16_5_to_4=int((b[t5] == 4).sum()),
                    **{f'val_F1_{i + 1}': float(x) for i, x in enumerate(f)})

    runs, ens = [], []
    sample = pd.read_csv(args.data_root / 'sample_submission.csv')
    for arm in args.arms:
        acc = {s: 0 for s in ['dev', 'validation', 'test']}
        for seed in args.seeds:
            probs, ep = train(arm, seed, data, prep, args, device)
            for s in acc: acc[s] = acc[s] + probs[s] / len(args.seeds)
            r = dict(arm=arm, seed=seed, best_epoch=ep, **scores(probs['dev'], probs['validation']))
            runs.append(r); print('RUN', json.dumps(r), flush=True)
            pd.DataFrame(runs).to_csv(args.output / 'runs.csv', index=False)
            if device.type == 'cuda': torch.cuda.empty_cache()
        e = dict(arm=arm, seeds=len(args.seeds), **scores(acc['dev'], acc['validation']))
        ens.append(e); pd.DataFrame(ens).to_csv(args.output / 'ensemble.csv', index=False)
        if not args.smoke:
            pred = pd.Series(acc['test'].argmax(1) + 1, index=frames['test'].sample_id)
            sub = sample[['sample_id']].copy(); sub['label'] = sub.sample_id.map(pred).astype(int)
            sub.to_csv(args.output / f'submission_{arm}.csv', index=False)
    cols = ['arm', 'dev_f1', 'val_f1', 'val_noE16_f1', 'E16_5_correct', 'E16_5_to_4'] + [f'val_F1_{i}' for i in range(1, 7)]
    print('\n== 3-seed 앙상블\n' + pd.DataFrame(ens)[cols].round(4).to_string(index=False))
    print(f'\n총 {(time.monotonic() - t0) / 60:.1f} 분. 결과: {args.output}', flush=True)


if __name__ == '__main__':
    main()
