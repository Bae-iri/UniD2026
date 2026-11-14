"""최종 앙상블: (B, B_gas, B_time) × seed(42,43,44) = 9개 ABC B 계열 모델, 고정 epoch, 확률 평균.

  check : fit(62실험)으로 학습 → dev / validation(전체, E16 제외)로 레시피 검증. 기준 = 0.91 모델(dev 0.7815, noE16 0.8142)
  final : fit+dev+validation(97실험) 전부로 학습 → test 제출 파일

dev로 epoch을 고르지 않고 고정 epoch(기본 11 = 기존 run 최고 epoch 중앙값)을 쓴다. 이미지·전처리·손실·optimizer는 원래 ABC와 같다.

  python -u final_ensemble.py --mode check --output final_runs/check --device cuda
  python -u final_ensemble.py --mode final --output final_runs/final --device cuda
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score

from abc_reference import abc_fusion_experiment as abc
import abc_color_gas as cg

ROOT = Path(__file__).resolve().parent


def macro(y, p):
    return float(f1_score(y, p, labels=range(1, 7), average='macro', zero_division=0))


def fit_fixed(arm, seed, split, prep, epochs, device):
    color, use_gas, use_time = cg.ARMS[arm]
    abc.common.seed_everything(seed)
    model = cg.Model(color, use_gas, use_time).to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(prep.weights, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed); aug = torch.Generator().manual_seed(seed + 1000)
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.
        order = torch.randperm(len(split.frame), generator=gen)
        for s in range(0, len(order), 64):
            ids = order[s:s + 64]
            y = split.y.index_select(0, ids.to(split.y.device)).to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(*split.batch(ids, device, color, use_gas, True, aug, use_time)), y)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            tot += loss.item() * len(ids)
        print(f'  {arm} seed{seed} ep{ep:02d} CE {tot / len(order):.4f}', flush=True)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['check', 'final'], required=True)
    ap.add_argument('--data-root', type=Path, default=Path('/home/ubuntu/data/battery_data_light'))
    ap.add_argument('--base-run', type=Path, default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--arms', nargs='+', default=['B', 'B_gas', 'B_time'], choices=list(cg.ARMS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=11)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke: a.seeds, a.epochs = a.seeds[:1], 1
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit(f'{a.output} 가 이미 있음. 새 --output 경로를 지정하세요.')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))
    device = torch.device(a.device); torch.set_num_threads(4); t0 = time.monotonic()

    frames = abc.load_frames(argparse.Namespace(data_root=a.data_root, split_seed=20261114, smoke=a.smoke))
    imgs = cg.load_images(argparse.Namespace(base_run=a.base_run, smoke=a.smoke, data_root=a.data_root), frames)
    parts = ['fit'] if a.mode == 'check' else ['fit', 'dev', 'validation']
    train_frame = pd.concat([frames[p] for p in parts], ignore_index=True)
    train_imgs = torch.cat([imgs[p] for p in parts]) if len(parts) > 1 else imgs['fit']
    evals = {'dev': frames['dev'], 'validation': frames['validation'], 'test': frames['test']} if a.mode == 'check' \
        else {'test': frames['test']}
    store = {k: imgs[k] for k in evals}
    need = train_imgs.numel() + sum(v.numel() for v in store.values())
    if device.type == 'cuda' and torch.cuda.mem_get_info()[0] > need + 4 * 1024 ** 3:
        train_imgs = train_imgs.to(device); store = {k: v.to(device) for k, v in store.items()}
    print(f'mode={a.mode} 학습 {len(train_frame)}행 {train_frame.experiment_id.nunique()}실험, '
          f'이미지 {need / 1024 ** 3:.2f} GiB on {train_imgs.device}', flush=True)
    prep = abc.SensorPreprocessor(train_frame)
    train = cg.Split(train_frame, train_imgs, prep, device)
    data = {k: cg.Split(f, store[k], prep, device) for k, f in evals.items()}

    probs = {}  # (arm, seed) -> {split: prob}
    for arm in a.arms:
        color, use_gas, use_time = cg.ARMS[arm]
        for seed in a.seeds:
            m = fit_fixed(arm, seed, train, prep, a.epochs, device)
            probs[(arm, seed)] = {k: cg.predict(m, d, color, use_gas, device, use_time) for k, d in data.items()}
            del m
            if device.type == 'cuda': torch.cuda.empty_cache()
            print(f'done {arm} seed{seed} ({(time.monotonic() - t0) / 60:.1f} min)', flush=True)
    np.savez(a.output / 'test_probs.npz', **{f'{arm}_{s}': p['test'] for (arm, s), p in probs.items()})

    groups = {f'{arm} 3seed': [k for k in probs if k[0] == arm] for arm in a.arms}
    groups['전체 앙상블'] = list(probs)
    if a.mode == 'check':
        va = frames['validation']; yv = va.label.to_numpy(int); yd = frames['dev'].label.to_numpy(int)
        vexp = sorted(va.experiment_id.unique()); e16 = vexp[15] if len(vexp) >= 16 else None
        keep = (va.experiment_id != e16).to_numpy(); t5 = ((va.experiment_id == e16) & (va.label == 5)).to_numpy()
        rows = []
        for name, keys in list(groups.items()) + [(f'{k[0]} seed{k[1]}', [k]) for k in probs]:
            pdv = np.mean([probs[k]['dev'] for k in keys], 0).argmax(1) + 1
            pv = np.mean([probs[k]['validation'] for k in keys], 0).argmax(1) + 1
            rows.append(dict(model=name, dev_f1=macro(yd, pdv), val_f1=macro(yv, pv), val_noE16_f1=macro(yv[keep], pv[keep]),
                             E16_5_correct=int((pv[t5] == 5).sum())))
        df = pd.DataFrame(rows); df.to_csv(a.output / 'check.csv', index=False)
        print('\n기준(0.91 모델): dev 0.7815, val 0.6963, val_noE16 0.8142, E16_5_correct 6')
        print(df.round(4).to_string(index=False))

    sample = pd.read_csv(a.data_root / 'sample_submission.csv')
    test = frames['test']; B0 = pd.read_csv(a.base_run / 'submission_B.csv').set_index('sample_id').label
    for name, keys in groups.items():
        if a.smoke: break
        pred = pd.Series(np.mean([probs[k]['test'] for k in keys], 0).argmax(1) + 1, index=test.sample_id)
        sub = sample[['sample_id']].copy(); sub['label'] = sub.sample_id.map(pred).astype(int)
        assert len(sub) == len(sample) and sub.label.between(1, 6).all()
        fn = a.output / f"submission_{a.mode}_{name.replace(' ', '_')}.csv"; sub.to_csv(fn, index=False)
        print(f'{fn.name}: 0.91 파일 대비 변경 {(sub.set_index("sample_id").label != B0.loc[sub.sample_id].values).sum()}행')
    print(f'총 {(time.monotonic() - t0) / 60:.1f} 분', flush=True)


if __name__ == '__main__':
    main()
