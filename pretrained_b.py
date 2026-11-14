"""ABC B + 경과시간(B_time)에서 작은 CNN만 ImageNet 사전학습 ResNet18(layer3까지)로 교체.

사전학습 모델: torchvision resnet18, ResNet18_Weights.IMAGENET1K_V1 (BSD-3-Clause, torchvision 0.26).
센서 query, 가스/이미지 attention, 직접 경로, 전처리, weighted CE, batch64는 원래 ABC와 같다.
backbone lr 1e-4, 나머지 3e-4. 입력 128px, ImageNet 정규화, 패딩 마스크 유지.

  check : fit(62실험) 학습, dev 최고 epoch 선택 → dev / validation 평가, fit 모델로 test 제출 파일
  final : fit+dev+validation(97실험) 전부, --epochs 고정 → test 제출 파일

  python -u pretrained_b.py --mode check --output pre_runs/check --device cuda
  python -u pretrained_b.py --mode final --epochs <check의 best epoch 중앙값> --output pre_runs/final --device cuda
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from torch import nn
from sklearn.metrics import f1_score

from abc_reference import abc_fusion_experiment as abc
import abc_color_gas as cg

LONG4 = 575  # train 78개 실험에서 실제 4단계 최장 지속시간(초)


def long4_rule(frame, pred, th=LONG4):
    """같은 실험에서 4 예측이 th초 넘게 이어지면 이후 4를 5로(과거 예측만 사용)."""
    pred = pred.copy(); t = frame.relative_time_s.to_numpy(float)
    order = np.lexsort((t, frame.experiment_id.to_numpy()))
    e = frame.experiment_id.to_numpy()[order]; first = None; prev = None
    for i in order:
        if frame.experiment_id.iat[i] != prev: first, prev = None, frame.experiment_id.iat[i]
        if pred[i] == 4:
            if first is None: first = t[i]
            elif t[i] - first > th: pred[i] = 5
    return pred


@torch.no_grad()
def predict_tta(model, split, device, use_time, tta):
    model.eval(); out = []
    for s in range(0, len(split.frame), 256):
        ids = torch.arange(s, min(s + 256, len(split.frame)))
        img, mask, q, gas = split.batch(ids, device, False, False, False, None, use_time)
        p = model(img, mask, q, gas).softmax(-1)
        if tta: p = (p + model(img.flip(-1), mask.flip(-1), q, gas).softmax(-1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def macro(y, p):
    return float(f1_score(y, p, labels=range(1, 7), average='macro', zero_division=0))


ARCHS = {  # 이름: (생성자, 가중치, layer3 출력 채널)
    'resnet18': (tv.models.resnet18, tv.models.ResNet18_Weights.IMAGENET1K_V1, 256),
    'resnet34': (tv.models.resnet34, tv.models.ResNet34_Weights.IMAGENET1K_V1, 256),
    'resnet50': (tv.models.resnet50, tv.models.ResNet50_Weights.IMAGENET1K_V2, 1024),
    'convnext_tiny': (tv.models.convnext_tiny, tv.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 384),
}


class Backbone(nn.Module):
    def __init__(self, d, pretrained=True, arch='resnet18'):
        super().__init__()
        ctor, w, ch = ARCHS[arch]
        r = ctor(weights=w if pretrained else None)
        if arch.startswith('convnext'):
            self.body = r.features[:6]  # stride 16, 384ch (resnet layer3와 같은 해상도)
        else:
            self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)  # 128px → 8x8
        self.proj = nn.Sequential(nn.Conv2d(ch, d, 1, bias=False), nn.GroupNorm(8, d), nn.ReLU())
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return self.proj(self.body((x - self.mean) / self.std))


class Model(cg.Model):
    def __init__(self, use_time=True, pretrained=True, d=64, arch='resnet18'):
        super().__init__(False, False, use_time, d)
        self.cnn = Backbone(d, pretrained, arch)


def build_opt(model):
    bb = [p for n, p in model.named_parameters() if n.startswith('cnn.body')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('cnn.body')]
    return torch.optim.AdamW([{'params': bb, 'lr': 1e-4}, {'params': rest, 'lr': 3e-4}], weight_decay=1e-4)


def run(seed, train, prep, epochs, device, use_time, dev=None, arch='resnet18'):
    abc.common.seed_everything(seed)
    model = Model(use_time, arch=arch).to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(prep.weights, device=device))
    opt = build_opt(model)
    gen = torch.Generator().manual_seed(seed)
    best, state, best_ep = -1, None, epochs
    for ep in range(1, epochs + 1):
        model.train(); t0 = time.monotonic()
        order = torch.randperm(len(train.frame), generator=gen)
        for s in range(0, len(order), 64):
            ids = order[s:s + 64]
            y = train.y.index_select(0, ids.to(train.y.device)).to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(*train.batch(ids, device, False, False, False, None, use_time)), y)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
        msg = f'  seed{seed} ep{ep:02d} ({time.monotonic() - t0:.0f}s)'
        if dev is not None:
            f = macro(dev.frame.label, cg.predict(model, dev, False, False, device, use_time).argmax(1) + 1)
            if f > best:
                best, best_ep, state = f, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
            msg += f' dev {f:.4f} best {best:.4f}'
        print(msg, flush=True)
    if state is not None: model.load_state_dict(state)
    return model, best_ep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['check', 'final'], required=True)
    ap.add_argument('--data-root', type=Path, default=Path('/home/ubuntu/data/battery_data_light'))
    ap.add_argument('--base-run', type=Path, default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--no-time', action='store_true')
    ap.add_argument('--arch', choices=list(ARCHS), default='resnet18')
    ap.add_argument('--image-size', type=int, default=128)
    ap.add_argument('--tta', action='store_true', help='좌우 반전 평균')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    use_time = not a.no_time
    if a.smoke: a.seeds, a.epochs = a.seeds[:1], 1
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit(f'{a.output} 가 이미 있음. 새 --output 경로를 지정하세요.')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'config.json').write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2))
    device = torch.device(a.device); torch.set_num_threads(4); torch.backends.cudnn.benchmark = True
    T0 = time.monotonic()

    frames = abc.load_frames(argparse.Namespace(data_root=a.data_root, split_seed=20261114, smoke=a.smoke))
    if a.image_size == 128:
        imgs = cg.load_images(argparse.Namespace(base_run=a.base_run, smoke=a.smoke, data_root=a.data_root), frames)
    else:
        imgs = {s: torch.from_numpy(np.ascontiguousarray(abc.common.cache_images(
            a.data_root, f, 'train' if s in ['fit', 'dev'] else s, Path(__file__).resolve().parent / '.abc_cache',
            a.image_size, 4, 8)[0])) for s, f in frames.items()}
    parts = ['fit'] if a.mode == 'check' else ['fit', 'dev', 'validation']
    tf = pd.concat([frames[p] for p in parts], ignore_index=True)
    ti = torch.cat([imgs[p] for p in parts]) if len(parts) > 1 else imgs['fit']
    ev = ['dev', 'validation', 'test'] if a.mode == 'check' else ['test']
    store = {k: imgs[k] for k in ev}
    need = ti.numel() + sum(v.numel() for v in store.values())
    if device.type == 'cuda' and torch.cuda.mem_get_info()[0] > need + 5 * 1024 ** 3:
        ti = ti.to(device); store = {k: v.to(device) for k, v in store.items()}
    print(f'mode={a.mode} arch={a.arch} 학습 {len(tf)}행 {tf.experiment_id.nunique()}실험, time={use_time}', flush=True)
    prep = abc.SensorPreprocessor(tf)
    train = cg.Split(tf, ti, prep, device)
    data = {k: cg.Split(frames[k], store[k], prep, device) for k in ev}

    probs, eps = [], []
    for seed in a.seeds:
        m, ep = run(seed, train, prep, a.epochs, device, use_time, data.get('dev') if a.mode == 'check' else None, a.arch)
        probs.append({k: cg.predict(m, d, False, False, device, use_time) for k, d in data.items()}); eps.append(ep)
        if a.tta or a.mode == 'check':
            probs[-1].update({k + '_tta': predict_tta(m, d, device, use_time, True) for k, d in data.items()})
        torch.save({'state': {k: v.cpu() for k, v in m.state_dict().items()}, 'arch': a.arch, 'use_time': use_time,
                    'image_size': a.image_size, 'seed': seed, 'epoch': ep, 'mode': a.mode,
                    'prep': {'median': prep.median.tolist(), 'mean': prep.mean.tolist(), 'std': prep.std.tolist(),
                             'weights': prep.weights.tolist()}}, a.output / f'model_seed{seed}.pt')
        print(f'done seed{seed} best_epoch {ep} ({(time.monotonic() - T0) / 60:.1f} min)', flush=True)
        del m
        if device.type == 'cuda': torch.cuda.empty_cache()
    np.savez(a.output / 'probs.npz', **{f'{k}_{i}': v for i, p in enumerate(probs) for k, v in p.items()})

    if a.mode == 'check':
        va = frames['validation']; yv = va.label.to_numpy(int); yd = frames['dev'].label.to_numpy(int)
        vexp = sorted(va.experiment_id.unique()); e16 = vexp[15] if len(vexp) >= 16 else None
        keep = (va.experiment_id != e16).to_numpy(); t5 = ((va.experiment_id == e16) & (va.label == 5)).to_numpy()
        rows = []
        variants = [('', False, False), ('+TTA', True, False), ('+규칙', False, True), ('+TTA+규칙', True, True)]
        for name, ps in [('3seed 앙상블', probs)] + [(f'seed{s}', [p]) for s, p in zip(a.seeds, probs)]:
            for suf, tta, rule in (variants if name == '3seed 앙상블' else variants[:1]):
                k = '_tta' if tta else ''
                pdv = np.mean([p['dev' + k] for p in ps], 0).argmax(1) + 1
                pv = np.mean([p['validation' + k] for p in ps], 0).argmax(1) + 1
                if rule:
                    pdv = long4_rule(frames['dev'], pdv); pv = long4_rule(va, pv)
                rows.append(dict(model=name + suf, dev_f1=macro(yd, pdv), val_f1=macro(yv, pv),
                                 val_noE16_f1=macro(yv[keep], pv[keep]), E16_5_correct=int((pv[t5] == 5).sum())))
        df = pd.DataFrame(rows); df.to_csv(a.output / 'check.csv', index=False)
        print('\n비교 기준  0.91 모델: dev 0.7815 / noE16 0.8142 | resnet18 3seed: dev 0.8349 / noE16 0.8325')
        print(df.round(4).to_string(index=False))
        print(f'best epochs {eps} → final에는 --epochs {int(np.median(eps))} 권장')

    if not a.smoke:
        sample = pd.read_csv(a.data_root / 'sample_submission.csv')
        key = 'test_tta' if a.tta else 'test'
        tp = np.mean([p[key] for p in probs], 0).argmax(1) + 1
        tp = long4_rule(frames['test'], tp)  # 긴 4단계 규칙 적용
        pred = pd.Series(tp, index=frames['test'].sample_id)
        sub = sample[['sample_id']].copy(); sub['label'] = sub.sample_id.map(pred).astype(int)
        assert len(sub) == len(sample) and sub.label.between(1, 6).all()
        fn = a.output / f'submission_pretrained_{a.mode}_{a.arch}_{a.image_size}{"_tta" if a.tta else ""}_long4rule.csv'; sub.to_csv(fn, index=False)
        B0 = pd.read_csv(a.base_run / 'submission_B.csv').set_index('sample_id').label
        print(f'{fn}: 0.91 파일 대비 변경 {(sub.label.values != B0.loc[sub.sample_id].values).sum()}행')
    print(f'총 {(time.monotonic() - T0) / 60:.1f} 분', flush=True)


if __name__ == '__main__':
    main()
