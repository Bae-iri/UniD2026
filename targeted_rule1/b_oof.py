"""원본 ABC B를 train 78실험 5-fold(실험 단위)로 재학습해 OOF 확률을 만든다.

각 fold: 나머지 실험 중 15%(실험 단위)를 checkpoint 선택용으로 떼고, 원본 ABC와 같은 설정
(FusionModel 'B', weighted CE, AdamW 3e-4/1e-4, batch 64, 15 epoch, 선택용 최고 macro F1 epoch)으로 학습.
출력: train 각 행의 OOF 확률(seed별), validation 확률(seed별·fold 평균). Test는 읽지 않는다.
전처리(SensorPreprocessor)는 각 fold의 학습 실험으로만 적합한다.

  python -u targeted_rule1/b_oof.py --output targeted_rule1/b_oof_v1 --device cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abc_reference import abc_fusion_experiment as abc

DATA = Path('/home/ubuntu/data/battery_data_light')
CACHE = ROOT/'.abc_cache'
SEEDS = [42, 43, 44]


class View:
    def __init__(self, base, idx):
        self.base, self.idx = base, torch.as_tensor(np.asarray(idx), dtype=torch.long)
        self.frame = base.frame.iloc[np.asarray(idx)].reset_index(drop=True)
        self.y = None if base.y is None else base.y.index_select(0, self.idx.to(base.y.device))

    def batch(self, ids, device):
        return self.base.batch(self.idx[ids.cpu()], device)


def make_data(frame, images, prep):
    q, gas = prep.transform(frame)
    dev = images.device
    y = torch.tensor(frame.label.to_numpy(int)-1, device=dev) if 'label' in frame else None
    return abc.Data(frame, images, torch.from_numpy(q).to(dev), torch.from_numpy(gas).to(dev), y)


def train_b(fit, sel, weights, seed, epochs, device):
    abc.common.seed_everything(seed)
    model = abc.FusionModel('B').to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    best, state = -1, None
    for ep in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(fit.frame), generator=gen)
        for s in range(0, len(order), 64):
            ids = order[s:s+64]
            y = fit.y.index_select(0, ids.to(fit.y.device)).to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(*fit.batch(ids, device)), y)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
        f = f1_score(sel.frame.label, abc.predict(model, sel, None, device).argmax(1)+1,
                     labels=range(1, 7), average='macro', zero_division=0)
        if f > best:
            best, state = f, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=15)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device); torch.set_num_threads(4); t0 = time.monotonic()
    tr = pd.read_csv(DATA/'train/train.csv'); va = pd.read_csv(DATA/'validation/validation.csv')
    imgs = {s: torch.from_numpy(np.ascontiguousarray(abc.common.cache_images(DATA, f, s, CACHE, 128, 4, 8)[0]))
            for s, f in [('train', tr), ('validation', va)]}
    if device.type == 'cuda' and torch.cuda.mem_get_info()[0] > sum(v.numel() for v in imgs.values()) + 4*1024**3:
        imgs = {k: v.to(device) for k, v in imgs.items()}
    folds = list(GroupKFold(5).split(tr, groups=tr.experiment_id))
    fold_of = np.zeros(len(tr), int)
    oof = np.zeros((len(SEEDS), len(tr), 6)); val = np.zeros((len(SEEDS), len(va), 6))
    log = []
    for k, (a_idx, b_idx) in enumerate(folds):
        fold_of[b_idx] = k
        i, j = next(GroupShuffleSplit(1, test_size=.15, random_state=20261114+k).split(a_idx, groups=tr.experiment_id.values[a_idx]))
        fit_i, sel_i = a_idx[i], a_idx[j]
        assert not set(tr.experiment_id.values[b_idx]) & set(tr.experiment_id.values[a_idx])
        prep = abc.SensorPreprocessor(tr.iloc[fit_i])
        dtr, dva = make_data(tr, imgs['train'], prep), make_data(va, imgs['validation'], prep)
        for si, seed in enumerate(SEEDS):
            m, best = train_b(View(dtr, fit_i), View(dtr, sel_i), prep.weights, seed, a.epochs, device)
            oof[si, b_idx] = abc.predict(m, View(dtr, b_idx), None, device)
            val[si] += abc.predict(m, dva, None, device)/len(folds)
            log.append(dict(fold=k, seed=seed, sel_f1=best, minutes=(time.monotonic()-t0)/60))
            print(json.dumps(log[-1]), flush=True)
            del m
            if device.type == 'cuda': torch.cuda.empty_cache()
    np.savez(a.output/'b_oof.npz', sample_id=tr.sample_id.to_numpy(str), fold=fold_of, oof=oof,
             val_sample_id=va.sample_id.to_numpy(str), validation=val)
    y = tr.label.to_numpy(int)
    print('OOF B macro F1', f1_score(y, oof.mean(0).argmax(1)+1, average='macro'), flush=True)
    (a.output/'log.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
