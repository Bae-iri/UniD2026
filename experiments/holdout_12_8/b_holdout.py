"""고정 holdout(leewon qt_mixed_runs/precision65_v1, 원래 Train 12 + Validation 8실험)에 맞춘 원본 ABC B 재학습.

- 개발 77실험(fit+tune): 실험 단위 5-fold × seed 42/43/44 → 개발 행의 B OOF 확률.
  각 fold의 학습 실험 중 15%(실험 단위)는 checkpoint 선택용. 전처리는 그 fold의 학습 실험으로만 적합.
- 최종 B: 개발 77실험 전체(15%는 선택용) → holdout 20실험 확률.
- holdout 행은 어떤 B의 학습·전처리·checkpoint 선택에도 쓰이지 않는다. Test는 읽지 않는다.
설정은 targeted_rule1/b_oof.py(원본 ABC와 같은 B 레시피)를 그대로 쓴다.

  python -u experiments/holdout_12_8/b_holdout.py --output experiments/holdout_12_8/b_v1 --device cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from abc_reference import abc_fusion_experiment as abc
from targeted_rule1.b_oof import View, make_data, train_b, DATA, CACHE, SEEDS

MANIFEST = Path('/home/ubuntu/leewon/qt_mixed_runs/precision65_v1/split_manifest.csv')


def pooled():
    tr = pd.read_csv(DATA/'train/train.csv').assign(original_split='train')
    va = pd.read_csv(DATA/'validation/validation.csv').assign(original_split='validation')
    frame = pd.concat([tr, va], ignore_index=True)
    man = pd.read_csv(MANIFEST).set_index('sample_id')
    assert set(man.index) == set(frame.sample_id) and len(man) == len(frame)
    frame['role'] = man.loc[frame.sample_id, 'split'].map({'fit': 'dev', 'tune': 'dev', 'holdout': 'holdout'}).to_numpy()
    assert (man.loc[frame.sample_id, 'original_split'].to_numpy() == frame.original_split.to_numpy()).all()
    assert frame.groupby('experiment_id').role.nunique().max() == 1
    return tr, va, frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=15)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device); torch.set_num_threads(4); t0 = time.monotonic()
    tr, va, frame = pooled()
    imgs = torch.cat([torch.from_numpy(np.ascontiguousarray(abc.common.cache_images(DATA, f, s, CACHE, 128, 4, 8)[0]))
                      for s, f in [('train', tr), ('validation', va)]])
    assert len(imgs) == len(frame)
    if device.type == 'cuda' and torch.cuda.mem_get_info()[0] > imgs.numel() + 4*1024**3:
        imgs = imgs.to(device)
    dev = np.flatnonzero(frame.role == 'dev'); hold = np.flatnonzero(frame.role == 'holdout')
    exps = frame.experiment_id.to_numpy()
    assert not set(exps[dev]) & set(exps[hold])
    print('dev exps', len(set(exps[dev])), 'holdout exps', len(set(exps[hold])), flush=True)
    fold_of = np.full(len(frame), -1)
    oof = np.zeros((len(SEEDS), len(frame), 6)); hp = np.zeros((len(SEEDS), len(frame), 6))
    log = []
    plans = [(k, dev[a_], dev[b_]) for k, (a_, b_) in enumerate(GroupKFold(5).split(dev, groups=exps[dev]))]
    plans.append(('final', dev, hold))
    for k, fit_all, target in plans:
        assert not set(exps[fit_all]) & set(exps[target])
        assert not set(exps[fit_all]) & set(exps[hold])
        i, j = next(GroupShuffleSplit(1, test_size=.15, random_state=20261114+(9 if k == 'final' else k))
                    .split(fit_all, groups=exps[fit_all]))
        fit_i, sel_i = fit_all[i], fit_all[j]
        prep = abc.SensorPreprocessor(frame.iloc[fit_i])
        data = make_data(frame, imgs, prep)
        if k != 'final':
            fold_of[target] = k
        for si, seed in enumerate(SEEDS):
            m, best = train_b(View(data, fit_i), View(data, sel_i), prep.weights, seed, a.epochs, device)
            p = abc.predict(m, View(data, target), None, device)
            (hp if k == 'final' else oof)[si, target] = p
            log.append(dict(fold=k, seed=seed, sel_f1=best, minutes=(time.monotonic()-t0)/60))
            print(json.dumps(log[-1]), flush=True)
            del m
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    np.savez(a.output/'b_probs.npz', sample_id=frame.sample_id.to_numpy(str), role=frame.role.to_numpy(str),
             fold=fold_of, dev_oof=oof[:, dev], dev_index=dev, holdout=hp[:, hold], holdout_index=hold)
    (a.output/'log.json').write_text(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()
