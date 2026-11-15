"""Nested CV용 B 재학습. train+validation 97실험을 실험 단위 바깥 5-fold(배터리 조합 층화)로 나누고,
각 바깥 fold마다 b_holdout.py와 같은 절차를 반복한다.
  - 나머지 실험(개발): 안쪽 실험 단위 5-fold × seed 42/43/44 → 개발 행 B OOF
  - 개발 전체로 학습한 B → 바깥 fold(평가) 행 확률
평가 fold 행은 그 fold의 어떤 B 학습·전처리·checkpoint 선택에도 쓰이지 않는다. Test는 읽지 않는다.
  python -u experiments/nested_cv/b_nested.py --output experiments/nested_cv/b_v1 --device cuda
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from abc_reference import abc_fusion_experiment as abc
from targeted_rule1.b_oof import View, make_data, train_b, DATA, CACHE, SEEDS
from experiments.holdout_12_8.b_holdout import pooled

OUTER_SEED = 20261115


def outer_folds(frame):
    exps = frame.experiment_id.to_numpy()
    combo = (frame.bat_class.astype(int).astype(str)+'/'+frame.bat_cc.astype(str)).to_numpy()
    sgk = StratifiedGroupKFold(5, shuffle=True, random_state=OUTER_SEED)
    out = np.full(len(frame), -1)
    for k, (_, b) in enumerate(sgk.split(frame, combo, groups=exps)):
        out[b] = k
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--epochs', type=int, default=15)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device); torch.set_num_threads(4); t0 = time.monotonic()
    tr, va, frame = pooled()
    imgs = torch.cat([torch.from_numpy(np.ascontiguousarray(abc.common.cache_images(DATA, f, s, CACHE, 128, 4, 8)[0]))
                      for s, f in [('train', tr), ('validation', va)]])
    if device.type == 'cuda' and torch.cuda.mem_get_info()[0] > imgs.numel()+4*1024**3:
        imgs = imgs.to(device)
    exps = frame.experiment_id.to_numpy(); outer = outer_folds(frame)
    np.save(a.output/'outer_fold.npy', outer)
    log = []
    for o in range(5):
        path = a.output/f'outer{o}.npz'
        if path.exists():
            continue
        dev = np.flatnonzero(outer != o); hold = np.flatnonzero(outer == o)
        assert not set(exps[dev]) & set(exps[hold])
        fold_of = np.full(len(dev), -1)
        oof = np.zeros((len(SEEDS), len(frame), 6)); hp = np.zeros((len(SEEDS), len(frame), 6))
        plans = [(k, dev[x], dev[y]) for k, (x, y) in enumerate(GroupKFold(5).split(dev, groups=exps[dev]))]
        for k, _, tg in plans:
            fold_of[np.isin(dev, tg)] = k
        plans.append(('final', dev, hold))
        for k, fit_all, target in plans:
            assert not set(exps[fit_all]) & set(exps[target]) and not set(exps[fit_all]) & set(exps[hold])
            i, j = next(GroupShuffleSplit(1, test_size=.15, random_state=20261114+(9 if k == 'final' else k)+100*o)
                        .split(fit_all, groups=exps[fit_all]))
            fit_i, sel_i = fit_all[i], fit_all[j]
            prep = abc.SensorPreprocessor(frame.iloc[fit_i]); data = make_data(frame, imgs, prep)
            for si, seed in enumerate(SEEDS):
                m, best = train_b(View(data, fit_i), View(data, sel_i), prep.weights, seed, a.epochs, device)
                (hp if k == 'final' else oof)[si, target] = abc.predict(m, View(data, target), None, device)
                log.append(dict(outer=o, fold=k, seed=seed, sel_f1=best, minutes=(time.monotonic()-t0)/60))
                print(json.dumps(log[-1]), flush=True)
                del m
                if device.type == 'cuda': torch.cuda.empty_cache()
        np.savez(path, sample_id=frame.sample_id.to_numpy(str), dev_index=dev, hold_index=hold,
                 dev_fold=fold_of, dev_oof=oof[:, dev], hold=hp[:, hold])
    (a.output/'log.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
