"""최종 모델 학습 전체 절차. 결과 가중치는 weights/에 저장한다(운영진은 실행하지 않고 읽기만 함).

  python train.py --data_root <battery_data_light> --work runs --device cuda

단계 (Test는 읽지 않는다)
 1. B 융합 모델(FusionModel 'B') seed 42/43/44 학습 → weights/B_seed*.pt, weights/preprocessing.json
    train 실험의 80%로 학습, 20%(실험 단위)로 epoch 선택. weighted CE, AdamW lr 3e-4, batch 64, 15 epoch.
    최종 제출에 필요한 B만 학습하며 Test와 공식 Validation은 읽지 않는다.
 2. B OOF 확률: train 78실험을 실험 단위 5-fold로 나눠 B를 다시 학습하고, 각 fold 밖 실험의 확률을 만든다.
    보정 모델은 B가 그 실험을 보지 않은 상태의 확률로 학습해야 test 때와 조건이 같다.
 3. 5→6 보정(HGB): 기준 예측(B+t1 45°C)이 5인 행 대상. 기준값은 학습 실험 3-fold로 {0.5,0.7,0.9} 중 자동 선택.
    (같은 함수가 4→5 헤드도 학습하지만, 처음 보는 배터리 평가에서 손해라 추론에서 끈다: model.upper_heads)
 4. 6→5 보정(HGB): 기준 예측이 6인 행 대상. 기준값 {0.7,0.9,0.95,0.99} 중 자동 선택.
    선택에 쓰는 기준 예측은 fold 헤드로 만든 OOF 예측이다.
 5. 하위 경계 보정 4종(2→3, 3→2, 3→4, 4→3)을 같은 방식으로 학습. 개발 검증의 사전 채택 규칙으로 3→4·4→3만 사용
    (model.LOWER_USED). 3→4는 학습 내부 기준값 선택에서 꺼진다.
 6. S3는 model.py의 고정 규칙(H=300, d=5)이며 추가 학습이 없다.
 모든 피처는 현재·과거 관측만 쓴다(model.py). 정규화·결측 대체·모델·기준값은 학습 실험에서만 적합한다.
"""
import argparse
import copy
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
import torch
from torch import nn

import model as M
from model import T, P

HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------- 1·2. B
def train_b(data_root, work, device):
    out = work/'abc'
    out.mkdir(parents=True, exist_ok=False)
    abc = M.abc
    device = torch.device(device)
    torch.set_num_threads(4)
    frame = pd.read_csv(Path(data_root)/'train/train.csv')
    fit_ids, dev_ids = next(GroupShuffleSplit(1, test_size=.2, random_state=20261114).split(
        frame, groups=frame.experiment_id))
    assert not set(frame.experiment_id.iloc[fit_ids]) & set(frame.experiment_id.iloc[dev_ids])
    frames = {'fit': frame.iloc[fit_ids].reset_index(drop=True),
              'dev': frame.iloc[dev_ids].reset_index(drop=True)}
    prep = abc.SensorPreprocessor(frames['fit'])
    abc.common.write_json(out/'preprocessing.json', prep.metadata())
    args = argparse.Namespace(data_root=Path(data_root), output=out)
    datasets = abc.prepare(args, frames, prep, device)
    (HERE/'weights').mkdir(exist_ok=True)
    abc.common.write_json(HERE/'weights/preprocessing.json', prep.metadata())
    abc.common.write_json(out/'split_manifest.json', {
        name: sorted(f.experiment_id.unique().tolist()) for name, f in frames.items()})
    for seed in M.SEEDS:
        # 원래 A/B/C 비교와 같은 초기값·난수 상태로 B만 학습한다.
        abc.common.seed_everything(seed)
        initial = copy.deepcopy(abc.FusionModel('A').state_dict())
        abc.common.seed_everything(seed)
        model = abc.FusionModel('B').to(device)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(prep.weights, device=device))
        generator = torch.Generator().manual_seed(seed)
        fit = datasets['fit']
        best = -1.
        history = []
        for epoch in range(1, 16):
            model.train()
            order = torch.randperm(len(fit.frame), generator=generator)
            for start in range(0, len(order), 64):
                ids = order[start:start+64]
                y = fit.y.index_select(0, ids.to(fit.y.device)).to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(*fit.batch(ids, device)), y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            score = abc.metrics(frames['dev'], abc.predict(model, datasets['dev'], None, device))['macro_f1']
            if score > best:
                best = score
                torch.save({'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            'epoch': epoch, 'dev_macro_f1': score, 'mode': 'B', 'seed': seed},
                           HERE/f'weights/B_seed{seed}.pt')
            history.append({'epoch': epoch, 'dev_macro_f1': score})
            print(f'B seed={seed} epoch={epoch}/15 dev={score:.6f} best={best:.6f}', flush=True)
        abc.common.write_json(out/f'seed_{seed}_history.json', history)
        del model, optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def b_oof(data_root, work, device):
    sys.path.insert(0, str(HERE/'lib'))
    from targeted_rule1 import b_oof as BO
    out = work/'b_oof'
    sys.argv = ['b_oof', '--data-root', str(data_root), '--cache', str(work/'oof_image_cache'),
                '--output', str(out), '--device', device]
    BO.main()
    return out/'b_oof.npz'


# ----------------------------------------------------------------------------- 하위 경계 보정 학습
def lower_fit(x, y, base, groups, pair):
    s, tg = M.LOWER_PAIRS[pair]
    keep = np.isin(y, [s, tg]) | (base == s)
    yy = (y[keep] == tg).astype(int)
    if len(np.unique(yy)) < 2:
        return None
    keys = pd.DataFrame({'e': groups[keep], 'y': yy})
    w = 1/keys.groupby(['e', 'y']).y.transform('size').to_numpy(float)       # 실험×클래스 균형
    m = HistGradientBoostingClassifier(**T.PARAMS); m.fit(x[keep], yy, sample_weight=w/w.mean())
    return m


def lower_select(score, y, base, groups, pair):
    """targeted_rule1.select_threshold와 같은 통과 조건."""
    ids = list(pd.DataFrame({'g': groups}).groupby('g').indices.values())
    cb = np.stack([T.matrix(y[i], base[i]) for i in ids]); choices = []
    for th in T.THRESHOLDS:
        p = M.lower_apply(base, score, pair, th)
        cp = np.stack([T.matrix(y[i], p[i]) for i in ids])
        delta = float(T.f1(cp.sum(0)).mean()-T.f1(cb.sum(0)).mean()); per = T.f1(cp).mean(1)-T.f1(cb).mean(1)
        fixed = int(((base != y) & (p == y)).sum()); harmed = int(((base == y) & (p != y)).sum())
        loo = [float(T.f1(cp.sum(0)-a).mean()-T.f1(cb.sum(0)-b).mean()) for a, b in zip(cp, cb)]
        ok = (delta > 0 and fixed >= max(3, 2*harmed) and (per > 1e-12).sum() >= 3
              and (per > 1e-12).sum() > (per < -1e-12).sum() and min(loo) > 0)
        choices.append(dict(threshold=th, delta=delta, fixed=fixed, harmed=harmed, passed=bool(ok)))
    valid = [c for c in choices if c['passed']]
    return (max(valid, key=lambda c: (c['delta'], c['threshold']))['threshold'] if valid else 1.01), choices


def lower_train(frame, probs, y, base):
    x, names = M.lower_features(frame, probs); g = frame.experiment_id.to_numpy()
    art = dict(models={}, thresholds={}, choices={}, names=names)
    folds = list(GroupKFold(3).split(x, groups=g))
    for pair in M.LOWER_PAIRS:
        sc = np.zeros(len(y))
        for a, b in folds:
            sc[b] = T.probability(lower_fit(x[a], y[a], base[a], g[a], pair), x[b])
        art['thresholds'][pair], art['choices'][pair] = lower_select(sc, y, base, g, pair)
        art['models'][pair] = lower_fit(x, y, base, g, pair)
    return art


# ----------------------------------------------------------------------------- 3·4·5. 보정 학습
def train_heads(data_root, oof_path):
    frame = pd.read_csv(Path(data_root)/'train/train.csv'); y = frame.label.to_numpy(int)
    with np.load(oof_path) as z:
        assert np.array_equal(z['sample_id'], frame.sample_id.to_numpy(str))
        p, folds = z['oof'], z['fold']
    groups = frame.experiment_id.to_numpy()
    base = M.rule1(frame, p.mean(0).argmax(1)+1)                       # 기준 예측: B OOF + t1 45°C

    # 5→6(·4→5) 헤드: fold 헤드로 OOF 기준 예측을 만든다(6→5 기준값 선택용)
    baseline = np.zeros(len(frame), int)
    for k in range(5):
        fit, held = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        heads_k = T.train(frame.iloc[fit].reset_index(drop=True), p[:, fit], y[fit], base[fit])
        hf = frame.iloc[held].reset_index(drop=True); x, _ = T.features(hf, p[:, held])
        b = base[held].copy(); out = b.copy()
        for pair in ('45', '56'):
            s, tg = T.PAIRS[pair]
            out[(b == s) & (T.probability(heads_k['models'][pair], x[pair]) >= heads_k['thresholds'][pair])] = tg
        baseline[held] = out
    heads = T.train(frame, p, y, base)                                 # 최종 5→6 헤드(4→5는 추론에서 끔)

    # 6→5 헤드
    x56 = T.features(frame, p)[0]['56']
    gate = P.train(x56, y, baseline, base, groups)

    # 하위 경계 헤드(4→3 등)
    lower = lower_train(frame, p, y, base)

    joblib.dump(dict(baseline_heads=heads, precision_head=gate), HERE/'weights/pipeline.joblib')
    joblib.dump(lower, HERE/'weights/lower_heads.joblib')
    print('5→6', heads['thresholds'], '6→5', gate['threshold'], 'lower', lower['thresholds'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', type=Path, required=True)
    ap.add_argument('--work', type=Path, default=HERE/'runs')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('PYTHONHASHSEED', '0')
    train_b(a.data_root, a.work, a.device)
    train_heads(a.data_root, b_oof(a.data_root, a.work, a.device))
