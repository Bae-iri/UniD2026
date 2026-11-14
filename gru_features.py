"""사전학습 이미지 모델의 프레임별 융합 특징 + 센서 + 경과시간 → 실험별 시간순 단방향 GRU.

pretrained_b.py가 저장한 model_seed*.pt(seed별)를 읽어, head 직전의 융합 특징(센서 query, 가스·이미지 요약,
가스·이미지 attention 출력)과 기본 모델 확률을 뽑는다. GRU는 같은 실험의 현재·과거 행만 보며(단방향),
출력 = 기본 모델 logit + 학습된 보정(초기 0)이라 시작 시 기본 모델과 같다. 보정이 해로우면 epoch0(기본 모델)으로 복귀.

  check: --models pre_runs/check_X  (fit 학습 모델) → fit으로 GRU 학습, dev/validation 평가, test 파일
  final: --models pre_runs/final_X --final  (97실험 모델) → fit+dev+validation으로 GRU 학습, test 파일

  python -u gru_features.py --models pre_runs/check_r18_224 --output gru_runs/check --device cuda
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from abc_reference import abc_fusion_experiment as abc
import abc_color_gas as cg
import pretrained_b as pb
import two_stage_temporal as ts

ROOT = Path(__file__).resolve().parent


def macro(y, p):
    return float(f1_score(y, p, labels=range(1, 7), average='macro', zero_division=0))


def load_imgs(frames, size, data_root, base_run):
    if size == 128:
        return cg.load_images(argparse.Namespace(base_run=base_run, smoke=False, data_root=data_root), frames)
    return {s: torch.from_numpy(np.ascontiguousarray(abc.common.cache_images(
        data_root, f, 'train' if s in ['fit', 'dev'] else s, ROOT / '.abc_cache', size, 4, 8)[0])) for s, f in frames.items()}


@torch.no_grad()
def extract(model, split, device, use_time):
    cap = {}
    h = model.head.register_forward_pre_hook(lambda mod, inp: cap.__setitem__('f', inp[0]))
    model.eval(); feats, probs = [], []
    try:
        for s in range(0, len(split.frame), 256):
            ids = torch.arange(s, min(s + 256, len(split.frame)))
            p = model(*split.batch(ids, device, False, False, False, None, use_time)).softmax(-1)
            feats.append(cap['f'].float().cpu().numpy()); probs.append(p.cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(feats), np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', type=Path, required=True)
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, default=Path('/home/ubuntu/data/battery_data_light'))
    ap.add_argument('--base-run', type=Path, default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--epochs', type=int, default=30)
    a = ap.parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise SystemExit(f'{a.output} 가 이미 있음.')
    a.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device); T0 = time.monotonic()
    cks = sorted(a.models.glob('model_seed*.pt'))
    if not cks: raise SystemExit('model_seed*.pt 없음 (가중치 저장 기능 추가 후 실행한 run이어야 함)')
    meta = torch.load(cks[0], map_location='cpu', weights_only=False)
    frames = abc.load_frames(argparse.Namespace(data_root=a.data_root, split_seed=20261114, smoke=False))
    # 시간순 정렬(실험별 연속 블록) 후 사용
    order = {s: f.sort_values(['experiment_id', 'relative_time_s'], kind='mergesort').index.to_numpy() for s, f in frames.items()}
    imgs = load_imgs(frames, meta['image_size'], a.data_root, a.base_run)
    prep = abc.SensorPreprocessor.__new__(abc.SensorPreprocessor)
    for k in ['median', 'mean', 'std']: setattr(prep, k, pd.Series(meta['prep'][k], index=abc.NUMERIC))
    prep.weights = np.asarray(meta['prep']['weights'], np.float32)
    data = {s: cg.Split(f, imgs[s], prep, device) for s, f in frames.items()}
    train_parts = ['fit', 'dev', 'validation'] if a.final else ['fit']
    eval_parts = ['test'] if a.final else ['dev', 'validation', 'test']

    sf = {s: frames[s].loc[order[s]].reset_index(drop=True) for s in frames}
    tr = pd.concat([sf[s] for s in train_parts], ignore_index=True)
    ytr = tr.label.to_numpy(int)
    base_p, gru_p = {s: 0 for s in eval_parts}, {s: 0 for s in eval_parts}
    for ck in cks:
        c = torch.load(ck, map_location='cpu', weights_only=False)
        m = pb.Model(c['use_time'], pretrained=False, arch=c['arch']).to(device); m.load_state_dict(c['state'])
        F, P = {}, {}
        for s in frames:
            f, p = extract(m, data[s], device, c['use_time']); F[s], P[s] = f[order[s]], p[order[s]]
        del m
        if device.type == 'cuda': torch.cuda.empty_cache()
        Ftr = np.concatenate([F[s] for s in train_parts]); mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6

        def X(s_list):
            fr = pd.concat([sf[s] for s in s_list], ignore_index=True)
            f = (np.concatenate([F[s] for s in s_list]) - mu) / sd
            p = np.concatenate([P[s] for s in s_list])
            return np.concatenate([np.log(np.clip(p, 1e-4, 1)), np.clip(f, -8, 8), cg.gas_status(fr)], 1).astype(np.float32), fr
        Xtr, _ = X(train_parts); ttr = tr.relative_time_s.to_numpy(float)
        bl = ts.blocks(tr); exps = tr.experiment_id.unique()
        rng = np.random.default_rng(c['seed']); rng.shuffle(exps); sel = set(exps[:max(1, len(exps) * 15 // 100)])
        bl_fit = [b for b in bl if tr.experiment_id.iat[b[0]] not in sel]
        bl_sel = [b for b in bl if tr.experiment_id.iat[b[0]] in sel]
        g, best = ts.train_stage2(bl_fit, bl_sel, Xtr, ttr, ytr, True, c['seed'], a.epochs, device)
        print(f'seed{c["seed"]}: GRU sel F1 {best:.4f} ({(time.monotonic() - T0) / 60:.1f} min)', flush=True)
        for s in eval_parts:
            Xs, fr = X([s])
            gru_p[s] = gru_p[s] + ts.predict2(g, ts.blocks(fr), Xs, fr.relative_time_s.to_numpy(float), len(fr), device) / len(cks)
            base_p[s] = base_p[s] + P[s] / len(cks)

    if not a.final:
        va, dv = sf['validation'], sf['dev']; yv, yd = va.label.to_numpy(int), dv.label.to_numpy(int)
        e16 = sorted(va.experiment_id.unique())[15]; keep = (va.experiment_id != e16).to_numpy()
        t5 = ((va.experiment_id == e16) & (va.label == 5)).to_numpy(); rows = []
        for name, P_ in [('기본 모델', base_p), ('기본+GRU', gru_p)]:
            for rule in [False, True]:
                pdv = P_['dev'].argmax(1) + 1; pv = P_['validation'].argmax(1) + 1
                if rule: pdv, pv = pb.long4_rule(dv, pdv), pb.long4_rule(va, pv)
                d, n = macro(yd, pdv), macro(yv[keep], pv[keep])
                rows.append(dict(model=name + ('+규칙' if rule else ''), dev_f1=d, val_f1=macro(yv, pv), val_noE16_f1=n,
                                 평균_dev_noE16=(d + n) / 2, E16_5_correct=int((pv[t5] == 5).sum())))
        df = pd.DataFrame(rows); df.to_csv(a.output / 'check.csv', index=False)
        print(df.round(4).to_string(index=False))
    te = sf['test']; sample = pd.read_csv(a.data_root / 'sample_submission.csv')
    B0 = pd.read_csv(a.base_run / 'submission_B.csv').set_index('sample_id').label
    for name, P_ in [('base', base_p), ('gru', gru_p)]:
        pred = pb.long4_rule(te, P_['test'].argmax(1) + 1)
        sub = sample[['sample_id']].copy(); sub['label'] = sub.sample_id.map(pd.Series(pred, index=te.sample_id)).astype(int)
        assert len(sub) == 3451 and sub.label.between(1, 6).all()
        fn = a.output / f'submission_{name}_{"final" if a.final else "check"}_long4rule.csv'; sub.to_csv(fn, index=False)
        print(f'{fn}: 0.91 파일 대비 변경 {(sub.label.values != B0.loc[sub.sample_id].values).sum()}행')
    np.savez(a.output / 'probs.npz', **{f'{n}_{s}': P_[s] for n, P_ in [('base', base_p), ('gru', gru_p)] for s in eval_parts})
    print(f'총 {(time.monotonic() - T0) / 60:.1f} 분')


if __name__ == '__main__':
    main()
