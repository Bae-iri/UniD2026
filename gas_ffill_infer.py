"""가스 끊김 입력 교정 추론: 재학습 없이 public 0.91 ABC B checkpoint(seed 42/43/44)를 그대로 사용.

가스 농도 10종이 동시에 정확히 0인 행(측정 끊김)에만, 같은 실험에서 직전에 정상 측정된 가스값을 넣는다
(과거 관측만 사용, 압력은 그대로). 나머지 행은 입력이 같으므로 예측도 같다.
정상 입력의 앙상블 예측이 저장된 0.91 예측과 정확히 일치해야 진행한다.

  python -u gas_ffill_infer.py --output gas_ffill_runs/v1 --device cuda
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from abc_reference import abc_fusion_experiment as abc
from diagnose_abc_gas_dependency import restore_preprocessor

CONC = [c for c in abc.GAS if c != 'meas_p']


def gas_ffill(frame):
    """반환: (교정된 frame, 가스 끊김 mask). 행 순서는 그대로 유지."""
    f = frame.copy()
    g = f[CONC].apply(pd.to_numeric, errors='coerce')
    off = (g.fillna(0) == 0).all(axis=1)
    order = f.sort_values(['experiment_id', 'relative_time_s'], kind='mergesort').index
    gs = g.loc[order].mask(off.loc[order])                 # 끊김 행을 비우고
    gs = gs.groupby(f.experiment_id.loc[order]).ffill()    # 같은 실험의 직전 정상값으로(과거만)
    gs = gs.fillna(g.loc[order])                           # 직전 정상값이 없으면 원래값(0) 유지
    f.loc[order, CONC] = gs.values
    return f, off.to_numpy()


def load_split(run, root, split):
    ref = pd.read_csv(run / f'ensemble_B_{split}_predictions.csv')
    src = 'train' if split == 'dev' else split
    raw = pd.read_csv(root / src / f'{src}.csv').set_index('sample_id', drop=False)
    frame = raw.loc[ref.sample_id].reset_index(drop=True)
    cache = Path(json.loads((run / 'image_cache_manifest.json').read_text())[split])
    images = np.load(cache, mmap_mode='c')
    if images.shape != (len(frame), 4, 128, 128):
        raise ValueError(f'{split}: cache shape differs')
    return frame, ref, torch.from_numpy(images)


@torch.inference_mode()
def predict(models, frame, images, prep, device):
    q, gas = prep.transform(frame)
    y = torch.tensor(frame.label.to_numpy(int) - 1) if 'label' in frame else None
    data = abc.Data(frame, images, torch.from_numpy(q), torch.from_numpy(gas), y)
    return np.mean([abc.predict(m, data, None, device) for m in models], axis=0)


def macro(y, p):
    return float(f1_score(y, p, labels=range(1, 7), average='macro', zero_division=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-run', type=Path, default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    ap.add_argument('--data-dir', type=Path, default=Path('/home/ubuntu/data/battery_data_light'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit(f'{a.output} 가 이미 있음. 새 경로를 지정하세요.')
    device = torch.device(a.device)
    prep = restore_preprocessor(a.base_run)
    models = []
    for seed in [42, 43, 44]:
        ck = torch.load(a.base_run / f'seed_{seed}/B/best.pt', map_location='cpu', weights_only=True)
        m = abc.FusionModel('B').to(device); m.load_state_dict(ck['state']); m.eval(); models.append(m)

    rows, out = [], {}
    for split in ['dev', 'validation', 'test']:
        frame, ref, images = load_split(a.base_run, a.data_dir, split)
        p0 = predict(models, frame, images, prep, device)
        if not np.array_equal(p0.argmax(1) + 1, ref.pred.to_numpy()):
            raise SystemExit(f'{split}: 정상 입력 예측이 저장된 0.91 예측과 다름. 중단.')
        fixed, off = gas_ffill(frame)
        p1 = predict(models, fixed, images, prep, device)
        a0, a1 = p0.argmax(1) + 1, p1.argmax(1) + 1
        out[split] = (frame, a1)
        r = dict(split=split, rows=len(frame), gas_off_rows=int(off.sum()), changed=int((a0 != a1).sum()))
        if 'label' in frame:
            y = frame.label.to_numpy(int)
            r.update(f1_before=macro(y, a0), f1_after=macro(y, a1),
                     gasoff_acc_before=float((a0[off] == y[off]).mean()) if off.any() else np.nan,
                     gasoff_acc_after=float((a1[off] == y[off]).mean()) if off.any() else np.nan,
                     fixed_right=int(((a0 != y) & (a1 == y)).sum()), made_wrong=int(((a0 == y) & (a1 != y)).sum()))
            exps = sorted(frame.experiment_id.unique()); better = worse = 0
            for e in exps:
                m = (frame.experiment_id == e).to_numpy()
                if not (off & m).any(): continue
                d = (a1[m] == y[m]).mean() - (a0[m] == y[m]).mean()
                better += d > 0; worse += d < 0
            r.update(exps_with_gasoff=int(sum((off & (frame.experiment_id == e).to_numpy()).any() for e in exps)),
                     exps_better=int(better), exps_worse=int(worse))
            if split == 'validation':  # 기존 보고서 별칭 E16(정렬 16번째) 제외
                keep = (frame.experiment_id != exps[15]).to_numpy()
                r.update(f1_before_noE16=macro(y[keep], a0[keep]), f1_after_noE16=macro(y[keep], a1[keep]))
        else:
            ch = a0 != a1
            r.update(change_pairs={f'{x}->{z}': int(n) for (x, z), n in
                                   pd.Series(list(zip(a0[ch], a1[ch]))).value_counts().items()} if ch.any() else {})
        rows.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)

    a.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(a.output / 'summary.csv', index=False)
    frame, pred = out['test']
    sample = pd.read_csv(a.data_dir / 'sample_submission.csv')
    sub = sample[['sample_id']].copy(); sub['label'] = sub.sample_id.map(pd.Series(pred, index=frame.sample_id)).astype(int)
    sub.to_csv(a.output / 'submission_ABC091_gasffill.csv', index=False)
    print('wrote', a.output / 'submission_ABC091_gasffill.csv')


if __name__ == '__main__':
    main()
