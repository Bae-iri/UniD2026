"""ABC B: matched head-only fine-tuning with elapsed time and causal extrema.
Encoders/query stay frozen. Existing checkpoints and data are never modified.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from abc_reference import abc_fusion_experiment as abc

MODES = ['B', 'B_time', 'B_extrema', 'B_extrema_time']
SIGNALS = ['meas_v', 'meas_t1', 'meas_t2', 'thermal_t']
EXTRA_NAMES = ['elapsed_log1p'] + [f'{c}_{s}' for c in SIGNALS for s in ['extreme', 'difference', 'age_log1p', 'valid']]


def metrics(y, pred):
    cm = np.zeros((6, 6), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, int)-1, np.asarray(pred, int)-1), 1)
    denominator = cm.sum(0)+cm.sum(1)
    f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(6, float), where=denominator > 0)
    return {'macro_f1': float(f1.mean()), 'class_f1': f1.tolist(), 'confusion': cm.tolist()}


def causal_features(frame):
    """All observations up to current; never uses labels, future or duration."""
    if frame.experiment_id.isna().any():
        raise ValueError('Missing experiment_id')
    ts = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    if not np.isfinite(ts).all() or (ts < 0).any():
        raise ValueError('Time must be finite and nonnegative')
    raw = frame[SIGNALS].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    out = np.zeros((len(frame), 17), dtype=np.float64)
    out[:, 0] = np.log1p(ts)
    for indices in frame.groupby('experiment_id', sort=False).indices.values():
        ordered = indices[np.argsort(ts[indices], kind='stable')]
        if (np.diff(ts[ordered]) <= 0).any():
            raise ValueError('Duplicate timestamps within experiment')
        extrema = np.full(4, np.nan)
        when = np.zeros(4)
        for row in ordered:
            for k in range(4):
                value = raw[row, k]
                if np.isfinite(value):
                    if not np.isfinite(extrema[k]) or (value < extrema[k] if k == 0 else value > extrema[k]):
                        extrema[k], when[k] = value, ts[row]
                j = 1+4*k
                if np.isfinite(extrema[k]):
                    out[row, j] = extrema[k]
                    # Difference is unavailable when current observation is missing.
                    out[row, j+1] = value-extrema[k] if np.isfinite(value) else np.nan
                    out[row, j+2] = np.log1p(ts[row]-when[k])
                    out[row, j+3] = 1
                else:
                    out[row, j:j+3] = np.nan
    return out


def fit_extra(raw):
    mean = np.zeros(17); scale = np.ones(17)
    for j in range(17):
        if j in [4, 8, 12, 16]:
            continue
        finite = raw[:, j][np.isfinite(raw[:, j])]
        if len(finite):
            mean[j] = finite.mean()
            scale[j] = max(finite.std(), 1e-6)
    return {'mean': mean.tolist(), 'scale': scale.tolist(), 'names': EXTRA_NAMES}


def transform_extra(raw, prep):
    scaled = (raw-np.asarray(prep['mean']))/np.asarray(prep['scale'])
    return np.where(np.isfinite(scaled), scaled, 0).astype(np.float32)


def mode_features(extra, mode):
    if mode not in MODES:
        raise ValueError('Unknown mode')
    mask = extra.new_zeros(17)
    if mode in ['B_time', 'B_extrema_time']:
        mask[0] = 1
    if mode in ['B_extrema', 'B_extrema_time']:
        for k in range(4):
            j = 1+4*k
            mask[j:j+2] = 1
            mask[j+3] = 1
            if mode == 'B_extrema_time':
                mask[j+2] = 1
    return extra*mask


class PhaseHead(nn.Module):
    def __init__(self, original_head, mode):
        super().__init__()
        self.mode = mode
        old = original_head[0]
        self.net = nn.Sequential(nn.Linear(old.in_features+17, old.out_features), nn.ReLU(),
                                 nn.Dropout(.1), nn.Linear(old.out_features, 6))
        with torch.no_grad():
            self.net[0].weight.zero_()
            self.net[0].weight[:, :old.in_features].copy_(old.weight)
            self.net[0].bias.copy_(old.bias)
            self.net[3].load_state_dict(original_head[3].state_dict())

    def forward(self, features, extra):
        return self.net(torch.cat([features, mode_features(extra, self.mode)], 1))


@torch.inference_mode()
def extract(model, data, device):
    result = []; captured = {}
    def capture(module, args):
        captured['features'] = args[0]
    handle = model.head.register_forward_pre_hook(capture)
    model.eval()
    try:
        for start in range(0, len(data.frame), 128):
            ids = torch.arange(start, min(start+128, len(data.frame)))
            model(*data.batch(ids, device))
            result.append(captured['features'].cpu().clone())
    finally:
        handle.remove()
    return torch.cat(result)


@torch.inference_mode()
def evaluate(head, features, extra, device):
    head.eval(); parts = []
    for start in range(0, len(features), 256):
        parts.append(head(features[start:start+256].to(device), extra[start:start+256].to(device)).argmax(1).cpu().numpy()+1)
    return np.concatenate(parts)


def load_data(args):
    config = json.loads((args.base_run/'config.json').read_text())
    for split in ['train', 'validation']:
        if hashlib.sha256((args.data_dir/split/f'{split}.csv').read_bytes()).hexdigest() != config['csv_sha256'][split]:
            raise ValueError('CSV differs from original ABC run')
    frames = abc.common.split_frames(args.data_dir, config['split_seed'], .2)
    meta = json.loads((args.base_run/'preprocessing.json').read_text())
    if meta['electro'] != abc.ELECTRO or meta['gas'] != abc.GAS:
        raise ValueError('Feature schema differs')
    prep = abc.SensorPreprocessor.__new__(abc.SensorPreprocessor)
    for key in ['median', 'mean', 'std']:
        setattr(prep, key, pd.Series(meta[key], index=abc.NUMERIC))
    manifest = json.loads((args.base_run/'image_cache_manifest.json').read_text())
    # Saved split manifest also checks fit order against its cached images.
    split_manifest = args.base_run/'split_manifest.csv'
    saved = pd.read_csv(split_manifest) if split_manifest.exists() else None
    datasets = {}
    for split, frame in frames.items():
        if saved is not None:
            original = saved[saved.split == split]
            if not np.array_equal(original.sample_id, frame.sample_id):
                raise ValueError('Original split/cache order differs')
        if split in ['dev', 'validation']:
            ref = pd.read_csv(args.base_run/f'seed_{args.seeds[0]}/B/{split}_predictions.csv')
            if not np.array_equal(ref.sample_id, frame.sample_id) or not np.array_equal(ref.label, frame.label):
                raise ValueError('Original targets/order differ')
        pixels = np.load(manifest[split], mmap_mode='c')
        if pixels.shape != (len(frame), 4, 128, 128):
            raise ValueError('Cache shape differs')
        q, gas = prep.transform(frame)
        datasets[split] = abc.Data(frame, torch.from_numpy(pixels), torch.from_numpy(q), torch.from_numpy(gas), None)
    raw = {split: causal_features(f) for split, f in frames.items()}
    norm = fit_extra(raw['fit'])
    extras = {split: torch.from_numpy(transform_extra(v, norm)) for split, v in raw.items()}
    return frames, datasets, extras, norm, meta


def run(args):
    if args.output.exists():
        raise FileExistsError('Use a NEW output directory')
    torch.set_num_threads(2)
    frames, datasets, extras, norm, prep = load_data(args)
    # All checkpoint paths are checked before creating output or training.
    paths = {seed: args.base_run/f'seed_{seed}/B/best.pt' for seed in args.seeds}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(scope='frozen encoders/query; head-only matched continued training',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  base_checkpoint_sha256={str(seed): hashlib.sha256(p.read_bytes()).hexdigest() for seed, p in paths.items()})
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    (args.output/'extra_preprocessing.json').write_text(json.dumps(norm, indent=2))
    (args.output/'preprocessing.json').write_text(json.dumps(prep, indent=2))
    labels = {s: torch.tensor(f.label.to_numpy(int)-1) for s, f in frames.items()}
    weights = torch.tensor(prep['class_weights'], device=args.device)
    summary = []
    for seed in args.seeds:
        ck = torch.load(paths[seed], map_location='cpu', weights_only=True)
        base = abc.FusionModel('B').to(args.device)
        base.load_state_dict(ck['state']); base.eval()
        features = {}
        for split, data in datasets.items():
            features[split] = extract(base, data, args.device)
            print(f'Frozen features ready: seed={seed} split={split} rows={len(data.frame)}', flush=True)
        # Original head supplies the shared baseline and exact epoch-0 predictions.
        original_head = base.head.cpu()
        for mode in args.modes:
            abc.common.seed_everything(seed)
            head = PhaseHead(original_head, mode).to(args.device)
            optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
            folder = args.output/f'{mode}_seed{seed}'; folder.mkdir()
            history = []; best = -1.; order_rng = torch.Generator().manual_seed(seed)
            started = time.monotonic()
            for epoch in range(args.epochs+1):
                if epoch:
                    head.train(); order = torch.randperm(len(labels['fit']), generator=order_rng)
                    for start in range(0, len(order), 64):
                        ids = order[start:start+64]
                        optimizer.zero_grad(set_to_none=True)
                        loss = nn.functional.cross_entropy(head(features['fit'][ids].to(args.device), extras['fit'][ids].to(args.device)),
                                                           labels['fit'][ids].to(args.device), weight=weights)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite loss')
                        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 5); optimizer.step()
                pred = evaluate(head, features['dev'], extras['dev'], args.device)
                result = metrics(frames['dev'].label, pred)
                history.append({'epoch': epoch, 'dev_macro_f1': result['macro_f1']})
                (folder/'history.json').write_text(json.dumps(history, indent=2))
                if result['macro_f1'] > best:
                    best = result['macro_f1']
                    torch.save({'head_state': {k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
                                'base_state': ck['state'], 'seed': seed, 'mode': mode, 'best_epoch': epoch,
                                'dev_macro_f1': best, 'extra_preprocessing': norm, 'preprocessing': prep,
                                'signals': SIGNALS, 'config': config}, folder/'best.pt')
                print(f'{mode} seed={seed} epoch={epoch}/{args.epochs} dev_macro_f1={result["macro_f1"]:.6f}', flush=True)
            selected = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
            head.load_state_dict(selected['head_state'])
            pred = evaluate(head, features['validation'], extras['validation'], args.device)
            result = metrics(frames['validation'].label, pred)
            groups = []
            for i, ids in enumerate(frames['validation'].groupby('experiment_id', sort=True).indices.values()):
                y = frames['validation'].label.to_numpy(int)[ids]; p = pred[ids]
                groups.append({'experiment': f'E{i+1:02}', **metrics(y, p),
                               'five_to_four': int(((y == 5)&(p == 4)).sum()), 'five_correct': int(((y == 5)&(p == 5)).sum())})
            (folder/'validation_aggregates.json').write_text(json.dumps({'overall':result, 'experiments':groups}, indent=2))
            e16 = next((g for g in groups if g['experiment'] == 'E16'), None)
            row = {'mode':mode, 'seed':seed, 'best_epoch':selected['best_epoch'], 'dev_macro_f1':best,
                   'val_macro_f1':result['macro_f1'], 'F1_3':result['class_f1'][2], 'F1_4':result['class_f1'][3],
                   'F1_5':result['class_f1'][4], 'F1_6':result['class_f1'][5], 'seconds':time.monotonic()-started,
                   'E16_5_to_4':e16['five_to_four'] if e16 else None, 'E16_5_correct':e16['five_correct'] if e16 else None}
            summary.append(row)
            pd.DataFrame(summary).to_csv(args.output/'summary.csv', index=False)
            print('VALIDATION', json.dumps(row), flush=True)
        del base, features
    table = pd.DataFrame(summary).pivot(index='seed', columns='mode', values='val_macro_f1')
    differences = pd.DataFrame(index=table.index)
    for a, b in [('B_time','B'), ('B_extrema','B'), ('B_extrema_time','B'), ('B_extrema_time','B_time'), ('B_extrema_time','B_extrema')]:
        if a in table and b in table:
            differences[f'{a}-{b}'] = table[a]-table[b]
    differences.to_csv(args.output/'paired_differences.csv')
    print(differences.to_string(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run',type=Path,default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    p.add_argument('--data-dir',type=Path,default=Path('/home/ubuntu/data/battery_data_light'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    if args.epochs<1 or not np.isfinite(args.lr) or args.lr<=0 or len(set(args.seeds))!=len(args.seeds) or len(set(args.modes))!=len(args.modes):
        p.error('Invalid epochs, learning rate, seeds or modes')
    run(args)
