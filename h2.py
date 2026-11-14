"""H2 image history comparison. Standalone: no h1.py import required.
I0=current image, I1=past mean+difference, I2=time-aware unidirectional GRU.
Common current-observation branch follows H1 late. No competition data is read
unless the user runs train/predict on their server.
"""
import argparse
import csv
import json
import random
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ELECTRIC = ['meas_v', 'meas_t1', 'meas_t2', 'thermal_t', 'atmp_t', 'indu_t']
GAS = ['meas_p', 'meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn',
       'meas_hcl', 'meas_hf', 'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2']
CATS = [('bat_class', 3), ('soc_status', 2), ('tr_method', 2)]
NUMERIC = ELECTRIC + GAS + ['relative_time_s', 'bat_cc']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def raw_numeric(frame):
    a = frame[NUMERIC].apply(pd.to_numeric, errors='coerce').to_numpy(np.float64)
    a[~np.isfinite(a)] = np.nan
    # Fixed, sample-independent transformation for potentially skewed signals.
    for col in GAS + ['relative_time_s', 'bat_cc']:
        k = NUMERIC.index(col)
        a[:, k] = np.sign(a[:, k]) * np.log1p(np.abs(a[:, k]))
    return a


def fit_preprocess(frame):
    a = raw_numeric(frame)
    med = []
    for k in range(a.shape[1]):
        good = a[:, k][np.isfinite(a[:, k])]
        med.append(float(np.median(good)) if len(good) else 0.0)
    a = np.where(np.isnan(a), np.array(med), a)
    std = a.std(0)
    std[std < 1e-6] = 1.0
    return {'median': med, 'mean': a.mean(0).tolist(), 'std': std.tolist()}


def transform(frame, prep):
    a = raw_numeric(frame)
    missing = np.isnan(a).astype(np.float32)
    a = np.where(np.isnan(a), np.array(prep['median']), a)
    a = ((a - np.array(prep['mean'])) / np.array(prep['std'])).astype(np.float32)
    # Categorical mappings are fixed by the published schema, not fitted on Test.
    cats = []
    for col, maximum in CATS:
        v = pd.to_numeric(frame[col], errors='coerce').to_numpy(float)
        valid = np.isfinite(v) & (v >= 1) & (v <= maximum) & (v == np.floor(v))
        cats.append(np.where(valid, v, 0).astype(np.int64))
    ne, ng = len(ELECTRIC), len(GAS)
    return {
        'electric': np.concatenate([a[:, :ne], missing[:, :ne]], 1),
        'gas': np.stack([a[:, ne:ne+ng], missing[:, ne:ne+ng]], -1),
        'time': np.stack([a[:, -2], missing[:, -2]], -1),
        'capacity': np.stack([a[:, -1], missing[:, -1]], -1),
        'cats': np.stack(cats, -1),
    }


class H1Model(nn.Module):
    def __init__(self, mode, image_size=96, dim=64):
        super().__init__()
        if mode not in ('late', 'conditioned') or image_size % 16:
            raise ValueError('mode=late|conditioned; image_size must be divisible by 16')
        self.mode = mode
        self.category_embeddings = nn.ModuleList([nn.Embedding(n+1, 8) for _, n in CATS])
        self.static = nn.Sequential(nn.Linear(26, dim), nn.GELU(), nn.LayerNorm(dim))
        self.electric = nn.Sequential(nn.Linear(2*len(ELECTRIC)+2, dim),
                                      nn.GELU(), nn.LayerNorm(dim))
        self.gas_value = nn.Linear(2, dim)
        self.gas_id = nn.Embedding(len(GAS), dim)
        layers, cin = [], 3
        for cout in (16, 32, 64, dim):
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1),
                       nn.GroupNorm(4, cout), nn.GELU()]
            cin = cout
        self.cnn = nn.Sequential(*layers)
        self.image_position = nn.Parameter(torch.zeros(1, (image_size//16)**2, dim))
        nn.init.normal_(self.image_position, std=0.02)
        self.query = nn.Sequential(nn.Linear(2*dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.gas_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.image_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.head = nn.Sequential(nn.Linear(6*dim, 2*dim), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(2*dim, 6))

    def encode(self, batch):
        c = torch.cat([emb(batch['cats'][:, i])
                       for i, emb in enumerate(self.category_embeddings)] + [batch['capacity']], -1)
        static = self.static(c)
        electric = self.electric(torch.cat([batch['electric'], batch['time']], -1))
        ids = torch.arange(len(GAS), device=electric.device)
        gas = self.gas_value(batch['gas']) + self.gas_id(ids)[None]
        image_map = self.cnn(batch['image'])
        # Downsample validity mask; keep original global features in both arms.
        valid = torch.nn.functional.adaptive_max_pool2d(
            batch['image_mask'], image_map.shape[-2:]).flatten(1).bool()
        image = image_map.flatten(2).transpose(1, 2)
        image_global = (image * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return static, electric, gas, image + self.image_position, image_global, valid

    def select(self, electric, static, gas, image, valid):
        # The sole architectural intervention: static context on/off in query.
        context = static if self.mode == 'conditioned' else torch.zeros_like(static)
        query = self.query(torch.cat([electric, context], -1))[:, None]
        g = self.gas_attention(query, gas, gas, need_weights=False)[0][:, 0]
        v = self.image_attention(query, image, image,
                                 key_padding_mask=~valid, need_weights=False)[0][:, 0]
        return g, v

    def forward(self, batch):
        static, electric, gas, image, image_global, valid = self.encode(batch)
        g, v = self.select(electric, static, gas, image, valid)
        return self.head(torch.cat([static, electric, gas.mean(1), image_global, g, v], -1))


def metrics(y, pred):
    cm = np.zeros((6, 6), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, int), np.asarray(pred, int)), 1)
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2*tp, den, out=np.zeros(6, float), where=den != 0)
    return {'macro_f1': float(f1.mean()), 'class_f1': f1.tolist(), 'confusion': cm.tolist()}


def read_frame(path):
    return pd.read_csv(path, dtype={'sample_id': 'string', 'experiment_id': 'string'})


def history_indices(frame, window_s):
    """[t-window_s, t], within one experiment, independent of input row order.

    Duplicate timestamps within an experiment are rejected rather than ordered
    arbitrarily. No future observation, label, or full-experiment statistic is used.
    """
    if window_s <= 0 or not np.isfinite(window_s):
        raise ValueError('window_s must be positive and finite')
    if frame.experiment_id.isna().any():
        raise ValueError('experiment_id contains missing values')
    t = pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError('relative_time_s must be finite and nonnegative')
    histories = [None] * len(frame)
    for indices in frame.groupby('experiment_id', sort=False).indices.values():
        ordered = indices[np.argsort(t[indices], kind='mergesort')]
        times = t[ordered]
        if (np.diff(times) <= 0).any():
            raise ValueError('Duplicate observation times within one experiment')
        for j, row in enumerate(ordered):
            start = np.searchsorted(times, times[j]-window_s, side='left')
            histories[row] = ordered[start:j+1].copy()
    return histories, t


class ImageHistoryDataset(Dataset):
    def __init__(self, frame, image_root, prep, image_size, window_s, variant):
        self.frame = frame.reset_index(drop=True).copy()
        required = NUMERIC + [c for c, _ in CATS] + ['sample_id', 'experiment_id', 'image_path']
        if not set(required).issubset(self.frame):
            missing = sorted(set(required)-set(self.frame))
            raise ValueError(f'Missing required column names: {missing}')
        if self.frame.sample_id.isna().any() or not self.frame.sample_id.is_unique:
            raise ValueError('sample_id must be nonmissing and unique')
        self.histories, self.times = history_indices(self.frame, window_s)
        self.window, self.size, self.variant = window_s, image_size, variant
        self.arrays = {k: torch.from_numpy(v) for k, v in transform(self.frame, prep).items()}
        self.y = torch.full((len(frame),), -1, dtype=torch.long)
        if 'label' in self.frame:
            y = pd.to_numeric(self.frame.label, errors='raise').to_numpy(float)
            if not np.isin(y, np.arange(1, 7)).all():
                raise ValueError('labels must be integers 1..6')
            self.y = torch.tensor(y.astype(np.int64)-1)
        self.paths = []
        for rel in self.frame.image_path.astype(str):
            rel = Path(rel)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('image_path must stay within its split directory')
            path = Path(image_root)/rel
            if not path.is_file():
                raise FileNotFoundError('An image is missing; check --image-root/data split paths')
            self.paths.append(path)

    def __len__(self):
        return len(self.frame)

    @lru_cache(maxsize=256)
    def image(self, row):
        with Image.open(self.paths[row]) as src:
            im = src.convert('RGB')
            im.thumbnail((self.size, self.size), Image.Resampling.BILINEAR)
            canvas = Image.new('RGB', (self.size, self.size))
            x, y = (self.size-im.width)//2, (self.size-im.height)//2
            canvas.paste(im, (x, y))
            pixels = np.asarray(canvas, dtype=np.uint8).copy()
            mask = np.zeros((1, self.size, self.size), dtype=np.uint8)
            mask[:, y:y+im.height, x:x+im.width] = 1
        return torch.from_numpy(pixels.transpose(2, 0, 1)), torch.from_numpy(mask)

    def __getitem__(self, row):
        indices = self.histories[row]
        # I0 reads only the current image. I2_single is an optional capacity control.
        if self.variant in ('I0', 'I2_single'):
            indices = np.array([row])
        ts = self.times[indices]
        delta = np.diff(ts, prepend=ts[0])
        age = self.times[row]-ts
        time_features = np.stack([np.log1p(delta), np.log1p(age)], -1)
        time_features = (time_features/np.log1p(self.window)).astype(np.float32)
        images, masks = zip(*(self.image(int(i)) for i in indices))
        return {**{k: v[row] for k, v in self.arrays.items()},
                'images': torch.stack(images), 'masks': torch.stack(masks),
                'temporal_time': torch.from_numpy(time_features), 'target': self.y[row]}


def collate_history(items):
    padded = ('images', 'masks', 'temporal_time')
    result = {k: torch.stack([item[k] for item in items])
              for k in items[0] if k not in padded}
    result['lengths'] = torch.tensor([len(item['images']) for item in items])
    for key in padded:
        result[key] = nn.utils.rnn.pad_sequence([item[key] for item in items], batch_first=True)
    return result


class H2Model(H1Model):
    """Same late current branch in every variant; replace only the history branch."""
    def __init__(self, variant, image_size=96, dim=64):
        super().__init__('late', image_size, dim)
        if variant not in ('I0', 'I1', 'I2', 'I2_single'):
            raise ValueError('Unknown variant')
        self.variant, self.dim = variant, dim
        self.temporal = nn.GRU(dim+2, 2*dim, batch_first=True, bidirectional=False)
        # Same head width and initialized weights in every H2 arm.
        self.head = nn.Sequential(nn.Linear(8*dim, 2*dim), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(2*dim, 6))

    def forward(self, batch):
        lengths = batch['lengths']
        b, steps = batch['images'].shape[:2]
        valid_steps = torch.arange(steps, device=lengths.device)[None] < lengths[:, None]
        # Padding frames never enter CNN or its normalization layers.
        pixels = batch['images'][valid_steps].float()/255.0
        pixel_mask = batch['masks'][valid_steps].float()
        feature_map = self.cnn(pixels)
        valid_pixels = nn.functional.adaptive_max_pool2d(
            pixel_mask, feature_map.shape[-2:]).flatten(1).bool()
        spatial = feature_map.flatten(2).transpose(1, 2)
        global_features = ((spatial*valid_pixels[:, :, None]).sum(1)
                           / valid_pixels.sum(1, keepdim=True).clamp_min(1))
        current = lengths.cumsum(0)-1
        image_global = global_features[current]
        image_tokens = spatial[current]+self.image_position
        current_valid = valid_pixels[current]

        c = torch.cat([embedding(batch['cats'][:, i])
                       for i, embedding in enumerate(self.category_embeddings)]
                      + [batch['capacity']], -1)
        static = self.static(c)
        electric = self.electric(torch.cat([batch['electric'], batch['time']], -1))
        gas_ids = torch.arange(len(GAS), device=pixels.device)
        gas = self.gas_value(batch['gas'])+self.gas_id(gas_ids)[None]
        selected_gas, selected_image = self.select(
            electric, static, gas, image_tokens, current_valid)

        if self.variant == 'I0':
            history = image_global.new_zeros(b, 2*self.dim)
        else:
            seq = nn.utils.rnn.pad_sequence(
                list(global_features.split(lengths.cpu().tolist())), batch_first=True)
            if self.variant == 'I1':
                # Strictly earlier observations. The current frame is excluded from mean.
                past = torch.arange(steps, device=lengths.device)[None] < (lengths-1)[:, None]
                mean = (seq*past[:, :, None]).sum(1)/(lengths-1).clamp_min(1)[:, None]
                history = torch.cat([mean, image_global-mean], -1)
                history = history*(lengths > 1)[:, None]
            else:
                temporal_input = torch.cat([seq, batch['temporal_time']], -1)
                packed = nn.utils.rnn.pack_padded_sequence(
                    temporal_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, last = self.temporal(packed)
                history = last[0]
        fused = torch.cat([static, electric, gas.mean(1), image_global,
                           selected_gas, selected_image, history], -1)
        return self.head(fused)


def make_loader(dataset, args, training=False, seed=0):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                      num_workers=args.workers, collate_fn=collate_history,
                      pin_memory=args.device.startswith('cuda'),
                      persistent_workers=args.workers > 0,
                      generator=torch.Generator().manual_seed(seed))


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate_h2(model, batches, device):
    model.eval()
    labels, predictions = [], []
    for batch in batches:
        batch = to_device(batch, device)
        predictions.extend(model(batch).argmax(1).cpu().tolist())
        labels.extend(batch['target'].cpu().tolist())
    return np.asarray(labels), np.asarray(predictions)


def check_split_separation(train, val):
    for col in ('sample_id', 'experiment_id'):
        if not set(train[col]).isdisjoint(val[col]):
            raise ValueError(f'Train and Validation overlap in {col}')


def comparison_table(summary, destination):
    table = pd.DataFrame(summary).pivot(index='seed', columns='variant', values='val_macro_f1')
    differences = pd.DataFrame(index=table.index)
    for first, second in [('I1', 'I0'), ('I2', 'I1'), ('I2', 'I0'), ('I2', 'I2_single')]:
        if first in table and second in table:
            differences[f'{first}-{second}'] = table[first]-table[second]
    if len(differences.columns):
        differences.to_csv(destination/'paired_differences.csv')
        print('\nPAIRED COMPARISON (Macro F1)', flush=True)
        for name in differences:
            for seed, delta in differences[name].items():
                print(f'{name} seed={seed}: {delta:+.6f}', flush=True)
            print(f'{name} mean_delta={differences[name].mean():+.6f}', flush=True)
    aggregates = pd.DataFrame(summary).groupby('variant')['val_macro_f1'].agg(['mean', 'std', 'count'])
    aggregates.to_csv(destination/'aggregate.csv')
    print(aggregates.to_string(), flush=True)


def train_h2(args):
    # Preflight every run before training anything; never overwrite earlier runs.
    folders = [args.output/f'{mode}_seed{seed}' for seed in args.seeds for mode in args.modes]
    if any(p.exists() for p in folders):
        raise FileExistsError('Run folder already exists. Choose a NEW --output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config['torch_version'] = str(torch.__version__)
    config['features'] = NUMERIC
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    train_frame = read_frame(args.data_dir/'train/train.csv')
    val_frame = read_frame(args.data_dir/'validation/validation.csv')
    if 'label' not in train_frame or 'label' not in val_frame:
        raise ValueError('Train and Validation labels are required')
    check_split_separation(train_frame, val_frame)
    prep = fit_preprocess(train_frame)
    summary = []
    device = torch.device(args.device)
    for seed in args.seeds:
        for variant in args.modes:
            seed_all(seed)
            ds_train = ImageHistoryDataset(train_frame, args.data_dir/'train', prep,
                                           args.image_size, args.window_s, variant)
            ds_val = ImageHistoryDataset(val_frame, args.data_dir/'validation', prep,
                                         args.image_size, args.window_s, variant)
            train_batches = make_loader(ds_train, args, True, seed)
            val_batches = make_loader(ds_val, args, False, seed)
            model = H2Model(variant, args.image_size, args.dim).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
            folder = args.output/f'{variant}_seed{seed}'
            folder.mkdir()
            history, best, best_epoch = [], -1.0, 0
            started = time.monotonic()
            print(f'\nSTART {variant} seed={seed} window={args.window_s}s', flush=True)
            parameter_total = sum(p.numel() for p in model.parameters())
            active_parameters = None
            for epoch in range(1, args.epochs+1):
                model.train()
                epoch_started = time.monotonic()
                total, rows = 0.0, 0
                for step, batch in enumerate(train_batches, 1):
                    batch = to_device(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(batch), batch['target'])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    loss.backward()
                    if active_parameters is None:
                        active_parameters = sum(p.numel() for p in model.parameters() if p.grad is not None)
                        print(f'parameters: allocated={parameter_total}, active={active_parameters}', flush=True)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    n = len(batch['target'])
                    total += loss.item()*n
                    rows += n
                    if step % args.log_every == 0:
                        print(f'{variant} seed={seed} epoch={epoch}/{args.epochs} '
                              f'batch={step}/{len(train_batches)} loss={total/rows:.4f}', flush=True)
                y, pred = evaluate_h2(model, val_batches, device)
                result = metrics(y, pred)
                epoch_seconds = time.monotonic()-epoch_started
                record = {'epoch': epoch, 'train_loss': total/rows,
                          'seconds': epoch_seconds, **result}
                history.append(record)
                (folder/'history.json').write_text(json.dumps(history, indent=2))
                print(f'{variant} seed={seed} epoch={epoch}/{args.epochs} '
                      f'val_macro_f1={result["macro_f1"]:.6f} seconds={epoch_seconds:.1f}', flush=True)
                if result['macro_f1'] > best:
                    best, best_epoch = result['macro_f1'], epoch
                    torch.save({'state_dict': model.state_dict(), 'variant': variant,
                                'image_size': args.image_size, 'dim': args.dim,
                                'window_s': args.window_s, 'preprocess': prep,
                                'seed': seed, 'best_epoch': epoch, 'validation': result,
                                'config': config}, folder/'best.pt')
            summary.append({'variant': variant, 'seed': seed, 'best_epoch': best_epoch,
                            'val_macro_f1': best, 'parameters_allocated': parameter_total,
                            'parameters_active': active_parameters,
                            'train_validation_seconds': time.monotonic()-started})
            pd.DataFrame(summary).to_csv(args.output/'summary.csv', index=False)
            del model, optimizer, train_batches, val_batches, ds_train, ds_val
            # Per-dataset image caches use method self as a key; explicitly release them.
            ImageHistoryDataset.image.cache_clear()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    comparison_table(summary, args.output)


def predict_h2(args):
    if args.output.exists():
        raise FileExistsError('Submission output exists; choose a new --output filename')
    # Load only this script's own trusted checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    frame = read_frame(args.input)
    ds = ImageHistoryDataset(frame, args.image_root, checkpoint['preprocess'],
                             checkpoint['image_size'], checkpoint['window_s'], checkpoint['variant'])
    model = H2Model(checkpoint['variant'], checkpoint['image_size'], checkpoint['dim']).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    _, predictions = evaluate_h2(model, make_loader(ds, args), args.device)
    by_id = pd.Series(predictions+1, index=frame.sample_id.to_numpy())
    order = frame.sample_id if args.sample_submission is None else read_frame(args.sample_submission).sample_id
    if order.isna().any() or not order.is_unique or set(order) != set(by_id.index):
        raise ValueError('Input and sample submission IDs do not match')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'sample_id': order.to_numpy(), 'label': by_id.loc[order].to_numpy()}).to_csv(args.output, index=False)
    print(f'Saved {args.output}; rows={len(order)}')


def smoke_test_h2():
    """No competition inputs. Tests live in this same standalone file."""
    import tempfile
    torch.set_num_threads(1)
    seed_all(42)
    with tempfile.TemporaryDirectory(prefix='h2-synthetic-') as temporary:
        root = Path(temporary)
        # Deliberately unsorted rows; two separate experiments and irregular gaps.
        times = [25, 0, 10, 80, 15, 0, 5]
        experiments = ['A', 'A', 'A', 'A', 'B', 'B', 'B']
        frame = pd.DataFrame({col: np.arange(7, dtype=float)+1 for col in NUMERIC})
        frame['sample_id'] = [f'synthetic_{i}' for i in range(7)]
        frame['experiment_id'] = experiments
        frame['relative_time_s'] = times
        frame['bat_class'], frame['soc_status'], frame['tr_method'] = 1, 1, 1
        frame['image_path'] = [f'{i}.png' for i in range(7)]
        frame['label'] = [1, 2, 3, 4, 5, 6, 1]
        frame.loc[0, 'indu_t'] = np.nan
        for i in range(7):
            rgb = np.random.default_rng(i).integers(0, 256, (29+i, 23, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(root/f'{i}.png')
        prep = fit_preprocess(frame)
        all_histories, _ = history_indices(frame, 60)
        assert all_histories[0].tolist() == [1, 2, 0]
        assert all_histories[3].tolist() == [0, 3]
        assert all_histories[4].tolist() == [5, 6, 4]
        datasets = {v: ImageHistoryDataset(frame, root, prep, 32, 60, v)
                    for v in ('I0', 'I1', 'I2', 'I2_single')}
        initial = None
        for variant, ds in datasets.items():
            seed_all(123)
            model = H2Model(variant, 32, 16)
            if initial is None:
                initial = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                assert all(torch.equal(initial[k], v) for k, v in model.state_dict().items())
            batch = collate_history([ds[0], ds[1], ds[4]])
            model.train()
            output = model(batch)
            assert output.shape == (3, 6) and torch.isfinite(output).all()
            nn.functional.cross_entropy(output, batch['target']).backward()
            assert model.cnn[0].weight.grad.abs().sum() > 0
            assert model.electric[0].weight.grad.abs().sum() > 0
            if variant.startswith('I2'):
                assert model.temporal.weight_ih_l0.grad.abs().sum() > 0
            model.eval()
            with torch.no_grad():
                expected = model(collate_history([ds[0]]))
                # Batch/padding independence.
                together = model(batch)[:1]
                assert torch.allclose(expected, together, atol=2e-5)
                poison = {k: v.clone() for k, v in batch.items()}
                invalid = torch.arange(batch['images'].shape[1])[None] >= batch['lengths'][:, None]
                poison['images'][invalid] = 255
                poison['masks'][invalid] = 1
                poison['temporal_time'][invalid] = 1000
                assert torch.allclose(model(batch), model(poison), atol=2e-5)
                # Remove every future observation and every other experiment.
                prefix = frame[(frame.experiment_id == 'A') & (frame.relative_time_s <= 25)].copy()
                truncated = ImageHistoryDataset(prefix, root, prep, 32, 60, variant)
                row = truncated.frame.index[truncated.frame.sample_id == frame.sample_id.iloc[0]][0]
                actual = model(collate_history([truncated[row]]))
                assert torch.allclose(expected, actual, atol=2e-5)
                # Changing a future image does not change this input or prediction.
                if variant == 'I2':
                    Image.new('RGB', (23, 32), (255, 0, 0)).save(root/'3.png')
                    ImageHistoryDataset.image.cache_clear()
                    assert torch.allclose(expected, model(collate_history([ds[0]])), atol=2e-5)
                # Checkpoint weights roundtrip.
                weights_path = root/f'{variant}.pt'
                torch.save(model.state_dict(), weights_path)
                restored = H2Model(variant, 32, 16).eval()
                restored.load_state_dict(torch.load(weights_path, weights_only=True))
                assert torch.allclose(expected, restored(collate_history([ds[0]])), atol=2e-5)
        try:
            history_indices(pd.concat([frame, frame.iloc[[0]]], ignore_index=True), 60)
            raise AssertionError('Duplicate times were not rejected')
        except ValueError:
            pass
        ImageHistoryDataset.image.cache_clear()
    print('Synthetic tests passed: histories, no future/cross-experiment leakage, padding, '
          'batch independence, initialization, gradients, checkpoint restore. No competition data read.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true')
    commands = parser.add_subparsers(dest='command')
    tr = commands.add_parser('train')
    tr.add_argument('--data-dir', type=Path, required=True)
    tr.add_argument('--output', type=Path, default=Path('runs_h2_v1'))
    tr.add_argument('--modes', nargs='+', choices=['I0', 'I1', 'I2', 'I2_single'], default=['I0', 'I1', 'I2'])
    tr.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    tr.add_argument('--epochs', type=int, default=20)
    tr.add_argument('--window-s', type=float, default=60.0)
    tr.add_argument('--image-size', type=int, default=96)
    tr.add_argument('--dim', type=int, default=64)
    tr.add_argument('--lr', type=float, default=3e-4)
    tr.add_argument('--log-every', type=int, default=50)
    pr = commands.add_parser('predict')
    pr.add_argument('--input', type=Path, required=True)
    pr.add_argument('--image-root', type=Path, required=True)
    pr.add_argument('--checkpoint', type=Path, required=True)
    pr.add_argument('--sample-submission', type=Path)
    pr.add_argument('--output', type=Path, required=True)
    for command in (tr, pr):
        command.add_argument('--batch-size', type=int, default=32)
        command.add_argument('--workers', type=int, default=4)
        command.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test_h2()
    elif args.command == 'train':
        if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.log_every < 1
                or args.dim < 4 or args.dim % 4 or args.image_size < 16 or args.image_size % 16
                or not np.isfinite(args.window_s) or args.window_s <= 0
                or not np.isfinite(args.lr) or args.lr <= 0
                or len(set(args.seeds)) != len(args.seeds) or len(set(args.modes)) != len(args.modes)):
            parser.error('Invalid arguments: check sizes, window, seeds, modes, epochs and learning rate')
        train_h2(args)
    elif args.command == 'predict':
        if args.batch_size < 1 or args.workers < 0:
            parser.error('batch-size must be positive, workers nonnegative')
        predict_h2(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()


