"""B vs B+gate: current-observation cross-attention with optional scalar gates.
Standalone; no external weights or historical inputs. Synthetic smoke test only.
"""
import argparse
import csv
import json
import random
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


class Observations(Dataset):
    def __init__(self, frame, image_root, prep, image_size):
        if not frame.sample_id.is_unique:
            raise ValueError('sample_id must be unique')
        self.frame = frame.reset_index(drop=True)
        self.root, self.size = Path(image_root), image_size
        self.arrays = {k: torch.from_numpy(v) for k, v in transform(frame, prep).items()}
        self.y = None
        if 'label' in frame:
            y = pd.to_numeric(frame.label, errors='raise').to_numpy(float)
            if not np.isin(y, np.arange(1, 7)).all():
                raise ValueError('labels must be integers 1..6')
            self.y = torch.tensor(y.astype(np.int64) - 1)
        self.paths = []
        for rel in frame.image_path.astype(str):
            rel = Path(rel)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('image_path must be a relative path within its split')
            path = self.root / rel
            if not path.is_file():
                raise FileNotFoundError('Image missing under image_root; check split path')
            self.paths.append(path)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        batch = {k: v[i] for k, v in self.arrays.items()}
        with Image.open(self.paths[i]) as src:
            im = src.convert('RGB')
            im.thumbnail((self.size, self.size), Image.Resampling.BILINEAR)
            canvas = Image.new('RGB', (self.size, self.size))
            x, y = (self.size-im.width)//2, (self.size-im.height)//2
            canvas.paste(im, (x, y))
            mask = np.zeros((self.size, self.size), dtype=np.float32)
            mask[y:y+im.height, x:x+im.width] = 1
            arr = np.asarray(canvas, dtype=np.float32).copy() / 255.0
        batch['image'] = torch.from_numpy(arr.transpose(2, 0, 1))
        batch['image_mask'] = torch.from_numpy(mask[None])
        batch['target'] = self.y[i] if self.y is not None else torch.tensor(-1)
        return batch


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


class BGateModel(H1Model):
    def __init__(self, mode, image_size=96, dim=64):
        if mode not in ('B', 'B_gate'):
            raise ValueError('mode must be B or B_gate')
        super().__init__('late', image_size, dim)
        self.comparison_mode = mode
        # Created in both arms AFTER shared modules, preserving shared initialization.
        self.gate = nn.Linear(dim, 2)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, batch):
        static, electric, gas, image, image_global, valid = self.encode(batch)
        g, v = self.select(electric, static, gas, image, valid)
        if self.comparison_mode == 'B_gate':
            weights = torch.sigmoid(self.gate(electric))
            g = g * weights[:, :1]
            v = v * weights[:, 1:]
        return self.head(torch.cat([static, electric, gas.mean(1), image_global, g, v], -1))


def metrics(y, pred):
    cm = np.zeros((6, 6), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, int), np.asarray(pred, int)), 1)
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2*tp, den, out=np.zeros(6, float), where=den != 0)
    return {'macro_f1': float(f1.mean()), 'class_f1': f1.tolist(), 'confusion': cm.tolist()}


def loader(dataset, args, training=False, seed=0):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                      num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed),
                      persistent_workers=args.workers > 0)


def on_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    predictions, labels = [], []
    for batch in batches:
        batch = on_device(batch, device)
        predictions.extend(model(batch).argmax(1).cpu().tolist())
        labels.extend(batch['target'].cpu().tolist())
    return np.array(labels), np.array(predictions)


def read_frame(path):
    return pd.read_csv(path, dtype={'sample_id': 'string', 'experiment_id': 'string'})


def train(args):
    out = args.output
    folders = [out/f'{mode}_seed{seed}' for seed in args.seeds for mode in ('B', 'B_gate')]
    if any(p.exists() for p in folders):
        raise FileExistsError('Run folder exists; choose a NEW --output directory')
    out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (out/'config.json').write_text(json.dumps(config, indent=2))
    train_frame = read_frame(args.data_dir/'train/train.csv')
    val_frame = read_frame(args.data_dir/'validation/validation.csv')
    if (not set(train_frame.experiment_id).isdisjoint(val_frame.experiment_id)
            or not set(train_frame.sample_id).isdisjoint(val_frame.sample_id)):
        raise ValueError('Train and Validation sample/experiment IDs overlap')
    prep = fit_preprocess(train_frame)
    ds_train = Observations(train_frame, args.data_dir/'train', prep, args.image_size)
    ds_val = Observations(val_frame, args.data_dir/'validation', prep, args.image_size)
    device = torch.device(args.device)
    summary = []
    for seed in args.seeds:
        for mode in ('B', 'B_gate'):
            folder = out/f'{mode}_seed{seed}'
            folder.mkdir(exist_ok=True)
            checkpoint = folder/'best.pt'
            if checkpoint.exists():
                raise FileExistsError(f'{checkpoint}: use a new --output directory')
            seed_all(seed)
            model = BGateModel(mode, args.image_size, args.dim).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
            train_batches = loader(ds_train, args, True, seed)
            val_batches = loader(ds_val, args, False, seed)
            history, best, best_epoch = [], -1.0, 0
            print(f'START {mode} seed={seed} params={sum(p.numel() for p in model.parameters())}', flush=True)
            for epoch in range(1, args.epochs+1):
                model.train()
                total, count = 0.0, 0
                for batch in train_batches:
                    batch = on_device(batch, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(batch), batch['target'])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total += loss.item()*len(batch['target'])
                    count += len(batch['target'])
                y, pred = evaluate(model, val_batches, device)
                result = metrics(y, pred)
                score = result['macro_f1']
                history.append({'epoch': epoch, 'loss': total/count, **result})
                (folder/'history.json').write_text(json.dumps(history, indent=2))
                print(f'{mode} seed={seed} epoch={epoch}/{args.epochs} loss={total/count:.4f} val_macro_f1={score:.6f}', flush=True)
                if score > best:
                    best, best_epoch = score, epoch
                    torch.save({'state_dict': model.state_dict(), 'mode': mode, 'seed': seed,
                                'image_size': args.image_size, 'dim': args.dim,
                                'preprocess': prep, 'best_epoch': epoch,
                                'validation': result, 'features': NUMERIC,
                                'torch_version': str(torch.__version__), 'config': config}, checkpoint)
            (folder/'history.json').write_text(json.dumps(history, indent=2))
            summary.append({'mode': mode, 'seed': seed, 'best_epoch': best_epoch,
                            'val_macro_f1': best})
            pd.DataFrame(summary).to_csv(out/'summary.csv', index=False)
            del model, optimizer, train_batches, val_batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print('\nPAIRED COMPARISON (B_gate - B)', flush=True)
    diffs = []
    for seed in args.seeds:
        scores = {r['mode']: r['val_macro_f1'] for r in summary if r['seed'] == seed}
        delta = scores['B_gate'] - scores['B']
        diffs.append(delta)
        print(f'seed={seed}: {delta:+.6f}', flush=True)
    pd.DataFrame({'seed': args.seeds, 'B_gate-B': diffs}).to_csv(out/'paired_differences.csv', index=False)
    pd.DataFrame(summary).groupby('mode')['val_macro_f1'].agg(['mean', 'std', 'count']).to_csv(out/'aggregate.csv')
    print(f'mean_delta={np.mean(diffs):+.6f}; see summary.csv', flush=True)


def predict(args):
    # Only load checkpoints generated by this script or another trusted source.
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    frame = read_frame(args.input)
    ds = Observations(frame, args.image_root, ck['preprocess'], ck['image_size'])
    device = torch.device(args.device)
    model = BGateModel(ck['mode'], ck['image_size'], ck['dim']).to(device)
    model.load_state_dict(ck['state_dict'])
    _, pred = evaluate(model, loader(ds, args), device)
    series = pd.Series(pred+1, index=frame.sample_id.to_numpy())
    order = frame.sample_id if args.sample_submission is None else read_frame(args.sample_submission).sample_id
    if not order.is_unique or set(order) != set(series.index):
        raise ValueError('Submission template and input sample IDs do not match')
    if args.output.exists():
        raise FileExistsError('Output already exists; choose a new filename')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'sample_id': order.to_numpy(), 'label': series.loc[order].to_numpy()}).to_csv(args.output, index=False)
    print(f'Saved submission: {args.output}; rows={len(order)}')


def smoke_test():
    """Synthetic forward/backward, pairing, gate operation and checkpoint restore."""
    import tempfile
    torch.set_num_threads(1)
    seed_all(42)
    batch = {'electric': torch.randn(4, 2*len(ELECTRIC)),
             'gas': torch.randn(4, len(GAS), 2), 'capacity': torch.randn(4, 2),
             'cats': torch.ones(4, 3, dtype=torch.long), 'time': torch.randn(4, 2),
             'image': torch.rand(4, 3, 32, 32), 'image_mask': torch.ones(4, 1, 32, 32)}
    seed_all(42)
    baseline = BGateModel('B', 32, 16).eval()
    seed_all(42)
    gated = BGateModel('B_gate', 32, 16).eval()
    assert all(torch.equal(v, gated.state_dict()[k]) for k, v in baseline.state_dict().items())
    # Baseline remains exactly the existing late architecture with matching weights.
    original = H1Model('late', 32, 16).eval()
    original.load_state_dict({k: v for k, v in baseline.state_dict().items() if not k.startswith('gate.')})
    assert torch.allclose(original(batch), baseline(batch), atol=1e-6)
    for model in (baseline, gated):
        output = model(batch)
        assert output.shape == (4, 6) and torch.isfinite(output).all()
        nn.functional.cross_entropy(output, torch.arange(4)).backward()
        assert (model.gate.weight.grad is not None) == (model.comparison_mode == 'B_gate')
        if model.comparison_mode == 'B_gate':
            assert model.gate.weight.grad.abs().sum() > 0
        assert torch.allclose(output[:1], model({k: v[:1] for k, v in batch.items()}), atol=1e-5)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'weights.pt'
            torch.save(model.state_dict(), path)
            restored = BGateModel(model.comparison_mode, 32, 16).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            assert torch.allclose(output, restored(batch), atol=1e-6)
    # Near-one gates must recover B, proving that only attention branches are gated.
    with torch.no_grad():
        gated.gate.bias.fill_(30)
        assert torch.allclose(baseline(batch), gated(batch), atol=1e-6)
        gated.gate.bias.fill_(-30)
        assert not torch.allclose(baseline(batch), gated(batch))
    print('Synthetic tests passed: shared initialization, B equivalence, gate gradients, batching, restore.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true')
    subs = parser.add_subparsers(dest='command')
    tr = subs.add_parser('train')
    tr.add_argument('--data-dir', type=Path, required=True)
    tr.add_argument('--output', type=Path, default=Path('runs_b_gate_v1'))
    tr.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    tr.add_argument('--epochs', type=int, default=20)
    tr.add_argument('--image-size', type=int, default=96)
    tr.add_argument('--dim', type=int, default=64)
    tr.add_argument('--lr', type=float, default=3e-4)
    pr = subs.add_parser('predict')
    pr.add_argument('--input', type=Path, required=True)
    pr.add_argument('--image-root', type=Path, required=True)
    pr.add_argument('--checkpoint', type=Path, required=True)
    pr.add_argument('--sample-submission', type=Path)
    pr.add_argument('--output', type=Path, required=True)
    for p in (tr, pr):
        p.add_argument('--batch-size', type=int, default=128)
        p.add_argument('--workers', type=int, default=4)
        p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
    elif args.command == 'train':
        if (args.dim % 4 or args.dim < 4 or args.epochs < 1 or args.batch_size < 1
                or args.workers < 0 or args.image_size < 16 or args.image_size % 16
                or not np.isfinite(args.lr) or args.lr <= 0
                or len(set(args.seeds)) != len(args.seeds)):
            parser.error('dim must be a positive multiple of 4, epochs >= 1, seeds unique')
        train(args)
    elif args.command == 'predict':
        predict(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
