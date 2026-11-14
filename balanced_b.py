"""Original B fine-tuning with stage/experiment balanced sampling.

No extra expert, static conditions, temporal rules, or validation training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from abc_reference import abc_fusion_experiment as abc


def sampling_weights(frame, arm):
    if not len(frame) or set(frame.label) != set(range(1, 7)):
        raise ValueError('Fit must contain all six stages')
    if frame.experiment_id.isna().any():
        raise ValueError('Missing experiment ID')
    if arm == 'control':
        weights = np.ones(len(frame), dtype=float)
    elif arm == 'stage_balanced':
        weights = 1 / frame.groupby('label').label.transform('size').to_numpy(float)
    elif arm == 'experiment_balanced':
        size = frame.groupby(['label', 'experiment_id']).label.transform('size').to_numpy(float)
        experiments = frame[['label', 'experiment_id']].drop_duplicates().groupby('label').size()
        weights = 1 / (size * frame.label.map(experiments).to_numpy(float))
    else:
        raise ValueError(arm)
    return weights / weights.sum()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_data(frame, packed, prep):
    q, gas = prep.transform(frame)
    return abc.Data(frame, torch.from_numpy(packed), torch.from_numpy(q), torch.from_numpy(gas),
                    torch.tensor(frame.label.to_numpy(int) - 1) if 'label' in frame else None)


def fit_one(arm, seed, data, prep, args, device):
    abc.common.seed_everything(seed)
    folder = args.output / f'seed_{seed}' / arm
    folder.mkdir(parents=True)
    source = args.base_run / f'seed_{seed}/B/best.pt'
    ck = torch.load(source, map_location='cpu', weights_only=True)
    model = abc.FusionModel('B').to(device)
    model.load_state_dict(ck['state'])
    baseline = abc.predict(model, data['dev'], None, device)
    best = abc.metrics(data['dev'].frame, baseline)['macro_f1']
    checkpoint = dict(state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                      epoch=0, dev_macro_f1=best, seed=seed, arm=arm,
                      preprocessing=prep.metadata(), source_sha256=sha(source))
    torch.save(checkpoint, folder / 'best.pt')
    # Reproduce the original checkpoint before any update; smoke uses a subset.
    if not args.smoke:
        ref = pd.read_csv(args.base_run / f'seed_{seed}/B/dev_predictions.csv')
        if not np.array_equal(ref.sample_id, data['dev'].frame.sample_id):
            raise ValueError('Original dev row identity mismatch')
        if not np.array_equal(ref.pred, baseline.argmax(1) + 1):
            raise ValueError('Original B predictions not reproduced')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    fit = data['fit']
    weights = torch.from_numpy(sampling_weights(fit.frame, arm))
    pd.DataFrame(dict(sample_id=fit.frame.sample_id, probability=weights.numpy())).to_csv(
        folder / 'sampling_probabilities.csv', index=False)
    class_weights = torch.as_tensor(prep.weights, device=device) if arm == 'control' else None
    generator = torch.Generator().manual_seed(seed)
    history = [dict(epoch=0, train_loss=None, dev_macro_f1=best, best_epoch=0)]
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        # Balanced sampling already gives uniform stage mass; do not double-weight the loss.
        order = (torch.randperm(len(fit.frame), generator=generator) if arm == 'control' else
                 torch.multinomial(weights, len(fit.frame), replacement=True, generator=generator))
        model.train()
        total = 0.
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            opt.zero_grad(set_to_none=True)
            logits = model(*fit.batch(ids, device))
            loss = F.cross_entropy(logits, fit.y[ids].to(device), weight=class_weights)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            total += loss.item() * len(ids)
        score = abc.metrics(data['dev'].frame, abc.predict(model, data['dev'], None, device))['macro_f1']
        current = dict(state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       epoch=epoch, dev_macro_f1=score, seed=seed, arm=arm,
                       preprocessing=prep.metadata(), source_sha256=checkpoint['source_sha256'])
        torch.save(current, folder / 'last.pt')
        if score > best:
            best, best_epoch = score, epoch
            torch.save(current, folder / 'best.pt')
        history.append(dict(epoch=epoch, train_loss=total / len(order),
                            dev_macro_f1=score, best_epoch=best_epoch))
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        print(f'{arm} seed={seed} epoch={epoch} dev={score:.6f} best={best:.6f} '
              f'best_epoch={best_epoch} seconds={time.monotonic()-started:.1f}', flush=True)
    model.load_state_dict(torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)['state'])
    probs = {}
    metrics = dict(arm=arm, seed=seed, best_epoch=best_epoch, baseline_dev_f1=history[0]['dev_macro_f1'])
    for split in data:
        if split == 'fit':
            continue
        probs[split] = abc.predict(model, data[split], None, device)
        abc.save_predictions(folder, split, data[split].frame, probs[split])
        if split != 'test':
            metrics[split] = abc.metrics(data[split].frame, probs[split])
    write_json(folder / 'metrics.json', metrics)
    return metrics, probs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=Path('/home/ubuntu/data/battery_data_light'))
    p.add_argument('--base-run', type=Path, default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--arms', nargs='+', choices=['control', 'stage_balanced', 'experiment_balanced'],
                   default=['control', 'stage_balanced', 'experiment_balanced'])
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--predict-test', action='store_true')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable in this process. Use the GPU-enabled environment or --device cpu.')
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        p.error('epochs, batch-size and lr must be positive')
    if args.smoke:
        args.seeds, args.epochs = args.seeds[:1], 1
    torch.set_num_threads(args.threads)
    config = json.loads((args.base_run / 'config.json').read_text())
    frames = abc.common.split_frames(args.data_root, config['split_seed'], .2)
    prep = abc.SensorPreprocessor(frames['fit'])
    expected = json.loads((args.base_run / 'preprocessing.json').read_text())
    for key in ['median', 'mean', 'std']:
        np.testing.assert_allclose(prep.metadata()[key], expected[key], rtol=1e-12, atol=1e-12)
    if args.predict_test:
        frames['test'] = pd.read_csv(args.data_root / 'test/test.csv')
        assert not set(frames['test'].experiment_id) & set(pd.concat(list(frames.values())[:-1]).experiment_id)
    split_manifest = pd.read_csv(args.base_run / 'split_manifest.csv')
    image_manifest = json.loads((args.base_run / 'image_cache_manifest.json').read_text())
    data = {}
    for split, frame in frames.items():
        if not np.array_equal(frame.sample_id, split_manifest.loc[split_manifest.split == split, 'sample_id']):
            raise ValueError(f'Split/cache row mismatch: {split}')
        ids = np.arange(len(frame))
        if args.smoke:
            ids = (np.concatenate([np.flatnonzero(frame.label.to_numpy() == label)[:12] for label in range(1, 7)])
                   if 'label' in frame else ids[:72])
        images = np.load(image_manifest[split], mmap_mode='r')
        assert images.shape == (len(frame), 4, 128, 128)
        data[split] = make_data(frame.iloc[ids].reset_index(drop=True), np.array(images[ids], copy=True), prep)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'config.json', {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                            'source_sha256': sha(Path(__file__)), 'split_seed': config['split_seed'],
                                            'selection': 'dev macro F1, including epoch 0',
                                            'validation_used_for_training': False})
    write_json(args.output / 'preprocessing.json', prep.metadata())
    pd.concat([d.frame[['sample_id', 'experiment_id']].assign(split=s) for s, d in data.items()]).to_csv(
        args.output / 'split_manifest.csv', index=False)
    rows = []
    for arm in args.arms:
        ensemble = {}
        for seed in args.seeds:
            metrics, probs = fit_one(arm, seed, data, prep, args, torch.device(args.device))
            rows.append(metrics)
            write_json(args.output / 'runs.json', rows)
            for split, prob in probs.items():
                ensemble[split] = ensemble.get(split, 0) + prob / len(args.seeds)
        folder = args.output / f'ensemble_{arm}'
        folder.mkdir()
        scores = {}
        for split, prob in ensemble.items():
            abc.save_predictions(folder, split, data[split].frame, prob)
            if split != 'test':
                scores[split] = abc.metrics(data[split].frame, prob)
            elif not args.smoke:
                template = pd.read_csv(args.data_root / 'sample_submission.csv')[['sample_id']]
                pred = pd.Series(prob.argmax(1) + 1, index=data[split].frame.sample_id)
                template['label'] = template.sample_id.map(pred)
                assert len(template) == 3451 and template.label.notna().all() and template.sample_id.is_unique
                assert template.label.between(1, 6).all()
                template.label = template.label.astype(int)
                template.to_csv(folder / 'submission.csv', index=False)
        write_json(folder / 'metrics.json', scores)
        print(f'ENSEMBLE {arm}: {json.dumps(scores)}', flush=True)


if __name__ == '__main__':
    main()
