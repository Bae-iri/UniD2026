"""Regenerate the group-isolated backbone inputs consumed by validate.py.

This is optional when the audited existing prediction artifacts are available.
Uses the repository's unchanged ABC B architecture and current observations only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from abc_reference import abc_fusion_experiment as abc
from reproduce import sha


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def prepare(args):
    train = pd.read_csv(args.data/'train/train.csv')
    if args.suite == 'train78':
        frame = train.copy()
        frame['_official_split'] = 'train'
        plans = [(f'outer_{k}', a, b) for k, (a, b) in enumerate(GroupKFold(5).split(frame, groups=frame.experiment_id))]
        root = args.output
        frame[['sample_id', 'experiment_id']].to_csv(root/'row_manifest.csv', index=False)
    else:
        val = pd.read_csv(args.data/'validation/validation.csv')
        assert not set(train.experiment_id) & set(val.experiment_id)
        pool = pd.concat([train.assign(_official_split='train'), val.assign(_official_split='validation')], ignore_index=True)
        a, b = next(GroupShuffleSplit(1, test_size=20, random_state=2026111501).split(pool, groups=pool.experiment_id))
        frame = pool.iloc[a].reset_index(drop=True)
        frame.to_csv(args.output/'development.csv', index=False)
        # The previously spent final confirmation groups are excluded entirely.
        write(args.output/'excluded_confirmation_experiments.json', sorted(set(pool.experiment_id.iloc[b])))
        plans = []
        for repeat, seed in enumerate([2026111511, 2026111522]):
            for k, (a, b) in enumerate(GroupKFold(5, shuffle=True, random_state=seed).split(frame, groups=frame.experiment_id)):
                plans.append((f'repeat_{repeat}_fold_{k}', a, b))
        root = args.output/'backbones'
    jobs = []
    for index, (scope, training, held) in enumerate(plans):
        folder = root/scope
        folder.mkdir(parents=True, exist_ok=True)
        splitter = GroupKFold(3) if args.suite == 'train78' else GroupKFold(3, shuffle=True, random_state=2026111600+index)
        parts = [training[b] for _, b in splitter.split(training, groups=frame.experiment_id.to_numpy()[training])]
        np.savez(folder/'partitions.npz', training=training, held=held, **{f'inner_{j}': p for j, p in enumerate(parts)})
        for j in range(4):
            allowed = np.setdiff1d(training, parts[j]) if j < 3 else training
            split_seed = (20261115+100*(index*7+(2*j+1 if j < 3 else 6)) if args.suite == 'train78'
                          else 2026111700+index*4+j)
            jobs.append(dict(folder=folder/(f'leave_{j}' if j < 3 else 'full'), allowed=allowed,
                             held=held, split_seed=split_seed, selection_fraction=.15 if args.suite == 'train78' else .2))
    return frame, jobs


def train_one(frame, images, plan, seed, epochs, device):
    folder = plan['folder']/f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    allowed = plan['allowed']
    a, b = next(GroupShuffleSplit(1, test_size=plan['selection_fraction'], random_state=plan['split_seed']).split(
        allowed, groups=frame.experiment_id.to_numpy()[allowed]))
    fit, selection = allowed[a], allowed[b]
    fit_exps, select_exps = set(frame.experiment_id.iloc[fit]), set(frame.experiment_id.iloc[selection])
    excluded = set(frame.experiment_id.iloc[plan['held']])
    assert not fit_exps & select_exps and not (fit_exps | select_exps) & excluded
    prep = abc.SensorPreprocessor(frame.iloc[fit])
    q, gas = prep.transform(frame)
    data = abc.Data(frame, images, torch.from_numpy(q).to(device), torch.from_numpy(gas).to(device),
                    torch.tensor(frame.label.to_numpy(int)-1, device=device))
    abc.common.seed_everything(seed)
    model = abc.FusionModel('B').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(prep.weights, device=device))
    generator = torch.Generator().manual_seed(seed)
    best, state, best_epoch = -1., None, -1
    history = []
    start = time.monotonic()
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.as_tensor(fit)[torch.randperm(len(fit), generator=generator)]
        total = 0.
        for offset in range(0, len(order), 64):
            ids = order[offset:offset+64]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*data.batch(ids, device)), data.y.index_select(0, ids.to(device)))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += float(loss.detach())*len(ids)
        model.eval()
        probs = []
        with torch.inference_mode():
            for offset in range(0, len(selection), 128):
                ids = torch.as_tensor(selection[offset:offset+128])
                probs.append(model(*data.batch(ids, device)).softmax(-1).cpu().numpy())
        score = float(f1_score(frame.label.iloc[selection], np.concatenate(probs).argmax(1)+1,
                               labels=range(1, 7), average='macro', zero_division=0))
        if score > best:
            best, best_epoch = score, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch, fit_ce=total/len(fit), selection_macro_f1=score))
    model.load_state_dict(state)
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for offset in range(0, len(frame), 128):
            ids = torch.arange(offset, min(offset+128, len(frame)))
            batch = data.batch(ids, device)
            size = len(ids)
            batch = tuple(torch.cat([v, torch.zeros((128-size, *v.shape[1:]), dtype=v.dtype, device=v.device)]) if size < 128 else v for v in batch)
            probabilities.append(model(*batch).softmax(-1)[:size].cpu().numpy())
    np.save(folder/'probabilities.npy', np.concatenate(probabilities))
    torch.save(dict(state=state, epoch=best_epoch, seed=seed, mode='B'), folder/'best.pt')
    pd.DataFrame(history).to_csv(folder/'history.csv', index=False)
    write(folder/'preprocessing.json', prep.metadata())
    manifest = frame[['sample_id', 'experiment_id']].copy()
    manifest['role'] = 'excluded'
    manifest.loc[fit, 'role'], manifest.loc[selection, 'role'] = 'fit', 'selection'
    manifest.to_csv(folder/'split_manifest.csv', index=False)
    write(folder/'complete.json', dict(seed=seed, best_epoch=best_epoch, epochs=epochs,
        fit_experiments=sorted(fit_exps), selection_experiments=sorted(select_exps),
        checkpoint_sha256=sha(folder/'best.pt'), probability_sha256=sha(folder/'probabilities.npy'),
        preprocessing_sha256=sha(folder/'preprocessing.json'), split_manifest_sha256=sha(folder/'split_manifest.csv'),
        selection_macro_f1=best, seconds=time.monotonic()-start, fixed_inference_batch_size=128))
    del data, model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(folder, round(best, 5), flush=True)


def main(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    frame, jobs = prepare(args)
    write(args.output/'regeneration_protocol.json', dict(suite=args.suite, epochs=args.epochs,
        seeds=[42,43,44], scopes=len(jobs)//4, source_sha256=sha(Path(__file__)),
        frozen_structure=True, checkpoint_selection='experiment-disjoint internal split',
        numeric_limit='GPU and library versions can change floating point values; labels and metrics must be rechecked.'))
    paths = [args.data/s/str(p).replace('\\', '/') for s,p in zip(frame['_official_split'], frame.image_path)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        packed = np.stack(list(executor.map(lambda p: abc.common.read_image(p,128,4)[0], paths)))
    device = torch.device(args.device)
    images = torch.from_numpy(packed).to(device)
    for plan in jobs:
        for seed in [42,43,44]:
            train_one(frame, images, plan, seed, args.epochs, device)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--suite', choices=['train78','repeat77'], required=True)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    main(ap.parse_args())
