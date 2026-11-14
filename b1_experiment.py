"""Opt-in B1 warm-start training and inference. Importing never starts a run."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import abc_fusion_experiment as abc
import b1_history
import b1_model
from b1_history import history_features, HistoryNormalizer, HISTORY_STEPS
from b1_model import B1Model

BASE = Path(__file__).resolve().parent


class FrozenBPreprocessor(abc.SensorPreprocessor):
    """Restore B's input coordinate system AND loss weights, without refitting."""
    def __init__(self, metadata):
        if metadata['electro'] != abc.ELECTRO or metadata['gas'] != abc.GAS or metadata['query_dim'] != 12:
            raise ValueError('Preprocessing schema differs from the preserved B')
        self.median = pd.Series(metadata['median'], index=abc.NUMERIC, dtype=float)
        self.mean = pd.Series(metadata['mean'], index=abc.NUMERIC, dtype=float)
        self.std = pd.Series(metadata['std'], index=abc.NUMERIC, dtype=float)
        self.weights = np.asarray(metadata['class_weights'], np.float32)
        if not (np.isfinite(self.median).all() and np.isfinite(self.mean).all()
                and np.isfinite(self.std).all() and (self.std > 0).all()
                and self.weights.shape == (6,) and np.isfinite(self.weights).all() and (self.weights > 0).all()):
            raise ValueError('Invalid B preprocessing statistics')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frame(frame, labelled=False):
    required = ['sample_id', 'experiment_id', 'image_path', 'relative_time_s'] + abc.NUMERIC
    if not set(required).issubset(frame):
        raise ValueError(f'Missing columns: {set(required)-set(frame.columns)}')
    if frame.empty or frame.sample_id.isna().any() or not frame.sample_id.is_unique:
        raise ValueError('Require nonempty data and unique, nonmissing sample IDs')
    if labelled and ('label' not in frame or not frame.label.isin(range(1,7)).all()):
        raise ValueError('Require integer labels 1..6')
    # history_features also checks experiment IDs and finite timestamps.


def checkpoint_state(path, seed):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('mode') != 'B' or checkpoint.get('seed') != seed:
        raise ValueError(f'Expected B checkpoint for seed {seed}: {path}')
    return checkpoint['state']


@torch.no_grad()
def predict(model, data, normalized_history, device, baseline=False):
    model.eval()
    probabilities = []
    for start in range(0, len(data.frame), 128):
        ids = torch.arange(start, min(start+128, len(data.frame)))
        inputs = data.batch(ids, device)
        logits = model(*inputs) if baseline else model(*inputs, normalized_history[ids].to(device))
        probabilities.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(probabilities)


def train_one(seed, datasets, train_histories, eval_histories, prep, args, device):
    folder = args.output / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    completed = folder / 'metrics.json'
    if args.resume and completed.exists():
        return json.loads(completed.read_text())
    base_path = args.base_run / f'seed_{seed}' / 'B/best.pt'
    state = checkpoint_state(base_path, seed)
    abc.common.seed_everything(seed)
    model = B1Model().to(device)
    model.load_b_weights(state)
    # Preserve a byte-identical reference in the new run, never modify the original.
    (folder / 'B_reference.pt').write_bytes(base_path.read_bytes())
    initial_prob = predict(model, datasets['validation'], eval_histories['validation'], device)
    reference = abc.metrics(datasets['validation'].frame, initial_prob)
    abc.save_predictions(folder, 'B_reference_validation', datasets['validation'].frame, initial_prob)
    best = reference['macro_f1']
    best_epoch = 0

    def save_best(epoch, score):
        torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        model='B1', seed=seed, epoch=epoch, validation_macro_f1=score,
                        history_hidden=32, eval_history_step=args.eval_history_step), folder/'best.pt')

    # Epoch zero is eligible: validation must improve to replace the original B behavior.
    save_best(0, best)
    print(f'{seed} B_reference validation {best:.4f}', flush=True)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(prep.weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    train = datasets['train']
    order_rng = torch.Generator().manual_seed(seed)
    history_rng = torch.Generator().manual_seed(seed+17001)
    log = [dict(epoch=0, train_ce=None, validation_macro_f1=best)]
    started = time.monotonic()
    for epoch in range(1, args.epochs+1):
        model.train()
        # EVERY target appears once. Only its past-history view changes.
        order = torch.randperm(len(train.frame), generator=order_rng)
        view_for_target = torch.randint(len(HISTORY_STEPS), (len(train.frame),), generator=history_rng)
        counts = torch.bincount(view_for_target, minlength=3).tolist()
        total, count = 0.0, 0
        for start in range(0, len(order), 64):
            ids = order[start:start+64]
            history = train_histories[view_for_target[ids], ids].to(device)
            y = train.y.index_select(0, ids.to(train.y.device)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*train.batch(ids, device), history), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss: {seed=} {epoch=}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()*len(ids)
            count += len(ids)
        if count != len(train.frame):
            raise RuntimeError('Target rows were lost during history augmentation')
        probability = predict(model, datasets['validation'], eval_histories['validation'], device)
        score = abc.metrics(datasets['validation'].frame, probability)['macro_f1']
        if score > best:
            best, best_epoch = score, epoch
            save_best(epoch, score)
        log.append(dict(epoch=epoch, train_ce=total/count, validation_macro_f1=score,
                        targets=count, history_5s=counts[0], history_10s=counts[1], history_15s=counts[2]))
        pd.DataFrame(log).to_csv(folder/'history.csv', index=False)
        print(f'{seed} B1 {epoch:02}/{args.epochs} CE {total/count:.4f} validation {score:.4f} '
              f'best {best:.4f} epoch {best_epoch} targets {count}', flush=True)
    saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['state'])
    for split in ['validation', 'test']:
        probability = predict(model, datasets[split], eval_histories[split], device)
        abc.save_predictions(folder, split, datasets[split].frame, probability)
    final = abc.metrics(datasets['validation'].frame, predict(
        model, datasets['validation'], eval_histories['validation'], device))
    final.update(seed=seed, model='B1', best_epoch=best_epoch, B_reference_macro_f1=reference['macro_f1'],
                 delta_from_B_reference=final['macro_f1']-reference['macro_f1'],
                 minutes=(time.monotonic()-started)/60)
    abc.common.write_json(completed, final)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return final


def summarize(results, frames, args):
    out = args.output
    summary = pd.DataFrame(results)
    summary.to_csv(out/'summary.csv', index=False)
    columns = ['macro_f1'] + [f'F1_{i}' for i in range(1,7)]
    aggregate = pd.DataFrame([dict(model='B1', **{c:summary[c].mean() for c in columns},
                                   seed_std=float(summary.macro_f1.std()) if len(summary)>1 else 0.0)])
    aggregate.to_csv(out/'aggregate.csv', index=False)
    ensemble_metrics, tidy = [], []
    for row in results:
        for metric in columns:
            tidy.append(dict(row_type='run', seed=row['seed'], model='B1', metric=metric, value=row[metric]))
    for column in columns:
        tidy.append(dict(row_type='aggregate', model='B1', metric=column, value=summary[column].mean()))
    for split in ['validation', 'test']:
        probabilities = []
        for seed in args.seeds:
            frame = pd.read_csv(out/f'seed_{seed}'/f'{split}_predictions.csv')
            if frame.sample_id.tolist() != frames[split].sample_id.tolist():
                raise ValueError('Ensemble prediction row mismatch')
            probabilities.append(frame[[f'p{i}' for i in range(1,7)]].to_numpy())
        probability = np.mean(probabilities, axis=0)
        abc.save_predictions(out, f'ensemble_B1_{split}', frames[split], probability)
        if split == 'validation':
            ensemble_metrics.append(dict(model='B1', split=split, **abc.metrics(frames[split], probability)))
            abc.save_predictions(out, 'validation', frames[split], probability)
        else:
            sample = pd.read_csv(args.data_root/'sample_submission.csv')
            if list(sample.columns) != ['sample_id','label'] or not sample.sample_id.is_unique:
                raise ValueError('Invalid submission template')
            submission = abc.make_submission(sample, frames[split], probability)
            submission.to_csv(out/'submission_B1.csv', index=False)
    pd.DataFrame(ensemble_metrics).to_csv(out/'ensemble_scores.csv', index=False)
    pd.DataFrame(tidy).to_csv(out/'results_combined.csv', index=False)
    abc.common.write_json(out/'submission_info.json', dict(model='B1', seeds=args.seeds,
        filename='submission_B1.csv', rows=len(frames['test']), selection='per-seed best Validation Macro F1, including epoch zero',
        eval_history_step=args.eval_history_step, original_B_untouched=True))
    lines = ['# B1: causal 30-second history residual', '',
        'B is preserved; B1 starts from each existing B checkpoint. Static variables excluded; current CNN and gas/image cross-attention unchanged.',
        'q = original_q + history_MLP(history). The last linear layer starts at zero. Query direct path and whole-modality summary skips remain.',
        'Train target rows are all kept. Each target independently receives a 5/10/15-second past-history view per epoch.',
        'Slopes use finite raw measurements before any imputation. History normalization uses only official Train, pooled over the three views.',
        'Current-observation normalization and weighted CE class weights are restored from the original B run, not refitted.',
        'Checkpoint selection uses provided Validation Macro F1, including the initial B-equivalent epoch zero. Validation is a selection set, not an independent final test.',
        'This is warm-start fine-tuning on full Train; changes relative to old B also include additional optimization/data exposure, not only history features.', '',
        abc.markdown_table(summary.round(4)), '', '## Seed-average F1', '', abc.markdown_table(aggregate.round(4)), '',
        '## Probability ensemble', '', abc.markdown_table(pd.DataFrame(ensemble_metrics).round(4)), '',
        'submission_B1.csv uses the official sample_id order and label 1..6. Original B code, checkpoints and submissions are not overwritten.', '']
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(aggregate.round(4).to_string(index=False), flush=True)
    print(f'Submission: {out/"submission_B1.csv"}', flush=True)


def train(args, device):
    frames = {s:pd.read_csv(args.data_root/s/f'{s}.csv').reset_index(drop=True)
              for s in ['train','validation','test']}
    for name, frame in frames.items():
        validate_frame(frame, labelled=name!='test')
    if any(set(frames[a].experiment_id)&set(frames[b].experiment_id)
           or set(frames[a].sample_id)&set(frames[b].sample_id)
           for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Official splits overlap')
    base_config = json.loads((args.base_run/'config.json').read_text())
    if base_config['source_sha256'] != sha(abc.__file__) or base_config['common_sha256'] != sha(abc.common.__file__):
        raise ValueError('Preserved B model source differs from the imported B implementation')
    sources = {'source.py':Path(__file__), 'b1_model.py':Path(b1_model.__file__),
               'b1_history.py':Path(b1_history.__file__), 'abc_source.py':Path(abc.__file__),
               'common_source.py':Path(abc.common.__file__)}
    config = dict(model='B1', seeds=args.seeds, epochs=args.epochs, base_run=str(args.base_run.resolve()),
        data_root=str(args.data_root.resolve()), eval_history_step=args.eval_history_step,
        source_sha256={name:sha(path) for name,path in sources.items()},
        csv_sha256={s:sha(args.data_root/s/f'{s}.csv') for s in frames},
        preprocessing_sha256=sha(args.base_run/'preprocessing.json'),
        template_sha256=sha(args.data_root/'sample_submission.csv'),
        B_checkpoint_sha256={str(seed):sha(args.base_run/f'seed_{seed}/B/best.pt') for seed in args.seeds},
        loss='original B weighted CE, label_smoothing=0', batch_size=64, lr=3e-4,
        weight_decay=1e-4, history_hidden=32, checkpoint_metric='provided Validation Macro F1',
        history_sampling='independent per target each epoch; all targets retained',
        device=str(device), torch_version=str(torch.__version__))
    if config['csv_sha256'] != base_config['csv_sha256']:
        raise ValueError('Data differ from preserved B run')
    path = args.output/'config.json'
    if path.exists() and (not args.resume or json.loads(path.read_text()) != config):
        raise ValueError('Output exists or resume configuration changed; choose a new output folder')
    args.output.mkdir(parents=True, exist_ok=True)
    abc.common.write_json(path, config)
    for name, source in sources.items():
        (args.output/name).write_bytes(source.read_bytes())
    metadata = json.loads((args.base_run/'preprocessing.json').read_text())
    prep = FrozenBPreprocessor(metadata)
    abc.common.write_json(args.output/'preprocessing.json', metadata)
    print('Generating causal features from raw observations (no sensor imputation)', flush=True)
    raw_train = [history_features(frames['train'], step) for step in HISTORY_STEPS]
    normalizer = HistoryNormalizer.fit(raw_train)
    abc.common.write_json(args.output/'history_normalization.json', normalizer.metadata())
    train_histories = torch.from_numpy(np.stack([normalizer.transform(raw) for raw in raw_train]))
    del raw_train
    eval_histories = {}
    for split in ['validation','test']:
        eval_histories[split] = torch.from_numpy(normalizer.transform(history_features(frames[split], args.eval_history_step)))
    pd.concat([f[['sample_id','experiment_id']].assign(split=s) for s,f in frames.items()]).to_csv(
        args.output/'split_manifest.csv', index=False)
    datasets = abc.prepare(args, frames, prep, device)
    results = []
    for seed in args.seeds:
        results.append(train_one(seed, datasets, train_histories, eval_histories, prep, args, device))
        pd.DataFrame(results).to_csv(args.output/'summary.csv', index=False)
    summarize(results, frames, args)


def infer(args, device):
    """No Train access, no normalization fit, no optimizer in prediction-only mode."""
    config = json.loads((args.output/'config.json').read_text())
    step = config['eval_history_step']
    frame = pd.read_csv(args.input_csv or args.data_root/'test/test.csv').reset_index(drop=True)
    validate_frame(frame)
    prep = FrozenBPreprocessor(json.loads((args.output/'preprocessing.json').read_text()))
    stored = json.loads((args.output/'history_normalization.json').read_text())
    normalizer = HistoryNormalizer(stored['mean'], stored['std'])
    history = torch.from_numpy(normalizer.transform(history_features(frame, step)))
    image_root = args.image_root or args.data_root/'test'
    images = abc.common.cache_images(image_root, frame, '.', BASE/'.hierarchy_cache', 128, 4, 8)[0]
    q, gas = prep.transform(frame)
    data = abc.Data(frame, torch.from_numpy(images), torch.from_numpy(q), torch.from_numpy(gas), None)
    probabilities = []
    for seed in config['seeds']:
        checkpoint = torch.load(args.output/f'seed_{seed}/best.pt', map_location='cpu', weights_only=True)
        if checkpoint['model'] != 'B1' or checkpoint['seed'] != seed:
            raise ValueError('Unexpected inference checkpoint')
        model = B1Model(history_hidden=checkpoint['history_hidden']).to(device)
        model.load_state_dict(checkpoint['state'], strict=True)
        probabilities.append(predict(model, data, history, device))
        del model
    probability = np.mean(probabilities, axis=0)
    destination = args.prediction_output or args.output/'inference'
    destination.mkdir(parents=True, exist_ok=True)
    abc.save_predictions(destination, 'B1', frame, probability)
    sample = pd.read_csv(args.sample_submission or args.data_root/'sample_submission.csv')
    if list(sample.columns) != ['sample_id','label'] or not sample.sample_id.is_unique:
        raise ValueError('Invalid submission template')
    abc.make_submission(sample, frame, probability).to_csv(destination/'submission_B1.csv', index=False)
    print(f'Inference submission: {destination/"submission_B1.csv"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=BASE.parent/'data/battery_data_light')
    p.add_argument('--base-run', type=Path, default=BASE/'abc_runs/current_query_v1')
    p.add_argument('--output', type=Path, default=BASE/'b1_runs/history30_v1')
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--eval-history-step', type=int, choices=HISTORY_STEPS, default=5)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--predict-only', action='store_true')
    p.add_argument('--input-csv', type=Path)
    p.add_argument('--image-root', type=Path)
    p.add_argument('--sample-submission', type=Path)
    p.add_argument('--prediction-output', type=Path)
    args = p.parse_args()
    if args.epochs<1 or len(set(args.seeds)) != len(args.seeds):
        p.error('Require positive epochs and distinct seeds')
    if args.output.resolve() == args.base_run.resolve() or args.base_run.resolve() in args.output.resolve().parents:
        p.error('B1 output must be outside the preserved B run')
    torch.set_num_threads(4)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on GPU host or explicitly use --device cpu')
    device = torch.device(args.device)
    if args.predict_only:
        infer(args, device)
    else:
        if any([args.input_csv, args.image_root, args.sample_submission, args.prediction_output]):
            p.error('Custom inference paths require --predict-only')
        train(args, device)


if __name__ == '__main__':
    main()
