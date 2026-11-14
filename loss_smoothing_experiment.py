"""B architecture: weighted CE vs weighted label-smoothed CE. Training is opt-in."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import confusion_matrix

import abc_fusion_experiment as abc

BASE = Path(__file__).resolve().parent
LOSSES = {'ce': 0.0, 'ls005': 0.05, 'ls010': 0.10}


def train_one(name, seed, initial, datasets, prep, args, device):
    epsilon = LOSSES[name]
    folder = args.output / f'seed_{seed}' / name
    folder.mkdir(parents=True, exist_ok=True)
    if args.resume and (folder / 'metrics.json').exists():
        return json.loads((folder / 'metrics.json').read_text())
    abc.common.seed_everything(seed)
    model = abc.FusionModel('B').to(device)
    model.load_state_dict(initial)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(prep.weights, device=device), label_smoothing=epsilon)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    fit = datasets['fit']
    generator = torch.Generator().manual_seed(seed)
    best, history = -1.0, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        order = torch.randperm(len(fit.frame), generator=generator)
        for start in range(0, len(order), 64):
            ids = order[start:start+64]
            y = fit.y.index_select(0, ids.to(fit.y.device)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(*fit.batch(ids, device)), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite loss: {seed=} {name=} {epoch=}')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(ids)
            count += len(ids)
        dev = abc.metrics(datasets['dev'].frame, abc.predict(model, datasets['dev'], args, device))['macro_f1']
        if dev > best:
            best = dev
            torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            epoch=epoch, dev_macro_f1=dev, model='B', loss=name,
                            label_smoothing=epsilon, seed=seed), folder / 'best.pt')
        history.append(dict(epoch=epoch, train_loss=total/count, dev_macro_f1=dev))
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        print(f'{seed} B {name} eps={epsilon:g} {epoch:02}/{args.epochs} '
              f'loss {total/count:.4f} dev {dev:.4f} best {best:.4f}', flush=True)
    checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state'])
    scores = {}
    for split in ['dev', 'validation', 'test']:
        prob = abc.predict(model, datasets[split], args, device)
        abc.save_predictions(folder, split, datasets[split].frame, prob)
        if split == 'validation':
            scores = abc.metrics(datasets[split].frame, prob)
    scores.update(model='B', loss=name, label_smoothing=epsilon, seed=seed,
                  best_dev_epoch=checkpoint['epoch'], best_dev_f1=best,
                  minutes=(time.monotonic()-started)/60)
    abc.common.write_json(folder / 'metrics.json', scores)
    print(f'FINAL {seed} B {name} validation {scores["macro_f1"]:.4f}', flush=True)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return scores


def summarize(results, frames, args):
    out = args.output
    summary = pd.DataFrame(results)
    summary.to_csv(out / 'summary.csv', index=False)
    columns = ['macro_f1'] + [f'F1_{i}' for i in range(1,7)]
    grouped = summary.groupby('loss', sort=False)[columns]
    means, stds = grouped.mean(), grouped.std().fillna(0)
    means.add_suffix('_mean').join(stds.add_suffix('_std')).to_csv(out / 'aggregate.csv')
    confusion, ensemble_rows, tidy, matrices = {}, [], [], {}
    sample = pd.read_csv(args.data_root / 'sample_submission.csv')
    if args.smoke:
        sample = sample[sample.sample_id.isin(frames['test'].sample_id)].copy()
    if list(sample.columns) != ['sample_id', 'label'] or not sample.sample_id.is_unique:
        raise ValueError('Invalid official submission template')
    for name in args.losses:
        per_seed = []
        for seed in args.seeds:
            prediction = pd.read_csv(out / f'seed_{seed}' / name / 'validation_predictions.csv')
            if prediction.sample_id.tolist() != frames['validation'].sample_id.tolist():
                raise ValueError('Validation row mismatch')
            per_seed.append(np.stack([confusion_matrix(g.label, g.pred, labels=range(1,7))
                                      for _,g in prediction.groupby('experiment_id', sort=True)]))
        confusion[name] = np.stack(per_seed)
        mean_cm = confusion[name].sum(1).mean(0)
        matrices[name] = pd.DataFrame(mean_cm, index=range(1,7), columns=range(1,7))
        matrices[name].to_csv(out / f'confusion_{name}_mean.csv', index_label='true_label')
        for i in range(6):
            for j in range(6):
                tidy.append(dict(row_type='confusion_mean', loss=name, true_label=i+1,
                                 pred_label=j+1, metric='count', value=mean_cm[i,j]))
        for metric in columns:
            tidy.append(dict(row_type='aggregate', loss=name, metric=metric,
                             value=means.loc[name,metric], std=stds.loc[name,metric]))
        for split in ['dev', 'validation', 'test']:
            probabilities = []
            for seed in args.seeds:
                p = pd.read_csv(out / f'seed_{seed}' / name / f'{split}_predictions.csv')
                if p.sample_id.tolist() != frames[split].sample_id.tolist():
                    raise ValueError('Ensemble row mismatch')
                probabilities.append(p[[f'p{i}' for i in range(1,7)]].to_numpy())
            probability = np.mean(probabilities, axis=0)
            abc.save_predictions(out, f'ensemble_{name}_{split}', frames[split], probability)
            if split != 'test':
                ensemble_rows.append(dict(loss=name, label_smoothing=LOSSES[name], split=split,
                                          **abc.metrics(frames[split], probability)))
            else:
                submission = abc.make_submission(sample, frames['test'], probability)
                submission.to_csv(out / f'submission_{name}.csv', index=False)
    for row in results:
        for metric in columns:
            tidy.append(dict(row_type='run', loss=row['loss'], seed=row['seed'],
                             metric=metric, value=row[metric]))
    n_groups = confusion['ce'].shape[1]
    weights = np.random.default_rng(args.split_seed+971).multinomial(
        n_groups, np.full(n_groups, 1/n_groups), size=2000)
    boot = {name:abc.common.macro_from_confusions(np.einsum('be,semn->bsmn',weights,cms)).mean(1)
            for name,cms in confusion.items()}
    comparisons = []
    for name in args.losses:
        if name == 'ce':
            continue
        low, high = np.quantile(boot[name]-boot['ce'], [.025,.975])
        delta = float(means.loc[name,'macro_f1']-means.loc['ce','macro_f1'])
        comparisons.append(dict(candidate=name, reference='ce', delta=delta, ci_low=low, ci_high=high))
        tidy.append(dict(row_type='contrast', loss=name, reference='ce', metric='macro_f1_delta',
                         value=delta, ci_low=low, ci_high=high))
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out / 'comparisons.csv', index=False)
    seed_deltas = summary.pivot(index='seed', columns='loss', values='macro_f1')
    for name in args.losses:
        if name != 'ce':
            seed_deltas[f'{name}_minus_ce'] = seed_deltas[name]-seed_deltas.ce
    seed_deltas.to_csv(out / 'seed_deltas.csv')
    pd.DataFrame(tidy).to_csv(out / 'results_combined.csv', index=False)
    ensemble = pd.DataFrame(ensemble_rows)
    ensemble.to_csv(out / 'ensemble_scores.csv', index=False)
    selected = ensemble[ensemble.split=='dev'].sort_values('macro_f1', ascending=False, kind='stable').iloc[0].loss
    # Each loss keeps its own explicit submission filename. No ambiguous submission.csv.
    (out / 'submission_best_dev.csv').write_bytes((out / f'submission_{selected}.csv').read_bytes())
    abc.common.write_json(out / 'submission_info.json', dict(model='B', selected_loss=selected,
        label_smoothing=LOSSES[selected], selection='highest dev ensemble macro-F1',
        best_dev_file='submission_best_dev.csv', per_loss_files=[f'submission_{n}.csv' for n in args.losses],
        seeds=args.seeds, rows=len(sample), columns=['sample_id','label'], smoke=args.smoke))
    table = means.round(4).reset_index()
    table.insert(1, 'label_smoothing', [LOSSES[n] for n in table.loss])
    table['seed_std'] = stds.macro_f1.round(4).values
    lines = ['# B 모델의 Cross Entropy / Label Smoothing 비교', '',
        f'Smoke={args.smoke}; seeds={args.seeds}; epochs={args.epochs}.', '',
        '모든 모델은 기존 B 구조. 클래스 가중치도 동일하고 CrossEntropyLoss의 label_smoothing만 변경.',
        'ce=0, ls005=0.05, ls010=0.1. 정적/시계열 특징 없음. 전기·열과 가스·이미지 전체 요약의 직접 연결 유지.',
        'fit에서만 전처리/클래스 가중치 계산. 같은 초기 state와 배치 순서로 처음부터 학습. dev로 checkpoint 선택, validation으로 최종 평가.', '',
        '## 검증 F1 (개별 seed 점수의 평균)', '', abc.markdown_table(table), '',
        '## 일반 CE 대비 차이', '', abc.markdown_table(comparison.round(4)), '',
        '95% 구간은 검증 experiment_id 단위 2,000회 paired bootstrap. 고정된 학습 결과에 대한 검증 표본 불확실성으로 seed 표준편차와 구분.',
        '차이가 양수여도 구간이 0을 포함하면 일관된 개선으로 확정하지 않음. 서로 다른 손실의 train_loss 크기만으로 성능을 비교하지 않음.', '',
        '## 혼동행렬 (seed 평균 count)', '행은 실제 label, 열은 예측 label. 앙상블 혼동행렬과 다름.', '']
    for name in args.losses:
        cm = matrices[name].round(1).reset_index().rename(columns={'index':'true / pred'})
        lines += [f'### {name}', '', abc.markdown_table(cm), '']
    lines += ['## 제출 파일', '',
        '모든 파일은 B 구조의 3-seed 확률 평균 예측이며, 손실별 파일명을 유지함.',
        '`submission_ce.csv`: 일반 CE. `submission_ls005.csv`: smoothing 0.05. `submission_ls010.csv`: smoothing 0.1.',
        f'`submission_best_dev.csv`는 dev 앙상블 점수로 선택한 **{selected}**의 복사본. 자동 선택 내역은 submission_info.json에 저장.',
        '기존 제출 파일은 변경하지 않음. 공식 sample_submission의 sample_id 순서와 label 1~6 정수 형식을 검사.',
        '외부 제출/업로드는 하지 않으며 test 점수는 정답이 없어 계산할 수 없음.', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(table.to_string(index=False), flush=True)
    print(comparison.to_string(index=False), flush=True)
    print(f'Best dev: {selected}; submission file: {out / "submission_best_dev.csv"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=BASE.parent/'data/battery_data_light')
    parser.add_argument('--output', type=Path, default=BASE/'loss_runs/b_smoothing_v1')
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    parser.add_argument('--losses', nargs='+', choices=list(LOSSES), default=list(LOSSES))
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--split-seed', type=int, default=20261114)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='User-invoked small end-to-end check; not research results')
    args = parser.parse_args()
    if 'ce' not in args.losses or len(args.losses)<2:
        parser.error('Include ce and at least one smoothing condition')
    if len(set(args.losses))!=len(args.losses) or len(set(args.seeds))!=len(args.seeds):
        parser.error('Duplicate losses or seeds')
    if args.epochs<1:
        parser.error('epochs must be positive')
    if args.smoke:
        args.seeds, args.epochs = [args.seeds[0]], 1
    torch.set_num_threads(4)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run on GPU host or explicitly set --device cpu')
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {'source.py':Path(__file__), 'abc_source.py':Path(abc.__file__),
               'common_source.py':Path(abc.common.__file__)}
    config = {key:str(value.resolve()) if isinstance(value,Path) else value
              for key,value in vars(args).items() if key not in ['output','resume']}
    config.update(model='B', smoothing_values={n:LOSSES[n] for n in args.losses},
                  batch_size=64, lr=3e-4, weight_decay=1e-4, loss='weighted_cross_entropy',
                  image_size=128, feature_dim=64, torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name() if device.type=='cuda' else None)
    config['source_sha256'] = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()}
    config['csv_sha256'] = {split:hashlib.sha256((args.data_root/split/f'{split}.csv').read_bytes()).hexdigest()
                            for split in ['train','validation','test']}
    config['template_sha256'] = hashlib.sha256((args.data_root/'sample_submission.csv').read_bytes()).hexdigest()
    path = args.output/'config.json'
    if path.exists() and (not args.resume or json.loads(path.read_text())!=config):
        raise ValueError('Output exists or code/settings/data changed; use a new output directory')
    abc.common.write_json(path, config)
    for name,source in sources.items():
        (args.output/name).write_bytes(source.read_bytes())
    frames = abc.load_frames(args)
    for name,frame in frames.items():
        print(name, len(frame), frame.experiment_id.nunique(), 'experiments', flush=True)
    pd.concat([frame[['sample_id','experiment_id']].assign(split=name) for name,frame in frames.items()]).to_csv(
        args.output/'split_manifest.csv', index=False)
    prep = abc.SensorPreprocessor(frames['fit'])
    abc.common.write_json(args.output/'preprocessing.json', prep.metadata())
    datasets = abc.prepare(args, frames, prep, device)
    results = []
    for seed in args.seeds:
        abc.common.seed_everything(seed)
        initial = copy.deepcopy(abc.FusionModel('B').state_dict())
        for name in args.losses:
            results.append(train_one(name, seed, initial, datasets, prep, args, device))
            pd.DataFrame(results).to_csv(args.output/'summary.csv', index=False)
    summarize(results, frames, args)


if __name__ == '__main__':
    main()
