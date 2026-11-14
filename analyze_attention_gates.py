"""Aggregate-only checkpoint diagnostics. No training or row-level exports.
Uses compare_gate_branches architecture; supports existing b_gate/image_gate checkpoints.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import compare_gate_branches as common

TIME_EDGES = [0, 60, 180, 600, 1800, float('inf')]


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'count': 0}
    return dict(count=len(a), mean=float(a.mean()), std=float(a.std()),
                **{name: float(v) for name, v in zip(
                    ['min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'],
                    np.quantile(a, [0, .05, .25, .5, .75, .95, 1]))})


@torch.inference_mode()
def analyze(model, batches, times, device):
    captured = {}
    original_select = model.select

    def select(*args):
        g, v = original_select(*args)
        captured['gas_before'], captured['image_before'] = g, v
        captured['gates'] = torch.sigmoid(model.gate(args[0]))
        return g, v

    def head_input(module, inputs):
        captured['head_input'] = inputs[0]

    model.select = select
    handle = model.head.register_forward_pre_hook(head_input)
    records, predictions, targets = [], {k: [] for k in ['normal', 'bypass_image', 'bypass_gas', 'bypass_both']}, []
    d = model.gate.in_features
    active = {'gas': model.comparison_mode in ('B_gate', 'B_gas_gate'),
              'image': model.comparison_mode in ('B_gate', 'B_image_gate')}
    model.eval()
    try:
        for batch in batches:
            batch = common.on_device(batch, device)
            logits = model(batch)
            features = captured['head_input'].clone()
            gates = captured['gates']
            columns = [gates]
            for branch, slot in [('gas', 4), ('image', 5)]:
                before = captured[branch+'_before']
                after = features[:, slot*d:(slot+1)*d]
                columns += [before.norm(dim=1, keepdim=True), after.norm(dim=1, keepdim=True)]
            records.append(torch.cat(columns, dim=1).cpu().numpy())
            predictions['normal'].extend(logits.argmax(1).cpu().tolist())
            targets.extend(batch['target'].cpu().tolist())
            # Same learned checkpoint/head; inference intervention, NOT a retrained B.
            for name, branches in [('bypass_image', ['image']), ('bypass_gas', ['gas']),
                                   ('bypass_both', ['gas', 'image'])]:
                changed = features.clone()
                for branch in branches:
                    slot = 4 if branch == 'gas' else 5
                    changed[:, slot*d:(slot+1)*d] = captured[branch+'_before']
                predictions[name].extend(model.head(changed).argmax(1).cpu().tolist())
    finally:
        handle.remove()
        model.select = original_select
    a = np.concatenate(records)
    if len(a) != len(times):
        raise ValueError('Unexpected loader order/length')
    result = {'count': len(a), 'gates': {}, 'time_bins': [], 'inference_interventions': {}}
    for branch, index, normcol in [('gas', 0, 2), ('image', 1, 4)]:
        gate = a[:, index]
        corr = None
        if len(gate)>1 and gate.std()>1e-10 and np.std(times)>1e-10:
            corr = float(np.corrcoef(gate, times)[0, 1])
        result['gates'][branch] = dict(applied=active[branch], raw_sigmoid=describe(gate),
            fraction_below_0_1=float(np.mean(gate < .1)), fraction_above_0_9=float(np.mean(gate > .9)),
            correlation_with_elapsed_seconds=corr,
            l2_before=describe(a[:, normcol]), l2_after=describe(a[:, normcol+1]))
    for lo, hi in zip(TIME_EDGES[:-1], TIME_EDGES[1:]):
        mask = (times >= lo) & (times < hi)
        result['time_bins'].append(dict(start_seconds=lo, end_seconds=hi if np.isfinite(hi) else None,
            count=int(mask.sum()), gas_gate=describe(a[mask, 0]), image_gate=describe(a[mask, 1])))
    target = np.asarray(targets)
    normal = np.asarray(predictions['normal'])
    for name, pred in predictions.items():
        pred = np.asarray(pred)
        entry = {'fraction_predictions_changed': float(np.mean(pred != normal))}
        if (target >= 0).all():
            entry.update(common.metrics(target, pred))
        result['inference_interventions'][name] = entry
    return result


def run(args):
    if args.output.exists():
        raise FileExistsError('Output exists; use a new filename')
    torch.set_num_threads(args.threads)
    reports = []
    for checkpoint in args.checkpoints:
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = common.BGateModel(ck['mode'], ck['image_size'], ck['dim']).to(args.device)
        model.load_state_dict(ck['state_dict'])
        report = dict(checkpoint=str(checkpoint), mode=ck['mode'], seed=ck['seed'],
                      best_epoch=ck['best_epoch'], saved_validation=ck['validation'], splits={})
        for split in args.splits:
            frame = common.read_frame(args.data_dir/split/f'{split}.csv')
            times = common.pd.to_numeric(frame.relative_time_s, errors='raise').to_numpy(float)
            ds = common.Observations(frame, args.data_dir/split, ck['preprocess'], ck['image_size'])
            loader_args = SimpleNamespace(batch_size=args.batch_size, workers=args.workers)
            report['splits'][split] = analyze(model, common.loader(ds, loader_args), times, args.device)
            print(f'Aggregated {ck["mode"]} seed={ck["seed"]} {split}; rows={len(ds)}', flush=True)
        reports.append(report)
    payload = dict(notes=[
        'Only aggregate statistics are exported. No sample IDs, row labels, predictions, images or probabilities.',
        'Time bins are fixed seconds, not normalized by experiment duration.',
        'Inactive gate outputs are untrained and must not be interpreted.',
        'Bypass scores are inference interventions on the gated checkpoint, not a retrained baseline.',
        'Feature L2 norm is not a causal importance measurement.'], checkpoints=reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(f'Saved aggregate report: {args.output}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--splits', nargs='+', choices=['train','validation','test'], default=['validation','test'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    args = p.parse_args()
    if args.threads<1 or args.batch_size<1 or args.workers<0:
        p.error('Invalid thread, batch or worker count')
    run(args)
