"""관측 간격(5/10/15초)에 강건한 MLP456 + 위험도 지표 검증.

배경: train/validation은 모두 5초 간격이지만 test 11실험 중 2개(E_a221, E_c8e730)는 10~15초 간격이다.
5초 간격으로만 학습·검증한 이력 기반 보정은 간격이 바뀌면 이득이 사라진다(b_cond_gate README 참고).

검증 구조
  - Event-aware: 실험 단위 5-fold(targeted_rule1/b_oof_v1과 같은 fold, B 확률도 그 fold의 OOF).
  - 간격: 떼어 둔 실험을 5초 / 10초 / 15초 / 10·15초 교대로 솎아 각각 평가. 특징은 솎은 관측만으로 다시 계산.
  - Risk-aware: Macro F1과 함께 5·6단계 재현율, 위험 과소 판정(실제≥5인데 더 낮게), 과잉 경보(실제≤4인데 ≥5),
    5·6단계 진입 감지 지연(초)을 보고.
  - 간격 증강 없이 해결: 정규화 float64 + 학습에서 상수인 특징은 0, 관측 개수·간격 특징 제거.
    증강 모델(V0_aug_ref)은 비교용으로만 학습하고 채택 대상에서 뺀다.
  - 채택(사전 고정, 증강 없는 후보만): 4개 간격 모두에서 OOF Macro F1이 기준보다 높고, 최악 간격에서 개선 실험 > 악화 실험.
    후보 중 최악 간격 OOF 이득이 가장 큰 것을 고른 뒤에만 validation을 읽는다.
  - test: 고정 모델로 추론만. 실험별·셔플·prefix 불변성 검증.

  python -u experiments/b_cadence_robust/run.py --output experiments/b_cadence_robust/run_v1
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.b_cond_gate import model as M
from targeted_rule1.model import matrix, f1, infer as targeted_infer
from targeted_rule1.run import rule1

DATA = Path('/home/ubuntu/data/battery_data_light')
BASE = Path('/home/ubuntu/leewon/abc_runs/current_query_v1')
BOOF = ROOT/'targeted_rule1/b_oof_v1/b_oof.npz'
TARGETED = ROOT/'targeted_rule1/run_v1'
SEEDS = [42, 43, 44]      # MLP 학습 seed
B_SEEDS = [42, 43, 44]    # 원본 B seed
E16 = 'E_e5bcddc7d77f459ea541cf658ab4346e'
COUNT_FEATURES = ['observation_gap_log1p', 'observed_count_log1p']   # 관측 개수·간격에 직접 의존
EVAL_CADENCES = {'5s': (1, 0), '10s': (2, 0), '15s': (3, 0), '10/15s': ('alt', 0)}
TRAIN_VIEWS = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), ('alt', 0), ('alt', 1)]
CANDIDATES = {   # name: (augment, drop_count_features, decode)
    'V0_orig': (False, False, 'all'),            # 정규화 수정만 적용
    'V0_nocount': (False, True, 'all'),          # + 관측 개수·간격 특징 제거
    'V0_nocount_56only': (False, True, '56only'),
    'V0_aug_ref': (True, True, 'all'),           # 간격 증강(비교용, 채택 대상 아님)
}
SELECTABLE = [c for c, (aug, _, _) in CANDIDATES.items() if not aug]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def b_probs(split, sample_ids):
    return np.stack([pd.read_csv(BASE/f'seed_{s}/B/{split}_predictions.csv').set_index('sample_id')
                     .loc[sample_ids, [f'p{i}' for i in range(1, 7)]].to_numpy() for s in B_SEEDS])


def macro(y, p):
    return float(f1(matrix(y, p)).mean())


def subsample(frame, view):
    """실험별 시간순 관측에서 stride/phase로 솎은 행 번호(원래 순서). 'alt'는 10·15초 교대."""
    stride, phase = view
    t = frame.relative_time_s.to_numpy(float)
    keep = []
    for ids in frame.groupby('experiment_id', sort=False).indices.values():
        ids = ids[np.argsort(t[ids], kind='stable')]
        if stride == 'alt':
            pos, k, step = [], phase, 0
            while k < len(ids):
                pos.append(k); k += (2, 3)[step % 2]; step += 1
            keep += list(ids[pos])
        else:
            keep += list(ids[phase::stride])
    return np.sort(np.asarray(keep, int))


def view_data(frame, probs, view):
    idx = subsample(frame, view)
    f = frame.iloc[idx].reset_index(drop=True)
    p = probs[:, idx]
    x, names, b, c = M.build_features(f, p)
    return dict(idx=idx, frame=f, probs=p, x=x, b=b, names=names,
                base=rule1(f, p.mean(0).argmax(1)+1), exps=f.experiment_id.to_numpy(),
                y=f.label.to_numpy(int) if 'label' in f else None)


class Model:
    def __init__(self, x, b, y, exps, keep_cols, seeds=None):
        m = np.isin(y, [4, 5, 6])
        self.cols = keep_cols
        self.prep = M.Prep().fit(x[m][:, keep_cols], np.zeros((m.sum(), 2)), exps[m])
        xs = self.prep.x(x[m][:, keep_cols])
        w = M.sample_weights(y[m]-4, exps[m])
        self.nets = []
        for s in (SEEDS if seeds is None else seeds):
            torch.manual_seed(s)
            net = M.MLP456(xs.shape[1]); net.train()
            M._loop(net.parameters(), lambda xx, bb, cc, n=net: n(xx, bb), xs, b[m], np.zeros((m.sum(), 1), np.float32),
                    y[m]-4, w, M.BASE_EPOCHS, s)
            net.eval(); self.nets.append(net)

    @torch.no_grad()
    def p456(self, x, b):
        xs = torch.from_numpy(self.prep.x(x[:, self.cols])); tb = torch.from_numpy(b)
        return np.mean([torch.softmax(n(xs, tb), 1).numpy() for n in self.nets], 0)


def decode(base, p, mode):
    out = M.apply456(base, p)
    if mode == '56only':
        base = np.asarray(base, int)
        ok = ((base == 5) & (out == 6)) | ((base == 6) & (out == 5))
        out = np.where(ok, out, base)
    return out


def onset_delays(frame, y, p, stage):
    """실험별 (예측이 처음 stage 이상이 된 시각) - (실제 처음 stage 이상 시각). 예측이 없으면 missed."""
    t = frame.relative_time_s.to_numpy(float)
    delays, missed = [], 0
    for ids in frame.groupby('experiment_id', sort=False).indices.values():
        ids = ids[np.argsort(t[ids])]
        true = ids[y[ids] >= stage]
        if not len(true):
            continue
        pred = ids[p[ids] >= stage]
        if not len(pred):
            missed += 1; continue
        delays.append(t[pred[0]]-t[true[0]])
    return delays, missed


def metrics(frame, y, base, p):
    exps = frame.experiment_id.to_numpy()
    groups = list(pd.Series(range(len(y))).groupby(exps).indices.values())
    cb = np.stack([matrix(y[i], base[i]) for i in groups]); cp = np.stack([matrix(y[i], p[i]) for i in groups])
    rng = np.random.default_rng(20261114)
    draws = rng.integers(0, len(groups), (2000, len(groups)))
    boot = f1(cp[draws].sum(1)).mean(-1)-f1(cb[draws].sum(1)).mean(-1)
    per = f1(cp).mean(1)-f1(cb).mean(1)
    cm = cp.sum(0)
    r = dict(macro_f1=macro(y, p), d_vs_base=macro(y, p)-macro(y, base),
             ci_lo=float(np.quantile(boot, .025)), ci_hi=float(np.quantile(boot, .975)),
             fixed=int(((base != y) & (p == y)).sum()), broken=int(((base == y) & (p != y)).sum()),
             changed=int((base != p).sum()), better_exps=int((per > 1e-12).sum()), worse_exps=int((per < -1e-12).sum()),
             true4_pred5=int(cm[3, 4]), true5_pred4=int(cm[4, 3]), true5_pred6=int(cm[4, 5]), true6_pred5=int(cm[5, 4]),
             recall5=float(cm[4, 4]/max(cm[4].sum(), 1)), recall6=float(cm[5, 5]/max(cm[5].sum(), 1)),
             danger_under=int(((y >= 5) & (p < y)).sum()), false_alarm=int(((y <= 4) & (p >= 5)).sum()))
    for s in (5, 6):
        d, miss = onset_delays(frame, y, p, s)
        r[f'onset{s}_median_delay_s'] = float(np.median(d)) if d else np.nan
        r[f'onset{s}_late_exps'] = int((np.asarray(d) > 0).sum()) if d else 0
        r[f'onset{s}_missed_exps'] = miss
    return r


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(6)
    src = [Path(__file__), Path(M.__file__), ROOT/'targeted_rule1/model.py', ROOT/'targeted_rule1/run.py',
           ROOT/'compare_abc_phase_history.py']
    (out/'protocol.json').write_text(json.dumps(dict(
        candidates=CANDIDATES, train_views=TRAIN_VIEWS, eval_cadences=EVAL_CADENCES, count_features=COUNT_FEATURES,
        seeds=SEEDS, epochs=M.BASE_EPOCHS, hidden=M.HIDDEN,
        adoption='OOF d_vs_base>0 at all cadences and better_exps>worse_exps at worst cadence; '
                 'choose max worst-cadence OOF gain; validation read after choice',
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in src}), indent=2, default=str))

    frame = pd.read_csv(DATA/'train/train.csv')
    z = np.load(BOOF)
    assert np.array_equal(z['sample_id'], frame.sample_id.to_numpy(str))
    fold_of_exp = dict(zip(frame.experiment_id, z['fold']))
    probs = z['oof']

    views = {v: view_data(frame, probs, v) for v in TRAIN_VIEWS}
    names = views[(1, 0)]['names']
    all_cols = np.arange(len(names)); robust_cols = np.array([i for i, n in enumerate(names) if n not in COUNT_FEATURES])
    print('views ready', {str(k): len(v['idx']) for k, v in views.items()}, flush=True)

    def stack(view_keys, exp_set):
        parts = [views[k] for k in view_keys]
        m = [np.isin(v['exps'], list(exp_set)) for v in parts]
        return (np.concatenate([v['x'][i] for v, i in zip(parts, m)]), np.concatenate([v['b'][i] for v, i in zip(parts, m)]),
                np.concatenate([v['y'][i] for v, i in zip(parts, m)]),
                # 같은 실험의 다른 간격 버전은 같은 실험으로 묶어 가중치 균형
                np.concatenate([v['exps'][i] for v, i in zip(parts, m)]))

    eval_views = {name: views[v] for name, v in EVAL_CADENCES.items()}
    oof = {c: {e: np.zeros(len(v['y']), int) for e, v in eval_views.items()} for c in CANDIDATES}
    targeted_oof = {e: np.zeros(len(v['y']), int) for e, v in eval_views.items()}
    exps_all = np.array(sorted(fold_of_exp))
    for k in range(5):
        fit_exps = set(exps_all[[fold_of_exp[e] != k for e in exps_all]])
        held_exps = set(exps_all) - fit_exps
        fitted = {}
        for c, (aug, drop, _) in CANDIDATES.items():
            key = (aug, drop)
            if key not in fitted:
                x, b, y, ex = stack(TRAIN_VIEWS if aug else [(1, 0)], fit_exps)
                fitted[key] = Model(x, b, y, ex, robust_cols if drop else all_cols)
        heads = joblib.load(TARGETED/f'fold_{k}_heads.joblib')
        for e, v in eval_views.items():
            m = np.isin(v['exps'], list(held_exps))
            for c, (aug, drop, mode) in CANDIDATES.items():
                oof[c][e][m] = decode(v['base'][m], fitted[(aug, drop)].p456(v['x'][m], v['b'][m]), mode)
            hf = v['frame'][m].reset_index(drop=True)
            targeted_oof[e][m] = targeted_infer(heads, hf, v['probs'][:, m], v['base'][m])['both']
        print('fold', k, {c: round(macro(eval_views['10/15s']['y'][np.isin(eval_views['10/15s']['exps'], list(held_exps))],
                                          oof[c]['10/15s'][np.isin(eval_views['10/15s']['exps'], list(held_exps))]), 4)
                          for c in CANDIDATES}, flush=True)

    rows = []
    for e, v in eval_views.items():
        rows.append(dict(split='train_oof', cadence=e, method='B_base', **metrics(v['frame'], v['y'], v['base'], v['base'])))
        rows.append(dict(split='train_oof', cadence=e, method='targeted_rule1', **metrics(v['frame'], v['y'], v['base'], targeted_oof[e])))
        for c in CANDIDATES:
            rows.append(dict(split='train_oof', cadence=e, method=c, **metrics(v['frame'], v['y'], v['base'], oof[c][e])))
    table = pd.DataFrame(rows)

    # ---- 사전 고정 채택 규칙 (OOF만 사용) ----
    choice = {}
    for c in SELECTABLE:
        t = table[(table.method == c)]
        worst = t.loc[t.d_vs_base.idxmin()]
        choice[c] = dict(all_positive=bool((t.d_vs_base > 0).all()), worst_cadence=worst.cadence,
                         worst_gain=float(worst.d_vs_base), worst_better_gt_worse=bool(worst.better_exps > worst.worse_exps))
    ok = [c for c, r in choice.items() if r['all_positive'] and r['worst_better_gt_worse']]
    selected = max(ok, key=lambda c: choice[c]['worst_gain']) if ok else None
    (out/'selection.json').write_text(json.dumps(dict(choice=choice, selected=selected), indent=2))
    print('SELECTION', json.dumps(dict(choice=choice, selected=selected), indent=1), flush=True)

    # ---- 최종 모델: train 전체 (선택 이후 validation 읽기) ----
    finals = {}
    for c, (aug, drop, _) in CANDIDATES.items():
        if (aug, drop) not in finals:
            x, b, y, ex = stack(TRAIN_VIEWS if aug else [(1, 0)], set(exps_all))
            finals[(aug, drop)] = Model(x, b, y, ex, robust_cols if drop else all_cols)
    torch.save({str(k): dict(prep=m.prep.__dict__, cols=m.cols, states=[n.state_dict() for n in m.nets]) for k, m in finals.items()},
               out/'final_models.pt')
    targeted_final = joblib.load(TARGETED/'deployment_heads.joblib')

    val = pd.read_csv(DATA/'validation/validation.csv')
    vp = b_probs('validation', val.sample_id)
    for e, view in EVAL_CADENCES.items():
        v = view_data(val, vp, view)
        preds = {'B_base': v['base'],
                 'targeted_rule1': targeted_infer(targeted_final, v['frame'], v['probs'], v['base'])['both']}
        for c, (aug, drop, mode) in CANDIDATES.items():
            preds[c] = decode(v['base'], finals[(aug, drop)].p456(v['x'], v['b']), mode)
        keep = v['exps'] != E16
        sub = v['frame'][keep].reset_index(drop=True)
        for c, p in preds.items():
            rows.append(dict(split='val', cadence=e, method=c, **metrics(v['frame'], v['y'], v['base'], p)))
            rows.append(dict(split='val_noE16', cadence=e, method=c, **metrics(sub, v['y'][keep], v['base'][keep], p[keep])))
    table = pd.DataFrame(rows)
    table.to_csv(out/'summary.csv', index=False)
    cols = ['split', 'cadence', 'method', 'macro_f1', 'd_vs_base', 'ci_lo', 'fixed', 'broken', 'better_exps', 'worse_exps',
            'recall6', 'danger_under', 'false_alarm', 'onset6_median_delay_s']
    pd.set_option('display.width', 250)
    print(table[cols].round(4).to_string(index=False), flush=True)

    # ---- Test: 선택된 모델로 추론만 ----
    test = pd.read_csv(DATA/'test/test.csv'); assert 'label' not in test
    tp = b_probs('test', test.sample_id)
    sample = pd.read_csv(DATA/'sample_submission.csv')
    manifest = dict(selected=selected)
    for c in ([selected] if selected else []):
        aug, drop, mode = CANDIDATES[c]; model = finals[(aug, drop)]

        def run_all(f, pp):
            base = rule1(f, pp.mean(0).argmax(1)+1)
            x, n, b, _ = M.build_features(f, pp)
            assert n == names
            return decode(base, model.p456(x, b), mode), base

        full, tb = run_all(test, tp)
        checks = dict(per_experiment=True, shuffled=True, prefixes=0, prefix_mismatch=0)
        t = test.relative_time_s.to_numpy()
        for ids in test.groupby('experiment_id').indices.values():
            o, _ = run_all(test.iloc[ids].reset_index(drop=True), tp[:, ids])
            checks['per_experiment'] &= bool(np.array_equal(o, full[ids]))
            ids = ids[np.argsort(t[ids], kind='stable')]
            for end in sorted(set(np.linspace(1, len(ids), 12).astype(int))):
                s = ids[:end]; o, _ = run_all(test.iloc[s].reset_index(drop=True), tp[:, s])
                checks['prefixes'] += 1; checks['prefix_mismatch'] += int(not np.array_equal(o, full[s]))
        perm = np.random.default_rng(0).permutation(len(test))
        o, _ = run_all(test.iloc[perm].reset_index(drop=True), tp[:, perm])
        checks['shuffled'] = bool(np.array_equal(o, full[perm]))
        subm = sample[['sample_id']].merge(pd.DataFrame({'sample_id': test.sample_id, 'label': full}), on='sample_id', how='left')
        assert len(subm) == len(sample) and subm.label.between(1, 6).all()
        subm['label'] = subm.label.astype(int)
        path = out/f'submission_{c}.csv'; subm.to_csv(path, index=False)
        changed = pd.Series(test.experiment_id[full != tb]).str[:8].value_counts().to_dict()
        manifest.update(checks=checks, file=path.name, sha256=sha(path), changed_vs_base=int((full != tb).sum()),
                        changed_by_experiment=changed,
                        changes={f'{a}->{b}': int(n) for (a, b), n in
                                 pd.DataFrame({'b': tb[full != tb], 'n': full[full != tb]}).value_counts().items()})
    (out/'test_manifest.json').write_text(json.dumps(manifest, indent=2, default=int))
    print('TEST', json.dumps(manifest, indent=1, default=int), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output', type=Path, required=True)
    main(ap.parse_args().output)
