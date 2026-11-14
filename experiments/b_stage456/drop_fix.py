"""4/5/6 specialist (robust) + post-drop fix.
A: duration rule - stage 4 never lasts longer than T seconds after the first drop in TRAIN; beyond T, pred 4 -> 5.
B: train-derived post-drop stage prior (b_more_data/drop_prior.py) applied before the 4/5/6 decision."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/ubuntu/gahyun/experiments/b_more_data')
from drop_prior import table, apply as prior_apply
from tune import F, FEATS, MODELS, specialist, spec_proba, PB, BR, te
from pair_experts import report
cols = FEATS['temp_dyn']; W = .7
tr = pd.concat([F['fit'], F['dev']]); T_MAX = tr[(tr.label == 4) & (tr.drop_occurred > 0)].time_since_drop.max()
T_P99 = np.percentile(tr[(tr.label == 4) & (tr.drop_occurred > 0)].time_since_drop, 99)
print('train stage-4 post-drop duration: max', T_MAX, 'p99', T_P99)
R, _ = table()

_PS = {}
def run(split, frames, fix, T=None):
    Pb = PB[split]
    if split not in _PS: _PS[split] = spec_proba(specialist(frames, cols, MODELS['mid']), F[split][cols])
    if fix == 'prior': Pb = prior_apply(Pb, split, F[split].sample_id, R)
    pred = Pb.argmax(1) + 1; m = pred >= 4
    ps = _PS[split]
    b456 = Pb[m, 3:] / Pb[m, 3:].sum(1, keepdims=True); mix = W * ps[m] + (1 - W) * b456
    if fix == 'prior':  # also apply the prior to the mixed 4/5/6 decision
        mix = mix * R[np.digitize(F[split].time_since_drop.values[m], [60, 300, 600])][:, 3:] ** (F[split].drop_occurred.values[m] > 0)[:, None]
    out = pred.copy(); out[m] = mix.argmax(1) + 4
    if fix == 'duration': out[(out == 4) & (F[split].drop_occurred.values > 0) & (F[split].time_since_drop.values > T)] = 5
    return out

if __name__ == '__main__':
    rows = []
    for split, frames in [('dev', [F['fit']]), ('validation', [F['fit'], F['dev']])]:
        rows.append(report(split, PB[split].argmax(1) + 1, 'S0 B', '-'))
        rows.append(report(split, run(split, frames, None), 'robust 456', '-'))
        rows.append(report(split, run(split, frames, 'duration', T_MAX), 'robust + duration(max)', T_MAX))
        rows.append(report(split, run(split, frames, 'duration', T_P99), 'robust + duration(p99)', round(T_P99)))
        rows.append(report(split, run(split, frames, 'prior'), 'robust + drop prior', '-'))
    r = pd.DataFrame(rows); r.to_csv('drop_fix_results.csv', index=False); print(r.to_string(index=False))
    allf = [F['fit'], F['dev'], F['validation']]; o = pd.read_csv(BR + 'submission_B.csv')
    ids = F['test'].sample_id; ex = F['test'].experiment_id.values; tsd = F['test'].time_since_drop.values
    base = run('test', allf, None)
    a = (ex == 'E_a221e4b97e024678bc423606fe6a2961') & (F['test'].drop_occurred.values > 0)
    d = pd.DataFrame(dict(tsd=tsd[a], t1=F['test'].meas_t1.values[a], B=PB['test'].argmax(1)[a] + 1, robust=base[a])).sort_values('tsd')
    print('\nE_a221 after drop (robust):'); print(d.iloc[::8].to_string(index=False))
    for tag, fix, T in [('robust', None, None), ('robust_duration', 'duration', T_MAX), ('robust_dropprior', 'prior', None)]:
        p = run('test', allf, fix, T); out = o.copy(); out['label'] = pd.Series(p, index=ids).loc[o.sample_id].values.astype(int)
        assert len(out) == 3451 and out.label.between(1, 6).all()
        fn = f'submission_B_spec456_{tag}_v2.csv'; out.to_csv(fn, index=False)
        print(fn, 'changed vs B', int((out.label != o.label).sum()), '| vs robust', int((p != base).sum()))
