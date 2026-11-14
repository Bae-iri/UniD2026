"""S0-S3: original B (frozen) + pairwise experts 3<->5 and 4<->5 (small boosted trees, causal tabular features).
An expert only acts on rows where B predicts one of its two stages, and flips only if expert prob of the other stage >= tau."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from tune import F, FEATS, PB, mf, BR, te
COLS = FEATS['temp_dyn']; PARAMS = dict(max_iter=300, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1., class_weight='balanced')

def expert(frames, a, b):
    d = pd.concat(frames); d = d[d.label.isin([a, b])]
    return [HistGradientBoostingClassifier(random_state=s, **PARAMS).fit(d[COLS], (d.label == b).values) for s in range(5)]

def p_b(models, x): return np.mean([m.predict_proba(x)[:, 1] for m in models], 0)

def apply(pred, x, experts, tau):
    out = pred.copy()
    for (a, b), ms in experts.items():
        m = np.isin(pred, [a, b]); pb = np.full(len(pred), np.nan); pb[m] = p_b(ms, x[m])
        out[(pred == a) & (pb >= tau)] = b; out[(pred == b) & (pb <= 1 - tau)] = a
    return out

def report(split, out, tag, tau):
    y = F[split].label.values; e5 = F[split].experiment_id.str.startswith('E_e5bc').values; base = PB[split].argmax(1) + 1
    c = lambda a, b: int(((y == a) & (out == b)).sum())
    f = f1_score(y, out, labels=range(1, 7), average=None)
    return dict(split=split, exp=tag, tau=tau, macro=round(100 * f.mean(), 2), macro_noE5=round(mf(y[~e5], out[~e5]), 2),
                F1_3=round(100 * f[2], 1), F1_4=round(100 * f[3], 1), F1_5=round(100 * f[4], 1),
                t5p3=c(5, 3), t5p4=c(5, 4), t3p5=c(3, 5), t4p5=c(4, 5), t2p3=c(2, 3), changed=int((out != base).sum()))

if __name__ == '__main__':
    E = {'dev': {p: expert([F['fit']], *p) for p in [(3, 5), (4, 5)]},
         'validation': {p: expert([F['fit'], F['dev']], *p) for p in [(3, 5), (4, 5)]}}
    rows = []
    for split in ['dev', 'validation']:
        base = PB[split].argmax(1) + 1; x = F[split][COLS]
        rows.append(report(split, base, 'S0 B', '-'))
        for tau in [.7, .9, .97]:
            for tag, keys in [('S1 B+3v5', [(3, 5)]), ('S2 B+4v5', [(4, 5)]), ('S3 B+both', [(3, 5), (4, 5)])]:
                rows.append(report(split, apply(base, x, {k: E[split][k] for k in keys}, tau), tag, tau))
    r = pd.DataFrame(rows); r.to_csv('pair_expert_results.csv', index=False)
    for s in ['dev', 'validation']: print(r[r.split == s].drop(columns='split').to_string(index=False), '\n')
