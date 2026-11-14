"""C0/C1/C2: does battery type (bat_class) / capacity (bat_cc) help the 4/5/6 specialist on top of FROZEN original B?
Specialist = existing V3 boosted-tree specialist (HistGradientBoosting x5 bags, same params/seeds/weights), mixed 0.7 with B's 4/5/6 probs,
applied only where B predicts >=4. Only the feature set changes. Encoding stats from the training frames of each evaluation.
Nothing existing is overwritten; outputs go to battery_check/."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from tune import F, MODELS, specialist, spec_proba, combine, PB, BR
from variants import V
OUT = HERE
BASE = V['V3 C - all current temps']
E5 = 'E_e5bcddc7d77f459ea541cf658ab4346e'

def encode(train_frames, frames):
    """one-hot bat_class (categories seen in training), log1p(bat_cc) standardized with training stats."""
    tr = pd.concat(train_frames); cats = sorted(tr.bat_class.dropna().unique())
    lc = np.log1p(tr.bat_cc.astype(float)); mu, sd = lc.mean(), lc.std() or 1.
    out = []
    for f in frames:
        f = f.copy()
        for c in cats: f[f'bat_class_{int(c)}'] = (f.bat_class == c).astype(float)
        f['bat_cc_z'] = (np.log1p(f.bat_cc.astype(float)) - mu) / sd
        out.append(f)
    return out, [f'bat_class_{int(c)}' for c in cats]

def featsets(cls_cols):
    return {'C0': BASE, 'C1': BASE + cls_cols, 'C2': BASE + cls_cols + ['bat_cc_z']}

def counts(y, p):
    c = lambda a, b: int(((y == a) & (p == b)).sum())
    f = f1_score(y, p, labels=range(1, 7), average=None, zero_division=0)
    return dict(macro=round(100 * f.mean(), 2), **{f'F1_{i+1}': round(100 * f[i], 1) for i in range(6)},
                t4p5=c(4, 5), t5p4=c(5, 4), t4p6=c(4, 6), t6p4=c(6, 4), t5p6=c(5, 6), t6p5=c(6, 5))

# ---------- 1. battery combos: stage 4/5 rows and experiments per split ----------
def combo_table():
    rows = []
    for s in ['fit', 'dev', 'validation', 'test']:
        f = F[s]; g = f.groupby(['bat_class', 'bat_cc'])
        for (bc, cc), d in g:
            r = dict(split=s, bat_class=int(bc), bat_cc=cc, experiments=d.experiment_id.nunique(), rows=len(d))
            if 'label' in d:
                r.update(rows_4=int((d.label == 4).sum()), rows_5=int((d.label == 5).sum()),
                         exps_with_4=d[d.label == 4].experiment_id.nunique(), exps_with_5=d[d.label == 5].experiment_id.nunique())
            rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(OUT / '1_combo_counts.csv', index=False); return t

# ---------- 2/3/5. B + specialist on B-out-of-sample splits ----------
def evaluate_split(split, train_frames):
    enc, cls = encode(train_frames, train_frames + [F[split]]); trf, ev = enc[:-1], enc[-1]
    y = ev.label.values; base = PB[split].argmax(1) + 1; e5 = (ev.experiment_id == E5).values
    preds, rows = {}, []
    for name, cols in featsets(cls).items():
        p = combine(PB[split], spec_proba(specialist(trf, cols, MODELS['mid']), ev[cols]), .7); preds[name] = p
        for scope, m in [('incl E_e5bc', np.ones(len(y), bool)), ('excl E_e5bc', ~e5)]:
            if split == 'dev' and scope == 'excl E_e5bc': continue
            rows.append(dict(split=split, config=name, scope=scope, n_feat=len(cols), **counts(y[m], p[m])))
    rows.append(dict(split=split, config='B only', scope='incl E_e5bc', **counts(y, base)))
    if split == 'validation': rows.append(dict(split=split, config='B only', scope='excl E_e5bc', **counts(y[~e5], base[~e5])))
    # 5. flips vs C0
    flips = []
    for name in ['C1', 'C2']:
        for scope, m in [('incl E_e5bc', np.ones(len(y), bool)), ('excl E_e5bc', ~e5)]:
            a, b = preds['C0'][m], preds[name][m]; yy = y[m]
            ch = a != b
            flips.append(dict(split=split, config=name, scope=scope, changed=int(ch.sum()),
                              right_to_wrong=int((ch & (a == yy)).sum()), wrong_to_right=int((ch & (b == yy)).sum()),
                              wrong_to_wrong=int((ch & (a != yy) & (b != yy)).sum()),
                              detail=' '.join(f'true{t}:{x}>{z}={n}' for (t, x, z), n in pd.Series(list(zip(yy[ch], a[ch], b[ch]))).value_counts().items()) if ch.any() else ''))
    detail = ev[['sample_id', 'experiment_id', 'bat_class', 'bat_cc', 'label']].assign(B=base, C0=preds['C0'], C1=preds['C1'], C2=preds['C2'])
    detail[(detail.C0 != detail.C1) | (detail.C0 != detail.C2)].to_csv(OUT / f'5_changed_rows_{split}.csv', index=False)
    return rows, flips

# ---------- 4. GroupKFold inside train (specialist alone on true 4/5/6 rows; B has no OOF probs on fit) ----------
def groupkfold():
    tr = pd.concat([F['fit'], F['dev']], ignore_index=True); rows = []
    for k, (i, j) in enumerate(GroupKFold(5).split(tr, groups=tr.experiment_id)):
        a, b = tr.iloc[i], tr.iloc[j]; (a, b), cls = encode([a], [a, b])
        bt = b[b.label >= 4]; y = bt.label.values
        for name, cols in featsets(cls).items():
            p = spec_proba(specialist([a], cols, MODELS['mid']), bt[cols]).argmax(1) + 4
            f = f1_score(y, p, labels=[4, 5, 6], average=None, zero_division=0)
            rows.append(dict(fold=k, config=name, test_exps=b.experiment_id.nunique(), macro456=round(100 * f.mean(), 2),
                             F1_4=round(100 * f[0], 1), F1_5=round(100 * f[1], 1), F1_6=round(100 * f[2], 1),
                             t4p5=int(((y == 4) & (p == 5)).sum()), t5p4=int(((y == 5) & (p == 4)).sum())))
    t = pd.DataFrame(rows); t.to_csv(OUT / '4_groupkfold_specialist.csv', index=False); return t

# ---------- 6. test combos covered by train ----------
def test_coverage(combos):
    tr = combos[combos.split.isin(['fit', 'dev'])].groupby(['bat_class', 'bat_cc'])[['experiments', 'rows_4', 'rows_5', 'exps_with_4', 'exps_with_5']].sum()
    va = combos[combos.split == 'validation'].groupby(['bat_class', 'bat_cc'])[['experiments', 'rows_4', 'rows_5']].sum().add_prefix('val_')
    te = combos[combos.split == 'test'].groupby(['bat_class', 'bat_cc'])[['experiments', 'rows']].sum().add_prefix('test_')
    t = te.join(tr, how='left').join(va, how='left').fillna(0).astype(int); t.to_csv(OUT / '6_test_combo_coverage.csv'); return t

if __name__ == '__main__':
    pd.set_option('display.width', 250)
    combos = combo_table()
    print('== 1. stage 4/5 rows & experiments per (bat_class, bat_cc) [train = fit+dev]')
    tt = combos[combos.split.isin(['fit', 'dev'])].groupby(['bat_class', 'bat_cc'])[['experiments', 'rows_4', 'rows_5', 'exps_with_4', 'exps_with_5']].sum()
    print(tt.astype(int).to_string())
    print('\n== 6. test combinations vs train/validation coverage'); print(test_coverage(combos).to_string())
    res, flips = [], []
    for split, trf in [('dev', [F['fit']]), ('validation', [F['fit'], F['dev']])]:
        r, f = evaluate_split(split, trf); res += r; flips += f
    res = pd.DataFrame(res); res.to_csv(OUT / '2_3_scores.csv', index=False)
    print('\n== 2/3. B + specialist (B frozen; dev and validation are out-of-sample for B)')
    print(res[['split', 'config', 'scope', 'macro', 'F1_3', 'F1_4', 'F1_5', 'F1_6', 't4p5', 't5p4', 't4p6', 't6p4', 't5p6', 't6p5']].to_string(index=False))
    flips = pd.DataFrame(flips); flips.to_csv(OUT / '5_flips_vs_C0.csv', index=False)
    print('\n== 5. prediction changes vs C0'); print(flips.to_string(index=False))
    g = groupkfold()
    print('\n== 4. GroupKFold(5) inside train, specialist alone on true 4/5/6 rows')
    piv = g.pivot(index='fold', columns='config', values='macro456'); print(piv.to_string())
    print(' mean:', g.groupby('config')[['macro456', 'F1_4', 'F1_5', 'F1_6', 't4p5', 't5p4']].mean().round(2).to_string())
    print(' C1-C0 per fold:', (piv.C1 - piv.C0).round(2).tolist(), '| C2-C0 per fold:', (piv.C2 - piv.C0).round(2).tolist())
    # test CSVs (new names only); specialist trained on fit+dev+validation, encoding stats from those frames
    allf = [F['fit'], F['dev'], F['validation']]; enc, cls = encode(allf, allf + [F['test']]); trf, te = enc[:-1], enc[-1]
    o = pd.read_csv(BR + 'submission_B.csv'); v3 = pd.read_csv(HERE.parent / 'submission_B_spec456_V3.csv').label.values
    for name, cols in featsets(cls).items():
        p = pd.Series(combine(PB['test'], spec_proba(specialist(trf, cols, MODELS['mid']), te[cols]), .7), index=te.sample_id).loc[o.sample_id].values.astype(int)
        out = o.copy(); out['label'] = p; assert len(out) == 3451 and out.label.between(1, 6).all() and out.sample_id.is_unique
        out.to_csv(OUT / f'submission_battery_check_{name}.csv', index=False)
        print(f'test {name}: changed vs B {int((p != o.label.values).sum())} | vs existing V3 file {int((p != v3).sum())}')
