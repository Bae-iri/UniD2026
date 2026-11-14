"""Frozen original B + LINEAR correction for stages 4/5/6 (only rows B predicts >=4).
L1: logistic-regression specialist trained on train 4-6 rows (independent of B), mixed with B's 4-6 probs.
L2: logistic regression on [B log-probs(4,5,6) + features], cross-fitted on B-out-of-sample splits
    (train on dev -> eval validation, train on validation -> eval dev), experiment-balanced weights."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tune import F, FEATS, PB, combine, mf

def clean(x):
    x = x.replace([np.inf, -np.inf], np.nan).astype(float)
    return x.clip(x.quantile(.001), x.quantile(.999), axis=1) if len(x) > 1000 else x

def lr(C): return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(), LogisticRegression(C=C, max_iter=3000))

def weights(d, by_exp):
    w = d.label.map(1 / d.label.value_counts()).values
    if by_exp: w = w / d.experiment_id.map(d.experiment_id.value_counts()).values
    return w / w.mean()

def blog(split): P = PB[split]; return np.log(P[:, 3:] / P[:, 3:].sum(1, keepdims=True) + 1e-6)

def apply_l2(split, model, cols):
    pred = PB[split].argmax(1) + 1; m = pred >= 4
    x = np.hstack([blog(split)[m], F[split][cols].pipe(clean).values[m]])
    out = pred.copy(); out[m] = model.predict(x); return out

rows = []; yd, yv = F['dev'].label.values, F['validation'].label.values
keep = ~F['validation'].experiment_id.str.startswith('E_e5bc').values
def rec(kind, fname, C, w, d, v):
    rows.append(dict(kind=kind, features=fname, C=C, w=w, dev=round(mf(yd, d), 2), val=round(mf(yv, v), 2), val_noE5=round(mf(yv[keep], v[keep]), 2),
                     mean=round((mf(yd, d) + mf(yv, v)) / 2, 2), dev_changed=int((d != PB['dev'].argmax(1) + 1).sum()), val_changed=int((v != PB['validation'].argmax(1) + 1).sum())))
rec('B only', '-', '-', '-', PB['dev'].argmax(1) + 1, PB['validation'].argmax(1) + 1)
for fname, cols in FEATS.items():
    for C in [.01, .1, 1.]:
        # L1
        tr1 = pd.concat([F['fit']]); tr1 = tr1[tr1.label >= 4]; tr2 = pd.concat([F['fit'], F['dev']]); tr2 = tr2[tr2.label >= 4]
        m1 = lr(C).fit(clean(tr1[cols]), tr1.label, logisticregression__sample_weight=weights(tr1, False))
        m2 = lr(C).fit(clean(tr2[cols]), tr2.label, logisticregression__sample_weight=weights(tr2, False))
        pd_, pv_ = m1.predict_proba(clean(F['dev'][cols])), m2.predict_proba(clean(F['validation'][cols]))
        for w in [.5, .7, 1.]: rec('L1 linear specialist', fname, C, w, combine(PB['dev'], pd_, w), combine(PB['validation'], pv_, w))
        # L2 cross-fit on B-out-of-sample rows (true label 4-6 and B pred >=4)
        fits = {}
        for s in ['dev', 'validation']:
            m = (F[s].label.values >= 4) & (PB[s].argmax(1) >= 3); d = F[s][m]
            x = np.hstack([blog(s)[m], clean(F[s][cols]).values[m]])
            fits[s] = lr(C).fit(x, d.label, logisticregression__sample_weight=weights(d, True))
        rec('L2 B-logit correction', fname, C, '-', apply_l2('dev', fits['validation'], cols), apply_l2('validation', fits['dev'], cols))
r = pd.DataFrame(rows); r.to_csv('linear_results.csv', index=False)
print(r.iloc[:1].to_string(index=False)); print(r.iloc[1:].sort_values('mean', ascending=False).groupby('kind').head(4).to_string(index=False))
