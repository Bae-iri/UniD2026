"""Check whether suppressing unconfirmed 3 predictions improves the 2->3 direction."""
import numpy as np
import pandas as pd
from analyze import DATA,BASE,OUT,features,rule1,evaluate
from sklearn.model_selection import GroupShuffleSplit


def debounce(d,base,seconds):
    p=base.copy()
    for ids in d.groupby('experiment_id').indices.values():
        ids=ids[np.argsort(d.relative_time_s.to_numpy()[ids])]
        start=None;active=False;prev=None
        for i in ids:
            t=float(d.relative_time_s.iat[i]);gap=prev is not None and t-prev>15;prev=t
            if base[i]==3:
                if start is None or gap:start=t
                if t-start>=seconds:active=True
            else:start=None
            if base[i] in [2,3]:p[i]=3 if active else 2
    return p


def main():
    train=pd.read_csv(DATA/'train/train.csv');_,idx=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261114).split(train,groups=train.experiment_id))
    rows=[]
    for split,raw in [('dev',train.iloc[idx].reset_index(drop=True)),('validation',pd.read_csv(DATA/'validation/validation.csv'))]:
        d=features(raw);ref=pd.read_csv(BASE/f'ensemble_B_{split}_predictions.csv');assert np.array_equal(d.sample_id,ref.sample_id)
        base=rule1(d,ref.pred.to_numpy())
        for seconds in [5,10]:
            rows.append(dict(split=split,confirmation_seconds=seconds,**evaluate(d,debounce(d,base,seconds),base)))
    pd.DataFrame(rows).to_csv(OUT/'debounce.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
