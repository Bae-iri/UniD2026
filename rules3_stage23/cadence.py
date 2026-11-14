"""Hold per-frame B probabilities fixed; thin the observable history in all phases."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from rules3_stage23.evaluate import DATA,BASE,ROOT,rules,odds_table,variants,metrics


def main():
    out=ROOT/'rules3_stage23/results_v2'
    tr=pd.read_csv(DATA/'train/train.csv');fi,di=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261114).split(tr,groups=tr.experiment_id))
    fit=tr.iloc[fi];floor=rules.t1_floor(fit);tab=odds_table(fit);f=rules.prepare(fit)
    upper=f[f.label<3].groupby('bat_cc').t1max.quantile(.995).to_dict();rows=[]
    for split,frame in [('dev',tr.iloc[di]),('validation',pd.read_csv(DATA/'validation/validation.csv'))]:
        pr=pd.read_csv(BASE/f'ensemble_B_{split}_predictions.csv')
        d=rules.prepare(frame.merge(pr[['sample_id','pred']+[f'p{i}' for i in range(1,7)]],on='sample_id',validate='1:1'))
        for step in [1,2,3]:
            ds=[];bs=[];ps=[]
            for phase in range(step):
                ids=np.concatenate([g[phase::step] for g in d.groupby('experiment_id').indices.values()])
                dd=rules.prepare(d.iloc[ids].reset_index(drop=True))
                b=rules.apply_rules(dd,dd.pred.to_numpy(int),floor)
                p=variants(dd,b,floor,tab,upper)['latch10_plus_upper_q995']
                ds.append(dd);bs.append(b);ps.append(p)
            df=pd.concat(ds,ignore_index=True);b=np.concatenate(bs);p=np.concatenate(ps)
            for scope,mask in [('all',np.ones(len(df),bool)),('noE16',(df.experiment_id!=rules.E16).to_numpy())]:
                if split=='dev' and scope!='all':continue
                rows.append(dict(split=split,scope=scope,nominal_seconds=step*5,**metrics(df.loc[mask],p[mask],b[mask])))
    pd.DataFrame(rows).to_csv(out/'combined_cadence_stress.csv',index=False)
    print(pd.DataFrame(rows)[['split','scope','nominal_seconds','macro_f1','corrected','harmed']].round(6).to_string(index=False))

if __name__=='__main__':main()
