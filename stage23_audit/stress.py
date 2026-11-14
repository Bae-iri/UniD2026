"""Fixed 10-second causal rule: seed/cadence checks, and threshold contradictions."""
import json
import numpy as np
import pandas as pd
from analyze import BASE,DATA,OUT,STATIC,features,rule1,latch,evaluate
from sklearn.model_selection import GroupShuffleSplit


def main():
    train=pd.read_csv(DATA/'train/train.csv');val=pd.read_csv(DATA/'validation/validation.csv')
    _,dev=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261114).split(train,groups=train.experiment_id))
    frames={'dev':features(train.iloc[dev].reset_index(drop=True)),'validation':features(val)}
    rows=[];cadence=[]
    for split,d in frames.items():
        for seed in [42,43,44]:
            ref=pd.read_csv(BASE/f'seed_{seed}/B/{split}_predictions.csv')
            assert np.array_equal(ref.sample_id,d.sample_id)
            for c in [f'p{i}' for i in range(1,7)]:d[c]=ref[c].to_numpy()
            base=rule1(d,ref.pred.to_numpy(int));pred=latch(d,base,10)
            rows.append(dict(split=split,seed=seed,**evaluate(d,pred,base)))
        ref=pd.read_csv(BASE/f'ensemble_B_{split}_predictions.csv')
        for c in [f'p{i}' for i in range(1,7)]:d[c]=ref[c].to_numpy()
        for step in [1,2,3]:
            combined=[];combined_pred=[];combined_base=[]
            for phase in range(step):
                indexes=[]
                for ids in d.groupby('experiment_id').indices.values():
                    ids=ids[np.argsort(d.relative_time_s.to_numpy()[ids])];indexes.extend(ids[phase::step])
                sub=features(d.iloc[indexes].reset_index(drop=True))
                base=rule1(sub,sub[[f'p{i}' for i in range(1,7)]].to_numpy().argmax(1)+1)
                pred=latch(sub,base,10)
                combined.append(sub);combined_pred.extend(pred);combined_base.extend(base)
            allrows=pd.concat(combined,ignore_index=True)
            cadence.append(dict(split=split,nominal_cadence_s=step*5,**evaluate(allrows,np.array(combined_pred),np.array(combined_base))))
    pd.DataFrame(rows).to_csv(OUT/'seed_stress.csv',index=False)
    pd.DataFrame(cadence).to_csv(OUT/'cadence_stress.csv',index=False)
    b=pd.read_csv(OUT/'boundary_values.csv');contradictions=[]
    for keys,g in b.groupby(STATIC):
        contradictions.append(dict(**dict(zip(STATIC,keys)),experiments=len(g),
          last2_runningmax_max=float(g.meas_t1_max_before.max()),first3_runningmax_min=float(g.meas_t1_max.min()),
          common_T1_threshold_possible=bool(g.meas_t1_max_before.max()<g.meas_t1_max.min())))
    pd.DataFrame(contradictions).to_csv(OUT/'condition_threshold_feasibility.csv',index=False)
    print('SEEDS',pd.DataFrame(rows).round(6).to_string(index=False))
    print('CADENCE',pd.DataFrame(cadence).round(6).to_string(index=False))
    print('CONDITIONS',pd.DataFrame(contradictions).to_string(index=False))

if __name__=='__main__':main()
