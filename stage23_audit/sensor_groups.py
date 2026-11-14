"""Fixed-capacity sensor-group probes for stage 2/3. No MLP or Test access."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from analyze import DATA,BASE,OUT,SENS,STATIC,features,rule1,evaluate


def classifier():
    return HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=40,
                l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=42)


def train_model(d,cols):
    w=1/d.groupby(['experiment_id','label']).label.transform('size').to_numpy(float)
    w=w/w.mean()
    model=classifier();model.fit(d[cols],(d.label==3).astype(int),sample_weight=w)
    return model


def main():
    train=pd.read_csv(DATA/'train/train.csv');fit_i,dev_i=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261114).split(train,groups=train.experiment_id))
    fit=features(train.iloc[fit_i].reset_index(drop=True));full=features(train);full=full[full.label.isin([2,3])].reset_index(drop=True)
    history=SENS+[c for c in fit if c.endswith(('_max','_delta_start','_rate30','_rate60','_minus_ambient'))]+['t1_minus_t2','since45']
    groups={'temperatures':['meas_t1','meas_t2','thermal_t','atmp_t','indu_t'],
            'temperatures_voltage_pressure':['meas_t1','meas_t2','thermal_t','atmp_t','indu_t','meas_v','meas_p'],
            'temperatures_chemical_gas':['meas_t1','meas_t2','thermal_t','atmp_t','indu_t']+[c for c in SENS if c.startswith('meas_') and c not in ['meas_t1','meas_t2','meas_v','meas_p']],
            'current_sensors':SENS+['atmp_t'],
            'causal_history':list(dict.fromkeys(history)),
            'history_and_conditions':list(dict.fromkeys(history+STATIC))}
    folds=list(GroupKFold(5).split(full,groups=full.experiment_id));cv=[];scores=[];imp=[]
    with threadpool_limits(2):
        for name,cols in groups.items():
            for k,(a,b) in enumerate(folds):
                tr=full.iloc[a];te=full.iloc[b];model=train_model(tr,cols);p=model.predict(te[cols])+2
                cv.append(dict(features=name,fold=k,macro23=f1_score(te.label,p,labels=[2,3],average='macro'),
                               accuracy=float((p==te.label).mean()),rows=len(te)))
            model=train_model(fit[fit.label.isin([2,3])],cols)
            for split,d in [('dev',features(train.iloc[dev_i].reset_index(drop=True))),('validation',features(pd.read_csv(DATA/'validation/validation.csv')))]:
                ref=pd.read_csv(BASE/f'ensemble_B_{split}_predictions.csv');assert np.array_equal(ref.sample_id,d.sample_id)
                bp=rule1(d,ref.pred.to_numpy(int));p=bp.copy();mask=np.isin(bp,[2,3]);p[mask]=model.predict(d.loc[mask,cols])+2
                scores.append(dict(features=name,split=split,**evaluate(d,p,bp)))
                if name=='causal_history':
                    # Permutation is descriptive heldout sensitivity, not physical causality.
                    keep=d.label.isin([2,3]);r=permutation_importance(model,d.loc[keep,cols],(d.loc[keep,'label']==3).astype(int),
                        scoring='balanced_accuracy',n_repeats=3,random_state=42)
                    imp.extend(dict(split=split,feature=c,importance=float(v)) for c,v in zip(cols,r.importances_mean))
    pd.DataFrame(cv).to_csv(OUT/'sensor_groups_train_groupcv.csv',index=False)
    pd.DataFrame(scores).to_csv(OUT/'sensor_groups_holdout.csv',index=False)
    pd.DataFrame(imp).to_csv(OUT/'sensor_groups_permutation.csv',index=False)
    print('TRAIN GROUPCV',pd.DataFrame(cv).groupby('features').macro23.agg(['mean','min','max']).round(5).to_string())
    print('HOLDOUT',pd.DataFrame(scores).round(6).to_string(index=False))
    print('IMPORTANCE',pd.DataFrame(imp).sort_values('importance',ascending=False).groupby('split',sort=False).head(8).to_string(index=False))

if __name__=='__main__':main()
