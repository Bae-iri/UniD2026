"""Summarize completed four-rule CV, including fitting support and seed sensitivity."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from four_rule_cv.run import fit_rules,predict_rules,stats,NAMES


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
    if not json.loads((root/'status.json').read_text()).get('complete'):raise RuntimeError('CV not complete')
    config=json.loads((root/'config.json').read_text());tr=pd.read_csv(Path(config['data_root'])/'train/train.csv')
    oof=pd.read_csv(root/'train_oof/predictions.csv');rows=[];seedrows=[]
    for fold in range(config['folds']):
        f=root/f'fold_{fold}';thr=json.loads((f/'thresholds.json').read_text())
        # JSON object keys are strings; restore numeric capacity keys used by pandas map.
        for key in ['floor3','floor6','upper3']:
            thr[key]['by_capacity']={float(k):v for k,v in thr[key]['by_capacity'].items()}
            thr[key]['support']={float(k):v for k,v in thr[key]['support'].items()}
            for cc,v in thr[key]['by_capacity'].items():
                rows.append(dict(fold=fold,rule=key,capacity=cc,value=v,support_experiments=thr[key]['support'][cc]))
        ids=np.flatnonzero(oof.fold.to_numpy()==fold);frame=tr.iloc[ids].reset_index(drop=True)
        split_manifest=pd.read_csv(f/'split_manifest.csv')
        assert np.array_equal(frame.sample_id,split_manifest.loc[split_manifest.role=='holdout','sample_id'])
        for seed in config['seeds']:
            z=np.load(f/f'seed{seed}_probs.npz');prob=z['holdout']
            # GroupKFold holdout indices are ascending, same order as saved OOF assignment.
            outputs,_=predict_rules(frame,prob.argmax(1)+1,thr)
            for n,pred in outputs.items():
                for sid,y,b,new in zip(frame.sample_id,frame.label,outputs['rules3'],pred):
                    seedrows.append((seed,n,sid,int(y),int(b),int(new)))
    t=pd.DataFrame(rows);t.to_csv(root/'fold_thresholds.csv',index=False)
    t.groupby(['rule','capacity']).agg(min_threshold=('value','min'),max_threshold=('value','max'),
       min_training_experiments=('support_experiments','min'),max_training_experiments=('support_experiments','max')).to_csv(root/'threshold_ranges.csv')
    s=pd.DataFrame(seedrows,columns=['seed','method','sample_id','true','base','pred']);aggregate=[]
    for (seed,n),d in s.groupby(['seed','method']):aggregate.append(dict(seed=seed,method=n,**stats(d.true.to_numpy(),d.pred.to_numpy(),d.base.to_numpy())))
    pd.DataFrame(aggregate).to_csv(root/'individual_seed_summary.csv',index=False)
    failures=pd.read_csv(root/'train_oof/per_experiment.csv');failures=failures[(failures.method!='rules3')&((failures.delta_f1< -1e-12)|(failures.harmed>0))]
    failures.sort_values(['method','delta_f1']).to_csv(root/'failure_cases.csv',index=False)
    print('OOF summary');print(pd.read_csv(root/'train_oof/summary.csv').round(6).to_string(index=False))
    print('Failures');print(failures.sort_values('delta_f1')[['method','experiment_id','bat_cc','corrected','harmed','delta_f1','thermal_floor','thermal_max_peak','thermal_training_experiments','true2_above_upper']].head(20).round(6).to_string(index=False))
    print('Threshold ranges');print(pd.read_csv(root/'threshold_ranges.csv').round(3).to_string(index=False))

if __name__=='__main__':main()
