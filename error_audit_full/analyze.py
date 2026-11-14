"""Descriptive audit of frozen rules3 predictions; no Test access or tuning."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import final_rules3 as rules
OUT=Path(__file__).resolve().parent
DATA=Path('/home/ubuntu/data/battery_data_light')
BASE=Path('/home/ubuntu/leewon/abc_runs/current_query_v1')

def score(y,p):
    return f1_score(y,p,labels=list(range(1,7)),average='macro',zero_division=0)

def main():
    tables={k:[] for k in ['stages','directions','experiments','boundaries','rule_steps','confidence','conditions','sensors','summary']}
    floor_raw=json.loads((ROOT/'rules3_stage23/results_v2/fit_thresholds.json').read_text())['floor']
    floor=({float(k):v for k,v in floor_raw[0].items()},floor_raw[1])
    all_rows=[]
    for split in ['dev','validation']:
        source=pd.read_csv(DATA/('train/train.csv' if split=='dev' else 'validation/validation.csv'))
        pr=pd.read_csv(BASE/f'ensemble_B_{split}_predictions.csv')
        d=rules.prepare(source.merge(pr[['sample_id','pred']+[f'p{i}' for i in range(1,7)]],on='sample_id',validate='1:1'))
        d['baseline']=rules.apply_rules(d,d.pred.to_numpy(int),floor)
        frozen=pd.read_csv(ROOT/f'rules3_stage23/results_v2/{split}_predictions.csv').set_index('sample_id')
        assert np.array_equal(d.baseline,frozen.loc[d.sample_id,'rules3'])
        assert np.array_equal(d.label,frozen.loc[d.sample_id,'label'])
        y=d.label.to_numpy(int);raw=d.pred.to_numpy(int);p=d.baseline.to_numpy(int)
        prob=d[[f'p{i}' for i in range(1,7)]].to_numpy()
        d['raw_confidence']=prob.max(1)
        d['true_rank']=1+(prob>prob[np.arange(len(d)),y-1,None]).sum(1)
        d['raw_margin']=np.sort(prob,axis=1)[:,-1]-np.sort(prob,axis=1)[:,-2]
        d['seed_true_votes']=0;d['seed_distinct']=0
        seedpred=[]
        for seed in [42,43,44]:
            sp=pd.read_csv(BASE/f'seed_{seed}/B/{split}_predictions.csv').set_index('sample_id').loc[d.sample_id,'pred'].to_numpy()
            seedpred.append(sp);d['seed_true_votes']+=(sp==y).astype(int)
        d['seed_distinct']=[len(set(x)) for x in zip(*seedpred)]
        d['nearest_transition_s']=np.nan;d['nearest_transition']='none';d['previously_predicted_true']=False
        for exp,ids in d.groupby('experiment_id').indices.items():
            g=d.iloc[ids];t=g.relative_time_s.to_numpy();yy=g.label.to_numpy();pp=g.baseline.to_numpy()
            j=np.flatnonzero(yy[1:]!=yy[:-1])+1
            if len(j):
                dist=np.abs(t[:,None]-t[j][None,:]);near=dist.argmin(1)
                d.loc[ids,'nearest_transition_s']=dist.min(1)
                d.loc[ids,'nearest_transition']=[f'{yy[j[z]-1]}->{yy[j[z]]}' for z in near]
            for k in range(1,7):
                seen=pd.Series(pp==k).cummax().shift(fill_value=False).to_numpy()
                d.loc[ids[yy==k],'previously_predicted_true']=seen[yy==k]
            for a,b in zip(j,zip(yy[j-1],yy[j])):
                old,new=b;onset=t[a];ix=np.flatnonzero(pp==new)
                tables['boundaries'].append(dict(split=split,experiment_id=exp,old=int(old),new=int(new),onset_s=float(onset),first_prediction_offset_s=float(t[ix[0]]-onset) if len(ix) else np.nan,missing_prediction=not len(ix)))
        # Sequential attribution: compare each rule with the immediately previous output.
        steps=[('raw_B',raw.copy())];q=raw.copy();hot=d.t1max.to_numpy()>=45
        q[(q==1)&hot]=2;q[(q>=2)&~hot]=1;steps.append(('temperature45',q.copy()))
        q[(q>=4)&~d.vdrop.to_numpy()]=3;steps.append(('voltage80',q.copy()))
        bound=d.bat_cc.map(floor[0]).fillna(floor[1]).to_numpy();q[(q==3)&(d.t1max.to_numpy()<bound)]=2;steps.append(('capacity_floor',q.copy()))
        assert np.array_equal(q,p)
        scopes=[('all',np.ones(len(d),bool))]+([('noE16',(d.experiment_id!=rules.E16).to_numpy())] if split=='validation' else [])
        for scope,mask in scopes:
            z=d.loc[mask];yy=y[mask];pp=p[mask];rr=raw[mask];err=yy!=pp;base_score=score(yy,pp)
            c=confusion_matrix(yy,pp,labels=range(1,7))
            pd.DataFrame(c,index=range(1,7),columns=range(1,7)).to_csv(OUT/f'confusion_{split}_{scope}.csv',index_label='true_pred')
            tables['summary'].append(dict(split=split,scope=scope,rows=len(z),experiments=z.experiment_id.nunique(),errors=int(err.sum()),accuracy=float((~err).mean()),macro_f1=base_score,raw_macro_f1=score(yy,rr),errors_from_unchanged_raw=int((err&(pp==rr)).sum()),raw_correct_but_rules_wrong=int((err&(rr==yy)).sum()),raw_wrong_rules_correct=int(((~err)&(rr!=yy)).sum()),changed_wrong_to_other_wrong=int((err&(rr!=yy)&(rr!=pp)).sum())))
            for k in range(1,7):
                tp=c[k-1,k-1];fn=c[k-1].sum()-tp;fp=c[:,k-1].sum()-tp;f=2*tp/(2*tp+fn+fp) if 2*tp+fn+fp else 0
                tables['stages'].append(dict(split=split,scope=scope,stage=k,support=int(tp+fn),tp=int(tp),fn=int(fn),fp=int(fp),precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn) if tp+fn else 0,f1=f,macro_deficit_pp=(1-f)*100/6))
            for a,b in zip(*np.where(c>0)):
                if a==b:continue
                m=(yy==a+1)&(pp==b+1);g=z.loc[m];fix=pp.copy();fix[m]=yy[m]
                tables['directions'].append(dict(split=split,scope=scope,true=a+1,pred=b+1,errors=int(m.sum()),experiments=g.experiment_id.nunique(),top_experiment=g.experiment_id.value_counts().index[0],top_experiment_errors=int(g.experiment_id.value_counts().iloc[0]),within30=int((g.nearest_transition_s<=30).sum()),within60=int((g.nearest_transition_s<=60).sum()),beyond120=int((g.nearest_transition_s>120).sum()),raw_already_same_wrong=int((g.pred==g.baseline).sum()),rules_broke_correct=int((g.pred==g.label).sum()),true_in_raw_top2=int((g.true_rank<=2).sum()),raw_confident90=int((g.raw_confidence>=.9).sum()),three_seeds_wrong=int((g.seed_true_votes==0).sum()),previously_predicted_true=int(g.previously_predicted_true.sum()),oracle_gain_pp=(score(yy,fix)-base_score)*100))
            for exp,g in z.groupby('experiment_id'):
                ec=int((g.label!=g.baseline).sum());counts=g.loc[g.label!=g.baseline].groupby(['label','baseline']).size().sort_values(ascending=False)
                remove=z.experiment_id!=exp
                tables['experiments'].append(dict(split=split,scope=scope,experiment_id=exp,rows=len(g),errors=ec,accuracy=float((g.label==g.baseline).mean()),largest_direction=f'{counts.index[0][0]}->{counts.index[0][1]}' if len(counts) else '',largest_direction_errors=int(counts.iloc[0]) if len(counts) else 0,macro_without_experiment=score(yy[remove],pp[remove]),**{k:g[k].iloc[0] for k in ['bat_class','bat_cc','soc_status','tr_method']}))
            prev=rr
            for name,v in steps:
                vv=v[mask]
                tables['rule_steps'].append(dict(split=split,scope=scope,step=name,macro_f1=score(yy,vv),errors=int((yy!=vv).sum()),changed=int((vv!=prev).sum()),corrected=int(((prev!=yy)&(vv==yy)).sum()),harmed=int(((prev==yy)&(vv!=yy)).sum())))
                prev=vv
            for col in ['bat_class','bat_cc','soc_status','tr_method']:
                for value,g in z.groupby(col):
                    tables['conditions'].append(dict(split=split,scope=scope,condition=col,value=value,rows=len(g),experiments=g.experiment_id.nunique(),errors=int((g.label!=g.baseline).sum()),error_rate=float((g.label!=g.baseline).mean()),macro_f1=score(g.label,g.baseline)))
            for correct,g in z.groupby(z.label==z.baseline):
                tables['confidence'].append(dict(split=split,scope=scope,correct=bool(correct),rows=len(g),mean_raw_confidence=g.raw_confidence.mean(),mean_margin=g.raw_margin.mean(),true_top2=float((g.true_rank<=2).mean()),all_seeds_wrong=float((g.seed_true_votes==0).mean())))
        # Within true class only, descriptive sensor comparisons (no causal claims).
        for (label,pred),g in d.groupby(['label','baseline']):
            ref=d[(d.label==label)&(d.baseline==label)]
            if label==pred or not len(ref):continue
            for col in ['meas_t1','t1max','meas_t2','thermal_t','meas_v','meas_p','relative_time_s']:
                tables['sensors'].append(dict(split=split,true=label,pred=pred,sensor=col,error_rows=len(g),correct_rows=len(ref),error_median=g[col].median(),correct_median=ref[col].median(),error_q10=g[col].quantile(.1),error_q90=g[col].quantile(.9),correct_q10=ref[col].quantile(.1),correct_q90=ref[col].quantile(.9)))
        d['split']=split;all_rows.append(d)
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    pd.concat(all_rows,ignore_index=True).to_csv(OUT/'row_diagnostics.csv',index=False)
    # Reconcile all counts and the Macro F1 formula independently.
    for r in tables['summary']:
        st=[x for x in tables['stages'] if x['split']==r['split'] and x['scope']==r['scope']]
        dr=[x for x in tables['directions'] if x['split']==r['split'] and x['scope']==r['scope']]
        assert sum(x['fn'] for x in st)==r['errors']==sum(x['errors'] for x in dr)
        assert abs(np.mean([x['f1'] for x in st])-r['macro_f1'])<1e-12
    print(pd.DataFrame(tables['summary']).to_string(index=False))
    print(pd.DataFrame(tables['directions']).sort_values(['split','scope','oracle_gain_pp'],ascending=[True,True,False]).to_string(index=False))

if __name__=='__main__':main()
