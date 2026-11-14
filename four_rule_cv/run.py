"""Fixed four-rule comparison with fresh, experiment-held-out original B models."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold,GroupShuffleSplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import final_rules3 as rules
from abc_reference import abc_fusion_experiment as abc

NAMES=['rules3','plus_thermal_floor','plus_t1_upper','plus_both']
STATIC=['bat_class','bat_cc','soc_status','tr_method']
E16=rules.E16


def write(path,data):
    abc.common.write_json(path,data)


def prepare(frame):
    d=frame.copy();d['_pos']=np.arange(len(d));d=rules.prepare(d)
    d['thermal_max']=d.groupby('experiment_id').thermal_t.cummax()
    return d.sort_values('_pos').reset_index(drop=True)


def quantile_fit(d,mask,col,q):
    source=d.loc[mask&np.isfinite(d[col])]
    if source.empty:raise ValueError(f'No fitting support: {col}')
    by=source.groupby('bat_cc')[col].quantile(q).to_dict()
    support=source.groupby('bat_cc').experiment_id.nunique().to_dict()
    return dict(by_capacity=by,fallback=float(source[col].quantile(q)),support=support,q=q)


def fit_rules(frame):
    d=prepare(frame)
    return dict(floor3=quantile_fit(d,d.label>=3,'t1max',.01),
                floor6=quantile_fit(d,d.label>=6,'thermal_max',.01),
                upper3=quantile_fit(d,d.label<3,'t1max',.995),
                experiments=sorted(frame.experiment_id.unique().tolist()))


def bound(d,t):
    return d.bat_cc.map(t['by_capacity']).fillna(t['fallback']).to_numpy(float)


def predict_rules(frame,raw,thresholds):
    d=prepare(frame)
    floor=(thresholds['floor3']['by_capacity'],thresholds['floor3']['fallback'])
    base=rules.apply_rules(d,np.asarray(raw,int),floor)
    thermal=base.copy();low=bound(d,thresholds['floor6'])
    change=(base==6)&np.isfinite(d.thermal_max.to_numpy())&(d.thermal_max.to_numpy()<low)
    thermal[change]=5
    upper=base.copy();high=bound(d,thresholds['upper3'])
    change=(base<=2)&np.isfinite(d.t1max.to_numpy())&(d.t1max.to_numpy()>high)
    upper[change]=3
    both=thermal.copy();both[upper!=base]=upper[upper!=base]
    outputs=dict(zip(NAMES,[base,thermal,upper,both]))
    assert np.array_equal(both!=base,(thermal!=base)|(upper!=base))
    return outputs,d


def matrix(y,p):
    return np.bincount((np.asarray(y)-1)*6+np.asarray(p)-1,minlength=36).reshape(6,6)


def f1(c):
    den=c.sum(-1)+c.sum(-2)
    return np.divide(2*np.diagonal(c,axis1=-2,axis2=-1),den,out=np.zeros_like(den,dtype=float),where=den>0)


def stats(y,p,base):
    values=f1(matrix(y,p));b=f1(matrix(y,base))
    return dict(macro_f1=float(values.mean()),delta_f1=float(values.mean()-b.mean()),
                corrected=int(((base!=y)&(p==y)).sum()),harmed=int(((base==y)&(p!=y)).sum()),
                changed=int((p!=base).sum()),errors=int((p!=y).sum()),
                **{f'F1_{i+1}':float(v) for i,v in enumerate(values)})


def record_protocol(out,protocol,frame,outputs,diagnostic=None):
    folder=out/protocol;folder.mkdir(exist_ok=False)
    y=frame.label.to_numpy(int);base=outputs['rules3']
    per=[];summary=[];changes=[];conditions=[];directions=[]
    rows=frame[['sample_id','experiment_id','label']+STATIC].copy()
    if 'fold' in frame:rows['fold']=frame.fold
    for name,p in outputs.items():rows[name]=p
    rows.to_csv(folder/'predictions.csv',index=False)
    groups=list(frame.groupby('experiment_id',sort=True).indices.items())
    rng=np.random.default_rng(20261115)
    draws=rng.integers(0,len(groups),(2000,len(groups)))
    cb=np.stack([matrix(y[i],base[i]) for _,i in groups])
    for name,p in outputs.items():
        cp=np.stack([matrix(y[i],p[i]) for _,i in groups])
        delta=f1(cp[draws].sum(1)).mean(-1)-f1(cb[draws].sum(1)).mean(-1)
        lo,hi=np.quantile(delta,[.025,.975]);exprows=[]
        jack=[]
        for k,(exp,ids) in enumerate(groups):
            g=frame.iloc[ids];m=stats(y[ids],p[ids],base[ids]);rec=dict(method=name,experiment_id=exp,rows=len(ids),**m,
               **{c:float(g[c].iloc[0]) for c in STATIC})
            if 'fold' in frame:rec['fold']=int(g.fold.iloc[0])
            if diagnostic is not None:
                dd=diagnostic.iloc[ids]
                rec.update(thermal_max_peak=float(dd.thermal_max.max()),thermal_floor=float(dd.floor6.iloc[0]),
                    thermal_training_experiments=int(dd.floor6_support.iloc[0]),
                    thermal_fallback=bool(dd.floor6_fallback.iloc[0]),
                    true6_below_floor=int(((g.label.to_numpy()==6)&(dd.thermal_max.to_numpy()<dd.floor6.to_numpy())).sum()),
                    true2_above_upper=int(((g.label.to_numpy()==2)&(dd.t1max.to_numpy()>dd.upper3.to_numpy())).sum()))
            exprows.append(rec);per.append(rec)
            if len(groups)>1:
                jack.append(float(f1(cp.sum(0)-cp[k]).mean()-f1(cb.sum(0)-cb[k]).mean()))
        e=pd.DataFrame(exprows)
        totalcorrect=int(e.corrected.sum())
        summary.append(dict(method=name,experiments=len(groups),rows=len(frame),**stats(y,p,base),
             ci_lower=float(lo),ci_upper=float(hi),better_experiments=int((e.delta_f1>1e-12).sum()),
             worse_experiments=int((e.delta_f1< -1e-12).sum()),ties=int((e.delta_f1.abs()<=1e-12).sum()),
             experiments_with_any_harm=int((e.harmed>0).sum()),worst_experiment_delta=float(e.delta_f1.min()),
             best_experiment_delta=float(e.delta_f1.max()),largest_corrected_share=float(e.corrected.max()/totalcorrect) if totalcorrect else 0.,
             leave_one_experiment_out_min_delta=min(jack) if jack else 0.))
        for group_cols in [['bat_cc'],['bat_class'],['soc_status'],['tr_method'],['bat_class','tr_method']]:
            for key,ids in frame.groupby(group_cols,sort=True).indices.items():
                if not isinstance(key,tuple):key=(key,)
                conditions.append(dict(method=name,condition='+'.join(group_cols),value='/'.join(map(str,key)),
                    experiments=int(frame.iloc[ids].experiment_id.nunique()),rows=len(ids),**stats(y[ids],p[ids],base[ids])))
        for i in np.flatnonzero(p!=base):
            rec=dict(method=name,sample_id=frame.sample_id.iloc[i],experiment_id=frame.experiment_id.iloc[i],
                     true=int(y[i]),before=int(base[i]),after=int(p[i]),
                     corrected=bool(p[i]==y[i]),harmed=bool(base[i]==y[i]))
            if diagnostic is not None:
                dd=diagnostic.iloc[i]
                rec.update(t1max=float(dd.t1max),thermal_max=float(dd.thermal_max),
                    floor6=float(dd.floor6),upper3=float(dd.upper3),floor6_support=int(dd.floor6_support))
            changes.append(rec)
        for (a,b),n in pd.DataFrame(dict(true=y,pred=p)).groupby(['true','pred']).size().items():
            if a!=b:directions.append(dict(method=name,true=int(a),pred=int(b),count=int(n)))
    for filename,data in [('summary',summary),('per_experiment',per),('conditions',conditions),('changes',changes),('error_directions',directions)]:
        pd.DataFrame(data).to_csv(folder/f'{filename}.csv',index=False)
    return summary


def diagnostics(frame,thresholds):
    d=prepare(frame);t=thresholds['floor6']
    d['floor6']=bound(d,t);d['upper3']=bound(d,thresholds['upper3'])
    d['floor6_support']=d.bat_cc.map(t['support']).fillna(0).astype(int)
    d['floor6_fallback']=~d.bat_cc.isin(t['by_capacity'])
    return d


class View:
    def __init__(self,data,indices):
        self.data=data;self.ids=torch.as_tensor(indices,dtype=torch.long)
        self.frame=data.frame.iloc[indices].reset_index(drop=True)
    def batch(self,indices,device):return self.data.batch(self.ids[indices.cpu()],device)


def load_images(train,val,base,device):
    manifest=pd.read_csv(base/'split_manifest.csv');paths=json.loads((base/'image_cache_manifest.json').read_text())
    images=torch.empty((len(train),4,128,128),dtype=torch.uint8,device=device)
    locations=pd.Index(train.sample_id);seen=np.zeros(len(train),bool)
    for split in ['fit','dev']:
        ids=locations.get_indexer(manifest.loc[manifest.split==split,'sample_id'])
        if (ids<0).any() or seen[ids].any():raise ValueError('Cache ID mismatch')
        seen[ids]=True;a=np.array(np.load(paths[split],mmap_mode='r'),copy=True)
        images.index_copy_(0,torch.tensor(ids,device=device),torch.from_numpy(a).to(device));del a
    assert seen.all()
    assert np.array_equal(val.sample_id,manifest.loc[manifest.split=='validation','sample_id'])
    vi=torch.from_numpy(np.array(np.load(paths['validation'],mmap_mode='r'),copy=True)).to(device)
    return images,vi


def data_for(frame,images,prep,device):
    q,gas=prep.transform(frame)
    return abc.Data(frame,images,torch.from_numpy(q).to(device),torch.from_numpy(gas).to(device),
                    torch.tensor(frame.label.to_numpy(int)-1,device=device))


def train_seed(data,fit_idx,select_idx,prep,args,fold,seed,folder,device):
    abc.common.seed_everything(seed)
    model=abc.FusionModel('B').to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(prep.weights,device=device))
    generator=torch.Generator().manual_seed(seed);selection=View(data,select_idx)
    fit=torch.tensor(fit_idx,dtype=torch.long);best=-1.;state=None;best_epoch=0;history=[]
    started=time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train();order=fit[torch.randperm(len(fit),generator=generator)];total=0.
        for start in range(0,len(order),64):
            ids=order[start:start+64].to(device);opt.zero_grad(set_to_none=True)
            loss=criterion(model(*data.batch(ids,device)),data.y.index_select(0,ids))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            total+=float(loss.detach())*len(ids)
        prob=abc.predict(model,selection,None,device);score=float(f1(matrix(selection.frame.label,prob.argmax(1)+1)).mean())
        if score>best:
            best=score;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(dict(epoch=epoch,train_loss=total/len(fit),selection_f1=score,best_epoch=best_epoch))
        if epoch==1 or epoch%5==0:print(f'fold={fold} seed={seed} epoch={epoch} inner={score:.5f} best={best:.5f} seconds={time.monotonic()-started:.1f}',flush=True)
    model.load_state_dict(state)
    torch.save(dict(state=state,fold=fold,seed=seed,epoch=best_epoch,inner_f1=best,preprocessing=prep.metadata()),folder/f'seed{seed}.pt')
    pd.DataFrame(history).to_csv(folder/f'seed{seed}_history.csv',index=False)
    return model,best_epoch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=Path('/home/ubuntu/data/battery_data_light'))
    parser.add_argument('--base-run',type=Path,default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--epochs',type=int,default=15)
    parser.add_argument('--split-seed',type=int,default=20261115)
    parser.add_argument('--saved-only',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'config.json',{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
      'recipes':dict(rules3='45C, voltage .8, capacity T1 floor q.01',thermal_floor_q=.01,t1_upper_q=.995),
      'test_access':False,'threshold_search':False,'candidate_selection':'Compare fixed variants; do not replace submission automatically',
      'limitations':'Previously inspected data; fixed-hypothesis outer evaluation, not new blind data'})
    tr=pd.read_csv(args.data_root/'train/train.csv');val=pd.read_csv(args.data_root/'validation/validation.csv')
    assert not set(tr.experiment_id)&set(val.experiment_id)
    fi,di=next(GroupShuffleSplit(1,test_size=.2,random_state=20261114).split(tr,groups=tr.experiment_id))
    threshold=fit_rules(tr.iloc[fi]);write(args.output/'saved_fit_thresholds.json',threshold)
    for split,frame in [('dev',tr.iloc[di].reset_index(drop=True)),('validation',val)]:
        ref=pd.read_csv(args.base_run/f'ensemble_B_{split}_predictions.csv')
        assert np.array_equal(ref.sample_id,frame.sample_id) and np.array_equal(ref.label,frame.label)
        outputs,_=predict_rules(frame,ref.pred.to_numpy(int),threshold);d=diagnostics(frame,threshold)
        record_protocol(args.output,f'saved_{split}',frame,outputs,d)
        if split=='validation':
            ids=np.flatnonzero(frame.experiment_id.to_numpy()!=E16)
            record_protocol(args.output,'saved_validation_noE16',frame.iloc[ids].reset_index(drop=True),{n:p[ids] for n,p in outputs.items()},d.iloc[ids].reset_index(drop=True))
    if args.saved_only:return
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    train_images,val_images=load_images(tr,val,args.base_run,device)
    folds=list(GroupKFold(args.folds,shuffle=True,random_state=args.split_seed).split(tr,groups=tr.experiment_id))
    oof=np.full((len(tr),6),np.nan);assignment=np.full(len(tr),-1);oof_rules={n:np.zeros(len(tr),int) for n in NAMES};diag=[]
    validation_prob=np.zeros((len(val),6));fold_details=[]
    for k,(outer,holdout) in enumerate(folds):
        folder=args.output/f'fold_{k}';folder.mkdir()
        a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=args.split_seed+k).split(outer,groups=tr.iloc[outer].experiment_id))
        fit_idx=outer[a];selection_idx=outer[b]
        group_sets=[set(tr.iloc[i].experiment_id) for i in [fit_idx,selection_idx,holdout]]
        assert not any(group_sets[i]&group_sets[j] for i in range(3) for j in range(i+1,3))
        thresholds=fit_rules(tr.iloc[outer]);assert not set(thresholds['experiments'])&group_sets[2]
        write(folder/'thresholds.json',thresholds)
        assignments=pd.concat([tr.iloc[i][['sample_id','experiment_id']].assign(role=role) for role,i in [('fit',fit_idx),('selection',selection_idx),('holdout',holdout)]])
        assignments.to_csv(folder/'split_manifest.csv',index=False)
        prep=abc.SensorPreprocessor(tr.iloc[fit_idx]);write(folder/'preprocessing.json',prep.metadata())
        data=data_for(tr,train_images,prep,device);vdata=data_for(val,val_images,prep,device);view=View(data,holdout)
        prob=np.zeros((len(holdout),6))
        for seed in args.seeds:
            model,epoch=train_seed(data,fit_idx,selection_idx,prep,args,k,seed,folder,device)
            hp=abc.predict(model,view,None,device);vp=abc.predict(model,vdata,None,device)
            np.savez(folder/f'seed{seed}_probs.npz',holdout=hp,validation=vp)
            prob+=hp/len(args.seeds);validation_prob+=vp/(args.folds*len(args.seeds));del model
            write(args.output/'status.json',dict(fold=k,completed_seed=seed,total_folds=args.folds))
        oof[holdout]=prob;assignment[holdout]=k
        frame=tr.iloc[holdout].reset_index(drop=True);outputs,_=predict_rules(frame,prob.argmax(1)+1,thresholds)
        for n,p in outputs.items():oof_rules[n][holdout]=p
        d=diagnostics(frame,thresholds);d['global_row']=holdout;diag.append(d)
        fold_details.extend(dict(fold=k,method=n,**stats(frame.label.to_numpy(int),p,outputs['rules3'])) for n,p in outputs.items())
        pd.DataFrame(fold_details).to_csv(args.output/'fold_scores.csv',index=False)
        print(f'FOLD {k} COMPLETE '+json.dumps({n:round(stats(frame.label.to_numpy(int),p,outputs["rules3"])["macro_f1"],5) for n,p in outputs.items()}),flush=True)
        del data,vdata
    assert np.isfinite(oof).all() and (assignment>=0).all()
    np.savez(args.output/'probs.npz',train_oof=oof,validation=validation_prob)
    frame=tr.copy();frame['fold']=assignment;d=pd.concat(diag).sort_values('global_row').reset_index(drop=True)
    summaries=record_protocol(args.output,'train_oof',frame,oof_rules,d)
    vt=fit_rules(tr);write(args.output/'validation_thresholds_train78.json',vt)
    vp,_=predict_rules(val,validation_prob.argmax(1)+1,vt);vd=diagnostics(val,vt)
    record_protocol(args.output,'validation_fold_ensemble',val,vp,vd)
    ids=np.flatnonzero(val.experiment_id.to_numpy()!=E16)
    record_protocol(args.output,'validation_fold_ensemble_noE16',val.iloc[ids].reset_index(drop=True),{n:p[ids] for n,p in vp.items()},vd.iloc[ids].reset_index(drop=True))
    selection=[]
    for s in summaries:
        if s['method']=='rules3':continue
        passed=s['delta_f1']>0 and s['ci_lower']>0 and s['better_experiments']>s['worse_experiments'] and s['leave_one_experiment_out_min_delta']>0
        selection.append(dict(method=s['method'],passes_fixed_internal_checks=bool(passed),
          note='Internal criterion only; does not establish private improvement. Baseline submission is not replaced.'))
    write(args.output/'decision.json',dict(primary_comparison='train_oof',checks=selection,submission_replaced=False,
      criteria=['positive pooled delta','positive paired bootstrap lower bound','more improved than worsened experiments','positive delta after removing any one experiment']))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'final_rules3.py',args.data_root/'train/train.csv',args.data_root/'validation/validation.csv']}
    write(args.output/'verification.json',dict(fresh_backbone_training=True,heldout_experiments_excluded_from_all_training=True,
         seeds=args.seeds,folds=args.folds,epochs=args.epochs,test_access=False,source_sha256=hashes))
    write(args.output/'status.json',dict(complete=True,models=args.folds*len(args.seeds)))
    print('TRAIN OOF',pd.DataFrame(summaries)[['method','macro_f1','delta_f1','corrected','harmed','better_experiments','worse_experiments','ci_lower','ci_upper']].to_string(index=False),flush=True)

if __name__=='__main__':main()
