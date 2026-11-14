"""Aggregate-only Validation comparison of existing B/B_gate checkpoints. No training."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import compare_attention_gate as common
from analyze_experiment_performance import confusion,scores


@torch.inference_mode()
def probabilities(model, loader):
    model.eval()
    return np.concatenate([model(batch).softmax(1).cpu().numpy() for batch in loader])


def run(args):
    if args.output.exists():raise FileExistsError('Choose new output directory')
    torch.set_num_threads(2)
    frame=common.read_frame(args.data_dir/'validation/validation.csv')
    reference=pd.read_csv(args.abc_run/'ensemble_B_validation_predictions.csv')
    if not frame.sample_id.is_unique or set(frame.sample_id)!=set(reference.sample_id):raise ValueError('IDs differ')
    frame=frame.set_index('sample_id').loc[reference.sample_id].reset_index()
    if not np.array_equal(frame.label,reference.label) or not np.array_equal(frame.experiment_id.astype(str),reference.experiment_id.astype(str)):
        raise ValueError('Targets/groups differ')
    groups=list(frame.groupby('experiment_id',sort=True).indices.values())
    y=frame.label.to_numpy(int)
    predictions={};metadata={};ensemble={}
    for mode in ['B','B_gate']:
        ps=[]
        for seed in [42,43,44]:
            path=args.run/f'{mode}_seed{seed}/best.pt'
            ck=torch.load(path,map_location='cpu',weights_only=False)
            if ck['mode']!=mode or ck['seed']!=seed:raise ValueError('Checkpoint mismatch')
            ds=common.Observations(frame,args.data_dir/'validation',ck['preprocess'],ck['image_size'])
            model=common.BGateModel(mode,ck['image_size'],ck['dim'])
            model.load_state_dict(ck['state_dict'])
            p=probabilities(model,common.loader(ds,SimpleNamespace(batch_size=32,workers=0)))
            ps.append(p)
            key=f'{mode}_seed{seed}';predictions[key]=p.argmax(1)+1
            metadata[key]={'best_epoch':ck['best_epoch'],'saved_macro_f1':ck['validation']['macro_f1'],
                'cpu_macro_f1':scores(confusion(y,predictions[key]))['macro_f1_6']}
            print('Evaluated',key,json.dumps(metadata[key]),flush=True)
        ensemble[mode]=np.mean(ps,axis=0).argmax(1)+1
        predictions[mode+'_ensemble']=ensemble[mode]
    for mode in ['A','B','C']:
        f=pd.read_csv(args.abc_run/f'ensemble_{mode}_validation_predictions.csv')
        if not f.sample_id.is_unique or set(f.sample_id)!=set(frame.sample_id):raise ValueError('ABC IDs differ')
        f=f.set_index('sample_id').loc[frame.sample_id].reset_index()
        if not np.array_equal(f.label,y):raise ValueError('ABC labels differ')
        predictions['ABC_'+mode]=f.pred.to_numpy(int)
    overall={k:scores(confusion(y,v)) for k,v in predictions.items()}
    details={};rows=[]
    for i,idx in enumerate(groups):
        name=f'E{i+1:02}';details[name]={};row={'experiment':name,'rows':len(idx)}
        for key,pred in predictions.items():
            c=confusion(y[idx],pred[idx]);s=scores(c)
            details[name][key]=dict(**s,errors=int(c.sum()-np.trace(c)))
            row.update({key+'_present_f1':s['macro_f1_true_present'],key+'_5_to_4':int(c[4,3]),key+'_errors':int(c.sum()-np.trace(c))})
        row['gate_minus_B_ensemble']=row['B_gate_ensemble_present_f1']-row['B_ensemble_present_f1']
        rows.append(row)
    pairs={}
    for seed in [42,43,44]:
        a=overall[f'B_seed{seed}']['macro_f1_6'];b=overall[f'B_gate_seed{seed}']['macro_f1_6']
        pairs[str(seed)]=b-a
    rng=np.random.default_rng(42);w=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    boot={}
    for name in ['B_ensemble','B_gate_ensemble']:
        cms=np.stack([np.asarray(details[f'E{i+1:02}'][name]['confusion']) for i in range(len(groups))])
        b=np.einsum('be,eij->bij',w,cms);den=b.sum(1)+b.sum(2)
        boot[name]=np.divide(2*np.diagonal(b,axis1=1,axis2=2),den,out=np.zeros_like(den,float),where=den>0).mean(1)
    result=dict(rows=len(frame),experiments=len(groups),overall=overall,checkpoint_metadata=metadata,
        seed_gate_minus_B=pairs,ensemble_gate_minus_B=overall['B_gate_ensemble']['macro_f1_6']-overall['B_ensemble']['macro_f1_6'],
        paired_experiment_bootstrap_95ci=np.quantile(boot['B_gate_ensemble']-boot['B_ensemble'],[.025,.975]).tolist(),
        experiment_details=details,notes=['Existing checkpoints only; no training/Test access or row-level exports.',
        'ABC models differ in architecture/inputs/loss/split; cross-family differences are descriptive, not gate effects.',
        'New B/B_gate ensembles use fixed equal seed probability averaging; original submissions used seed42.',
        'CPU reproduction may differ slightly from saved GPU metrics.','Experiment aliases match earlier ABC report.'])
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    pd.DataFrame(rows).to_csv(args.output/'experiment_comparison.csv',index=False)
    pd.DataFrame([{'model':k,'macro_f1':s['macro_f1_6'],**{f'F1_{i+1}':v for i,v in enumerate(s['class_f1'])}} for k,s in overall.items()]).to_csv(args.output/'overall_comparison.csv',index=False)
    print('OVERALL',json.dumps({k:s['macro_f1_6'] for k,s in overall.items()}),flush=True)
    print('ENSEMBLE_DELTA',result['ensemble_gate_minus_B'],'CI',result['paired_experiment_bootstrap_95ci'],flush=True)
    for e in ['E16','E14','E05']:
        print(e,json.dumps({k:{'present_f1':s['macro_f1_true_present'],'errors':s['errors'],'5_to_4':s['confusion'][4][3]} for k,s in details[e].items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('runs_b_gate_v1'))
    p.add_argument('--abc-run',type=Path,default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    p.add_argument('--data-dir',type=Path,default=Path('/home/ubuntu/data/battery_data_light'))
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
