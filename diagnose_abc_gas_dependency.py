"""Aggregate-only Validation intervention: zero both gas feature paths. No training."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from abc_reference import abc_fusion_experiment as abc
from analyze_experiment_performance import confusion,scores


def restore_preprocessor(run):
    meta=json.loads((run/'preprocessing.json').read_text())
    if meta['electro']!=abc.ELECTRO or meta['gas']!=abc.GAS:raise ValueError('Feature schema differs')
    prep=abc.SensorPreprocessor.__new__(abc.SensorPreprocessor)
    for name in ['median','mean','std']:setattr(prep,name,pd.Series(meta[name],index=abc.NUMERIC))
    prep.weights=np.asarray(meta['class_weights'],np.float32)
    return prep


def validation_data(run,root):
    frame=pd.read_csv(root/'validation/validation.csv')
    ref=pd.read_csv(run/'ensemble_B_validation_predictions.csv')
    if not frame.sample_id.is_unique or not np.array_equal(frame.sample_id,ref.sample_id) or not np.array_equal(frame.label,ref.label):
        raise ValueError('Cache/reference row alignment differs')
    if not np.array_equal(frame.experiment_id,ref.experiment_id):raise ValueError('Experiment alignment differs')
    manifest=json.loads((run/'image_cache_manifest.json').read_text())
    cache=Path(manifest['validation'])
    images=np.load(cache,mmap_mode='c')
    if images.shape!=(len(frame),4,128,128):raise ValueError('Cache shape differs')
    q,gas=restore_preprocessor(run).transform(frame)
    return abc.Data(frame,torch.from_numpy(images),torch.from_numpy(q),torch.from_numpy(gas),torch.tensor(frame.label.to_numpy(int)-1)),ref


@torch.inference_mode()
def evaluate_pair(model,data,device):
    captured={}
    def capture(module,inputs):captured['features']=inputs[0]
    handle=model.head.register_forward_pre_hook(capture)
    normal=[];disabled=[]
    model.eval()
    try:
        for start in range(0,len(data.frame),128):
            ids=torch.arange(start,min(start+128,len(data.frame)))
            logits=model(*data.batch(ids,device))
            f=captured['features'].clone();d=f.shape[1]//5
            f[:,d:2*d]=0;f[:,3*d:4*d]=0
            off=model.head(f)
            normal.append(logits.softmax(1).cpu().numpy());disabled.append(off.softmax(1).cpu().numpy())
    finally:handle.remove()
    return np.concatenate(normal),np.concatenate(disabled)


def run(args):
    if args.output.exists():raise FileExistsError('Use new output directory')
    torch.set_num_threads(2)
    data,ref=validation_data(args.base_run,args.data_dir)
    frame=data.frame;y=frame.label.to_numpy(int)
    groups=list(frame.groupby('experiment_id',sort=True).indices.values())
    subsets={'all':np.ones(len(frame),bool)}
    # Pressure is part of the model gas branch but is excluded from concentration-zero definition.
    gas=frame[[c for c in abc.GAS if c!='meas_p']].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    subsets['all_10_gas_finite_zero']=np.isfinite(gas).all(1)&(gas==0).all(1)
    subsets['any_gas_missing']=(~np.isfinite(gas)).any(1)
    masks={f'E{i+1:02}':idx for i,idx in enumerate(groups)}
    normal=[];off=[];seed_scores={}
    for seed in [42,43,44]:
        ck=torch.load(args.base_run/f'seed_{seed}/B/best.pt',map_location='cpu',weights_only=True)
        model=abc.FusionModel('B').to(args.device);model.load_state_dict(ck['state'])
        a,b=evaluate_pair(model,data,args.device);normal.append(a);off.append(b)
        seed_scores[str(seed)]={'normal':scores(confusion(y,a.argmax(1)+1)),'gas_off':scores(confusion(y,b.argmax(1)+1))}
        print(f'Evaluated ABC B seed={seed}; aggregates only',flush=True)
    pa=np.mean(normal,0).argmax(1)+1;pb=np.mean(off,0).argmax(1)+1
    if not np.array_equal(pa,ref.pred):raise ValueError('CPU baseline predictions differ from saved ensemble; investigate before interpreting')
    report={'overall':{},'groups':{},'subsets':{},'seed_scores':seed_scores,'notes':[
      'Both gas mean and selected gas features are zeroed, including pressure. Images and query unchanged.',
      'Inference intervention differs from a retrained model and introduces out-of-training feature combinations.',
      'Zero concentration is not treated as missing. Zero subset requires all ten finite concentrations exactly zero.',
      'Validation-only diagnostics; no Test or row-level exports. E aliases match prior reports.']}
    for name,idx in [('overall',np.arange(len(y)))]+list(masks.items()):
        entry={'normal':scores(confusion(y[idx],pa[idx])),'gas_off':scores(confusion(y[idx],pb[idx])),
            'prediction_change_fraction':float(np.mean(pa[idx]!=pb[idx])),
            '5_to_4_normal':int(((y[idx]==5)&(pa[idx]==4)).sum()),'5_to_4_gas_off':int(((y[idx]==5)&(pb[idx]==4)).sum())}
        if name=='overall':report['overall']=entry
        else:report['groups'][name]=entry
    for name,mask in subsets.items():
        entry={'rows':int(mask.sum()),'true_5_rows':int(((y==5)&mask).sum())}
        if mask.any():
            entry.update(normal=scores(confusion(y[mask],pa[mask])),gas_off=scores(confusion(y[mask],pb[mask])))
        report['subsets'][name]=entry
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    rows=[]
    for name,e in report['groups'].items():
        rows.append(dict(experiment=name,normal_f1=e['normal']['macro_f1_true_present'],gas_off_f1=e['gas_off']['macro_f1_true_present'],
            normal_5_to_4=e['5_to_4_normal'],gas_off_5_to_4=e['5_to_4_gas_off'],prediction_change_fraction=e['prediction_change_fraction']))
    pd.DataFrame(rows).to_csv(args.output/'experiment_comparison.csv',index=False)
    print('OVERALL',json.dumps(report['overall']),flush=True)
    print('E16',json.dumps(report['groups']['E16']),flush=True)
    print('ZERO/MISSING SUBSETS',json.dumps(report['subsets']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run',type=Path,default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    p.add_argument('--data-dir',type=Path,default=Path('/home/ubuntu/data/battery_data_light'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(p.parse_args())
