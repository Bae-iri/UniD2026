"""Validation-only factorial interventions and attention diagnostics on frozen ABC B.
No training, no Test reads, no row-level exports. Cached features remain in memory only.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from abc_reference import abc_fusion_experiment as abc
from diagnose_abc_gas_dependency import validation_data
from analyze_experiment_performance import confusion,scores

MODES=['normal','gas_off','image_off','gas_image_off','sensor_direct_off',
 'gas_uniform','image_uniform','both_uniform','concentrations_masked','pressure_masked',
 'gas_selected_off','gas_global_off','image_selected_off','image_global_off']


def head_variant(model,features,mode,gas_args,image_args,gas_input):
    f=features.clone();d=f.shape[1]//5
    slots={'gas_off':[1,3],'image_off':[2,4],'gas_image_off':[1,2,3,4],
        'sensor_direct_off':[0],'gas_selected_off':[3],'gas_global_off':[1],
        'image_selected_off':[4],'image_global_off':[2]}
    for slot in slots.get(mode,[]):f[:,slot*d:(slot+1)*d]=0
    if mode in ['gas_uniform','both_uniform']:f[:,3*d:4*d]=model.gas_readout(*gas_args[:3],'A')
    if mode in ['image_uniform','both_uniform']:f[:,4*d:5*d]=model.image_readout(*image_args[:3],'A')
    if mode in ['concentrations_masked','pressure_masked']:
        changed=gas_input.clone()
        idx=[i for i,c in enumerate(abc.GAS) if (c!='meas_p')==(mode=='concentrations_masked')]
        changed[:,idx,0]=0;changed[:,idx,1]=1
        identity=model.gas_id(model.gas_indices)[None].expand(len(f),-1,-1)
        tokens=model.gas_encoder(torch.cat([changed,identity],-1));valid=changed[...,1]<.5
        f[:,d:2*d]=abc.common.masked_mean(tokens,valid)
        f[:,3*d:4*d]=model.gas_readout(tokens,gas_args[1],valid,'B')
    return model.head(f)


def attention_summary(readout,args):
    tokens,q,valid=args[:3]
    s=(readout.key(tokens)*readout.query(q)[:,None]).sum(-1)/(tokens.shape[-1]**.5)
    a=s.masked_fill(~valid,torch.finfo(s.dtype).min).softmax(-1)*valid
    a=a/a.sum(-1,keepdim=True).clamp_min(1e-8)
    entropy=-(a*a.clamp_min(1e-12).log()).sum(1)
    entden=valid.sum(1).clamp_min(2).float().log()
    return torch.stack([entropy/entden,a.max(1).values,a[:,0]],1)


@torch.inference_mode()
def evaluate(model,data,device):
    captured={}
    def head_hook(module,args):captured['head']=args[0]
    def gas_hook(module,args):captured['gas']=args
    def image_hook(module,args):captured['image']=args
    handles=[model.head.register_forward_pre_hook(head_hook),model.gas_readout.register_forward_pre_hook(gas_hook),model.image_readout.register_forward_pre_hook(image_hook)]
    probs={m:[] for m in MODES};margins={m:[] for m in MODES};att=[]
    model.eval()
    try:
        for start in range(0,len(data.frame),128):
            ids=torch.arange(start,min(start+128,len(data.frame)))
            batch=data.batch(ids,device);normal=model(*batch)
            features=captured['head'].clone();ga=captured['gas'];ia=captured['image']
            att.append(torch.cat([attention_summary(model.gas_readout,ga),attention_summary(model.image_readout,ia)],1).cpu().numpy())
            for mode in MODES:
                out=normal if mode=='normal' else head_variant(model,features,mode,ga,ia,batch[3])
                probs[mode].append(out.softmax(1).cpu().numpy());margins[mode].append((out[:,4]-out[:,3]).cpu().numpy())
    finally:
        for h in handles:h.remove()
    return {m:np.concatenate(p) for m,p in probs.items()},{m:np.concatenate(p) for m,p in margins.items()},np.concatenate(att)


def describe(a):
    a=np.asarray(a,float)
    return {'count':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9))} if len(a) else {'count':0}


def run(args):
    if args.output.exists():raise FileExistsError('Choose new output directory')
    torch.set_num_threads(2);data,ref=validation_data(args.base_run,args.data_dir)
    frame=data.frame;y=frame.label.to_numpy(int);groups=list(frame.groupby('experiment_id',sort=True).indices.values())
    masks={'all':np.ones(len(y),bool),'all_true5':y==5,'other_true5':y==5}
    for i,idx in enumerate(groups):
        mask=np.zeros(len(y),bool);mask[idx]=True;masks[f'E{i+1:02}']=mask
    masks['E16_true5']=masks['E16']&(y==5);masks['other_true5']&=~masks['E16']
    sums={m:None for m in MODES};margin_sums={m:None for m in MODES};attention=[];seed_reports={}
    for seed in args.seeds:
        model=abc.FusionModel('B').to(args.device)
        ck=torch.load(args.base_run/f'seed_{seed}/B/best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state'])
        ps,ms,at=evaluate(model,data,args.device);attention.append(at)
        for m in MODES:
            sums[m]=ps[m] if sums[m] is None else sums[m]+ps[m]
            margin_sums[m]=ms[m] if margin_sums[m] is None else margin_sums[m]+ms[m]
        seed_reports[str(seed)]={m:{'overall':scores(confusion(y,p.argmax(1)+1)),'E16':scores(confusion(y[masks['E16']],p[masks['E16']].argmax(1)+1))} for m,p in ps.items()}
        print(f'Completed 14 interventions on seed {seed}',flush=True)
    pred={m:p.argmax(1)+1 for m,p in sums.items()}
    if args.seeds==[42,43,44] and not np.array_equal(pred['normal'],ref.pred):raise ValueError('Baseline predictions differ')
    report={'seeds':args.seeds,'interventions':{},'attention':{},'interaction_margin':{},'seed_reports':seed_reports,
      'notes':['All interventions use the same frozen checkpoints and Validation only.',
      'Zeroing feature paths or masking observations changes inference distribution; not a retrained-model comparison.',
      'Concentration masking preserves pressure; pressure masking preserves concentration tokens. Both summaries and readouts are recomputed.',
      'Uniform selection changes readout only and retains learned projections/FFN.',
      'Class-5 minus class-4 logit interaction = both - no_gas - no_image + neither; not physical causality.',
      'Experimental comparisons are exploratory after observing Validation errors; require dev/group-held-out confirmation.']}
    rows=[];normal=pred['normal']
    for mode,p in pred.items():
        entry={}
        for name,mask in masks.items():
            s=scores(confusion(y[mask],p[mask]));s.update(rows=int(mask.sum()),five_to_four=int(((y==5)&(p==4)&mask).sum()),
                five_correct=int(((y==5)&(p==5)&mask).sum()),corrected=int(((normal!=y)&(p==y)&mask).sum()),
                harmed=int(((normal==y)&(p!=y)&mask).sum()),margin_5_minus_4=describe((margin_sums[mode]/len(args.seeds))[mask]))
            entry[name]=s
        report['interventions'][mode]=entry
        a=entry['all'];e=entry['E16']
        rows.append(dict(mode=mode,overall_f1=a['macro_f1_6'],class5_f1=a['class_f1'][4],E16_f1=e['macro_f1_true_present'],
          E16_5_to_4=e['five_to_four'],E16_5_correct=e['five_correct'],corrected=a['corrected'],harmed=a['harmed']))
    at=np.mean(attention,0)
    interaction=(margin_sums['normal']-margin_sums['gas_off']-margin_sums['image_off']+margin_sums['gas_image_off'])/len(args.seeds)
    for name in ['all_true5','E16_true5','other_true5']:
        mask=masks[name];report['attention'][name]={n:describe(at[mask,i]) for i,n in enumerate(['gas_entropy','gas_max_weight','pressure_weight','image_entropy','image_max_weight','first_image_token_weight'])}
        report['interaction_margin'][name]=describe(interaction[mask])
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    pd.DataFrame(rows).to_csv(args.output/'intervention_summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('ATTENTION',json.dumps(report['attention']),flush=True);print('INTERACTION',json.dumps(report['interaction_margin']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run',type=Path,default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    p.add_argument('--data-dir',type=Path,default=Path('/home/ubuntu/data/battery_data_light'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    run(p.parse_args())
