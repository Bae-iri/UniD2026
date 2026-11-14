"""Controlled short fine-tuning: same ABC B checkpoint, with/without gas dropout.
No Test access, no row-level exports. Training occurs only when explicitly run.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from abc_reference import abc_fusion_experiment as abc
from diagnose_abc_gas_dependency import restore_preprocessor
from analyze_experiment_performance import confusion,scores


class GasDropoutB(abc.FusionModel):
    def __init__(self,p=0.,seed=0,d=64):
        super().__init__('B',d=d)
        self.drop_probability=p
        self.mask_rng=torch.Generator().manual_seed(seed+11719)
        self.head.register_forward_pre_hook(self.drop_gas)

    def drop_gas(self,module,inputs):
        if not self.training or self.drop_probability==0:return None
        f=inputs[0];d=f.shape[1]//5
        keep=(torch.rand(len(f),generator=self.mask_rng)>=self.drop_probability).to(f.device,f.dtype)[:,None]
        # Zero both summary and attention outputs together, including pressure.
        return (torch.cat([f[:,:d],f[:,d:2*d]*keep,f[:,2*d:3*d],f[:,3*d:4*d]*keep,f[:,4*d:]],1),)


@torch.inference_mode()
def predict(model,data,device):
    model.eval();parts=[]
    for start in range(0,len(data.frame),128):
        ids=torch.arange(start,min(start+128,len(data.frame)))
        parts.append(model(*data.batch(ids,device)).softmax(1).cpu().numpy())
    return np.concatenate(parts)


def run(args):
    if args.output.exists():raise FileExistsError('Output exists; use a new directory')
    torch.set_num_threads(2)
    config=json.loads((args.base_run/'config.json').read_text())
    # Protect original splits/statistics from silent changes; do not read Test.
    for split in ['train','validation']:
        path=args.data_dir/split/f'{split}.csv'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=config['csv_sha256'][split]:raise ValueError('Source CSV changed')
    frames=abc.common.split_frames(args.data_dir,config['split_seed'],.2)
    prep=restore_preprocessor(args.base_run)
    manifest=json.loads((args.base_run/'image_cache_manifest.json').read_text())
    datasets={}
    for split,f in frames.items():
        packed=np.load(manifest[split],mmap_mode='c')
        if packed.shape!=(len(f),4,128,128):raise ValueError('Cache shape differs')
        if split in ['dev','validation']:
            ref=pd.read_csv(args.base_run/f'seed_{args.seeds[0]}/B/{split}_predictions.csv')
            if not np.array_equal(ref.sample_id,f.sample_id) or not np.array_equal(ref.label,f.label):raise ValueError('Saved split differs')
        q,g=prep.transform(f)
        datasets[split]=abc.Data(f,torch.from_numpy(packed),torch.from_numpy(q),torch.from_numpy(g),torch.tensor(f.label.to_numpy(int)-1))
    args.output.mkdir(parents=True)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings['selection']='highest original Train-dev Macro F1, including epoch 0'
    settings['comparison']='matched continued training from same ABC B checkpoint; fresh optimizer in both arms'
    settings['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    settings['base_checkpoint_sha256']={str(seed):hashlib.sha256((args.base_run/f'seed_{seed}/B/best.pt').read_bytes()).hexdigest() for seed in args.seeds}
    (args.output/'preprocessing.json').write_bytes((args.base_run/'preprocessing.json').read_bytes())
    (args.output/'config.json').write_text(json.dumps(settings,indent=2))
    summary=[]
    for seed in args.seeds:
        initial=torch.load(args.base_run/f'seed_{seed}/B/best.pt',map_location='cpu',weights_only=True)
        for mode,p in [('B_continue',0.),('B_gas_dropout',args.drop_probability)]:
            abc.common.seed_everything(seed)
            model=GasDropoutB(p,seed).to(args.device);model.load_state_dict(initial['state'])
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
            lossfn=torch.nn.CrossEntropyLoss(weight=torch.tensor(prep.weights,device=args.device))
            folder=args.output/f'{mode}_seed{seed}';folder.mkdir()
            best=-1.;history=[]
            order_rng=torch.Generator().manual_seed(seed)
            for epoch in range(args.epochs+1):
                if epoch:
                    model.train();order=torch.randperm(len(frames['fit']),generator=order_rng)
                    for start in range(0,len(order),64):
                        ids=order[start:start+64];opt.zero_grad(set_to_none=True)
                        loss=lossfn(model(*datasets['fit'].batch(ids,args.device)),datasets['fit'].y[ids].to(args.device))
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                prob=predict(model,datasets['dev'],args.device)
                dev=scores(confusion(frames['dev'].label,prob.argmax(1)+1))['macro_f1_6']
                history.append(dict(epoch=epoch,dev_macro_f1=dev))
                if dev>best:
                    best=dev
                    torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},seed=seed,
                        epoch=epoch,dev_macro_f1=dev,drop_probability=p,mode=mode),folder/'best.pt')
                (folder/'history.json').write_text(json.dumps(history,indent=2))
                print(f'{mode} seed={seed} epoch={epoch}/{args.epochs} dev_macro_f1={dev:.6f}',flush=True)
            ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state'])
            prob=predict(model,datasets['validation'],args.device);pred=prob.argmax(1)+1
            s=scores(confusion(frames['validation'].label,pred));bygroup=[]
            for i,idx in enumerate(frames['validation'].groupby('experiment_id',sort=True).indices.values()):
                c=confusion(frames['validation'].label.to_numpy()[idx],pred[idx])
                bygroup.append(dict(experiment=f'E{i+1:02}',**scores(c),five_to_four=int(c[4,3])))
            (folder/'validation_aggregates.json').write_text(json.dumps(dict(overall=s,experiments=bygroup),indent=2))
            summary.append(dict(mode=mode,seed=seed,best_epoch=ck['epoch'],dev_macro_f1=ck['dev_macro_f1'],validation_macro_f1=s['macro_f1_6']))
            pd.DataFrame(summary).to_csv(args.output/'summary.csv',index=False)
    table=pd.DataFrame(summary).pivot(index='seed',columns='mode',values='validation_macro_f1')
    table['dropout_minus_continue']=table.B_gas_dropout-table.B_continue
    table.to_csv(args.output/'paired_differences.csv');print(table.to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run',type=Path,default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    p.add_argument('--data-dir',type=Path,default=Path('/home/ubuntu/data/battery_data_light'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--drop-probability',type=float,default=.2)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    if args.epochs<1 or not 0<args.drop_probability<1 or len(set(args.seeds))!=len(args.seeds):p.error('Invalid arguments')
    run(args)
