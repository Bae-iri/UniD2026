"""Independent B experiments. Every run starts from ORIGINAL B, never another study."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import confusion_matrix

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'common'))
import abc_fusion_experiment as original
from experiment1_voltage.model import VoltageBoostB,Preprocessor as VoltagePreprocessor
from experiment1_voltage.event_memory import event_memory,NAMES
from experiment2_gas_temperature.model import GasTemperatureB,Preprocessor as TemperaturePreprocessor,all_observed_zero

MODES=['B_control','experiment1_voltage','experiment2_gas_temperature']


def make_model(mode):
    if mode=='B_control': return original.FusionModel('B')
    if mode=='experiment1_voltage': return VoltageBoostB()
    if mode=='experiment2_gas_temperature': return GasTemperatureB()
    raise ValueError(mode)


def make_preprocessor(mode,fit):
    if mode=='B_control': return original.SensorPreprocessor(fit)
    if mode=='experiment1_voltage': return VoltagePreprocessor(fit)
    if mode=='experiment2_gas_temperature': return TemperaturePreprocessor(fit)
    raise ValueError(mode)


def extended_metrics(frame,p):
    result=original.metrics(frame,p); y=frame.label.to_numpy(int);pred=p.argmax(1)+1
    cm=confusion_matrix(y,pred,labels=range(1,7))
    for label in [3,4,5]:
        n=int(cm[label-1].sum());wrong=int(n-cm[label-1,label-1])
        result.update({f'support_{label}':n,f'wrong_{label}':wrong,f'error_rate_{label}':wrong/n if n else None})
    for a,b in [(3,5),(5,3),(4,5),(5,4)]:result[f'true{a}_pred{b}']=int(cm[a-1,b-1])
    zero=all_observed_zero(frame)>.5
    for label in [4,5]:
        mask=zero&(y==label);n=int(mask.sum());correct=int((pred[mask]==label).sum());to5=int((pred[mask]==5).sum())
        result.update({f'zero_support_{label}':n,f'zero_correct_{label}':correct,f'zero_true{label}_pred5':to5,f'zero_error_rate_{label}':1-correct/n if n else None})
    return result


@torch.no_grad()
def predict(model,data,device):
    model.eval(); probabilities=[]
    for start in range(0,len(data.frame),128):
        ids=torch.arange(start,min(start+128,len(data.frame)))
        probabilities.append(model(*data.batch(ids,device)).softmax(1).cpu().numpy())
    p=np.concatenate(probabilities)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid probability')
    return p


def original_state_into(model,state):
    keys=set(original.FusionModel('B').state_dict())
    if set(state)!=keys:raise ValueError('Source checkpoint is not exact original B')
    missing,extra=model.load_state_dict(state,strict=False)
    if extra or set(missing)!=(set(model.state_dict())-keys):raise ValueError('Unexpected warm-start mismatch')


def train_one(mode,seed,state,data,weights,args,out,device):
    folder=out/mode/f'seed_{seed}';folder.mkdir(parents=True)
    original.common.seed_everything(seed);model=make_model(mode).to(device);original_state_into(model,state)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device))
    # Reset after constructor/loading: common head dropout uses the same RNG in every study.
    original.common.seed_everything(seed+999);generator=torch.Generator().manual_seed(seed)
    dev=original.metrics(data['dev'].frame,predict(model,data['dev'],device))['macro_f1']
    best=dev;history=[dict(epoch=0,train_loss=None,dev_macro_f1=dev)];started=time.monotonic()
    torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=0,dev_macro_f1=best,mode=mode,seed=seed),folder/'best.pt')
    fit=data['fit']
    for epoch in range(1,args.epochs+1):
        model.train();total=0
        for ids in torch.randperm(len(fit.frame),generator=generator).split(64):
            y=fit.y[ids.to(fit.y.device)].to(device);optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(*fit.batch(ids,device)),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=loss.item()*len(ids)
        dev=original.metrics(data['dev'].frame,predict(model,data['dev'],device))['macro_f1']
        if dev>best:
            best=dev;torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,dev_macro_f1=dev,mode=mode,seed=seed),folder/'best.pt')
        history.append(dict(epoch=epoch,train_loss=total/len(fit.frame),dev_macro_f1=dev));pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        print(f'{mode} seed={seed} {epoch}/{args.epochs} dev={dev:.4f} best={best:.4f}',flush=True)
    checkpoint=torch.load(folder/'best.pt',weights_only=True);model.load_state_dict(checkpoint['state'])
    rows=[]
    for split in ['dev','validation','test']:
        p=predict(model,data[split],device);original.save_predictions(folder,split,data[split].frame,p)
        if split!='test':rows.append(dict(model=mode,seed=seed,split=split,best_epoch=checkpoint['epoch'],**extended_metrics(data[split].frame,p)))
    original.common.write_json(folder/'metrics.json',dict(rows=rows,seconds=time.monotonic()-started,initialized_from='ORIGINAL B, not B_control or another experiment'))
    del model,optimizer;torch.cuda.empty_cache()
    return rows


def finalize(rows,frames,args,out):
    summary=pd.DataFrame(rows);summary.to_csv(out/'summary.csv',index=False)
    val=summary[summary.split=='validation'];metrics=['macro_f1','F1_3','F1_4','F1_5','error_rate_3','error_rate_4','error_rate_5']
    val.groupby('model')[metrics].agg(['mean','std']).to_csv(out/'seed_aggregate.csv')
    sample=pd.read_csv(args.data_root/'sample_submission.csv');ensemble_rows=[];ensembles={}
    for mode in args.modes:
        ensembles[mode]={}
        for split in ['dev','validation','test']:
            predictions=[]
            for seed in args.seeds:
                p=pd.read_csv(out/mode/f'seed_{seed}'/f'{split}_predictions.csv')
                if p.sample_id.tolist()!=frames[split].sample_id.tolist():raise ValueError('Ensemble row mismatch')
                predictions.append(p[[f'p{i}' for i in range(1,7)]].to_numpy())
            p=np.mean(predictions,axis=0);ensembles[mode][split]=p
            folder=out/mode/'ensemble';folder.mkdir(exist_ok=True);original.save_predictions(folder,split,frames[split],p)
            if split!='test':ensemble_rows.append(dict(model=mode,split=split,**extended_metrics(frames[split],p)))
        submission=original.make_submission(sample,frames['test'],ensembles[mode]['test'])
        submission.to_csv(out/mode/'submission.csv',index=False)
        submission.to_csv(out/f'submission_{mode}.csv',index=False)
    ensemble=pd.DataFrame(ensemble_rows);ensemble.to_csv(out/'ensemble_scores.csv',index=False)
    percent=ensemble.copy()
    for c in ['macro_f1']+[f'F1_{i}' for i in range(1,7)]+[f'error_rate_{i}' for i in [3,4,5]]+[f'zero_error_rate_{i}' for i in [4,5]]:percent[c]*=100
    percent.to_csv(out/'ensemble_percent.csv',index=False)
    comparisons=[];paired=[];ev=ensemble[ensemble.split=='validation'].set_index('model')
    if 'B_control' in ev.index:
        for mode in args.modes:
            if mode=='B_control':continue
            row=dict(model=mode,reference='B_control')
            for c in metrics:row[c+'_delta_pp']=100*(ev.loc[mode,c]-ev.loc['B_control',c])
            comparisons.append(row)
            for seed in args.seeds:
                a=val[(val.model==mode)&(val.seed==seed)].iloc[0];b=val[(val.model=='B_control')&(val.seed==seed)].iloc[0]
                paired.append(dict(model=mode,seed=seed,macro_f1_delta_pp=100*(a.macro_f1-b.macro_f1),error3_delta_pp=100*(a.error_rate_3-b.error_rate_3),error4_delta_pp=100*(a.error_rate_4-b.error_rate_4),error5_delta_pp=100*(a.error_rate_5-b.error_rate_5)))
    pd.DataFrame(comparisons).to_csv(out/'ensemble_differences.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_seed_differences.csv',index=False)
    text=['# 原本 B 기반 독립 실험 결과','',original.markdown_table(percent[percent.split=='validation'].round(2)),'',original.markdown_table(pd.DataFrame(comparisons).round(2)),'',
       'F1과 오류율은 0~100, 차이는 %p. 오류율은 해당 실제 클래스 표본 중 다른 단계로 예측한 비율(1-recall). 음수 오류율 차이가 개선임.',
       '각 실험은 원본 B seed42/43/44 체크포인트에서 독립 시작. B_control도 동일 추가 학습량으로 비교하며 이전 실험의 학습 결과를 다음 실험으로 넘기지 않음.',
       '실험1: 전압 급락 전용 branch의 logit 잔차 ×2. 실험2: 화학 가스 모두 관측된 0일 때 chemical value ×0.5, 현재 온도 전용 logit 잔차 ×2. 압력 및 이미지 경로는 유지. 실험2는 전압 사건 특징을 계산/입력하지 않음.',
       '계수는 실행 전에 고정했으며 실제 학습된 중요도가 정확히 2배/절반이라는 뜻은 아님. 실험1은 특징 추가+강조의 결합 효과, 실험2는 gas downweight+temperature branch의 결합 효과임.',
       '모든 checkpoint는 dev 기준으로만 선택. validation은 이전에 확인한 탐색 데이터이며 새 blind test가 아님. 제출 점수는 실제 평가 전 알 수 없음.',
       '각 실험 submission.csv는 해당 실험의 seed 확률 평균이며 원본 template의 sample_id,label 순서로 저장. 원본 B 제출 및 checkpoint는 변경하지 않음.','']
    (out/'report.md').write_text('\n'.join(text));print(percent[percent.split=='validation'].to_string(index=False),flush=True)


def main(forced_modes=None,default_output=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=Path('/home/ubuntu/data/battery_data_light'))
    p.add_argument('--base-run',type=Path,default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    p.add_argument('--output',type=Path,default=default_output or BASE/'runs/independent_seed42_43_44_v1')
    p.add_argument('--modes',nargs='+',choices=MODES,default=MODES);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--cache-dir',type=Path,default=BASE.parent/'battery_query_rates/.hierarchy_cache');p.add_argument('--device',choices=['cuda','cpu'],default='cuda');args=p.parse_args()
    if forced_modes is not None:args.modes=forced_modes
    if args.epochs<1 or len(set(args.modes))!=len(args.modes) or len(set(args.seeds))!=len(args.seeds):p.error('Invalid training settings')
    torch.set_num_threads(4);device=torch.device(args.device);out=args.output;out.mkdir(parents=True,exist_ok=False);args.smoke=False
    reference_cfg=json.loads((args.base_run/'config.json').read_text());args.split_seed=reference_cfg['split_seed']
    for split,digest in reference_cfg['csv_sha256'].items():
        if hashlib.sha256((args.data_root/split/f'{split}.csv').read_bytes()).hexdigest()!=digest:raise ValueError('Original B data changed')
    source_files=[p for p in BASE.rglob('*.py') if 'runs' not in p.parts and '__pycache__' not in p.parts]
    cfg={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    cfg.update(source_sha256={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},original_csv_sha256=reference_cfg['csv_sha256'],gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,lr=3e-4,weight_decay=1e-4,batch_size=64,initialization='Independent warm-start from ORIGINAL B per seed',dropout_seed='seed+999 reset after model construction',experiment1_event_gain=2.,experiment2_chemical_gain=.5,experiment2_temperature_gain=2.)
    original.common.write_json(out/'config.json',cfg)
    for path in source_files:
        target=out/'sources'/path.relative_to(BASE);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    frames=original.load_frames(args)
    manifest=pd.concat([f[['sample_id','experiment_id']].assign(split=s) for s,f in frames.items()]);manifest.to_csv(out/'split_manifest.csv',index=False)
    pd.testing.assert_frame_equal(pd.read_csv(out/'split_manifest.csv'),pd.read_csv(args.base_run/'split_manifest.csv'))
    if not manifest.sample_id.is_unique:raise ValueError('Sample overlap')
    control_prep=original.SensorPreprocessor(frames['fit']);preps={mode:make_preprocessor(mode,frames['fit']) for mode in args.modes}
    original_cache=original.common.cache_images
    def cached(root,frame,split,unused,size,grid,workers):return original_cache(root,frame,split,args.cache_dir,size,grid,workers)
    original.common.cache_images=cached
    cache_data=original.prepare(args,frames,control_prep,device);data={}
    for mode,prep in preps.items():
        folder=out/mode;folder.mkdir();original.common.write_json(folder/'preprocessing.json',prep.metadata());data[mode]={}
        for split,f in frames.items():
            q,gas=prep.transform(f);existing=cache_data[split]
            np.testing.assert_allclose(gas,existing.gas.cpu().numpy(),atol=0,rtol=0)
            data[mode][split]=original.Data(f,existing.images,torch.tensor(q,device=existing.q.device),existing.gas,existing.y)
            if mode=='experiment1_voltage':
                raw=event_memory(f,prep.detector);table=f[['sample_id','experiment_id','relative_time_s']].copy()
                for k,name in enumerate(NAMES):table[name]=raw[:,k]
                table.to_csv(folder/f'{split}_voltage_event_columns.csv',index=False)
            if mode=='experiment2_gas_temperature':
                f[['sample_id','experiment_id']].assign(all_chemical_gases_observed_zero=all_observed_zero(f).astype(bool)).to_csv(folder/f'{split}_zero_state.csv',index=False)
    originals=out/'original_B_checkpoints';originals.mkdir();rows=[];original_rows=[]
    for seed in args.seeds:
        source=args.base_run/f'seed_{seed}/B/best.pt';target=originals/f'seed_{seed}.pt';shutil.copy2(source,target)
        checkpoint=torch.load(target,map_location='cpu',weights_only=True)
        original.common.seed_everything(seed);before=original.FusionModel('B').to(device);before.load_state_dict(checkpoint['state'])
        p=predict(before,cache_data['validation'],device)
        reference=pd.read_csv(args.base_run/f'seed_{seed}/B/validation_predictions.csv')
        if reference.sample_id.tolist()!=frames['validation'].sample_id.tolist():raise ValueError('Original prediction ID mismatch')
        np.testing.assert_allclose(p,reference[[f'p{i}' for i in range(1,7)]].to_numpy(),atol=2e-6,rtol=1e-5)
        original_rows.append(dict(seed=seed,original_checkpoint_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),**extended_metrics(frames['validation'],p)))
        pd.DataFrame(original_rows).to_csv(out/'original_B_reference_scores.csv',index=False);del before
        for mode in args.modes:
            rows+=train_one(mode,seed,checkpoint['state'],data[mode],control_prep.weights,args,out,device)
            pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)
    finalize(rows,frames,args,out);print('FINISHED',out,flush=True)

if __name__=='__main__':main()
