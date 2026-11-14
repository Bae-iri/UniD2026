"""One leakage-controlled gated hybrid training run, with final submission CSV."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,'/tmp/gated_hybrid_deps')
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.models import convnext_tiny,ConvNeXt_Tiny_Weights
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score,confusion_matrix,classification_report
from features import temporal_features,Normalizer
from model import ImageHead,GatedHybrid,GROUPS,hierarchy_prob,relation_loss

BASE=Path(__file__).resolve().parent


def write_json(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


class Images(Dataset):
    def __init__(self,frame,root):
        self.paths=[]
        for value in frame.image_path:
            rel=Path(str(value).replace('\\','/'))
            if rel.is_absolute() or '..' in rel.parts: raise ValueError('Invalid image path')
            path=root/rel
            if not path.is_file(): raise FileNotFoundError(path)
            self.paths.append(path)
        self.mean=torch.tensor([.485,.456,.406])[:,None,None]
        self.std=torch.tensor([.229,.224,.225])[:,None,None]

    def __len__(self): return len(self.paths)

    def __getitem__(self,i):
        with Image.open(self.paths[i]) as image:
            image=ImageOps.pad(image.convert('RGB'),(224,224),method=Image.Resampling.BILINEAR,color=(0,0,0))
            x=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()).permute(2,0,1)/255
        return (x-self.mean)/self.std


def extract_images(frames,args,device,out):
    torch.hub.set_dir(str(BASE/'.weights'))
    backbone=convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    backbone.classifier[-1]=nn.Identity(); backbone.eval().requires_grad_(False).to(device)
    arrays={}; manifest={}
    cache=BASE/'.image_features'; cache.mkdir(exist_ok=True)
    for split,frame in frames.items():
        dataset=Images(frame,args.data_root/split)
        digest=hashlib.sha256(b'convnext-tiny-imagenet1k-v1-letterbox224-normalized-v1')
        for sample,path in zip(frame.sample_id,dataset.paths):
            stat=path.stat(); digest.update(f'{sample}:{path}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
        path=cache/f'{split}_{digest.hexdigest()[:20]}.npy'; manifest[split]=str(path)
        if not path.exists():
            partial=path.with_suffix('.partial.npy')
            features=np.lib.format.open_memmap(partial,mode='w+',dtype='float32',shape=(len(frame),768))
            loader=DataLoader(dataset,batch_size=64,num_workers=4,pin_memory=True)
            start=0; started=time.monotonic()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                for x in loader:
                    output=backbone(x.to(device,non_blocking=True)).float().cpu().numpy()
                    features[start:start+len(x)]=output; start+=len(x)
                    if start%4096==0: print(f'ConvNeXt cache {split}: {start}/{len(frame)}',flush=True)
            features.flush(); del features; partial.replace(path)
            print(f'Image extraction {split}: {time.monotonic()-started:.1f}s',flush=True)
        arrays[split]=np.load(path,mmap_mode='r')
    weight_path=BASE/'.weights/checkpoints/convnext_tiny-983f1562.pth'
    write_json(out/'image_cache_manifest.json',dict(paths=manifest,pretrained_weights_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest(),backbone_frozen=True,preprocessing='RGB letterbox224; ImageNet normalization; no crop'))
    del backbone; torch.cuda.empty_cache()
    return arrays


def split_data(args,out):
    official={s:pd.read_csv(args.data_root/s/f'{s}.csv') for s in ['train','validation','test']}
    for s,f in official.items():
        if f.empty or f.experiment_id.isna().any() or f.sample_id.isna().any() or not f.sample_id.is_unique: raise ValueError(f'Invalid IDs: {s}')
        if s!='test' and set(f.label)!=set(range(1,7)): raise ValueError(f'Invalid labels: {s}')
        if s=='test' and 'label' in f: raise ValueError('Test labels must not be used')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for col in ['experiment_id','sample_id']:
            if set(official[a][col])&set(official[b][col]): raise ValueError('Official split overlap')
    train=official['train']; groups=train.experiment_id
    fit,dev=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.split_seed).split(train,groups=groups))
    expert,gate=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.seed+901).split(train.iloc[fit],groups=groups.iloc[fit]))
    indices={'expert':('train',fit[expert]),'gate':('train',fit[gate]),'dev':('train',dev),
             'validation':('validation',np.arange(len(official['validation']))),'test':('test',np.arange(len(official['test'])))}
    frames={name:official[s].iloc[ids].reset_index(drop=True) for name,(s,ids) in indices.items()}
    for name,f in frames.items():
        if name!='test' and set(f.label)!=set(range(1,7)): raise ValueError(f'Missing class in {name}')
        print(name,len(f),f.experiment_id.nunique(),'experiments',flush=True)
    for a,f in frames.items():
        for b,g in frames.items():
            if a!=b and set(f.experiment_id)&set(g.experiment_id): raise ValueError('Internal experiment overlap')
    pd.concat([f[['sample_id','experiment_id']].assign(split=n) for n,f in frames.items()]).to_csv(out/'split_manifest.csv',index=False)
    return official,frames,indices


def scores(frame,p):
    if p.shape!=(len(frame),6) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5): raise ValueError('Invalid probabilities')
    y=frame.label.to_numpy(int); pred=p.argmax(1)+1
    f=f1_score(y,pred,labels=range(1,7),average=None,zero_division=0); cm=confusion_matrix(y,pred,labels=range(1,7))
    return dict(macro_f1=float(f.mean()),**{f'F1_{i+1}':float(v) for i,v in enumerate(f)},true4_pred5=int(cm[3,4]),true5_pred4=int(cm[4,3]),support4=int(cm[3].sum()),support5=int(cm[4].sum()))


def save_predictions(out,name,frame,p,alpha=None):
    pred=frame[['sample_id','experiment_id']+(['label'] if 'label' in frame else [])].copy()
    pred['prediction']=p.argmax(1)+1
    for i in range(6): pred[f'p{i+1}']=p[:,i]
    if alpha is not None: pred['sensor_gate_alpha']=alpha
    pred.to_csv(out/f'{name}_predictions.csv',index=False)
    if 'label' in frame:
        cm=confusion_matrix(frame.label,pred.prediction,labels=range(1,7))
        pd.DataFrame(cm,index=range(1,7),columns=range(1,7)).to_csv(out/f'{name}_confusion_matrix.csv',index_label='true_label')
        pd.DataFrame(classification_report(frame.label,pred.prediction,labels=range(1,7),output_dict=True,zero_division=0)).T.to_csv(out/f'{name}_classification_report.csv')


def fit_catboost(frames,x,args,out):
    y={k:f.label.to_numpy(int)-1 for k,f in frames.items() if k!='test'}
    group=np.array(GROUPS)
    tasks={'flat':(np.ones(len(y['expert']),bool),np.ones(len(y['dev']),bool),y['expert'],y['dev']),
           'group':(np.ones(len(y['expert']),bool),np.ones(len(y['dev']),bool),group[y['expert']],group[y['dev']]),
           'early':(y['expert']<3,y['dev']<3,y['expert'][y['expert']<3],y['dev'][y['dev']<3]),
           'pair45':(np.isin(y['expert'],[3,4]),np.isin(y['dev'],[3,4]),y['expert'][np.isin(y['expert'],[3,4])]-3,y['dev'][np.isin(y['dev'],[3,4])]-3)}
    models={}; logs=[]
    for name,(fit_mask,dev_mask,yt,yd) in tasks.items():
        counts=np.bincount(yt); weights=counts.astype(float)**-.5; weights/=weights.mean()
        model=CatBoostClassifier(iterations=args.trees,depth=6,learning_rate=.05,loss_function='MultiClass',
              eval_metric='TotalF1:average=Macro;use_weights=false',class_weights=weights.tolist(),random_seed=args.seed,
              thread_count=4,l2_leaf_reg=5,allow_writing_files=False,task_type='CPU')
        print(f'CatBoost {name} training',flush=True)
        model.fit(x['expert'][fit_mask],yt,eval_set=(x['dev'][dev_mask],yd),use_best_model=True,early_stopping_rounds=80,verbose=100)
        model.save_model(str(out/f'sensor_{name}.cbm')); models[name]=model
        logs.append(dict(task=name,best_iteration=int(model.get_best_iteration()),trained_iterations=int(len(model.get_evals_result()['learn']['MultiClass']))))
        pd.DataFrame(logs).to_csv(out/'sensor_training.csv',index=False)
        pd.DataFrame(dict(feature=x['names'],importance=model.feature_importances_)).sort_values('importance',ascending=False).to_csv(out/f'sensor_{name}_feature_importance.csv',index=False)
    probabilities={}
    for split in frames:
        parts={k:m.predict_proba(x[split]) for k,m in models.items()}
        hierarchical=np.concatenate([parts['group'][:,:1]*parts['early'],parts['group'][:,1:2]*parts['pair45'],parts['group'][:,2:3]],1)
        probabilities[split]=(.7*parts['flat']+.3*hierarchical).astype(np.float32)
    return probabilities


@torch.no_grad()
def image_predict(model,x):
    model.eval(); return torch.cat([model(batch) for batch in x.split(512)]).cpu().numpy()


@torch.no_grad()
def gate_predict(model,sensor,image,ps,pi):
    model.eval(); p=[]; a=[]
    for start in range(0,len(sensor),512):
        sl=slice(start,start+512); prob,alpha=model(sensor[sl],image[sl],ps[sl],pi[sl]); p.append(prob.cpu().numpy()); a.append(alpha.cpu().numpy())
    return np.concatenate(p),np.concatenate(a).ravel()


def train_image(frames,images,args,out,device):
    seed_all(args.seed); model=ImageHead().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    y=torch.tensor(frames['expert'].label.to_numpy()-1,device=device)
    counts=np.bincount(y.cpu(),minlength=6); weights=counts.astype(np.float32)**-.5; weights/=weights.mean(); weights=torch.tensor(weights,device=device)
    generator=torch.Generator().manual_seed(args.seed); best=-1; history=[]
    for epoch in range(1,args.image_epochs+1):
        model.train(); total=0
        for ids in torch.randperm(len(y),generator=generator).to(device).split(256):
            optimizer.zero_grad(set_to_none=True); p=model(images['expert'][ids]); loss=relation_loss(p,y[ids],weights)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); total+=loss.item()*len(ids)
        result=scores(frames['dev'],image_predict(model,images['dev']))
        if result['macro_f1']>best:
            best=result['macro_f1']; torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,dev_macro_f1=best),out/'image_head_best.pt')
        history.append(dict(epoch=epoch,loss=total/len(y),**result)); pd.DataFrame(history).to_csv(out/'image_history.csv',index=False)
        print(f'IMAGE {epoch}/{args.image_epochs} dev F1={result["macro_f1"]:.4f} best={best:.4f}',flush=True)
    model.load_state_dict(torch.load(out/'image_head_best.pt',weights_only=True)['state'])
    return {s:image_predict(model,x) for s,x in images.items()}


def train_gate(frames,sensor,images,ps,pi,args,out,device):
    seed_all(args.seed+1); model=GatedHybrid(sensor['gate'].shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    y=torch.tensor(frames['gate'].label.to_numpy()-1,device=device)
    counts=np.bincount(y.cpu(),minlength=6); weights=counts.astype(np.float32)**-.5; weights/=weights.mean(); weights=torch.tensor(weights,device=device)
    generator=torch.Generator().manual_seed(args.seed+1)
    initial,_=gate_predict(model,sensor['dev'],images['dev'],ps['dev'],pi['dev'])
    best=scores(frames['dev'],initial)['macro_f1']; history=[]
    torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=0,dev_macro_f1=best),out/'gate_best.pt')
    history.append(dict(epoch=0,loss=None,dev_macro_f1=best))
    for epoch in range(1,args.gate_epochs+1):
        model.train(); total=0
        for ids in torch.randperm(len(y),generator=generator).to(device).split(128):
            optimizer.zero_grad(set_to_none=True)
            p,a=model(sensor['gate'][ids],images['gate'][ids],ps['gate'][ids],pi['gate'][ids])
            loss=relation_loss(p,y[ids],weights)+.002*(a-.5).square().mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); total+=loss.item()*len(ids)
        p,_=gate_predict(model,sensor['dev'],images['dev'],ps['dev'],pi['dev'])
        dev=scores(frames['dev'],p)['macro_f1']
        if dev>best:
            best=dev; torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,dev_macro_f1=dev),out/'gate_best.pt')
        history.append(dict(epoch=epoch,loss=total/len(y),dev_macro_f1=dev)); pd.DataFrame(history).to_csv(out/'gate_history.csv',index=False)
        print(f'GATE {epoch}/{args.gate_epochs} dev F1={dev:.4f} best={best:.4f}',flush=True)
    checkpoint=torch.load(out/'gate_best.pt',weights_only=True); model.load_state_dict(checkpoint['state'])
    write_json(out/'gate_selection.json',dict(epoch=checkpoint['epoch'],dev_macro_f1=checkpoint['dev_macro_f1']))
    return model


def make_submission(sample,test,p):
    if list(sample.columns)!=['sample_id','label'] or not sample.sample_id.is_unique or not test.sample_id.is_unique or set(sample.sample_id)!=set(test.sample_id): raise ValueError('Submission ID mismatch')
    result=sample.copy(); result['label']=result.sample_id.map(pd.Series(p.argmax(1)+1,index=test.sample_id))
    if result.label.isna().any() or not result.label.isin(range(1,7)).all(): raise ValueError('Invalid labels')
    result['label']=result.label.astype(int); return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('/home/ubuntu/data/battery_data_light'))
    p.add_argument('--output',type=Path,default=BASE/'runs/seed42_v1')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--split-seed',type=int,default=20261114)
    p.add_argument('--trees',type=int,default=600); p.add_argument('--image-epochs',type=int,default=30); p.add_argument('--gate-epochs',type=int,default=40)
    p.add_argument('--device',default='cuda',choices=['cuda','cpu']); args=p.parse_args()
    if min(args.trees,args.image_epochs,args.gate_epochs)<1: p.error('Training counts must be positive')
    out=args.output; out.mkdir(parents=True,exist_ok=False); torch.set_num_threads(4); seed_all(args.seed); started=time.monotonic()
    device=torch.device(args.device)
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(sources={f:hashlib.sha256((BASE/f).read_bytes()).hexdigest() for f in ['train.py','model.py','features.py']},
         csv_sha256={s:hashlib.sha256((args.data_root/s/f'{s}.csv').read_bytes()).hexdigest() for s in ['train','validation','test']},
         gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,torch_version=str(torch.__version__),
         static_excluded=['bat_class','bat_cc','soc_status','tr_method'],grouping=[[1,2,3],[4,5],[6]],image_backbone='ConvNeXt_Tiny_Weights.IMAGENET1K_V1 frozen',
         image_hierarchy_blend=.3,gate_loss_weights=dict(coarse=.2,pair45=.4,ordinal=.1),sensor_catboost_device='CPU',
         limitations='Validation was examined in earlier experiments; this is exploratory, not a fresh blind test.')
    write_json(out/'config.json',config)
    for f in ['train.py','model.py','features.py']: (out/f).write_bytes((BASE/f).read_bytes())
    official,frames,indices=split_data(args,out)
    raw={s:temporal_features(f) for s,f in official.items()}
    x={name:raw[s].iloc[ids].to_numpy() for name,(s,ids) in indices.items()}; x['names']=raw['train'].columns.tolist()
    normalizer=Normalizer().fit(x['expert']); sensor={s:torch.tensor(normalizer.transform(x[s]),device=device) for s in frames}
    write_json(out/'preprocessing.json',dict(features=x['names'],normalizer=normalizer.state()))
    embeddings=extract_images(official,args,device,out)
    images={name:torch.tensor(np.asarray(embeddings[s][ids]).copy(),device=device) for name,(s,ids) in indices.items()}
    ps=fit_catboost(frames,x,args,out); pi=train_image(frames,images,args,out,device)
    for name in frames:
        if name!='expert':
            save_predictions(out,f'{name}_sensor',frames[name],ps[name]); save_predictions(out,f'{name}_image',frames[name],pi[name])
    ts={s:torch.tensor(v,device=device) for s,v in ps.items()}; ti={s:torch.tensor(v,device=device) for s,v in pi.items()}
    gate=train_gate(frames,sensor,images,ts,ti,args,out,device)
    rows=[]; outputs={}
    for split in ['dev','validation','test']:
        gp,alpha=gate_predict(gate,sensor[split],images[split],ts[split],ti[split])
        outputs[split]=dict(sensor=ps[split],image=pi[split],uniform=.5*(ps[split]+pi[split]),gated=gp)
        save_predictions(out,f'{split}_gated',frames[split],gp,alpha)
        for name,prob in outputs[split].items():
            if split!='test':
                rows.append(dict(split=split,model=name,**scores(frames[split],prob)))
                save_predictions(out,f'{split}_{name}',frames[split],prob,alpha if name=='gated' else None)
        pd.DataFrame(dict(split=split,mean_alpha=[float(alpha.mean())],std_alpha=[float(alpha.std())],min_alpha=[float(alpha.min())],max_alpha=[float(alpha.max())])).to_csv(out/f'{split}_gate_summary.csv',index=False)
    summary=pd.DataFrame(rows); summary.to_csv(out/'summary.csv',index=False)
    sample=pd.read_csv(args.data_root/'sample_submission.csv')
    for name,prob in outputs['test'].items(): make_submission(sample,frames['test'],prob).to_csv(out/f'submission_{name}.csv',index=False)
    make_submission(sample,frames['test'],outputs['test']['gated']).to_csv(out/'submission.csv',index=False)
    selected=summary[summary.split=='dev'].sort_values('macro_f1',ascending=False,kind='stable').iloc[0].model
    make_submission(sample,frames['test'],outputs['test'][selected]).to_csv(out/'submission_dev_best.csv',index=False)
    write_json(out/'submission_info.json',dict(submission_model='gated',dev_best_model=selected,rows=len(sample),columns=['sample_id','label'],external_upload=False))
    validation=summary[summary.split=='validation'].set_index('model')
    deltas=[]
    for ref in ['sensor','image','uniform']:
        deltas.append(dict(candidate='gated',reference=ref,macro_f1_delta=float(validation.loc['gated','macro_f1']-validation.loc[ref,'macro_f1']),F1_4_delta=float(validation.loc['gated','F1_4']-validation.loc[ref,'F1_4']),F1_5_delta=float(validation.loc['gated','F1_5']-validation.loc[ref,'F1_5'])))
    pd.DataFrame(deltas).to_csv(out/'differences.csv',index=False)
    def table(f): return '\n'.join(['| '+' | '.join(map(str,f.columns))+' |','| '+' | '.join(['---']*len(f.columns))+' |']+['| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None)])
    lines=['# Gated hybrid + hierarchy + 4/5 specialist','',table(summary.round(4)),'',table(pd.DataFrame(deltas).round(4)), '',
       '센서 CatBoost + 고정 사전학습 ConvNeXt-Tiny 특징의 학습 가능한 이미지 head. 센서·이미지 expert를 먼저 학습하고, 별도 실험의 gate 학습 데이터에서 alpha를 학습. dev에서만 checkpoint 선택.',
       'p_final = alpha * p_sensor + (1-alpha) * p_image. 단계군 [1,2,3]/[4,5]/[6] 및 그룹 내부 조건부 예측과 ordinal/4·5 보조 손실. 단계군은 설계 가정이며 물리적 인과 학습을 의미하지 않음.',
       '정적 4개 변수 제외. 동일 experiment의 현재·과거 센서만 사용. 현재 열화상 RGB만 사용. CNN backbone은 frozen이며 end-to-end fine-tuning은 하지 않음.',
       '단일 seed의 탐색 실험. validation은 이전 실험에서 이미 확인했으므로 독립적인 신규 최종 평가가 아님. 이전 모델과 학습 데이터·backbone이 달라 단순 점수 차이를 게이트나 계층 단독 효과로 해석하지 않음.',
       'submission.csv는 요청한 gated 모델의 공식 제출 파일. submission_dev_best.csv는 dev 점수 기준 후보 중 최상 모델. 모든 CSV는 sample_id,label 두 열과 공식 ID 순서를 확인함. 외부 제출하지 않음.','',
       f'Elapsed seconds: {time.monotonic()-started:.1f}', '']
    (out/'report.md').write_text('\n'.join(lines))
    print(summary.to_string(index=False),flush=True); print(f'FINISHED {out}',flush=True)

if __name__=='__main__': main()
