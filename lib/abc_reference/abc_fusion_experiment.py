"""A/B/C current-observation fusion comparison, without static covariates."""
import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit

from . import hierarchy_experiment as common

BASE = Path(__file__).resolve().parent
ELECTRO, GAS = common.ELECTRO, common.GAS
NUMERIC = ELECTRO + GAS
MODES = ['A', 'B', 'C']
EXCLUDED = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']


class SensorPreprocessor:
    def __init__(self, fit):
        numeric = self.numeric(fit)
        self.median = numeric.median().fillna(0)
        filled = numeric.fillna(self.median)
        self.mean, self.std = filled.mean(), filled.std(ddof=0).clip(lower=1e-6)
        counts = np.bincount(fit.label.to_numpy(int)-1, minlength=6)
        if (counts == 0).any():
            raise ValueError('All six classes must be present in fit')
        weights = counts.astype(np.float32)**-0.5
        self.weights = weights / weights.mean()

    @staticmethod
    def numeric(frame):
        return frame[NUMERIC].astype(float).replace([np.inf, -np.inf], np.nan)

    def transform(self, frame):
        numeric = self.numeric(frame)
        missing = numeric.isna().to_numpy(np.float32)
        values = ((numeric.fillna(self.median)-self.mean)/self.std).to_numpy(np.float32)
        q = np.concatenate([values[:, :len(ELECTRO)], missing[:, :len(ELECTRO)]], axis=1)
        gas = np.stack([values[:, len(ELECTRO):], missing[:, len(ELECTRO):]], axis=-1)
        return q.astype(np.float32), gas.astype(np.float32)

    def metadata(self):
        return dict(electro=ELECTRO, gas=GAS, excluded=EXCLUDED, query_dim=2*len(ELECTRO),
                    median=self.median.tolist(), mean=self.mean.tolist(), std=self.std.tolist(),
                    class_weights=self.weights.tolist(), temporal_features=False)


class FusionReadout(common.Readout):
    def forward(self, tokens, q, valid, mode):
        if mode == 'C':
            q = self.fixed_query.expand(len(q), -1)
        return super().forward(tokens, q, valid, 'flat' if mode == 'A' else 'conditional')


class FusionModel(nn.Module):
    def __init__(self, mode, d=64):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.q_encoder = nn.Sequential(nn.Linear(2*len(ELECTRO), d), nn.ReLU(), nn.Linear(d, d))
        self.gas_id = nn.Embedding(len(GAS), 8)
        self.register_buffer('gas_indices', torch.arange(len(GAS)))
        self.gas_encoder = nn.Sequential(nn.Linear(10, d), nn.ReLU(), nn.Linear(d, d))
        layers = []
        for cin, cout in [(3,16), (16,32), (32,64), (64,d)]:
            layers += [nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8,cout), nn.ReLU()]
        self.cnn = nn.Sequential(*layers)
        self.position = nn.Linear(2,d,bias=False)
        self.gas_readout, self.image_readout = FusionReadout(d), FusionReadout(d)
        self.head = nn.Sequential(nn.Linear(d*5,128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128,6))

    def forward(self, image, image_mask, q_input, gas_input):
        q = self.q_encoder(q_input)
        identity = self.gas_id(self.gas_indices).unsqueeze(0).expand(len(q),-1,-1)
        gas = self.gas_encoder(torch.cat([gas_input,identity],-1))
        gas_valid = gas_input[...,1] < .5
        features = self.cnn(image)
        h,w = features.shape[-2:]
        yy,xx = torch.meshgrid(torch.linspace(-1,1,h,device=image.device),
                               torch.linspace(-1,1,w,device=image.device), indexing='ij')
        xy = torch.stack([xx,yy],-1).reshape(h*w,2)
        pixels = features.flatten(2).transpose(1,2) + self.position(xy).unsqueeze(0)
        pixel_valid = F.adaptive_max_pool2d(image_mask.float(),(h,w)).flatten(1) > 0
        # These first three direct paths are identical in every model, including C.
        combined = torch.cat([q, common.masked_mean(gas,gas_valid), common.masked_mean(pixels,pixel_valid),
                              self.gas_readout(gas,q,gas_valid,self.mode),
                              self.image_readout(pixels,q,pixel_valid,self.mode)],-1)
        return self.head(combined)


def load_frames(args):
    train = pd.read_csv(args.data_root/'train/train.csv')
    val = pd.read_csv(args.data_root/'validation/validation.csv')
    test = pd.read_csv(args.data_root/'test/test.csv')
    if any(set(x.experiment_id)&set(y.experiment_id) for x,y in [(train,val),(train,test),(val,test)]):
        raise ValueError('Experiment overlap between official splits')
    indices = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=args.split_seed).split(train,groups=train.experiment_id))
    frames = {'fit':train.iloc[indices[0]], 'dev':train.iloc[indices[1]], 'validation':val, 'test':test}
    for name,frame in frames.items():
        if not frame.sample_id.is_unique:
            raise ValueError(f'Duplicate sample IDs: {name}')
        if args.smoke:
            frame = frame.sample(min(len(frame),384 if name=='fit' else 192),random_state=args.split_seed)
        frames[name] = frame.reset_index(drop=True)
        if name!='test' and set(frame.label)!=set(range(1,7)):
            raise ValueError(f'Missing/invalid class in {name}')
    return frames


@dataclass
class Data:
    frame: pd.DataFrame
    images: torch.Tensor
    q: torch.Tensor
    gas: torch.Tensor
    y: torch.Tensor | None

    def batch(self, ids, device):
        ids = ids.to(self.q.device)
        packed = self.images.index_select(0,ids)
        return (packed[:,:3].to(device,dtype=torch.float32).div_(255), packed[:,3:4].to(device),
                self.q.index_select(0,ids).to(device), self.gas.index_select(0,ids).to(device))


def prepare(args,frames,prep,device):
    cached = {name:common.cache_images(args.data_root,frame,'train' if name in ['fit','dev'] else name,
                                      BASE/'.hierarchy_cache',128,4,8)[0] for name,frame in frames.items()}
    required = sum(v.nbytes for v in cached.values())
    resident = device if device.type=='cuda' and torch.cuda.mem_get_info()[0]>required+3*1024**3 else torch.device('cpu')
    print(f'Image cache {required/1024**3:.2f} GiB on {resident}',flush=True)
    cache_manifest = {name:str(images.filename) for name,images in cached.items()}
    path = args.output/'image_cache_manifest.json'
    if path.exists() and json.loads(path.read_text())!=cache_manifest:
        raise ValueError('Image files changed; use a fresh output directory')
    common.write_json(path,cache_manifest)
    datasets = {}
    for name,frame in frames.items():
        q,gas = prep.transform(frame)
        images = torch.from_numpy(cached.pop(name)).to(resident)
        y = torch.tensor(frame.label.to_numpy(int)-1,device=resident) if 'label' in frame else None
        datasets[name] = Data(frame,images,torch.from_numpy(q).to(resident),torch.from_numpy(gas).to(resident),y)
    return datasets


@torch.no_grad()
def predict(model,data,args,device):
    model.eval()
    probabilities=[]
    for start in range(0,len(data.frame),128):
        ids=torch.arange(start,min(start+128,len(data.frame)))
        probabilities.append(model(*data.batch(ids,device)).softmax(-1).cpu().numpy())
    return np.concatenate(probabilities)


def metrics(frame,prob):
    y,pred=frame.label.to_numpy(int),prob.argmax(1)+1
    f1=f1_score(y,pred,labels=range(1,7),average=None,zero_division=0)
    return dict(macro_f1=float(f1.mean()),**{f'F1_{i+1}':float(value) for i,value in enumerate(f1)})


def save_predictions(folder,split,frame,prob):
    columns=['sample_id','experiment_id']+(['label'] if 'label' in frame else [])
    result=frame[columns].copy()
    result['pred']=prob.argmax(1)+1
    for i in range(6): result[f'p{i+1}']=prob[:,i]
    result.to_csv(folder/f'{split}_predictions.csv',index=False)
    if split=='validation':
        pd.DataFrame(classification_report(frame.label,result.pred,labels=range(1,7),output_dict=True,zero_division=0)).T.to_csv(folder/'class_report.csv')
        pd.DataFrame(confusion_matrix(frame.label,result.pred,labels=range(1,7)),index=range(1,7),columns=range(1,7)).to_csv(folder/'confusion_matrix.csv',index_label='true_label')


def run_one(mode,seed,state,datasets,prep,args,device):
    folder=args.output/f'seed_{seed}'/mode
    folder.mkdir(parents=True,exist_ok=True)
    if args.resume and (folder/'metrics.json').exists():
        return json.loads((folder/'metrics.json').read_text())
    common.seed_everything(seed)
    model=FusionModel(mode).to(device)
    model.load_state_dict(state)
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(prep.weights,device=device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    fit=datasets['fit']; generator=torch.Generator().manual_seed(seed)
    best=-1; history=[]; active=None; started=time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; count=0.
        order=torch.randperm(len(fit.frame),generator=generator)
        for start in range(0,len(order),64):
            ids=order[start:start+64]
            y=fit.y.index_select(0,ids.to(fit.y.device)).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(*fit.batch(ids,device)),y)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward()
            if active is None: active=sum(p.numel() for p in model.parameters() if p.grad is not None)
            nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            total+=loss.item()*len(ids); count+=len(ids)
        dev=metrics(datasets['dev'].frame,predict(model,datasets['dev'],args,device))['macro_f1']
        if dev>best:
            best=dev
            torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            epoch=epoch,dev_macro_f1=dev,mode=mode,seed=seed),folder/'best.pt')
        history.append(dict(epoch=epoch,train_ce=total/count,dev_macro_f1=dev))
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        print(f'{seed} {mode} {epoch:02}/{args.epochs} CE {total/count:.4f} dev {dev:.4f} best {best:.4f}',flush=True)
    checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state'])
    scores={}
    for split in ['dev','validation','test']:
        prob=predict(model,datasets[split],args,device)
        save_predictions(folder,split,datasets[split].frame,prob)
        if split=='validation': scores=metrics(datasets[split].frame,prob)
    scores.update(model=mode,seed=seed,best_dev_epoch=checkpoint['epoch'],best_dev_f1=best,
                  parameters_total=sum(p.numel() for p in model.parameters()),parameters_active=active,
                  minutes=(time.monotonic()-started)/60)
    common.write_json(folder/'metrics.json',scores)
    print(f'FINAL {seed} {mode} validation {scores["macro_f1"]:.4f}',flush=True)
    del model,optimizer
    if device.type=='cuda': torch.cuda.empty_cache()
    return scores


def markdown_table(frame):
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)])


def summarize(results,frames,args):
    out=args.output; summary=pd.DataFrame(results)
    summary.to_csv(out/'summary.csv',index=False)
    columns=['macro_f1']+[f'F1_{i}' for i in range(1,7)]
    grouped=summary.groupby('model')[columns]
    means,stds=grouped.mean(),grouped.std().fillna(0)
    aggregate=means.add_suffix('_mean').join(stds.add_suffix('_std'))
    aggregate.to_csv(out/'aggregate.csv')
    cm={}; ensemble={}; tidy=[]; confusion_tables={}
    for model in MODES:
        per_seed=[]
        for seed in args.seeds:
            pred=pd.read_csv(out/f'seed_{seed}'/model/'validation_predictions.csv')
            if pred.sample_id.tolist()!=frames['validation'].sample_id.tolist(): raise ValueError('Row mismatch')
            per_seed.append(np.stack([confusion_matrix(g.label,g.pred,labels=range(1,7))
                                      for _,g in pred.groupby('experiment_id',sort=True)]))
        cm[model]=np.stack(per_seed)
        mean_cm=cm[model].sum(1).mean(0)
        pd.DataFrame(mean_cm,index=range(1,7),columns=range(1,7)).to_csv(out/f'confusion_{model}_mean.csv',index_label='true_label')
        confusion_tables[model]=pd.DataFrame(mean_cm,index=range(1,7),columns=range(1,7))
        for i in range(6):
            for j in range(6):
                tidy.append(dict(row_type='confusion_mean',model=model,true_label=i+1,pred_label=j+1,metric='count',value=mean_cm[i,j]))
        ensemble[model]={}
        for split in ['dev','validation','test']:
            probabilities=[]
            for seed in args.seeds:
                p=pd.read_csv(out/f'seed_{seed}'/model/f'{split}_predictions.csv')
                if p.sample_id.tolist()!=frames[split].sample_id.tolist(): raise ValueError('Ensemble row mismatch')
                probabilities.append(p[[f'p{i}' for i in range(1,7)]].to_numpy())
            prob=np.mean(probabilities,axis=0)
            ensemble[model][split]=prob
            save_predictions(out,f'ensemble_{model}_{split}',frames[split],prob)
        for metric in columns:
            tidy.append(dict(row_type='aggregate',model=model,metric=metric,value=means.loc[model,metric],std=stds.loc[model,metric]))
    for row in results:
        for metric in columns:
            tidy.append(dict(row_type='run',model=row['model'],seed=row['seed'],metric=metric,value=row[metric]))
    n_groups=cm['A'].shape[1]
    weights=np.random.default_rng(args.split_seed+971).multinomial(n_groups,np.full(n_groups,1/n_groups),size=2000)
    boot={m:common.macro_from_confusions(np.einsum('be,semn->bsmn',weights,cms)).mean(1) for m,cms in cm.items()}
    contrasts=[]
    for reference in ['A','C']:
        low,high=np.quantile(boot['B']-boot[reference],[.025,.975])
        delta=means.loc['B','macro_f1']-means.loc[reference,'macro_f1']
        contrasts.append(dict(candidate='B',reference=reference,delta=delta,ci_low=low,ci_high=high))
        tidy.append(dict(row_type='contrast',model='B',reference=reference,metric='macro_f1_delta',value=delta,ci_low=low,ci_high=high))
    pd.DataFrame(contrasts).to_csv(out/'comparisons.csv',index=False)
    seed_deltas=summary.pivot(index='seed',columns='model',values='macro_f1')
    seed_deltas['B_minus_A']=seed_deltas.B-seed_deltas.A
    seed_deltas['B_minus_C']=seed_deltas.B-seed_deltas.C
    seed_deltas.to_csv(out/'seed_deltas.csv')
    pd.DataFrame(tidy).to_csv(out/'results_combined.csv',index=False)
    # Submission selection uses dev ensemble scores only, never final validation or test labels.
    ensemble_scores=pd.DataFrame([dict(model=m,split=s,**metrics(frames[s],ensemble[m][s]))
                                  for m in MODES for s in ['dev','validation']])
    ensemble_scores.to_csv(out/'ensemble_scores.csv',index=False)
    selected=ensemble_scores[ensemble_scores.split=='dev'].sort_values('macro_f1',ascending=False,kind='stable').iloc[0].model
    sample=pd.read_csv(args.data_root/'sample_submission.csv')
    if args.smoke: sample=sample[sample.sample_id.isin(frames['test'].sample_id)].copy()
    if list(sample.columns)!=['sample_id','label'] or not sample.sample_id.is_unique: raise ValueError('Invalid template')
    for m in MODES:
        submission=make_submission(sample,frames['test'],ensemble[m]['test'])
        submission.to_csv(out/f'submission_{m}.csv',index=False)
        if m==selected: submission.to_csv(out/'submission.csv',index=False)
    common.write_json(out/'submission_info.json',dict(selected_model=selected,selection='highest dev ensemble macro-F1',
                                                    seed_ensemble=args.seeds,rows=len(sample),columns=['sample_id','label'],smoke=args.smoke))
    labels={'A':'균등 pooling','B':'현재 상태 Query','C':'학습 가능한 고정 Query'}
    table=means.round(4).reset_index(); table.insert(1,'설명',[labels[m] for m in table.model]); table['seed_std']=stds.macro_f1.values.round(4)
    lines=['# A/B/C 상태 조건부 결합 실험','',f'Smoke={args.smoke}; seeds={args.seeds}; epochs={args.epochs}.','',
           '정적 변수 4개 제외. 현재 시점 전기·열 6개와 결측 표시만으로 Query 구성. meas_a는 train에서 항상 0이므로 제외. 시계열 특징 없음.',
           '세 모델 모두 동일한 encoder/head/초기 state/배치 순서/weighted CE/AdamW 설정. 전기·열과 가스·이미지 전체 요약의 직접 연결 유지.',
           'A는 균등 pooling, B는 상태 Query, C는 두 readout의 학습 가능한 고정 Query. 선택 모듈의 Query만 변경하며 최종 head에는 실제 상태를 항상 전달.',
           'train을 실험 단위 fit/dev(62/16개)로 분리하고 dev로 checkpoint 선택. 제공 validation(19개 실험)은 최종 평가에 사용. 전처리는 fit에서만 계산.','',
           '## 전체 및 클래스별 F1 (seed 평균)','',markdown_table(table),'',
           '## B−A 및 B−C','',markdown_table(pd.DataFrame(contrasts).round(4)),'',
           '95% 구간은 experiment_id 단위 2,000회 paired bootstrap. 고정된 학습 결과의 검증 실험 표본 불확실성이며 seed 변동은 표준편차/개별 차이로 별도 표시.','',
           '점추정 B>A 및 B>C: '+str(all(r['delta']>0 for r in contrasts)),
           '두 비교의 95% 하한이 모두 양수: '+str(all(r['ci_low']>0 for r in contrasts)),
           '가스와 이미지 attention을 동시에 변경한 결합 효과다. 이미지 단독 효과나 물리적 인과 관계로 해석하지 않는다.','',
           '## 혼동행렬 (seed 평균 count)','행은 실제 label, 열은 예측 label. 여러 seed의 count 평균이며 앙상블 혼동행렬과 다르다.','']
    for m in MODES:
        frame=confusion_tables[m].round(1).reset_index().rename(columns={'index':'true / pred'})
        lines += [f'### {m}', '', markdown_table(frame),'']
    lines += ['## 제출 파일','',f'`submission.csv`: dev 앙상블 점수로 선택한 {selected}, {len(sample)}행. 공식 sample_submission의 ID 순서 유지.',
              '`submission_A.csv`, `submission_B.csv`, `submission_C.csv`도 개별 저장. 제출 양식은 sample_id,label 두 열.',
              '`results_combined.csv`에는 실행별/평균 F1·평균 혼동행렬·B−A/B−C 차이를 long format으로 통합.',
              '최종 test 점수는 label이 없어 알 수 없다. 실제 서비스 업로드는 수행하지 않는다.','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(table.to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)
    print(f'Submission: {out / "submission.csv"} (model {selected})',flush=True)


def make_submission(sample,test,prob):
    if set(sample.sample_id)!=set(test.sample_id) or not test.sample_id.is_unique:
        raise ValueError('Submission IDs do not exactly match test')
    predictions=pd.Series(prob.argmax(1)+1,index=test.sample_id)
    result=sample.copy(); result['label']=result.sample_id.map(predictions)
    if result.label.isna().any() or not result.label.isin(range(1,7)).all(): raise ValueError('Invalid labels')
    result['label']=result.label.astype(int)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=BASE.parent/'data/battery_data_light')
    p.add_argument('--output',type=Path,default=BASE/'abc_runs/current_query_v1')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--split-seed',type=int,default=20261114)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.smoke: args.seeds,args.epochs=[args.seeds[0]],1
    torch.set_num_threads(4)
    if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; run on GPU host')
    device=torch.device(args.device); args.output.mkdir(parents=True,exist_ok=True)
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ['output','resume']}
    config['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['common_sha256']=hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest()
    config['csv_sha256']={s:hashlib.sha256((args.data_root/f'{s}/{s}.csv').read_bytes()).hexdigest() for s in ['train','validation','test']}
    config['template_sha256']=hashlib.sha256((args.data_root/'sample_submission.csv').read_bytes()).hexdigest()
    config.update(batch_size=64,lr=3e-4,loss='weighted_cross_entropy',image_size=128,feature_dim=64,
                  excluded=EXCLUDED,torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name() if device.type=='cuda' else None)
    path=args.output/'config.json'
    if path.exists() and (not args.resume or json.loads(path.read_text())!=config): raise ValueError('Existing run/config differs')
    common.write_json(path,config)
    (args.output/'source.py').write_bytes(Path(__file__).read_bytes())
    (args.output/'common_source.py').write_bytes(Path(common.__file__).read_bytes())
    frames=load_frames(args)
    for name,frame in frames.items(): print(name,len(frame),frame.experiment_id.nunique(),'experiments',flush=True)
    pd.concat([f[['sample_id','experiment_id']].assign(split=n) for n,f in frames.items()]).to_csv(args.output/'split_manifest.csv',index=False)
    prep=SensorPreprocessor(frames['fit']); common.write_json(args.output/'preprocessing.json',prep.metadata())
    datasets=prepare(args,frames,prep,device); results=[]
    for seed in args.seeds:
        common.seed_everything(seed); state=copy.deepcopy(FusionModel('A').state_dict())
        for mode in MODES:
            results.append(run_one(mode,seed,state,datasets,prep,args,device))
            pd.DataFrame(results).to_csv(args.output/'summary.csv',index=False)
    summarize(results,frames,args)


if __name__=='__main__': main()
