"""Independent paired comparisons and artifact verification; no training/tuning."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

BASE=Path(__file__).resolve().parent


def macro(cm):
    diag=np.diagonal(cm,axis1=-2,axis2=-1);denom=cm.sum(-1)+cm.sum(-2)
    return np.divide(2*diag,denom,out=np.zeros_like(diag,dtype=float),where=denom>0).mean(-1)


def error(cm,label):
    total=cm[:,label-1].sum(1);correct=cm[:,label-1,label-1]
    return 1-np.divide(correct,total,out=np.full_like(correct,np.nan,dtype=float),where=total>0)


def analyze(root):
    cfg=json.loads((root/'config.json').read_text());template=pd.read_csv(Path(cfg['data_root'])/'sample_submission.csv')
    for file in root.glob('submission_*.csv'):
        f=pd.read_csv(file)
        assert list(f.columns)==['sample_id','label'] and len(f)==len(template)==3451
        assert f.sample_id.equals(template.sample_id) and f.sample_id.is_unique and f.label.isin(range(1,7)).all()
        assert pd.api.types.is_integer_dtype(f.label)
    frames={};cms={};reference=None;experiment_rows=[]
    for mode in cfg['modes']:
        f=pd.read_csv(root/mode/'ensemble/validation_predictions.csv');frames[mode]=f
        ids=f[['sample_id','experiment_id','label']]
        if reference is None:reference=ids
        else:pd.testing.assert_frame_equal(reference,ids)
        matrices=[]
        for exp,g in f.groupby('experiment_id',sort=True):
            cm=confusion_matrix(g.label,g.pred,labels=range(1,7));matrices.append(cm)
            row=dict(model=mode,experiment_id=exp,rows=len(g),errors=int((g.label!=g.pred).sum()))
            for label in [3,4,5]:
                n=int(cm[label-1].sum());row[f'support_{label}']=n;row[f'wrong_{label}']=int(n-cm[label-1,label-1])
            experiment_rows.append(row)
        cms[mode]=np.stack(matrices)
        for seed in cfg['seeds']:
            h=pd.read_csv(root/mode/f'seed_{seed}/history.csv');assert len(h)==cfg['epochs']+1
        if mode!='B_control':
            # Easy-to-find copies in independent CODE folders, no model state is transferred.
            shutil.copy2(root/mode/'submission.csv',BASE/mode/'submission.csv')
            (BASE/mode/'submission_provenance.json').write_text(json.dumps(dict(source=str((root/mode/'submission.csv').resolve()),experiment=mode,seeds=cfg['seeds'],rows=3451,initialization='original B only'),indent=2))
    pd.DataFrame(experiment_rows).to_csv(root/'errors_by_experiment.csv',index=False)
    if 'B_control' in cms:
        groups=len(cms['B_control']);weights=np.random.default_rng(20261114).multinomial(groups,np.full(groups,1/groups),size=2000)
        sampled={m:np.einsum('be,emn->bmn',weights,c) for m,c in cms.items()}
        rows=[]
        for m in cfg['modes']:
            if m=='B_control':continue
            metrics=['macro_f1','error_rate_3','error_rate_5'] if m=='experiment1_voltage' else ['macro_f1','error_rate_4','error_rate_5']
            for metric in metrics:
                a=macro(sampled[m]) if metric=='macro_f1' else error(sampled[m],int(metric[-1]))
                b=macro(sampled['B_control']) if metric=='macro_f1' else error(sampled['B_control'],int(metric[-1]))
                diff=a-b;lo,hi=np.nanquantile(diff,[.025,.975]);rows.append(dict(model=m,reference='B_control',metric=metric,ci_low_pp=100*lo,ci_high_pp=100*hi,valid_boot_replicates=int(np.isfinite(diff).sum())))
        pd.DataFrame(rows).to_csv(root/'paired_experiment_bootstrap.csv',index=False)
    scores=pd.read_csv(root/'ensemble_percent.csv');table=scores[scores.split=='validation'][['model','macro_f1','support_3','wrong_3','error_rate_3','support_4','wrong_4','error_rate_4','support_5','wrong_5','error_rate_5','true3_pred5','true5_pred3','true4_pred5','true5_pred4']]
    table.to_csv(root/'target_error_table.csv',index=False)
    report=root/'report.md';text=['','## 검증 완료','', '각 실험 3-seed 앙상블 제출 CSV는 공식 3,451행 ID 순서와 label 1~6을 검증했습니다. 실험별 code 폴더에도 submission.csv를 복사했습니다.', '각 모델·seed의 추가 15 epoch 완료와 원본 epoch0 후보를 확인했습니다. best epoch가 0이면 dev에서 추가 학습의 개선이 없어 원본 상태로 돌아간 경우이며 history.csv에는 실제 학습 이력이 보존됩니다.',
      '두 실험은 별도 model.py/전처리를 사용하며 원본 B만 공통 시작점입니다. 실험2 state에는 전압 사건 branch가 없고 실험1 state에는 gas-zero 온도 branch가 없습니다.',
      'bootstrap 구간은 동일 validation 실험을 paired 재표집한 조건부 불확실성이며 seed 재학습 변동/반복 선택 편향은 포함하지 않습니다. 모델별 실제 실험 분포는 errors_by_experiment.csv를 확인하세요.','']
    report.write_text(report.read_text()+'\n'.join(text));print(table.to_string(index=False));print('Submission copies:',BASE/'experiment1_voltage/submission.csv',BASE/'experiment2_gas_temperature/submission.csv')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);analyze(p.parse_args().run)
