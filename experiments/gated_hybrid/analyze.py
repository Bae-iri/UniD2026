"""Post-training paired experiment bootstrap and error review; does not tune models."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def macro(cm):
    diag=np.diagonal(cm,axis1=-2,axis2=-1); denom=cm.sum(-1)+cm.sum(-2)
    return np.divide(2*diag,denom,out=np.zeros_like(diag,dtype=float),where=denom>0).mean(-1)


def analyze(root):
    arrays={}; reference=None
    predictions={m:pd.read_csv(root/f'validation_{m}_predictions.csv') for m in ['sensor','image','uniform','gated']}
    for name,f in predictions.items():
        ids=f[['sample_id','experiment_id','label']]
        if reference is None: reference=ids
        else: pd.testing.assert_frame_equal(ids,reference)
        arrays[name]=np.stack([confusion_matrix(g.label,g.prediction,labels=range(1,7)) for _,g in f.groupby('experiment_id',sort=True)])
    n=arrays['gated'].shape[0]; rng=np.random.default_rng(20261114)
    weights=rng.multinomial(n,np.full(n,1/n),size=2000)
    draws={m:macro(np.einsum('be,emn->bmn',weights,cm)) for m,cm in arrays.items()}
    summary=pd.read_csv(root/'summary.csv'); val=summary[summary.split=='validation'].set_index('model')
    rows=[]
    for ref in ['sensor','image','uniform']:
        low,high=np.quantile(draws['gated']-draws[ref],[.025,.975])
        rows.append(dict(candidate='gated',reference=ref,delta_pp=100*(val.loc['gated','macro_f1']-val.loc[ref,'macro_f1']),ci_low_pp=100*low,ci_high_pp=100*high))
    pd.DataFrame(rows).to_csv(root/'paired_bootstrap.csv',index=False)
    percent=summary.copy()
    for c in ['macro_f1']+[f'F1_{i}' for i in range(1,7)]: percent[c]*=100
    percent.to_csv(root/'scores_percent.csv',index=False)
    gated=predictions['gated'].copy()
    for m in ['sensor','image','uniform']: gated[f'{m}_prediction']=predictions[m].prediction
    errors=gated[gated.label!=gated.prediction]; errors.to_csv(root/'errors.csv',index=False)
    errors[((errors.label==4)&(errors.prediction==5))|((errors.label==5)&(errors.prediction==4))].to_csv(root/'stage45_errors.csv',index=False)
    errors[(errors.label==5)&(errors.prediction==3)].to_csv(root/'stage5_to3_errors.csv',index=False)
    report=root/'report.md'
    lines=['','## 최종 검증','', '모델별 제출 CSV와 submission.csv의 공식 ID 순서·3,451행·sample_id,label 열·정수 label 1~6을 검증했습니다. submission.csv는 gated 모델이며 dev 점수 기준 최상 후보도 gated입니다.',
       '', '| 비교 | Macro F1 차이 (%p) | 실험 bootstrap 95% 구간 |','|---|---:|---:|']
    for r in rows: lines.append(f'| gated − {r["reference"]} | {r["delta_pp"]:+.2f} | [{r["ci_low_pp"]:+.2f}, {r["ci_high_pp"]:+.2f}] |')
    lines += ['', '구간은 동일 validation experiment를 쌍으로 2,000회 재표집한 결과로, 학습 seed·재학습 변동이나 반복된 모델 선택의 불확실성은 포함하지 않습니다.',
       '', '4/5 혼동뿐 아니라 다른 단계로 이동한 오답을 함께 확인해야 합니다. gated 모델의 실제 5단계 중 484/1,134개만 정답이며(재현율 42.68%), 3단계로 예측한 오답이 479개 남아 있습니다. stage45_errors.csv와 stage5_to3_errors.csv에 센서·이미지 예측과 alpha를 같이 저장했습니다.',
       '', '4→5 26건, 5→4 85건. 4/5 혼동 감소가 5단계의 모든 오류 해결을 의미하지 않습니다. 계층/4·5 손실의 개별 효과는 별도 ablation 없이 확정하지 않습니다.','']
    report.write_text(report.read_text()+'\n'.join(lines))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run',type=Path); analyze(p.parse_args().run)
