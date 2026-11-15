"""Conservative original/S3 fusion. No guarantee for unknown Test labels.

S3 is the default. A fixed, training-only disagreement gate may revert to the
original label. Agreement rows and the shared4->3 corrections are immutable.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pipeline import history, predict_stages
from reproduce import sha

GATE_THRESHOLD=.95
GATE_C=.2
FEATURE_NAMES=['original_is6','original_p','current_p','probability_margin',
               'unanimity_original','p6_seed_sd','temperature_peak_log',
               'temperature_drop_log','sum_rate_signed_log','original6_run_log']


def features(frame,probs,original,current):
    p=np.asarray(probs,float);mean=p.mean(0);n=len(frame)
    if p.shape!=(3,n,6) or not np.isfinite(p).all():
        raise ValueError('Three aligned finite backbone arrays required')
    original=np.asarray(original,int);current=np.asarray(current,int)
    _,peak,drop,_=history(frame)
    t=frame.relative_time_s.to_numpy(float)
    total=pd.to_numeric(frame.meas_t1_plus_t2,errors='coerce').to_numpy(float)
    rate=np.full(n,np.nan);run=np.zeros(n)
    for ids in frame.groupby('experiment_id',sort=False).indices.values():
        ids=ids[np.argsort(t[ids],kind='stable')]
        previous=previous_time=start=np.nan
        for row in ids:
            if np.isfinite(total[row]):
                if np.isfinite(previous) and t[row]>previous_time:
                    rate[row]=(total[row]-previous)/(t[row]-previous_time)
                previous,previous_time=total[row],t[row]
            if original[row]==6:
                if not np.isfinite(start):start=t[row]
                run[row]=np.log1p(t[row]-start)
            else:start=np.nan
    old_p=mean[np.arange(n),original-1]
    new_p=mean[np.arange(n),current-1]
    return np.column_stack([original==6,old_p,new_p,old_p-new_p,
        (p.argmax(2)+1==original[None,:]).mean(0),p[:,:,5].std(0),
        np.log1p(np.maximum(peak,0)),np.log1p(np.maximum(drop,0)),
        np.sign(rate)*np.log1p(abs(rate)),run])


def fit_gate(frame,x,original,current):
    y=frame.label.to_numpy(int);groups=frame.experiment_id.to_numpy()
    old=np.asarray(original,int);new=np.asarray(current,int)
    eligible=(old!=new)&((old==y)^(new==y))&np.isfinite(x).all(1)
    target=(old[eligible]==y[eligible]).astype(int)
    gate=dict(model=None,scaler=None,threshold=GATE_THRESHOLD,feature_names=FEATURE_NAMES,
              fit_experiments=sorted(set(groups)),informative_rows=int(eligible.sum()),
              original_correct_rows=int(target.sum()),current_correct_rows=int((1-target).sum()))
    if eligible.sum()<30 or target.sum()<5 or (1-target).sum()<15:
        gate['disabled_reason']='Insufficient informative training disagreements'
        return gate
    xx=x[eligible]
    scaler=StandardScaler().fit(xx)
    table=pd.DataFrame({'group':groups[eligible],'target':target})
    weight=1/table.groupby(['group','target']).target.transform('size').to_numpy(float)
    model=LogisticRegression(C=GATE_C,max_iter=1000,random_state=42)
    model.fit(scaler.transform(xx),target,sample_weight=weight/weight.mean())
    gate.update(model=model,scaler=scaler,minimum=xx.min(0),maximum=xx.max(0))
    return gate


def apply_gate(gate,x,original,current):
    old=np.asarray(original,int);new=np.asarray(current,int);result=new.copy()
    if gate['model'] is None:return result
    if gate['feature_names']!=FEATURE_NAMES:raise ValueError('Gate feature schema changed')
    valid=np.isfinite(x).all(1)&(old!=new)
    # Bounds are fitted only on informative TRAIN disagreements. This is a
    # conservative extrapolation check, not proof against distribution shift.
    tolerance=1e-8*np.maximum(1,np.maximum(abs(gate['minimum']),abs(gate['maximum'])))
    valid&=((x>=gate['minimum']-tolerance)&(x<=gate['maximum']+tolerance)).all(1)
    if valid.any():
        score=gate['model'].predict_proba(gate['scaler'].transform(x[valid]))[:,1]
        ids=np.flatnonzero(valid)[score>=gate['threshold']]
        result[ids]=old[ids]
    if not np.array_equal(result[old==new],new[old==new]):
        raise AssertionError('Fusion changed an agreement row')
    if not np.all((result==old)|(result==new)):
        raise AssertionError('Fusion invented a third label')
    return result


def predict_fusion(bundle,frame,probs):
    stages=predict_stages(bundle,frame,probs)
    if not bundle.get('fusion_enabled',False):return stages['S3']
    x=features(frame,probs,stages['J096'],stages['S3'])
    return apply_gate(bundle['disagreement_gate'],x,stages['J096'],stages['S3'])


class Runtime:
    def __init__(self,base,artifact,device='cpu'):
        from infer import Runtime as BaseRuntime
        self.base=BaseRuntime(base,combined=artifact,device=device)
        self.bundle=self.base.artifact
        for relative,digest in self.bundle.get('backbone_sha256',{}).items():
            if sha(Path(base)/relative)!=digest:
                raise ValueError('Backbone/preprocessing differs from validated bundle: '+relative)

    def labels(self,frame,image_root):
        return predict_fusion(self.bundle,frame,self.base.probabilities(frame,image_root))


def main(args):
    import torch
    torch.set_num_threads(4)
    frame=pd.read_csv(args.input)
    if not frame.sample_id.is_unique:raise ValueError('Duplicate sample ID')
    labels=Runtime(args.base,args.artifact,args.device).labels(frame,args.image_root)
    table=pd.DataFrame({'sample_id':frame.sample_id,'label':labels.astype(int)})
    if args.sample_submission:
        sample=pd.read_csv(args.sample_submission)
        assert sample.sample_id.is_unique and set(sample.sample_id)==set(table.sample_id)
        table=table.set_index('sample_id').loc[sample.sample_id].reset_index()
    assert table.label.isin(range(1,7)).all()
    table.to_csv(args.output,index=False,mode='x')
    print('Wrote',len(table),'rows:',args.output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input','image-root','base','artifact','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--sample-submission',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    main(parser.parse_args())
