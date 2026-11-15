"""Fit and run the causal raw50 expert alongside the preserved S3 classifier."""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from boost_model import features,image_grid,fit,probability,combine,PARAMS
from infer import Runtime as BaseRuntime,abc
from pipeline import predict_stages
from reproduce import sha
from validate import summary

SPEC=dict(name='raw_0.5',mode='raw',alpha=.5)


def descriptors(frame,image_root):
    root=Path(image_root).resolve();out=[]
    for value in frame.image_path:
        relative=Path(str(value).replace('\\','/'));path=(root/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):raise ValueError('Invalid image path')
        packed=abc.common.read_image(path,128,4)[0]
        out.append(image_grid(packed))
    return np.asarray(out,np.float32)


def predict(bundle,frame,probs,rgb):
    stages=predict_stages(bundle['baseline'],frame,probs)
    x=features(frame,rgb)
    expert=probability(bundle['expert'],x,probs)
    labels=combine(stages,probs,expert,bundle['spec'])
    return labels,stages,x


def audit_prefix(bundle,frame,p,rgb):
    labels,stages,x=predict(bundle,frame,p,rgb);checks=0
    for ids in frame.groupby('experiment_id',sort=False).indices.values():
        ids=ids[np.argsort(frame.relative_time_s.to_numpy()[ids],kind='stable')]
        for count in sorted(set([1,min(3,len(ids)),min(5,len(ids)),len(ids)//4,len(ids)//2,3*len(ids)//4,len(ids)])):
            if count<1:continue
            take=ids[:count];f=frame.iloc[take].reset_index(drop=True)
            out,_,xx=predict(bundle,f,p[:,take],rgb[take])
            np.testing.assert_allclose(xx,x[take],rtol=0,atol=0,equal_nan=True)
            np.testing.assert_array_equal(out,labels[take]);checks+=1
    order=np.random.default_rng(20261115).permutation(len(frame))
    shuffled=frame.iloc[order].reset_index(drop=True).copy()
    shuffled['sample_id']=[f'unused_{i}' for i in range(len(frame))]
    shuffled['label']=999
    out,_,xx=predict(bundle,shuffled,p[:,order],rgb[order])
    np.testing.assert_array_equal(out,labels[order])
    np.testing.assert_allclose(xx,x[order],rtol=0,atol=0,equal_nan=True)
    return dict(experiments=frame.experiment_id.nunique(),prefixes=checks,feature_and_label_mismatches=0,
                shuffle_and_sample_id_and_label_invariance=True)


class Runtime:
    def __init__(self,base,artifact,device='cpu'):
        self.bundle=joblib.load(artifact)
        self.base=BaseRuntime(base,device=device,artifact=self.bundle['baseline'])

    def labels(self,frame,image_root):
        p=self.base.probabilities(frame,image_root)
        rgb=descriptors(frame,image_root)
        return predict(self.bundle,frame,p,rgb)[0]


def main(args):
    torch.set_num_threads(4);args.output.mkdir(parents=True,exist_ok=False)
    f=pd.read_csv(args.prepared/'frame.csv');x=np.load(args.prepared/'features.npy',mmap_mode='r')
    train=f._official_split.eq('train').to_numpy()
    assert train.sum()==33522 and f.experiment_id[train].nunique()==78
    rgb=np.load(args.prepared/'rgb.npy',mmap_mode='r')
    baseline=joblib.load(args.combined)
    models={}
    for name,mask in [('train78',train),('all97',np.ones(len(f),bool))]:
        expert=fit(f.loc[mask].reset_index(drop=True),x[mask],None,'raw')
        bundle=dict(baseline=baseline,expert=expert,spec=SPEC,protocol='raw50 causal expert; preserve class1/6,43,S3')
        joblib.dump(bundle,args.output/f'{name}.joblib');models[name]=bundle
        print('FIT',name,len(expert['fit_experiments']),flush=True)
    runtime=BaseRuntime(args.base,device='cpu',combined=args.combined)
    validation=f.loc[~train].reset_index(drop=True);p=runtime.probabilities(validation,args.data/'validation')
    np.save(args.output/'validation_probabilities.npy',p)
    val_rgb=rgb[~train]
    pred,stages,_=predict(models['train78'],validation,p,val_rgb)
    val=validation[['sample_id','experiment_id','label']].copy();val['current']=stages['S3'];val['candidate']=pred
    val.to_csv(args.output/'validation_predictions.csv',index=False)
    result=dict(validation=summary(val,'candidate',reference='current'),validation_is_reused_not_blind=True,
                validation_prefix=audit_prefix(models['train78'],validation,p,val_rgb))
    e='E_e5bcddc7d77f459ea541cf658ab4346e'
    g=val[val.experiment_id==e];result['E16']=summary(g,'candidate',reference='current')
    result['E16']['old_true5_pred4']=int(((g.label==5)&(g.current==4)).sum())
    result['E16']['new_true5_pred4']=int(((g.label==5)&(g.candidate==4)).sum())
    print('VALIDATION',result['validation']['macro_f1'],result['validation']['delta'],result['validation']['bootstrap_delta_95'],flush=True)
    test=pd.read_csv(args.data/'test/test.csv');sample=pd.read_csv(args.data/'sample_submission.csv')
    assert test.sample_id.is_unique and set(sample.sample_id)==set(test.sample_id)
    p=runtime.probabilities(test,args.data/'test');rgb=descriptors(test,args.data/'test')
    np.save(args.output/'test_probabilities.npy',p);np.save(args.output/'test_rgb.npy',rgb)
    reference=pd.read_csv(args.reference).set_index('sample_id').loc[test.sample_id].label.to_numpy(int)
    result['submissions']={}
    for name,bundle in models.items():
        pred,stages,_=predict(bundle,test,p,rgb)
        np.testing.assert_array_equal(stages['S3'],reference)
        out=pd.DataFrame(dict(sample_id=test.sample_id,label=pred)).set_index('sample_id').loc[sample.sample_id].reset_index()
        assert len(out)==3451 and out.label.between(1,6).all()
        path=args.output/f'submission_expert50_{name}.csv';out.to_csv(path,index=False,mode='x')
        result['submissions'][name]=dict(file=str(path),rows=len(out),sha256=sha(path),changed_from_current=int((pred!=reference).sum()),
            class6_preserved=bool(np.array_equal(pred==6,reference==6)),prefix=audit_prefix(bundle,test,p,rgb))
        print('SUBMISSION',name,result['submissions'][name]['changed_from_current'],flush=True)
    result['source_hashes']={str(Path(__file__).name):sha(Path(__file__)),
        'boost_model.py':sha(Path(__file__).with_name('boost_model.py')),'baseline':sha(args.combined),'reference_csv':sha(args.reference)}
    result['params']=PARAMS;result['spec']=SPEC
    (args.output/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','base','combined','data','reference','output']:parser.add_argument('--'+name,type=Path,required=True)
    with threadpool_limits(limits=2):main(parser.parse_args())
