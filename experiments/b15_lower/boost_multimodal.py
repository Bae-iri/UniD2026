"""Canonical image/sensor expert inference; original B network runs on CPU.

The frozen image encoder uses one current image per call. Histories use only
observations at or before each row. Sample IDs and labels are never features.
"""
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from boost_model import features,image_grid,fit,probability
from boost_family import policies
from boost_visual import Encoder
from infer import Runtime as BaseRuntime,abc
from pipeline import predict_stages
from reproduce import sha
from validate import summary


def descriptors(frame,image_root,encoder):
    root=Path(image_root).resolve();raw=[];visual=[]
    for start in range(0,len(frame),128):
        packed=[]
        for value in frame.image_path.iloc[start:start+128]:
            relative=Path(str(value).replace('\\','/'));path=(root/relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root):raise ValueError('Invalid image path')
            packed.append(abc.common.read_image(path,128,4)[0])
        packed=np.stack(packed);raw.append(image_grid(packed));visual.append(encoder(packed))
    return np.concatenate(raw),np.concatenate(visual)


def predict(bundle,frame,p,rgb,visual):
    x=features(frame,rgb);vx=np.c_[x,visual]
    stages=predict_stages(bundle['baseline'],frame,p)
    preds=policies(stages,p,probability(bundle['experts']['raw'],x,p),probability(bundle['experts']['visual'],vx,p))
    return preds,stages,vx


def prefix_audit(bundle,frame,p,rgb,visual):
    full,_,x=predict(bundle,frame,p,rgb,visual);checks=0
    for ids in frame.groupby('experiment_id',sort=False).indices.values():
        ids=ids[np.argsort(frame.relative_time_s.to_numpy()[ids],kind='stable')]
        for n in sorted(set([1,min(3,len(ids)),min(5,len(ids)),len(ids)//4,len(ids)//2,3*len(ids)//4,len(ids)])):
            if n<1:continue
            take=ids[:n];pred,_,xx=predict(bundle,frame.iloc[take].reset_index(drop=True),p[:,take],rgb[take],visual[take])
            np.testing.assert_allclose(xx,x[take],rtol=0,atol=0,equal_nan=True)
            for name in full:np.testing.assert_array_equal(pred[name],full[name][take])
            checks+=1
    perm=np.random.default_rng(20261115).permutation(len(frame))
    f=frame.iloc[perm].reset_index(drop=True).copy();f['label']=-99;f['sample_id']=[str(i) for i in perm]
    pred,_,xx=predict(bundle,f,p[:,perm],rgb[perm],visual[perm])
    np.testing.assert_allclose(xx,x[perm],rtol=0,atol=0,equal_nan=True)
    for name in full:np.testing.assert_array_equal(pred[name],full[name][perm])
    return dict(experiments=frame.experiment_id.nunique(),prefixes=checks,all_four_policies_mismatches=0,
                raw_feature_mismatches=0,shuffled_and_label_and_sample_id_invariant=True)


class Runtime:
    def __init__(self,base,artifact,weights,device='cuda'):
        self.bundle=joblib.load(artifact)
        if sha(weights)!=self.bundle['weights_sha256']:raise ValueError('Image encoder weights changed')
        self.base=BaseRuntime(base,device='cpu',artifact=self.bundle['baseline'])
        self.encoder=Encoder(weights,device)

    def labels(self,frame,image_root,policy=None):
        p=self.base.probabilities(frame,image_root);rgb,visual=descriptors(frame,image_root,self.encoder)
        preds,_,_=predict(self.bundle,frame,p,rgb,visual)
        return preds[policy or self.bundle['policy']]


def main(args):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    args.output.mkdir(parents=True,exist_ok=False)
    root=args.root;f=pd.read_csv(root/'prepared/frame.csv');train=f._official_split.eq('train').to_numpy()
    x=np.load(root/'prepared/features.npy',mmap_mode='r');rgb=np.load(root/'prepared/rgb.npy',mmap_mode='r')
    visual=np.load(root/'visual_exact/visual.npy',mmap_mode='r');vx=np.c_[x,visual].astype(np.float32)
    weights_sha=sha(args.weights);bundles={}
    for name,mask in [('train78',train),('all97',np.ones(len(f),bool))]:
        old=joblib.load(root/f'deploy_raw50/{name}.joblib')
        expert=fit(f.loc[mask].reset_index(drop=True),vx[mask],None,'raw')
        bundle=dict(baseline=old['baseline'],experts=dict(raw=old['expert'],visual=expert),
                    policy='average50',weights_sha256=weights_sha,
                    feature_protocol='causal sensor216 plus frozen current image512; raw50 fixed blend')
        joblib.dump(bundle,args.output/f'{name}.joblib');bundles[name]=bundle
        print('FIT MULTIMODAL',name,flush=True)
    # Canonical B probabilities came from direct CPU image/sensor inference.
    val=f.loc[~train].reset_index(drop=True);p=np.load(root/'deploy_raw50/validation_probabilities.npy')
    preds,stages,_=predict(bundles['train78'],val,p,rgb[~train],visual[~train])
    table=val[['sample_id','experiment_id','label']].copy();table['current']=stages['S3']
    for name,pred in preds.items():table[name]=pred
    table.to_csv(args.output/'validation_predictions.csv',index=False)
    report=dict(validation={name:summary(table,name,reference='current') for name in preds},
                validation_reused_not_blind=True,
                validation_prefix=prefix_audit(bundles['train78'],val,p,rgb[~train],visual[~train]))
    e='E_e5bcddc7d77f459ea541cf658ab4346e';g=table[table.experiment_id==e]
    report['E16']={name:dict(summary(g,name,reference='current'),
        old_true5_pred4=int(((g.label==5)&(g.current==4)).sum()),new_true5_pred4=int(((g.label==5)&(g[name]==4)).sum())) for name in preds}
    print('VALIDATION MULTIMODAL',{name:round(v['delta'],6) for name,v in report['validation'].items()},flush=True)
    test=pd.read_csv(args.data/'test/test.csv');sample=pd.read_csv(args.data/'sample_submission.csv')
    p=np.load(root/'deploy_raw50/test_probabilities.npy');encoder=Encoder(args.weights,args.device)
    test_rgb,test_visual=descriptors(test,args.data/'test',encoder)
    np.testing.assert_allclose(test_rgb,np.load(root/'deploy_raw50/test_rgb.npy'),rtol=0,atol=0,equal_nan=True)
    np.save(args.output/'test_visual.npy',test_visual)
    reference=pd.read_csv(args.reference).set_index('sample_id').loc[test.sample_id].label.to_numpy(int)
    report['submissions']={}
    for fit_name,bundle in bundles.items():
        preds,stages,_=predict(bundle,test,p,test_rgb,test_visual)
        np.testing.assert_array_equal(stages['S3'],reference)
        report['submissions'][fit_name]={}
        for name,labels in preds.items():
            out=pd.DataFrame(dict(sample_id=test.sample_id,label=labels)).set_index('sample_id').loc[sample.sample_id].reset_index()
            assert len(out)==3451 and out.sample_id.is_unique and out.label.between(1,6).all()
            path=args.output/f'submission_{name}_{fit_name}.csv';out.to_csv(path,index=False,mode='x')
            report['submissions'][fit_name][name]=dict(file=str(path),rows=len(out),sha256=sha(path),
                changes=int((labels!=reference).sum()),stage6_preserved=bool(np.array_equal(labels==6,reference==6)))
        report['submissions'][fit_name]['prefix']=prefix_audit(bundle,test,p,test_rgb,test_visual)
        print('EXPORTED',fit_name,{name:v['changes'] for name,v in report['submissions'][fit_name].items() if name!='prefix'},flush=True)
    report['hashes']={name:sha(Path(__file__).with_name(name)) for name in ['boost_multimodal.py','boost_model.py','boost_family.py','boost_visual.py']}
    report['hashes']['weights']=weights_sha
    (args.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','weights','data','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda')
    with threadpool_limits(limits=2):main(p.parse_args())
