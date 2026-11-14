"""CPU frozen-feature probes, fitted only on original ABC fit split. Aggregate exports."""
import argparse,json,warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from abc_reference import abc_fusion_experiment as abc
from diagnose_abc_gas_dependency import restore_preprocessor
from analyze_experiment_performance import confusion,scores


@torch.inference_mode()
def extract(model,data):
    features=[];pred=[];captured={}
    def hook(module,args):captured['f']=args[0]
    h=model.head.register_forward_pre_hook(hook);model.eval()
    try:
        for start in range(0,len(data.frame),128):
            ids=torch.arange(start,min(start+128,len(data.frame)))
            logits=model(*data.batch(ids,'cpu'))
            features.append(captured['f'].cpu().numpy());pred.extend((logits.argmax(1)+1).tolist())
    finally:h.remove()
    return np.concatenate(features),np.asarray(pred)


def run(args):
    if args.output.exists():raise FileExistsError('Choose new output directory')
    torch.set_num_threads(2)
    config=json.loads((args.base_run/'config.json').read_text());prep=restore_preprocessor(args.base_run)
    frames=abc.common.split_frames(args.data_dir,config['split_seed'],.2)
    manifest=json.loads((args.base_run/'image_cache_manifest.json').read_text())
    ck=torch.load(args.base_run/'seed_42/B/best.pt',map_location='cpu',weights_only=True)
    model=abc.FusionModel('B');model.load_state_dict(ck['state'])
    xs={};ys={};normal={}
    for name,frame in frames.items():
        if name in ['dev','validation']:
            ref=pd.read_csv(args.base_run/f'seed_42/B/{name}_predictions.csv')
            if not np.array_equal(ref.sample_id,frame.sample_id) or not np.array_equal(ref.label,frame.label):raise ValueError('Cache/split mismatch')
        pixels=np.load(manifest[name],mmap_mode='c');q,g=prep.transform(frame)
        if pixels.shape!=(len(frame),4,128,128):raise ValueError('Image cache shape')
        data=abc.Data(frame,torch.from_numpy(pixels),torch.from_numpy(q),torch.from_numpy(g),None)
        xs[name],normal[name]=extract(model,data);ys[name]=frame.label.to_numpy(int)
        if name in ['dev','validation'] and not np.array_equal(normal[name],ref.pred):raise ValueError('Frozen baseline predictions differ')
        print('Extracted frozen features for',name,'rows=',len(frame),flush=True)
    d=xs['fit'].shape[1]//5
    blocks={'sensor':list(range(d)),'gas':list(range(d,2*d)),'image':list(range(2*d,3*d)),
      'sensor_image':list(range(d))+list(range(2*d,3*d)), 'gas_image':list(range(d,3*d)),
      'sensor_gas':list(range(2*d)),'all_global':list(range(3*d)),'all_original_head':list(range(5*d))}
    output={'seed':42,'normal_frozen_head':{},'probes':{},'centroid_diagnostics':{},'notes':[
      'Encoders frozen from original B seed42; new regularized linear classifiers fit on original fit only.',
      'All preprocessing/PCA/centroids fit only on fit. No Test access or row exports.',
      'Fixed C=.1, original class weights, no hyperparameter search. Validation results are exploratory.',
      'Gas and image-only use global summaries before attention; encoder training still used all modalities.',
      'Cross terms are outer products of separately standardized PCA8 representations, not physical alignment.',
      'Centroid proximity is descriptive geometry, not proof of label correctness or causal relevance.']}
    valgroups=list(frames['validation'].groupby('experiment_id',sort=True).indices.values())
    for split in ['dev','validation']:output['normal_frozen_head'][split]=scores(confusion(ys[split],normal[split]))
    transformed={name:{} for name in xs}
    for b,start in [('sensor',0),('gas',d),('image',2*d)]:
        scaler=StandardScaler().fit(xs['fit'][:,start:start+d])
        zs={split:scaler.transform(x[:,start:start+d]) for split,x in xs.items()}
        pca=PCA(n_components=8,whiten=True,random_state=42).fit(zs['fit'])
        for split in xs:transformed[split][b]=pca.transform(zs[split])
        c4=zs['fit'][ys['fit']==4].mean(0);c5=zs['fit'][ys['fit']==5].mean(0)
        v=zs['validation'];dist4=((v-c4)**2).mean(1);dist5=((v-c5)**2).mean(1)
        e16=np.zeros(len(v),bool);e16[valgroups[15]]=True
        output['centroid_diagnostics'][b]={}
        for name,mask in [('E16_true5',e16&(ys['validation']==5)),('other_true5',~e16&(ys['validation']==5))]:
            output['centroid_diagnostics'][b][name]={'rows':int(mask.sum()),'fraction_closer_to_fit_class4':float((dist4[mask]<dist5[mask]).mean()),
                'median_distance4':float(np.median(dist4[mask])),'median_distance5':float(np.median(dist5[mask]))}
    cross={}
    for split,x in xs.items():
        t=transformed[split];parts=[x[:,:3*d]]
        for a,b in [('sensor','gas'),('sensor','image'),('gas','image')]:parts.append((t[a][:,:,None]*t[b][:,None,:]).reshape(len(x),-1))
        cross[split]=np.concatenate(parts,1)
    rows=[]
    with threadpool_limits(limits=2):
        for name,idx in list(blocks.items())+[('all_global_cross_terms',None)]:
            x={split:(v[:,idx] if idx is not None else cross[split]) for split,v in xs.items()}
            scaler=StandardScaler().fit(x['fit']);x={k:scaler.transform(v) for k,v in x.items()}
            classifier=LogisticRegression(C=.1,max_iter=300,class_weight={k+1:float(w) for k,w in enumerate(prep.weights)},random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');classifier.fit(x['fit'],ys['fit'])
            result={'fit_warnings':[str(w.message) for w in caught]}
            for split in ['dev','validation']:
                pred=classifier.predict(x[split]);s=scores(confusion(ys[split],pred));result[split]=s
                if split=='validation':
                    result['groups']={f'E{i+1:02}':scores(confusion(ys[split][ids],pred[ids])) for i,ids in enumerate(valgroups)}
                    ids=valgroups[15];e=result['groups']['E16']
                    rows.append(dict(probe=name,dev_f1=result['dev']['macro_f1_6'],validation_f1=s['macro_f1_6'],class5_f1=s['class_f1'][4],E16_f1=e['macro_f1_true_present'],
                        E16_5_to_4=e['confusion'][4][3],E16_5_correct=e['confusion'][4][4]))
            output['probes'][name]=result
            print('Probe',name,json.dumps(rows[-1]),flush=True)
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    pd.DataFrame(rows).to_csv(args.output/'probe_summary.csv',index=False)
    print('CENTROIDS',json.dumps(output['centroid_diagnostics']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run',type=Path,default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1'))
    p.add_argument('--data-dir',type=Path,default=Path('/home/ubuntu/data/battery_data_light'))
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
