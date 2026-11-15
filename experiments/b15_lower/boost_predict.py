"""Regenerate a submission directly from images and sensors, with optional audit."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
from boost_multimodal import Runtime
from reproduce import sha


def main(args):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    f=pd.read_csv(args.input)
    if not f.sample_id.is_unique:raise ValueError('Duplicate sample ID')
    runtime=Runtime(args.base,args.artifact,args.weights,args.device)
    pred=runtime.labels(f,args.image_root,args.policy)
    out=pd.DataFrame(dict(sample_id=f.sample_id,label=pred.astype(int)))
    if args.sample_submission:
        sample=pd.read_csv(args.sample_submission)
        assert sample.sample_id.is_unique and set(sample.sample_id)==set(out.sample_id)
        out=out.set_index('sample_id').loc[sample.sample_id].reset_index()
    assert out.label.between(1,6).all();out.to_csv(args.output,index=False,mode='x')
    report=dict(rows=len(out),sha256=sha(args.output),policy=args.policy or runtime.bundle['policy'],
                bundle_sha256=sha(args.artifact),direct_image_and_sensor_inference=True)
    if args.audit_prefix:
        checks=0
        for ids in f.groupby('experiment_id',sort=False).indices.values():
            ids=ids[np.argsort(f.relative_time_s.to_numpy()[ids],kind='stable')]
            for n in sorted(set([1,max(1,len(ids)//2),len(ids)])):
                take=ids[:n];partial=runtime.labels(f.iloc[take].reset_index(drop=True),args.image_root,args.policy)
                np.testing.assert_array_equal(partial,pred[take]);checks+=1
        perm=np.random.default_rng(20261115).permutation(len(f))
        shuffled=f.iloc[perm].reset_index(drop=True).copy();shuffled['label']=99
        shuffled['sample_id']=[f'ignored_{i}' for i in perm]
        np.testing.assert_array_equal(runtime.labels(shuffled,args.image_root,args.policy),pred[perm])
        report['raw_image_prefix_audit']=dict(prefixes=checks,experiments=f.experiment_id.nunique(),mismatches=0,
            shuffled_and_label_and_sample_id_invariant=True)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input','image-root','base','artifact','weights','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sample-submission',type=Path)
    p.add_argument('--policy',choices=['raw50','visual50','average50','agreement50'])
    p.add_argument('--device',default='cuda');p.add_argument('--audit-prefix',action='store_true')
    with threadpool_limits(limits=2):main(p.parse_args())
