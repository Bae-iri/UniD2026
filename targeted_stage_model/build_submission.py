"""Create review candidates only after training and causality audits pass.

Test observations are used only for fixed inference and prefix checks. The
existing public-0.93 submission is reproduced and preserved byte-for-byte.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from targeted_stage_model.model import infer
from targeted_stage_model.run import DATA, BASE, SEEDS
from temporal45_model.causal_rules import predict_rules
import final_rules3


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(run):
    assert json.loads((run/'audit_checks.json').read_text())['passed']
    decision=json.loads((run/'decision.json').read_text())
    assert decision['accepted']=={'45':True,'56':True}
    for path,digest in json.loads((run/'protocol.json').read_text())['source_sha256'].items():
        assert sha(Path(path))==digest
    artifact=joblib.load(run/'deployment_heads.joblib')
    assert artifact['accepted']==decision['accepted']
    baseline_path=ROOT/'final_submit/submission_rules3.csv'
    baseline_hash=sha(baseline_path)
    original=pd.read_csv(baseline_path)
    frame=pd.read_csv(DATA/'test/test.csv')
    assert 'label' not in frame and frame.sample_id.is_unique
    p=np.stack([pd.read_csv(BASE/f'seed_{s}/B/test_predictions.csv').set_index('sample_id').loc[
                 frame.sample_id,[f'p{i}' for i in range(1,7)]].to_numpy() for s in SEEDS])
    raw_reference=pd.read_csv(BASE/'submission_B.csv').set_index('sample_id').loc[frame.sample_id,'label'].to_numpy(int)
    np.testing.assert_array_equal(p.mean(0).argmax(1)+1,raw_reference)
    # Preserve the public baseline's existing threshold recipe. The specialists
    # and their thresholds are not fitted or adjusted here.
    labeled=pd.concat([pd.read_csv(DATA/'train/train.csv'),pd.read_csv(DATA/'validation/validation.csv')])
    floor=final_rules3.t1_floor(labeled[labeled.experiment_id!=final_rules3.E16])
    thresholds={'floor3':{'by_capacity':floor[0],'fallback':floor[1]}}
    base=predict_rules(frame,raw_reference,thresholds)[0]['rules3']
    np.testing.assert_array_equal(base,pd.Series(original.label.to_numpy(),index=original.sample_id).loc[frame.sample_id])
    outputs=infer(artifact,frame,p,base)
    prefix_checks=0
    for ids in frame.groupby('experiment_id').indices.values():
        ids=ids[np.argsort(frame.relative_time_s.to_numpy()[ids],kind='stable')]
        for end in sorted(set([1,5,max(1,len(ids)//2),len(ids)])):
            if end>len(ids):continue
            subset=ids[:end];f=frame.iloc[subset].reset_index(drop=True);pp=p[:,subset]
            pb=predict_rules(f,pp.mean(0).argmax(1)+1,thresholds)[0]['rules3']
            np.testing.assert_array_equal(pb,base[subset])
            predicted=infer(artifact,f,pp,pb)
            for name in outputs:np.testing.assert_array_equal(predicted[name],outputs[name][subset])
            prefix_checks+=1
    files=[]
    for head,name in [('both','submission_targeted45_56_past.csv'),('56','submission_targeted56_past.csv')]:
        path=ROOT/'final_submit'/name
        if path.exists():raise FileExistsError(path)
        labels=pd.Series(outputs[head],index=frame.sample_id).loc[original.sample_id].to_numpy(int)
        sub=pd.DataFrame({'sample_id':original.sample_id,'label':labels})
        assert len(sub)==3451 and sub.sample_id.is_unique and sub.label.isin(range(1,7)).all()
        before=original.label.to_numpy(int)
        allowed=((before==4)&(labels==5))|((before==5)&(labels==6))
        assert ((before==labels)|allowed).all()
        sub.to_csv(path,index=False,mode='x')
        files.append(dict(file=str(path),rows=len(sub),changed=int((before!=labels).sum()),
                          pred4_to5=int(((before==4)&(labels==5)).sum()),
                          pred5_to6=int(((before==5)&(labels==6)).sum()),sha256=sha(path)))
    assert sha(baseline_path)==baseline_hash
    result=dict(files=files,test_prefix_checks=prefix_checks,original_submission_unchanged=True,
                model_selection_used_test=False,model_thresholds=artifact['thresholds'],
                public_score=None,source_sha256=sha(Path(__file__)))
    (run/'submission_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    with threadpool_limits(limits=2):main(p.parse_args().run)
