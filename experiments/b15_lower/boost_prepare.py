"""Prepare fixed causal/raw-image descriptors; no fitted global transform."""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from abc_reference import abc_fusion_experiment as abc
from boost_model import features,image_grid,PARAMS,SPECS
from reproduce import sha


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    frame=pd.read_csv(args.source/'frame.csv');packed=np.load(args.source/'images.npy',mmap_mode='r')
    rgb=np.empty((len(frame),48),np.float32)
    for start in range(0,len(frame),256):rgb[start:start+256]=image_grid(packed[start:start+256])
    for row in np.random.default_rng(20261115).choice(len(frame),12,replace=False):
        f=frame.iloc[row];path=args.data/f['_official_split']/str(f.image_path).replace('\\','/')
        raw=abc.common.read_image(path,128,4)[0]
        np.testing.assert_array_equal(raw,packed[row]);np.testing.assert_allclose(image_grid(raw),rgb[row],rtol=0,atol=0,equal_nan=True)
    x=features(frame,rgb)
    np.save(args.output/'features.npy',x);np.save(args.output/'rgb.npy',rgb)
    frame.to_csv(args.output/'frame.csv',index=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(
        method='Conditional2/3/4/5 multimodal expert, fixed parameters, nested group policy selection',
        params=PARAMS,specs=SPECS,E16_included=True,leaderboard_input=False,
        protected='all current1/6 labels, original4->3 changes, all S3 changes',
        selection='Inner3 experiment groups, within0.0015 prefer lower alpha and raw feature mode',
        acceptance='Primary full97paired95 lower>0; all5fold deltas>=0; all cadence pooled deltas>=0',
        full_final_training='Raw expert fits all97 causal features only; stack diagnostics require B OOF. Do not report training scores as validation.',
        limitation='Datasets have prior exploration exposure; no new blind benchmark and no hidden-score guarantee.',
        source_hashes={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('boost_model.py')]},
        frame_sha256=sha(args.source/'frame.csv')),indent=2)+'\n')
    print('PREPARED',x.shape,'raw image parity12/12',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','data','output']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
