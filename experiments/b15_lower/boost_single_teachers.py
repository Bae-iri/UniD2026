"""Additional single-inner-group teachers for leakage-free nested screening."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from train_backbones import train_one
from reproduce import sha


def main(args):
    torch.set_num_threads(3)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for the one-hour training run')
    frame=pd.read_csv(args.source/'frame.csv')
    images=torch.from_numpy(np.load(args.source/'images.npy',mmap_mode='c')).to('cuda')
    jobs=[]
    for k in range(5):
        with np.load(args.source/f'outer_{k}/partitions.npz') as z:
            held=z['held']
            for j in range(3):
                jobs.append(dict(folder=args.output/f'outer_{k}'/f'single_{j}',allowed=z[f'inner_{j}'],
                    held=held,split_seed=20261115+100*(k*7+2*j),selection_fraction=.15))
    count=0
    for plan in jobs:
        for seed in [42,43,44]:
            if count%args.workers==args.worker:
                done=plan['folder']/f'seed_{seed}'/'complete.json'
                if done.exists():
                    r=json.loads(done.read_text())
                    assert sha(done.parent/'best.pt')==r['checkpoint_sha256']
                    assert sha(done.parent/'probabilities.npy')==r['probability_sha256']
                else:train_one(frame,images,plan,seed,15,torch.device('cuda'))
            count+=1


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--worker',type=int,required=True)
    main(p.parse_args())
