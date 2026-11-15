"""Portable submission entry point; the released archive includes all weights."""
import argparse,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'src/experiments/b15_lower' if (HERE/'src').is_dir() else HERE.parent
sys.path.insert(0,str(SOURCE))
from boost_predict import main
from threadpoolctl import threadpool_limits


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--image-root','--image_root',dest='image_root',type=Path,required=True)
    p.add_argument('--model','--artifact',dest='artifact',type=Path,default=HERE/'model.joblib')
    p.add_argument('--base',type=Path,default=HERE/'base')
    p.add_argument('--weights',type=Path,default=HERE/'resnet18_weights.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample-submission','--sample_submission',dest='sample_submission',type=Path)
    p.add_argument('--policy',choices=['raw50','visual50','average50','agreement50'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--audit-prefix',action='store_true')
    with threadpool_limits(limits=2):main(p.parse_args())
