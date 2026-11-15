"""Simultaneous bootstrap check for the finite policy search, not blind testing."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from validate import matrix,macro


def main(args):
    q=pd.read_csv(args.family/'all97_predictions.csv');q=q[q.view=='original'].reset_index(drop=True)
    groups=list(q.groupby('experiment_id').indices.values())
    before=np.stack([matrix(q.label.to_numpy()[i],q.current.to_numpy()[i]) for i in groups])
    weights=np.random.default_rng(20261115).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=100000)
    baseline=macro(np.einsum('be,eij->bij',weights,before));results={}
    # Include the additional rejected bagging policies and fit-scope variants.
    # This count covers this run, not earlier manual exploration of the data.
    family_size=32;tail=.025/family_size
    for name in ['raw50','visual50','average50','agreement50']:
        after=np.stack([matrix(q.label.to_numpy()[i],q[name].to_numpy()[i]) for i in groups])
        delta=macro(np.einsum('be,eij->bij',weights,after))-baseline
        results[name]=dict(delta=float(macro(after.sum(0))-macro(before.sum(0))),
            ordinary_paired_95=np.quantile(delta,[.025,.975]).tolist(),
            selection_adjusted_paired_95=np.quantile(delta,[tail,1-tail]).tolist())
    args.output.write_text(json.dumps(dict(results=results,draws=len(weights),family_size=family_size,
        resampling_unit='experiment',limitation='Approximate CV bootstrap; folds share training data; no blind/private guarantee.'),indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
