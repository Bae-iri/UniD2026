"""Apply a saved gate and past-only rules with exact sample-ID alignment."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from temporal45_model.conservative import infer
from temporal45_model.causal_rules import predict_rules


def predict(artifact, frame, thresholds, seeds):
    if not frame.sample_id.is_unique:
        raise ValueError('Duplicate sample_id')
    required = set(frame.sample_id)
    probs = []
    for seed in seeds:
        if not seed.sample_id.is_unique or set(seed.sample_id) != required:
            raise ValueError('Seed probability IDs do not exactly match observations')
        probs.append(seed.set_index('sample_id').loc[frame.sample_id, [f'p{i}' for i in range(1, 7)]].to_numpy(float))
    p = np.stack(probs)
    # Rebuild the baseline ourselves; an externally supplied legacy prediction
    # file must not silently reintroduce a rule that reads future observations.
    base = predict_rules(frame, p.mean(0).argmax(1)+1, thresholds)[0]['rules3']
    result = infer(artifact, frame, p, base)
    return pd.DataFrame(dict(sample_id=frame.sample_id.to_numpy(), pred=result, baseline=base))


def main(args):
    if args.output.exists():
        raise FileExistsError('Use a new output filename')
    artifact = joblib.load(args.model)
    if not artifact.get('enabled', False):
        print('Deployment gate is disabled: output preserves the supplied baseline.', flush=True)
        artifact = dict(artifact, threshold=1.01)
    result = predict(artifact, pd.read_csv(args.observations), json.loads(args.rule_thresholds.read_text()),
                     [pd.read_csv(p) for p in args.seeds])
    result.to_csv(args.output, index=False)
    print(f'Wrote {len(result)} predictions; changed {(result.pred != result.baseline).sum()} rows')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--observations', type=Path, required=True)
    p.add_argument('--rule-thresholds', type=Path, required=True)
    p.add_argument('--seeds', type=Path, nargs=3, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
