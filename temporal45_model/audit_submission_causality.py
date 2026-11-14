"""Reproduce the existing submission using current/past observations only.

There is no legacy read-ahead implementation in this audit. It never trains or
selects a model and never writes a submission. Test has no labels.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import final_rules3 as original
from temporal45_model.causal_rules import prepare


def main():
    data = Path('/home/ubuntu/data/battery_data_light')
    base = Path('/home/ubuntu/leewon/abc_runs/current_query_v1/submission_B.csv')
    path = ROOT/'final_submit/submission_rules3.csv'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sub = pd.read_csv(path)
    frame = pd.read_csv(base).rename(columns={'label': 'B'}).merge(
        pd.read_csv(data/'test/test.csv'), on='sample_id', validate='1:1')
    assert 'label' not in frame
    causal = prepare(frame)
    labeled = pd.concat([pd.read_csv(data/'train/train.csv'), pd.read_csv(data/'validation/validation.csv')])
    floor = original.t1_floor(labeled[labeled.experiment_id != original.E16])
    after = original.apply_rules(causal, causal.B.to_numpy(int), floor)
    after = pd.Series(after, index=causal.sample_id).loc[sub.sample_id].to_numpy(int)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(file='final_submit/submission_rules3.csv', rows=len(sub),
                  experiments=int(causal.experiment_id.nunique()), sha256=digest,
                  implementation='current/past only; initial reference updated on arrival and then frozen',
                  prediction_differences_after_removing_future_reference=int((sub.label.to_numpy() != after).sum()),
                  causal_predictions_match_submitted_file=bool(np.array_equal(after, sub.label)),
                  test_labels_present=False, file_unchanged=True)
    print(json.dumps(report, indent=2))
    (ROOT/'temporal45_model/submission_rules3_causality_audit.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
