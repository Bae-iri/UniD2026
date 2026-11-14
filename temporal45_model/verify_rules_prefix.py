"""Verify production rule state is unchanged when observations end early."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import final_rules3 as rules


def main():
    data = Path('/home/ubuntu/data/battery_data_light')
    results = []
    for split in ['train', 'validation', 'test']:
        frame = pd.read_csv(data/split/f'{split}.csv')
        count = 0
        for _, group in frame.groupby('experiment_id'):
            group = group.sort_values('relative_time_s', kind='stable').reset_index(drop=True)
            full = rules.prepare(group)
            for end in sorted(set([1, 2, 3, 4, 5, max(1, len(group)//2), len(group)])):
                if end > len(group):
                    continue
                prefix = rules.prepare(group.iloc[:end])
                for column in ['t1max', 'voltage_reference', 'vdrop']:
                    np.testing.assert_allclose(prefix[column], full[column].iloc[:end], equal_nan=True)
                count += 1
        results.append(dict(split=split, experiments=int(frame.experiment_id.nunique()), prefix_checks=count))
    result = dict(passed=True, results=results,
                  total_prefix_checks=sum(r['prefix_checks'] for r in results),
                  production_source_sha256=hashlib.sha256((ROOT/'final_rules3.py').read_bytes()).hexdigest(),
                  test_labels_used=False, model_training_or_selection=False)
    (ROOT/'temporal45_model/rules_prefix_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
