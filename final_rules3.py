"""원본 ABC B 예측(public 0.9121) + 센서 기반 단계 규칙 3개 → public 0.93 제출 파일.

규칙(모두 같은 experiment_id의 현재·과거 관측만 사용):
  1) 1↔2 경계 정의: meas_t1 누적 최고 ≥ 45°C 이면 2단계 이상, 미만이면 1단계
     (train+validation 97개 실험의 1→2 전환 시점 meas_t1 = 45.0~45.4°C)
  2) 전압 필요조건: meas_v가 관측된 초기 전압 기준의 80% 아래로 내려간 적이 없으면 4단계 이상 불가 → 3
     초기 5행 동안은 현재까지 관측한 최대값만 사용하고, 이후 기준을 고정한다.
     (3→4 전환이 전압 하락 시점과 78% 실험에서 ±5초 일치)
  3) t1 필요조건: meas_t1 누적 최고가 배터리 용량(bat_cc)별 하한(3단계 이상 행의 1% 분위) 미만이면 3단계 불가 → 2

검증(원본 B 대비, 기준값은 fit 실험으로만 계산): dev F1 0.7815 → 0.8102, validation(E16 제외) 0.8142 → 0.8685.
제출 파일의 하한값은 train+validation(E16 제외)으로 계산한다. 정답·행별 예측은 출력하지 않는다.

  python final_rules3.py --base /home/ubuntu/leewon/abc_runs/current_query_v1/submission_B.csv \
      --data-dir /home/ubuntu/data/battery_data_light --output submission_rules3.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

E16 = 'E_e5bcddc7d77f459ea541cf658ab4346e'  # validation의 3→5 점프 실험(하한 계산에서만 제외)
T1_STAGE2 = 45.0
V_DROP = 0.8
Q_T1 = 0.01


def prepare(frame):
    d = frame.sort_values(['experiment_id', 'relative_time_s'], kind='mergesort').reset_index(drop=True)
    g = d.groupby('experiment_id')
    d['t1max'] = g.meas_t1.cummax()
    # No read-ahead: update the reference as each of the first five observations
    # arrives, then freeze it. Never broadcast a future voltage to an earlier row.
    v0 = pd.Series(np.nan, index=d.index, dtype=float)
    for indices in g.indices.values():
        reference = np.nan
        for position, row in enumerate(indices):
            voltage = d.at[row, 'meas_v']
            if position < 5 and np.isfinite(voltage):
                reference = voltage if not np.isfinite(reference) else max(reference, voltage)
            v0.at[row] = reference
    d['voltage_reference'] = v0
    d['vdrop'] = (d.meas_v < V_DROP * v0).groupby(d.experiment_id).cummax()
    return d


def t1_floor(labeled):
    d = prepare(labeled)
    x = d[d.label >= 3]
    return x.groupby('bat_cc').t1max.quantile(Q_T1).to_dict(), float(x.t1max.quantile(Q_T1))


def apply_rules(d, pred, floor):
    p = pred.copy()
    hot = d.t1max.to_numpy() >= T1_STAGE2
    p[(p == 1) & hot] = 2
    p[(p >= 2) & ~hot] = 1
    p[(p >= 4) & ~d.vdrop.to_numpy()] = 3
    th = d.bat_cc.map(floor[0]).fillna(floor[1]).to_numpy()
    p[(p == 3) & (d.t1max.to_numpy() < th)] = 2
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', type=Path, default=Path('/home/ubuntu/leewon/abc_runs/current_query_v1/submission_B.csv'))
    ap.add_argument('--data-dir', type=Path, default=Path('/home/ubuntu/data/battery_data_light'))
    ap.add_argument('--output', type=Path, default=Path('submission_rules3.csv'))
    a = ap.parse_args()
    root = a.data_dir
    labeled = pd.concat([pd.read_csv(root / 'train/train.csv'), pd.read_csv(root / 'validation/validation.csv')])
    floor = t1_floor(labeled[labeled.experiment_id != E16])
    test = prepare(pd.read_csv(a.base).rename(columns={'label': 'B'}).merge(pd.read_csv(root / 'test/test.csv'), on='sample_id'))
    pred = apply_rules(test, test.B.to_numpy(), floor)
    sample = pd.read_csv(root / 'sample_submission.csv')
    sub = sample[['sample_id']].merge(pd.DataFrame({'sample_id': test.sample_id, 'label': pred}), on='sample_id', how='left')
    if len(sub) != len(sample) or sub.label.isna().any() or not sub.label.between(1, 6).all():
        raise SystemExit('제출 형식 오류')
    sub['label'] = sub.label.astype(int)
    sub.to_csv(a.output, index=False)
    print(f'wrote {a.output}: {len(sub)} rows, changed vs base {(sub.label.values != pd.read_csv(a.base).set_index("sample_id").label.loc[sub.sample_id].values).sum()}')


if __name__ == '__main__':
    main()
