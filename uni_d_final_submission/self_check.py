"""운영진 점검 5가지를 test.py로 직접 재현하는 자가 점검(공식 check_submission.py와 별개).

  python self_check.py --data_dir <battery_data_light> --final <최종 CSV>

1 최종 파일 재현  2 같은 입력 재실행  3 행 순서 섞기  4 실험별로 나눠 넣기  5 실험마다 앞 60%만 넣기
"""
import argparse, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent


def run(frame, image_root, tmp, name):
    inp, out = tmp/f'{name}_in.csv', tmp/f'{name}_out.csv'
    frame.to_csv(inp, index=False)
    subprocess.run([sys.executable, str(HERE/'test.py'), '--input', str(inp), '--image_root', str(image_root),
                    '--output', str(out)], check=True, capture_output=True)
    return pd.read_csv(out).set_index('sample_id').label


def main(a):
    test = pd.read_csv(a.data_dir/'test/test.csv'); root = a.data_dir/'test'
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d); res = {}
        full = run(test, root, tmp, 'full')
        final = pd.read_csv(a.final).set_index('sample_id').label
        res['1 최종 파일 재현'] = set(final.index) == set(full.index) and bool((final.loc[full.index] == full).all())
        res['2 같은 입력 재실행'] = bool((run(test, root, tmp, 'again').loc[full.index] == full).all())
        perm = test.sample(frac=1, random_state=0)
        res['3 행 순서 섞기'] = bool((run(perm, root, tmp, 'shuffle').loc[full.index] == full).all())
        parts = [run(g, root, tmp, f'exp{i}') for i, (_, g) in enumerate(test.groupby('experiment_id'))]
        res['4 실험 나눠 넣기'] = bool((pd.concat(parts).loc[full.index] == full).all())
        t = test.relative_time_s.astype(float)
        cut = test[t <= test.assign(t=t).groupby('experiment_id').t.transform(lambda s: s.quantile(.6))]
        part = run(cut, root, tmp, 'prefix60')
        res['5 미래 행 빼고 넣기(앞 60%)'] = bool((part == full.loc[part.index]).all())
    for k, v in res.items():
        print(('✓ ' if v else '✗ ')+k)
    sys.exit(0 if all(res.values()) else 1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--data_dir', type=Path, required=True); ap.add_argument('--final', type=Path, required=True)
    main(ap.parse_args())
