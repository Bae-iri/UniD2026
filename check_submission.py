"""가이드의 5가지 검사를 수행하는 자체 점검. 운영진 배포 스크립트 원본은 아니다.

python check_submission.py --data_dir data/battery_data_light --final submission_stage6.csv
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def read_predictions(path):
    frame = pd.read_csv(path)
    if list(frame.columns) != ['sample_id', 'label'] or not frame.sample_id.is_unique:
        raise ValueError(f'Invalid prediction IDs/columns: {path}')
    if not frame.label.isin(range(1, 7)).all():
        raise ValueError(f'Invalid prediction labels: {path}')
    return frame.set_index('sample_id').label


def run(input_path, image_root, tmp, name):
    output = tmp/f'{name}_out.csv'
    result = subprocess.run(
        [sys.executable, str(HERE/'test.py'), '--input', str(input_path),
         '--image_root', str(image_root), '--output', str(output)],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f'test.py failed for {name}:\n{result.stdout}\n{result.stderr}')
    expected = pd.read_csv(input_path).sample_id
    predictions = read_predictions(output)
    if predictions.index.tolist() != expected.tolist():
        raise ValueError(f'Output must preserve the supplied CSV row order: {name}')
    return predictions, output


def same(left, right):
    return set(left.index) == set(right.index) and bool((left.loc[right.index] == right).all())


def main(args):
    input_path = args.data_dir/'test/test.csv'
    image_root = args.data_dir/'test'
    test = pd.read_csv(input_path)
    reference = read_predictions(args.final)
    report = {'checker': 'self-authored implementation of the five checks in the participant guide',
              'final_file': args.final.name,
              'final_sha256': hashlib.sha256(args.final.read_bytes()).hexdigest(),
              'rows': len(test), 'experiments': int(test.experiment_id.nunique()), 'checks': {}}

    def record(name, passed):
        report['checks'][name] = bool(passed)
        print(('✓ ' if passed else '✗ ')+name, flush=True)

    with tempfile.TemporaryDirectory(prefix='unid_submission_') as directory:
        tmp = Path(directory)

        def run_frame(frame, name):
            path = tmp/f'{name}_in.csv'
            frame.to_csv(path, index=False)
            return run(path, image_root, tmp, name)[0]

        full, full_path = run(input_path, image_root, tmp, 'full')
        record('1 최종 선택 파일 재현', same(reference, full))
        record('1a 최종 파일 바이트 일치', args.final.read_bytes() == full_path.read_bytes())
        again, _ = run(input_path, image_root, tmp, 'again')
        record('2 같은 입력 다시 실행', same(full, again))
        shuffled = run_frame(test.sample(frac=1, random_state=0), 'shuffled')
        record('3 행 순서 섞기', same(full, shuffled))

        pieces = []
        for i, (_, frame) in enumerate(test.groupby('experiment_id', sort=False)):
            pieces.append(run_frame(frame, f'experiment_{i}'))
        record('4 실험 나눠 넣기', same(full, pd.concat(pieces)))

        ordered = [g.sort_values('relative_time_s', kind='stable')
                   for _, g in test.groupby('experiment_id', sort=False)]
        prefix60 = pd.concat([g.iloc[:max(1, int(np.ceil(.6*len(g))))] for g in ordered])
        truncated = run_frame(prefix60, 'prefix60')
        record('5 미래 행 빼고 넣기(앞 60%)', same(full.loc[truncated.index], truncated))

        for count in [1, 2, 5]:
            early = run_frame(pd.concat([g.iloc[:count] for g in ordered]), f'first{count}')
            record(f'추가 처음 {count}개 관측 prefix', same(full.loc[early.index], early))

        changed = test.copy()
        changed['label'] = 999
        record('추가 정답 열 무관성', same(full, run_frame(changed, 'labels')))

    report['passed'] = all(report['checks'].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Report: {args.report}', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument('--final', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=HERE/'verification/self_check.json')
    sys.exit(main(parser.parse_args()))
