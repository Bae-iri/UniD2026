"""저장된 가중치로 예측만 한다(학습·전처리 재적합 없음).

  python test.py --input <CSV> --image_root <이미지 폴더> --output <저장 경로>

입력 CSV에 있는 행만 본다. 각 행의 예측은 같은 실험에서 그 행 시각까지 관측한 행만 사용하므로,
행 순서를 섞거나 실험을 나눠 넣거나 뒤쪽 행을 빼도 남은 행의 예측은 같다. CPU에서 실행한다.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import model as M

WEIGHTS = Path(__file__).resolve().parent/'weights'


def main(args):
    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    frame = pd.read_csv(args.input)
    if len(frame) == 0:
        raise ValueError('empty input')
    if not frame.sample_id.is_unique:
        raise ValueError('duplicate sample_id')
    bundle = joblib.load(WEIGHTS/'pipeline.joblib')          # 5→6 헤드 + 6→5 헤드 (train.py가 저장)
    lower = joblib.load(WEIGHTS/'lower_heads.joblib')        # 4→3 헤드 (train.py가 저장)
    probs = M.BModel(WEIGHTS).probabilities(frame, args.image_root)
    labels = M.predict_labels(frame, probs, bundle, lower)
    if not np.isin(labels, range(1, 7)).all():
        raise ValueError('invalid label')
    out = pd.DataFrame({'sample_id': frame.sample_id, 'label': labels.astype(int)})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'wrote {len(out)} rows -> {args.output}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--image_root', required=True)
    ap.add_argument('--output', required=True)
    with threadpool_limits(limits=4):
        main(ap.parse_args())
