`submission_J096_S3_final.csv`의 예측을 보존한 독립 재현 코드와 일반화 감사다. 결과와 한계는 [REPORT.md](REPORT.md)에 기록했다. public 0.9719에 맞춘 추가 튜닝은 하지 않았다.

`pipeline.py`는 B 3-seed 평균 → T1 45℃ 누적 규칙 → 5→6 보정 → 6→5 보정 → 4→3 하위 보정 → T1+T2 합계 S3 규칙을 실행한다. 제출본에서 4→5와 3→4는 꺼져 있다. 현재·과거 관측만 사용하며 실험별 시간순으로 계산한 뒤 입력 순서대로 반환한다. 전압 초기 기준은 관측 도착 시점까지 갱신하고 과거로 소급하지 않는다.

저장소 루트에서 실행한다. Python 3.10과 `requirements.txt`의 패키지가 필요하다. 기존 B 구조는 저장소의 `abc_reference/`를 사용한다. 학습 데이터·이미지·체크포인트·joblib·행별 대형 평가 파일은 Git에 포함하지 않는다. 제출 CSV 한 개와 작은 검증 결과는 포함한다. 재현에는 아래 경로의 기존 artifact가 필요하며, 해시는 `reproduction.json`, `raw_audit.json`에서 확인할 수 있다.

이 작업 환경의 경로 예시:

```bash
PYTHON=/home/ubuntu/venv/bin/python
DATA=/home/ubuntu/data/battery_data_light
BASE=/home/ubuntu/leewon/abc_runs/current_query_v1
UPPER=/home/ubuntu/gahyun/experiments/rule1_precision65/artifacts/pipeline.joblib
LOWER=/home/ubuntu/JEONGYUN/experiments/lower_heads/submission_v1/lower_heads.joblib
```

저장된 B 확률로 제출 파일을 재생성하고 prefix·특징·행 순서·라벨 무관성을 검사한다. 출력 파일이 존재하면 덮어쓰지 않는다.

```bash
"$PYTHON" experiments/b15_lower/reproduce.py \
  --input "$DATA/test/test.csv" --base-predictions "$BASE" \
  --upper-artifact "$UPPER" --lower-artifact "$LOWER" \
  --sample-submission "$DATA/sample_submission.csv" \
  --reference experiments/b15_lower/submission_J096_S3_final.csv \
  --output /tmp/j096_reproduced.csv --report /tmp/j096_reproduced.json --audit
```

새 이미지·센서에서 직접 추론하려면 `infer.py`를 사용한다. `--device cpu`가 기본이며 GPU도 지원한다. 출력 ID 순서는 `--sample-submission`으로 지정할 수 있다.

```bash
"$PYTHON" experiments/b15_lower/infer.py \
  --input "$DATA/test/test.csv" --image-root "$DATA/test" --base "$BASE" \
  --upper-artifact "$UPPER" --lower-artifact "$LOWER" \
  --sample-submission "$DATA/sample_submission.csv" --output /tmp/j096_raw.csv

"$PYTHON" experiments/b15_lower/raw_audit.py \
  --data "$DATA" --base "$BASE" --upper-artifact "$UPPER" --lower-artifact "$LOWER" \
  --reference experiments/b15_lower/submission_J096_S3_final.csv --report /tmp/j096_raw_audit.json
```

독립 바깥 분할 검증은 기존에 학습된 180개 B 모델의 계보와 정규화를 다시 감사하고, 그 분할 안의 OOF 확률로 고정된 보정 구조를 새로 적합한다. 이미 살펴본 데이터이므로 새로운 맹검 테스트라는 의미는 없다. 공식 validation은 제출 가중치로 별도 평가한다.

```bash
"$PYTHON" experiments/b15_lower/validate.py --data "$DATA" \
  --nested /home/ubuntu/gahyun/experiments/nested_stability/fresh_backbones \
  --repeated /home/ubuntu/gahyun/experiments/frozen_five_holdout/run_v1 \
  --base-predictions "$BASE" --upper-artifact "$UPPER" --lower-artifact "$LOWER" \
  --output /tmp/j096_audit
```

기존 검증용 기본 모델이 없는 환경에서는 `train_backbones.py`로 두 검증 묶음을 재생성할 수 있다. 각 출력 경로는 새 디렉터리여야 한다. 전체 15 epoch 재학습은 이번 감사에서 다시 실행하지 않았고, 기존 모델을 해시 검증 후 사용했다. 새 학습 코드의 분할 인덱스가 기존 모든 분할과 일치함을 확인했고, 원본 이미지 70행·6실험에서 1 epoch의 학습·체크포인트 선택·확률 내보내기를 시험했다. 하드웨어·라이브러리 차이로 새 학습의 숫자는 달라질 수 있다.

```bash
"$PYTHON" experiments/b15_lower/train_backbones.py --data "$DATA" --suite train78 --output /tmp/j096_train78
"$PYTHON" experiments/b15_lower/train_backbones.py --data "$DATA" --suite repeat77 --output /tmp/j096_repeat77
# validate.py의 --nested /tmp/j096_train78 --repeated /tmp/j096_repeat77 로 사용

"$PYTHON" experiments/b15_lower/test_pipeline.py
```

`core/upper.py`, `core/lower.py`, `core/precision.py`는 제출 때 사용한 작은 HGB 보정 구현의 소스 스냅샷이다. import만 로컬 상대 경로로 옮겼다. `core/causal.py`는 기존 `compare_abc_phase_history.py`의 현재·과거 특징 함수만 추출했다. 학습 진입점을 import하지 않으며 다른 사용자의 작업 디렉터리에 대한 코드 의존성을 제거했다. 모델 파일 경로는 실행 인자로 받는다.
