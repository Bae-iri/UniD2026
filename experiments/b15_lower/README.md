정윤 코드의 **4→3 보정 + 6단계 보정(S3)** 통합 구현이다. 기존 제출 예측을 그대로 재현하며, 검증은 **E16을 포함한 전체 자료**를 사용한다. 리더보드 점수를 입력하거나 모델·임계값 선택에 사용하는 경로는 없다. 결과와 한계는 [REPORT.md](REPORT.md)에 기록한다.

후속 2→3·4→5 보정 검토는 채택 기준을 통과하지 못해 현재 제출을 유지했다. [결과와 다음 조치](boundary_review_v1/REPORT.md), [재현 코드](boundary_review.py).

정윤 원본과 현재 코드의 같은 조건 직접 비교는 [비교 보고서](original_comparison_v1/REPORT.md)에 있다. 두 제출 재현·E16 포함 검증·테스트 관측 간격 및 조건 차이를 확인한 근거는 현재 S3 코드 쪽을 지지한다.

가장 직접적인 실행 경로는 `build_combined.py`다. 고정된 두 보정 artifact를 하나로 묶고, 공식 검증 19실험 전부(E16 1,543행 포함)의 전체·구성요소별 지표와 prefix 불변성을 검사한 뒤 제출 CSV를 생성한다. `combined_v2/manifest.json`에 검증 근거를 남긴다. 6단계 오탐(FP)과 놓친 정답(FN)을 모두 기록한다. 기존 제출 CSV와 다른 예측이 하나라도 나오거나 4→3 보정이 덮이면 실패한다.

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

통합 artifact와 검증·제출 결과를 한 번에 만든다. 출력은 새 디렉터리를 지정한다.

```bash
"$PYTHON" experiments/b15_lower/build_combined.py \
  --data "$DATA" --base-predictions "$BASE" \
  --upper-artifact "$UPPER" --lower-artifact "$LOWER" \
  --reference experiments/b15_lower/submission_J096_S3_final.csv \
  --output /tmp/jeongyun43_stage6

"$PYTHON" experiments/b15_lower/infer.py \
  --input "$DATA/test/test.csv" --image-root "$DATA/test" --base "$BASE" \
  --combined-artifact /tmp/jeongyun43_stage6/combined.joblib \
  --sample-submission "$DATA/sample_submission.csv" --output /tmp/jeongyun43_stage6_raw.csv
```

`reference`는 요청한 예측과의 일치 여부만 검사한다. 정답·리더보드 점수로 파라미터를 조정하는 용도가 아니다. `combined.joblib`은 Git에서 제외되며, 이 환경에는 `experiments/b15_lower/combined_v2/combined.joblib`이 생성되어 있다.

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

독립 바깥 분할 검증의 주 평가는 **97실험 전체의 새 5-fold**다. E16이 평가 fold에 포함되고 해당 fold의 기본 모델·전처리·체크포인트 선택·보정 학습에서 모두 제외된다. 내부 OOF 교사를 포함한 60개 기본 모델을 새로 학습했다. 기존 78실험 5-fold와 77실험 반복 검증은 보조 평가로 함께 재실행한다. 이전 77실험 자료에는 E16이 없으므로, 이를 전체 자료의 검증으로 취급하지 않는다. 이미 살펴본 데이터의 강건성 검증이며 새로운 맹검은 아니다.

```bash
"$PYTHON" experiments/b15_lower/validate.py --data "$DATA" \
  --nested /home/ubuntu/gahyun/experiments/nested_stability/fresh_backbones \
  --repeated /home/ubuntu/gahyun/experiments/frozen_five_holdout/run_v1 \
  --all97 /home/ubuntu/gahyun/experiments/b15_lower/all97_backbones \
  --base-predictions "$BASE" --upper-artifact "$UPPER" --lower-artifact "$LOWER" \
  --output /tmp/j096_audit
```

`train_backbones.py`로 검증용 기본 모델을 학습할 수 있다. 각 준비 출력 경로는 새 디렉터리여야 한다. 97실험 검증은 모델 60개를 각각 15 epoch 학습하며, 이전 학습에 쓰이지 않았던 E16도 전체 분할에 포함한다. 과거 20실험 확인 자료도 이번에는 전체 CV에 속한다. 따라서 이를 다시 맹검 확인 자료라고 부르지 않는다. 하드웨어·라이브러리 차이로 새 학습의 숫자는 달라질 수 있다.

```bash
"$PYTHON" experiments/b15_lower/train_backbones.py --data "$DATA" --suite all97 --output /tmp/j096_all97
"$PYTHON" experiments/b15_lower/train_backbones.py --data "$DATA" --suite train78 --output /tmp/j096_train78
"$PYTHON" experiments/b15_lower/train_backbones.py --data "$DATA" --suite repeat77 --output /tmp/j096_repeat77
# validate.py의 --all97 /tmp/j096_all97 --nested /tmp/j096_train78 --repeated /tmp/j096_repeat77 로 사용

"$PYTHON" experiments/b15_lower/test_pipeline.py
```

`core/upper.py`, `core/lower.py`, `core/precision.py`는 제출 때 사용한 작은 HGB 보정 구현의 소스 스냅샷이다. import만 로컬 상대 경로로 옮겼다. `core/causal.py`는 기존 `compare_abc_phase_history.py`의 현재·과거 특징 함수만 추출했다. 학습 진입점을 import하지 않으며 다른 사용자의 작업 디렉터리에 대한 코드 의존성을 제거했다. 모델 파일 경로는 실행 인자로 받는다.
