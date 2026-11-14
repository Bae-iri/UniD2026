# B 보존 및 B1의 30초 변화 Query

코드만 작성했다. 학습·추론·단위 테스트는 실행하지 않았다. 문법 검사는 코드 실행 없이 수행할 수 있다.

## 보존하는 B

- 기존 `abc_fusion_experiment.py`, B의 3개 checkpoint와 `submission_B.csv`는 수정하지 않는다.
- 기본 원본 경로: `abc_runs/current_query_v1/seed_{42,43,44}/B/best.pt`.
- 원본 B run의 전처리 통계와 클래스 가중치를 그대로 복원한다. 기존 센서 좌표계를 바꾸지 않고 B checkpoint를 strict하게 검사해 불러온다.
- B1 출력 기본 폴더는 별도 `b1_runs/history30_v1/`. B run 안에 출력 폴더를 두는 것은 거부한다.
- 각 seed의 원본 checkpoint는 새 폴더에도 `B_reference.pt`로 보존하고 SHA256을 기록한다.

## B1 구조

```text
현재 전기·열 입력 → 기존 q_encoder → q_current ─┐
                                              + → q_B1
과거 30초 변화 특징 → MLP(11→32→64) ──────────┘
```

history MLP 마지막 선형층의 weight/bias를 0으로 초기화한다. 초기 `q_B1=q_current`이므로 B 가중치를 로드한 B1은 동일한 현재 입력에서 기존 B와 같은 출력을 낸다. B 가중치 재로딩 시에도 마지막 층을 0으로 재설정한다.

q_B1은 가스·이미지 cross-attention Query와 최종 head의 직접 Query 경로에 모두 전달한다. 가스 전체 요약·이미지 전체 요약 skip, 현재 이미지 CNN, 기존 head를 유지한다. 정적 4개 변수 제외, 가스 그룹화 없음, 이미지 시계열 없음, label smoothing 및 새 손실 없음.

## 인과적인 특징 생성

학습과 추론은 모두 `b1_history.history_features`를 사용한다.

1. 같은 `experiment_id` 안에서 `relative_time_s`로 정렬하되 결과는 원래 target 행 순서로 복원한다. 원본 DataFrame index에 의존하지 않는다.
2. 각 target의 현재 관측은 항상 포함한다. 현재보다 엄격히 과거이며 `[현재−30초, 현재)` 범위인 행만 이력 후보로 쓴다. 같은 timestamp의 다른 행과 미래 행은 사용하지 않는다.
3. 현재 시간에서 역방향으로 최소 5/10/15초 간격을 만족하는 가장 최근 실제 관측을 선택한다. 이어서 그 관측에서 같은 최소 간격만큼 이전의 실제 관측을 선택한다. 보간하거나 없는 시점의 값을 만들지 않는다.
4. meas_v, meas_t1, meas_t2, thermal_t 각각에 대해 선택된 현재/과거의 **실제 finite 관측만**으로 시간에 대한 OLS 기울기(값/초)를 구한다. 2개 이상의 서로 다른 유효 관측 시간이 있어야 한다. 현재 값이 결측이어도 실제 과거 값 2개가 있으면 기울기를 구한다.
5. 유효 기울기를 만들 수 없으면 slope_valid=0과 placeholder 0을 제공한다. 센서의 median 대체값으로 기울기를 만들지 않는다.

총 11차원:

| 특징 | 차원 | 의미 |
|---|---:|---|
| 4개 기울기 | 4 | 유효 실제 관측의 OLS slope |
| 4개 slope_valid | 4 | 기울기의 유효 여부 |
| history_span_s | 1 | 현재부터 가장 오래된 선택 이력 timestamp까지 실제 범위. 항상 30이 아님 |
| previous_gap_s | 1 | 현재와 가장 최근 선택 이력 사이의 실제 시간 차이. 불규칙 간격일 때 nominal step과 다를 수 있음 |
| previous_gap_valid | 1 | 선택 가능한 과거 행이 있는지 |

span/gap은 선택된 관측 행의 시간 범위다. 센서마다 결측이 다르면 기울기에 실제로 참여한 timestamp는 다를 수 있으며 slope_valid로 사용 가능 여부를 제공한다. 유효한 과거가 전혀 없으면 raw span/gap/valid는 0이다. 정규화 후 span은 Train 기준 z-score이므로 이력이 없을 때도 0이 아닐 수 있고, gap_valid로 구별한다.

## Train 이력 augmentation 및 정규화

- 공식 Train의 모든 target 행을 유지한다. 전체 CSV를 5/10/15초 간격으로 subsampling하지 않는다.
- raw Train으로 3개 cadence의 변화 특징을 각각 처음부터 계산한다. 각 epoch에서 target별로 하나의 cadence를 독립적으로 선택한다. minibatch/label/현재 이미지/현재 센서 행은 그대로다.
- 예를 들어 현재 t=30이면 10초 뷰의 규칙적인 이력은 t=20,10,0이고 현재 t=30도 기울기에 포함된다. 5초 뷰의 기울기를 단순히 확대/축소하는 것이 아니다.
- 새 통계는 세 Train 이력 뷰를 합쳐서 계산한다. slope는 slope_valid인 Train 값만, gap은 gap_valid인 Train 값만으로 평균/표준편차를 계산한다. span은 모든 Train target에 대해 계산한다. 유효 flags는 0/1 그대로 둔다.
- 새 통계는 `history_normalization.json`에 저장한다. Validation/Test에서 fit하지 않는다. 무효 slope/gap은 정규화 후 0으로 마스킹한다.
- 검증·추론은 기본 최소 5초 cadence를 쓴다. `--eval-history-step 10` 또는 `15`를 지정할 수도 있으나 실행 설정에 기록되며 학습된 run의 추론에서도 동일하게 복원된다.

## 학습과 저장

- B checkpoint에서 warm start해 B1 전체를 fine-tuning한다. 기존 B checkpoint 파일을 갱신하지 않는다.
- 이번 B1 학습은 공식 Train 전체를 사용한다. 이전 B의 내부 fit/dev 분할과 다르며, 기존 전처리/클래스 가중치는 B를 보존하기 위해 그대로 유지한다.
- 기존 weighted CrossEntropyLoss(label_smoothing=0), batch 64, AdamW lr=3e-4, weight_decay=1e-4, gradient clipping 5를 유지한다.
- **제공 Validation Macro F1 최고점**으로 저장한다. 초기 B와 같은 epoch 0도 저장 후보이므로 모든 학습 epoch가 Validation을 악화시키면 epoch 0으로 복귀한다. 선택 epoch가 명시된다.
- Validation을 checkpoint 선택에 쓰므로 이 점수는 독립 최종 평가가 아니다. 기존 B 대비 차이는 이력 추가뿐 아니라 추가 fine-tuning/Train 행 사용 차이도 포함한다. 이력 자체의 효과를 분리하려면 동일한 추가 학습 예산의 B 대조군이 별도로 필요하다.

## 직접 실행

```bash
cd /home/ubuntu/leewon
# 선택 사항: 데이터 접근/학습이 없는 단위 테스트
/home/ubuntu/venv/bin/python -m unittest -v test_b1

# 실제 학습은 사용자가 실행
/home/ubuntu/venv/bin/python -u b1_experiment.py --device cuda
```

기본 seeds=42/43/44, epochs=15. 경로/설정을 바꾸려면 `--base-run`, `--data-root`, `--output`, `--epochs`, `--seeds`를 지정한다. 동일한 명령 뒤에 `--resume`을 붙이면 완료된 seed를 건너뛰고 미완료 seed는 원본 B부터 재시작한다. 코드·데이터·checkpoint·설정이 바뀌면 새 출력 경로를 써야 한다.

학습 완료 후 재추론만 실행:

```bash
/home/ubuntu/venv/bin/python -u b1_experiment.py --device cuda \
  --output b1_runs/history30_v1 --predict-only
```

다른 입력은 `--input-csv`, `--image-root`, `--sample-submission`, `--prediction-output`을 함께 지정한다. 입력 CSV에는 예측할 행과 그 이전 30초 관측이 있어야 한다. 입력에 없는 과거를 다른 split에서 가져오지 않는다. `--predict-only`는 Train을 읽거나 통계를 재학습하지 않고 저장된 전처리/이력 정규화/모델을 복원한다.

## 생성될 결과

- `submission_B1.csv`: 세 seed 확률 평균, sample_id/label 두 열. 공식 template ID 순서와 label 1~6 정수를 검사. 기존 B 제출 파일은 변경하지 않는다.
- `summary.csv`, `aggregate.csv`, `ensemble_scores.csv`, `results_combined.csv`, `report.md`.
- seed별 `B_reference.pt`, 초기 B Validation 예측, `best.pt`, `history.csv`, 클래스별 F1·혼동행렬·Validation/Test 예측.
- `preprocessing.json`, `history_normalization.json`, `config.json`, 원본 코드 snapshot과 체크포인트 hash.

별도 재추론 파일은 기본 `inference/submission_B1.csv`에 저장한다. 외부 업로드나 제출은 수행하지 않는다.
