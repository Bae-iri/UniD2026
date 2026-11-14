# 원본 B 기반 독립 실험 1 / 실험 2

사용자 지정 원본: 리원 `feat/leewon`의 `FusionModel('B')`. 원본 코드와 common attention은 `common/`에 수정 없이 복사했습니다. 초기 가중치는 `/home/ubuntu/leewon/abc_runs/current_query_v1/seed_{42,43,44}/B/best.pt`의 실제 B 체크포인트이며 기존 gated_hybrid나 다른 실험의 checkpoint는 사용하지 않습니다.

## 독립성

- `experiment1_voltage/model.py`: 원본 B + 전압 급락 전용 최종-logit 분기. 실험2의 gas/temperature 조절 없음.
- `experiment2_gas_temperature/model.py`: 원본 B + 가스0 조건부 gas value 조절과 현재 온도 전용 최종-logit 분기. 전압 사건/시간 기억을 계산·import·입력하지 않음.
- `B_control`: 원본 B를 같은 데이터·추가 학습 예산으로 더 학습한 대조군.

각 실험은 **원본 B에서 독립적으로 시작**합니다. 추가 학습한 B_control의 가중치도 다른 실험으로 넘기지 않습니다. 원본 checkpoint를 읽어 run 폴더에 복사하고 hash를 기록하며 원본 파일을 수정하지 않습니다. 원본 validation 확률을 현재 코드로 재현해 일치함을 검사합니다.

## 실험1: 전압 급락 전용 분기 ×2

현재 sensor/gas/image encoder와 Cross-Attention, 전체 요약 skip 및 원본 6분류 head는 유지합니다. 별도 전압 사건 7개 열을 MLP(7→64→6)에 입력하고 **logit 잔차에 2.0을 곱해 원본 B 출력에 더합니다.** 분기 마지막 층은 0으로 초기화하여 출발 출력은 원본 B와 같습니다. 이 실험은 전압 사건 정보를 최종 분류에 추가하는 효과이며 Query나 gas weighting은 변경하지 않습니다.

이력은 동일 experiment_id의 현재·과거 실제 관측만 사용합니다. 연속 두 유효 전압 관측(간격 ≤30초)의 하락량과 실제 초당 하락 속도를 계산합니다. 의미 있는 급락은 **fit 양수 전압 중앙값의 10% 이상 하락**이면서 **fit 양수 하락 속도의 90분위수 이상**인 경우입니다. detector는 label/validation을 사용하지 않습니다.

- max_voltage_drop: 현재까지 최대 연속 관측 하락량(V).
- max_drop_rate: 현재까지 최대 하락 속도(V/s).
- time_since_drop: 첫 의미 있는 급락 이후 경과초. 후속 급락에도 reset하지 않음.
- voltage_after_drop: 첫 급락 직후 전압.
- drop_occurred / event_not_observed / has_observed_voltage_pair: 사건 및 관측 가능 여부.

연속 관측 중 NaN/Inf/음수 전압이 있으면 연결하지 않습니다. 5/10/15초 간격은 실제 Δt로 반영합니다. 경과시간은 log1p 후 fit 통계로 표준화합니다. 아직 사건이 없으면 placeholder0과 별도 flag를 사용합니다. raw event 열은 split별 CSV에 저장합니다. 급락이 있었다는 이유로 5단계로 강제하지 않습니다.

**목표 지표**: 실제 3단계와 실제 5단계의 오류율(1−recall), 각 F1, 3→5/5→3 오답 수 및 전체 Macro F1.

## 실험2: 가스0일 때 gas value ×0.5, 온도 분기 ×2

원본 B의 입력은 현재 전기·열 센서와 gas/image 토큰 그대로입니다. raw 화학 가스 10종이 **모두 finite이고 정확히 0**일 때만 조절합니다. 압력 meas_p는 이 판정에서 제외합니다. NaN/Inf나 음수는 all-zero로 처리하지 않습니다. 검출한계가 없으므로 이는 물리적 미검출 확정이 아닌 기록값 proxy입니다.

- chemical gas token의 전체 요약 및 attention **value projection의 기여를 0.5배**로 조절합니다.
- 압력 token, gas attention key/query 및 valid mask는 유지합니다. 0인 토큰도 유효 관측으로 남습니다.
- 현재 meas_t1/meas_t2/thermal_t의 표준화 값·결측 flag만 별도 MLP(6→64→6)에 입력합니다.
- 온도 logit 잔차 ×2를 all-zero 표본에만 더합니다. gas가 0이 아닌 행은 원본 B와 동일 경로이며 초기 출력의 일치를 검사했습니다.
- 이미지 readout 및 전체 이미지 요약은 원본 그대로입니다. 전압 급락/Temporal Query/Event Memory를 추가하지 않습니다.

화학 gas를 줄여도 압력과 원래 zero 정보가 완전히 제거되지 않습니다. 온도 분기의 마지막 층은 0 초기화합니다. 계수는 실행 전 .5와 2로 고정하고 validation으로 조정하지 않습니다.

**목표 지표**: 실제 4단계·5단계의 오류율(1−recall)/F1, 4→5/5→4 건수, all-zero 4/5의 오류율과 분모. validation의 가스0 실제4는 12개, 실제5는 506개이므로 분모가 작은 오탐률에 주의합니다.

2배 계수는 분기 출력의 명시적 배율이지 학습된 feature importance가 반드시 2배라는 뜻은 아닙니다. 실험1은 사건 특징 추가+강조, 실험2는 gas downweight+온도 분기 강조의 결합 효과입니다. 각각을 세부 요인 하나의 효과로 단정하지 않습니다.

## 공통 학습

원본 B와 같은 experiment 단위 fit/dev 분할(62/16 실험)을 유지하고 기존 checkpoint의 공통 가중치에서 시작합니다. validation 19개/test11개는 추가 학습에 넣지 않습니다. detector/normalizer/class weight는 fit에서만 계산합니다. static4개 변수는 계속 제외합니다.

seeds42/43/44, **추가 15 epoch**, batch64, image128, weighted CE, AdamW lr3e-4/weight_decay1e-4, clip5. 모델 생성/초기화 후 dropout seed를 seed+999로 reset해 공통 head의 난수 흐름을 맞춥니다. 별도 분기에는 dropout을 넣지 않아 공통 batch 순서와 head dropout 흐름을 바꾸지 않습니다. 원본 상태 epoch0도 저장 후보에 포함하고 dev Macro F1만으로 best checkpoint를 선택합니다.

validation은 이전에 이미 분석한 탐색 데이터입니다. 새 blind test가 아니며 실제 제출 점수는 CSV 평가 전에는 알 수 없습니다. 원본 B 3-seed 앙상블과 이 추가학습 실험의 점수를 다른 모델의 84/91점과 직접 비교하지 않습니다.

## 실행

```bash
cd /home/ubuntu/gahyun/experiments/b_independent_studies
/home/ubuntu/venv/bin/python -m unittest -v test_independence
# B 대조군 + 두 독립 실험, 모두 원본 B에서 시작
/home/ubuntu/venv/bin/python -u run.py --epochs 15 --seeds 42 43 44
```

단일 실험만 독립 실행할 수도 있습니다.

```bash
/home/ubuntu/venv/bin/python experiment1_voltage/train.py --output experiment1_voltage/runs/new_v1
/home/ubuntu/venv/bin/python experiment2_gas_temperature/train.py --output experiment2_gas_temperature/runs/new_v1
```

기존 출력 경로를 덮어쓰지 않습니다. 다시 실행할 때 새 --output을 사용하세요. 이미지 캐시만 기존 battery_query_rates에서 재사용하며 가중치를 거기서 읽지 않습니다.

## 결과

기본 폴더: `runs/independent_seed42_43_44_v1/`.

- `experiment1_voltage/submission.csv`: 실험1의 seed42/43/44 확률 평균 제출.
- `experiment2_gas_temperature/submission.csv`: 실험2의 독립 seed 앙상블 제출.
- `B_control/submission.csv`: 동일 추가 학습량 B 대조군.
- 루트의 `submission_experiment1_voltage.csv`, `submission_experiment2_gas_temperature.csv`는 위 파일의 복사본입니다.
- `ensemble_scores.csv`, `ensemble_percent.csv`, `ensemble_differences.csv`: 전체/클래스 F1과 목표 오류율·혼동 건수. percent 파일의 F1/오류율은 0~100, 차이는 %p입니다. 오류율 차이가 음수면 개선입니다.
- `summary.csv`, `seed_aggregate.csv`, `paired_seed_differences.csv`: seed별 결과·평균/표준편차와 paired 차이.
- 실험별 ensemble 및 seed별 예측·혼동행렬·class report·history/best.pt·전처리.
- 원본 checkpoint 복사본 및 hash, 원본 B 확률 재현 검사, CSV/코드 해시·source snapshot·split manifest.

각 CSV는 공식 sample_submission의 ID를 기준으로 sample_id,label 두 열과 행 순서를 맞추며 3,451행·중복/결측 없음·정수1~6을 검사합니다. 두 실험 CSV를 서로 결합하지 않습니다. 외부 업로드/제출은 수행하지 않습니다.

## 실행 완료

세 seed 모두 각 모델에 추가15 epoch를 실행하고 실험1/2 제출 CSV를 각각 검증했습니다. 코드 폴더에도 제출 CSV를 복사해 두었습니다.

원본 B 3-seed 앙상블의 실제3/4/5 오답률은 17.91%/13.64%/62.61%입니다. 실험1은 3단계18.72%, 5단계62.61%로 목표 개선이 없었습니다. 실험2는 4단계15.91%, 5단계63.40%로 두 목표 오류율이 모두 증가했습니다. 동일 추가 학습량의 B 대조군도 같이 저장했으며 비교표는 `comparison_including_original_B.csv`, 오답 수와 분모는 `target_error_table.csv`에 있습니다.

이번 숫자는 로컬 validation의 3-seed **확률 앙상블** 결과입니다. seed 평균 점수와 혼동하지 마세요. 실험 결과가 목표를 달성하지 않았더라도 제출 후보를 바꾸거나 서로 섞지 않고 두 독립 CSV를 그대로 제공합니다. 실제 제출 점수는 아직 알 수 없습니다.

## 저장소에 포함된 결과

체크포인트와 `runs/` 출력은 저장소에 넣지 않았습니다. 주요 결과표는 `published_results/independent_seed42_43_44_v1/`에 있습니다.

## 리더보드 결과 및 회귀 분석

- 리더보드: 원본 B 91점, 실험1(전압) 88점 → 3점 하락.
- 실험1 체크포인트에서 전압 분기만 끄고 비교한 결과, 원본 B 대비 test 예측 변화 93개 중 89개가 분기를 끈 모델(공유 가중치 변화만 남은 모델)과 같았습니다. 하락은 전압 분기보다 추가 학습으로 B의 공유 가중치가 바뀐 영향이 큰 것으로 보입니다.
- seed 43은 dev 기준 best epoch가 0이라 원본 B와 동일합니다.
