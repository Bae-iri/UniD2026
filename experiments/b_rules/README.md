# 원본 B + 1→2단계 온도 규칙 (리더보드 92)

원본 B(리원 `FusionModel('B')` 3-seed 앙상블)의 **모델·가중치·예측 확률은 그대로** 두고, 최종 예측에 규칙 하나를 후처리로 적용했습니다. 학습한 모델은 없습니다.

## 리더보드

| 제출 | 원본 B 대비 test 변경 | 점수 |
|---|---|---|
| 원본 B | 0 | 91 |
| **원본 B + T1 45℃ 규칙 (`submission_originalB_rule1_T1_45.csv`)** | 54 (1↔2만) | **92** |

## 규칙

라벨의 1→2단계 경계는 `meas_t1`이 45℃에 도달하는 순간입니다. 97개 실험(train+validation)에서 경계 시점 T1이 45.0~45.4℃이고, train에서 "T1 누적 최고값 ≥ 45℃ ⇔ 2단계 이상"은 예외 0개입니다.

각 실험을 시간순 정렬 후 `t1max = cummax(meas_t1)` (과거·현재 값만 사용):
- B가 1단계인데 `t1max >= 45` → 2단계
- B가 2단계 이상인데 `t1max < 45` → 1단계
- `t1max`가 NaN이면 B 예측 유지. 3~6단계 예측은 변경 없음.

## 검증 (B가 학습하지 않은 데이터)

| 데이터 | 모델 | 정답률 | Macro F1 | 1↔2 오답 | 망가뜨린 행 |
|---|---|---|---|---|---|
| dev | 원본 B | 88.74% | 78.15 | 63 | – |
| dev | B + 규칙 | 89.67% | 78.79 | 0 | 0 |
| validation | 원본 B | 84.09% | 69.63 | 178 | – |
| validation | B + 규칙 | 85.83% | 71.29 | 0 | 0 |

## 분석 파일

- `b_errors.py`: 원본 B의 오답 유형(문제 실험 `E_e5bc` 제외 시 3→2 438, 2→3 214, 3→4 181, 1↔2 226 등), 단계 경계별 센서 변화.
- `threshold_rules.py`: 단계 경계 시점의 온도 분포. 1→2만 고정 온도 기준(45℃)이 존재. 3→4는 전압 급락과 일치(약 78%), 2→3·4→5·5→6은 뚜렷한 기준 없음.
- `rules_12_34.py`: 규칙 ①(T1 45℃)과 규칙 ②(급락 전 4단계 이상 불가) 평가. 규칙 ②는 dev 58개 수정/2개 오류, train 예외 0.7%로 미제출.

## 의존 경로 (저장소 미포함)

- 원본 B 예측: `/home/ubuntu/leewon/abc_runs/current_query_v1/ensemble_B_*_predictions.csv`, `submission_B.csv`
- 전압 급락 특징: `experiments/b_independent_studies/runs/independent_seed42_43_44_v1/experiment1_voltage/*_voltage_event_columns.csv`
