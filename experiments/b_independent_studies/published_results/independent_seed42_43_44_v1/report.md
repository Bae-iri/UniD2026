# 원본 B 기반 독립 실험 결과

| model | split | macro_f1 | F1_1 | F1_2 | F1_3 | F1_4 | F1_5 | F1_6 | support_3 | wrong_3 | error_rate_3 | support_4 | wrong_4 | error_rate_4 | support_5 | wrong_5 | error_rate_5 | true3_pred5 | true5_pred3 | true4_pred5 | true5_pred4 | zero_support_4 | zero_correct_4 | zero_true4_pred5 | zero_error_rate_4 | zero_support_5 | zero_correct_5 | zero_true5_pred5 | zero_error_rate_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B_control | validation | 69.98 | 97.97 | 71.55 | 89.83 | 46.46 | 50.58 | 63.49 | 3066 | 461 | 15.04 | 352 | 50 | 14.2 | 1134 | 718 | 63.32 | 0 | 44 | 35 | 545 | 12 | 6 | 0 | 50.0 | 506 | 22 | 22 | 95.65 |
| experiment1_voltage | validation | 69.84 | 98.43 | 72.03 | 87.96 | 43.85 | 51.3 | 65.49 | 3066 | 574 | 18.72 | 352 | 56 | 15.91 | 1134 | 710 | 62.61 | 0 | 41 | 38 | 548 | 12 | 3 | 1 | 75.0 | 506 | 27 | 27 | 94.66 |
| experiment2_gas_temperature | validation | 69.62 | 98.25 | 73.33 | 89.33 | 43.43 | 50.15 | 63.2 | 3066 | 491 | 16.01 | 352 | 56 | 15.91 | 1134 | 719 | 63.4 | 0 | 44 | 44 | 559 | 12 | 4 | 2 | 66.67 | 506 | 18 | 18 | 96.44 |

| model | reference | macro_f1_delta_pp | F1_3_delta_pp | F1_4_delta_pp | F1_5_delta_pp | error_rate_3_delta_pp | error_rate_4_delta_pp | error_rate_5_delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| experiment1_voltage | B_control | -0.13 | -1.86 | -2.61 | 0.72 | 3.69 | 1.7 | -0.71 |
| experiment2_gas_temperature | B_control | -0.36 | -0.5 | -3.03 | -0.43 | 0.98 | 1.7 | 0.09 |

F1과 오류율은 0~100, 차이는 %p. 오류율은 해당 실제 클래스 표본 중 다른 단계로 예측한 비율(1-recall). 음수 오류율 차이가 개선임.
각 실험은 원본 B seed42/43/44 체크포인트에서 독립 시작. B_control도 동일 추가 학습량으로 비교하며 이전 실험의 학습 결과를 다음 실험으로 넘기지 않음.
실험1: 전압 급락 전용 branch의 logit 잔차 ×2. 실험2: 화학 가스 모두 관측된 0일 때 chemical value ×0.5, 현재 온도 전용 logit 잔차 ×2. 압력 및 이미지 경로는 유지. 실험2는 전압 사건 특징을 계산/입력하지 않음.
계수는 실행 전에 고정했으며 실제 학습된 중요도가 정확히 2배/절반이라는 뜻은 아님. 실험1은 특징 추가+강조의 결합 효과, 실험2는 gas downweight+temperature branch의 결합 효과임.
모든 checkpoint는 dev 기준으로만 선택. validation은 이전에 확인한 탐색 데이터이며 새 blind test가 아님. 제출 점수는 실제 평가 전 알 수 없음.
각 실험 submission.csv는 해당 실험의 seed 확률 평균이며 원본 template의 sample_id,label 순서로 저장. 원본 B 제출 및 checkpoint는 변경하지 않음.

## 검증 완료

각 실험 3-seed 앙상블 제출 CSV는 공식 3,451행 ID 순서와 label 1~6을 검증했습니다. 실험별 code 폴더에도 submission.csv를 복사했습니다.
각 모델·seed의 추가 15 epoch 완료와 원본 epoch0 후보를 확인했습니다. best epoch가 0이면 dev에서 추가 학습의 개선이 없어 원본 상태로 돌아간 경우이며 history.csv에는 실제 학습 이력이 보존됩니다.
두 실험은 별도 model.py/전처리를 사용하며 원본 B만 공통 시작점입니다. 실험2 state에는 전압 사건 branch가 없고 실험1 state에는 gas-zero 온도 branch가 없습니다.
bootstrap 구간은 동일 validation 실험을 paired 재표집한 조건부 불확실성이며 seed 재학습 변동/반복 선택 편향은 포함하지 않습니다. 모델별 실제 실험 분포는 errors_by_experiment.csv를 확인하세요.

## 원본 B 자체와의 비교

| 모델 | Macro F1 | 3단계 오답률 | 4단계 오답률 | 5단계 오답률 |
|---|---:|---:|---:|---:|
| original_B_before_additional_training | 69.63 | 17.91 | 13.64 | 62.61 |
| B_control | 69.98 | 15.04 | 14.20 | 63.32 |
| experiment1_voltage | 69.84 | 18.72 | 15.91 | 62.61 |
| experiment2_gas_temperature | 69.62 | 16.01 | 15.91 | 63.40 |

오답률은 %이며 분모는 3단계3066/4단계352/5단계1134개입니다. 원본 B는 추가 학습 전 seed 확률 평균입니다.
실험1: 원본 B 대비 실제3 오답 549→574, 실제5 오답710→710. 실험2: 실제4 오답48→56, 실제5 오답710→719. 이번 설정은 사용자 목표 오류율 감소를 뒷받침하지 않습니다.
동일 추가학습량 B_control 대비로도 실험1의 3단계 오류가 증가하고, 실험2의 4/5단계 오류가 줄지 않았습니다. 전체 Macro F1의 작은 변화만으로 목표 개선을 주장하지 않습니다.
