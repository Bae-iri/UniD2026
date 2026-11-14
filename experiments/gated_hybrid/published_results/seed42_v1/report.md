# Gated hybrid + hierarchy + 4/5 specialist

| split | model | macro_f1 | F1_1 | F1_2 | F1_3 | F1_4 | F1_5 | F1_6 | true4_pred5 | true5_pred4 | support4 | support5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dev | sensor | 0.8475 | 1.0 | 0.8183 | 0.8787 | 0.8727 | 0.698 | 0.8174 | 42 | 45 | 358 | 285 |
| dev | image | 0.7798 | 0.9787 | 0.7641 | 0.824 | 0.6192 | 0.6703 | 0.8225 | 53 | 28 | 358 | 285 |
| dev | uniform | 0.8223 | 0.9917 | 0.8077 | 0.8648 | 0.7405 | 0.7008 | 0.8286 | 43 | 38 | 358 | 285 |
| dev | gated | 0.8484 | 0.9934 | 0.8205 | 0.8914 | 0.8478 | 0.7082 | 0.8291 | 38 | 38 | 358 | 285 |
| validation | sensor | 0.7798 | 1.0 | 0.8305 | 0.9299 | 0.5117 | 0.5681 | 0.8385 | 26 | 565 | 352 | 1134 |
| validation | image | 0.6603 | 0.9738 | 0.5102 | 0.7107 | 0.5882 | 0.4843 | 0.6945 | 35 | 70 | 352 | 1134 |
| validation | uniform | 0.7667 | 0.99 | 0.6797 | 0.8063 | 0.7391 | 0.5716 | 0.8137 | 40 | 76 | 352 | 1134 |
| validation | gated | 0.7824 | 0.9916 | 0.6873 | 0.8102 | 0.7853 | 0.5856 | 0.8345 | 26 | 85 | 352 | 1134 |

| candidate | reference | macro_f1_delta | F1_4_delta | F1_5_delta |
| --- | --- | --- | --- | --- |
| gated | sensor | 0.0026 | 0.2736 | 0.0175 |
| gated | image | 0.1221 | 0.197 | 0.1013 |
| gated | uniform | 0.0157 | 0.0462 | 0.014 |

센서 CatBoost + 고정 사전학습 ConvNeXt-Tiny 특징의 학습 가능한 이미지 head. 센서·이미지 expert를 먼저 학습하고, 별도 실험의 gate 학습 데이터에서 alpha를 학습. dev에서만 checkpoint 선택.
p_final = alpha * p_sensor + (1-alpha) * p_image. 단계군 [1,2,3]/[4,5]/[6] 및 그룹 내부 조건부 예측과 ordinal/4·5 보조 손실. 단계군은 설계 가정이며 물리적 인과 학습을 의미하지 않음.
정적 4개 변수 제외. 동일 experiment의 현재·과거 센서만 사용. 현재 열화상 RGB만 사용. CNN backbone은 frozen이며 end-to-end fine-tuning은 하지 않음.
단일 seed의 탐색 실험. validation은 이전 실험에서 이미 확인했으므로 독립적인 신규 최종 평가가 아님. 이전 모델과 학습 데이터·backbone이 달라 단순 점수 차이를 게이트나 계층 단독 효과로 해석하지 않음.
submission.csv는 요청한 gated 모델의 공식 제출 파일. submission_dev_best.csv는 dev 점수 기준 후보 중 최상 모델. 모든 CSV는 sample_id,label 두 열과 공식 ID 순서를 확인함. 외부 제출하지 않음.

Elapsed seconds: 85.4

## 최종 검증

모델별 제출 CSV와 submission.csv의 공식 ID 순서·3,451행·sample_id,label 열·정수 label 1~6을 검증했습니다. submission.csv는 gated 모델이며 dev 점수 기준 최상 후보도 gated입니다.

| 비교 | Macro F1 차이 (%p) | 실험 bootstrap 95% 구간 |
|---|---:|---:|
| gated − sensor | +0.26 | [-4.42, +1.75] |
| gated − image | +12.21 | [+8.02, +17.67] |
| gated − uniform | +1.57 | [+0.12, +4.03] |

구간은 동일 validation experiment를 쌍으로 2,000회 재표집한 결과로, 학습 seed·재학습 변동이나 반복된 모델 선택의 불확실성은 포함하지 않습니다.

4/5 혼동뿐 아니라 다른 단계로 이동한 오답을 함께 확인해야 합니다. gated 모델의 실제 5단계 중 484/1,134개만 정답이며(재현율 42.68%), 3단계로 예측한 오답이 479개 남아 있습니다. stage45_errors.csv와 stage5_to3_errors.csv에 센서·이미지 예측과 alpha를 같이 저장했습니다.

4→5 26건, 5→4 85건. 4/5 혼동 감소가 5단계의 모든 오류 해결을 의미하지 않습니다. 계층/4·5 손실의 개별 효과는 별도 ablation 없이 확정하지 않습니다.
