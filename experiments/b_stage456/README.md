# 원본 B + 4·5·6단계 전문가 (B 동결)

원본 B(리원 `FusionModel('B')` 3-seed 앙상블) 체크포인트와 예측 확률은 **그대로** 사용합니다. B가 4 이상으로 예측한 행에서만 별도 전문가가 4/5/6을 다시 결정합니다. 1~3단계 예측과 단계 경계는 바뀌지 않습니다.

## 리더보드

| 제출 | 원본 B 대비 test 변경 | 점수 |
|---|---|---|
| 원본 B | 0 | 91 |
| **V3 (`submission_B_spec456_V3.csv`)** | 55 | **90** |

로컬 validation(dev 83.28 / validation 73.54, 원본 B 78.15 / 69.63)에서는 개선됐지만 리더보드에서는 1점 낮았습니다.

## V3 구성

- 전문가: HistGradientBoosting × 5 bag 평균(`max_leaf_nodes=15`, `lr=0.05`, `300 iter`), 클래스 균형 가중치. (MLP가 아니라 트리 모델입니다.)
- 학습 데이터: 실제 4·5·6단계 행만 사용. 최종 제출은 fit+dev+validation.
- 특징(38개, 과거·현재 관측만): 현재 전압·가스·압력, 급락 사건 특징(급락 후 경과시간 등), 온도·전압·압력의 30초/2분/5분 변화율과 누적 최댓값. **현재 온도 절댓값은 제외**(`variants.py`의 V3).
- 결합: 4/5/6 확률 = 0.7 × 전문가 + 0.3 × B(4~6 재정규화).

## 주요 분석 (파일)

- `pre_analysis.py`, `info_ablation.py`, `info_importance.py`: 4 vs 5 구분 정보. 시간 이력(특히 급락 후 경과시간)이 핵심이고, 이미지는 낯선 실험에서 일반화가 약함.
- `pair_experts.py`, `s4.py`: 3↔5 / 4↔5 쌍 전문가. test에서는 3↔5 전문가가 바꾸는 행 없음.
- `linear.py`: 선형 보정(L1/L2)은 트리보다 약함.
- `soften_b.py`: B 비율을 낮추거나 B 확신을 완화해도 dev·`E_e5bc` 제외 성능 하락.
- `find_diff.py`, `gas_collapse.py`, `battery_duration.py`: validation `E_e5bc`(280Ah, 4단계 생략, 급락 후 장시간 5단계)의 5→4 오답 원인 분석. 배터리별 4단계 최대 지속시간 규칙은 dev 5→4를 49→22로 줄였으나 test 변경이 `E_a221` 한 실험에 집중되어 미제출.
- `battery_check/`: `bat_class` one-hot, `log1p(bat_cc)` 표준화를 특징으로 추가(C1/C2). dev·validation·train GroupKFold 모두 차이 ±0.3 수준으로 기여 없음.

## 의존 경로

`tune.py`는 아래 로컬 산출물을 읽습니다(저장소 미포함).
- 원본 B 예측: `/home/ubuntu/leewon/abc_runs/current_query_v1/ensemble_B_*_predictions.csv`
- 전압 급락 특징: `experiments/b_independent_studies/runs/independent_seed42_43_44_v1/experiment1_voltage/*_voltage_event_columns.csv`

트리 학습 시 다른 GPU/CPU 작업과 경합하면 매우 느려지므로 `OMP_NUM_THREADS=6` 정도로 제한해 실행하세요.
