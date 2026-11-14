# 현재 상태에 따른 가스·이미지 선택의 A/B/C 비교

`abc_fusion_experiment.py`는 이전 중간 반응 예측 실험과 독립적인 실험이다. 보조 예측·재구성 손실은 사용하지 않는다.

- A: 유효 가스 토큰과 CNN 공간 토큰을 각각 균등 pooling.
- B: 현재 전기·열·환경 관측으로 만든 Query가 각 모달리티를 scaled dot-product cross-attention으로 읽음.
- C: 선택용 Query만 모달리티별 학습 가능한 고정 벡터로 교체. 실제 센서 표현은 최종 head에 유지.

모든 모델의 head 입력은 `[현재 센서 표현, 가스 전체 요약, 이미지 전체 요약, 선택된 가스, 선택된 이미지]`다. encoder, value/output projection, FFN, head, 초기 state가 같고, A/B/C에서 정보 선택 규칙만 바뀐다. 모든 모델에 같은 파라미터를 선언하지만 실제 gradient가 있는 파라미터 수는 다를 수 있어 함께 기록한다. B/C는 학습 가능한 동일한 attention projection을 사용하고 C에 고정 Query 학습이 추가된다.

센서 입력은 meas_v, meas_t1, meas_t2, atmp_t, thermal_t, indu_t 및 각 결측 표시(총 12차원). 기존 입력 목록을 유지하며 train에서 항상 0인 meas_a는 제외한다. 중복/파생 열 indu_t.1, meas_t1_plus_t2, 시간, ID, 정적 4개 변수는 입력으로 사용하지 않는다. 가스는 압력 포함 기존 11개 변수다. 이미지의 padding과 가스 결측은 pooling/attention에서 제외한다. 정규화·결측 대체·클래스 가중치는 fit에서만 계산한다.

train은 실험 단위 fit/dev 62/16개로 나누고 checkpoint를 dev macro-F1로 선택한다. validation 19개 실험은 최종 평가에 사용한다. seeds 42/43/44, 15 epoch, batch 64, image 128, D=64, AdamW lr=3e-4/weight_decay=1e-4, inverse-sqrt 빈도의 weighted CE가 기본값이다. 같은 seed에서 모든 state와 minibatch 순서가 같다.

```bash
/home/ubuntu/venv/bin/python -m unittest -v test_abc_fusion_experiment
/home/ubuntu/venv/bin/python -u abc_fusion_experiment.py --device cuda --output abc_runs/current_query_v1
# 완료된 모델부터 재개: 같은 명령에 --resume 추가
# 축소 검사: --device cpu --smoke --output abc_runs/smoke_v1
```

주요 결과는 seed 평균 macro-F1, 클래스별 F1, seed별 차이, B−A 및 B−C와 실험 단위 paired bootstrap 95% 구간이다. 혼동행렬은 seed별 원본과 seed 평균 count를 모두 저장한다. 앙상블 성능은 별도 `ensemble_scores.csv`에 기록하여 단일 seed 평균과 혼동하지 않는다.

판단은 B>A와 B>C를 모두 확인한다. 두 차이가 양수여도 불확실성 구간이 0을 포함하면 확정적인 개선으로 해석하지 않는다. 가스·이미지 attention을 동시에 바꿨으므로 이미지 단독 효과로 해석하지 않는다.

## CSV 및 제출

- `summary.csv`: 9개 학습 실행별 macro/클래스별 F1.
- `aggregate.csv`, `comparisons.csv`, `seed_deltas.csv`: 평균/표준편차, 비교 차이, seed별 차이.
- `confusion_A_mean.csv` 등: seed 평균 혼동행렬.
- `results_combined.csv`: 실행/평균 F1, 평균 혼동행렬, 점수 차이를 row_type으로 구분한 long-format 통합 CSV.
- `submission_A.csv`, `submission_B.csv`, `submission_C.csv`: 각각 3-seed 확률 평균 test 예측.
- `submission.csv`: dev 앙상블 macro-F1이 가장 높은 모델의 파일. 선택 기준과 모델은 `submission_info.json`에 기록.

제출 양식은 공식 sample_submission과 동일한 sample_id,label 두 열, 3,451행이다. ID를 기준으로 결합하여 공식 행 순서를 유지한다. ID 집합, 누락, 중복, label 1~6 정수 여부를 검사한다. test 정답은 알 수 없고, 파일 생성까지 수행하며 외부 업로드는 하지 않는다. smoke 파일은 실제 제출용이 아니다.
