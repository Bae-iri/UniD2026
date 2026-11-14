# 4→5 보정의 과적합 억제 실험

주 비교는 시간 없는 과거 상태 보정(`history`)이다. 확률만 쓰는 보정과 시간까지 추가한 보정은 대조군이며, 결과가 더 좋아도 주 비교 대신 골라 제출하지 않는다. 전체 실험 조건은 실행 전에 `past_only_v3/protocol.json`에 기록했다.

## 분리 방식

1. 기존 공식 Train 78개 실험의 5개 outer fold를 그대로 사용한다. 이미 저장된 각 fold B 모델 3개(seed 42/43/44)는 그 fold의 평가 실험을 fit·checkpoint 선택 모두에서 제외했다.
2. 각 outer 학습 실험 안에서 새로 3개 inner fold를 만든다. Inner 학습 실험도 fit/selection으로 나누어 B 3개를 처음부터 학습한다. Scaler·결측값 대체·가중치는 fit만 사용하고 checkpoint는 selection만 사용한다. Inner holdout의 실험은 양쪽에서 제외한다. 총 45개 B 모델을 새로 학습한다.
3. 이 inner OOF 확률과 현재까지의 과거 극값·현재와의 차이로 작은 선형 보정기를 학습한다. B가 4라고 예측한 행만 학습 대상으로 삼고, 실제 5를 복원할지 예측한다. 다른 실제 단계는 모두 음성으로 둔다. 실험별 가중치 합은 1이며, L2 규제 C=1을 고정한다.
4. 수정 기준 .90/.95/.99는 outer 학습 데이터 안의 3-group meta CV로 고른다. 정답 복원이 훼손의 2배 이상, 최소 3개 실험 개선, 개선 실험 수가 악화보다 많음, 어떤 실험 하나를 빼도 pooled F1 개선이 남는 조건을 모두 요구한다. 통과 기준이 없으면 보정을 끈다. 이 안쪽 meta CV는 선택용으로만 사용하며 독립적인 전체 시스템 성능으로 보고하지 않는다.
5. 모든 선택을 끝낸 뒤 outer holdout의 B 확률과 이력을 보정한다. Outer holdout의 라벨은 이 최종 평가에만 사용한다. 바깥 평가 실험을 학습한 B의 OOF 값을 안쪽 보정기 학습에 섞지 않는다.

## 채택 기준

주 비교의 outer OOF F1 증가, 실험 단위 paired bootstrap 하한 양수, 개선 실험 수 우세, 실험 하나 제외 후에도 개선을 요구한다. 공식 Validation의 전체/E16 제외 성능이 기존 fold B 대비 악화하면 채택하지 않는다. 실제 최고 제출의 기존 B와의 호환 평가도 전체/E16 제외에서 악화하지 않아야 한다. Validation 결과에 따라 임계값·모델 용량·특징을 재조정하지 않는다.

통과하지 못하면 `deployment_gate.joblib`의 보정을 비활성화한다. 연구용 원래 후보와 예측은 별도로 보존한다. 이 중첩 학습은 Test를 읽거나 제출 파일을 생성하지 않는다. 별도 제출 인과 감사에서는 사용자의 질문에 답하기 위해 기존 Test 관측으로 원래 제출을 재현했다. 그 감사는 학습·모델 선택에 사용하지 않았다.

## 검증 및 한계

과거에 관찰한 dev·validation이므로 새로운 blind 평가라고 주장하지 않는다. 바깥 평가의 분리는 이번 실행의 계산상 누수를 막는 것이며, 이전 탐색의 선택 편향을 없애지는 못한다. Bootstrap은 고정 OOF 예측의 불확실성이고 가능한 모든 재학습 변동을 포함하지 않는다. 모델 출력 .90 등은 보정 허용 점수이며 실제 정답률 90% 보장이 아니다.

이력 특징은 같은 실험의 현재·과거만 사용한다. 기존 rules3의 초기 전압 기준에 초기 4개 관측에서 미래 참조 가능성이 발견되어 원래 실행들을 중단했다. 현재 `past_only_v3`는 원본 `final_rules3.py`의 미래 참조를 제거한 생산용 코드를 사용한다. 첫 5개 관측에서 현재까지 확인한 최대 전압만 쓰고 이후 그 기준을 고정한다. 이미 학습한 B는 현재 프레임만 입력으로 쓰므로, 분할·전처리·소스를 확인한 뒤 B checkpoint만 재사용한다. 보정기와 보정 기준은 전부 다시 학습한다. 관련 수정과 검증은 `CAUSALITY_FIX.md`에 있다.

실제 `final_submit/submission_rules3.csv`도 직접 재현해 확인했다. 이 미래 참조를 제거해도 3,451행 모두 동일했다. 이는 해당 제출의 실제 영향 확인이며, 미래 참조 구현을 유지할 이유로 사용하지 않는다. 별도 결과는 `submission_rules3_causality_audit.json`에 있다.

`test_conservative.py`는 미래 불변성, 실험 분리, 시간 없는 특징의 시간 배율 불변성, 4→5 외 변경 금지, 기준 미통과시 보정 중단, 중복 시각 거부, 초기 전압 규칙을 포함한 전체 예측의 prefix 불변성을 확인한다. `verify_nested.py`는 실행 후 모든 inner split·전처리·45개 checkpoint·저장 OOF 확률·outer 예측·채택 판정을 독립적으로 재구성하고 실제 실험을 여러 시점에서 잘라도 이전 예측이 같은지 검사한다.

```bash
/home/ubuntu/venv/bin/python -m unittest temporal45_model.test_conservative -v
/home/ubuntu/venv/bin/python -u temporal45_model/nested_train.py --output temporal45_model/past_only_v3 --reuse-inner temporal45_model/nested_causal_v2
/home/ubuntu/venv/bin/python temporal45_model/verify_nested.py temporal45_model/past_only_v3
```

기존 출력 폴더는 덮어쓰지 않는다. `predict_conservative.py`로 배포용 모델을 적용할 수 있다. 관측 CSV, 학습에서 고정한 규칙 임계값 JSON, seed별 `p1`~`p6` CSV 세 개를 입력받고 sample_id로 정확히 정렬한다. 기준 예측도 현재·과거 규칙으로 직접 계산하므로 외부의 미래 참조 예측을 그대로 받지 않는다. 비활성 gate는 이렇게 계산한 기준 예측을 보존한다.
