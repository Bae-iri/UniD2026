# 이미지–센서 attention 출력 게이트 비교

compare_gate_branches.py는 독립 스크립트다. compare_attention_gate.py, compare_condition_query.py, compare_image_history.py와 기존 결과는 수정하지 않는다.

## 비교

- B: 전기·열 센서와 현재 경과시간으로 query 생성. 가스 토큰과 CNN 공간 토큰을 cross-attention으로 읽음. 정적 조건은 최종 분류기에만 전달.
- B_image_gate: B에서 이미지 attention 출력에만 샘플별 스칼라 게이트 적용.
- B_gas_gate: 가스 attention 출력에만 게이트 적용.
- B_gate: 이미지와 가스 attention 출력에 각각 게이트 적용. 기존 compare_attention_gate.py의 B_gate와 같은 구조.

게이트 입력은 공통 전기·열+경과시간 인코더 특징이다. Linear(dim,2)→sigmoid로 가스/이미지 게이트를 만들며 해당 모드에서 필요한 출력만 사용한다. 원래 이미지 전역 요약·가스 요약·전기열·정적 조건 직접 연결은 유지한다. 과거 이미지·센서는 사용하지 않는다.

모든 모드는 같은 seed에서 동일한 state_dict로 시작하며 게이트도 모두 생성한다. B는 게이트 미사용, 이미지/가스 단독 모드는 출력 한 개만 사용한다. allocated 파라미터 수가 같아도 유효 용량은 다르다. 게이트는 처음 0.5이므로 초기 예측은 B와 다르다. 개선이 상황별 조절만의 효과인지는 고정 게이트 비교 없이 확정할 수 없다.

CNN은 현재 compare_condition_query.py의 작은 CNN, 외부 가중치 없음. Train만으로 전처리, 같은 전체 Validation Macro F1로 최고 epoch 선택. 원본 대회 데이터·정답·행별 예측은 AI로 전달하지 않는다.

## 실행

사용자가 실행한다. H2와 기존 B/B+ 학습이 진행 중이면 GPU 자원을 공유한다. 네 가지를 전부 돌릴 필요 없이 기본 두 모델로 시작한다.

```bash
source /home/ubuntu/venv/bin/activate
cd ~/JEONGYUN
python compare_gate_branches.py --smoke-test
python -u compare_gate_branches.py train \
  --data-dir /home/ubuntu/data/battery_data_light \
  --output runs_image_gate_quick \
  --modes B B_image_gate \
  --seeds 42 --epochs 10 --batch-size 32 --device cuda
```

기본 비교는 2모델×1seed×10epoch. 탐색 결과이므로 유망하면 같은 조건의 3seed로 재확인한다. 기존 B/B+ 실험과 구조는 호환되지만 seed·epoch·batch·lr·image-size·dim 등 조건이 맞는지 확인 없이 점수를 합치지 않는다. 이 스크립트는 기존 결과를 자동 재사용하지 않는다.

가스와 조합까지 비교하려면 새 output으로:

```bash
python -u compare_gate_branches.py train \
  --data-dir /home/ubuntu/data/battery_data_light \
  --output runs_gate_ablation_v1 \
  --modes B B_image_gate B_gas_gate B_gate \
  --seeds 42 43 44 --epochs 20 --batch-size 32 --device cuda
```

결과:

- summary.csv: 최고 Macro F1·epoch·학습+검증 총 시간.
- paired_differences.csv: 같은 seed에서 게이트별 B 대비 차이, 양쪽 게이트와 단독 게이트의 차이.
- aggregate.csv: 평균·표준편차·반복 수.
- 각 run의 history.json: epoch별 손실·Macro F1·클래스별 F1·혼동행렬·시간·전체 Validation 처리 시간.
- best.pt: 가중치·전처리·설정. predict 명령은 compare_attention_gate.py와 같은 인자 사용.

Validation 처리 시간은 로딩·전처리·전송·모델 연산을 포함하며 순수 GPU 추론 지연시간이 아니다. 동시 학습 부하가 다르면 속도 차이를 게이트 효과로 해석하지 않는다. 예측 품질과 시간 비용을 함께 비교한다.

기존 run 폴더는 덮어쓰지 않는다. 중간 재개는 미구현이므로 재실행 시 새 output 사용. 최종 재학습·앙상블 미구현.

합성 통합 검증: `python test_compare_gate_branches.py`. 네 모드 학습·복원·제출 순서·덮어쓰기 방지와 smoke test를 검사하며 대회 데이터는 읽지 않는다.
