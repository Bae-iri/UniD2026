# B vs B+게이트

`b_gate.py`는 h1.py/h2.py import 없이 실행하는 독립 스크립트다. 기존 코드와 결과는 수정하지 않는다. 사용자 직접 실행용이다.

## 비교 정의

- B: H1 late와 같은 현재 관측 경로. 전기·열과 경과시간으로 query를 만들고, 가스 토큰·CNN 공간 토큰에 cross-attention 적용.
- B_gate: 동일한 경로에 `sigmoid(Linear(electric_feature, 2))` 추가. 샘플별 가스·이미지 스칼라를 각각 attention 출력에 곱함.
- 정적 조건은 최종 분류기에만 입력. 원래 전기·열·가스 요약·이미지 전역 요약과 정적 특징의 직접 경로는 유지.
- 과거 입력, 외부 가중치, 색 증강, 클래스 가중치 없음. Train 전처리, 동일 Validation 전체 행으로 평가.
- 양쪽에 게이트 모듈을 생성하여 공통 초기값을 맞추지만 B는 사용하지 않는다. B_gate의 유효 학습 파라미터가 더 많다(기본 dim=64에서 130개).
- 게이트 가중치·bias는 0으로 초기화하므로 처음 출력은 0.5. 공통 가중치는 같지만 초기 예측은 다르다. 결과에는 초기 감쇠와 추가 표현력 효과도 포함되며, 상황별 조절만의 효과라고 단정하지 않는다.
- 후속으로 필요하면 모든 샘플에 같은 학습 게이트를 쓰는 대조군으로 상황별 조절 효과를 분리한다.
- B도 새로 학습한다. 기존 H1/H2 점수를 이 비교의 대조군으로 재사용하지 않는다.

## 실행

H2와 동시에 GPU 학습하면 메모리 경합과 속도 저하가 생길 수 있으므로, GPU 여유가 있을 때 사용자가 실행한다. 본 문서 작성 중 실제 학습은 실행하지 않았다.

```bash
source /home/ubuntu/venv/bin/activate
cd ~/JEONGYUN
python b_gate.py --smoke-test
```

한 seed, 한 epoch 점검:

```bash
python -u b_gate.py train \
  --data-dir /home/ubuntu/data/battery_data_light \
  --output runs_b_gate_check \
  --seeds 42 --epochs 1 --batch-size 32 --device cuda
```

본 비교(2모델 × 3 seeds):

```bash
python -u b_gate.py train \
  --data-dir /home/ubuntu/data/battery_data_light \
  --output runs_b_gate_v1 \
  --seeds 42 43 44 --epochs 20 --batch-size 32 --device cuda
```

같은 seed에서 초기 공통 가중치·샘플 순서·학습 설정을 맞춘다. 모델별 Validation Macro F1 최고 epoch를 저장한다. `summary.csv`, `paired_differences.csv`의 B_gate−B, `aggregate.csv`, 클래스별 F1을 함께 본다. 평균뿐 아니라 seed별 일관성을 확인한다.

기존 run 폴더가 있으면 시작 전에 중단한다. 중간 재개는 없다. 재실행은 새 output 경로를 지정한다. summary/비교 표는 각 run 또는 전체 완료 시 갱신되고 history는 epoch마다 저장된다.

추론은 선택한 모델의 checkpoint를 지정한다(B_gate 선택을 미리 가정하지 않음).

```bash
python b_gate.py predict \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image-root /home/ubuntu/data/battery_data_light/test \
  --checkpoint runs_b_gate_v1/B_seed42/best.pt \
  --sample-submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output submission_b_seed42.csv --device cuda
```

원본 대회 데이터·정답·행별 예측을 AI로 전달하지 않는다. 가중치만으로 추론 복원 가능하며 최종 재학습·앙상블·행별 확률 저장은 미구현이다.
