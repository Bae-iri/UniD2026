# 게이트 체크포인트 진단

`analyze_attention_gates.py`는 재학습 없이 저장된 모델의 게이트를 분석한다. `compare_gate_branches.py`가 같은 폴더에 필요하다. 기존 b_gate.py/image_gate.py에서 저장한 checkpoint도 지원한다.

- 가스·이미지 게이트의 평균, 표준편차, 분위수, 0.1 미만/0.9 초과 비율.
- 고정 경과시간 구간별 게이트 분포와 시간 상관계수.
- attention 출력 벡터의 게이트 적용 전후 L2 크기.
- 같은 학습 가중치에서 이미지/가스/양쪽 게이트를 1로 바꿨을 때 예측 변경 비율 및 정답 있는 split의 집계 F1.
- 마지막 항목은 재학습한 B와의 비교가 아니라 추론 시 개입이다. 특징 크기는 물리적 중요도나 최종 출력 기여도가 아니다.
- 사용되지 않는 게이트 출력은 미학습 상태이므로 해석하지 않는다.
- 행별 정보는 저장·출력하지 않으며 Test의 성능 지표는 계산하지 않는다. Test 분포는 사후 진단용이며 이를 이용해 파라미터·임계값·모델을 맞추지 않는다.

```bash
source /home/ubuntu/venv/bin/activate
cd ~/JEONGYUN
python -u analyze_attention_gates.py \
  --checkpoints runs_b_gate_v1/B_gate_seed42/best.pt runs_image_gate_quick/B_image_gate_seed42/best.pt \
  --data-dir /home/ubuntu/data/battery_data_light \
  --splits validation test \
  --device cpu --threads 2 --batch-size 64 \
  --output gate_diagnostics/seed42_validation_test.json
```

기존 출력은 덮어쓰지 않는다. CPU 기본값으로 진행 중인 GPU 학습과의 경합을 줄이지만 이미지 읽기와 CPU 부하는 발생한다. split에 train을 추가하면 학습 입력 분포의 게이트도 집계할 수 있다. 전처리를 다시 학습하지 않는다.
