# B와 B+의 실험별 Validation 진단

`compare_b_gate_experiments.py`는 기존6개 checkpoint(2모델×3seed)를 CPU에서 추론한다. 재학습하지 않고 Validation만 사용한다. `compare_attention_gate.py`, `analyze_experiment_performance.py`가 같은 폴더에 필요하다.

```bash
source /home/ubuntu/venv/bin/activate
cd ~/JEONGYUN
python -u compare_b_gate_experiments.py \
  --run runs_b_gate_v1 \
  --abc-run /home/ubuntu/leewon/abc_runs/current_query_v1 \
  --data-dir /home/ubuntu/data/battery_data_light \
  --output experiment_diagnostics/b_gate_validation_v2
```

- B=게이트 없음. B_gate=B+=가스·이미지 attention 출력에 각각 게이트 적용.
- 단일 seed별 비교와 고정 동일 가중치3seed 확률 평균을 구분한다.
- ABC의 A/B/C 앙상블은 참고용으로 함께 집계하지만 우리 B/B+와 기반 모델·학습법이 달라 게이트 효과 비교로 해석하지 않는다.
- 기존 ABC 예측과 sample ID·정답·실험 구분을 맞춰 같은 Validation 행인지 확인한다. E01~E19는 기존 보고서와 같은 별칭이다.
- 원본/행별 데이터는 저장하거나 출력하지 않는다. 혼동행렬·클래스별 F1·실험별 집계만 저장한다.
- Test를 읽거나 제출 CSV를 생성하지 않는다. 기존 학습 파일은 변경하지 않는다.
- 새3seed 앙상블 점수는 기존 단일seed 제출의 공개 점수와 다른 결과다.
- bootstrap은 고정 모델의 실험 구성 불확실성이며 재학습 불확실성은 포함하지 않는다.
- CPU 수치 차이가 발생할 수 있어 checkpoint의 저장 점수와 재현 점수를 함께 기록한다. 기존 결과 폴더를 덮어쓰지 않는다.
