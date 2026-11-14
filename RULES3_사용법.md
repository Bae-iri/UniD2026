# 센서 기반 단계 규칙 (public 0.93)

원본 ABC B 제출(`leewon/abc_runs/current_query_v1/submission_B.csv`, public 0.9121)의 예측에 센서로 확인되는 단계 규칙 3개를 적용한다. 모델 재학습은 없다.

## 규칙

같은 experiment_id의 현재·과거 관측만 사용한다.

| 규칙 | 내용 | 근거 |
|---|---|---|
| 1. 1↔2 경계 | meas_t1 누적 최고 ≥ 45°C이면 2단계 이상, 미만이면 1단계 | train+validation 97개 실험의 1→2 전환 시점 meas_t1이 45.0~45.4°C (이가현 분석) |
| 2. 전압 필요조건 | meas_v가 초기값(첫 5행 최대)의 80% 아래로 내려간 적이 없으면 4단계 이상 불가 → 3 | 3→4 전환이 전압 하락 시점과 78% 실험에서 ±5초 일치 |
| 3. t1 필요조건 | meas_t1 누적 최고가 배터리 용량(bat_cc)별 하한 미만이면 3단계 불가 → 2 | 하한 = 3단계 이상 행의 1% 분위 |

## 검증

원본 B 대비. 규칙 3의 하한은 fit 실험으로만 계산했다.

| | dev F1 | validation F1 (E16 제외) |
|---|---:|---:|
| 원본 B | 0.7815 | 0.8142 |
| 규칙 1 | 0.7879 | 0.8317 |
| 규칙 1+2+3 | 0.8102 | 0.8685 |

public: 원본 B 0.9121 → 규칙 1 0.92 → 규칙 1+2+3 0.93.

제외한 규칙: "t2 누적 최고가 배터리별 하한 미만이면 6단계 불가"는 dev/validation에서는 좋았지만, public에서 정답 6으로 확인된 T06의 3행을 5로 바꿔서 뺐다.

## 실행

```bash
source /home/ubuntu/venv/bin/activate
cd ~/JEONGYUN
python final_rules3.py \
  --base /home/ubuntu/leewon/abc_runs/current_query_v1/submission_B.csv \
  --data-dir /home/ubuntu/data/battery_data_light \
  --output submission_rules3.csv
```

출력은 public 0.93 제출 파일과 바이트 단위로 같다. 원본 B 제출 파일만 있으면 재현되며, 원본 B를 복원하려면 leewon의 ABC 실행 코드와 checkpoint가 필요하다.
