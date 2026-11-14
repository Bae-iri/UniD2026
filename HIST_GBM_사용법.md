# 센서 + 인과 이력 특징 GBM (최종 후보)

`hist_gbm.py` 단독 스크립트. 신경망·이미지·GPU 없이 CPU에서 약 1분. 기존 코드·결과는 수정하지 않는다.

## 근거

- train·validation 모든 실험에서 label은 시간에 따라 감소하지 않는다(단조 증가 100%). ABC B는 현재 행만 보고 과거를 쓰지 않는다.
- 예측 후처리(누적 최대, HMM, 평활)는 validation에서는 개선되지만 Train 내부 dev에서는 악화돼 채택하지 않았다. 오류가 잡음이 아니라 구간 단위의 체계적 오류다.
- 과거 이력을 **입력 특징**으로 넣으면 dev·validation 모두 개선된다.

| 모델 | dev | Validation |
|---|---:|---:|
| ABC B (3seed) | 0.78 | 0.696 |
| GBM 현재값만 | 0.82 | 0.73 |
| GBM + 인과 이력 | 0.87 | 0.813 |
| GBM + 인과 이력 + 첫 관측 대비 변화량 (최종) | - | **0.843** |

- Train+Validation 97실험 GroupKFold(5): 0.881 → 0.883.
- Validation 5단계 F1 0.51→0.86, 4단계 0.44→0.75.

## val 0.70 → public 0.91 차이의 원인

- 실험 단위 교차검증으로 train/val/test를 구분하는 분류기 AUC는 0.34~0.51(무작위 수준 이하). split 간 체계적 분포 차이는 없다. test의 정적 조건 조합 9개도 모두 train에 있다.
- 차이의 원인은 실험 간 편차, 그리고 val의 E16이다. E16 한 실험이 ABC B val 오류의 52%를 차지한다. ABC B는 E16을 빼면 val 0.814다.
- 따라서 E16 제외 val과 97실험 GroupKFold가 public의 더 나은 대리 지표다. E16 제외 val: ABC B 0.814, 이 모델 0.913. 실험별로는 19개 중 18개에서 ABC B보다 높다.
- 이미지를 실험 첫 프레임 대비 변화량으로 넣으면 E16 제외 val은 0.922로 오르지만 GroupKFold가 0.871로 떨어져 채택하지 않았다.
- 이미지 통계 추가 시 Validation 약 -7pt(실험별 과적합 추정)라 제외. ABC B와 확률 블렌딩도 악화.
- ABC B ↔ ABC B+이력(인코더 고정, 최종 MLP만 학습, `compare_abc_phase_history.py`) 비교는 +0.7pt 이하로 효과 미미.

## 특징

같은 experiment_id 안에서 relative_time_s ≤ 현재 시점의 관측만 사용한다. 미래 관측·실험 전체 통계는 쓰지 않는다.

- 현재 센서(전압, 압력, 온도 7종, 가스 10종), 경과시간, 정적 조건 4개
- 신호별 현재까지 최대/최소, 최대값과의 차이
- 전압·압력·가스합·온도의 실험 첫 관측 대비 변화량(실험별 기준값 차이 상쇄)
- 30초·120초 창 평균과 변화율(행 수가 아니라 초 단위 → test의 15초 간격 구간에도 같은 정의)
- 가스 최초 검출 후 경과시간, 과거 가스 검출 비율, meas_t1/thermal_t 최고점 이후 경과시간

Validation을 test처럼 일부 구간 15초 간격으로 솎아도 0.855→0.850으로 유지됐다.

## 실행

```bash
source /home/ubuntu/venv/bin/activate
cd ~/JEONGYUN
D=/home/ubuntu/data/battery_data_light

python hist_gbm.py eval --data-dir $D                          # Train→Val, Train+Val GroupKFold(5)
python hist_gbm.py fit  --data-dir $D --model hist_gbm.joblib  # Train+Val 전체, seed 0/1/2
python hist_gbm.py predict --data-dir $D --model hist_gbm.joblib \
  --input $D/test/test.csv --sample-submission $D/sample_submission.csv \
  --output submission_hist_gbm.csv
```

`hist_gbm.joblib`만 있으면 재학습 없이 같은 제출 파일을 복원한다. 두 번 독립 학습해 제출 파일이 바이트 단위로 같음을 확인했다.

- hist_gbm.joblib SHA256: `78e0d9d0d735d4d3bc0c834978ba5176e5ce938b5cb1332ccdfb834228119791`
- submission_hist_gbm.csv SHA256: `dc89addb7c3e78e732fff4fcefb3c7aeb5dae15188f1d14aa4aed56aef0a718b`
- 예측 분포: 1:1363, 2:417, 3:964, 4:268, 5:302, 6:137 (3451행)

## 주의

- ABC B는 validation 0.70인데 public 0.91이었다. test는 11실험뿐이라 validation이 public을 그대로 예측하지 않는다. 이 모델의 public 점수는 제출해서 확인해야 한다.
- `.gitignore`가 `*.joblib`, `*.csv`를 제외하므로 가중치·제출 파일은 git에 올라가지 않는다. 서버에 따로 보관한다.
- 원본 데이터·정답·행별 예측은 AI로 전달하지 않았다. 집계 지표만 확인했다.
