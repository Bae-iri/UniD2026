# 원본 B + t1 45°C 규칙 + 5→4·6→5 보정 모델

제출 파일: `final_submit/submission_rule1base_targeted.csv`

## 구성

1. **원본 ABC B** (`leewon/abc_runs/current_query_v1`, seed 42/43/44 평균, public 0.9121). 각 행의 현재 센서·가스·이미지만 사용한다.
2. **t1 45°C 규칙**: 같은 실험에서 현재 행까지의 meas_t1 최고값이 45°C 이상이면 2단계 이상, 미만이면 1단계로 둔다. train+validation 97개 실험의 1→2 전환 시점 meas_t1이 45.0~45.4°C다(이가현 분석). 이 단계의 출력은 public 0.92 파일과 같다.
3. **보정 모델 2개** (HistGradientBoosting, `targeted_rule1/model.py`)
   - 4→5: 현재 센서, 현재까지 온도 최고·전압 최저와 그 이후 경과시간, 최근 60초 변화, 관측 간격·경과시간을 쓴다. B 확률은 쓰지 않는다. 기준 예측이 4인 행만 5로 바꿀 수 있다.
   - 5→6: 위 특징에 B 3-seed 현재 확률, 최근 30·60초 확률 이력, 과거 6 예측 이력을 더한다. 기준 예측이 5인 행만 6으로 바꿀 수 있다.

모든 특징은 같은 실험에서 현재 행까지의 관측만 사용한다. 미래 행, 실험 전체 통계, 실험 길이는 쓰지 않는다.

## 학습·검증

- B 확률: `b_oof.py`가 원본 ABC B를 train 78개 실험 단위 5-fold × seed 3개로 재학습한다. 각 train 행은 그 실험을 보지 않은 B의 확률을 갖는다. 전처리는 fold별 학습 실험으로만 적합한다.
- 보정 모델 평가: 같은 5-fold에서 나머지 fold로 학습하고 해당 fold로 평가한다. 보정 기준(.5/.7/.9)은 학습 fold 안의 실험 단위 3-fold로만 고른다.
- 채택 조건(사전 고정): 교차검증 F1 개선, 실험 bootstrap 95% 하한 > 0, 개선 실험 수 > 악화 실험 수, 실험 하나를 빼도 개선 유지, validation 비악화. Validation은 학습·기준 선택이 모두 끝난 뒤에만 읽는다.
- Test는 고정된 모델로 추론에만 사용한다.

| 평가 (기준 = 원본 B + t1 45°C) | Macro F1 변화 | 고침 / 훼손 |
|---|---:|---:|
| train 78실험 교차검증 | +1.41pt (95% 하한 +0.72pt) | 151 / 31 |
| 실험별 | 개선 33 / 악화 7 | 한 실험 제외 최소 +1.25pt |
| validation (원본 B) | +2.03pt | 63 / 7 |
| validation (E16 제외) | +1.97pt | 44 / 7 |

최종 보정 기준: 4→5 = 0.9, 5→6 = 0.7. Test에서는 4→5 기준을 넘은 행이 없고, 0.92 파일 대비 5→6 22행이 바뀐다.

## 미래 관측 검증 (Test)

- 실험별로 나눠 추론: 전체 추론과 동일
- 행 순서를 섞어 추론: 동일
- 각 실험의 앞부분만 입력(132개 지점): 결과가 달라진 곳 0

과거 이력을 쓰므로, 실험 중간 행을 빼고 입력하면 남은 행의 예측이 달라질 수 있다.

## 실행

```bash
source /home/ubuntu/venv/bin/activate
cd ~/JEONGYUN
python -u targeted_rule1/b_oof.py --output targeted_rule1/b_oof_v1 --device cuda   # B 교차검증 재학습 (약 10분)
python -u targeted_rule1/run.py --output targeted_rule1/run_v1                     # 보정 모델 학습·평가·Test 추론
```

`run.py`는 `run_v1/submission_targeted_both.csv`를 만든다. 이 파일이 `final_submit/submission_rule1base_targeted.csv`다. 학습된 보정 모델은 `run_v1/deployment_heads.joblib`, 결과는 `run_v1/results.json`, Test 검증은 `run_v1/test_manifest.json`에 있다.

현재 Test 추론은 원본 B의 seed별 확률 CSV(`leewon/abc_runs/current_query_v1/seed_*/B/test_predictions.csv`)를 읽는다. 최종 `test.py`에서는 원본 B checkpoint로 직접 추론하도록 바꿔야 한다.
