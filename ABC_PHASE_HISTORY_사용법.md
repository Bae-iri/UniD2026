# ABC B: 경과시간과 과거 극값의 추가 가치 비교

## 비교하는 네 모델

| 모드 | 추가 정보 |
|---|---|
| B | 없음. 동일한 예산으로 최종 MLP만 추가 학습하는 대조군 |
| B_time | 현재 경과시간의 log1p |
| B_extrema | 현재까지 최소전압/최고온도, 현재값과의 차이, 유효 표시 |
| B_extrema_time | B_extrema + 현재 경과시간 + 각 극값 발생 이후 시간의 log1p |

전압은 meas_v, 온도는 meas_t1/meas_t2/thermal_t다. 네 번째 모델은 시간 대조군과의 비교가 가능하도록 **현재 경과시간도 포함**한다. 극값 발생 이후 시간의 독립 효과까지 완전히 분리하는 설계는 아니다.

모든 추가 입력은 최종 MLP에만 넣는다. 기존 센서 query, 가스/이미지 attention, 직접 전달 경로와 원래 전처리는 유지한다. 정적 조건을 새로 추가하지 않는다.

## 빠르게 비교하는 방법

- 기존 ABC B checkpoint에서 시작한다. 기존 파일을 수정하거나 덮어쓰지 않는다.
- CNN·센서/가스 인코더·query·attention을 고정한다. 각 split의 현재 특징을 한 번 추출한 뒤 네 모델의 최종 MLP만 학습한다. 특징은 메모리에만 유지하고 행별 파일로 저장하지 않는다.
- 각 모드의 분류기는 같은 크기로 만들며 추가 입력 가중치를 0으로 초기화한다. 시작 시 네 모드 모두 기존 B와 같은 출력을 낸다.
- B는 추가 입력을 모두 0으로 채운다. 모드별 유효 입력 차이는 실험 개입이며, 파라미터 수가 같다는 것이 유효 용량까지 동일하다는 뜻은 아니다.
- 같은 seed·샘플 순서·weighted CE·batch64·AdamW lr3e-4/weight_decay1e-4·epoch 수로 맞춘다.
- 기본값은 seed42, 각3epoch. 원래 Train 내부 fit/dev 분할을 유지하며 dev 최고 epoch를 선택한다. epoch0도 선택 후보라 추가 학습이 모두 나빠지면 기존 B로 복귀한다.
- Validation은 checkpoint 선택 후 전체 행을 평가한다. 기존 분석을 이미 본 뒤 설계한 탐색 실험이므로 독립적인 최종 성능 증거는 아니다.
- 이 비교는 인코더 고정 조건의 추가 가치 검증이다. 전체 네트워크 재학습 성능까지 결론 내리지 않는다.

## 실행

필요 파일: `compare_abc_phase_history.py`, `abc_reference/__init__.py`, `abc_reference/abc_fusion_experiment.py`, `abc_reference/hierarchy_experiment.py`. 테스트용으로 `test_abc_phase_history.py`도 둔다.

```bash
source /home/ubuntu/venv/bin/activate
cd ~/JEONGYUN

# 합성 데이터만 사용하는 검증
python -m unittest -v test_abc_phase_history

# 사용자가 직접 실행하는 짧은 실제 비교
python -u compare_abc_phase_history.py \
  --base-run /home/ubuntu/leewon/abc_runs/current_query_v1 \
  --data-dir /home/ubuntu/data/battery_data_light \
  --output phase_history_runs/quick_seed42 \
  --seeds 42 --epochs 3 --device cuda
```

같은 조건으로 재확인하려면 새 output 경로에 `--seeds 43 44`를 지정한다. 학습 시간은 측정 전이다. CNN은 seed당 fit/dev/Validation 각각 한 번만 추론하며, 실제로 몇 초/몇 분 걸리는지는 GPU 부하에 따라 다르다.

CPU를 쓰려면 `--device cpu`. 원래 run의 preprocessing, config, checkpoint, image_cache_manifest와 캐시 파일을 사용한다. CSV hash·split 순서가 원래 run과 다르면 오류를 내며 임의로 재학습/재정렬하지 않는다.

## 누수 방지와 결측

같은 experiment_id 안에서 relative_time_s로 정렬해 현재까지의 관측만 사용한다. 전압 최솟값·온도 최댓값에는 현재 관측도 포함한다. 미래, 실험 종료시각, 전체 길이, label은 입력 특징으로 사용하지 않는다. 중복 시각은 오류다. 극값이 유지되면 처음 도달한 시점을 기준으로 이후 시간을 센다.

정규화는 fit에서만 계산한다. 관측되지 않은 극값과 현재 결측에 따른 차이는 별도로 결측 처리하고, 정규화 후 0으로 채운다. 원래 현재 센서 경로의 결측 표시와 추가 극값 유효 표시를 함께 유지한다. 가스0을 결측으로 바꾸지 않는다.

## 확인할 결과

`summary.csv`에 전체 F1, 3/4/5/6단계 F1, E16의5→4 오류와5단계 정답 수, 최고 dev epoch를 기록한다. `paired_differences.csv`에는 같은 seed의 모드별 차이를 저장한다.

각 run의 `validation_aggregates.json`에는 전체/실험별 클래스 F1과 혼동행렬이 있다. `best.pt`에는 원래 B state, 선택된 head state, 전처리와 추가 특징 통계가 포함된다. 원본/행별 예측은 출력하거나 저장하지 않는다. Test 접근·제출 CSV 생성은 하지 않는다.

전체 F1과 E16 외 실험의 5단계 개선을 함께 본다. 5→4가 줄어도 5→3으로 옮겨갔다면 해결이 아니다. 3단계 성능 손실도 확인한다. B_extrema_time이 B_time보다 좋아야 단순 경과시간 이상으로 이력을 넣을 근거가 생긴다.

기존 output은 덮어쓰지 않는다. 중간 재개는 구현하지 않았다. 중단하면 완료된 epoch의 best.pt/history.json은 남고, summary는 완료된 run만 포함할 수 있다.

## 검증 상태

합성 테스트3개 통과: 미래 제거 불변성·실험 분리·중복 시간 거부·결측·모드별 입력, 기존 B와 초기 출력 일치·공통 가중치·gradient, 네 모드의1epoch 학습·checkpoint 복원·epoch0 선택·원본 checkpoint 불변·덮어쓰기 방지. 실제 데이터 학습은 실행하지 않았다.
