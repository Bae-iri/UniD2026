# ABC B：가스 의존 진단과 드롭아웃 비교

기존 팀원 코드·checkpoint는 수정하지 않는다. `abc_reference/`는 원래 ABC run의 source/common_source snapshot 사본이며 ABC 내부 import만 패키지 상대 경로로 바꿨다.

## 실행 완료한 진단

`diagnose_abc_gas_dependency.py`는 기존 B3seed를 CPU에서 평가한다. 가스 요약과 가스 attention 출력 둘 다0으로 바꾸는 추론 개입이며 압력도 포함한다. 농도0과 결측을 동일시하지 않는다. 기존 이미지 캐시를 COW mmap으로 열어 원본 캐시를 쓰지 않는다. 정상 앙상블 예측이 기존 저장 예측과 일치해야 결과를 저장한다. 원본 행/개별 예측은 출력하지 않고 Validation 집계만 저장한다.

```bash
python -u diagnose_abc_gas_dependency.py \
  --base-run /home/ubuntu/leewon/abc_runs/current_query_v1 \
  --data-dir /home/ubuntu/data/battery_data_light \
  --output experiment_diagnostics/abc_gas_off_v2
```

결과상 가스 제거는 전체/E16 성능을 악화시켰다. `5→4` 감소만 보고 성공으로 판단하면 안 된다. E16의5단계 정답 수가6→0으로 감소했다. 가스를 무시하는 규칙이나0을 일괄 결측으로 바꾸는 처리는 권하지 않는다.

## 준비한 짧은 학습 비교(실데이터 학습 미실행)

두 모델은 같은 기존 B checkpoint에서 시작한다.

- B_continue: 기존 B의 동일 예산 추가 학습.
- B_gas_dropout: 같은 추가 학습 중 샘플별20% 확률로 두 가스 경로를 함께0으로 만듦. 추론에서는 모든 입력 사용. 역확률 크기 보정은 하지 않음.

기존 encoder/head, 원래 fit/dev 분할, 전처리, weighted CE, batch64, lr3e-4, weight_decay1e-4 유지. 두 조건 모두 optimizer를 새로 만든다. 기존 dev에서 epoch0 포함 최고 checkpoint 선택. Validation은 선택 완료 뒤 집계 평가. 원래 B와 fine-tuning 결과의 차이는 드롭아웃 효과가 아니므로 반드시 B_continue와 비교한다.

가스 제거 진단이 부정적이므로 이 실험은 성공을 예상하는 권장안이 아니라 준비된 탐색 옵션이다. 기본2조건×seed42×3epoch이며 장시간 학습은 시작하지 않았다.

```bash
source /home/ubuntu/venv/bin/activate
cd ~/JEONGYUN
python test_abc_gas_dropout.py

# 사용자가 탐색하기로 결정한 경우에만 실행
python -u compare_abc_gas_dropout.py \
  --base-run /home/ubuntu/leewon/abc_runs/current_query_v1 \
  --data-dir /home/ubuntu/data/battery_data_light \
  --output gas_dropout_runs/quick_seed42 \
  --seeds 42 --epochs 3 --drop-probability 0.2 --device cuda
```

summary.csv, paired_differences.csv, run별best.pt/history.json/Validation 집계 JSON, 원본 전처리·설정·checkpoint SHA를 저장한다. 기존 output을 덮어쓰거나 resume하지 않는다. Test 접근·제출 생성·원본/행별 결과 저장은 없다. 합성 테스트는 eval의 기존 B 동일성, 마스킹 경로, 양쪽 추가 학습·checkpoint·집계·덮어쓰기 방지를 검사한다.
