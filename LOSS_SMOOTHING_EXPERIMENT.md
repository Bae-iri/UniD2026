# B 구조 고정: 일반 CE와 Label Smoothing CE 비교

코드 작성만 수행했다. 학습·smoke·추론 실험은 실행하지 않았다.

## 비교 조건

| 이름 | label_smoothing | 손실 |
|---|---:|---|
| ce | 0 | 클래스 가중치가 있는 일반 Cross Entropy |
| ls005 | 0.05 | 같은 클래스 가중치의 Label Smoothing CE |
| ls010 | 0.1 | 같은 클래스 가중치의 Label Smoothing CE |

PyTorch `CrossEntropyLoss(weight=..., label_smoothing=epsilon)`를 사용한다. 클래스 수 K=6일 때 목표 분포는 `(1-epsilon)*one_hot + epsilon/K`다. 클래스 가중치는 세 조건에서 동일하며 smoothing된 분포에도 PyTorch의 가중 CE 규칙으로 적용된다. 무가중 CE와 가중 CE의 비교는 이번 실험에 포함하지 않는다.

## 고정 조건

- 기존 `abc_fusion_experiment.FusionModel('B')`를 그대로 사용. 현재 전기·열 Query가 가스/이미지를 읽고 센서·전체 요약 직접 연결 유지.
- 정적 4개 변수와 시계열 특징 제외. 기존 전기·열 6개 및 결측 표시로 12차원 Query 입력.
- 같은 데이터 전처리, fit/dev/validation 분할, 클래스 가중치, 초기 state와 minibatch 순서.
- seeds 42/43/44, 15 epoch, batch 64, image 128, D=64, AdamW lr=3e-4, weight_decay=1e-4, gradient clipping 5.
- 기존 B의 checkpoint를 fine-tuning하지 않고 각 조건을 같은 초기 가중치부터 학습한다. 기존 결과 파일은 덮어쓰지 않는다.
- dev macro-F1로 checkpoint 선택. validation은 최종 평가에만 사용하고 test는 제출 예측에만 사용.

## 직접 실행

```bash
cd /home/ubuntu/leewon
/home/ubuntu/venv/bin/python -u loss_smoothing_experiment.py --device cuda
```

기본 출력 폴더: `loss_runs/b_smoothing_v1/`. 총 3조건×3seed=9회 학습. 중단 후 같은 명령에 `--resume`을 추가하면 완료된 조건은 건너뛰고 미완료 조건은 처음부터 학습한다. 코드·입력 CSV·이미지·설정이 바뀌면 새 출력 폴더를 지정한다.

```bash
# 사용자가 원할 때만 실행하는 작은 전체 경로 검사; 실제 비교 결과로 해석하지 않음
/home/ubuntu/venv/bin/python -u loss_smoothing_experiment.py \
  --device cpu --smoke --output loss_runs/smoke_smoothing
```

이 스크립트는 같은 폴더의 `abc_fusion_experiment.py`와 `hierarchy_experiment.py`를 import하므로 함께 유지해야 한다. import만으로 학습이 시작되지는 않는다.

## 결과와 제출 파일

- `summary.csv`: 9회 실행의 전체/클래스별 F1, 선택 epoch.
- `aggregate.csv`: 조건별 F1 seed 평균·표준편차.
- `comparisons.csv`: smoothing−일반 CE의 macro-F1 차이와 실험 단위 paired bootstrap 95% 구간.
- `seed_deltas.csv`: seed별 점수와 차이.
- `confusion_<loss>_mean.csv`: 평균 혼동행렬. seed별 원본은 각 실행 폴더에 저장.
- `results_combined.csv`: 실행/평균 F1, 혼동행렬, 차이의 long-format 통합 CSV.
- `ensemble_scores.csv`: seed 확률 평균 앙상블의 dev/validation 성능. 개별 seed 평균과 구분.
- `submission_ce.csv`, `submission_ls005.csv`, `submission_ls010.csv`: **모두 B 모델**이며 손실만 다른 test 예측. 공식 sample_id 순서와 label 1~6 정수 양식을 검사.
- `submission_best_dev.csv`: dev 앙상블 성능으로 자동 선택한 파일의 복사본. 선택 손실과 epsilon은 `submission_info.json` 및 콘솔에 명시. 모호한 이름의 `submission.csv`는 만들지 않는다.
- `report.md`, `config.json`, 코드 snapshot, 분할/전처리 파일, seed별 checkpoint/학습 이력/예측값.

각 smoothing 조건이 일반 CE보다 일관되게 높은 macro-F1을 보이는지 확인한다. 차이 구간에 0이 포함되면 불확실성을 명시한다. 클래스별 성능 저하도 함께 본다. smoothing에 따라 손실의 목표 분포가 달라지므로 train loss 숫자가 더 작다는 이유만으로 우수하다고 판단하지 않는다. 검증 데이터에서 많은 설정을 골라 반복 실험하면 별도 독립 평가가 필요하다. test 점수는 이 코드로 알 수 없으며 외부 제출/업로드는 수행하지 않는다.
