# 센서 → 중간 반응 → label 가설 검증

`hierarchy_experiment.py`는 기존 `h3.py`를 변경하거나 실행하지 않는 별도 실험이다.

## 검증할 주장

1. 전기·열·환경·배터리 조건으로 가스·압력과 이미지의 일부 반응 특징을 예측할 수 있는가?
2. 실제 중간 반응을 예측하도록 감독하면, 같은 구조의 감독 없는 모델보다 label 일반화 성능이 좋아지는가?
3. 잘못 연결한 중간 반응으로 학습하는 대조군보다 나은가? 기존 병렬 결합과 조건부 선택 모델보다 나은가?

실제 센서값이 가스나 이미지를 물리적으로 생성한다는 의미는 아니다. 동시점 관측 사이의 예측적 의존관계를 검사하며, 인과 방향을 입증하는 실험은 아니다.

## 모델

| 이름 | 정보 흐름 | 학습 손실 |
|---|---|---|
| `flat` | 센서 표현 + 가스/이미지 균등 요약 → label | weighted CE |
| `conditional` | 센서 표현으로 가스/이미지 attention + 직접 센서 연결 → label | weighted CE |
| `hierarchy_noaux` | 센서 → 가스/이미지 예측값 → 실제 가스/이미지와 결합 → label | weighted CE |
| `hierarchy_supervised` | 위와 같은 구조 | weighted CE + 0.1 × 가스 MSE + 0.1 × 이미지 MSE |
| `hierarchy_shuffled` | 위와 같은 구조, 중간 감독 목표만 다른 학습 샘플과 연결 | 위와 같은 손실/가중치 |

세 hierarchy 모델은 구조, 초기 가중치, 파라미터 수가 같다. 센서에서 최종 label로 가는 모든 경로는 예측된 가스/이미지 값과 그 값의 인코더를 거친다. 실제 가스·이미지도 입력으로 유지하므로 모든 모델이 동일한 관측 정보를 받는다. 센서만으로 관측 모달리티를 대체하는 실험은 아니다. 무작위 감독은 train 안에서 고정된 별도 permutation을 사용하며, label·실제 입력·minibatch 순서를 섞지 않는다.

이미지 목표는 letterbox padding을 제외한 4×4 RGB 영역 평균 48차원이다. 학습 모델과 독립적인 고정 특징이므로 목표 인코더가 함께 붕괴하는 문제를 피한다. 이미지 전체의 생성이나 화염/연기 의미 표현까지 검증하지 않는다. 가스 목표는 압력을 포함한 기존 `h3.py`의 11개 관측값이다. 결측값은 감독 손실과 예측 평가에서 제외한다.

## 공정한 비교와 평가

- 제공 train의 78개 실험을 고정 seed `20261114`로 fit/dev에 80/20 분할한다. 같은 실험의 행을 양쪽에 나누지 않는다.
- 전처리·평균값 기준선·클래스 가중치는 fit에서만 계산한다. sample ID, experiment ID, 시간, label을 입력 특징으로 사용하지 않는다.
- dev macro-F1이 가장 높은 checkpoint를 선택한다. 제공 validation 19개 실험은 최종 평가에만 사용한다. 이전 h3 점수와 평가 절차가 다르므로 직접 비교하지 않는다.
- 기본 설정은 seeds `42 43 44`, 15 epochs, batch 64, AdamW `lr=3e-4`, `weight_decay=1e-4`, D=64, image 128이다. 검증 결과로 보조 손실 가중치를 조정하지 않는다.
- 공통 모듈은 기존 h3와 같은 초기화이며, 같은 seed의 모든 모델에 같은 초기 state와 minibatch 순서를 적용한다. 실제 gradient를 받는 파라미터 수도 기록한다. 단계적 구조와 기존 구조의 용량 차이는 있을 수 있어, 핵심 감독 효과는 같은 구조의 noaux/shuffled와 비교한다.
- label은 macro-F1 및 클래스별 F1, 중간 반응은 정규화 MSE와 fit 평균값 예측 대비 skill을 평가한다. `skill=1-SSE/SSE_mean`이며, 0보다 크면 평균값 예측보다 낫다.
- paired bootstrap은 validation의 행 대신 experiment_id 19개를 복원 추출하며, 같은 추출을 모든 모델·seed에 적용한다. 기본 2,000회다. 이는 고정된 학습 모델에 대한 검증 실험 표본 불확실성으로, 새로운 학습 표본 재학습 불확실성까지 포함하지 않는다.
- label이 없는 공식 test는 평가나 학습에 사용하지 않는다.

## 실행

```bash
cd /home/ubuntu/leewon
/home/ubuntu/venv/bin/python -m unittest -v test_hierarchy_experiment
/home/ubuntu/venv/bin/python -u hierarchy_experiment.py \
  --device cuda --seeds 42 43 44 --epochs 15 \
  --output hierarchy_runs/derived_response_v1
```

동일한 명령 뒤에 `--resume`을 붙이면 완료된 모델은 다시 학습하지 않는다. 미완료 모델은 처음부터 학습한다. 코드·CSV·이미지 변경 또는 설정 변경 시 새 출력 폴더를 사용해야 한다.

```bash
# 작은 데이터로 전체 저장/평가 경로만 검사; 연구 결과로 해석하지 않음
/home/ubuntu/venv/bin/python -u hierarchy_experiment.py \
  --device cpu --smoke --output hierarchy_runs/smoke
```

전처리한 uint8 이미지 캐시는 `.hierarchy_cache`에 저장한다. CPU/GPU 메모리에 올려 PNG 재해독 비용을 줄이며, GPU 여유 공간이 부족하면 CPU 캐시를 사용한다. 데이터 원본과 기존 학습 결과는 덮어쓰지 않는다.

## 결과 파일

- `report.md`: 실험 방법, label 점수, 중간 예측 점수, 비교 구간, 해석.
- `summary.csv`, `aggregate.csv`, `comparisons.csv`: 실행별 결과, seed 평균/표준편차, 쌍별 차이.
- `split_manifest.csv`, `preprocessing.json`, `config.json`, `source.py`: 분할·전처리·설정·실제 실행 코드.
- `seed_<seed>/<mode>/best.pt`, `history.csv`, `metrics.json`: checkpoint와 학습 이력.
- 각 실행의 `val_predictions.csv`, `class_report.csv`, `confusion_matrix.csv`.
- hierarchy 실행의 `intermediate_predictions.npz`, `intermediate_features.csv`: 정규화된 중간 예측/목표/관측 mask와 변수별 점수. 행 순서는 `val_predictions.csv`와 같다.

핵심 근거는 실제 감독 모델이 noaux/shuffled보다 label 성능이 좋아지고, 가스·이미지 예측도 평균값 기준선을 넘는지다. 이 조건을 만족해도 인과 관계 전체를 입증하지는 않는다. 만족하지 못하면 이 구현·목표·학습 예산에서 가설을 뒷받침하지 못한 결과로 보고한다.

## 별도 선형 진단

```bash
/home/ubuntu/venv/bin/python sensor_response_probe.py hierarchy_runs/derived_response_v1
```

`sensor_response_probe.py`는 label 학습 없이 센서만으로 가스·이미지 특징을 예측하는 Ridge(alpha=1) 진단이다. 각 목표의 실제 관측 fit 행만 사용한다. 검증 결과로 alpha를 고르지 않으며, 기본 신경망 실험의 모델·설정·평가를 변경하지 않는다. 중간 반응의 예측 가능성을 label 학습과 분리해 보기 위한 보조 분석이며, 결과는 `sensor_only_probe`에 저장한다.
