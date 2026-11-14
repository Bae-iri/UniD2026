# Gated hybrid: 센서·이미지 확률 결합과 단계 관계 학습

위치: `<repo>/experiments/gated_hybrid/`. 기존 UniD2026 코드와 별도로 구성한 새 실험입니다.

## 모델

1. 센서: CatBoost. 현재 전기·열·가스 17개, 과거 30/60초 구간의 평균·표준편차·OLS 기울기·양 끝 변화율, 관측 수·실제 구간 길이·관측 간격, 온도 30/60초 가속도. 총 147개 원시 수치 특징이며 게이트는 표준화된 값과 결측 flag를 받습니다. 정적 bat_class/bat_cc/soc_status/tr_method와 ID, label, raw 상대 시간은 특징에서 제외합니다.
2. 이미지: ImageNet 사전학습 ConvNeXt-Tiny를 고정하고 768차원 특징을 추출합니다. 현재 RGB 열화상만 사용합니다. 종횡비를 유지한 letterbox 224와 ImageNet 정규화를 적용하며 중앙 crop은 하지 않습니다. 학습 가능한 이미지 헤드에는 256차원 trunk와 6단계/단계군/내부 구분 출력이 있습니다. **CNN 전체를 fine-tuning한 실험은 아닙니다.**
3. 계층 관계: 단계군 `[1,2,3] / [4,5] / [6]`의 확률과 각 그룹 내 조건부 확률을 곱해 정규화된 6단계 확률을 만듭니다. 두 expert의 최종 확률은 flat 6단계 예측 70% + 계층 예측 30%입니다. 그룹은 이번 실험의 설계 가정이며 공식 물리 분류 체계나 인과 모델이라고 주장하지 않습니다.
4. 4/5 전문가: CatBoost의 4/5 이진 구분 모델과 이미지의 4/5 조건부 head가 그룹 내부 구분을 학습합니다. gate와 이미지 head의 손실도 6단계 weighted NLL + 0.2 단계군 NLL + 0.4 실제 4/5 표본 조건부 NLL + 0.1 순서 누적확률 BCE로 구성합니다. 4/5 조건부 분포는 `[p4,p5]/(p4+p5)`이며 다른 단계 확률을 임의로 4/5로 바꾸지 않습니다.
5. Gate: 센서 특징·이미지 특징·두 expert 확률·entropy/margin → MLP → sigmoid alpha. 최종 확률은 그림과 동일하게 `alpha*p_sensor + (1-alpha)*p_image`입니다. alpha는 샘플별 스칼라이므로 합은 1이고 음수 확률이 없습니다. 0.5에서 시작하고 게이트 regularization 0.002를 적용합니다. 계층/순서/4·5 손실은 이 결합 확률에 적용합니다.

## 누수 방지와 분할

공식 train의 experiment_id를 expert 49개(20,380행), gate 13개(6,354행), dev 16개(6,788행)로 분리합니다. split seed는 20261114, gate split은 seed+901입니다. expert는 gate 표본의 label이나 이미지를 학습하지 않고 gate는 독립 실험에서 얻은 expert 예측을 입력으로 학습합니다. feature normalizer는 expert 데이터에서만 fit합니다. 전문가와 gate의 checkpoint 선택에는 dev만 사용합니다.

공식 validation 19개(10,200행)와 test 11개(3,451행)는 학습에 넣지 않습니다. 다만 이 validation을 이전 실험에서 여러 번 확인했으므로 이번 결과는 탐색적 평가이며 새로운 독립 blind test로 간주하지 않습니다. 이전 A/B/C 모델과 backbone·학습 데이터·history 설계가 달라 전체 점수 차이를 게이트나 계층 관계 하나의 효과로 분리해 주장하지 않습니다. 같은 run의 sensor/image/uniform/gated 비교는 동일한 experts를 사용합니다.

모든 split은 experiment_id/sample_id 중복을 검사합니다. 이력은 같은 experiment의 현재·과거만 참조하며 순서가 뒤섞여도 시간으로 정렬합니다. OLS 기울기는 구간 내 실제 finite 값과 실제 시각으로 계산하며 결측값을 먼저 채우지 않습니다. 가속도는 정확한 t, t−w, t−2w 값이 있을 때만 `(T(t)-2T(t-w)+T(t-2w))/w²`입니다. 없으면 결측 처리합니다. 센서 단위의 기울기는 값/s, 온도 가속도는 °C/s²입니다.

## 실행

현재 환경은 CatBoost를 `/tmp/gated_hybrid_deps`에 설치했습니다. 새 환경에서는 다음을 먼저 설치해야 합니다.

```bash
/home/ubuntu/venv/bin/python -m pip install --target /tmp/gated_hybrid_deps --no-deps catboost==1.2.10
cd experiments/gated_hybrid
/home/ubuntu/venv/bin/python -m unittest -v test_pipeline
/home/ubuntu/venv/bin/python -u train.py --output runs/seed42_v1
```

기본 학습: seed42, CatBoost 최대 600 trees/depth6/lr0.05/early stopping80(CPU), 이미지 head 30 epochs/AdamW lr3e-4/weight_decay1e-3, gate 40 epochs/동일 optimizer. 클래스 가중치는 학습 파트의 역제곱근 빈도로 각각 계산합니다. CNN 특징 추출 및 이미지 head·gate는 CUDA를 사용합니다. gate는 epoch0의 균등 결합부터 dev 최적 체크포인트 후보로 보존합니다. 기존 결과 폴더는 덮어쓰지 않습니다. 재실행할 때 새 output 경로를 사용하세요.

## 결과

- `summary.csv`: dev/validation의 모델별 Macro F1, 클래스별 F1, 4→5/5→4 혼동 건수와 support.
- `differences.csv`: gated − sensor/image/uniform의 Macro F1 및 4/5 F1 차이.
- `validation_*_confusion_matrix.csv`, `validation_*_classification_report.csv`: 행=실제 단계, 열=예측 단계.
- `image_history.csv`, `gate_history.csv`, `sensor_training.csv`: 실제 학습 진행 및 선택 결과.
- `*_gate_summary.csv`, `*_gated_predictions.csv`: alpha 분포와 샘플별 확률·gate.
- `submission.csv`, `submission_gated.csv`: 요청한 gated 모델의 공식 제출 파일.
- `submission_sensor.csv`, `submission_image.csv`, `submission_uniform.csv`, `submission_dev_best.csv`: 같은 experts의 비교 후보. 마지막 파일은 dev 점수 기준 최상 후보이며 요청 모델과 다를 수 있습니다.
- `.cbm`, `image_head_best.pt`, `gate_best.pt`, `preprocessing.json`: 모델과 전처리. `.weights/`에는 원본 사전학습 가중치, `.image_features/`에는 고정 특징 cache가 있습니다.
- `config.json`, source snapshots, `split_manifest.csv`: 설정·입력 해시·학습 분할. `report.md`는 결과 요약입니다.

공식 sample_submission과 같은 `sample_id,label` 두 열, 3,451행이며 ID를 기준으로 공식 행 순서에 맞춥니다. 라벨은 정수 1~6이고 test 정답은 사용하지 않습니다. 외부 업로드/제출은 하지 않습니다. 단일 seed 실험이므로 성능 개선이나 4/5 혼동 감소는 결과를 보고 판단해야 합니다.

## 참고

- [TorchVision ConvNeXt-Tiny 및 사전학습 가중치](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)
- [CatBoost 학습/최적 모델 선택](https://catboost.ai/docs/en/concepts/python-reference_catboost_fit)

## 실행 완료 결과 (seed 42)

이미지 헤드 30 epoch, 게이트 40 epoch를 학습했고 게이트는 dev 기준 epoch 1을 선택했습니다. validation Macro F1은 sensor 77.98, image 66.03, uniform 76.67, gated 78.24입니다. 게이트의 4단계 F1은 78.53, 5단계 F1은 58.56입니다. 단일 seed 및 반복 확인한 validation의 탐색 결과입니다.

같은 전문가의 균등 결합 대비 +1.57%p, 센서 단독 대비 +0.26%p입니다. experiment 단위 paired bootstrap 95% 구간은 각각 [+0.12,+4.03], [−4.42,+1.75]%p로 센서 단독 대비 일관된 개선을 확정할 수 없습니다. 최종 gate의 4→5 26건, 5→4 85건이지만 5→3 오답 479건이 남습니다. 5단계 재현율은 42.68%이므로 후속 개선 대상으로 봐야 합니다.

추가 분석은 아래 명령으로 실행했습니다. `scores_percent.csv`, `paired_bootstrap.csv`, `errors.csv`, `stage45_errors.csv`, `stage5_to3_errors.csv`를 생성합니다.

```bash
/home/ubuntu/venv/bin/python analyze.py runs/seed42_v1
```

학습 폴더 `runs/seed42_v1/submission.csv`가 gated 모델 제출 파일이며 dev 최상 후보도 gated입니다. 공식 템플릿의 3,451행과 모든 ID 순서, 라벨 1~6 정수 여부를 검증했습니다. 보고서는 `runs/seed42_v1/report.md`에 있습니다.

## Git에 포함한 결과

`published_results/seed42_v1/`에 결과 요약·혼동행렬·학습 이력·제출 CSV를 보존했습니다. 가중치와 이미지 특징 캐시는 용량 때문에 제외했고 위 명령으로 재학습할 수 있습니다.

사용자가 이 모델 제출 점수 84를 보고했습니다. 이는 사용자 보고 점수이며 코드로 확인한 로컬 validation Macro F1 78.24와 별개입니다.
