# 고정된 4개 규칙 조합의 실험 단위 검증

- 기준: `rules3` (45°C, 초기 전압 대비 80%, 용량별 t1 하한 1% 분위).
- 추가 A: 용량별 실제 6단계의 열화상 누적 최고값 **1% 분위** 미만이면 6→5.
- 추가 B: 용량별 실제 1·2단계의 t1 누적 최고값 **99.5% 분위** 초과이면 2 이하→3.
- 비교 조합: 기준 / 기준+A / 기준+B / 기준+A+B.

분위수·조합을 고정하고 새 threshold search는 하지 않는다. Test 데이터와 public에서 추측한 행별 정답을 사용하지 않는다. 기존 제출 CSV를 변경하지 않는다.

## 평가 설계

1. 기존 B의 저장 예측: 원래 fit 62개 실험에서 임계값을 계산하고 dev/validation에 공통 적용한다. validation은 E16 포함/제외를 별도로 저장한다.
2. 새로운 전체 시스템 GroupKFold: 공식 Train 78개 실험을 5-fold로 나눈다. 각 outer fold 안에서 fit/selection 실험을 80/20으로 나누고, scaler·class weights·B 모델은 inner fit에서만 학습한다. checkpoint는 inner selection에서만 고른다. 규칙의 임계값은 outer train에서 계산한다. Outer holdout은 모두에서 제외한다.
3. B는 원본 구조·AdamW lr3e-4·weighted CE·batch64·15epoch·3 seeds(42/43/44)를 사용한다. 15개 모델을 새로 학습하며 기존 checkpoint를 초기값으로 사용하지 않는다. 네 조합에는 같은 fold의 같은 B 확률 평균을 쓴다.
4. 공식 Validation은 checkpoint/임계값 선택에 사용하지 않는다. 마지막에 15개 새 모델의 평균과 Train 전체로 추정한 규칙으로 평가한다. 이전부터 탐색한 자료이므로 새 blind validation이라고 주장하지 않는다.

저장 자료의 사전 탐색과 네 후보 선택은 이미 이루어졌다. 이 평가는 학습/평가 실험 분리를 갖춘 고정 가설 재검증이며, 새로운 독립 데이터나 private 개선의 보증은 아니다. 각 fold의 B가 실제로 학습하는 실험 수는 inner fit 기준이며 원래 62개 fit보다 적을 수 있다. 비교의 목적은 네 규칙 조합의 차이를 분리하는 것이고, 절대 F1을 public 점수와 대응시키지 않는다.

## 산출물

- `fold_*/split_manifest.csv`: fit/selection/holdout sample ID와 실험 ID.
- `fold_*/thresholds.json`, `preprocessing.json`: 해당 fold의 임계값·전처리·학습 실험 목록.
- `fold_*/seed*.pt`, `seed*_history.csv`, `seed*_probs.npz`: 새 학습과 checkpoint 선택 증거. NPZ의 holdout 행 순서는 split_manifest의 holdout 순서와 같다.
- `train_oof/summary.csv`: pooled F1, 실험 단위 paired bootstrap 2000회, 실험별 승/패, 최악 손실, 특정 실험 제외 민감도.
- `train_oof/per_experiment.csv`, `conditions.csv`, `changes.csv`: 실패한 실험과 용량/SOC/방법별 조건, 실제 수정·훼손.
- `failure_cases.csv`, `threshold_ranges.csv`, `individual_seed_summary.csv`: 완료 후 추가 요약.
- `validation_fold_ensemble*`: 공식 Validation 포함/제외 평가.
- `decision.json`: 내부 기준 통과 여부. 기존 제출은 자동으로 교체하지 않는다.

과거 기록의 '좋아진 실험/나빠진 실험'이 정확도 기준인 경우와 혼용하지 않도록, 이번 승/패는 같은 실험의 6클래스 Macro F1 차이로 계산한다. 정답 훼손 건수도 따로 저장하므로 실험 전체가 좋아져도 내부 손해가 가려지지 않는다. bootstrap은 고정된 OOF 예측에 조건부이고 모델 재학습 및 과거의 적응적 탐색 불확실성은 포함하지 않는다.

## 실행

```bash
/home/ubuntu/venv/bin/python -m unittest four_rule_cv.test_run -v
/home/ubuntu/venv/bin/python -u four_rule_cv/run.py --output four_rule_cv/runs/new_run
/home/ubuntu/venv/bin/python -m four_rule_cv.summarize four_rule_cv/runs/new_run
```

본 실행 결과는 `runs/group5_seed42_43_44_v1/`에 있다. 기존 출력 폴더는 덮어쓰지 않는다. CPU 검증만 필요하면 `--saved-only`를 사용한다.
