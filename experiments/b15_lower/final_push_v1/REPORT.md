# 최종 보정 후보 검증

`submission_consensus.csv`를 우선 후보로 선택했다. 기존 `submission_stage6.csv`의
6단계 예측, 1단계 예측, 4→3 보정, S3에서 변경한 예측을 보존한다.
새 모델 두 개의 보정 결과가 같을 때만 2·3·4·5단계 예측을 수정한다.
실제 Test 3,451행 중 64행이 달라졌다. 리더보드 점수는 정책 선택에 사용하지 않았다.

첫 번째 모델은 현재 센서, 과거 15/60초 통계, 관측된 최초값·최댓값과의 차이,
실험 시작부터 알려진 조건, 현재 이미지의 4×4 RGB 평균을 사용한다.
두 번째 모델은 같은 특징에 고정된 ImageNet ResNet18의 현재 이미지 특징 512개를 추가한다.
둘 다 동일한 얕은 HistGradientBoosting 설정으로 2·3·4·5단계를 학습한다.
기존 B의 조건부 확률과 새 모델 확률을 각각 50:50으로 합친 뒤,
두 보정 결과가 일치할 때만 변경한다. ID별 예외나 실험 종료시각·길이는 입력하지 않는다.

새 CNN을 900 epochs 학습하지 않았다. 원래 B 가중치는 변경하지 않았고,
이미지 인코더도 고정했다. 새 보정기의 boosting 반복 상한은 180이다.
검증용 추가 B 교사는 각 15 epochs 안에서 내부 실험 선택만 사용한다.
기존 주 검증의 900 epochs는 60개 모델 × 15 epochs의 합계임을 학습 기록에서 확인했다.
반복 횟수 자체를 과적합 여부의 판단 근거로 삼지 않는다.

## 실험 단위 검증

| 검증 묶음 | 기존 S3 | 합의 보정 | 변화 |
|---|---:|---:|---:|
| 전체 97실험, 5개 외부 분할, E16 포함 | 0.852522 | 0.863345 | +0.010823 |
| 공식 Train 78실험, 별도 5개 분할 | 0.894886 | 0.904810 | +0.009923 |
| 기존 개발 77실험, 반복 2×5개 분할 | 0.875850 | 0.885771 | +0.009921 |
| 기존 제출 B 가중치, 공식 Validation 19실험 | 0.774527 | 0.780824 | +0.006297 |

97실험 주 평가의 5개 분할은 모두 개선됐다. 관측 간격 2배/3배, 교대 간격,
30% 누락을 포함한 8개 관측 조건에서도 풀링한 평균이 개선됐다.
배터리 용량 6개 그룹, SOC 2개 그룹, 유도 방식 2개 그룹으로 나눈 평균도 모두 개선됐다.
추가 15개 분할 중 12개는 개선되고 3개는 하락했다. 가장 큰 하락은 약 0.00349다.
검증 결과가 모든 실험에서 같다는 뜻은 아니다.

외부 평가 실험은 각 B의 학습·checkpoint 선택·정규화 및 보정기 학습에서 제외했다.
원래 S3 보정기 학습에는 외부 학습 실험 안에서 다시 교차 적합한 B 확률을 사용했다.
새 raw 보정기는 B 확률을 학습 입력으로 사용하지 않는다.
진단용 stack 보정기의 내부 평가에서는 별도 single-group B 교사까지 사용해
내부 평가 실험이 학습용 확률 생성에 섞이지 않도록 했다.

제출용 새 보정기는 공개된 97실험 전체로 적합했다.
위 19실험 검증 점수는 **Train 78실험으로 적합한 별도 보정기**의 결과다.
최종 97실험 적합 모델을 같은 19실험에서 다시 평가한 학습 점수를 검증으로 제시하지 않는다.
97실험 교차검증의 B 학습량은 기존 제출 B와 다르므로, 해당 수치는 방법의 비교다.
기존 제출 B를 그대로 사용하는 19실험 평가를 함께 제시했다.

## 선택 편향과 E16

raw/stack·표현 방식·혼합률을 비교한 외부 교차검증으로 최종 정책을 선택했다.
따라서 최종 정책의 97실험 점수를 완전히 보지 않은 nested 평가라고 부르지 않는다.
별도로 내부에서 선택한 conservative 정책의 외부 점수는 이미지 표현 사용 시
0.859294이며, 기존 대비 +0.006772, 실험 단위 paired bootstrap 95% 구간은
[+0.000934, +0.012254]다.

최종 합의 정책의 주 평가 개선 구간은 약 [+0.00544, +0.01744]다.
추가 탈락 후보 등을 포함해 유한 후보군 32개를 보수적으로 반영한 Bonferroni bootstrap 구간도
[+0.00246, +0.02240]이다. 이 보정이 이전의 모든 수동 탐색을 제거하지는 않는다.
실험 재표집은 학습을 다시 돌리지 않는 근사이며, 분할 간 학습 자료가 겹친다.
이미 여러 번 검토한 자료이므로 새 blind test나 비공개 70% 성능 보장은 아니다.

E16은 주 평가와 공식 Validation에서 제외하지 않았다.
기존 제출 B를 쓰는 19실험 평가에서 E16의 Macro F1은 0.489885→0.489757이다.
E16의 실제 5→예측 4 오류는 539→540으로 해결되지 않았다.
이 조건이 비공개 자료에 많다면 성능을 제한할 수 있다.
전체 평균 개선을 E16 문제 해결로 해석하지 않는다.

## 파일과 재현

- `submission_consensus.csv`: 우선 후보, 기존 대비 64행 변경.
- `submission_average.csv`: 두 전문가 확률을 평균하는 대안. 기존 대비 71행 변경,
  합의 후보와 7행 차이. 주 평가 0.864211이지만 추가 분할에서는 합의 후보가 더 안정적이었다.
- `cv_family.json`: 3개 검증 묶음·8개 관측 조건·각 분할의 전체 지표.
- `deployment_audit.json`: 기존 제출 B를 사용하는 검증 및 제출 파일 검사.
- `raw_inference_audit.json`: 실제 이미지 기반 전체 재생성과 33개 prefix 일치 검사.
- `selection_adjusted.json`: 후보 비교를 보수적으로 반영한 근사 구간.
- `condition_scores.json`: 용량·SOC·유도 방식별 그룹 검증 결과.

선택한 모델 묶음은 작업 공간의
`/home/ubuntu/gahyun/experiments/final_push_20261115/delivery/model.joblib`에 있다.
기존 B 가중치와 고정 이미지 인코더가 함께 필요하다. 아래 명령은 학습이나
캐시된 행별 예측 없이 이미지와 센서에서 제출 파일을 재생성한다.
모든 가중치와 코드를 포함하는 독립 실행 묶음은
`/home/ubuntu/gahyun/experiments/final_push_20261115/submission_bundle.zip`이다.
압축을 풀면 `test.py --input ... --image_root ... --output ...`으로 실행할 수 있다.

```bash
/home/ubuntu/venv/bin/python experiments/b15_lower/boost_predict.py \
  --input /home/ubuntu/data/battery_data_light/test/test.csv \
  --image-root /home/ubuntu/data/battery_data_light/test \
  --base /home/ubuntu/leewon/abc_runs/current_query_v1 \
  --artifact /home/ubuntu/gahyun/experiments/final_push_20261115/delivery/model.joblib \
  --weights /home/ubuntu/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  --sample-submission /home/ubuntu/data/battery_data_light/sample_submission.csv \
  --output /tmp/submission_consensus_recreated.csv --device cuda --audit-prefix
```

기존 B는 CPU에서, 이미지 인코더는 고정 single-image 연산으로 실행한다.
전체/잘린 시계열의 특징과 라벨, 행 순서 변경, label/sample_id 변경을 검사했다.
Test의 11개 실험을 실제 이미지에서 33번 잘라 실행한 결과도 모두 일치했다.
CPU 이미지 인코더로 다시 생성해도 동일한 CSV가 나왔다.
기존 `submission_stage6.csv`는 수정하지 않았다.

추가로 전체 학습 보정기와 80% 실험 부분집합 5개의 보정기를 평균하는
고정 앙상블을 검사했다. 주 평가에서 합의 정책이 0.863087로 내려갔고,
나머지 세 정책도 원래 보정기보다 낮아 사전에 정한 채택 조건에 미달했다.
추가 15개 분할은 중단했고 제출 파일은 변경하지 않았다.
이는 `bagging_rejected.json`에 별도로 기록했다.
