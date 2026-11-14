# 원본 B(동결) + MLP456 + 배터리 조건 Feature Gating (V0~V3)

> **주의 (2026-11-14):** `run_v1`은 정규화 버그가 있던 코드로 만들었다. train에서 상수인 `observation_gap_log1p`가 10·15초 간격 test 실험(E_a221, E_c8e730)에서 크게 증폭됐다. 5초 간격인 train·validation 지표는 영향이 없다. 그러나 `run_v1/submission_condgate_*.csv`는 제출에 쓰지 않는다. 수정과 재검증은 `../b_cadence_robust/`에 있다.

배터리 종류(`bat_class`)와 용량(`bat_cc`)이 4·5·6단계 판단에서 센서 특징의 반영 비중을 바꾸도록 학습시키는 실험이다.

## 구성

- **기준 예측**: 원본 ABC B(`leewon/abc_runs/current_query_v1`, 3-seed 평균) + t1 45°C 규칙. B 체크포인트와 확률은 읽기만 한다.
- **MLP456(V0)**: 기준 예측이 4·5·6인 행만 다시 결정한다. 출력 logit = log(B의 4·5·6 재정규화 확률) + MLP 보정이다. 보정 출력층을 0으로 초기화하므로 학습 전에는 B와 같다.
  - 입력 77개(현재·과거만): `targeted_rule1.model.features`의 센서 이력·B 확률 이력 67개와 현재 가스 10개(log1p).
  - 학습 대상: 실제 4·5·6단계 행. 가중치는 (실험, 클래스)별 균형이다. 2층 64 은닉, 40 epoch, seed 42/43/44의 평균을 쓴다.
- **V1~V3**: 학습된 V0을 동결하고 조건 모듈만 20 epoch 학습한다. 조건 입력 c = bat_class one-hot + 표준화한 log1p(bat_cc)이다.
  - V1: c를 첫 은닉층에 단순 결합한다(0 초기화).
  - V2: x' = x ⊙ (1 + α·tanh(g(c))), α=1. g의 마지막 층을 0으로 초기화해 시작 시 gate=1이다.
  - V3: V2와 같고 α=0.25, 페널티 1e-2·tanh(g)²를 둔다.
- 하이퍼파라미터는 모두 사전 고정이며 성능을 보고 고르지 않았다.

## 평가

- train 78실험: `targeted_rule1/b_oof_v1`과 같은 실험 단위 5-fold로 평가한다. 입력 B 확률도 그 fold의 OOF이다.
- validation 19실험: train 전체로 학습한 뒤 한 번만 평가한다. B 확률은 원본 B와 fold 앙상블 B 두 가지다.
- test: 고정 모델로 추론만 한다. 실험별 분리, 행 셔플, prefix 132개에서 결과가 모두 같았다(불일치 0).

## 결과 (Macro F1, 6클래스)

| 평가 | B 기준 | V0 | V1 | V2 | V3 |
|---|---:|---:|---:|---:|---:|
| train OOF | .8413 | **.8612** | .8613 | .8607 | .8614 |
| val (원본 B) | .7129 | .7820 | .7831 | **.7961** | .7858 |
| val (원본 B), E_e5bc 제외 | .8317 | **.8925** | .8920 | .8906 | .8913 |
| val (fold B) | .7125 | .7714 | .7752 | .7783 | .7737 |
| val (fold B), E_e5bc 제외 | .8633 | .9098 | **.9111** | .9073 | .9088 |

V0 대비 변경(정답→오답 / 오답→정답)과 5→4 오답 수:

| 평가 | V1 | V2 | V3 | 5→4 (V0 / V2) |
|---|---|---|---|---|
| train OOF | 12 / 13 | 36 / 31 | 16 / 18 | 98 / 97 |
| val (원본 B) | 1 / 8 | 7 / 86 | 3 / 26 | 505 / 422 |
| val (원본 B), E_e5bc 제외 | 1 / 0 | 6 / 2 | 3 / 1 | 14 / 15 |

4→5 오답은 모든 변형에서 OOF 125~127, validation 20~21로 같다. V0 대비 Macro F1 차이의 실험 bootstrap 95% 구간은 모든 변형·평가에서 0을 포함한다.

## 해석

1. **이득은 MLP456(V0)에서 나온다.** B 대비 OOF +1.98pt, validation +6.9pt(E_e5bc 제외 +6.1pt)다.
2. **배터리 조건 모듈은 일반화된 이득이 없다.** OOF 차이는 ±0.04pt이고, E_e5bc를 빼면 validation에서 V1·V2·V3 모두 V0보다 약간 낮다.
3. **V2의 validation +1.4pt는 E_e5bc 한 실험에서 나온다.** 오답→정답 86행 중 84행이 E_e5bc(280Ah)다.
4. **그 이득의 대부분은 배터리 조건과 무관하다.** 조건을 다른 배터리 조합으로 바꿔 넣어도 V2는 V0 대비 +0.8pt를 유지한다(5개 shift 평균). 실제 조건을 넣었을 때는 +1.4pt다. 학습된 gate는 모든 조합에서 공통으로 1보다 크다(평균 |gate−1| = 0.23). 조합별 차이는 그 절반 정도다(0.11). 따라서 V2는 배터리별 해석보다는 MLP 보정을 전체적으로 키우는 방향으로 학습됐다.
5. **센서 그룹별 제거 실험에서 일관되게 중요한 그룹이 없다.** 그룹 하나의 gate를 1로 되돌려도 OOF 변화는 0.2pt 이내이고, 부호도 평가마다 바뀐다(`summary.csv`의 `reset_*`, `only_*`). gate 크기는 가스·압력·시간 그룹이 가장 크다. 그러나 제거 실험이 이 크기를 성능 기여로 뒷받침하지 않는다.
6. V3(변화량 제한)는 V2보다 덜 흔들린다(OOF 변경 34행, V2는 67행). 그러나 V0 대비 이득은 없다.

결론: 이 데이터(6개 배터리 조합, train 78실험)에서는 배터리 조건별 특징 가중치를 학습해도 4·5·6 판단이 나아지지 않는다. 기존 `battery_check/`(HistGBM에 one-hot 추가, ±0.3pt)와 같은 결론이다. 제출 후보로는 V0만 볼 만하다.

## 파일 (`run_v1/`)

- `summary.csv`: 평가×방법별 지표(Macro F1, B·V0 대비 차이, bootstrap 구간, 4↔5·5↔6 혼동, 정답↔오답 변경, 개선/악화 실험 수). 제거 실험 `V*|cond_shift*`, `V*|reset_<그룹>`, `V*|only_<그룹>`도 들어 있다.
- `per_experiment.csv`: 실험별 정확도와 V0 대비 고침/훼손 수.
- `changed_rows_vs_V0.csv`: V0과 예측이 달라진 행 전체.
- `gates_by_combo.csv`: V2·V3 최종 모델의 배터리 조합×특징별 gate 값.
- `final_models.pt`, `protocol.json`(소스 해시 포함), `test_manifest.json`.
- `submission_condgate_V0~V3.csv`: B+t1 규칙 대비 test 변경 행 수는 V0 97, V1 97, V2 98, V3 97이다. V0 대비로는 V1 2행, V2 9행, V3 5행이 다르다. 리더보드 점수는 아직 없다. 기존 B, 기존 체크포인트, `final_submit/`의 파일은 건드리지 않았다.

## 실행

```bash
source /home/ubuntu/venv/bin/activate && cd ~/JEONGYUN
python -u experiments/b_cond_gate/run.py --output experiments/b_cond_gate/run_v2   # CPU 실행, 출력 폴더는 새 이름이어야 함
```
