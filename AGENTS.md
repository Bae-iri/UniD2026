# 대회 작업의 필수 조건

- 미래 데이터 참조는 금지다. 시각 t의 추론은 같은 실험에서 시각 t까지 실제로 관측한 데이터만 사용한다.
- 추론 특징·규칙·정규화·상태 갱신에 미래 행, 뒤 채움, 음수 shift, 중심 이동창, 양방향 시계열 모델, 실험 종료시각·전체 길이·전체 구간 통계를 사용하지 않는다.
- 초기 기준도 관측 도착 시점까지의 값으로만 갱신한다. 처음 N개 관측의 통계를 과거 행 전체에 소급 적용하지 않는다.
- 정답 라벨과 실제 단계 전환시각은 학습 목표와 사후 평가에만 쓰며 추론 입력에 넣지 않는다.
- Scaler·결측값 대체·모델·보정 기준은 학습 실험에서만 적합한다. 평가 실험은 실험 단위로 분리한다.
- 시계열 추론을 변경하면 prefix 불변성을 검증한다. 실험을 t에서 잘라 계산한 t까지의 예측이 전체 자료로 계산한 같은 예측과 일치해야 한다.
- 기존 코드에서 위반을 발견하면 예측 영향 유무와 관계없이 실행 경로에서 제거한다. 위반 코드를 비교·복원용으로 다시 실행하는 경로도 남기지 않는다.
- 제출 파일은 현재·과거만 사용하는 코드로 생성한다. 미래 참조를 허용하는 옵션이나 우회 경로를 만들지 않는다.

# 봉인 목록 (2026-11-14)

아래는 미래 관측 사용 이력이 있거나 그 결과에서 파생된 코드·산출물이다. 저장소에서 삭제했다. git 이력·다른 폴더·다른 사람 사본에서 **복원·재실행·import·비교 기준으로 사용하지 않는다.** 같은 기능이 필요하면 현재·과거만 쓰는 코드로 새로 작성한다.

- rules3 계열: `final_rules3.py`, `RULES3_사용법.md`, `temporal45_model/`, `four_rule_cv/`, `targeted_stage_model/`, `rules3_stage23/`, `stage56_audit/`, `error_audit_full/`. 옛 구현이 초기 5개 전압 관측의 최대값을 그 이전 행에도 소급 적용했다(rules5는 실험 전체의 첫 전압 하락 시각도 앞 행에 적용).
- `stage23_audit/`: `transform('first')`로 실험 전체의 첫 유효값을 그 이전 행에 사용했다.
- 제출: `submission_rules3/4/6.csv`, `submission_rules_robust.csv`, `submission_rule1_vdrop80/90.csv`, `submission_targeted56_past.csv`, `submission_targeted45_56_past.csv`.
- 외부 사본 `/home/ubuntu/gahyun/experiments/b_rules/jeongyun_rules/`의 `final_rules3.py`·`final_rules5.py`와 그 출력도 사용 금지다.
