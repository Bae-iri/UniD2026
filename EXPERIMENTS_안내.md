# 비교 실험 코드 안내

| 파일 | 비교 목적 |
|---|---|
| `compare_condition_query.py` | H1: 정적 조건을 최종 분류기에만 사용 vs attention query에도 사용 |
| `compare_image_history.py` | H2: 현재 이미지 vs 과거 이미지 평균·차이 vs 이미지 시계열 GRU |
| `compare_attention_gate.py` | B vs B+가스·이미지 attention 출력 게이트 |
| `compare_gate_branches.py` | 게이트 없음 vs 이미지 게이트만 vs 가스 게이트만 vs 둘 다. 기본은 없음 vs 이미지만 |
| `test_gate_branches.py` | 게이트 비교의 합성 데이터 검증 |

실행 인자·결과 폴더·체크포인트 형식은 기존 코드와 동일하다. 기존 결과를 새 파일명으로 추론할 수 있다. 실행 중인 학습을 위해 기존 파일은 유지했다. GitHub에는 위 이름의 코드와 이 안내, `ATTENTION_GATE_사용법.md`, `GATE_BRANCHES_사용법.md`를 올린다.

게이트 비교 실행법은 두 사용법 문서에 정리했다. 합성 검증은 `python test_gate_branches.py`로 실행하며 대회 데이터를 읽지 않는다.
