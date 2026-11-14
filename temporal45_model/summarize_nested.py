"""Write a reviewable result report after training and independent verification."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import final_rules3


def main(run):
    assert json.loads((run/'status.json').read_text()).get('complete')
    assert json.loads((run/'audit_checks.json').read_text()).get('passed')
    decision = json.loads((run/'decision.json').read_text())
    decisions = json.loads((run/'fold_decisions.json').read_text())
    labels = dict(rules3='B+rules3', probability='확률만', history='확률+과거 극값·차이',
                  history_time='확률+극값·차이+시간')
    report = ['# 중첩 검증을 적용한 4→5 보정 모델 결과', '',
              '공식 Train의 실험 단위 5-fold를 바깥 평가로 사용하고, 각 바깥 학습 집합 안에서 B를 새로 3-fold×3-seed 학습했다. '
              '새 B 모델 45개와 보정기를 실제 학습했다. 바깥 평가 실험은 안쪽 B의 학습·checkpoint 선택·전처리·보정기·임계값 선택에서 모두 제외했다.', '',
              '주 후보는 실행 전에 정한 시간 없는 과거 상태 모델(history)이다. 나머지 모델은 대조군이며 결과를 보고 주 후보로 교체하지 않았다. '
              '보정은 4→5만 가능하다. 실험별 총 가중치 1과 L2 규제를 적용했고, 안쪽 검증 기준을 통과하지 못하면 해당 fold는 보정하지 않는다.', '',
              '## 바깥 OOF 평가', '',
              '| 모델 | Macro F1 | 기존 대비 | 복원/훼손 | 개선/악화 실험 | bootstrap 차이 95% 구간 |',
              '|---|---:|---:|---:|---:|---:|']
    nested = pd.read_csv(run/'nested_oof/summary.csv')
    for _, s in nested.iterrows():
        report.append(f"| {labels[s.method]} | {s.macro_f1:.6f} | {s.delta_f1:+.6f} | {int(s.corrected)}/{int(s.harmed)} | {int(s.better_experiments)}/{int(s.worse_experiments)} | [{s.ci_lower:.6f}, {s.ci_upper:.6f}] |")
    report += ['', '## 수정 허용 기준', '', '| 후보 | 보정 기준이 통과한 outer fold 수 | 최종 전체 Train 학습 기준 |', '|---|---:|---:|']
    for mode in ['probability', 'history', 'history_time']:
        count = sum(d['mode'] == mode and d['threshold'] <= 1 for d in decisions)
        artifact = joblib.load(run/f'{mode}_candidate.joblib')
        threshold = artifact['threshold']
        report.append(f"| {labels[mode]} | {count}/5 | {'보정 중단' if threshold > 1 else str(threshold)} |")
    report += ['', '## 공식 Validation', '',
               '모든 gate 학습과 임계값 선택이 끝난 뒤 평가했다. 이미 여러 차례 탐색한 자료이므로 새로운 blind 검증은 아니다.', '',
               '| 기준 모델·범위 | B+rules3 | 주 후보 적용 | 복원/훼손 |', '|---|---:|---:|---:|']
    for protocol, title in [('validation_fold_ensemble', '새 fold B 전체'),
                            ('validation_fold_ensemble_noE16', '새 fold B E16 제외'),
                            ('validation_original_B', '기존 최고 제출 B 전체'),
                            ('validation_original_B_noE16', '기존 최고 제출 B E16 제외')]:
        s = pd.read_csv(run/protocol/'summary.csv').set_index('method')
        c = s.loc['history']
        report.append(f"| {title} | {s.loc['rules3', 'macro_f1']:.6f} | {c.macro_f1:.6f} | {int(c.corrected)}/{int(c.harmed)} |")
    valpred = pd.read_csv(run/'validation_original_B/predictions.csv')
    e16 = valpred[(valpred.experiment_id == final_rules3.E16) & (valpred.label == 5)]
    report += ['', f"기존 B 기준 E16 실제 5단계 정답: {int(e16.rules3.eq(5).sum())}→{int(e16.history.eq(5).sum())}건. "
               f"5→4 오류: {int(e16.rules3.eq(4).sum())}→{int(e16.history.eq(4).sum())}건."]
    report += ['', '## 채택 판단', '',
               ('주 후보가 사전에 고정한 채택 기준을 통과했다. 공개/private 성능 개선을 보장하는 것은 아니다.' if decision['accepted'] else
                '주 후보는 사전에 고정한 채택 기준을 통과하지 못했다. 배포용 모델은 보정을 끄고 기존 예측을 유지하도록 저장했다. '
                '검증 결과를 보고 기준을 완화하거나 대조군으로 바꾸지 않았다.'), '',
               '| 조건 | 통과 |', '|---|---:|']
    for key, value in decision['checks'].items():
        report.append(f"| {key} | {'예' if value else '아니오'} |")
    report += ['', '## 해석 및 한계', '',
               '규제를 둔 작은 선형 gate가 보정할 수 있다는 가설을 검증한 결과다. 보정 기준이 꺼진 경우는 안전하게 고칠 근거를 확보하지 못했다는 뜻이며, '
               '모든 형태의 이력 모델이 불가능하다는 증거는 아니다. Inner B는 outer B보다 적은 실험으로 학습되어 확률 분포 차이가 있을 수 있다. '
               '이전 탐색의 선택 편향과 새로운 재학습 변동은 이번 고정 OOF bootstrap으로 제거되지 않는다. 과적합 부재를 증명했다고 주장하지 않는다.', '',
               'OOF와 Validation의 절대 F1을 공개 제출 점수 0.93과 대응시키지 않는다. 이 중첩 학습은 Test를 읽거나 제출 파일을 만들지 않았다. '
               '별도 제출 감사에서는 기존 Test 관측으로 제출 파일을 재현했으며 정답·학습·모델 선택에는 사용하지 않았다.', '',
               '## 산출물·검증', '',
               '`nested_v1/protocol.json`: 실행 전 조건과 소스 hash. `nested_v1/outer_*/inner_*`: 실험 분할, 전처리, '
               'checkpoint 45개와 학습 곡선, holdout 확률. `nested_v1/nested_oof/`: 평가 예측, 전체·실험별·조건별 지표와 수정 내역. '
               '`nested_v1/*_candidate.joblib`: 연구 후보. `nested_v1/deployment_gate.joblib`: 채택 판단을 반영한 모델. '
               '`nested_v1/decision.json`: 사전 기준별 통과 여부.', '',
               '미래 불변성·실험 분리·시간 없는 특징의 시간 배율 불변성·4→5 외 변경 금지·근거 부족 시 보정 중단·중복 시각 검사·초기 전압 규칙을 포함한 전체 prefix 불변성을 포함한 '
               '단위검증 7개를 통과했다. 독립 감사는 45개 모델의 fit/selection/holdout 분리, 전처리와 checkpoint 선택, '
               'OOF 배열, 보정 결정, 전체 F1을 재구성했다. 결과는 `nested_v1/audit_checks.json`에 있다.', '',
               '설계와 재현 명령은 [NESTED_PROTOCOL.md](NESTED_PROTOCOL.md)에 있다.']
    report.insert(4, '원본 final_rules3.py에서 미래 참조를 제거하고 관측된 prefix만 사용하는 생산용 규칙을 모든 경로에서 공유한다. '
                  '현재 프레임만 사용하는 B checkpoint 30개는 중단한 이전 실행에서 재사용했고, 남은 15개를 추가 학습했다. 보정기와 기준은 모두 다시 학습했다. 실제 실험 prefix 불변성 감사도 수행했다.')
    destination = run.parent/'REPORT_NESTED.md'
    destination.write_text(('\n'.join(report)+'\n').replace('nested_v1/', run.name+'/'))
    print(destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    main(p.parse_args().run)
