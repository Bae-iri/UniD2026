"""Build Jeongyun 4->3 + stage6 correction with mandatory full validation.

No leaderboard input, parameter search, experiment exclusion, or E16 exception.
Archived checkpoint weights and the frozen rule are preserved exactly.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pipeline import predict_stages
from reproduce import load_artifact, load_probs, invariants, sha
from validate import summary, component_results

E16 = 'E_e5bcddc7d77f459ea541cf658ab4346e'


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    artifact = load_artifact(args.upper_artifact, args.lower_artifact)
    assert artifact['heads']['thresholds']['56'] == .7
    assert artifact['gate']['threshold'] == .7
    assert artifact['low']['thresholds']['43'] == .5
    joblib.dump(artifact, args.output/'combined.joblib')
    manifest = dict(recipe=['B seed42/43/44 mean', 'T1 causal45', '5->6 head0.7',
                           '6->5 gate0.7', 'Jeongyun4->3 head0.5', 'S3 H300 d5 with observed voltage drop'],
                    leaderboard_score_input=False, parameter_search=False,
                    evaluation_exclusions=[], inference_experiment_specific_rules=False,
                    source_artifacts={str(p):sha(p) for p in [args.upper_artifact,args.lower_artifact]},
                    combined_artifact_sha256=sha(args.output/'combined.joblib'))
    for split in ['validation','test']:
        frame = pd.read_csv(args.data/split/(split+'.csv'))
        probs = load_probs(args.base_predictions, split, frame.sample_id)
        stages = predict_stages(artifact, frame, probs)
        checks = invariants(frame, probs, artifact, stages['S3'])
        table = frame[['sample_id','experiment_id']+(['label'] if split == 'validation' else [])].copy()
        for name, values in stages.items():
            table[name] = values
        table.to_csv(args.output/(split+'_stages.csv'), index=False)
        changed43 = stages['precision65'] != stages['J096']
        assert np.array_equal(stages['S3'][changed43], stages['J096'][changed43])
        manifest[split] = dict(rows=len(frame), experiments=frame.experiment_id.nunique(),
            prefix_checks=checks, lower43_changes=int(changed43.sum()),
            lower43_preserved=int((stages['S3'][changed43] == 3).sum()))
        if split == 'validation':
            assert frame.experiment_id.nunique() == 19 and E16 in set(frame.experiment_id)
            original = pd.read_csv(args.data/'validation/validation.csv')
            np.testing.assert_array_equal(frame.sample_id, original.sample_id)
            manifest['validation'].update(E16_included=True, E16_rows=int((frame.experiment_id == E16).sum()),
                overall=summary(table), components=component_results(table),
                E16_components=component_results(table[table.experiment_id == E16]))
        else:
            sample = pd.read_csv(args.data/'sample_submission.csv')
            assert sample.sample_id.is_unique and set(sample.sample_id) == set(frame.sample_id)
            submission = pd.DataFrame({'sample_id':frame.sample_id,'label':stages['S3']}).set_index('sample_id').loc[sample.sample_id].reset_index()
            assert submission.label.isin(range(1,7)).all()
            reference = pd.read_csv(args.reference).set_index('sample_id')
            assert set(reference.index) == set(submission.sample_id) and reference.index.is_unique
            mismatch = int((submission.label.to_numpy() != reference.loc[submission.sample_id].label.to_numpy()).sum())
            assert mismatch == 0, 'Combined code changed the requested archived predictions'
            target = args.output/'submission_stage6.csv'
            submission.to_csv(target,index=False)
            manifest['test'].update(reference_label_mismatches=mismatch,
                reference_sha256=sha(args.reference),submission_sha256=sha(target))
        print(split,manifest[split]['rows'],'rows; prefixes PASS; lower4->3 preserved',manifest[split]['lower43_preserved'],flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print('Combined recipe verified; full validation F1',manifest['validation']['overall']['macro_f1'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data','base-predictions','upper-artifact','lower-artifact','reference','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    with threadpool_limits(limits=2):
        main(parser.parse_args())
