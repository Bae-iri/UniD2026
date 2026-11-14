"""V3 specialist + battery static variables (bat_class, bat_cc, soc_status, tr_method). Same adoption rule vs V3."""
import numpy as np, pandas as pd
from tune import F, MODELS, specialist, spec_proba, combine, PB, BR
from pair_experts import report
from variants import V
ST = ['bat_class', 'bat_cc', 'soc_status', 'tr_method']
SETS = {'V3': V['V3 C - all current temps'], 'V3+bat_class,bat_cc': V['V3 C - all current temps'] + ['bat_class', 'bat_cc'], 'V3+all static': V['V3 C - all current temps'] + ST}
rows = []
for name, cols in SETS.items():
    for split, frames in [('dev', [F['fit']]), ('validation', [F['fit'], F['dev']])]:
        rows.append(report(split, combine(PB[split], spec_proba(specialist(frames, cols, MODELS['mid']), F[split][cols]), .7), name, '-'))
r = pd.DataFrame(rows).drop(columns='tau'); r.to_csv('static_spec_results.csv', index=False)
for s in ['dev', 'validation']: print(r[r.split == s].drop(columns='split').to_string(index=False), '\n')
o = pd.read_csv(BR + 'submission_B.csv'); allf = [F['fit'], F['dev'], F['validation']]; v3 = pd.read_csv('submission_B_spec456_V3.csv').label.values
a = F['test'].set_index('sample_id').loc[o.sample_id]; ina = a.experiment_id.str.startswith('E_a221').values
for name, cols in list(SETS.items())[1:]:
    p = pd.Series(combine(PB['test'], spec_proba(specialist(allf, cols, MODELS['mid']), F['test'][cols]), .7), index=F['test'].sample_id).loc[o.sample_id].values.astype(int)
    fn = 'submission_B_spec456_' + name.replace('+', '_').replace(',', '_').replace(' ', '') + '.csv'; out = o.copy(); out['label'] = p; out.to_csv(fn, index=False)
    print(fn, '| vs B', int((p != o.label.values).sum()), '| vs V3', int((p != v3).sum()), '| E_a221 preds', pd.Series(p[ina]).value_counts().sort_index().to_dict(), '(V3:', pd.Series(v3[ina]).value_counts().sort_index().to_dict(), ')')
