"""Compare fixed stage-dependent mixtures on saved dev/validation only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BASE=Path('/home/ubuntu/leewon/abc_runs/current_query_v1')
OUT=Path(__file__).resolve().parent/'stage_weights'

def conditional_mix(b, other, weight):
    out=b.copy()
    mass=b[:,3:].sum(1,keepdims=True)
    q=b[:,3:]/np.maximum(mass,1e-12)
    r=other[:,3:]/np.maximum(other[:,3:].sum(1,keepdims=True),1e-12)
    out[:,3:]=mass*((1-weight)*q+weight*r)
    assert np.allclose(out.sum(1),1,atol=1e-5)
    assert np.array_equal(out[:,:3],b[:,:3])
    return out

def main():
    OUT.mkdir(exist_ok=False)
    rows=[]
    for split in ['dev','validation']:
        fs={m:pd.read_csv(BASE/f'ensemble_{m}_{split}_predictions.csv') for m in 'ABC'}
        f=fs['B']; y=f.label.to_numpy(); cols=[f'p{i}' for i in range(1,7)]
        for v in fs.values():
            assert np.array_equal(v.sample_id,f.sample_id) and np.array_equal(v.label,y)
        a,b,c=[fs[m][cols].to_numpy() for m in 'ABC']
        late=(a+c)/2
        bp=b.argmax(1)+1
        variants={'B':bp,'B_late_C50':conditional_mix(b,c,.5).argmax(1)+1,
                  'B_late_AC50':conditional_mix(b,late,.5).argmax(1)+1,
                  'B_late_C100':conditional_mix(b,c,1).argmax(1)+1}
        # Strict router: preserves every B prediction of 1,2,3 and the early/late boundary.
        for name,other in [('C',c),('AC',late)]:
            mix=conditional_mix(b,other,.5)
            p=bp.copy(); mask=bp>=4
            p[mask]=mix[mask,3:].argmax(1)+4
            assert np.array_equal(p[bp<=3],bp[bp<=3])
            variants[f'B_router_{name}50']=p
        for name,p in variants.items():
            keep=f.experiment_id.to_numpy()!='E_e5bcddc7d77f459ea541cf658ab4346e'
            rows.append(dict(split=split,method=name,f1=f1_score(y,p,labels=range(1,7),average='macro'),
               noE16_f1=f1_score(y[keep],p[keep],labels=range(1,7),average='macro'),
               corrected=int(((bp!=y)&(p==y)).sum()),harmed=int(((bp==y)&(p!=y)).sum()),
               early_B_predictions_changed=int(((bp<=3)&(p!=bp)).sum()),
               **{f'F1_{i+1}':v for i,v in enumerate(f1_score(y,p,labels=range(1,7),average=None))}))
    df=pd.DataFrame(rows);df.to_csv(OUT/'scores.csv',index=False)
    (OUT/'scope.json').write_text(json.dumps(dict(test_access=False,training=False,weight_search=False,
      note='Fixed hypotheses on repeatedly inspected dev/validation; exploratory, not fresh validation.'),indent=2))
    print(df.round(6).to_string(index=False))

if __name__=='__main__':main()
