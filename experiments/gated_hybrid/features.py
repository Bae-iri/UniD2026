"""Bounded causal sensor history with real irregular timestamps; no static covariates."""
import numpy as np
import pandas as pd

CURRENT=['meas_v','meas_t1','meas_t2','atmp_t','thermal_t','indu_t','meas_p','meas_ch4','meas_co','meas_co2','meas_hcn','meas_hcl','meas_hf','meas_n2o','meas_no','meas_no2','meas_so2']
HISTORY=['meas_v','meas_t1','meas_t2','thermal_t']+CURRENT[6:]


def temporal_features(frame):
    if frame[['experiment_id','relative_time_s']].isna().any().any(): raise ValueError('Missing experiment/time')
    t=pd.to_numeric(frame.relative_time_s,errors='raise').to_numpy(float)
    if not np.isfinite(t).all() or frame.duplicated(['experiment_id','relative_time_s']).any(): raise ValueError('Invalid/duplicate timestamps')
    raw=frame[CURRENT].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    raw[~np.isfinite(raw)]=np.nan
    names=CURRENT.copy(); blocks=[raw]
    for window in [30,60]:
        out=np.full((len(frame),len(HISTORY)*4+3),np.nan)
        for positions in frame.groupby('experiment_id',sort=False).indices.values():
            ids=positions[np.argsort(t[positions])]; times=t[ids]
            for i,target in enumerate(ids):
                left=np.searchsorted(times,times[i]-window)
                selected=ids[left:i+1]; tt=t[selected]
                for k,col in enumerate(HISTORY):
                    v=raw[selected,CURRENT.index(col)]; good=np.isfinite(v)
                    if good.any():
                        x=tt[good]; y=v[good]
                        out[target,4*k:4*k+2]=[y.mean(),y.std()]
                        if len(x)>1:
                            centered=x-x.mean(); denom=(centered**2).sum()
                            out[target,4*k+2]=(centered*(y-y.mean())).sum()/denom
                            out[target,4*k+3]=(y[-1]-y[0])/(x[-1]-x[0])
                out[target,-3:]=[len(selected),tt[-1]-tt[0],times[i]-times[i-1] if i>0 else np.nan]
        names += [f'{c}_{stat}_{window}s' for c in HISTORY for stat in ['mean','std','ols_slope','endpoint_rate']]+[f'history_count_{window}s',f'history_span_{window}s',f'previous_gap_{window}s']
        blocks.append(out)
    # Exact consecutive windows for acceleration; no interpolation or future values.
    for window in [30,60]:
        out=np.full((len(frame),2),np.nan)
        for ids in frame.groupby('experiment_id',sort=False).indices.values():
            lookup={t[j]:j for j in ids}
            for i in ids:
                j,k=lookup.get(t[i]-window),lookup.get(t[i]-2*window)
                if j is not None and k is not None:
                    for n,col in enumerate(['meas_t1','meas_t2']):
                        vals=raw[[i,j,k],CURRENT.index(col)]
                        if np.isfinite(vals).all(): out[i,n]=(vals[0]-2*vals[1]+vals[2])/window**2
        blocks.append(out); names += [f'{c}_acceleration_{window}s' for c in ['meas_t1','meas_t2']]
    values=np.concatenate(blocks,1)
    return pd.DataFrame(values,columns=names,index=frame.index)


class Normalizer:
    def fit(self,x):
        x=pd.DataFrame(x); self.median=x.median().fillna(0).to_numpy()
        filled=x.fillna(pd.Series(self.median)).to_numpy()
        self.mean=filled.mean(0); self.scale=filled.std(0); self.scale[self.scale<1e-6]=1
        return self

    def transform(self,x):
        x=np.asarray(x,float); mask=~np.isfinite(x)
        z=(np.where(mask,self.median,x)-self.mean)/self.scale
        return np.concatenate([np.clip(z,-20,20),mask],1).astype(np.float32)

    def state(self): return {k:getattr(self,k).tolist() for k in ['median','mean','scale']}
