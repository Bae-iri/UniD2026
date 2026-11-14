"""Experiment 2 alone: current-temperature emphasis for observed-zero chemical gases.
No voltage-drop/event/history feature is calculated, imported or consumed here.
"""
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import abc_fusion_experiment as original

CHEMICAL=[c for c in original.GAS if c!='meas_p']


def all_observed_zero(frame):
    values=frame[CHEMICAL].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    return (np.isfinite(values).all(1)&(values==0).all(1)).astype(np.float32)


class Preprocessor(original.SensorPreprocessor):
    def transform(self,frame):
        q,g=super().transform(frame)
        return np.concatenate([q,all_observed_zero(frame)[:,None]],1).astype(np.float32),g

    def metadata(self):
        d=super().metadata();d.update(chemical_gases=CHEMICAL,zero_rule='all ten chemicals finite and exactly zero; pressure excluded',chemical_value_gain_when_zero=.5,temperature_logit_gain_when_zero=2.,temperature_columns=['meas_t1','meas_t2','thermal_t'],temporal_features=False)
        return d


class GasTemperatureB(original.FusionModel):
    def __init__(self,gas_gain=.5,temperature_gain=2.):
        super().__init__('B');self.gas_gain=gas_gain;self.temperature_gain=temperature_gain
        self.temperature=nn.Sequential(nn.Linear(6,64),nn.ReLU(),nn.Linear(64,6))
        nn.init.zeros_(self.temperature[-1].weight);nn.init.zeros_(self.temperature[-1].bias)
        self.register_buffer('chemical_mask',torch.tensor([c!='meas_p' for c in original.GAS]))

    def forward(self,image,image_mask,q_input,gas_input):
        if q_input.shape[1]!=13: raise ValueError('Experiment2: current12 + all-observed-zero flag expected')
        current=q_input[:,:12];zero=q_input[:,12:13];q=self.q_encoder(current)
        identity=self.gas_id(self.gas_indices)[None].expand(len(q),-1,-1)
        gas=self.gas_encoder(torch.cat([gas_input,identity],-1));valid=gas_input[...,1]<.5
        # Reduce only chemical-gas values; retain the pressure token and every valid zero token.
        scale=1-zero[:,:,None]*(1-self.gas_gain)*self.chemical_mask[None,:,None].to(gas.dtype)
        hg=original.common.masked_mean(gas*scale,valid)
        read=self.gas_readout
        scores=(read.key(gas)*read.query(q)[:,None]).sum(-1)/math.sqrt(gas.shape[-1])
        scores=scores.masked_fill(~valid,torch.finfo(scores.dtype).min)
        attention=scores.softmax(-1)*valid.to(scores.dtype)
        attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-8)
        z=read.output((attention[:,:,None]*read.value(gas)*scale).sum(1));rg=z+read.ffn(z)
        feature=self.cnn(image);h,w=feature.shape[-2:]
        yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=image.device),torch.linspace(-1,1,w,device=image.device),indexing='ij')
        pixels=feature.flatten(2).transpose(1,2)+self.position(torch.stack([xx,yy],-1).reshape(-1,2))[None]
        pixel_valid=F.adaptive_max_pool2d(image_mask.float(),(h,w)).flatten(1)>0
        hi=original.common.masked_mean(pixels,pixel_valid);ri=self.image_readout(pixels,q,pixel_valid,'B')
        logits=self.head(torch.cat([q,hg,hi,rg,ri],1))
        # Re-emphasize only measured battery/thermal temperatures, not voltage or gas history.
        temp=current[:,[1,2,4,7,8,10]]
        return logits+zero*self.temperature_gain*self.temperature(temp)
