"""Experiment 1 alone: original B plus a dedicated voltage-event logit branch."""
import numpy as np
import torch
from torch import nn
import abc_fusion_experiment as original
from .event_memory import fit_detector,event_memory,NAMES


class Preprocessor(original.SensorPreprocessor):
    def __init__(self,fit):
        super().__init__(fit); self.detector=fit_detector(fit)
        e=event_memory(fit,self.detector).astype(float); e[:,2]=np.log1p(e[:,2])
        self.event_mean=e[:,:4].mean(0); self.event_scale=e[:,:4].std(0); self.event_scale[self.event_scale<1e-6]=1

    def transform(self,frame):
        q,g=super().transform(frame);e=event_memory(frame,self.detector)
        e[:,2]=np.log1p(e[:,2]);e[:,:4]=np.clip((e[:,:4]-self.event_mean)/self.event_scale,-20,20)
        return np.concatenate([q,e],1).astype(np.float32),g

    def metadata(self):
        d=super().metadata();d.update(event_names=NAMES,detector=self.detector,event_mean=self.event_mean.tolist(),event_scale=self.event_scale.tolist(),event_gain=2.,event_input='dedicated final-logit branch; no Query modification')
        return d


class VoltageBoostB(original.FusionModel):
    def __init__(self,event_gain=2.):
        super().__init__('B');self.event_gain=event_gain
        self.voltage_event=nn.Sequential(nn.Linear(7,64),nn.ReLU(),nn.Linear(64,6))
        nn.init.zeros_(self.voltage_event[-1].weight);nn.init.zeros_(self.voltage_event[-1].bias)

    def forward(self,image,image_mask,q_input,gas_input):
        if q_input.shape[1]!=19: raise ValueError('Experiment1: current12 + voltage-event7 expected')
        return super().forward(image,image_mask,q_input[:,:12],gas_input)+self.event_gain*self.voltage_event(q_input[:,12:])
