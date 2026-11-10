"""Observation-preserving residual audio edge, independent of frozen G1 code."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from models.slwm_generation import MultirateAudioHead


class SignalResidualAudioHead(nn.Module):
    """Observed 25 Hz [B,T,128] + condition [B,T,D] -> 100 Hz [B,4T,64].

    The fixed path preserves absolute signal scale even outside training
    statistics. A bounded learned residual cannot replace the observation.
    """
    def __init__(self,old: dict,new: dict):
        super().__init__(); self.correction=MultirateAudioHead()
        self.register_buffer("scale",torch.from_numpy((old["audio_scale"]/new["scale"]).astype(np.float32)))
        self.register_buffer("bias",torch.from_numpy(((old["audio_mean"]-new["mean"])/new["scale"]).astype(np.float32)))
        nn.init.zeros_(self.correction.output.weight); nn.init.zeros_(self.correction.output.bias)

    def baseline(self,observed: torch.Tensor) -> torch.Tensor:
        """Physical center-preserving interpolation [B,T,128] -> [B,4T,64]."""
        x=observed[...,:64].float()*self.scale+self.bias
        time=torch.arange(4*x.shape[1],device=x.device)/4; left=time.long(); right=(left+1).clamp_max(x.shape[1]-1)
        f=(time-left)[None,:,None]
        return x[:,left]*(1-f)+x[:,right]*f

    def forward(self,condition: torch.Tensor,observed: torch.Tensor) -> torch.Tensor:
        return self.baseline(observed)+.25*self.correction(condition).float().tanh()
