"""T2-M1 signal model: physical-time adapters -> shared latent -> edge heads.

No external neural encoder. Independent alignment calls hide the entire other
stream and omit absolute positions. Outputs are diagnostic proposals only.
"""
from __future__ import annotations
import math
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from training.t2_train_latents import T2SLWMBlock


class TemporalAdapter(nn.Module):
    """Modality features [B,T,F] and visibility [B,T] -> [B,T,D]."""
    def __init__(self,input_dim: int,width: int):
        super().__init__()
        self.project=nn.Linear(input_dim,width)
        self.norm=nn.LayerNorm(width)
        self.local=nn.Conv1d(width,width,5,padding=2,groups=width)
        self.mix=nn.Linear(width,width)

    def forward(self,x: torch.Tensor,visible: torch.Tensor) -> torch.Tensor:
        x=x.masked_fill(~visible[...,None],0.)
        z=self.project(x).masked_fill(~visible[...,None],0.)
        z=z+self.mix(F.silu(self.local(self.norm(z).transpose(1,2)).transpose(1,2)))
        return z.masked_fill(~visible[...,None],0.)


class MotionSLWM(nn.Module):
    """Audio [B,T,128], video [B,T,160] -> shared [B,T,D] and edge outputs.

    The legacy baseline uses [B,2T,D] internally. For every mode, missing values
    are masked before either adapter; target arrays are never accepted as inputs.
    """
    def __init__(self,cfg: Mapping):
        super().__init__()
        self.cfg=dict(cfg); d=int(cfg["width"])
        self.width=d; self.packing=cfg.get("packing","aligned")
        flags={"use_local_temporal_mixer":True,"use_spectral_mixer":True,"use_long_conv":True,
               "use_attention_binding":True,"use_gated_mlp":True,"local_kernel_size":5,
               "spectral_kernel_size":15,"long_kernel_size":31,"mlp_ratio":4}
        if cfg.get("processor")=="vanilla":
            flags.update(use_local_temporal_mixer=False,use_spectral_mixer=False,use_long_conv=False)
        if cfg.get("no_spectral"): flags["use_spectral_mixer"]=False
        self.audio_adapter=TemporalAdapter(128,d)
        self.video_adapter=TemporalAdapter(160,d)
        self.presence=nn.Embedding(4,d)
        self.modality=nn.Embedding(2,d)
        self.blocks=nn.ModuleList([T2SLWMBlock(latent_dim=d,n_head=cfg["heads"],flags=flags,dropout=0.) for _ in range(cfg["layers"])])
        self.norm=nn.LayerNorm(d)
        self.audio_head=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,128))
        self.video_head=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,160))
        self.audio_alignment=nn.Linear(d,96)
        self.video_alignment=nn.Linear(d,96)
        self.logit_scale=nn.Parameter(torch.tensor(math.log(10.)))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Conv1d,nn.Embedding)):
            nn.init.normal_(module.weight,std=.02)
            if getattr(module,"bias",None) is not None: nn.init.zeros_(module.bias)

    def positions(self,t: int,device: torch.device) -> torch.Tensor:
        """Return fixed physical-relative sinusoidal positions [1,T,D]."""
        time=torch.arange(t,device=device,dtype=torch.float32)[:,None]
        freq=torch.exp(torch.arange(0,self.width,2,device=device)*(-math.log(10000.)/self.width))
        p=torch.zeros(t,self.width,device=device)
        p[:,0::2]=torch.sin(time*freq); p[:,1::2]=torch.cos(time*freq)
        return p[None]*.05

    def encode(self,audio: torch.Tensor,video: torch.Tensor,a_mask: torch.Tensor,v_mask: torch.Tensor,
               *,positions: bool=True) -> tuple[torch.Tensor,torch.Tensor]:
        """Visible signals -> two [B,T,D] readouts of one shared latent field."""
        a=self.audio_adapter(audio,a_mask); v=self.video_adapter(video,v_mask)
        if self.packing=="aligned":
            count=(a_mask.float()+v_mask.float()).clamp_min(1.).sqrt()[...,None]
            z=(a+v)/count+self.presence(a_mask.long()+2*v_mask.long())
            observed=a_mask|v_mask
            if positions: z=z+self.positions(z.shape[1],z.device)
        else:
            a=a+self.modality.weight[0]; v=v+self.modality.weight[1]
            if positions:
                p=self.positions(a.shape[1],a.device); a=a+p; v=v+p
            z=torch.cat([a,v],1); observed=torch.cat([a_mask,v_mask],1)
        if not observed.any(dim=1).all():
            # Null-input probes may query a mean prior; one fixed zero-evidence
            # position prevents an all-masked attention softmax from becoming NaN.
            observed=observed.clone(); observed[~observed.any(dim=1),0]=True
        for block in self.blocks: z=block(z,observed)
        z=self.norm(z)
        if self.packing=="aligned": return z,z
        return z[:,:audio.shape[1]],z[:,audio.shape[1]:]

    def forward(self,audio: torch.Tensor,video: torch.Tensor,a_mask: torch.Tensor,v_mask: torch.Tensor) -> dict:
        a,v=self.encode(audio,video,a_mask,v_mask)
        return {"audio":self.audio_head(a),"video":self.video_head(v),"audio_world":a,"video_world":v}

    def independent(self,audio: torch.Tensor,video: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        """Two blind encodes -> normalized alignment features [B,T,96]."""
        visible=torch.ones(audio.shape[:2],device=audio.device,dtype=torch.bool)
        absent=torch.zeros_like(visible)
        a,_=self.encode(audio,torch.zeros_like(video),visible,absent,positions=False)
        _,v=self.encode(torch.zeros_like(audio),video,absent,visible,positions=False)
        return F.normalize(self.audio_alignment(a).float(),dim=-1),F.normalize(self.video_alignment(v).float(),dim=-1)

    def independent_video(self,video: torch.Tensor) -> torch.Tensor:
        """Video [B,T,160] alone -> normalized [B,T,96], for shift negatives."""
        audio=video.new_zeros((*video.shape[:2],128))
        visible=torch.ones(video.shape[:2],device=video.device,dtype=torch.bool)
        _,v=self.encode(audio,video,torch.zeros_like(visible),visible,positions=False)
        return F.normalize(self.video_alignment(v).float(),dim=-1)

    def parameter_counts(self) -> dict:
        """Exact counts including every trainable adapter/core/head."""
        return {"adapters":sum(p.numel() for n,p in self.named_parameters() if "adapter" in n),
                "processor":sum(p.numel() for n,p in self.named_parameters() if n.startswith(("blocks","norm"))),
                "total":sum(p.numel() for p in self.parameters()),"frozen_neural_parameters":0}
