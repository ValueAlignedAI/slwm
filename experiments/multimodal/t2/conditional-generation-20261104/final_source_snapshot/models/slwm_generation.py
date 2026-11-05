"""Small diagnostic edge decoders for a frozen shared signal field (T2-G1)."""
from __future__ import annotations
import math

import torch
from torch import nn
import torch.nn.functional as F


def pad_condition(x: torch.Tensor) -> torch.Tensor:
    """Raw/world [B,T,D<=256] -> matched decoder input [B,T,256]."""
    if x.shape[-1]>256: raise ValueError("Condition exceeds registered width")
    return F.pad(x,(0,256-x.shape[-1]))


def unpack_mel(x: torch.Tensor) -> torch.Tensor:
    """Four chronological subframes [B,T,256] -> [B,4T,64]."""
    if x.shape[-1]!=256: raise ValueError("Expected four 64-bin subframes")
    return x.reshape(x.shape[0],4*x.shape[1],64)


class TemporalResidual(nn.Module):
    """Local time processing [B,T,D] -> [B,T,D], with fixed shape."""
    def __init__(self,width: int,dilation: int):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.conv=nn.Conv1d(width,width,5,padding=2*dilation,dilation=dilation,groups=width)
        self.mix=nn.Sequential(nn.Linear(width,2*width),nn.SiLU(),nn.Linear(2*width,width))
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        z=self.norm(x); z=self.conv(z.transpose(1,2)).transpose(1,2)
        return x+self.mix(z)


class MultirateAudioHead(nn.Module):
    """World/raw [B,T,256] -> normalized 100 Hz log-mel [B,4T,64]."""
    def __init__(self):
        super().__init__()
        self.input=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,192),nn.SiLU())
        self.temporal=nn.Sequential(*(TemporalResidual(192,d) for d in (1,2,4)))
        self.output=nn.Linear(192,256)
        self.post=nn.Conv1d(64,64,5,padding=2,groups=64)
        nn.init.zeros_(self.post.weight); nn.init.zeros_(self.post.bias)
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x=unpack_mel(self.output(self.temporal(self.input(pad_condition(x)))))
        return x+self.post(x.transpose(1,2)).transpose(1,2)


class SeedMotionHead(nn.Module):
    """One image [B,32,48] + audio [B,T,256] -> grayscale [B,T,32,48].

    No image other than the seed is accepted. A zero output layer yields exact
    seed persistence. `warp=False` ablates flow while retaining a residual path.
    """
    def __init__(self,warp: bool=True):
        super().__init__(); self.warp=warp
        self.seed=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,32,4,stride=2,padding=1),nn.SiLU(),
                                nn.Conv2d(32,64,4,stride=2,padding=1),nn.SiLU())
        self.condition=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,96),nn.SiLU(),TemporalResidual(96,1),TemporalResidual(96,2),nn.Linear(96,64))
        self.time=nn.Sequential(nn.Linear(9,64),nn.SiLU(),nn.Linear(64,64))
        self.low=nn.Sequential(nn.Conv2d(64,64,3,padding=1),nn.SiLU())
        self.middle=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.SiLU())
        self.high=nn.Sequential(nn.Conv2d(32,16,3,padding=1),nn.SiLU())
        self.output=nn.Conv2d(16,3,3,padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self,condition: torch.Tensor,seed: torch.Tensor,offset: int=0) -> torch.Tensor:
        b,t=condition.shape[:2]
        steps=(torch.arange(t,device=seed.device,dtype=torch.float32)+offset)/25.
        angles=steps[:,None]*torch.tensor([1.,2.,4.,8.],device=seed.device)[None]*2*math.pi
        timing=torch.cat([steps[:,None],angles.sin(),angles.cos()],-1)
        context=self.condition(pad_condition(condition))+self.time(timing)[None]
        spatial=self.seed(seed[:,None])[:,None]+context[...,None,None]
        spatial=self.low(spatial.reshape(b*t,64,8,12))
        spatial=self.middle(F.interpolate(spatial,scale_factor=2,mode="bilinear",align_corners=False))
        spatial=self.high(F.interpolate(spatial,scale_factor=2,mode="bilinear",align_corners=False))
        output=self.output(spatial).float()
        progress=((steps*25)/4).clamp(0,1)[None].expand(b,-1).reshape(b*t,1,1,1)
        residual=.25*output[:,2].tanh()*progress[:,0]
        base=seed[:,None].expand(-1,t,-1,-1).reshape(b*t,1,32,48).float()
        if self.warp:
            y,x=torch.meshgrid(torch.linspace(-1,1,32,device=seed.device),torch.linspace(-1,1,48,device=seed.device),indexing="ij")
            grid=torch.stack([x,y],-1)[None].expand(b*t,-1,-1,-1)
            flow=output[:,:2].tanh()*progress*3.  # fixed maximum three-pixel displacement
            flow=flow.permute(0,2,3,1)*flow.new_tensor([2/47,2/31])
            # CUDA grid_sample backward is not bitwise deterministic. Two seeds
            # and actual checkpoint hashes are retained; never claim bit parity.
            with torch.autocast(device_type=seed.device.type,enabled=False):
                base=F.grid_sample(base,grid+flow,mode="bilinear",padding_mode="border",align_corners=True)
        return (base[:,0]+residual).reshape(b,t,32,48)


def visual_loss(pred: torch.Tensor,target: torch.Tensor) -> torch.Tensor:
    """Pixels [B,T,32,48] -> scalar loss; provided seed excluded."""
    p,y=pred[:,1:].float(),target[:,1:].float()
    edge=F.l1_loss(p[:,:,1:]-p[:,:,:-1],y[:,:,1:]-y[:,:,:-1])+F.l1_loss(p[:,:,:,1:]-p[:,:,:,:-1],y[:,:,:,1:]-y[:,:,:,:-1])
    return F.l1_loss(p,y)+2*F.l1_loss(p[:,:,8:24,10:38],y[:,:,8:24,10:38])+.25*edge+.5*F.l1_loss(pred[:,1:]-pred[:,:-1],target[:,1:]-target[:,:-1])


def visual_errors(pred: torch.Tensor,target: torch.Tensor) -> dict[str,torch.Tensor]:
    """Pixels [B,T,32,48] -> each metric [B], excluding the seed."""
    error=(pred[:,1:].float()-target[:,1:].float()).square()
    pd=pred[:,1:].float()-pred[:,:-1].float(); yd=target[:,1:].float()-target[:,:-1].float()
    return {"roi_mse":error[:,:,8:24,10:38].mean((1,2,3)),"pixel_mse":error.mean((1,2,3)),
            "temporal_mse":(pd-yd).square().mean((1,2,3)),"motion_energy":pd.square().mean((1,2,3)),"true_motion_energy":yd.square().mean((1,2,3))}


def audio_errors(pred: torch.Tensor,target: torch.Tensor) -> dict[str,torch.Tensor]:
    """100 Hz mel [B,4T,64] -> per-clip [B] reconstruction errors."""
    p,y=pred.float(),target.float()
    return {"mel_mse":(p-y).square().mean((1,2)),"mel_l1":(p-y).abs().mean((1,2)),
            "temporal_mse":((p[:,1:]-p[:,:-1])-(y[:,1:]-y[:,:-1])).square().mean((1,2))}
