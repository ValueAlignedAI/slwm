"""Keep observed audio detail linearly available in the shared world field."""
from __future__ import annotations
import torch
from torch import nn
from models.slwm_restoration import RestorationSLWM


class ContentRestorationSLWM(RestorationSLWM):
    """Same primary [B,T,F] -> [B,4T,64], plus training-only content readout."""
    def __init__(self,cfg: dict,run: dict):
        super().__init__(cfg,run)
        self.content_head=nn.Linear(int(cfg["model"]["width"]),64); nn.init.zeros_(self.content_head.weight); nn.init.zeros_(self.content_head.bias)
        self.content_prediction=None

    def forward(self,audio,video,visible,video_visible,base):
        z=self.world.encode(audio,video.masked_fill(~video_visible[...,None],0.),visible,video_visible)[0]
        self.content_prediction=self.content_head(z).float() if self.training else None
        correction=self.head(z).float(); bound=torch.where(visible,.25,3.).repeat_interleave(4,1)[...,None]
        return base+bound*correction.tanh()


def content_loss(prediction: torch.Tensor,target: torch.Tensor,visible: torch.Tensor) -> torch.Tensor:
    """Field-only [B,T,64] -> observed coarse-sound MSE; hidden values excluded."""
    coarse=target.reshape(target.shape[0],visible.shape[1],4,64)[:,:,0].float()
    error=(prediction.float()-coarse).square().mean(-1)
    return (error*visible).sum()/visible.sum().clamp_min(1)
