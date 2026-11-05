"""Observation-preserving shared core for masked multimodal signal restoration."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from models.slwm_motion import MotionSLWM
from models.slwm_generation import MultirateAudioHead,unpack_mel,pad_condition


def fill_observed(values: torch.Tensor,visible: torch.Tensor) -> torch.Tensor:
    """Observed values [B,T,F], mask [B,T] -> linear fill [B,T,F].

    Hidden values are zeroed before indexing. Values outside visible anchors use
    the closest anchor; completely absent rows return zero. No hidden delta leaks.
    """
    x=values.masked_fill(~visible[...,None],0.); b,t=visible.shape
    index=torch.arange(t,device=x.device)[None].expand(b,-1)
    left=torch.where(visible,index,-1).cummax(1).values
    right=torch.where(visible,index,t).flip(1).cummin(1).values.flip(1)
    left=torch.where(left<0,right,left).clamp(0,t-1); right=torch.where(right>=t,left,right).clamp(0,t-1)
    lo=x.gather(1,left[...,None].expand(-1,-1,x.shape[-1])); hi=x.gather(1,right[...,None].expand(-1,-1,x.shape[-1]))
    fraction=((index-left).float()/(right-left).clamp_min(1)).clamp(0,1)[...,None]
    return (lo*(1-fraction)+hi*fraction)*visible.any(1)[:,None,None]


class RestorationCodec(nn.Module):
    """Fixed CREMA-training statistics; [B,T,256] -> honest inputs/base/target."""
    def __init__(self,old: dict,new: dict):
        super().__init__()
        for name,key,source in (("mean100","mean",new),("scale100","scale",new),("mean25","audio_mean",old),("scale25","audio_scale",old),
                                 ("delta_mean","audio_delta_mean",old),("delta_scale","audio_delta_scale",old)):
            self.register_buffer(name,torch.as_tensor(source[key],dtype=torch.float32))

    def augment(self,y: torch.Tensor,gain_db: torch.Tensor,eq_db: torch.Tensor) -> torch.Tensor:
        """Targets [B,T,256] + per-recording gain/EQ -> same-rate [B,T,256]."""
        eq=F.interpolate(eq_db[:,None].float(),size=64,mode="linear",align_corners=True)[:,0]
        shift=(gain_db[:,None]+eq)*(.1*2.302585092994046)/self.scale100
        return (y.reshape(y.shape[0],y.shape[1],4,64)+shift[:,None,None]).reshape_as(y)

    def observe(self,y: torch.Tensor,visible: torch.Tensor) -> dict:
        """Target [B,T,256], declared visibility -> inputs without hidden evidence."""
        physical=y.reshape(y.shape[0],y.shape[1],4,64)[:,:,0].float()*self.scale100+self.mean100
        filled=fill_observed(physical,visible)
        # Derive differences only from observed/interpolated values, never clean
        # hidden frames. Even the visible frame after a hole must not leak its
        # hidden predecessor through a precomputed delta channel.
        delta=filled-torch.cat([filled[:,:1],filled[:,:-1]],1)
        audio=torch.cat([(filled-self.mean25)/self.scale25,(delta-self.delta_mean)/self.delta_scale],-1)
        normalized=(filled-self.mean100)/self.scale100
        t=y.shape[1]; clock=torch.arange(4*t,device=y.device)/4; left=clock.long(); right=(left+1).clamp_max(t-1); f=(clock-left)[None,:,None]
        base=normalized[:,left]*(1-f)+normalized[:,right]*f
        return {"audio":audio,"base":base,"target":unpack_mel(y),"visible":visible}


class RestorationSLWM(nn.Module):
    """Partial audio/video [B,T,128/160], masks -> repaired [B,4T,64].

    Only the externally supplied observation dictionary enters this model. No
    clean target is accepted. An explicit raw path preserves the observed signal.
    """
    def __init__(self,cfg: dict,run: dict):
        super().__init__(); self.arm=run["arm"]; self.has_core=self.arm not in {"raw_audio","raw_av"}
        if self.has_core:
            worldcfg={**cfg["model"],"processor":"vanilla" if self.arm=="vanilla" else "signal"}
            self.world=MotionSLWM(worldcfg)
            del self.world.audio_head,self.world.video_head
            if self.arm=="frozen_core": self.world.requires_grad_(False)
        elif self.arm=="raw_av": self.raw_projection=nn.Linear(290,256)
        # Match the decoder initialization across cores with different module
        # counts; core construction must not shift its random-number sequence.
        with torch.random.fork_rng():
            torch.manual_seed(int(run.get("seed",0))+9000)
            self.head=MultirateAudioHead()
        if self.arm!="no_skip": nn.init.zeros_(self.head.output.weight); nn.init.zeros_(self.head.output.bias)

    def initialize_world(self,state: dict):
        if self.has_core:
            kept={k:v for k,v in state.items() if not k.startswith(("audio_head.","video_head."))}
            self.world.load_state_dict(kept,strict=True)

    def forward(self,audio: torch.Tensor,video: torch.Tensor,visible: torch.Tensor,video_visible: torch.Tensor,base: torch.Tensor) -> torch.Tensor:
        video=video.masked_fill(~video_visible[...,None],0.)
        if self.has_core:
            z=self.world.encode(audio,video,visible,video_visible)[0]
        elif self.arm=="raw_av":
            z=self.raw_projection(torch.cat([audio,video,visible[...,None].float(),video_visible[...,None].float()],-1))
        else: z=pad_condition(torch.cat([audio,visible[...,None].float()],-1))
        output=self.head(z).float()
        if self.arm=="no_skip": return output
        bound=torch.where(visible,.25,3.).repeat_interleave(4,1)[...,None]
        return base+bound*output.tanh()


def restoration_loss(pred: torch.Tensor,target: torch.Tensor,visible: torch.Tensor) -> torch.Tensor:
    """Predicted/target [B,4T,64] -> weighted continuous reconstruction loss."""
    missing=(~visible).repeat_interleave(4,1); weight=(1+3*missing.float())[...,None]
    p,y=pred.float(),target.float(); loss=((p-y).abs()*weight).sum()/(weight.sum()*64)
    return loss+.5*F.l1_loss(p[:,1:]-p[:,:-1],y[:,1:]-y[:,:-1])


def restoration_errors(pred: torch.Tensor,target: torch.Tensor,visible: torch.Tensor) -> dict[str,torch.Tensor]:
    """Signals [B,4T,64], mask [B,T] -> per-recording errors [B]."""
    error=(pred.float()-target.float()).square().mean(-1); missing=(~visible).repeat_interleave(4,1)
    hole=(error*missing).sum(1)/missing.sum(1).clamp_min(1)
    return {"mel_mse":error.mean(1),"hole_mse":hole,"observed_mse":(error*(~missing)).sum(1)/(~missing).sum(1).clamp_min(1),
        "temporal_mse":((pred[:,1:].float()-pred[:,:-1].float())-(target[:,1:].float()-target[:,:-1].float())).square().mean((1,2))}
