"""Audio-anchored visual corrections with exact no-video fallback."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from models.slwm_content import ContentRestorationSLWM
from models.slwm_generation import MultirateAudioHead,TemporalResidual


def gap_masks(start: torch.Tensor,length: torch.Tensor,packets: int=32) -> dict:
    """True waveform gaps -> honest coarse inputs, true/unknown fine masks.

    16kHz, FFT1024, centers640 samples (coarse) /160 (fine). Strict support
    intersection removes even a visible center whose window sees hidden sound.
    """
    device=start.device; stop=start+length; coarse=torch.arange(packets,device=device)[None]*640
    begin=start[:,None]*640; end=stop[:,None]*640
    visible=~((coarse-512<end)&(coarse+512>begin))
    fine=torch.arange(4*packets,device=device)[None]*160
    unknown=(fine-512<end)&(fine+512>begin)
    target=(fine>=begin)&(fine<end)
    return {"visible":visible,"unknown":unknown,"target":target}


class RepairBranch(nn.Module):
    """Audio field/prediction [B,T,256], video -> residual [B,4T,64], gate[B,T]."""
    def __init__(self,arm: str,seed: int):
        super().__init__(); self.arm=arm; self.is_audio=arm=="audio"
        if self.is_audio:
            # The audio arm does not train the visual alignment/gate heads.
            # Extra frontend width approximately matches graph-connected
            # parameters, rather than padding the count with unused weights.
            self.front=nn.Sequential(nn.Linear(256,1088),nn.SiLU(),nn.Linear(1088,128))
        elif arm=="pca_video": self.front=nn.Linear(160,128)
        else:
            layers=[]; channels=(2,16,32,64,96)
            for i in range(4):
                layers += [nn.Conv2d(channels[i],channels[i+1],5 if i==0 else 3,stride=2,padding=2 if i==0 else 1),nn.GroupNorm(4,channels[i+1]),nn.SiLU()]
            self.front=nn.Sequential(*layers,nn.Flatten(),nn.Linear(96*4*6,128))
        self.temporal=nn.Sequential(TemporalResidual(128,1),TemporalResidual(128,2))
        self.context=nn.Sequential(nn.Linear(512,256),nn.SiLU())
        self.query=nn.Linear(256,128); self.attend=nn.MultiheadAttention(128,4,batch_first=True,dropout=0.)
        self.fuse=nn.Linear(384,256); self.visual_align=nn.Linear(128,96)
        self.gate_net=nn.Sequential(nn.Linear(385,64),nn.SiLU(),nn.Linear(64,1)); nn.init.constant_(self.gate_net[-1].bias,-1.)
        with torch.random.fork_rng():
            torch.manual_seed(seed+10000)
            for top in (self.temporal,self.context,self.query,self.attend,self.fuse,self.visual_align,self.gate_net):
                for module in top.modules():
                    if isinstance(module,nn.MultiheadAttention): module._reset_parameters()
                    elif hasattr(module,"reset_parameters"): module.reset_parameters()
        nn.init.constant_(self.gate_net[-1].bias,-1.)
        with torch.random.fork_rng():
            torch.manual_seed(seed+11000); self.head=MultirateAudioHead()
        nn.init.zeros_(self.head.output.weight); nn.init.zeros_(self.head.output.bias)

    def encode_hint(self,field: torch.Tensor,mouth: torch.Tensor,video: torch.Tensor) -> torch.Tensor:
        """Only declared modality -> hint [B,T,128]; pixels never reach audio arm."""
        b,t=field.shape[:2]
        if self.is_audio: hint=self.front(field)
        elif self.arm=="pca_video": hint=self.front(video)
        else:
            pixels=mouth.float()/255.; mean=pixels.mean((1,2,3),keepdim=True); scale=pixels.std((1,2,3),keepdim=True).clamp_min(.03)
            pixels=(pixels-mean)/scale; delta=pixels-torch.cat([pixels[:,:1],pixels[:,:-1]],1)
            hint=self.front(torch.stack([pixels,delta],2).reshape(b*t,2,64,96)).reshape(b,t,128)
        return self.temporal(hint)

    def forward(self,field,prediction,mouth,video,visible,unknown,alignment,present):
        b,t=field.shape[:2]; context=self.context(torch.cat([field,prediction.reshape(b,t,256)],-1))
        hint=self.encode_hint(field,mouth,video); projected=F.normalize(self.visual_align(hint).float(),dim=-1)
        similarity=(projected*alignment).sum(-1)
        confidence=(similarity*visible).sum(1)/visible.sum(1).clamp_min(1)
        query=self.query(context); index=torch.arange(t,device=field.device); mask=(index[:,None]-index[None,:]).abs()>2
        bound,_=self.attend(query,hint,hint,attn_mask=mask,need_weights=False)
        mixed=self.fuse(torch.cat([context,bound],-1)); raw=self.head(mixed).float().tanh()
        logits=self.gate_net(torch.cat([context,hint,confidence[:,None,None].expand(-1,t,1)],-1))[...,0].float()
        if self.is_audio or self.arm=="ungated_video": gate=torch.ones_like(logits)
        else: gate=logits.sigmoid()
        availability=torch.ones_like(present) if self.is_audio else present
        residual=raw*gate.repeat_interleave(4,1)[...,None]*unknown[...,None]*availability[:,None,None]
        return {"prediction":prediction+residual,"residual":residual,"gate_logits":logits,"gate":gate,"projected_hint":projected,"similarity":similarity}


class VideoRepairSystem(nn.Module):
    """Frozen content expert -> audio anchor -> optional learned gap correction."""
    def __init__(self,expert_cfg: dict,expert_run: dict,arm: str,seed: int,anchor_only: bool=False):
        super().__init__(); self.anchor_only=anchor_only
        self.expert=ContentRestorationSLWM(expert_cfg,expert_run).requires_grad_(False)
        self.anchor=RepairBranch("audio",seed)
        if not anchor_only:
            self.anchor.requires_grad_(False); self.branch=RepairBranch(arm,seed)

    def train(self,mode=True):
        super().train(mode); self.expert.eval()
        if not self.anchor_only: self.anchor.eval()
        return self

    def frozen_audio(self,observation):
        """Visible signals only -> independent audio field and initial [B,4T,64]."""
        audio=observation["audio"]; visible=observation["visible"]; zero=torch.zeros_like(visible); video=audio.new_zeros((*audio.shape[:2],160))
        with torch.no_grad():
            field=self.expert.world.encode(audio,video,visible,zero)[0]
            pred=observation["base"]+torch.where(visible,.25,3.).repeat_interleave(4,1)[...,None]*self.expert.head(field).float().tanh()
            align=F.normalize(self.expert.world.audio_alignment(field).float(),dim=-1)
        return field,pred,align

    def forward(self,observation,mouth,video,unknown,present):
        field,base,alignment=self.frozen_audio(observation)
        if self.anchor_only: return self.anchor(field,base,mouth,video,observation["visible"],unknown,alignment,present)
        with torch.no_grad(): anchor=self.anchor(field,base,mouth,video,observation["visible"],unknown,alignment,present)["prediction"]
        output=self.branch(field,anchor,mouth,video,observation["visible"],unknown,alignment,present)
        output["anchor"]=anchor; output["alignment_target"]=alignment
        output["field"]=field
        return output
