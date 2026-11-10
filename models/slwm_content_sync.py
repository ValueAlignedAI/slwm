"""Speech-content latents, timed-text conditioning and observed-only AV rejection."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from models.slwm_content import ContentRestorationSLWM
from models.slwm_generation import MultirateAudioHead,TemporalResidual


class SpeechContentBase(nn.Module):
    """Observed audio[B,T,128] -> field[B,T,256], word logits[B,T,K], mel[B,4T,64]."""
    def __init__(self,expert_cfg,expert_run,words,seed):
        super().__init__(); self.expert=ContentRestorationSLWM(expert_cfg,expert_run); self.words=words
        with torch.random.fork_rng():
            torch.manual_seed(seed+21000); self.unit=nn.Linear(256,words); self.text_embedding=nn.Embedding(words,64); self.text_mix=nn.Linear(64,256)
        nn.init.zeros_(self.text_mix.weight); nn.init.zeros_(self.text_mix.bias)
    def encode(self,obs):
        audio=obs["audio"]; visible=obs["visible"]
        return self.expert.world.encode(audio,audio.new_zeros((*audio.shape[:2],160)),visible,torch.zeros_like(visible))[0]
    def decode(self,obs,field,logits,text_ids=None):
        probabilities=logits.float().softmax(-1)
        if text_ids is not None:
            provided=text_ids>=0; supplied=F.one_hot(text_ids.clamp_min(0).long(),self.words).float()
            probabilities=torch.where(provided[...,None],supplied,probabilities)
        content=probabilities@self.text_embedding.weight.float(); mixed=field+self.text_mix(content.to(field.dtype))
        bound=torch.where(obs["visible"],.25,3.).repeat_interleave(4,1)[...,None]
        return obs["base"]+bound*self.expert.head(mixed).float().tanh()
    def forward(self,obs,text_ids=None):
        field=self.encode(obs); logits=self.unit(field).float()
        return {"prediction":self.decode(obs,field,logits,text_ids),"field":field,"unit_logits":logits,"content_readout":self.expert.content_head(field).float()}


class SpatialVideo(nn.Module):
    """Recorded uint8[B,T,64,96] -> translation-tolerant video hints[B,T,128]."""
    def __init__(self):
        super().__init__(); channels=(2,16,32,64,96); layers=[]
        for i in range(4): layers.extend((nn.Conv2d(channels[i],channels[i+1],5 if i==0 else 3,2,2 if i==0 else 1),nn.GroupNorm(4,channels[i+1]),nn.SiLU()))
        self.net=nn.Sequential(*layers,nn.AdaptiveAvgPool2d((2,3)),nn.Flatten(),nn.Linear(576,128))
    def forward(self,mouth):
        b,t=mouth.shape[:2]; pixels=mouth.float()/255.; pixels=(pixels-pixels.mean((1,2,3),keepdim=True))/pixels.std((1,2,3),keepdim=True).clamp_min(.03)
        if self.training:
            # One shift per complete clip; temporal differences remain physical.
            padded=F.pad(pixels,(4,4,4,4)); offsets=torch.randint(0,9,(b,2)).tolist()
            pixels=torch.stack([padded[i,:,dy:dy+64,dx:dx+96] for i,(dy,dx) in enumerate(offsets)])
        delta=pixels-torch.cat((pixels[:,:1],pixels[:,:-1]),1)
        return self.net(torch.stack((pixels,delta),2).reshape(b*t,2,64,96)).reshape(b,t,128)


class ContentCandidate(nn.Module):
    """Frozen base context + one declared modality -> candidate latent/mel corrections."""
    def __init__(self,arm,words,seed,audio_width=512):
        super().__init__(); self.arm=arm; self.is_audio=arm=="audio"
        self.front=nn.Sequential(nn.Linear(256,audio_width),nn.SiLU(),nn.Linear(audio_width,128)) if self.is_audio else SpatialVideo()
        with torch.random.fork_rng():
            torch.manual_seed(seed+22000)
            self.temporal=nn.Sequential(TemporalResidual(128,1),TemporalResidual(128,2)); self.context=nn.Sequential(nn.Linear(512,256),nn.SiLU()); self.query=nn.Linear(256,128); self.attend=nn.MultiheadAttention(128,4,batch_first=True)
            self.fuse=nn.Sequential(nn.Linear(384,256),nn.SiLU()); self.delta_field=nn.Linear(256,256); self.delta_unit=nn.Linear(256,words); self.hint_unit=nn.Linear(128,words); self.head=MultirateAudioHead()
        for module in (self.delta_field,self.delta_unit,self.head.output): nn.init.zeros_(module.weight); nn.init.zeros_(module.bias)
        if not self.is_audio:
            self.audio_sync=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,64)); self.video_sync=nn.Linear(128,64); self.log_temperature=nn.Parameter(torch.tensor(2.3)); self.match=nn.Sequential(nn.Linear(19,64),nn.SiLU(),nn.Linear(64,1))
    def hint(self,field,mouth):
        return self.temporal(self.front(field) if self.is_audio else self.front(mouth))
    def synchronize(self,observed_audio,hint,visible):
        """Only real observed audio frames ->17 offset scores and aligned quality[B]."""
        audio=F.normalize(self.audio_sync(observed_audio).float(),dim=-1); video=F.normalize(self.video_sync(hint).float(),dim=-1); t=audio.shape[1]; scores=[]
        for offset in range(-8,9):
            lo=max(0,-offset); hi=min(t,t-offset); valid=visible[:,lo:hi]
            cosine=(audio[:,lo:hi]*video[:,lo+offset:hi+offset]).sum(-1)
            scores.append((cosine*valid).sum(1)/valid.sum(1).clamp_min(1))
        raw=torch.stack(scores,1); scale=self.log_temperature.exp().clamp(1,50); logits=raw*scale
        features=torch.cat((raw,raw.max(1,keepdim=True).values,raw.std(1,keepdim=True)),1); aligned=self.match(features)[...,0].float()
        quality=aligned.sigmoid()*logits.softmax(-1)[:,8]
        return {"offset_logits":logits,"match_logits":aligned,"quality":quality,"predicted_offset":logits.argmax(1)-8}
    def forward(self,base,obs,field,base_logits,base_prediction,mouth,unknown):
        b,t=field.shape[:2]; hint=self.hint(field,mouth); context=self.context(torch.cat((field,base_prediction.reshape(b,t,256)),-1)); index=torch.arange(t,device=field.device)
        attended,_=self.attend(self.query(context),hint,hint,attn_mask=(index[:,None]-index[None,:]).abs()>2,need_weights=False); fused=self.fuse(torch.cat((context,attended),-1))
        coarse=unknown.reshape(b,t,4).any(-1)
        logits=base_logits+self.delta_unit(fused).float()*coarse[...,None]; changed=field+.5*self.delta_field(fused).tanh()*coarse[...,None]
        candidate=base.decode(obs,changed,logits)+self.head(fused).float().tanh()*unknown[...,None]
        prediction=torch.where(unknown[...,None],candidate,base_prediction)
        result={"prediction":prediction,"unit_logits":logits,"hint_unit_logits":self.hint_unit(hint).float(),"hint":hint,"candidate_field":changed}
        if not self.is_audio: result.update(self.synchronize(obs["audio"],hint,obs["visible"]))
        return result


class ContentSyncSystem(nn.Module):
    """Matched audio/video candidates; rejected V returns the strong audio candidate."""
    def __init__(self,expert_cfg,expert_run,words,seed,arm="audio",audio_width=512):
        super().__init__(); self.arm=arm; self.base=SpeechContentBase(expert_cfg,expert_run,words,seed).requires_grad_(False)
        if arm!="audio": self.audio_candidate=ContentCandidate("audio",words,seed,audio_width).requires_grad_(False)
        self.branch=ContentCandidate(arm,words,seed,audio_width)
    def train(self,mode=True):
        super().train(mode); self.base.eval()
        if self.arm!="audio": self.audio_candidate.eval()
        return self
    def forward(self,obs,mouth,unknown,present=None,threshold=None):
        with torch.no_grad(): base=self.base(obs)
        out=self.branch(self.base,obs,base["field"],base["unit_logits"],base["prediction"],mouth,unknown)
        out["base_field"]=base["field"]; out["base_unit_logits"]=base["unit_logits"]
        if self.arm=="audio": return out
        with torch.no_grad(): fallback_result=self.audio_candidate(self.base,obs,base["field"],base["unit_logits"],base["prediction"],mouth,unknown)
        fallback=fallback_result["prediction"]; out["raw_prediction"]=out["prediction"]; out["fallback"]=fallback
        out["raw_unit_logits"]=out["unit_logits"]; out["raw_candidate_field"]=out["candidate_field"]
        if threshold is not None:
            accept=(out["quality"]>=threshold)&(out["predicted_offset"]==0)
            if present is not None: accept=accept&present
            out["accepted"]=accept; out["prediction"]=torch.where(accept[:,None,None],out["raw_prediction"],fallback)
            out["unit_logits"]=torch.where(accept[:,None,None],out["raw_unit_logits"],fallback_result["unit_logits"])
            out["candidate_field"]=torch.where(accept[:,None,None],out["raw_candidate_field"],fallback_result["candidate_field"])
        elif present is not None:
            out["prediction"]=torch.where(present[:,None,None],out["prediction"],fallback)
            out["unit_logits"]=torch.where(present[:,None,None],out["unit_logits"],fallback_result["unit_logits"])
            out["candidate_field"]=torch.where(present[:,None,None],out["candidate_field"],fallback_result["candidate_field"])
        return out
