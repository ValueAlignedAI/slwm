"""G1 typed observation edges and a clock-consistent shared latent processor.

Edges accept observations only. Targets are supplied to losses, never forward.
Inference returns internal representations; callers explicitly request a proposal.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class TemporalSignal(nn.Module):
    """Field[B,T,K,D] -> same, causal filtering along chronological T only.

    Sequential normalization/nonlinearity and an input-dependent gate separate
    this path from the previously foldable sum of parallel linear FIR branches.
    Fourier coefficients parameterize a31-tap causal filter; no future FFT input.
    """
    def __init__(self,width):
        super().__init__(); self.norm=nn.LayerNorm(width); self.local=nn.Conv1d(width,width,5,groups=width)
        self.real=nn.Parameter(torch.randn(width,16)*.002); self.imag=nn.Parameter(torch.randn(width,15)*.002)
        self.scale=nn.Parameter(torch.zeros(width)); self.gate_weight=nn.Parameter(torch.zeros(width)); self.gate_bias=nn.Parameter(torch.zeros(width))
    def forward(self,z):
        b,t,k,d=z.shape; x=self.norm(z); sequence=x.permute(0,2,3,1).reshape(b*k,d,t)
        local=F.silu(self.local(F.pad(sequence,(4,0))))
        imaginary=F.pad(self.imag,(1,0))
        kernel=torch.fft.irfft(torch.complex(self.real,imaginary),n=31,dim=-1).flip(-1)[:,None].to(local.dtype)
        long=F.conv1d(F.pad(local,(30,0)),kernel,groups=d).reshape(b,k,d,t).permute(0,3,1,2)
        return z+torch.sigmoid(x*self.gate_weight+self.gate_bias)*torch.tanh(self.scale+.1)*long


class FieldBlock(nn.Module):
    """Time/slot field[B,T,K,D] -> same; same-time slots bind, future ticks masked."""
    def __init__(self,width,heads):
        super().__init__(); self.n1=nn.LayerNorm(width); self.attn=nn.MultiheadAttention(width,heads,batch_first=True)
        self.n2=nn.LayerNorm(width); self.mlp=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width)); self.signal=None
    def forward(self,z,valid):
        if self.signal is not None: z=self.signal(z)
        b,t,k,d=z.shape; x=z.reshape(b,t*k,d); q=self.n1(x)
        ticks=torch.arange(t,device=z.device).repeat_interleave(k)
        future=ticks[None,:]>ticks[:,None]
        x=x+self.attn(q,q,q,attn_mask=future,key_padding_mask=~valid.repeat_interleave(k,1),need_weights=False)[0]
        return (x+self.mlp(self.n2(x))).reshape(b,t,k,d)


class GeneralWorld(nn.Module):
    """Typed observations -> shared field[B,T,K,D], independent of dataset/task ID.

    Optional inputs: audio[B,T,4,100], audio_present[B,T,4];
    images[B,T,V,3,64,64], image_present[B,T,V]; numeric[B,T,64],
    numeric_present[B,T,64]; text[B,L], text_present[B,L]. All masks are Boolean.
    time[B,T] is relative physical seconds or explicitly flagged ordinal steps.
    text is a supplied observation/goal only; label captions stay outside forward.
    """
    def __init__(self,variant="slwm",width=512,layers=10,heads=8,slots=4,vocabulary=50257,phone_count=256,seed=101):
        super().__init__(); self.width=width; self.slots=slots; self.variant=variant; self.phone_count=phone_count
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.audio_edge=nn.Sequential(nn.Linear(100,width),nn.SiLU(),nn.Linear(width,width)); self.audio_subframe=nn.Embedding(4,width)
            self.vision_edge=nn.Sequential(nn.Conv2d(3,32,4,2,1),nn.SiLU(),nn.Conv2d(32,64,4,2,1),nn.SiLU(),nn.Conv2d(64,128,4,2,1),nn.SiLU(),nn.Conv2d(128,width,4,2,1))
            self.patch=nn.Embedding(16,width); self.view=nn.Embedding(4,width)
            self.numeric_edge=nn.Sequential(nn.Linear(1,width),nn.SiLU(),nn.Linear(width,width)); self.channel=nn.Embedding(64,width)
            self.text_edge=nn.Embedding(vocabulary,256); self.text_projection=nn.Linear(256,width); self.text_position=nn.Embedding(128,width)
            self.modality=nn.Embedding(4,width); self.clock=nn.Sequential(nn.Linear(10,width),nn.SiLU(),nn.Linear(width,width)); self.unit=nn.Embedding(2,width)
            self.latents=nn.Parameter(torch.randn(slots,width)*.02); self.resample_norm=nn.LayerNorm(width); self.resample=nn.MultiheadAttention(width,heads,batch_first=True)
            self.blocks=nn.ModuleList([FieldBlock(width,heads) for _ in range(layers)]); self.norm=nn.LayerNorm(width)
            self.query_norm=nn.LayerNorm(width); self.query_attention=nn.MultiheadAttention(width,heads,batch_first=True); self.query_kind=nn.Embedding(4,width)
            self.action_edge=nn.Sequential(nn.Linear(7,width),nn.SiLU(),nn.Linear(width,width))
            self.embedding=nn.Linear(width,128); self.future_embedding=nn.Linear(width,128); self.mel=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,400))
            self.numeric=nn.Linear(width,64); self.picture=nn.Linear(width,3*16*16); self.phones=nn.Linear(width,4*phone_count); self.characters=nn.Linear(width,29)
            self.action=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,14)); self.risk=nn.Linear(width,1)
            self.correspondence=nn.Sequential(nn.Linear(128*4,256),nn.SiLU(),nn.Linear(256,1))
            self.register_buffer("action_ready",torch.tensor(False))
            for layer in (self.mel[-1],self.numeric,self.picture):
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        if variant=="slwm":
            with torch.random.fork_rng():
                torch.manual_seed(seed+9000)
                for block in self.blocks: block.signal=TemporalSignal(width)
        elif variant=="pooled_transformer":
            # A conventional fixed-modality fusion control at almost the same
            # parameter budget. Replace learned latent resampling with dense
            # binding of fine audio bins, named scalar channels and camera views.
            del self.resample,self.resample_norm
            with torch.random.fork_rng():
                torch.manual_seed(seed+17000);self.fixed_audio=nn.Linear(400,width);self.fixed_numeric=nn.Linear(128,width);self.fixed_views=nn.Linear(3*width,width)
        elif variant!="transformer": raise ValueError(variant)

    def clock_features(self,time,unit,origin=None):
        """Relative timestamps[B,T] and unit[B] -> positional field[B,T,D]."""
        t=time-(time[:,:1] if origin is None else origin); frequency=torch.tensor([.5,1.,2.,4.],device=t.device); angle=t[...,None]*frequency
        interval=torch.cat((torch.zeros_like(t[:,:1]),(t[:,1:]-t[:,:-1]).clamp_min(0)),1)
        features=torch.cat((angle.sin(),angle.cos(),torch.log1p(t.clamp_min(0))[...,None],torch.log1p(interval)[...,None]),-1)
        return self.clock(features)+self.unit(unit)[:,None]

    def forward(self,obs):
        time=obs["time"]; b,t=time.shape; device=time.device; tokens=[]; masks=[]; kinds=[]; clock=self.clock_features(time,obs["time_unit"])
        def append(values,present,kind):
            # Erase missing content before *every* learned edge, as well as after.
            # Edges below already receive erased values; masked tokens cannot bind.
            tokens.append(values+self.modality.weight[kind]+clock[:,:,None]); masks.append(present);kinds.append(kind)
        if "audio" in obs:
            present=obs["audio_present"] & obs["valid"][...,None]; values=obs["audio"].masked_fill(~present[...,None],0)
            append(self.audio_edge(values)+self.audio_subframe.weight[None,None],present,0)
        if "images" in obs:
            present=obs["image_present"] & obs["valid"][...,None]; images=obs["images"].masked_fill(~present[...,None,None,None],0)
            if "pixel_present" in obs: images=images.masked_fill(~obs["pixel_present"],0)
            v=images.shape[2]; image=self.vision_edge(images.reshape(b*t*v,3,64,64)).flatten(2).transpose(1,2).reshape(b,t,v,16,self.width)
            image=image+self.patch.weight[None,None,None]+self.view.weight[:v][None,None,:,None]
            if "image_weight" in obs: image=image*obs["image_weight"][...,None,None]
            patches=present.repeat_interleave(16,2)
            if "patch_present" in obs: patches=patches&obs["patch_present"].reshape(b,t,v*16)
            append(image.reshape(b,t,v*16,self.width),patches,1)
        if "numeric" in obs:
            present=obs["numeric_present"] & obs["valid"][...,None]; values=obs["numeric"].masked_fill(~present,0)
            append(self.numeric_edge(values[...,None])+self.channel.weight[None,None],present,2)
        if "text" in obs:
            present=obs["text_present"]; ids=obs["text"].masked_fill(~present,0); length=ids.shape[1]
            values=self.text_projection(self.text_edge(ids))+self.text_position.weight[:length][None]
            append(values[:,None].expand(-1,t,-1,-1),present[:,None].expand(-1,t,-1)&obs["valid"][...,None],3)
        # One explicit missing/null observation keeps attention finite even when
        # all sensors are absent. It encodes missingness, not observed zero/silence.
        null=torch.zeros(b,t,1,self.width,device=device)+clock[:,:,None]
        tokens.append(null); masks.append(torch.ones(b,t,1,device=device,dtype=torch.bool))
        values=torch.cat(tokens,2); present=torch.cat(masks,2); values=values.masked_fill(~present[...,None],0)
        queries=self.latents[None,None]+clock[:,:,None]; queries=queries.expand(b,-1,-1,-1)
        if self.variant=="pooled_transformer":
            if self.slots!=4:raise ValueError("Fixed-modality control requires4 slots")
            slots=[clock+self.modality.weight[kind] for kind in range(4)]
            for value,mask,kind in zip(tokens[:-1],masks[:-1],kinds):
                feature=F.silu(value).masked_fill(~mask[...,None],0).sum(2)/mask.sum(2).clamp_min(1)[...,None]
                if kind==0:
                    raw=obs["audio"].masked_fill(~obs["audio_present"][...,None],0);feature=feature+self.fixed_audio(raw.flatten(2))
                elif kind==2:
                    raw=obs["numeric"].masked_fill(~obs["numeric_present"],0);feature=feature+self.fixed_numeric(torch.cat((raw,obs["numeric_present"].to(raw.dtype)),-1))
                elif kind==1:
                    views=obs["images"].shape[2]
                    if views>3:raise ValueError("Fixed-modality baseline supports the3 registered camera views")
                    local=value.reshape(b,t,views,16,self.width);visible=mask.reshape(b,t,views,16)
                    pooled=local.masked_fill(~visible[...,None],0).sum(3)/visible.sum(3).clamp_min(1)[...,None]
                    feature=feature+self.fixed_views(F.pad(pooled,(0,0,0,3-views)).flatten(2))
                slots[kind]=feature
            z=queries+torch.stack(slots,2)
        else:
            z=queries+self.resample(queries.reshape(b*t,self.slots,self.width),self.resample_norm(values).reshape(b*t,-1,self.width),self.resample_norm(values).reshape(b*t,-1,self.width),key_padding_mask=~present.reshape(b*t,-1),need_weights=False)[0].reshape(b,t,self.slots,self.width)
        for block in self.blocks: z=block(z,obs["valid"])
        z=self.norm(z); valid=obs["valid"][...,None,None]
        pooled=(z*valid).sum((1,2))/(obs["valid"].sum(1).clamp_min(1)*self.slots)[:,None]
        raw=self.embedding(pooled).float()
        return {"field":z,"raw_embedding":raw,"embedding":F.normalize(raw,dim=-1),"sequence_embedding":F.normalize(self.embedding(z.mean(2)).float(),dim=-1),"valid":obs["valid"],"last":z[:,-1].mean(1),"time_origin":time[:,:1]}

    def query(self,state,times,time_unit,kind,planned_actions=None,view=None):
        """Received field[B,T,K,D] + requested times[B,Q] -> readouts[B,Q,D].

        Repair queries may inspect the entire explicitly received window. Future
        queries see that same past context plus optional *supplied* action plans;
        future observation targets are never passed to this function.
        """
        z=state["field"]; b,t,k,d=z.shape; q=self.clock_features(times,time_unit,state["time_origin"])+self.query_kind.weight[kind]
        if planned_actions is not None: q=q+self.action_edge(planned_actions)
        if view is not None:q=q+self.view.weight[view]
        key=z.reshape(b,t*k,d); return q+self.query_attention(self.query_norm(q),key,key,key_padding_mask=~state["valid"].repeat_interleave(k,1),need_weights=False)[0]

    def propose(self,features,family):
        """Explicitly select one internal proposal; no physical command is sent."""
        if family=="audio": return self.mel(features).float()
        if family=="numeric": return self.numeric(features).float()
        if family=="image": return self.picture(features).float().reshape(*features.shape[:-1],3,16,16)
        if family=="phones": return self.phones(features).float().reshape(*features.shape[:-1],4,self.phone_count)
        if family=="characters": return self.characters(features).float()
        if family=="action":
            if not bool(self.action_ready):raise RuntimeError("Train/load the frozen-state actor before requesting actions")
            values=self.action(features).float(); return {"mean":values[...,:7],"log_scale":values[...,7:].clamp(-5,3),"risk":self.risk(features).float()[...,0],"status":"internal_proposal"}
        raise ValueError(family)

    def correspondence_logits(self,a,b):
        """Independently encoded pairs[B,128] -> synchronization logit[B]."""
        return self.correspondence(torch.cat((a,b,(a-b).abs(),a*b),-1)).float()[...,0]
