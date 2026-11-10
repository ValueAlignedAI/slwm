"""Bounded streaming shared fields, recurrent working memory and diagnostic heads."""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from training.t1_torch_text import CausalDepthwiseConv,CausalSpectralMixer


class StreamBlock(nn.Module):
    """Field[B,M+T,D] + valid[B,M+T] -> same; attention sees this received chunk only."""
    def __init__(self,width,heads):
        super().__init__(); self.norm_attn=nn.LayerNorm(width); self.attn=nn.MultiheadAttention(width,heads,batch_first=True); self.norm_mlp=nn.LayerNorm(width); self.mlp=nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width)); self.local=None; self.spectral=None; self.long=None
    def add_signals(self,width):
        self.norm_signal=nn.LayerNorm(width); self.local=CausalDepthwiseConv(width,5); self.spectral=CausalSpectralMixer(width,15); self.long=CausalDepthwiseConv(width,31)
    def forward(self,z,valid):
        if self.local is not None:
            x=self.norm_signal(z); z=z+.1*(self.local(x)+self.spectral(x)+self.long(x))
        q=self.norm_attn(z); z=z+self.attn(q,q,q,key_padding_mask=~valid,need_weights=False)[0]; return z+self.mlp(self.norm_mlp(z))


class StreamWorld(nn.Module):
    """Mel[B,T,400],video[B,T,64],phones[B,T] -> fields and proposals, memory[B,M,D].

    Inputs are explicitly observed values and presence masks. No target arrays
    enter this module. Each chunk is received in full before emitting its output;
    chunk32 means1.365s maximum algorithmic lookahead, not sample-causal control.
    """
    def __init__(self,variant="slwm",width=512,layers=6,heads=8,memory=8,chunk=32,phone_count=41,seed=91):
        super().__init__(); self.variant=variant; self.width=width; self.chunk=chunk; self.memory_slots=memory; self.phone_count=phone_count
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.audio=nn.Sequential(nn.Linear(400,width),nn.SiLU(),nn.Linear(width,width)); self.video=nn.Sequential(nn.Linear(64,width),nn.SiLU(),nn.Linear(width,width)); self.text=nn.Embedding(phone_count,width); self.appearance=nn.Linear(64,width); self.presence=nn.Embedding(8,width)
            nn.init.normal_(self.text.weight,std=.02); nn.init.normal_(self.presence.weight,std=.02)
            self.blocks=nn.ModuleList([StreamBlock(width,heads) for _ in range(layers)]); self.norm=nn.LayerNorm(width); self.memory_query=nn.Parameter(torch.randn(memory,width)*.02); self.memory_read=nn.MultiheadAttention(width,heads,batch_first=True); self.memory_gate=nn.Linear(width*2,width); self.memory_norm=nn.LayerNorm(width)
            self.audio_head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,400)); self.video_head=nn.Linear(width,64); self.phone_head=nn.Linear(width,phone_count); self.character_head=nn.Linear(width,29); self.alignment_head=nn.Linear(width,128); self.future_head=nn.Linear(width,8*464); self.activity_head=nn.Linear(width,2); self.risk_head=nn.Linear(width,1)
            self.phone_head.weight=self.text.weight
            nn.init.zeros_(self.audio_head[-1].weight); nn.init.zeros_(self.audio_head[-1].bias); nn.init.zeros_(self.video_head.weight); nn.init.zeros_(self.video_head.bias)
        # Add optional paths only after every common module is initialized. Paired
        # Transformer/SLWM adapters, attention, MLPs, memory and heads start equal.
        if variant=="slwm":
            with torch.random.fork_rng():
                torch.manual_seed(seed+8000)
                for block in self.blocks: block.add_signals(width)
        elif variant=="late_fusion":
            self.separate=nn.ModuleList([nn.ModuleList([self.blocks[2*k],self.blocks[2*k+1]]) for k in range(3)]); del self.blocks
        elif variant!="transformer": raise ValueError(variant)

    def positions(self,length,device):
        t=torch.arange(length,device=device)[:,None]; f=torch.exp(torch.arange(0,self.width,2,device=device)*(-math.log(10000.)/self.width)); z=torch.zeros(length,self.width,device=device); z[:,::2]=torch.sin(t*f); z[:,1::2]=torch.cos(t*f); return z*.05

    def initial_memory(self,batch,device):
        shape=(batch,3,self.memory_slots,self.width) if self.variant=="late_fusion" else (batch,self.memory_slots,self.width)
        return torch.zeros(shape,device=device)

    def process(self,packet,valid,memory,blocks):
        b=packet.shape[0]; z=torch.cat((memory.to(packet.dtype),packet),1); mask=torch.cat((torch.ones(b,self.memory_slots,device=valid.device,dtype=torch.bool),valid),1)
        for block in blocks: z=block(z,mask)
        z=self.norm(z); query=memory.to(z.dtype)+self.memory_query[None].to(z.dtype); update=self.memory_read(query,z,z,key_padding_mask=~mask,need_weights=False)[0]; gate=self.memory_gate(torch.cat((memory.to(z.dtype),update),-1)).sigmoid(); new=self.memory_norm((1-gate)*memory+gate*update)
        return z[:,self.memory_slots:],new

    def forward(self,audio,video,phones,a_mask,v_mask,t_mask,valid,appearance,memory=None):
        b,t=audio.shape[:2]
        if memory is None: memory=self.initial_memory(b,audio.device)
        audio_input=audio.masked_fill(~a_mask[...,None],0); video_input=video.masked_fill(~v_mask[...,None],0); text_input=phones.masked_fill(~t_mask,0).clamp_min(0)
        streams=[self.audio(audio_input)*a_mask[...,None],self.video(video_input)*v_mask[...,None],self.text(text_input)*t_mask[...,None]]; fields=[]; boundaries=[]; bases=[]
        presence=self.presence(a_mask.long()+2*v_mask.long()+4*t_mask.long()); style=self.appearance(appearance)[:,None]
        for start in range(0,t,self.chunk):
            stop=min(t,start+self.chunk); mask=valid[:,start:stop]; position=self.positions(stop-start,audio.device)[None]; tags=presence[:,start:stop]+style+position
            bases.append(self.interpolate(audio_input[:,start:stop],a_mask[:,start:stop]))
            if self.variant=="late_fusion":
                parts=[]; next_mem=[]
                for k in range(3):
                    h,m=self.process(streams[k][:,start:stop]+tags,mask,memory[:,k],self.separate[k]); parts.append(h); next_mem.append(m)
                active=torch.stack((a_mask[:,start:stop].any(1),v_mask[:,start:stop].any(1),t_mask[:,start:stop].any(1)),-1).float()[:,None]; active=torch.where(active.sum(-1,keepdim=True)>0,active,torch.ones_like(active)); counts=active.sum(-1,keepdim=True).clamp_min(1)
                h=sum(parts[k]*active[...,k:k+1] for k in range(3))/counts; memory=torch.stack(next_mem,1)
            else:
                count=(a_mask[:,start:stop].float()+v_mask[:,start:stop].float()+t_mask[:,start:stop].float()).clamp_min(1).sqrt()[...,None]
                packet=sum(s[:,start:stop] for s in streams)/count+tags; h,memory=self.process(packet,mask,memory,self.blocks)
            fields.append(h); boundaries.append(h[:,-1])
        field=torch.cat(fields,1); bound=torch.where(a_mask,.25,4.)[...,None]; mel=torch.cat(bases,1)+bound*self.audio_head(field).float().tanh(); picture=appearance[:,None]+self.video_head(field).float()
        pooled=(field*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        return {"mel":mel,"video":picture,"field":field,"phone_logits":self.phone_head(field).float(),"character_logits":self.character_head(field).float(),"embedding":F.normalize(self.alignment_head(pooled).float(),dim=-1),"future":self.future_head(torch.stack(boundaries,1)).float().reshape(b,len(boundaries),8,464),"activity":self.activity_head(field).float(),"risk":self.risk_head(field).float()[...,0],"memory":memory}

    @staticmethod
    def interpolate(observed,mask):
        """Chunk-local observed-only linear baseline[B,T,F]; all-missing -> zero."""
        b,t=mask.shape; clock=torch.arange(t,device=mask.device)[None].expand(b,-1)
        left=torch.where(mask,clock,-1).cummax(1).values; right=torch.where(mask,clock,t).flip(1).cummin(1).values.flip(1)
        left=torch.where(left<0,right,left).clamp(0,t-1); right=torch.where(right>=t,left,right).clamp(0,t-1)
        a=observed.gather(1,left[...,None].expand(-1,-1,observed.shape[-1])); c=observed.gather(1,right[...,None].expand(-1,-1,observed.shape[-1])); weight=((clock-left)/(right-left).clamp_min(1))[...,None]
        return torch.lerp(a,c,weight).masked_fill(~mask.any(1)[:,None,None],0.)
