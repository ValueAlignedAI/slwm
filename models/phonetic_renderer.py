"""Small supervised phone-to-mel codec; separate linguistic and acoustic errors."""
import torch
from torch import nn
import torch.nn.functional as F


def observed_condition(values,present):
    """Normalized mel[B,F,100] + observed[B,F] -> style[B,100], prosody[B,F,2].

    Erase unknown values before every statistic/interpolation. All-missing uses
    the training mean voice/energy (normalized zero), not a hidden reference.
    """
    b,t,c=values.shape;values=values.masked_fill(~present[...,None],0);style=values.sum(1)/present.sum(1).clamp_min(1)[:,None]
    clock=torch.arange(t,device=values.device)[None].expand(b,-1);left=torch.where(present,clock,-1).cummax(1).values;right=torch.where(present,clock,t).flip(1).cummin(1).values.flip(1)
    left=torch.where(left<0,right,left).clamp(0,t-1);right=torch.where(right>=t,left,right).clamp(0,t-1)
    lo=values.gather(1,left[...,None].expand(-1,-1,c));hi=values.gather(1,right[...,None].expand(-1,-1,c));fraction=((clock-left)/(right-left).clamp_min(1))[...,None]
    baseline=torch.lerp(lo,hi,fraction);prosody=torch.stack((baseline.mean(-1),baseline.std(-1,unbiased=False)),-1)
    return style,prosody


class PhoneticRenderer(nn.Module):
    """Phones[B,F], observed-only style[B,100]/prosody[B,F,2] -> mel[B,F,100].

    Train with supplied real phone annotations; G1 inference supplies its own
    predicted phones. Three fine frames (~32ms) of unit lookahead precede Vocos.
    No speaker ID, text teacher or pretrained TTS weights are used here.
    """
    def __init__(self,phone_count,mean,scale,seed=701):
        super().__init__();self.register_buffer("mean",torch.as_tensor(mean,dtype=torch.float32));self.register_buffer("scale",torch.as_tensor(scale,dtype=torch.float32))
        with torch.random.fork_rng():
            torch.manual_seed(seed);self.phones=nn.Embedding(phone_count,96);self.style=nn.Sequential(nn.Linear(100,32),nn.SiLU(),nn.Linear(32,32));self.net=nn.Sequential(nn.Conv1d(130,256,5,padding=2),nn.SiLU(),nn.Conv1d(256,128,3,padding=1),nn.SiLU(),nn.Conv1d(128,100,1))
    def forward(self,phones,style,prosody):
        x=torch.cat((self.phones(phones.clamp_min(0)),self.style(style)[:,None].expand(-1,phones.shape[1],-1),prosody),-1)
        return self.net(x.transpose(1,2)).transpose(1,2).float()
    def render_physical(self,phones,observed_mel,present):
        normalized=(observed_mel-self.mean)/self.scale;style,prosody=observed_condition(normalized,present)
        return self(phones,style,prosody)*self.scale+self.mean
