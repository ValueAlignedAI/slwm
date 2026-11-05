"""Inference-only algebraic collapse of B1's parallel linear signal branches.

The three branches share an input and have no intervening nonlinearity, so their
causally aligned kernels/biases sum into one31-tap depthwise convolution. This
does NOT apply to the sequentially normalized T1 text blocks. Training with the
folded parameterization has different optimizer geometry and is not substituted.
FP32 equivalence is numerical, not a claim of bit-identical AMP trajectories.
"""
import torch
from torch import nn
import torch.nn.functional as F


class FoldedStreamBlock(nn.Module):
    def __init__(self,block):
        super().__init__()
        for name in ("norm_signal","norm_attn","attn","norm_mlp","mlp"): setattr(self,name,getattr(block,name))
        with torch.no_grad():
            spectral=torch.fft.irfft(torch.complex(block.spectral.real,block.spectral.imag),n=block.spectral.kernel_size,dim=-1)[:,None]; longest=block.long.kernel_size; kernel=F.pad(block.local.conv.weight,(longest-block.local.kernel_size,0))+F.pad(spectral,(longest-block.spectral.kernel_size,0))+block.long.conv.weight; bias=block.local.conv.bias+block.spectral.bias+block.long.conv.bias
        self.kernel=nn.Parameter(kernel.detach().clone(),requires_grad=False); self.bias=nn.Parameter(bias.detach().clone(),requires_grad=False); self.length=longest

    def forward(self,z,valid):
        x=self.norm_signal(z).transpose(1,2); signal=F.conv1d(F.pad(x,(self.length-1,0)),self.kernel,self.bias,groups=self.kernel.shape[0]).transpose(1,2); z=z+.1*signal; q=self.norm_attn(z); z=z+self.attn(q,q,q,key_padding_mask=~valid,need_weights=False)[0]; return z+self.mlp(self.norm_mlp(z))


def fold_signal_branches(model):
    if model.training: raise ValueError("Signal folding is an inference transformation; call eval() first")
    if model.variant!="slwm": raise ValueError("Only B1 parallel-signal SLWM worlds are eligible")
    original=sum(p.numel() for p in model.parameters()); model.blocks=nn.ModuleList([FoldedStreamBlock(block) for block in model.blocks]); model.eval().requires_grad_(False); model.folding_metadata={"original_trained_parameters":original,"deployed_parameters":sum(p.numel() for p in model.parameters()),"training_weights_modified":False,"training_optimizer_equivalence_claimed":False,"function_equivalence_scope":"Real arithmetic; FP32/AMP numerical agreement measured separately"}; return model
