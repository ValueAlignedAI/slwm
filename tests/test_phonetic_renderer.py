import numpy as np
import torch
from models.phonetic_renderer import PhoneticRenderer,observed_condition


def test_renderer_condition_never_reads_hidden_mels():
    values=torch.randn(2,16,100);present=torch.ones(2,16,dtype=torch.bool);present[:,5:9]=False
    style,prosody=observed_condition(values,present);changed=values.masked_fill(~present[...,None],1e9)
    other_style,other_prosody=observed_condition(changed,present)
    torch.testing.assert_close(style,other_style,rtol=0,atol=0);torch.testing.assert_close(prosody,other_prosody,rtol=0,atol=0)
    a,b=observed_condition(values,torch.zeros_like(present));assert torch.count_nonzero(a)==torch.count_nonzero(b)==0


def test_renderer_shapes_gradients_and_physical_scale():
    model=PhoneticRenderer(79,np.full(100,-4,np.float32),np.full(100,2,np.float32));phones=torch.randint(0,79,(2,16));mel=torch.randn(2,16,100);present=torch.ones(2,16,dtype=torch.bool)
    physical=model.render_physical(phones,mel,present);assert physical.shape==mel.shape;physical.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters())==289860
