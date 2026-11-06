import torch
from training.predictive_losses import spread


def test_spread_penalizes_rank_one_features_even_with_adequate_variance():
    generator=torch.Generator().manual_seed(141)
    low=torch.randn(128,1,generator=generator).repeat(1,32).requires_grad_()
    healthy=torch.randn(128,32,generator=generator)
    va,ca=spread(low);vb,cb=spread(healthy)
    assert va<.01 and vb<.01
    assert ca>10*cb
    (va+ca).backward()
    assert torch.isfinite(low.grad).all() and low.grad.abs().sum()>0


def test_fixed_raw_baseline_cannot_bypass_the_predictive_state():
    from models.predictive_world import PredictiveWorld
    from training.predictive_losses import loss
    net=PredictiveWorld(width=32,layers=1,heads=4,slots=4,memory=24,phone_count=79).eval()
    obs={'time':torch.arange(4.)[None].repeat(2,1),'time_unit':torch.ones(2,dtype=torch.long),'valid':torch.ones(2,4,dtype=torch.bool),'numeric':torch.randn(2,4,64),'numeric_present':torch.ones(2,4,64,dtype=torch.bool)}
    batch={'family':'pamap2','obs':obs,'query_time':obs['time'],'numeric_target':torch.randn(2,4,64),'numeric_target_present':torch.ones(2,4,64,dtype=torch.bool),'numeric_hidden':torch.ones(2,4,64,dtype=torch.bool),'numeric_baseline':torch.zeros(2,4,64)}
    cfg={'loss_weights':{'reconstruction':1.,'state_variance':1.,'state_covariance':.01}}
    a=loss(net,None,batch,{},cfg,0)[0]
    batch['numeric_baseline'].fill_(1e6)
    b=loss(net,None,batch,{},cfg,0)[0]
    assert torch.equal(a,b)
