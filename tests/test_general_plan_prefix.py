import torch
from models.slwm_general import GeneralWorld


def test_current_g1_forecaster_is_point_conditioned_not_prefix_conditioned():
    model=GeneralWorld(width=32,layers=1,heads=4).eval()
    obs={"time":torch.arange(4.)[None],"time_unit":torch.ones(1,dtype=torch.long),"valid":torch.ones(1,4,dtype=torch.bool),"numeric":torch.randn(1,4,64),"numeric_present":torch.ones(1,4,64,dtype=torch.bool)}
    with torch.no_grad():
        state=model(obs);plan=torch.randn(1,8,7);changed=plan.clone();changed[:,:7]+=5
        times=torch.arange(4.,12.)[None]
        a=model.query(state,times,obs["time_unit"],2,plan)
        b=model.query(state,times,obs["time_unit"],2,changed)
    assert torch.equal(a[:,-1],b[:,-1])
    assert not torch.equal(a[:,0],b[:,0])
