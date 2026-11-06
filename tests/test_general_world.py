"""Observation isolation, actual-time causality and shared typed interfaces."""
import torch
import pytest
from models.slwm_general import GeneralWorld


def setup():
    model=GeneralWorld(width=32,layers=2,heads=4,vocabulary=64,phone_count=11).eval()
    obs={"time":torch.arange(6)[None].float().expand(2,-1)*.02,"time_unit":torch.zeros(2,dtype=torch.long),"valid":torch.ones(2,6,dtype=torch.bool),"numeric":torch.randn(2,6,64),"numeric_present":torch.rand(2,6,64)>.4}
    return model,obs


def test_missing_content_never_affects_the_field():
    model,obs=setup(); before=model(obs)["field"]
    changed={**obs,"numeric":obs["numeric"].masked_fill(~obs["numeric_present"],1e9)}
    torch.testing.assert_close(before,model(changed)["field"],rtol=0,atol=0)


def test_future_ticks_cannot_modify_earlier_fields():
    model,obs=setup(); before=model(obs)["field"]
    changed={**obs,"numeric":obs["numeric"].clone()}; changed["numeric"][:,4:]+=500
    torch.testing.assert_close(before[:,:4],model(changed)["field"][:,:4],rtol=1e-5,atol=1e-5)


def test_typed_edges_queries_and_proposals():
    model,obs=setup(); b,t=obs["time"].shape
    obs.update(audio=torch.randn(b,t,4,100),audio_present=torch.ones(b,t,4,dtype=torch.bool),images=torch.randn(b,t,2,3,64,64),image_present=torch.ones(b,t,2,dtype=torch.bool),text=torch.randint(0,64,(b,7)),text_present=torch.ones(b,7,dtype=torch.bool))
    state=model(obs); assert state["field"].shape==(b,t,4,32)
    query=model.query(state,torch.tensor([[.2,.4]]).expand(b,-1),obs["time_unit"],2,torch.randn(b,2,7))
    assert query.shape==(b,2,32)
    assert model.propose(query,"audio").shape==(b,2,400)
    assert model.propose(query,"phones").shape==(b,2,4,11)
    assert model.propose(query,"image").shape==(b,2,3,16,16)
    with pytest.raises(RuntimeError,match="frozen-state actor"):model.propose(query,"action")
    model.action_ready.fill_(True)
    assert model.propose(query,"action")["status"]=="internal_proposal"
    model({k:v for k,v in obs.items() if k in ("time","time_unit","valid")})
    shifted=model.query(state,torch.tensor([[.5,.7]]).expand(b,-1),obs["time_unit"],2,torch.zeros(b,2,7))
    earlier=model.query(state,torch.tensor([[.2,.4]]).expand(b,-1),obs["time_unit"],2,torch.zeros(b,2,7))
    assert not torch.allclose(shifted,earlier)  # Horizon offsets must survive query clocks.


def test_fixed_modality_transformer_has_distinct_typed_slots():
    model=GeneralWorld("pooled_transformer",width=32,layers=1,heads=4,vocabulary=64,phone_count=11).eval()
    _,obs=setup();obs.update(audio=torch.randn(2,6,4,100),audio_present=torch.ones(2,6,4,dtype=torch.bool),images=torch.randn(2,6,1,3,64,64),image_present=torch.ones(2,6,1,dtype=torch.bool))
    output=model(obs);assert output["field"].shape==(2,6,4,32) and torch.isfinite(output["field"]).all()
    assert not hasattr(model,"resample")
