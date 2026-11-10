import numpy as np
import pytest
torch=pytest.importorskip("torch")
pytest.importorskip("vocos")
pytest.importorskip("cmudict")
from models.slwm_stream import StreamWorld
from training.stream_sensing import RiskActor,choose,observe,features,raw_features,future_error,weight_digest,risk_masks


def test_risk_head_selection_splits_people_not_adjacent_ticks():
    actors=["person"+str(i) for i in range(10) for _ in range(4)]; fit,validation,held=risk_masks(actors,{"risk_holdout_people":3}); assert len(held)==3; assert not set(np.asarray(actors)[fit])&set(np.asarray(actors)[validation]); assert fit.sum()==28 and validation.sum()==12


def test_sensing_cost_and_goal_change_decisions():
    errors=torch.tensor([[[.1,3.],[3.,.1],[.1,.1],[4.,4.]]])
    assert choose(errors,"audio",.05).item()==0
    assert choose(errors,"video",.05).item()==1
    assert choose(errors,"both",.05).item()==2
    assert choose(torch.zeros_like(errors),"both",.05).item()==3


def test_counterfactuals_do_not_receive_future_and_preserve_weights():
    torch.set_num_threads(1); torch.manual_seed(18); model=StreamWorld(width=32,heads=4,layers=2,memory=2).eval(); audio=torch.randn(2,80,400); video=torch.randn(2,80,64); bits=torch.tensor([[1.,0.],[0.,1.]]); before=weight_digest(model)
    with torch.inference_mode():
        out=observe(model,audio,video,0,bits); changed=audio.clone(); changed[:,32:]+=1000; second=observe(model,changed,video,0,bits)
        assert torch.equal(out["future"],second["future"])
        altered=audio.clone(); altered[1,:32]+=1000; invisible=observe(model,altered,video,0,bits); assert torch.equal(out["field"][1],invisible["field"][1])
        assert torch.equal(out["raw_features"][1],invisible["raw_features"][1]); assert raw_features(out,bits).shape==(2,1027)
        error=future_error(out,audio,video,0); assert error.shape==(2,2)
        actor=RiskActor(67); assert actor(features(out,bits)).shape==(2,4,2)
    assert before==weight_digest(model)


def test_paced_replay_waits_for_observations_before_processing_them(monkeypatch):
    from types import SimpleNamespace
    import training.stream_sensing as sensing
    torch.set_num_threads(1)
    model=StreamWorld(width=32,heads=4,layers=1,memory=2).eval()
    actor=RiskActor(67).eval(); audio=torch.randn(1,106,400); video=torch.randn(1,106,64)
    data=SimpleNamespace(rows={"ami_test":[{"offset":0}]},arrays={"ami_test":(None,np.zeros(106,np.int16))})
    plan={"name":"ami_test","index":0,"id":"fixture","actor":"fixture","start":0,"start_seconds":0.}
    clock=[0.]; observed=[]; original=sensing.observe
    def observe_at_time(*args,**kwargs):
        observed.append((args[3],clock[0]))
        return original(*args,**kwargs)
    monkeypatch.setattr(sensing,"sequence",lambda *args:(audio,video))
    monkeypatch.setattr(sensing,"observe",observe_at_time)
    monkeypatch.setattr(sensing.time,"perf_counter",lambda:clock[0])
    monkeypatch.setattr(sensing.time,"sleep",lambda duration:clock.__setitem__(0,clock[0]+duration))
    cfg={"episode_chunks":2,"sensor_cost":.05,"actions":["listen","look","both","wait"]}
    rows,audit=sensing.replay(model,actor,(torch.zeros(67),torch.ones(67)),{},data,[plan],cfg,"both","learned",device="cpu",paced=True)
    duration=32*sensing.PACKET_SECONDS
    assert observed==[(0,0.),(32,duration),(64,2*duration)]
    assert [r["decision_wall_seconds"] for r in rows]==[0.,duration]
    assert all(r["observation_read_wall_seconds"]>=r["observation_available_after_bootstrap_seconds"] for r in rows)
    assert audit["wall_seconds"]==2*duration and audit["weights_unchanged"]
