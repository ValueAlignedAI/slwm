import torch

from models.predictive_world import PredictiveWorld,join_observations


def observations(steps=6,batch=2):
    return {'time':torch.arange(float(steps))[None].repeat(batch,1),'time_unit':torch.ones(batch,dtype=torch.long),'valid':torch.ones(batch,steps,dtype=torch.bool),'numeric':torch.randn(batch,steps,64),'numeric_present':torch.ones(batch,steps,64,dtype=torch.bool)}


def model(**kwargs):return PredictiveWorld(width=32,layers=1,heads=4,slots=4,memory=24,phone_count=79,**kwargs).eval()


def test_forecast_uses_action_history_and_never_later_commands():
    net=model();obs=observations();state=net(obs);times=torch.arange(6.,14.)[None].repeat(2,1)
    plan=torch.randn(2,8,7);earlier=plan.clone();earlier[:,:7]+=2;later=plan.clone();later[:,1:]+=3
    a=net.forecast_states(state,times,obs['time_unit'],plan)
    b=net.forecast_states(state,times,obs['time_unit'],earlier)
    c=net.forecast_states(state,times,obs['time_unit'],later)
    assert not torch.equal(a[:,-1],b[:,-1])
    assert torch.equal(a[:,0],c[:,0])
    absent=net.forecast_states(state,times,obs['time_unit'],None)
    mean_plan=net.forecast_states(state,times,obs['time_unit'],torch.zeros_like(plan))
    assert not torch.equal(absent,mean_plan)
    point=model(action_history=False);s=point(obs)
    assert torch.equal(point.forecast_states(s,times,obs['time_unit'],plan)[:,-1],point.forecast_states(s,times,obs['time_unit'],earlier)[:,-1])


def test_received_field_is_not_a_decoder_bypass_and_all_heads_train_memory():
    net=model().train();obs=observations();state=net(obs)
    assert 'field' not in state and state['shared'].shape==(2,24)
    query=net.query(state,obs['time'],obs['time_unit'],2)
    # Nonzero heads avoid the intentional zero-output initialization's first-step gate.
    loss=sum(net.propose(query,name).square().mean() for name in ('phones','characters'))
    loss.backward()
    assert net.history.weight_ih_l0.grad.abs().sum()>0
    assert net.history_input[0].weight.grad.abs().sum()>0
    assert net.state_slots.weight.grad.abs().sum()>0


def test_stream_consumes_only_newest_packet_and_matches_causal_prefix_pass():
    net=model();obs=observations();memory=None
    with torch.no_grad():
        batch=net(obs)
        for end in range(1,7):
            prefix={k:(v[:,:end] if v.ndim>=2 else v) for k,v in obs.items()}
            current=net.stream_step(prefix,memory);memory=current['memory']
        assert torch.allclose(current['shared'],batch['shared'],atol=2e-6)


def test_controls_keep_parameter_budget_and_teacher_join_erases_absent_edges():
    counts=[sum(p.numel() for p in model(**kw).parameters()) for kw in ({},{'bypass':True},{'action_history':False})]
    assert len(set(counts))==1
    past=observations(4);future=observations(2);future['time']+=4
    past['images']=torch.randn(2,4,1,3,64,64);past['image_present']=torch.ones(2,4,1,dtype=torch.bool)
    joined=join_observations(past,future)
    assert joined['time'].shape==(2,6)
    assert not joined['image_present'][:,4:].any()
    assert joined['images'][:,4:].eq(0).all()
