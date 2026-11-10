import torch
from models.predictive_world import PredictiveWorld
from exploration.predictive_infer import LiveWorld,weight_hash


def test_stale_and_missing_packets_cannot_mutate_memory_or_propose_actions():
    net=PredictiveWorld(width=32,layers=1,heads=4,slots=4,memory=24,phone_count=79).eval();loop=LiveWorld(net)
    obs={'time':torch.arange(4.)[None],'time_unit':torch.ones(1,dtype=torch.long),'valid':torch.ones(1,4,dtype=torch.bool),'numeric':torch.randn(1,4,64),'numeric_present':torch.ones(1,4,64,dtype=torch.bool)}
    before=weight_hash(net);loop.observe(obs,1.);memory=loop.memory.clone()
    with torch.inference_mode():expected=net(obs)['memory']
    assert torch.equal(memory,expected)
    assert loop.observe(obs,.9)['status']=='stale_rejected'
    assert torch.equal(memory,loop.memory)
    obs['numeric_present'][:,-1]=False
    assert loop.observe(obs,2.)['status']=='missing_observation'
    assert torch.equal(memory,loop.memory) and loop.updates==1
    assert before==weight_hash(net)
