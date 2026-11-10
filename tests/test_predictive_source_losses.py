import copy,json
from pathlib import Path
import pytest
import torch
from models.predictive_world import PredictiveWorld
from training.general_batch import Sources
from training.predictive_losses import loss


@pytest.mark.parametrize('family',['coco','speech','ami','clotho','pamap2','droid'])
def test_every_real_source_has_a_finite_gradient_through_shared_memory(family):
    if not Path('artifacts/general_world/diverse_v1/prepared/dataset_contract.json').exists():pytest.skip('Real prepared source integration runs on Fedora')
    cfg=json.loads(Path('configs/t2/general_transfer.json').read_text());recipe=json.loads(Path('configs/t3/predictive_world.json').read_text())
    batch=Sources(cfg,'train').make(family,22000000,777,'cpu',2)
    net=PredictiveWorld(width=32,layers=1,heads=4,slots=4,memory=24,phone_count=79).train()
    teacher=copy.deepcopy(net).eval().requires_grad_(False)
    objective,metrics=loss(net,teacher,batch,{},recipe,0)
    assert torch.isfinite(objective)
    objective.backward()
    assert net.history.weight_ih_l0.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
