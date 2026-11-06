import torch
import torch.nn.functional as F
from training.general_angular_pilot import angular_regularizer


def test_angular_penalty_catches_raw_variance_satisfied_by_radial_collapse():
    torch.manual_seed(5);direction=F.normalize(torch.randn(128),dim=0)
    radial=torch.arange(1,65.)[:,None]*direction[None]
    collapsed=F.normalize(radial,dim=-1).requires_grad_(True)
    healthy=F.normalize(torch.randn(64,128),dim=-1)
    assert float(angular_regularizer(collapsed))>5*float(angular_regularizer(healthy))
    angular_regularizer(collapsed).backward()
    assert torch.isfinite(collapsed.grad).all()
