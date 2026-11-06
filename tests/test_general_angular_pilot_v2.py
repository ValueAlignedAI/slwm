import torch
import torch.nn.functional as F
from training.general_angular_pilot_v2 import angular_regularizer,geometry


def test_geometry_stays_float32_inside_mixed_precision_context():
    values=F.normalize(torch.randn(32,128),dim=-1).requires_grad_(True)
    with torch.autocast("cpu",dtype=torch.bfloat16):
        penalty=angular_regularizer(values)
        result=geometry(values)
    penalty.backward()
    assert result["covariance_participation_ratio"]>8
    assert penalty.dtype==torch.float32 and torch.isfinite(values.grad).all()
