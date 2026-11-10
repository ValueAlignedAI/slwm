import numpy as np
from training.general_readout_eval import paired_effect,physical_actor


def test_transfer_interval_resamples_groups_not_repeated_frames():
    groups=np.repeat(["one","two","three","four","five"],20)
    target=np.zeros((100,7));keep=np.ones_like(target,bool)
    result=paired_effect(np.full_like(target,.5),np.ones_like(target),target,keep,groups,"actor")
    assert result["groups"]==5 and result["passes_10pct_transfer_gate"]
    assert np.allclose(result["gain_95ci"],[.5,.5])


def test_actor_rotation_error_uses_so3_not_coordinate_difference():
    target=np.zeros((2,7));prediction=target.copy();prediction[:,3]=2*np.pi
    assert physical_actor(prediction,target)["rotation_command_error_rad"]<1e-10
