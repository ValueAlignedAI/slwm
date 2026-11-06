import numpy as np
from training.general_consequences import physical_state,state_errors


def test_pose_metric_roundtrip_and_invalid_rotation_penalty():
    poses=np.zeros((2,3,7));poses[...,3:6]=[.2,-.1,3.13]
    result=state_errors(physical_state(poses),poses)
    assert np.max(result["rotation_rad"])<1e-7
    assert np.max(result["translation_m"])==0
    broken=physical_state(poses);broken[...,3:9]=0
    assert np.allclose(state_errors(broken,poses)["rotation_rad"],np.pi)
