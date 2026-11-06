import numpy as np
import pytest
from training.predictive_robot import native_clock


def test_native_clock_requires_pose_alignment_and_preserves_nanosecond_differences(tmp_path):
    h5py=pytest.importorskip('h5py');path=tmp_path/'raw.h5';pose=np.zeros((3,6),np.float32)
    with h5py.File(path,'w') as f:
        f['observation/robot_state/cartesian_position']=pose
        f['observation/timestamp/robot_state/robot_timestamp_seconds']=np.full(3,1690000000,np.int64)
        f['observation/timestamp/robot_state/robot_timestamp_nanos']=np.array([100,40000200,80000300],np.int64)
    clock,audit=native_clock(path,pose)
    assert audit['verified'] and np.allclose(clock,[0,.0400001,.0800002],atol=1e-10,rtol=0)
    assert native_clock(path,pose+1)[0] is None
    short,audit=native_clock(path,pose[:2])
    assert audit['verified'] and audit['raw_tail_frames_not_in_rlds']==1
    assert np.array_equal(short,clock[:2])
    assert native_clock(path,pose[:1])[0] is None
