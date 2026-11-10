import numpy as np
from training.predictive_features import fit_projection
from training.predictive_readouts import readout_inputs,Readout


def test_state_actor_cannot_use_raw_proprioception_and_controls_keep_equal_capacity(tmp_path):
    rng=np.random.default_rng(441);features=rng.normal(size=(160,160)).astype(np.float32)
    np.save(tmp_path/'target_common.npy',np.zeros((160,17),np.float32))
    state={**fit_projection(features,128),'common_mean':np.zeros(0,np.float32),'common_scale':np.ones(0,np.float32)}
    control={**fit_projection(features,111),'common_mean':np.zeros(17,np.float32),'common_scale':np.ones(17,np.float32)}
    first=readout_inputs(tmp_path,features,state);raw=readout_inputs(tmp_path,features,control)
    np.save(tmp_path/'target_common.npy',np.full((160,17),999,np.float32))
    assert np.array_equal(first,readout_inputs(tmp_path,features,state))
    assert not np.array_equal(raw,readout_inputs(tmp_path,features,control))
    assert first.shape==raw.shape==(160,128)
    assert sum(p.numel() for p in Readout(first.shape[1],7).parameters())==sum(p.numel() for p in Readout(raw.shape[1],7).parameters())
