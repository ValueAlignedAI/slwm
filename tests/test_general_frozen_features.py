import numpy as np
from training.general_frozen_features import fit_projection, project


def test_training_projection_preserves_low_rank_signal_without_test_statistics():
    rng=np.random.default_rng(2)
    train=(rng.normal(size=(80,2))@rng.normal(size=(2,16))+3).astype(np.float32)
    fitted=fit_projection(train,4)
    reconstructed=(project(train,fitted)@fitted["basis"].T)*fitted["scale"]+fitted["mean"]
    assert np.mean((reconstructed-train)**2)<1e-8
    original=fitted["mean"].copy()
    assert project(np.full((3,16),1000,np.float32),fitted).shape==(3,4)
    assert np.array_equal(fitted["mean"],original)
