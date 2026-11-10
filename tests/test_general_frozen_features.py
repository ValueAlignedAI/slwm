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


def test_pooled_control_matches_the_alignment_projection_input():
    import torch
    from models.slwm_general import GeneralWorld
    from training.general_frozen_features import frozen_features
    model=GeneralWorld(width=32,layers=1,heads=4).eval()
    obs={"time":torch.arange(4.)[None].repeat(2,1),"time_unit":torch.zeros(2,dtype=torch.long),"valid":torch.ones(2,4,dtype=torch.bool),"numeric":torch.randn(2,4,64),"numeric_present":torch.ones(2,4,64,dtype=torch.bool)}
    with torch.no_grad():
        values=frozen_features(model,obs,"mhealth")
        recovered=torch.nn.functional.normalize(model.embedding(torch.from_numpy(values["pooled"])),dim=-1).numpy()
    assert np.allclose(recovered,values["aligned"],atol=1e-6)
