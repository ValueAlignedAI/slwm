"""Conditioning, time transforms, and real seed controls for T2-G1."""
import numpy as np
import pytest
import torch

from models.slwm_generation import MultirateAudioHead,SeedMotionHead,pad_condition,unpack_mel,visual_loss,visual_errors
from training.t2_prepare_generation import packetize
from training.t2_generation_train import condition,crop
from models.slwm_generation_residual import SignalResidualAudioHead


def test_chronological_subframes_and_edge_padding():
    mel=np.arange(7*64,dtype=np.float32).reshape(7,64)
    packed,padded=packetize(mel,2)
    assert packed.shape==(2,256) and padded==1
    restored=unpack_mel(torch.from_numpy(packed)[None])[0].numpy()
    np.testing.assert_array_equal(restored[:7],mel)
    np.testing.assert_array_equal(restored[-1],mel[-1])
    with pytest.raises(ValueError): packetize(mel,3)


def test_pad_condition_preserves_every_observation():
    x=torch.randn(2,4,128); padded=pad_condition(x)
    torch.testing.assert_close(padded[...,:128],x); assert padded[...,128:].count_nonzero()==0
    with pytest.raises(ValueError): pad_condition(torch.zeros(1,1,257))


@pytest.mark.parametrize("warp",[True,False])
def test_seed_initialization_and_target_exclusion(warp):
    torch.set_num_threads(1); model=SeedMotionHead(warp); seed=torch.rand(2,32,48)
    output=model(torch.randn(2,5,256),seed)
    assert output.shape==(2,5,32,48)
    torch.testing.assert_close(output,seed[:,None].expand_as(output),atol=4e-6,rtol=1e-5)
    # Targets enter only the objective; no future image is a model argument.
    target=torch.rand_like(output); loss=visual_loss(output,target); loss.backward()
    assert torch.isfinite(loss) and model.output.weight.grad.abs().sum()>0
    errors=visual_errors(seed[:,None].expand_as(target),target)
    assert errors["motion_energy"].count_nonzero()==0
    assert all(v.shape==(2,) for v in errors.values())


def test_audio_head_shapes_and_gradient():
    torch.set_num_threads(1); model=MultirateAudioHead(); x=torch.randn(2,7,160)
    y=model(x); assert y.shape==(2,28,64)
    y.square().mean().backward(); assert model.output.weight.grad.abs().sum()>0


def test_wrong_condition_never_changes_seed_target_and_null_is_zero():
    data={"field_audio":torch.arange(2*40*256).reshape(2,40,256).half(),
          "mouth":torch.zeros(2,40,32,48,dtype=torch.uint8),"wrong":np.array([1,0]),"lengths":np.array([40,40])}
    ids=np.array([0]); starts=np.array([2]); run={"input":"field_audio"}
    correct=condition(data,run,ids,starts,8); wrong=condition(data,run,ids,starts,8,"wrong")
    torch.testing.assert_close(correct,crop(data,"field_audio",ids,starts,8))
    torch.testing.assert_close(wrong,crop(data,"field_audio",np.array([1]),np.array([16]),8))
    assert condition(data,run,ids,starts,8,"null").count_nonzero()==0
    assert condition(data,{"input":"time_only"},ids,starts,8).count_nonzero()==0
    training_run={"input":"field_audio","shuffled":True}
    torch.testing.assert_close(condition(data,training_run,ids,starts,8),wrong)
    # Shuffling belongs to training; frozen inference must explicitly disable it.
    torch.testing.assert_close(condition(data,{**training_run,"shuffled":False},ids,starts,8),correct)


def test_seed_frame_not_rewarded_in_visual_metric():
    target=torch.zeros(1,4,32,48); pred=target.clone(); pred[:,0]=1
    errors=visual_errors(pred,target)
    assert errors["roi_mse"].item()==0 and errors["pixel_mse"].item()==0
    assert errors["temporal_mse"].item()>0


def test_residual_path_preserves_unseen_signal_level_and_cannot_replace_it():
    torch.set_num_threads(1)
    old={"audio_mean":np.zeros(64,np.float32),"audio_scale":np.ones(64,np.float32)}
    new={"mean":np.zeros(64,np.float32),"scale":np.ones(64,np.float32)}
    model=SignalResidualAudioHead(old,new)
    observed=torch.full((2,4,128),8.); condition=torch.randn(2,4,256)
    output=model(condition,observed); torch.testing.assert_close(output,torch.full((2,16,64),8.))
    with torch.no_grad(): model.correction.output.bias.fill_(100.)
    changed=model(condition,observed)
    assert (changed-output).abs().max()<=.250001
    ramp=torch.arange(4).float()[None,:,None].expand(1,4,128)
    torch.testing.assert_close(model.baseline(ramp)[0,:,0],torch.tensor([0,.25,.5,.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3,3,3]))
