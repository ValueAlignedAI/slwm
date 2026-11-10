"""Visibility, shape and synchronization checks for the motion architecture."""
import json
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip("torch")
from models.slwm_motion import MotionSLWM
from training.t2_prepare_motion import encode,fit_pca
from training.t2_motion_train import crop,masked_inputs,shuffled_indices,offset_baseline


def model(packing="aligned"):
    torch.manual_seed(3)
    return MotionSLWM({"width":32,"layers":1,"heads":4,"packing":packing,"processor":"signal"}).eval()


@pytest.mark.parametrize("packing",["aligned","concatenated"])
def test_missing_stream_and_independent_encodes_never_read_partner(packing):
    m=model(packing); a,v=torch.randn(3,12,128),torch.randn(3,12,160)
    am,vm=masked_inputs(a,v,"audio")
    with torch.no_grad():
        before=m(a,v,am,vm); after=m(a,torch.randn_like(v)*1e4,am,vm)
        ar,vr=m.independent(a,v)
        ar2,_=m.independent(a,v.roll(1,0)); _,vr2=m.independent(a.roll(1,0),v)
    assert before["audio"].shape==(3,12,128)
    assert before["video"].shape==(3,12,160)
    for name in ("audio","video"): torch.testing.assert_close(before[name],after[name],atol=0,rtol=0)
    torch.testing.assert_close(ar,ar2,atol=0,rtol=0); torch.testing.assert_close(vr,vr2,atol=0,rtol=0)
    assert ar.shape==vr.shape==(3,12,96)


def test_null_input_is_finite_and_checkpoint_reload_is_exact(tmp_path):
    m=model(); a,v=torch.randn(2,10,128),torch.randn(2,10,160); am,vm=masked_inputs(a,v,"null")
    with torch.no_grad(): expected=m(a,v,am,vm)
    assert all(torch.isfinite(value).all() for value in expected.values())
    torch.save(m.state_dict(),tmp_path/"model.pt")
    reloaded=model(); reloaded.load_state_dict(torch.load(tmp_path/"model.pt",weights_only=True))
    with torch.no_grad(): actual=reloaded(a,v,am,vm)
    for name in expected: torch.testing.assert_close(expected[name],actual[name],atol=0,rtol=0)


def test_physical_windows_are_exact_and_never_use_padding():
    a=torch.arange(2*80*128).reshape(2,80,128).float(); v=torch.randn(2,80,160)
    aa,vv=crop({"audio":a,"video":v},np.array([1,0]),np.array([7,12]),32)
    assert aa.shape==(2,32,128) and vv.shape==(2,32,160)
    torch.testing.assert_close(aa[0],a[1,7:39],atol=0,rtol=0)
    torch.testing.assert_close(aa[1],a[0,12:44],atol=0,rtol=0)


def test_fresh_confirmation_people_never_enter_previous_data():
    c=json.loads(Path("configs/t2/cremad_motion_prepare.json").read_text())
    old=json.loads(Path("configs/t2/cremad_prepare_fresh_residual.json").read_text())
    discovery_test={"1090","1049","1074","1070","1006","1042","1045"}
    assert not set(c["actor_splits"]["test"]) & (set(sum(old["actor_splits"].values(),[]))|discovery_test)
    assert not set(c["actor_splits"]["train"]) & (set(old["actor_splits"]["test"])|discovery_test)
    assert len(c["actor_splits"]["train"])==62


def test_train_only_pca_and_codec_roundtrip_shapes():
    rng=np.random.default_rng(2); x=rng.normal(size=(40,20)).astype(np.float32)
    mean,basis,scale=fit_pca(x,8)
    assert mean.shape==(20,) and basis.shape==(20,8) and scale.shape==(8,)
    np.testing.assert_allclose(basis.T@basis,np.eye(8),atol=1e-6)
    restored=((x-mean)@basis)@basis.T+mean
    assert np.mean((x-restored)**2)<np.mean((x-mean)**2)


def test_rendered_mouth_is_derived_from_predictions_and_explicit_prior():
    from exploration.motion_outputs import inverse_appearance,inverse_motion
    appearance=np.zeros((1536,96),np.float32); appearance[:96]=np.eye(96)
    motion=np.zeros((1536,64),np.float32); motion[:64]=np.eye(64)
    codec={"appearance_mean":np.ones(1536,np.float32)*.25,"appearance_basis":appearance,"appearance_scale":np.ones(96,np.float32),
           "motion_mean":np.zeros(1536,np.float32),"motion_basis":motion,"motion_scale":np.ones(64,np.float32)}
    output=inverse_appearance(np.zeros((4,96),np.float32),codec)
    assert output.shape==(4,32,48)
    np.testing.assert_array_equal(output,np.ones_like(output)*.25)
    delta=np.zeros((4,64),np.float32); delta[:,0]=.01
    sequence=inverse_motion(delta,codec)
    np.testing.assert_allclose(sequence[:,0,0],[.26,.27,.28,.29],atol=1e-6)


def test_sequence_generation_cannot_use_unprovided_video():
    from exploration.motion_outputs import predict_sequence
    m=model(); a=torch.randn(40,128); v=torch.randn(40,160)
    left=predict_sequence(m,a,v,"audio")
    right=predict_sequence(m,a,v*10000,"audio")
    assert left["audio"].shape==(40,128) and left["video"].shape==(40,160)
    np.testing.assert_array_equal(left["video"],right["video"])
    np.testing.assert_array_equal(left["audio"],right["audio"])


def test_inverse_audio_codec_produces_finite_waveform():
    pytest.importorskip("torchaudio")
    from exploration.motion_outputs import invert_mel
    codec={"audio_scale":np.ones(64,np.float32),"audio_mean":np.zeros(64,np.float32)}
    sound=invert_mel(np.zeros((8,64),np.float32),codec,iterations=2)
    assert sound.ndim==1 and len(sound)>0 and np.isfinite(sound).all()


def test_streaming_decoder_fit_recovers_linear_map():
    from exploration.motion_linear_decoders import solve_stats
    rng=np.random.default_rng(77); x=rng.normal(size=(40,5)); truth=rng.normal(size=(5,3)); y=x@truth+.2
    stats={"n":len(x),"x":x.sum(0),"y":y.sum(0),"xx":x.T@x,"xy":x.T@y}
    weights=solve_stats(stats,regularization=1e-8)
    assert weights["coefficient"].shape==(5,3) and weights["intercept"].shape==(3,)
    np.testing.assert_allclose(x@weights["coefficient"]+weights["intercept"],y,atol=1e-6)


def test_frozen_probe_shapes_and_closed_set_predictions():
    from exploration.motion_outputs import probe_features,fit_probe,probe_scores
    features=probe_features(torch.randn(12,32,96))
    assert features.shape==(12,480)
    labels=np.arange(12)%3; weights=fit_probe(features,labels,3)
    scores=probe_scores(weights,features)
    assert weights["coefficient"].shape==(480,3) and scores.shape==(12,3)
    assert np.isfinite(scores).all()


def test_speech_diagnostic_counts_insertions_and_normalizes_apostrophes():
    from evals.motion_audio_intelligibility import word_error
    assert word_error("It's eleven o'clock.","its eleven oclock")==0
    assert word_error("hello","one two three")>1
