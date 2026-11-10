"""Leakage and control checks for the real recorded audiovisual pilot."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from training.t2_prepare_cremad import select_clips, fit_normalizer, log_mel_sequence
from training.t2_real_av import batch_from_arrays, cluster_interval, retrieval_metrics, within_person_permutation
from training.t2_train_latents import T2AudioVisualSLWM


def tiny_model():
    torch.manual_seed(2)
    return T2AudioVisualSLWM(model_cfg={"latent_dim":16,"n_layer":1,"n_head":2,"context_length":16,
            "architecture_flags":{"local_kernel_size":3,"spectral_kernel_size":3,"long_kernel_size":3,"mlp_ratio":2}},
            audio_feature_dim=4,visual_feature_dim=5).eval()


def test_hidden_future_never_affects_any_model_output():
    model = tiny_model()
    batch = batch_from_arrays(torch.randn(3,8,4),torch.randn(3,8,5),4)
    altered = dict(batch)
    for name in ("audio","visual"):
        altered[name+"_features"] = batch[name+"_features"].clone()
        altered[name+"_features"][:,4:] = 10000*torch.randn_like(altered[name+"_features"][:,4:])
    with torch.no_grad():
        before, after = model(batch),model(altered)
    for key in before:
        torch.testing.assert_close(before[key],after[key],atol=0,rtol=0)
    assert before["audio_prediction"].shape == (3,8,4)
    assert before["visual_prediction"].shape == (3,8,5)


def test_retrieval_encodes_cannot_consume_paired_other_stream():
    model = tiny_model()
    a,v = torch.randn(3,8,4),torch.randn(3,8,5)
    with torch.no_grad():
        ar,vr = model.independent_representations(batch_from_arrays(a,v,4))
        ar2,_ = model.independent_representations(batch_from_arrays(a,v.roll(1,0),4))
        _,vr2 = model.independent_representations(batch_from_arrays(a.roll(1,0),v,4))
    torch.testing.assert_close(ar,ar2,atol=0,rtol=0)
    torch.testing.assert_close(vr,vr2,atol=0,rtol=0)
    assert ar.shape == vr.shape == (3,16)


def test_train_only_statistics_are_invariant_to_heldout_people():
    rng = np.random.default_rng(2)
    audio,visual = rng.normal(size=(6,8,4)),rng.normal(size=(6,8,5))
    train = np.array([True,True,True,True,False,False])
    normal = fit_normalizer(audio,visual,train,3)
    audio[~train] += 100000
    visual[~train] -= 100000
    altered = fit_normalizer(audio,visual,train,3)
    for key in normal:
        np.testing.assert_array_equal(normal[key],altered[key])
    assert normal["visual_pca"].shape == (5,3)


def test_actor_splits_and_balanced_selection_are_deterministic():
    names = [f"{actor}_IEO_{emo}_{level}" for actor in range(1001,1007)
             for emo in ("ANG","DIS","FEA","HAP","NEU","SAD") for level in ("LO","HI")]
    cfg = {"seed":3,"actor_count":6,"train_actor_count":3,"validation_actor_count":1,"clips_per_emotion_per_actor":1}
    rows,splits = select_clips(names,cfg)
    assert (rows,splits) == select_clips(names,cfg)
    assert not (set(splits["train"])&set(splits["test"]))
    assert not (set(splits["validation"])&set(splits["test"]))
    assert len(rows) == len({r["sample_id"] for r in rows}) == 36


def test_independent_retrieval_reports_full_candidate_set_and_controls():
    rows = [{"actor":str(i//3)} for i in range(9)]
    result = retrieval_metrics(np.eye(9),np.eye(9),rows)
    assert result["candidate_count"] == 9
    assert result["audio_to_video"]["r1"] == result["video_to_audio"]["r1"] == 1
    assert result["chance_r1"] == 1/9
    assert result["within_person_auc"] == 1
    order = within_person_permutation(rows)
    assert np.all(order != np.arange(9))
    assert all(rows[i]["actor"] == rows[order[i]]["actor"] for i in range(9))
    assert cluster_interval(np.ones(9)*.1,[r["actor"] for r in rows],samples=20) == pytest.approx([.1,.1])


def test_audio_windows_cannot_see_across_future_boundary():
    torchaudio = pytest.importorskip("torchaudio")
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=400,hop_length=160,n_mels=64,center=False)
    waveform = torch.randn(32000)
    altered = waveform.clone()
    altered[16000:] *= 1000
    a,b = log_mel_sequence(waveform,mel,32),log_mel_sequence(altered,mel,32)
    assert a.shape == (32,64)
    torch.testing.assert_close(a[:16],b[:16],atol=0,rtol=0)
    assert not torch.equal(a[16:],b[16:])


def test_residual_forecast_starts_at_observed_persistence_and_absent_mean():
    model = T2AudioVisualSLWM(model_cfg={"latent_dim":16,"n_layer":1,"n_head":2,"context_length":16,
                "prediction_parameterization":"residual_persistence",
                "architecture_flags":{"local_kernel_size":3,"spectral_kernel_size":3,"long_kernel_size":3,"mlp_ratio":2}},
                audio_feature_dim=4,visual_feature_dim=5).eval()
    a,v = torch.randn(3,8,4),torch.randn(3,8,5)
    with torch.no_grad():
        out = model(batch_from_arrays(a,v,4))
        absent = model(batch_from_arrays(a,v,4,"audio_only"))
    torch.testing.assert_close(out["audio_prediction"],a[:,3:4].expand(-1,8,-1),atol=0,rtol=0)
    torch.testing.assert_close(out["visual_prediction"],v[:,3:4].expand(-1,8,-1),atol=0,rtol=0)
    torch.testing.assert_close(absent["visual_prediction"],torch.zeros_like(v),atol=0,rtol=0)


def test_fresh_followup_excludes_every_discovery_person():
    cfg = json.loads(Path("configs/t2/cremad_prepare_fresh_residual.json").read_text())
    discovery = json.loads(Path("configs/t2/cremad_prepare_pilot.json").read_text())
    # Reconstruct the discovery actor selection with all 91 source IDs, without
    # using media or any test outcomes.
    names = [f"{i}_IEO_{emo}_XX" for i in range(1001,1092)
             for emo in ("ANG","DIS","FEA","HAP","NEU","SAD")]
    _, old = select_clips(names,discovery)
    assert not set(cfg["actor_splits"]["test"]) & set(sum(old.values(),[]))
    assert cfg["actor_splits"]["train"] == old["train"]
    assert cfg["actor_splits"]["validation"] == old["validation"]


def test_linear_control_predicts_new_inputs_with_documented_shapes():
    from evals.t2_feature_controls import fit_linear,predict_linear
    rng = np.random.default_rng(8)
    x = rng.normal(size=(20,3))
    truth = rng.normal(size=(3,2))
    weights = fit_linear(x,x@truth+.3,regularization=1e-8)
    heldout = rng.normal(size=(4,3))
    predicted = predict_linear(weights,heldout)
    assert weights["coefficient"].shape == (3,2)
    assert predicted.shape == (4,2)
    np.testing.assert_allclose(predicted,heldout@truth+.3,atol=1e-6)


def test_reprepare_preserves_checkpoint_bound_corpus_and_detects_changes(tmp_path):
    from training.t2_prepare_cremad import prepare,sha256
    corpus = tmp_path/"prepared"
    corpus.mkdir()
    artifact = corpus/"normalizer.npz"
    artifact.write_bytes(b"fixed checkpoint-bound preprocessing")
    cfg = {"output_dir":str(corpus),"seed":2}
    card = {"config":cfg,"files":{"normalizer.npz":sha256(artifact)}}
    original = json.dumps(card)
    (corpus/"dataset_card.json").write_text(original)
    path = tmp_path/"config.json"
    path.write_text(json.dumps(cfg))
    assert prepare(str(path)) == card
    assert (corpus/"dataset_card.json").read_text() == original
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError,match="Prepared artifact changed"):
        prepare(str(path))
