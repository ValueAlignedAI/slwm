"""Information isolation, real fallback, calibration and scale-data contracts."""
import json
import tarfile
import io
import numpy as np
import pytest
import torch
from models.slwm_content_sync import SpeechContentBase,ContentCandidate,ContentSyncSystem
from models.slwm_restoration import RestorationCodec
from models.slwm_video_repair import gap_masks


def fixture():
    torch.set_num_threads(1); torch.manual_seed(31)
    cfg={"model":{"width":256,"layers":1,"heads":8,"packing":"aligned","processor":"signal"}}; run={"arm":"signal","seed":71}
    old={k:np.zeros(64,np.float32) for k in ("audio_mean","audio_delta_mean")}; old.update({k:np.ones(64,np.float32) for k in ("audio_scale","audio_delta_scale")})
    codec=RestorationCodec(old,{"mean":np.zeros(64,np.float32),"scale":np.ones(64,np.float32)})
    masks=gap_masks(torch.tensor([12,12]),torch.tensor([8,8])); obs=codec.observe(torch.randn(2,32,256),masks["visible"]); mouth=torch.randint(0,255,(2,32,64,96),dtype=torch.uint8)
    return cfg,run,obs,mouth,masks


def test_primary_content_path_ignores_labels_and_only_explicit_text_conditions():
    cfg,run,obs,mouth,masks=fixture(); model=SpeechContentBase(cfg,run,52,71).eval()
    with torch.no_grad(): model.expert.head.output.weight.normal_(std=.02); model.text_mix.weight.normal_(std=.05)
    with torch.inference_mode():
        first=model(obs)["prediction"]; obs["target"]=torch.randn_like(obs["target"])*100; obs["word_labels"]=torch.randint(0,52,(2,32)); second=model(obs)["prediction"]
        torch.testing.assert_close(first,second,atol=0,rtol=0)
        supplied=model(obs,torch.ones(2,32,dtype=torch.long))["prediction"]; assert not torch.equal(first,supplied)


def test_observed_only_sync_cannot_read_erased_audio_frames():
    cfg,run,obs,mouth,masks=fixture(); branch=ContentCandidate("video",52,71).eval(); hint=branch.hint(torch.randn(2,32,256),mouth)
    changed=obs["audio"].clone(); changed[~obs["visible"]]=torch.randn_like(changed[~obs["visible"]])*100
    a,b=branch.synchronize(obs["audio"],hint,obs["visible"]),branch.synchronize(changed,hint,obs["visible"])
    for key in ("quality","offset_logits","match_logits"): torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)


def test_rejection_returns_strong_audio_candidate_including_unit_state():
    cfg,run,obs,mouth,masks=fixture(); model=ContentSyncSystem(cfg,run,52,71,"video").eval()
    with torch.no_grad():
        model.branch.head.output.weight.normal_(std=.02); model.branch.delta_unit.weight.normal_(std=.02); model.audio_candidate.head.output.weight.normal_(std=.01)
    with torch.inference_mode():
        base=model.base(obs); expected=model.audio_candidate(model.base,obs,base["field"],base["unit_logits"],base["prediction"],mouth,masks["unknown"])
        rejected=model(obs,mouth,masks["unknown"],torch.ones(2,dtype=torch.bool),1.1)
        absent=model(obs,255-mouth,masks["unknown"],torch.zeros(2,dtype=torch.bool),0.)
        for result in (rejected,absent):
            for key in ("prediction","unit_logits","candidate_field"): torch.testing.assert_close(result[key],expected[key],atol=0,rtol=0)
        assert not torch.equal(rejected["raw_prediction"],expected["prediction"])
        torch.testing.assert_close(rejected["raw_prediction"][~masks["unknown"]],base["prediction"][~masks["unknown"]],atol=0,rtol=0)


def test_audio_candidate_ignores_pixels_and_shared_weights_match():
    cfg,run,obs,mouth,masks=fixture(); a=ContentCandidate("audio",52,71).eval(); v=ContentCandidate("video",52,71).eval()
    for name in ("head","context","query","attend","fuse","delta_field","delta_unit"):
        for key,value in getattr(a,name).state_dict().items(): torch.testing.assert_close(value,getattr(v,name).state_dict()[key],atol=0,rtol=0)
    base=SpeechContentBase(cfg,run,52,71).eval(); values=base(obs)
    with torch.no_grad(): a.head.output.weight.normal_(std=.02)
    first=a(base,obs,values["field"],values["unit_logits"],values["prediction"],mouth,masks["unknown"])["prediction"]
    second=a(base,obs,values["field"],values["unit_logits"],values["prediction"],255-mouth,masks["unknown"])["prediction"]
    torch.testing.assert_close(first,second,atol=0,rtol=0)


def test_calibration_caps_each_direction_and_never_hides_coverage_failure():
    from training.content_sync_train import rejection_threshold
    cfg={"threshold_grid_points":1001,"max_shift_acceptance":.05,"required_correct_coverage":.5}
    accepted=rejection_threshold(np.full(100,.9),np.stack((np.full(100,.2),np.full(100,.4))),cfg)
    assert accepted["threshold"]>.4 and accepted["coverage_gate_passes"] and accepted["shift_acceptance"]==0
    rejected=rejection_threshold(np.full(100,.1),np.full((2,100),.9),cfg)
    assert rejected["aligned_coverage"]==0 and not rejected["coverage_gate_passes"]


def test_token_windows_are_unique_and_common_scale_initialization_is_causal(tmp_path):
    from training.scale_text_train import UniqueWindows,initialize
    tokens=np.arange(4097,dtype="<u2"); path=tmp_path/"tokens.bin"; tokens.tofile(path); stream=UniqueWindows(path,32,81)
    x,y=stream.batch(0,stream.blocks,"cpu"); assert len(torch.unique(x[:,0]))==stream.blocks; torch.testing.assert_close(y,x+1)
    with pytest.raises(ValueError): stream.batch(stream.blocks-1,2,"cpu")
    cfg={"model":{"vocab_size":128,"context_length":64,"n_layer":1,"n_embd":32,"n_head":4,"dropout":0.}}
    g,s=initialize(cfg,"gpt2",81).eval(),initialize(cfg,"slwm",81).eval()
    torch.testing.assert_close(g.token_embedding.weight,s.token_embedding.weight,atol=0,rtol=0)
    ids=torch.randint(0,128,(2,32)); altered=ids.clone(); altered[:,16:]=torch.randint(0,128,(2,16))
    for model in (g,s): torch.testing.assert_close(model(ids)[:,:16],model(altered)[:,:16],atol=1e-6,rtol=1e-6)


def test_domains_and_exact_documents_are_isolated():
    pytest.importorskip("pyarrow"); pytest.importorskip("publicsuffix2"); pytest.importorskip("tiktoken")
    from training.prepare_scale_text import document_keys
    a=document_keys("Some   text here","https://one.example.co.uk/a")
    b=document_keys("SOME text HERE","https://two.example.co.uk/b")
    assert a[0]==b[0] and a[1]==b[1]=="example.co.uk" and a[3]==b[3]


def test_grid_alignment_clock_is_25000_samples_and_split_uses_words(tmp_path):
    from training.grid_content_data import alignments
    payload=b"0 2500 sil\n2500 25000 set\n25000 75000 soon\n"; path=tmp_path/"align.tar"
    with tarfile.open(path,"w") as t:
        info=tarfile.TarInfo("s1/example.align"); info.size=len(payload); t.addfile(info,io.BytesIO(payload))
    row=alignments(path)["example"]; assert row["spans"][-1][1]==3. and row["transcript"]=="set soon"


def test_wave_inference_erases_globally_and_rejects_uncovered_gaps():
    from exploration.content_sync_infer import predict_sequence
    cfg,run,obs,mouth,masks=fixture()
    old={k:np.zeros(64,np.float32) for k in ("audio_mean","audio_delta_mean")}; old.update({k:np.ones(64,np.float32) for k in ("audio_scale","audio_delta_scale")})
    stats={"mean":np.zeros(64,np.float32),"scale":np.ones(64,np.float32)}; codec=RestorationCodec(old,stats)
    class Model:
        def __call__(self,obs,text=None): return {"prediction":obs["base"],"unit_logits":torch.zeros(1,32,52)}
    sound=np.random.default_rng(6).normal(0,.02,64*640).astype(np.float32); hidden=np.zeros(len(sound),bool); hidden[28*640:36*640]=True
    pixels=np.zeros((64,64,96),np.uint8); changed=sound.copy(); changed[hidden]=100
    a,_=predict_sequence(Model(),"base",codec,stats,sound,hidden,pixels); b,_=predict_sequence(Model(),"base",codec,stats,changed,hidden,pixels)
    np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match="does not cover"): predict_sequence(Model(),"base",codec,stats,sound,hidden,pixels[:32])
