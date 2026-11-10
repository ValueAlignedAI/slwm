"""Support-level isolation, preservation and matched-capacity branch contracts."""
import io
import zipfile
import numpy as np
import torch
import torchaudio
from models.slwm_restoration import RestorationCodec
from models.slwm_video_repair import RepairBranch,gap_masks,VideoRepairSystem
from exploration.video_repair_audio import constrained_audio,write_pcm


def make_codec():
    old={k:np.zeros(64,np.float32) for k in ("audio_mean","audio_delta_mean")}
    old.update({k:np.ones(64,np.float32) for k in ("audio_scale","audio_delta_scale")})
    return RestorationCodec(old,{"mean":np.zeros(64,np.float32),"scale":np.ones(64,np.float32)})


def test_fft_support_mask_removes_hidden_sample_boundary_leak():
    torch.set_num_threads(1); torch.manual_seed(4); x=torch.randn(32*640); changed=x.clone(); changed[12*640:20*640]=torch.randn(8*640)*100
    transform=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=1024,hop_length=160,n_mels=64,power=2.)
    y=transform(x).clamp_min(1e-8).log().T[:128].reshape(1,32,256); z=transform(changed).clamp_min(1e-8).log().T[:128].reshape(1,32,256)
    masks=gap_masks(torch.tensor([12]),torch.tensor([8])); assert masks["target"].sum()==32 and masks["unknown"].sum()==39
    assert not masks["visible"][0,20] and masks["visible"][0,11] and masks["visible"][0,21]
    codec=make_codec(); a,b=codec.observe(y,masks["visible"]),codec.observe(z,masks["visible"])
    for key in ("audio","base"): torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    naive=torch.ones(1,32,dtype=torch.bool); naive[:,12:20]=False
    assert not torch.equal(codec.observe(y,naive)["audio"],codec.observe(z,naive)["audio"])


def inputs():
    torch.set_num_threads(1); b,t=1,32
    return (torch.randn(b,t,256),torch.randn(b,4*t,64),torch.randint(0,255,(b,t,64,96),dtype=torch.uint8),torch.randn(b,t,160),
        torch.ones(b,t,dtype=torch.bool),torch.ones(b,4*t,dtype=torch.bool),torch.nn.functional.normalize(torch.randn(b,t,96),dim=-1),torch.ones(b,dtype=torch.bool))


def test_video_null_fallback_and_support_are_exact_after_nonzero_head():
    branch=RepairBranch("video",61).eval()
    with torch.no_grad(): branch.head.output.weight.normal_(std=.01)
    args=list(inputs()); args[-1]=torch.zeros(1,dtype=torch.bool); out=branch(*args)
    torch.testing.assert_close(out["prediction"],args[1],atol=0,rtol=0)
    args[-1]=torch.ones(1,dtype=torch.bool); args[5][:,:20]=False; out=branch(*args)
    assert out["prediction"].shape==(1,128,64)
    torch.testing.assert_close(out["prediction"][:,:20],args[1][:,:20],atol=0,rtol=0)


def test_audio_control_cannot_read_video_and_common_weights_are_matched():
    audio,video=RepairBranch("audio",61).eval(),RepairBranch("video",61).eval()
    for name in ("head","context","query","attend","fuse","temporal"):
        for key,value in getattr(audio,name).state_dict().items(): torch.testing.assert_close(value,getattr(video,name).state_dict()[key],atol=0,rtol=0)
    with torch.no_grad(): audio.head.output.weight.normal_(std=.01)
    args=list(inputs()); a=audio(*args)["prediction"]; args[2]=255-args[2]; args[3]*=1000
    torch.testing.assert_close(a,audio(*args)["prediction"],atol=0,rtol=0)


def test_real_visual_frontend_gets_gradients_and_initializes_at_audio_anchor():
    branch=RepairBranch("video",61); args=inputs(); out=branch(*args)
    torch.testing.assert_close(out["prediction"],args[1],atol=0,rtol=0)
    with torch.no_grad(): branch.head.output.weight.normal_(std=.01)
    branch(*args)["prediction"].square().mean().backward(); assert branch.front[0].weight.grad.abs().sum()>0


def test_observed_waveform_is_preserved_and_hidden_samples_are_never_used(tmp_path):
    torch.set_num_threads(1); rng=np.random.default_rng(3); x=(rng.integers(-1000,1000,4096).astype(np.float32)/32768)
    missing=np.zeros(len(x),bool); missing[1600:2400]=True; changed=x.copy(); changed[missing]=rng.normal(size=missing.sum())
    codec={"mean":np.zeros(64,np.float32)-5,"scale":np.ones(64,np.float32)}; mel=np.zeros((26,64),np.float32)
    a=constrained_audio(mel,x,missing,codec,iterations=2); b=constrained_audio(mel,changed,missing,codec,iterations=2)
    np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(a[~missing],x[~missing])
    path=tmp_path/"repair.wav"; write_pcm(path,a)
    import wave
    with wave.open(str(path),"rb") as f: samples=np.frombuffer(f.readframes(f.getnframes()),"<i2")
    np.testing.assert_array_equal(samples[~missing],np.rint(x[~missing]*32768).astype(np.int16))


def test_public_zip_range_reader_checks_byte_ranges(monkeypatch):
    from training.video_repair_data import VerifiedRanges
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,"w") as z: z.writestr("example.txt","real payload")
    payload=buffer.getvalue()
    class Response:
        def __init__(self,start,end): self.data=payload[start:end+1]; self.headers={"Content-Range":f"bytes {start}-{end}/{len(payload)}"}
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def read(self): return self.data
    def request(req,timeout):
        interval=req.headers["Range"].removeprefix("bytes="); start,end=map(int,interval.split("-")); return Response(start,end)
    monkeypatch.setattr("training.video_repair_data.urllib.request.urlopen",request)
    with zipfile.ZipFile(VerifiedRanges("https://example.invalid/archive?download=1",len(payload))) as archive: assert archive.read("example.txt")==b"real payload"


def test_full_recording_erasure_precedes_every_window_and_transform():
    from exploration.video_repair_infer import predict_mel,support_masks,repair_windows
    torch.set_num_threads(1); rng=np.random.default_rng(9); sound=rng.normal(0,.03,64*640).astype(np.float32)
    missing=np.zeros(len(sound),bool); missing[28*640:36*640]=True; changed=sound.copy(); changed[missing]=100
    seen,unknown=support_masks(missing,64); assert repair_windows(unknown,64)==[16]
    class Model:
        def __call__(self,obs,mouth,video,unknown,present): return {"prediction":obs["base"]+.1*unknown[...,None],"gate":torch.ones(1,32)}
    mouth=np.zeros((64,64,96),np.uint8); stats={"mean":np.zeros(64,np.float32),"scale":np.ones(64,np.float32)}
    a,_=predict_mel(Model(),make_codec(),stats,sound,missing,mouth); b,_=predict_mel(Model(),make_codec(),stats,changed,missing,mouth)
    np.testing.assert_array_equal(a,b)


def test_terminal_subframe_padding_matches_existing_codec_without_hidden_evidence():
    from exploration.video_repair_infer import predict_mel,pad_subframes
    torch.set_num_threads(1); sound=np.random.default_rng(5).normal(0,.03,64*640-320).astype(np.float32)
    hidden=np.zeros(len(sound),bool); hidden[28*640:36*640]=True; mouth=np.zeros((64,64,96),np.uint8)
    stats={"mean":np.zeros(64,np.float32),"scale":np.ones(64,np.float32)}
    class Model:
        def __call__(self,obs,mouth,video,unknown,present): return {"prediction":obs["base"],"gate":torch.ones(1,32)}
    prediction,_=predict_mel(Model(),make_codec(),stats,sound,hidden,mouth)
    assert prediction.shape==(256,64)
    np.testing.assert_array_equal(prediction[-1],prediction[-2])
    import pytest
    with pytest.raises(ValueError): pad_subframes(np.zeros((120,64)),32)
