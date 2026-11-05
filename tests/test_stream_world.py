import numpy as np
import pytest
torch=pytest.importorskip("torch")
pytest.importorskip("vocos")
pytest.importorskip("cmudict")
pytest.importorskip("cv2")
pytest.importorskip("soundfile")
from models.slwm_stream import StreamWorld
from training.stream_codec import observed_mel,support_visibility
from training.broad_data import physical_mel
from vocos.feature_extractors import MelSpectrogramFeatures


def inputs():
    torch.manual_seed(11); b,t=2,24
    return [torch.randn(b,t,400),torch.randn(b,t,64),torch.randint(0,41,(b,t)),torch.ones(b,t,dtype=torch.bool),torch.zeros(b,t,dtype=torch.bool),torch.zeros(b,t,dtype=torch.bool),torch.ones(b,t,dtype=torch.bool),torch.zeros(b,64)]


@pytest.mark.parametrize("variant",["slwm","transformer","late_fusion"])
def test_stream_chunks_equal_batched_and_no_future_chunk(variant):
    torch.set_num_threads(1); model=StreamWorld(variant,width=64,layers=6,heads=4,memory=4,chunk=8,seed=11).eval(); args=inputs()
    with torch.inference_mode():
        full=model(*args); memory=None; outputs=[]
        for start in range(0,24,8):
            out=model(*(x[:,start:start+8] for x in args[:7]),args[7],memory); memory=out["memory"]; outputs.append(out["field"])
        torch.testing.assert_close(full["field"],torch.cat(outputs,1),rtol=1e-5,atol=1e-5); torch.testing.assert_close(full["memory"],memory,rtol=1e-5,atol=1e-5)
        changed=[x.clone() for x in args]; changed[0][:,16:]*=100
        alternate=model(*changed); torch.testing.assert_close(full["field"][:,:16],alternate["field"][:,:16],atol=0,rtol=0); torch.testing.assert_close(full["future"][:,:2],alternate["future"][:,:2],atol=0,rtol=0)
        assert full["mel"].shape==(2,24,400) and full["video"].shape==(2,24,64) and full["phone_logits"].shape==(2,24,41)


def test_hidden_feature_and_unprovided_text_invariance():
    model=StreamWorld(width=64,layers=6,heads=4,memory=4,chunk=8).eval(); args=inputs(); args[3][:,3:6]=False
    with torch.inference_mode():
        first=model(*args); changed=[x.clone() for x in args]; changed[0][:,3:6]=1e6; changed[1]*=999; changed[2]=torch.randint(0,41,changed[2].shape); second=model(*changed)
        for key in ("mel","field","video","phone_logits","risk","memory"): torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    assert model.phone_head.weight is model.text.weight


def test_common_initializers_and_memory_is_working_state():
    hybrid=StreamWorld("slwm",width=64,layers=6,heads=4,memory=4,chunk=8,seed=11); baseline=StreamWorld("transformer",width=64,layers=6,heads=4,memory=4,chunk=8,seed=11)
    shared=baseline.state_dict(); actual=hybrid.state_dict()
    for name,value in shared.items(): torch.testing.assert_close(actual[name],value,atol=0,rtol=0)
    args=inputs(); before={k:v.clone() for k,v in hybrid.state_dict().items()}
    with torch.inference_mode():
        first=hybrid(*(x[:,:8] for x in args[:7]),args[7]); warm=hybrid(*(x[:,8:16] for x in args[:7]),args[7],first["memory"]); cold=hybrid(*(x[:,8:16] for x in args[:7]),args[7])
        assert not torch.allclose(warm["field"],cold["field"])
    for name,value in hybrid.state_dict().items(): torch.testing.assert_close(before[name],value,rtol=0,atol=0)


def test_waveform_erased_before_features_including_edge_support():
    rng=np.random.default_rng(42); sound=rng.normal(0,.1,24000*3).astype(np.float32); missing=np.zeros(len(sound),bool); missing[512:4608]=True; changed=sound.copy(); changed[missing]=rng.normal(10,100,missing.sum()); packets=len(sound)//1024
    first,_,visible=observed_mel(sound,missing,packets); second,_,other=observed_mel(changed,missing,packets); np.testing.assert_array_equal(first,second); np.testing.assert_array_equal(visible,other); assert not visible[0]
    extractor=MelSpectrogramFeatures(); clean=extractor(torch.tensor(sound)[None])[0].T[:packets*4].reshape(packets,400); altered=extractor(torch.tensor(changed)[None])[0].T[:packets*4].reshape(packets,400)
    torch.testing.assert_close(clean[visible],altered[visible],rtol=0,atol=0)


def test_bounded_fft_preserves_long_recording_clock():
    torch.set_num_threads(1); sound=torch.randn(1,24000*65+17)*.05; extractor=MelSpectrogramFeatures(); expected=extractor(sound)[0].T; actual=physical_mel(sound,extractor)
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
