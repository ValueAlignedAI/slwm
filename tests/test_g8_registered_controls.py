import itertools
import numpy as np
import pytest


def test_fixed_rate_optimizer_handles_small_payload_and_matches_exhaustive_solution():
    from training.g8_rate_codec import allocation
    costs=np.array([[9.,3.,1.,.5],[4.,2.,1.3,1.1],[1.,.9,.7,.6]])
    for budget in (0,1,2,4,7):
        found=allocation(costs,budget)
        options=[v for v in itertools.product(range(4),repeat=3) if sum(v)==budget]
        assert sum(found)==budget
        assert sum(costs[i,k] for i,k in enumerate(found))==pytest.approx(min(sum(costs[i,k] for i,k in enumerate(v)) for v in options))


def test_acoustic_bitstream_roundtrip_and_padding():
    from training.g8_rate_codec import pack,unpack
    bits=np.array([3,2,0,1],np.int16);rows=np.array([[7,0,0,1],[1,3,0,0],[5,1,0,1]],np.uint8)
    payload=pack(rows,bits)
    assert len(payload)==3
    np.testing.assert_array_equal(unpack(payload,bits,3),rows)
    with pytest.raises(ValueError,match="Malformed"):
        unpack(payload[:-1]+bytes([payload[-1]|1]),bits,3)


def test_conditional_regret_handles_tied_predictions():
    pytest.importorskip("jax")
    from training.g8_pushing import summarize
    real=np.array([[[0.,0.],[1.,0.],[3.,0.]]])
    goals=np.array([[[3.,0.],[-3.,0.]]])
    assert summarize(real,real,goals)["regret_px"][0]==pytest.approx(0.)
    uniform=summarize(np.zeros_like(real),real,goals)
    truth=np.linalg.norm(real[:,None]-goals[:,:,None],axis=-1)
    assert uniform["regret_px"][0]==pytest.approx((truth.mean(-1)-truth.min(-1)).mean())


def test_partial_mixtures_restore_every_excluded_family():
    pytest.importorskip("optax")
    from training.g8_factorial import FactorialData
    class Stream:
        def __init__(self,fresh):self.fresh=fresh
        def requests(self,step,batch):
            fam=np.array([0,1,2,3,4,5,6,7,8])
            rows={k:((fam+10*self.fresh)[:,None]).copy() for k in ("a","v","x","input_x","n","u","outcome","target_n","event")}
            return dict(rows,family=fam)
    data=object.__new__(FactorialData);data.base=Stream(0);data.bank=Stream(1);data.arm="sim";data.families=(8,)
    values=data.requests(8192,9)
    for k in ("a","v","x","input_x","n","u","outcome","target_n","event"):
        np.testing.assert_array_equal(values[k].ravel(),[0,1,2,3,4,5,6,7,18])


def test_lossless_optimizer_packet_preserves_even_and_odd_array_boundaries():
    from scripts.train.g8_file_blocks import frame,unframe
    rng=np.random.default_rng(888)
    for size in (1,2,3,4,17,4096,8201):
        data=rng.integers(0,256,size,dtype=np.uint8).tobytes();encoded=frame(data)
        assert unframe(encoded,size)==data
        with pytest.raises(ValueError):unframe(encoded+b"x",size)
        with pytest.raises(ValueError):unframe(encoded,size+1)


def test_released_audio_file_roundtrip_preserves_samples_and_detects_corruption(tmp_path):
    from pathlib import Path
    pytest.importorskip("vocos")
    import soundfile as sf
    from training.g8_audio_codec import PUBLIC,encode_audio,decode_audio
    renderer=Path("artifacts/decoders/vocos-mel-24khz")
    if not (renderer/"pytorch_model.bin").is_file():pytest.skip("External frozen Vocos fixture unavailable")
    path=tmp_path/"input.wav";stream=tmp_path/"clip.slwma";decoded=tmp_path/"decoded.wav"
    samples=np.sin(2*np.pi*337*np.arange(2333)/16000).astype(np.float32)*.06
    sf.write(path,samples,16000)
    meta=encode_audio(path,stream,PUBLIC,48)
    assert meta["samples"]==3500
    result=decode_audio(stream,decoded,PUBLIC,renderer,"cpu")
    audio,sr=sf.read(decoded)
    assert sr==24000 and len(audio)==3500 and result==meta and np.isfinite(audio).all()
    content=bytearray(stream.read_bytes());content[-1]^=1;stream.write_bytes(content)
    with pytest.raises(ValueError,match="checksum"):decode_audio(stream,decoded,PUBLIC,renderer,"cpu")
