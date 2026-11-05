from pathlib import Path
import subprocess
import numpy as np
import pytest
torch=pytest.importorskip("torch")
pytest.importorskip("vocos")
pytest.importorskip("cmudict")
pytest.importorskip("soundfile")
from exploration.stream_media import synthesize,restore_known,ctc_text,write_audio,encode_video,rollout
from exploration.broad_stream_evaluate import gap_interval,edit_distance
from training.stream_codec import load_vocoder,mel_features
from models.slwm_stream import StreamWorld


def test_plain_text_and_packed_codec_types_work_in_cpu_stream():
    torch.set_num_threads(1); model=StreamWorld(width=32,heads=4,layers=2,memory=2).eval(); result=rollout(model,np.zeros((64,400),np.float16),np.zeros((64,64),np.float16),np.full(64,3,np.int16),np.zeros(64,np.float16),"text"); assert result["mel"].shape==(64,400); assert len(result["trace"])==2; assert np.isfinite(result["mel"]).all()


def test_repair_preserves_observed_pcm_and_gap_placement_is_deterministic():
    wave=np.arange(1200,dtype=np.float32)/32768; missing=np.zeros(1200,bool); missing[300:540]=True; generated=np.full(1200,.25,np.float32); repaired=restore_known(generated,wave,missing)
    assert np.array_equal(repaired[~missing],wave[~missing]); assert repaired[420]>.1
    row={"spans":[[3.,3.5,"a"],[5.,5.7,"b"],[6.,6.7,"c"]]}; assert gap_interval(row,10.,.16)==pytest.approx((5.27,5.43))
    assert ctc_text([4,4,0,4])=="bb"; assert edit_distance("one two three","one four three")== (1,3)


@pytest.mark.skipif(not Path("artifacts/decoders/vocos-mel-24khz/pytorch_model.bin").exists(),reason="Pinned external renderer not installed")
def test_bounded_vocoder_clock_matches_complete_decode():
    torch.set_num_threads(1); torch.manual_seed(6); sound=torch.randn(1,71680)*.02; physical=mel_features(sound)[0].T.numpy()[:280]; packed=physical.reshape(-1,400); codec={"audio_mean":np.zeros(100,np.float32),"audio_scale":np.ones(100,np.float32)}; vocoder=load_vocoder()
    with torch.inference_mode(): full=vocoder.decode(torch.from_numpy(np.concatenate((physical,physical[-1:]),0).T.copy()[None]))[0].numpy()
    streamed,meta=synthesize(packed,codec,vocoder); assert len(streamed)==len(full)==71680; assert np.max(np.abs(full-streamed))<5e-5; assert meta["source_phase_used"] is False


def test_encoded_video_has_actual_packet_clock(tmp_path):
    if subprocess.run(["which","ffmpeg"],capture_output=True).returncode: pytest.skip("FFmpeg unavailable")
    frames=np.zeros((32,64,64),np.uint8); frames[::2]=120; write_audio(tmp_path/"sound.wav",np.zeros(32768,np.float32)); meta=encode_video(tmp_path/"movie.mp4",frames,tmp_path/"sound.wav"); assert meta["frames"]==32; assert meta["unique_frames"]==2; assert meta["seconds"]==pytest.approx(32768/24000)
