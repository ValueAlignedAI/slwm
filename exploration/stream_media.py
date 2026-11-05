"""Continuous field inference, transparent PCA video and bounded-context neural audio."""
from __future__ import annotations
from contextlib import nullcontext
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import soundfile as sf
import torch
import torchaudio
from training.stream_codec import PACKET_SECONDS,SAMPLE_RATE,load_vocoder
from training.research_sources import file_sha


def load_renderer(directory=None,device="cpu"):
    """Load the separately declared frozen waveform renderer from a portable bundle."""
    if directory is None or not(Path(directory)/"pytorch_model.bin").exists(): return load_vocoder(device)
    from vocos import Vocos
    folder=Path(directory); model=Vocos.from_hparams(str(folder/"config.yaml")); model.load_state_dict(torch.load(folder/"pytorch_model.bin",map_location="cpu",weights_only=True),strict=True); return model.to(device).eval().requires_grad_(False)


def check_video_cache(cache,source,required_frames):
    marker=Path(cache)/"input_pixels.json"
    if marker.exists():
        meta=json.loads(marker.read_text())
        if meta["raw_video_sha256"]!=file_sha(source) or meta["frames"]<required_frames: raise ValueError("Use a fresh output folder for a changed or longer video input; the existing cache belongs to another request")


def read_audio(path,start=0.,seconds=None):
    """Decode/resample -> quantized mono float32 PCM[N] at24kHz."""
    info=sf.info(path); audio,sr=sf.read(path,start=int(round(start*info.samplerate)),frames=-1 if seconds is None else int(round(seconds*info.samplerate)),dtype="float32",always_2d=True); wave=torch.from_numpy(audio.mean(1))[None]
    if sr!=SAMPLE_RATE: wave=torchaudio.functional.resample(wave,sr,SAMPLE_RATE)
    pcm=np.rint(wave[0].numpy()*32768.).clip(-32768,32767).astype("<i2"); return pcm.astype(np.float32)/32768.


def write_audio(path,sound):
    pcm=np.rint(np.asarray(sound)*32768.).clip(-32768,32767).astype("<i2"); sf.write(path,pcm,SAMPLE_RATE,subtype="PCM_16"); return {"samples":len(pcm),"seconds":len(pcm)/SAMPLE_RATE,"peak_before_clipping":float(np.max(np.abs(sound))) if len(sound) else 0.,"clipped_fraction":float(np.mean(np.abs(sound)>=1.))}


def ctc_text(ids):
    alphabet=" 'abcdefghijklmnopqrstuvwxyz"; last=None; chars=[]
    for item in ids:
        item=int(item)
        if item and item!=last: chars.append(alphabet[item-1])
        last=item
    return "".join(chars).strip()


def rollout(model,audio,video,phones,appearance,mode,*,visible=None,prefix_chunks=None,reset_memory=False,device="cpu",keep_fields=False):
    """Received chunks -> proposals with persistent memory; no targets or labels assumed.

    Full audio/video/text arrays represent an incoming stream. Only the requested
    modality is exposed at each tick; prefix-only probes suppress subsequent input.
    Text phones are used exclusively in explicitly supplied-text mode.
    """
    length=len(audio); memory=None; records=[]; outputs={k:[] for k in ("mel","video","character_ids","phone_ids")}; fields=[]
    amp=torch.autocast("cuda",dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    with torch.inference_mode(),amp:
        for tick,start in enumerate(range(0,length,model.chunk)):
            end=min(start+model.chunk,length); n=end-start; valid=torch.ones(1,n,device=device,dtype=torch.bool); yes=valid; no=torch.zeros_like(valid)
            a=yes if mode in ("audio","repair","repair_av") else no; v=yes if mode in ("video","repair_av") else no; t=yes if mode=="text" else no
            if mode in ("repair","repair_av"): a=torch.as_tensor(visible[None,start:end],device=device)
            if prefix_chunks is not None and tick>=prefix_chunks: a=v=t=no
            if reset_memory: memory=None
            before=time.perf_counter(); out=model(torch.as_tensor(audio[None,start:end],device=device,dtype=torch.float32),torch.as_tensor(video[None,start:end],device=device,dtype=torch.float32),torch.as_tensor(phones[None,start:end],device=device,dtype=torch.long),a,v,t,valid,torch.as_tensor(appearance[None],device=device,dtype=torch.float32),memory); memory=out["memory"]
            # Transfer the emitted proposal before timing, making CUDA completion
            # part of latency instead of measuring kernel dispatch alone.
            for key in ("mel","video"): outputs[key].append(out[key][0].float().cpu().numpy())
            outputs["character_ids"].append(out["character_logits"][0].argmax(-1).cpu().numpy()); outputs["phone_ids"].append(out["phone_logits"][0].argmax(-1).cpu().numpy())
            if keep_fields: fields.append(out["field"][0].float().cpu().numpy())
            records.append({"tick":tick,"start_packet":start,"source_start_seconds":start*PACKET_SECONDS,"received_until_seconds":end*PACKET_SECONDS,"observed_audio_fraction":float(a.float().mean()),"observed_video_fraction":float(v.float().mean()),"provided_text_fraction":float(t.float().mean()),"memory_norm":float(memory.float().norm()),"field_norm":float(out["field"].float().norm()),"predicted_error":float(out["risk"].clamp(-10,10).exp().mean()),"annotated_activity_probability":float(out["activity"].softmax(-1)[...,1].mean()),"compute_seconds":time.perf_counter()-before})
    result={k:np.concatenate(v) for k,v in outputs.items()}; result["text"]=ctc_text(result["character_ids"]); result["trace"]=records
    if keep_fields: result["field"]=np.concatenate(fields)
    return result


def synthesize(packed,codec,vocoder,device="cpu",context=48,chunk_frames=128):
    """Physical logmel -> waveform, using bounded overlapping decoder context.

    A final repeated analysis frame closes the center-padded inverse-STFT clock.
    Interior chunks use48 predicted mel frames on each side, never source phase.
    """
    normalized=packed.reshape(-1,100); physical=normalized*codec["audio_scale"]+codec["audio_mean"]; clipping=float(np.mean((physical< -16.12)|(physical>8.))); physical=physical.clip(-16.12,8.).astype(np.float32); outputs=[]
    with torch.inference_mode():
        for start in range(0,len(physical),chunk_frames):
            end=min(start+chunk_frames,len(physical)); lo=max(0,start-context); hi=min(len(physical),end+context); block=physical[lo:hi]
            if hi==len(physical): block=np.concatenate((block,block[-1:]),0)
            wave=vocoder.decode(torch.as_tensor(block.T.copy()[None],device=device))[0].float().cpu().numpy(); begin=(start-lo)*256; segment=wave[begin:begin+(end-start)*256]
            if len(segment)!=(end-start)*256: raise ValueError("Vocoder sample clock mismatch")
            outputs.append(segment)
    return np.concatenate(outputs),{"external_vocoder_parameters":sum(p.numel() for p in vocoder.parameters()),"physical_mel_clipped_fraction":clipping,"decoder_context_mel_frames":context,"terminal_repeated_analysis_frames":1,"source_phase_used":False}


def restore_known(generated,sound,missing,fade_samples=240):
    """Prediction inside gaps; exact observed PCM outside, with10ms internal fades."""
    result=np.zeros_like(sound); n=min(len(generated),len(sound)); result[:n]=generated[:n]; missing=np.asarray(missing,bool)
    if missing.any() and np.flatnonzero(missing)[-1]>=n: raise ValueError("Model proposals do not cover the complete gap")
    edges=np.flatnonzero(np.diff(np.r_[False,missing,False]));
    for start,end in edges.reshape(-1,2):
        width=end-start; fade=min(fade_samples,width//2); baseline=np.linspace(sound[start-1] if start else 0.,sound[end] if end<len(sound) else 0.,width+2,dtype=np.float32)[1:-1]; weights=np.ones(width,np.float32)
        if fade:
            ramp=np.sin(np.linspace(0,np.pi/2,fade+2,dtype=np.float32)[1:-1])**2; weights[:fade]=ramp; weights[-fade:]=ramp[::-1]
        result[start:end]=weights*result[start:end]+(1-weights)*baseline
    result[~missing]=sound[~missing]; return result


def decode_video(codes,codec,appearance_pixels,appearance_code):
    """Normalized PCA proposals -> actual64×64 frames; supplied appearance is explicit."""
    delta=((codes-appearance_code)*codec["video_scale"])@codec["video_basis"]; pictures=appearance_pixels.reshape(1,-1).astype(np.float32)/255.+delta
    return np.rint(pictures.clip(0,1)*255).astype(np.uint8).reshape(-1,64,64)


@lru_cache(maxsize=1)
def video_encoder():
    encoders=subprocess.check_output(["ffmpeg","-hide_banner","-encoders"],stderr=subprocess.DEVNULL).decode()
    if "libx264" in encoders: return "libx264",["-preset","fast","-crf","18"]
    if "libopenh264" in encoders: return "libopenh264",["-b:v","512k"]
    raise RuntimeError("A CPU H.264 encoder (libx264 or libopenh264) is required")


def encode_video(path,frames,audio_path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    encoder,options=video_encoder(); command=["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","gray","-s","64x64","-framerate","375/16","-i","pipe:0","-i",str(audio_path),"-c:v",encoder,*options,"-pix_fmt","yuv420p","-c:a","aac","-b:a","96k","-shortest",str(path)]
    process=subprocess.run(command,input=np.ascontiguousarray(frames).tobytes(),capture_output=True)
    if process.returncode: raise RuntimeError(process.stderr.decode()[-2000:])
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames,duration,width,height","-of","json",str(path)]))["streams"][0]; assert int(probe["nb_read_frames"])==len(frames)
    decoded=subprocess.check_output(["ffmpeg","-v","error","-i",str(path),"-an","-f","rawvideo","-pix_fmt","gray","pipe:1"]); actual=np.frombuffer(decoded,np.uint8).reshape(-1,64,64); assert len(actual)==len(frames)
    return {"frames":len(frames),"seconds":len(frames)*PACKET_SECONDS,"native_width":64,"native_height":64,"unique_frames":len({frame.tobytes() for frame in frames}),"decoded_unique_frames":len({frame.tobytes() for frame in actual}),"encoder":encoder,"sha256":file_sha(path)}
