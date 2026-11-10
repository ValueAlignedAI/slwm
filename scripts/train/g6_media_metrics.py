"""Verify both media streams and quantify source alignment on the registered, continuous development clip."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def corr(x,y):
    x=x-x.mean(); y=y-y.mean()
    return float(x@y/max(float(np.linalg.norm(x)*np.linalg.norm(y)),1e-12))


def compare(x,y):
    n=min(len(x),len(y)); x=np.asarray(x[:n],np.float64); y=np.asarray(y[:n],np.float64)
    shifts=[corr(x,np.roll(y,k)) for k in range(16,n-15,8)]
    return {"ticks":n,"aligned":corr(x,y),"shifted_median":float(np.median(shifts)),
            "shifted_95pct":float(np.quantile(shifts,.95))}


def analyze(folder,stage):
    note=json.loads((folder/"live.json").read_bytes())
    rec=json.loads((stage/"verified.json").read_bytes())
    if rec["step"]!=8192 or note["weights_sha256"]!=rec["sha256"]["inference.npz"]:
        raise ValueError("Wrong terminal recording weights")
    def energy(p):
        audio,sr=sf.read(p,dtype="float32")
        if sr!=24000 or audio.ndim!=1: raise ValueError("Unexpected waveform schema")
        width=1024
        return np.log(np.maximum(np.sqrt((audio[:len(audio)//width*width].reshape(-1,width)**2).mean(1)),1e-8))
    reference=energy(folder/"audio-to-video.wav"); gen=energy(folder/"video-to-audio.wav")
    with np.load(folder/"predictions.npz",allow_pickle=False) as z:
        frames=np.rint(np.clip((z["audio"][:,100:]+1)*127.5,0,255))
        observed=z["orig_v"].astype(np.float32)
    motion=lambda x:np.abs(np.diff(x,axis=0)).mean(-1)
    clips={}
    for name in ("video-to-audio","audio-to-video","text-to-audio-video"):
        file=folder/(name+".mp4")
        report=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration:stream=codec_type,codec_name",
                                                   "-of","json",file],text=True))
        seconds=float(report["format"]["duration"])
        if abs(seconds-1024*1024/24000)>.1 or {r["codec_type"] for r in report["streams"]}!={"video","audio"}:
            raise ValueError("Truncated media or missing track")
        clips[name]={"seconds":seconds,"sha256":sha(file),"bytes":file.stat().st_size}
    result={"run":rec["run"],"inference_sha256":note["weights_sha256"],"recording":note["recording"],"role":"one development clip",
            "audio_log_rms_envelope":compare(reference,gen),"frame_absolute_motion_envelope":compare(motion(observed),motion(frames)),
            "prompt_effect_rms_mel":note["prompt_effect_rms_mel"],"clips":clips}
    out=folder/"media-verified.json"; tmp=out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); tmp.replace(out)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("folder",type=Path); p.add_argument("stage",type=Path)
    a=p.parse_args(); analyze(a.folder,a.stage)
