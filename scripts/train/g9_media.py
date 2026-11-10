"""Render frozen cross-modal predictions for a disjoint room using only the named sensor inputs."""
import argparse
import json
import subprocess
import wave
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g9_transfer import MASKS,records
from models.g9_shared import Spec,encode,decode
from scripts.train.g9_fit import numpy_weights


def sound(bins,hz=16000):
    # A deterministic carrier renders predicted band energies. Raw phase and spoken phonemes are unavailable.
    centers=np.sqrt(np.geomspace(60,7900,17)[:-1]*np.geomspace(60,7900,17)[1:])
    frames=len(bins);t=np.arange(frames*hz//10)/hz
    level=np.clip(bins.reshape(frames,4,16)[:,0],-1,1)
    amp=10**((40*(level+1)-90)/20)
    wave_out=np.zeros(len(t))
    for i,f in enumerate(centers):
        envelope=np.interp(t,np.arange(frames)/10,amp[:,i])
        wave_out+=envelope*np.sin(2*np.pi*f*t+i)
    return np.clip(wave_out,-1,1)


def render(stage,data,dest):
    meta=json.loads((stage/"receipt.json").read_text());spec=Spec(**meta["spec"])
    p=jax.tree.map(jnp.asarray,numpy_weights(stage/"weights.npz"))
    source=records(data/"starss-test.npz")
    name=next(k for k in sorted(source) if len(source[k]["a"])>=400)
    r=source[name];dest.mkdir(exist_ok=True,parents=True);rows={}
    for mode in ("audio_only","video_only","text_only"):
        pred=[];keep=jnp.asarray([MASKS[mode]],bool)
        # Carry temporal state across all 400 observations in one causal sequence.
        from evals.g9_live import initial_memory
        memory=initial_memory(spec,1)
        @jax.jit
        def chunk(params,obs,mem,offset):
            z,mem=encode(params,spec,obs,keep,memory=mem,offset=offset)
            return decode(params,z,obs["u"]),mem
        for offset in range(0,400,16):
            row={k:jnp.asarray(v[None,offset:offset+16]) for k,v in r.items()}
            row["u"]=jnp.zeros((1,16),jnp.int32)
            out,memory=chunk(p,row,memory,jnp.int32(offset))
            pred.append({k:np.asarray(out[k])[0] for k in ("a","v")})
        a=np.concatenate([v["a"] for v in pred]);v=np.concatenate([v["v"] for v in pred])
        tag=meta["tag"]+"-"+mode
        if mode in ("video_only","text_only"):
            samples=sound(a);audio=dest/(tag+".wav")
            with wave.open(str(audio),"wb") as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes((samples*32767).astype("<i2").tobytes())
        if mode in ("audio_only","text_only"):
            frames=np.clip((v+1)*127.5,0,255).astype(np.uint8).reshape(400,16,16,3)
            cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s","16x16","-r","10","-i","pipe:0"]
            if mode=="text_only":cmd+=["-i",str(dest/(tag+".wav")),"-c:a","aac"]
            cmd+=["-vf","scale=128:128:flags=neighbor","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(dest/(tag+".mp4"))]
            subprocess.run(cmd,input=frames.tobytes(),check=True)
        rows[tag]={"recording":name,"seconds":40,"inputs":mode,"emitted_field":spec.field,"weights_sha256":meta["inference_sha256"],
                   "visual_grid":[16,16],"audio":"four-channel spectral-envelope prediction, first-channel deterministic carrier"}
    (dest/(meta["tag"]+"-examples.json")).write_text(json.dumps(rows,indent=2)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("stage",type=Path);ap.add_argument("data",type=Path);ap.add_argument("dest",type=Path)
    a=ap.parse_args();render(a.stage,a.data,a.dest)
