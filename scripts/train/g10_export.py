"""Export predetermined seed-1003 complete models and actual 60-second held-source predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import wave
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_portable import pack,load
from scripts.train.g9_fit import save_json
from scripts.train.g9_media import sound,video_encoder
from scripts.infer.g10_world import stream
from evals.g10_control import load_stage,live_kernel
from evals.g10_real import Bank,predictor,SOURCES
from evals.g9_invariance import center,matching
from g10_store import roots
from g9_storage import sha,write


def av(p,s,bank):
    starts,groups=bank.selection();fields=[[],[]]
    for i in range(0,len(starts),64):
        row=bank.samples(starts[i:i+64],groups[i:i+64])
        for n in (0,1):
            mask=np.eye(4,dtype=bool)[n];z,_=predictor(s,False)(p,row,mask);fields[n].append(np.asarray(z))
    a,v=[center(np.concatenate(h).reshape(-1,8,s.field).astype(np.float64)) for h in fields]
    return matching(a,v)


def make_media(p,s,rows,tag,dest):
    for source in ("audio","vision","words"):
        emit=("vision",) if source=="audio" else ("audio",) if source=="vision" else ("vision","audio")
        pred=stream(p,s,rows,(source,),emit);path=dest/(tag+"-"+source)
        if "audio" in pred:
            with wave.open(str(path)+".wav","wb") as fd:
                fd.setnchannels(1);fd.setsampwidth(2);fd.setframerate(16000);fd.writeframes((sound(pred["audio"])*32767).astype("<i2").tobytes())
        if "vision" in pred:
            v=np.clip((pred["vision"]+1)*127.5,0,255).astype(np.uint8)
            cmd=["ffmpeg","-nostdin","-y","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","16x16","-r","10","-i","pipe:0"]
            if "audio" in pred:cmd+=["-i",str(path)+".wav","-c:a","aac"]
            cmd+=["-vf","scale=128:128:flags=neighbor","-c:v",video_encoder(),"-pix_fmt","yuv420p","-movflags","+faststart",str(path)+".mp4"]
            subprocess.run(cmd,input=v.tobytes(),check=True)


def main(runs,data,out):
    models=out/"inference";media=out/"examples";models.mkdir(parents=True,exist_ok=True);media.mkdir(exist_ok=True);receipts={};info={}
    bank=Bank(data/"holo-novel.npz",True)
    order=sorted(range(len(bank.names)),key=lambda i:hashlib.sha256(("G10-example/"+bank.names[i]).encode()).hexdigest())
    i=next(i for i in order if bank.lengths[i]>=600 and np.any(bank.data["x"][bank.origins[i]:bank.origins[i]+600]))
    rows={k:v[bank.origins[i]:bank.origins[i]+600] for k,v in bank.data.items()};rows.update(u=np.zeros(600,np.int32),plan=np.zeros(600,np.int32))
    np.savez_compressed(media/"held-public-input.npz",**rows)
    tags=["align-hybrid-paired-s1003","align-gpt2-paired-s1003","context-gated-variable-s1003"]
    for tag in tags:
        step=1536 if tag.startswith("align") else 5120;stage=runs/tag/f"step-{step:05d}";d,p,s=load_stage(stage)
        file=models/(tag+".npz")
        pack(file,{k:np.asarray(v) for k,v in p.items()},d["spec"],d["inference_sha256"])
        meta,decoded=load(file);compact=jax.tree.map(jnp.asarray,decoded)
        pts={}
        for mode in ("continuous","brief","latched"):
            for key,weights in (("original",p),("portable",compact)):
                scores=live_kernel(s,"full",0,128,mode)(weights,jnp.int32(247667));pts[key+"_"+mode]=np.asarray(scores)[:,:,0].mean(0).tolist()
        rec=dict(sha256=sha(file),parent_sha256=d["inference_sha256"],bytes=file.stat().st_size,original_bytes=(stage/"weights.npz").stat().st_size,control=pts)
        if tag.startswith("align"):
            rec["original_audio_to_vision"]=av(p,s,bank);rec["portable_audio_to_vision"]=av(compact,s,bank)
            make_media(compact,s,rows,tag,media)
        receipts[tag]=rec
        print(tag,file.stat().st_size,float(np.mean(pts["portable_continuous"])),flush=True)
    save_json(models/"portable.json",receipts)
    save_json(media/"source.json",dict(source="HoloAssist",recording=bank.names[i],role="held coffee/computer families",first_tick=0,ticks=600,seconds=60,
           inputs="10 Hz released AV and acceleration/gyro plus completed speech n-gram features",visual_grid=[16,16],
           audio="predicted spectral envelopes with deterministic carriers",parents={k:v["parent_sha256"] for k,v in receipts.items()}))
    disks=roots()
    for k in models.iterdir():info[k.name]={"sha256":sha(k),"bytes":k.stat().st_size}
    for a in disks:
        dest=a.parent/"public-inference";dest.mkdir(exist_ok=True)
        for name,r in info.items():
            shutil.copyfile(models/name,dest/name)
            if sha(dest/name)!=r["sha256"]:raise ValueError("Compact package readback failed")
        write(dest/"verified.json",info)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("runs",type=Path);p.add_argument("holo",type=Path);p.add_argument("out",type=Path);p.add_argument("--cache")
    a=p.parse_args()
    if a.cache:jax.config.update("jax_compilation_cache_dir",a.cache)
    main(a.runs,a.holo,a.out)
