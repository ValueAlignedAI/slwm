"""Prepare a timestamp-joined, scene-held external contact panel from the pinned public ROMA release."""
import csv
import functools
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
from PIL import Image
from g9_recordings import pack, spectrum

REV = "b738c7c2ca20c5e1f9b975c92749da0e51ac4e22"
TREE = f"https://huggingface.co/api/datasets/GeWu-Lab/ROMA/tree/{REV}/ROMA-Bench/"
RAW = f"https://huggingface.co/datasets/GeWu-Lab/ROMA/resolve/{REV}/ROMA-Bench/"
VERBS = ["clench","knock","lift","press","push","shake","slide","tap","touch","weigh"]


def fetch(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"SLWM-G9-research"}),timeout=35) as reader:
                return reader.read()
        except OSError:
            if attempt==4: raise
            time.sleep(2**attempt)


@functools.lru_cache(maxsize=512)
def children(path):
    return sorted(row["path"].split("/")[-1] for row in json.loads(fetch(TREE+urllib.parse.quote(path.rstrip("/"))+"?limit=200")) if row["type"]=="directory")


def fingerprint(s):
    return hashlib.sha256(("SLWM-G9/"+s).encode()).hexdigest()


def trial(name,verb):
    base=RAW+urllib.parse.quote(name)+"/"
    def get(filename):
        return fetch(base+urllib.parse.quote(filename))
    stamp=dict(line.split(":",1) for line in get("mic/time.txt").decode().splitlines() if ":" in line)
    start,end=float(stamp["start"]),float(stamp["end"])
    audio,rate=sf.read(io.BytesIO(get("mic/audio.wav")),always_2d=True,dtype="float32")
    spec=spectrum(audio,rate)
    forces=list(csv.DictReader(io.StringIO(get("force/data.csv").decode())))
    times=np.array([float(row["time"]) for row in forces])
    force=np.array([[float(row[k]) for k in ("fx","fy","fz")] for row in forces])
    frames=sorted((float(t),f.strip()) for t,f in csv.reader(io.StringIO(get("ThirdCamera/time.txt").decode())) if f.endswith(".png"))
    ft=np.array([t for t,_ in frames])
    ticks=start+.1*(1+np.arange(len(spec)))
    # Select the most recent sample at the end of each audio bin. Never read a future sensor value.
    fi=np.searchsorted(ft,ticks,"right")-1
    ni=np.searchsorted(times,ticks,"right")-1
    good=(fi>=0)&(ni>=0)&(ticks<=end+.015)&(ticks-ft[np.maximum(fi,0)]<=.16)&(ticks-times[np.maximum(ni,0)]<=.16)
    inds=np.flatnonzero(good)
    groups=np.split(inds,np.where(np.diff(inds)!=1)[0]+1)
    inds=max(groups,key=len)
    if len(inds)<8: raise ValueError("Fewer than eight consecutive synchronized ticks")
    fi,ni=fi[inds],ni[inds]
    def frame(idx):
        with Image.open(io.BytesIO(get("ThirdCamera/"+frames[idx][1]))) as image:
            return np.asarray(image.convert("RGB").resize((16,16),Image.Resampling.BOX)).reshape(-1)
    unique=np.unique(fi)
    with ThreadPoolExecutor(4) as pool:
        images=dict(zip(unique,pool.map(frame,unique)))
    n=np.zeros((len(inds),12),np.float32)
    n[:,:3]=np.clip(force[ni]/10,-2,2)
    words=np.zeros((len(inds),8),np.uint8)
    words[:,0]=6
    words[:,1]=8+VERBS.index(verb) if verb in VERBS else 22
    meta=dict(ticks=len(inds),clock_start=float(ticks[inds[0]]),max_camera_age_s=float((ticks[inds]-ft[fi]).max()),
              max_force_age_s=float((ticks[inds]-times[ni]).max()),verb=verb,raw_rate=rate)
    return dict(a=spec[inds],v=np.stack([images[i] for i in fi]),n=n,w=words),meta


def main(dest):
    dest.mkdir(parents=True,exist_ok=True)
    scenes=sorted([s for s in children("") if s.startswith("b") and s[1:].isdigit()],key=fingerprint)
    log={"revision":REV,"selection":"first 24 sha256-ranked scene IDs; first 6 calibrate, next 18 held","scenes":scenes[:24],"trials":{},"failed":{}}
    for split,selected in (("cal",scenes[:6]),("held",scenes[6:24])):
        records=[]
        for scene in selected:
            objs=sorted(children(scene),key=lambda o:fingerprint(scene+"/"+o))
            obj=objs[0]
            acts=sorted(children(f"{scene}/{obj}"),key=lambda v:fingerprint(scene+"/"+v))
            for verb in acts[:2]:
                name="/".join((scene,obj,verb))
                cached=dest/(name.replace("/","--")+".npz")
                try:
                    if cached.exists():
                        with np.load(cached) as db:
                            data={k:db[k] for k in ("a","v","n","w")};meta=json.loads(str(db["metadata"]))
                    else:
                        data,meta=trial(name,verb);np.savez_compressed(cached,**data,metadata=json.dumps(meta))
                    records.append((name,data));log["trials"][name]=meta
                except (OSError,ValueError,KeyError,IndexError) as exc:
                    log["failed"][name]=str(exc)[:180]
            print(scene,len(records),flush=True)
        if records:log[split]=pack(records,dest/f"roma-{split}.npz")
        (dest/"roma.json").write_text(json.dumps(log,indent=2)+"\n")


if __name__=="__main__":
    import sys
    main(Path(sys.argv[1]))
