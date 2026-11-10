"""Independent physical timing confirmation media from untouched actors and STARSS recording rooms."""
import argparse
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from training.g4_data import number,quant,DT
from training.g7_pools import save_pack
from training.g7_events import window

REV="1658cd342dff90010aa843eaeebd53610a08b1dc"
URL="https://github.com/CheyneyComputerScience/CREMA-D"
RAW="https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/"+REV+"/VideoFlash/"
MEDIA="https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/"+REV+"/VideoFlash/"


def fetch(job):
    from training.g7_speech import read_clip
    import torch
    import torchaudio
    from vocos.feature_extractors import MelSpectrogramFeatures
    folder,name=job;dest=folder/(name+".npz")
    if dest.is_file():return dest
    raw=folder/(name+".flv");pointer=folder/(name+".lfs")
    if not pointer.exists():
        subprocess.run(["curl","-fsSL","--retry","3","--max-time","55",RAW+raw.name,"-o",str(pointer)+".part"],check=True)
        Path(str(pointer)+".part").replace(pointer)
    expected=next(i.split("sha256:")[1] for i in pointer.read_text().splitlines() if i.startswith("oid sha256:"))
    if not raw.exists():
        subprocess.run(["curl","-fsSL","--retry","3","--max-time","70",MEDIA+raw.name,"-o",str(raw)+".part"],check=True)
        Path(str(raw)+".part").replace(raw)
    with raw.open("rb") as h:actual=hashlib.file_digest(h,"sha256").hexdigest()
    if actual!=expected:raise ValueError("Pinned CREMA media hash changed")
    vf="fps=25,crop=iw*0.30:ih*0.24:iw*0.35:ih*0.42,scale=48:32"
    vid=subprocess.check_output(["ffmpeg","-v","error","-threads","1","-i",str(raw),"-an","-vf",vf,"-frames:v","128","-f","rawvideo","-pix_fmt","gray","-threads","1","-"])
    decoded=folder/(name+".mouth.npz");np.savez(decoded,mouth=np.frombuffer(vid,np.uint8).reshape(-1,32,48))
    pcm,pixels=read_clip((name,"","",str(raw),str(decoded)))
    mel=MelSpectrogramFeatures(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=100).eval()
    with torch.inference_mode():m=mel(torchaudio.functional.resample(torch.from_numpy(pcm)[None],16000,24000))[0].T.numpy()
    ticks=min(int(len(pcm)/16000/DT),int(len(pixels)/25/DT),len(m)//4);size=min(ticks,64);start=(ticks-size)//2
    if size<=20:raise ValueError("No eligible timing interior")
    a=np.zeros((64,100),np.uint8);v=np.zeros((64,256),np.uint8)
    a[:size]=quant(m[4*start:4*(start+size)].reshape(size,4,100).mean(1))
    v[:size]=pixels[np.minimum(((start+np.arange(size)+.5)*DT*25).astype(int),len(pixels)-1)]
    actor,phrase=name.split("_")[:2]
    with dest.with_suffix(".npz.part").open("wb") as h:np.savez(h,a=a,v=v,uid=np.uint64(number("crema/"+name)),group=np.uint64(number("crema/"+actor)),phrase=phrase,length=np.uint8(size),sha=np.asarray(actual))
    dest.with_suffix(".npz.part").replace(dest);return dest


def speech(root,old):
    import torch
    torch.set_num_threads(1);base=old/"artifacts/t2_audio_visual/cremad_motion_v3"
    if json.loads((base/"dataset_card.json").read_bytes())["source_revision"]!=REV:raise ValueError("CREMA revision changed")
    names=[Path(r["Filename"]).stem for r in csv.DictReader((base/"SentenceFilenames.csv").open())]
    used={r["actor"] for k in ("train","validation","test") for r in json.loads((base/(k+".json")).read_bytes())}
    people=sorted({n.split("_")[0] for n in names}-used);chosen=[]
    for actor in people:
        members=sorted((n for n in names if n.startswith(actor+"_")),key=lambda n:number("G7-physical-time/"+n))
        phrases=sorted({n.split("_")[1] for n in members})
        chosen.extend(n for ph in phrases for n in [k for k in members if k.split("_")[1]==ph][:2])
    if not people:raise ValueError("No untouched CREMA actors")
    folder=root/"physical-time/crema/files";folder.mkdir(exist_ok=True,parents=True)
    with ThreadPoolExecutor(max_workers=6) as pool:paths=list(pool.map(fetch,[(folder,n) for n in chosen]))
    keys=("a","v","uid","group","phrase","length");arr={k:[] for k in keys};pins={}
    for file in paths:
        with np.load(file) as f:
            for k in keys:arr[k].append(np.array(f[k]))
            pins[file.stem]=str(f["sha"])
    save_pack(root/"physical-time/crema/confirm",{k:np.stack(v) for k,v in arr.items()},
        {"role":"confirmation","actors":people,"excluded_prior_actors":len(used),"source_revision":REV,"source":URL,"source_sha256":pins})


def events(root):
    cfg=json.loads(Path("configs/t7/study.json").read_bytes())
    with zipfile.ZipFile(root/"starss/video_dev.zip") as z:
        stems=sorted(Path(p).stem for p in z.namelist() if "/dev-train-sony/" in p and p.endswith(".mp4"))
    with ProcessPoolExecutor(max_workers=4) as pool:
        paths=list(pool.map(window,[(root,"dev-train-sony",s,"sync-confirm",set(cfg["starss_contact_classes"])) for s in stems],chunksize=1))
    arrays={k:[] for k in ("a","v","event","group","recording","uid")}
    for f in paths:
        with np.load(f) as rec:
            for k in arrays:arrays[k].append(np.array(rec[k]))
    arrays={k:np.concatenate(v) for k,v in arrays.items()};new=set(arrays["group"].tolist())
    old={int(g) for role in ("train","dev","test") for g in np.load(root/"fresh/events"/role/"group.npy")}
    if old&new:raise ValueError("Confirmation rooms overlap primary G7 rooms")
    save_pack(root/"physical-time/events/confirm",arrays,{"role":"confirmation","official":"dev-train-sony","rooms":len(new),"recordings":len(stems),"source":"10.5281/zenodo.7880637"})


if __name__=="__main__":
    q=argparse.ArgumentParser();q.add_argument("--root",type=Path,required=True);q.add_argument("--repo",type=Path,required=True);q.add_argument("phase",choices=("crema","events"))
    a=q.parse_args()
    if a.phase=="crema":speech(a.root,a.repo)
    else:events(a.root)
