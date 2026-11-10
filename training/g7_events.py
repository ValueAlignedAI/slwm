"""Pack real, synchronized STARSS23 pressure/video and the first annotated tick of each contact event."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

import numpy as np
from training.g4_data import DT,number,quant
from training.g7_pools import save_pack

RECORD=None


def window(arg):
    root,official,stem,role,class_ids=arg
    cache=root/"event-records"/role; cache.mkdir(parents=True,exist_ok=True)
    target=cache/(stem+".npz")
    if target.is_file():return str(target)
    import soundfile as sf
    import torch
    from vocos.feature_extractors import MelSpectrogramFeatures
    global RECORD
    if RECORD is None:
        torch.set_num_threads(1)
        RECORD=MelSpectrogramFeatures(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=100).eval()
    with zipfile.ZipFile(root/"starss/foa_dev.zip") as archive:
        wave,sr=sf.read(io.BytesIO(archive.read(f"foa_dev/{official}/{stem}.wav")),dtype="float32")
    if sr!=24000 or wave.ndim!=2 or wave.shape[1]!=4:raise ValueError("Unexpected FOA schema")
    with zipfile.ZipFile(root/"starss/video_dev.zip") as z,tempfile.NamedTemporaryFile(dir=cache,suffix=".mp4") as vid:
        with z.open(f"video_dev/{official}/{stem}.mp4") as source:
            import shutil
            shutil.copyfileobj(source,vid,4<<20)
        vid.flush()
        raw=subprocess.check_output(["ffmpeg","-v","error","-threads","1","-i",vid.name,"-an","-filter_threads","1",
            "-vf","fps=375/16,scale=16:16:flags=area,format=gray","-threads","1","-f","rawvideo","pipe:1"])
    images=np.frombuffer(raw,np.uint8).reshape(-1,256)
    with torch.inference_mode():features=RECORD(torch.from_numpy(np.ascontiguousarray(wave[:,0]))[None])[0].T.numpy()
    ticks=min(len(images),len(wave)//1024,len(features)//4)//64*64
    if ticks<64:raise ValueError("Insufficient synchronous data")
    event=np.zeros(ticks,np.uint8); active={}
    with zipfile.ZipFile(root/"starss/metadata_dev.zip") as z:
        items=list(csv.reader(io.StringIO(z.read(f"metadata_dev/{official}/{stem}.csv").decode())))
    for row in sorted(items,key=lambda v:int(v[0])):
        frame,kind,track=map(int,row[:3]);key=(kind,track)
        if kind in class_ids and frame>active.get(key,-2)+1:
            tick=round(frame*.1/DT)
            if 0<=tick<ticks:event[tick]=1
        active[key]=frame
    count=ticks//64;room=re.search("room([0-9]+)",stem)[1]
    group=number("STARSS23/"+official.split("-")[-1]+"/"+room)
    arr={"a":quant(features[:ticks*4].reshape(count,64,4,100).mean(2)),
         "v":images[:ticks].reshape(count,64,256),"event":event.reshape(count,64),
         "group":np.full(count,group,np.uint64),"recording":np.full(count,number("STARSS23/"+stem),np.uint64),
         "uid":np.array([number("STARSS23/"+stem+"/"+str(j)) for j in range(count)],np.uint64)}
    with target.with_suffix(".part").open("wb") as f:np.savez(f,**arr)
    target.with_suffix(".part").replace(target);return str(target)


def prepare(root,workers):
    with zipfile.ZipFile(root/"starss/video_dev.zip") as z:files=sorted(x for x in z.namelist() if x.endswith(".mp4"))
    rolemap={"dev-train-tau":"train","dev-test-tau":"dev","dev-test-sony":"test"}
    classes=set(json.loads(Path("configs/t7/study.json").read_bytes())["starss_contact_classes"])
    for official,role in rolemap.items():
        files_here=[p for p in files if Path(p).parts[1]==official]
        jobs=[(root,official,Path(file).stem,role,classes) for file in files_here]
        with ProcessPoolExecutor(max_workers=workers) as pool: paths=list(pool.map(window,jobs,chunksize=1))
        a={k:[] for k in ("a","v","event","group","recording","uid")}
        for p in paths:
            with np.load(p) as f:
                for k in a:a[k].append(np.array(f[k]))
        a={k:np.concatenate(v) for k,v in a.items()}
        save_pack(root/"fresh/events"/role,a,{"family":"events","role":role,"site":official.split("-")[-1],
            "official_partition":official,"rooms":len(np.unique(a["group"])),"recordings":len(files_here),"onsets":int(a["event"].sum()),
            "labels":"First 100-ms activity bin per labeled source track in clapping, footsteps, door and knocking, on the joint 1024/24000 clock."})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--workers",type=int,default=3);a=p.parse_args();prepare(a.root,a.workers)
