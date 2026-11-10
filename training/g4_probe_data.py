"""Prepare two held-out downstream tasks; the training pipeline never opens these files."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time
import urllib.request
import zipfile

import numpy as np

from training.g4_data import T, number, quant, raster


def download(url,path):
    if path.exists(): return
    tmp=path.with_suffix(".partial")
    with urllib.request.urlopen(url,timeout=120) as response,tmp.open("wb") as out:
        while block:=response.read(4<<20): out.write(block)
    tmp.replace(path)


def sorted_class(items, label, identity, cap):
    by=defaultdict(list)
    for item in items: by[label(item)].append(item)
    out=[]
    for key in sorted(by):
        out.extend(sorted(by[key],key=lambda x:number(identity(x)))[:cap])
    return out


def cifar(root,cfg):
    file=root/"cifar-10-binary.tar.gz"
    download("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",file)
    with file.open("rb") as stream:
        if hashlib.file_digest(stream,"md5").hexdigest()!=cfg["cifar10_binary_md5"]: raise ValueError("Unexpected CIFAR release")
    result={}; seen=set()
    with tarfile.open(file) as tar:
        for source,splits in (("data_batch",("fit","dev")),("test_batch",("test",))):
            members=sorted((m for m in tar.getmembers() if m.isfile() and Path(m.name).name.startswith(source)),key=lambda m:m.name)
            data=np.concatenate([np.frombuffer(tar.extractfile(m).read(),np.uint8).reshape(-1,3073) for m in members])
            labels=data[:,0]; digest=[hashlib.sha256(row.data).digest() for row in data[:,1:]]
            order=sorted(range(len(data)),key=lambda i:number(digest[i]))
            ordered=[]
            for i in order:
                if digest[i] not in seen:
                    ordered.append(i); seen.add(digest[i])
            for role in splits:
                offset,limit=(0,8) if role=="fit" else ((8,48) if role=="dev" else (0,100))
                idx=np.array([i for c in range(10) for i in [j for j in ordered if labels[j]==c][offset:offset+limit]])
                image=data[idx,1:].reshape(-1,3,32,32).transpose(0,2,3,1).repeat(2,1).repeat(2,2)
                result[role]={"data":raster(image),"labels":labels[idx],"group":idx,"kind":np.int32(1)}
    return result


def esc_archive(root,rev):
    full=root/("esc50-"+rev+".zip")
    if full.exists(): return full
    subset=root/("esc50-selected-"+rev+".zip")
    if subset.exists(): return subset
    cache=root/"esc50-pinned"; cache.mkdir(exist_ok=True)
    url=f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{rev}"
    meta=cache/"esc50.csv"; download(url+"/meta/esc50.csv",meta)
    rows=list(csv.DictReader(io.StringIO(meta.read_text())))
    fit=sorted_class([r for r in rows if int(r["fold"])<=3],lambda r:int(r["target"]),lambda r:r["filename"],8)
    use=fit+[r for r in rows if int(r["fold"])>=4]
    def recording(row):
        path=cache/row["filename"]
        for attempt in range(3):
            try:
                download(url+"/audio/"+path.name,path)
                with path.open("rb") as stream:
                    if stream.read(4)!=b"RIFF": raise ValueError("Unexpected ESC file encoding")
                return path
            except OSError:
                if attempt==2: raise
                time.sleep(2**attempt)
    with ThreadPoolExecutor(max_workers=12) as pool:
        recordings=list(pool.map(recording,use))
    tmp=subset.with_suffix(".part")
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_STORED) as z:
        z.write(meta,"esc/meta/esc50.csv")
        for path in recordings: z.write(path,"esc/audio/"+path.name)
    tmp.replace(subset)
    return subset


def esc(root,cfg):
    import soundfile as sf
    import torch
    import torchaudio
    from vocos.feature_extractors import MelSpectrogramFeatures
    torch.set_num_threads(3)
    rev=cfg["esc50_revision"]
    file=esc_archive(root,rev)
    result={}
    extractor=MelSpectrogramFeatures(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=100)
    with zipfile.ZipFile(file) as z:
        meta=next(k for k in z.namelist() if k.endswith("/meta/esc50.csv"))
        prefix=meta.split("/meta/")[0]+"/audio/"
        rows=list(csv.DictReader(io.StringIO(z.read(meta).decode())))
        for role,folds in (("fit",{1,2,3}),("dev",{4}),("test",{5})):
            subset=[r for r in rows if int(r["fold"]) in folds]
            use=sorted_class(subset,lambda r:int(r["target"]),lambda r:r["filename"],8 if role=="fit" else 100)
            mel=[]
            for i in range(0,len(use),32):
                waves=[]
                for r in use[i:i+32]:
                    sound,rate=sf.read(io.BytesIO(z.read(prefix+r["filename"])),dtype="float32")
                    if rate!=44100 or sound.ndim!=1: raise ValueError("Unexpected ESC-50 codec")
                    waves.append(sound)
                with torch.inference_mode():
                    audio=torchaudio.functional.resample(torch.from_numpy(np.stack(waves)),44100,24000)
                    spec=extractor(audio).transpose(1,2).numpy()
                at=(spec.shape[1]-4*T)//2
                mel.extend(quant(spec[:,at:at+4*T].reshape(len(waves),T,4,100).mean(2)))
            result[role]={"data":np.stack(mel),"labels":np.array([int(r["target"]) for r in use]),
                          "group":np.array([int(r["src_file"]) for r in use]),"kind":np.int32(0)}
    if set(result["fit"]["group"]) & set(result["test"]["group"]): raise ValueError("Overlapping ESC recording group")
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True); cfg=json.loads(a.config.read_bytes())["transfer"]
    for name,prepare in (("cifar10",cifar),("esc50",esc)):
        if all((a.out/f"{name}-{role}.npz").exists() for role in ("fit","dev","test")): continue
        for role,data in prepare(a.out,cfg).items():
            file=a.out/f"{name}-{role}.npz"
            with file.with_suffix(".tmp").open("wb") as out: np.savez_compressed(out,**data)
            file.with_suffix(".tmp").replace(file)
            print(name,role,len(data["labels"]),len(np.unique(data["labels"])),flush=True)


if __name__=="__main__": main()
