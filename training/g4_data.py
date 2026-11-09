"""Pack real G4 examples into fixed-size arrays on the data SSD; all clocks and roles are explicit."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


T = 64
DT = 1024 / 24000
CFG = Path("configs/t4/crossmodal_tpu.json")
FAMILIES = ("grid","libri","coco","clotho","droid","text","ami")
FIELDS = {"a": (np.uint8,(T,100)), "v": (np.uint8,(T,256)), "n": (np.float16,(T,17)),
          "x": (np.uint16,(T,)), "ok": (np.uint8,(T,)), "has": (np.uint8,(4,)),
          "family": (np.uint8,()), "group": (np.int32,()), "cap": (np.int32,()), "uid": (np.uint64,())}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def number(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8],"little")


def tokens(text):
    return np.array([b+3 for b in text.strip().encode("utf8")[:T-1]]+[2], np.uint16)


def quant(mel):
    return np.rint((np.clip(mel,-12,3.9375)+12)*16).astype(np.uint8)


@lru_cache(maxsize=24)
def mapped(path):
    return np.load(path,mmap_mode="r",allow_pickle=False)


def gray(image):
    if image.ndim == 4:
        image = image[...,0]*.299 + image[...,1]*.587 + image[...,2]*.114
    frames, h, w = image.shape
    return np.rint(image.reshape(frames,16,h//16,16,w//16).mean((2,4))).astype(np.uint8).reshape(frames,256)


def raster(images):
    """64x64 RGB -> 64 rasterized 8x8 grayscale patches, replicated 2x for the shared visual edge."""
    g = images[...,0]*.299 + images[...,1]*.587 + images[...,2]*.114
    patches = g.reshape(len(g),8,8,8,8).transpose(0,1,3,2,4).reshape(-1,64,8,8)
    return np.rint(patches).astype(np.uint8).repeat(2,axis=2).repeat(2,axis=3).reshape(-1,T,256)


def select(rows, role, cap, key):
    order = sorted((row for row in rows if row["role"]==role),key=lambda r:number(f"{key}/{r['id']}"))
    return order[:cap]


class Writer:
    def __init__(self, dest, capacity):
        dest.mkdir(parents=True, exist_ok=False)
        self.dest, self.index = dest, 0
        self.arrays = {k:np.lib.format.open_memmap(dest/f"{k}.npy",mode="w+",dtype=typ,shape=(capacity,*shape))
                       for k,(typ,shape) in FIELDS.items()}
        self.groups = {f:set() for f in FAMILIES}

    def add(self, family, uid, group, text="", audio=None, video=None, numeric=None, length=T):
        j = self.index
        n = min(length,T)
        self.arrays["ok"][j,:n] = 1
        present = [audio is not None,video is not None,numeric is not None,bool(text.strip())]
        self.arrays["has"][j] = present
        self.arrays["family"][j] = FAMILIES.index(family)
        self.arrays["uid"][j] = number(uid)
        self.arrays["group"][j] = number(group) % (2**31-1)
        self.arrays["cap"][j] = number(" ".join(text.lower().split())) % (2**31-1)
        self.groups[family].add(str(group))
        if text.strip():
            ids = tokens(text)
            self.arrays["x"][j,:len(ids)] = ids
        for key,value in (("a",audio),("v",video),("n",numeric)):
            if value is not None:
                self.arrays[key][j,:n] = value[:n]
        self.index += 1

    def finish(self, role):
        n = self.index
        if n == 0:
            raise ValueError("Empty data role")
        fam = self.arrays["family"][:n]
        if len(np.unique(self.arrays["uid"][:n])) != n:
            raise ValueError("Duplicate example identity")
        stats = {"role":role,"examples":n,"valid_positions":int(self.arrays["ok"][:n].sum()),
                 "families":{f:int((fam==i).sum()) for i,f in enumerate(FAMILIES)},
                 "groups":{f:len(groups) for f,groups in self.groups.items()}}
        # Reserve upper bounds while packing and compact every finalized file to its actual row count.
        self.arrays = {}
        for key in FIELDS:
            file = self.dest/f"{key}.npy"
            src = np.load(file,mmap_mode="r")
            if len(src) != n:
                target=self.dest/f"{key}.compact.npy"
                out=np.lib.format.open_memmap(target,mode="w+",dtype=src.dtype,shape=(n,*src.shape[1:]))
                for j in range(0,n,512):
                    out[j:j+512]=src[j:min(n,j+512)]
                del out
                target.replace(file)
            del src
        stats["sha256"]={k:digest(self.dest/f"{k}.npy") for k in FIELDS}
        (self.dest/"card.json").write_text(json.dumps(stats,sort_keys=True,indent=2)+"\n")
        return stats


def grid(writer, root, role, cap, source_split=None):
    import torch
    import torchaudio
    from vocos.feature_extractors import MelSpectrogramFeatures
    torch.set_num_threads(3)
    split=source_split if source_split is not None else {"train":"train","dev":"validation","test":"test"}[role]
    original=read(root/(split+".json"))
    ids=sorted(range(len(original)),key=lambda j:number(original[j]["sample_id"]))[:cap]
    mouth=mapped(root/(split+"_mouth.npy"))
    extract=MelSpectrogramFeatures(sample_rate=24000,n_fft=1024,hop_length=256,n_mels=100)
    for base in range(0,len(ids),48):
        batch=ids[base:base+48]
        wave=np.zeros((len(batch),48000),np.float32)
        sizes=[]
        for k,j in enumerate(batch):
            with np.load(original[j]["decoded_path"]) as rec:
                pcm=rec["pcm"][:48000]
            wave[k,:len(pcm)]=pcm.astype(np.float32)/32768
            sizes.append(len(pcm))
        with torch.inference_mode():
            hi=torchaudio.functional.resample(torch.from_numpy(wave),16000,24000)
            mel=extract(hi).transpose(1,2).numpy()
        for k,j in enumerate(batch):
            row=original[j]
            duration=min(row["frames"]/25,sizes[k]/16000)
            count=min(T,max(1,int(duration/DT)))
            start=max(0,int((duration/DT-count)/2))
            clock=(start+np.arange(count)+.5)*DT
            vi=np.minimum((clock*25).astype(int),row["frames"]-1)
            audio=quant(mel[k,4*start:4*(start+count)].reshape(count,4,100).mean(1))
            writer.add("grid",row["sample_id"],row["actor"],row["sentence"],audio,gray(mouth[j,vi]),length=count)


def pack_role(repo,cfg,dest,role):
    src={k:repo/v for k,v in cfg["sources"].items()}
    cap=dict(zip(FAMILIES,cfg[role+"_caps"]))
    w=Writer(dest,sum(cap.values()))
    original_role={"train":"train","dev":"selection","test":"test"}[role]
    grid(w,src["grid"],role,cap["grid"])
    print(role,"grid",w.index,flush=True)
    for family in ("libri","coco","clotho","droid","ami"):
        if not cap[family]:
            continue
        folder={"libri":"speech"}.get(family,family)
        metadata=src["g2"]/"robot_rows.json" if family=="droid" else src["cache"]/folder/"rows.json"
        rows=select(read(metadata),original_role,cap[family],family)
        limit=w.index+cap[family]
        if family=="coco":
            for base in range(0,len(rows),128):
                sample=rows[base:base+128]
                pixels=np.stack([mapped(src["cache"]/folder/r["array"])[r["index"]] for r in sample])
                for r,v in zip(sample,raster(pixels)):
                    w.add(family,r["id"],r["group"],r["captions"][number(r["id"])%len(r["captions"])],video=v)
            continue
        for r in rows:
            if w.index>=limit:
                break
            if family=="clotho":
                m=mapped(src["cache"]/folder/r["file"]).reshape(-1,100)
                at=np.linspace(0,len(m),T+1,dtype=int)
                a=quant(np.stack([m[l:max(l+1,h)].mean(0,dtype=np.float32) for l,h in zip(at,at[1:])]))
                w.add(family,r["id"],r["group"],r["captions"][number(r["id"])%len(r["captions"])],audio=a)
                continue
            if family=="droid":
                path=src["cache"]/"droid_g2"/r["id"]
                with np.load(path/"trajectory.npz") as p:
                    data=np.concatenate([p["state"],p["action"]],axis=1).astype(np.float32)
                if not np.isfinite(data).all():
                    continue
                pixels=mapped(path/"rgb.npy")
                length=min(len(data),len(pixels))
            else:
                length=r["length"]
                mel=mapped(repo/r["mel_file"])
                if family=="ami":
                    pixels=mapped(repo/r["pixel_file"])
                    length=min(length,len(pixels))
            repeats=8 if family=="droid" else (max(1,cap[family]//max(1,len(rows))) if family=="ami" else 1)
            if length<12:
                continue
            count=min(T,length)
            starts=np.unique(np.linspace(0,length-count,repeats+2,dtype=int)[1:-1]) if repeats>1 else [number(r["id"])%(length-count+1)]
            for start in starts:
                if w.index>=limit:
                    break
                sl=slice(start,start+count)
                uid=f"{family}/{r['id']}/{start}"
                if family=="droid":
                    desc=r["captions"][number(uid)%len(r["captions"])] if r["captions"] else ""
                    w.add(family,uid,r["group"],desc,video=gray(pixels[sl,0]),numeric=data[sl],length=count)
                else:
                    m=np.asarray(mel[r["offset"]+start:r["offset"]+start+count],np.float32).reshape(count,4,100).mean(1)
                    txt=" ".join(word for l,h,word in r["spans"] if l>=start*DT and h<=(start+count)*DT and "<" not in word)
                    v=gray(pixels[sl]) if family=="ami" else None
                    w.add(family,uid,r["group"],txt,audio=quant(m),video=v,length=count)
        print(role,family,w.index,flush=True)
    import tiktoken
    enc=tiktoken.get_encoding("gpt2")
    split={"train":"train","dev":"validation","test":"test"}[role]
    corpus=np.memmap(src["text"]/(split+".tokens.bin"),dtype="<u2",mode="r")
    blocks=len(corpus)//64
    positions=np.random.default_rng(number("g4-text-"+role)).choice(blocks,cap["text"],replace=False)
    for index in positions:
        ids=corpus[index*64:(index+1)*64].tolist()
        text=enc.decode(ids).strip()
        if text:
            w.add("text",f"text/{role}/{index}",f"text/{role}/{index}",text)
    card=w.finish(role)
    print(json.dumps(card["families"]),flush=True)
    return card


def live(repo,cfg,dest):
    cache=repo/cfg["sources"]["cache"]
    r=next(r for r in sorted(read(cache/"ami/rows.json"),key=lambda x:number("live/"+x["id"]))
           if r["role"]=="test" and r["length"]>=3600)
    count=1600
    start=min(1200,r["length"]-count)
    m=mapped(repo/r["mel_file"])[r["offset"]+start:r["offset"]+start+count].reshape(count,4,100).mean(1,dtype=np.float32)
    v=gray(mapped(repo/r["pixel_file"])[start:start+count])
    spans=[(a-start*DT,b-start*DT,s) for a,b,s in r["spans"] if a>=start*DT and b<=(start+count)*DT]
    np.savez(dest/"live.npz",a=quant(m),v=v,spans=np.array(json.dumps(spans)),dt=DT,group=np.array(r["group"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--role",choices=["train","dev","test"],required=True)
    a=p.parse_args(); cfg=read(a.config)
    os.chdir(a.repo)
    a.output.mkdir(parents=True,exist_ok=True)
    dest=a.output/a.role
    if not (dest/"card.json").exists():
        with tempfile.TemporaryDirectory(dir=a.output,prefix=".build-") as tmp:
            new=Path(tmp)/a.role
            pack_role(a.repo,cfg,new,a.role)
            if a.role=="test":
                live(a.repo,cfg,new)
            new.rename(dest)
    archive=a.output/(a.role+".tar.zst")
    if not archive.exists():
        part=archive.with_suffix(".partial")
        subprocess.run(["tar","--create","--file",part,"--use-compress-program","zstd -T2 -1 -q",
                        "--directory",a.output,a.role],check=True)
        part.replace(archive)
    print(json.dumps({"role":a.role,"archive_bytes":archive.stat().st_size,"sha256":digest(archive)}),flush=True)


if __name__=="__main__":
    main()
