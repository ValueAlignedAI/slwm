"""Immutable mixed-corpus data for T2-R1; public sources, no new codec fitting."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request
import zipfile
import zlib
import numpy as np
import torch

from training.generation_external import RangeFile
from training.t2_prepare_generation import high_mel,packetize
from training.t2_prepare_motion import encode
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json,stable_hash

ARCHIVES={"dev-clean":"42e2234ba48799c1f50f24a7926300a1","test-clean":"32fa31d27d2e1cad72775fee3f4849a9"}


def checksum(path: Path,kind: str="md5") -> str:
    h=hashlib.new(kind)
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


def flac_duration(path: Path) -> float:
    """Read STREAMINFO duration without decoding or external process launches."""
    with path.open("rb") as f:
        if f.read(4)!=b"fLaC": raise ValueError("Not native FLAC")
        header=f.read(4); data=f.read(int.from_bytes(header[1:],"big"))
    if header[0]&127 or len(data)!=34: raise ValueError("Missing FLAC STREAMINFO")
    value=int.from_bytes(data[10:18],"big"); rate=value>>44
    return (value&((1<<36)-1))/rate


def download_libri(root: Path,subset: str):
    path=root/(subset+".tar.gz"); url=f"https://www.openslr.org/resources/12/{path.name}"
    if not path.exists() or checksum(path)!=ARCHIVES[subset]:
        for attempt in range(3):
            try:
                tmp=path.with_suffix(".part")
                with urllib.request.urlopen(url,timeout=90) as source,tmp.open("wb") as target: shutil.copyfileobj(source,target,1024*1024)
                if checksum(tmp)!=ARCHIVES[subset]: raise ValueError("Official LibriSpeech MD5 mismatch")
                tmp.replace(path); break
            except Exception:
                if attempt==2: raise
                time.sleep(2)
    out=root/"raw"; marker=root/(subset+"_extracted.json")
    if not marker.exists():
        with tarfile.open(path,"r:gz") as archive:
            for member in archive:
                if not member.isfile(): continue
                target=(out/member.name).resolve()
                if not target.is_relative_to(out.resolve()): raise ValueError("Unsafe archive member")
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.extractfile(member) as source,target.open("wb") as dest: shutil.copyfileobj(source,dest)
        write_json(marker,{"archive_sha256":sha256(path),"official_md5":ARCHIVES[subset],"url":url})
    return {"subset":subset,"url":url,"md5":ARCHIVES[subset],"sha256":sha256(path)}


def select_libri(root: Path) -> dict:
    catalog={}
    for subset in ARCHIVES:
        base=root/"raw/LibriSpeech"/subset; text={}
        for path in base.rglob("*.trans.txt"):
            for line in path.read_text().splitlines():
                key,value=line.split(" ",1); text[key]=value
        rows=[]
        for path in sorted(base.rglob("*.flac")):
            seconds=flac_duration(path)
            if seconds>=1.28:
                rows.append({"sample_id":path.stem,"actor":"LIBRI-"+path.stem.split("-")[0],"raw_path":str(path),
                    "duration":seconds,"transcript":text[path.stem],"complete_transcript":seconds<=5.12,"source_subset":subset,"video_present":False})
        catalog[subset]=rows
    speakers=sorted({r["actor"] for r in catalog["dev-clean"]})
    rng=np.random.default_rng(7301); speakers=list(rng.permutation(speakers)); train=set(speakers[:32]); val=set(speakers[32:])
    order=lambda r:hashlib.sha256(("7301/"+r["sample_id"]).encode()).hexdigest()
    def take(rows,people,limit,short=False):
        selected=[]
        for actor in sorted(people):
            pool=[r for r in rows if r["actor"]==actor and (not short or r["complete_transcript"])]
            selected.extend(sorted(pool,key=order)[:limit])
        return selected
    candidates=sorted({r["actor"] for r in catalog["test-clean"] if sum(x["actor"]==r["actor"] and x["complete_transcript"] for x in catalog["test-clean"])>=4})[:12]
    groups={"libri_train":take(catalog["dev-clean"],train,40),"libri_validation":take(catalog["dev-clean"],val,8),
            "libri_test":take(catalog["test-clean"],candidates,4,True)}
    if len(train)!=32 or len(val)!=8 or len(groups["libri_test"])!=48: raise ValueError("Unexpected LibriSpeech selection")
    if (train|val)&set(candidates): raise ValueError("Official LibriSpeech speakers overlap")
    return groups


def fetch_ravdess(root: Path,actors: list[int]) -> list[dict]:
    metadata=json.load(urllib.request.urlopen("https://zenodo.org/api/records/1188976",timeout=60)); write_json(root/"ravdess_record.json",metadata)
    rows=[]
    for actor in actors:
        key=f"Video_Speech_Actor_{actor:02d}.zip"; file=next(f for f in metadata["files"] if f["key"]==key)
        reader=RangeFile(f"https://zenodo.org/records/1188976/files/{key}?download=1",file["size"])
        with zipfile.ZipFile(reader) as archive:
            selected=[]
            for member in archive.infolist():
                fields=Path(member.filename).stem.split("-")
                if len(fields)==7 and fields[0]=="01" and fields[1]=="01" and fields[3]=="01" and fields[5]=="01" and member.filename.endswith(".mp4"): selected.append(member)
            if len(selected)!=16: raise ValueError("Unexpected RAVDESS selection")
            for member in sorted(selected,key=lambda m:m.filename):
                path=root/"raw/ravdess"/Path(member.filename).name; path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists() or zlib.crc32(path.read_bytes())!=member.CRC:
                    tmp=path.with_suffix(".part"); tmp.write_bytes(archive.read(member)); tmp.replace(path)
                fields=path.stem.split("-")
                rows.append({"sample_id":path.stem,"actor":f"RAVDESS-{actor:02d}","raw_path":str(path),"raw_sha256":sha256(path),"crc32":member.CRC,
                    "transcript":"Kids are talking by the door." if fields[4]=="01" else "Dogs are sitting by the door.","complete_transcript":True,
                    "video_present":True,"sentence":"RAVDESS-"+fields[4],"archive_declared_md5":file["checksum"]})
            print(json.dumps({"new_ravdess_actor":actor,"downloaded":len(rows)}),flush=True)
        del reader
    return rows


def prepare_arrays(root: Path,split: str,rows: list[dict],cfg: dict):
    marker=root/(split+"_manifest.json")
    if marker.exists():
        manifest=json.loads(marker.read_text())
        if manifest["config_hash"]!=stable_hash(cfg): raise ValueError("Use a new corpus version")
        for name,digest in manifest["files"].items():
            if sha256(root/name)!=digest: raise ValueError("Prepared data changed")
        return
    with np.load(Path(cfg["old_corpus"])/"codec.npz",allow_pickle=False) as d: old={k:d[k] for k in d.files}
    with np.load(Path(cfg["generation_corpus"])/"codec100.npz",allow_pickle=False) as d: mean,scale=d["mean"],d["scale"]
    cache=root/"decoded"; cache.mkdir(exist_ok=True)
    def decode(row):
        path=cache/(row["sample_id"]+".npz")
        if path.exists():
            with np.load(path,allow_pickle=False) as d: return d["mel"],d["video"],int(d["length"])
        mel=high_mel(Path(row["raw_path"]))
        length=min(128,len(mel)//4)
        if row["video_present"]:
            visual=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",row["raw_path"],"-an","-vf",cfg["external_crop"],"-frames:v","128",
                "-f","rawvideo","-pix_fmt","gray","-threads","1","-"],capture_output=True,check=True,timeout=45)
            mouth=np.frombuffer(visual.stdout,np.uint8).reshape(-1,32,48).copy(); length=min(length,len(mouth))
            _,video=encode(mouth[:length],mel[:4*length:4],old)
        else: video=np.zeros((length,160),np.float32)
        if length<32: raise ValueError("Short decoded clip")
        packed,_=packetize(mel,length); packed=((packed.reshape(-1,64)-mean)/scale).reshape(length,256)
        np.savez_compressed(path,mel=packed.astype(np.float16),video=video.astype(np.float16),length=np.array(length))
        return packed,video,length
    n=len(rows); arrays={"mel100":np.zeros((n,128,256),np.float16),"video":np.zeros((n,128,160),np.float16),"lengths":np.zeros(n,np.int32)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(mel,video,length) in enumerate(pool.map(decode,rows)):
            arrays["mel100"][i,:length]=mel; arrays["video"][i,:length]=video; arrays["lengths"][i]=length
            rows[i]={**rows[i],"frames":length,"raw_sha256":sha256(Path(rows[i]["raw_path"]))}
    files={}
    for name,value in arrays.items():
        path=root/f"{split}_{name}.npy"; np.save(path,value); files[path.name]=sha256(path)
    write_json(root/(split+".json"),rows); files[split+".json"]=sha256(root/(split+".json"))
    write_json(marker,{"config_hash":stable_hash(cfg),"samples":n,"people":sorted({r["actor"] for r in rows}),"files":files,
        "codec_fit":"Existing CREMA training-only statistics; no LibriSpeech/RAVDESS fit","field_cache":False})
    print(json.dumps({"prepared":split,"samples":n}),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--external-only",action="store_true"); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["new_corpus"]); root.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1)
    if args.external_only:
        rows=fetch_ravdess(root,cfg["external_actors"]); prepare_arrays(root,"ravdess_test",rows,cfg); return
    with ThreadPoolExecutor(max_workers=2) as pool: archives=list(pool.map(lambda s:download_libri(root,s),ARCHIVES))
    selection=select_libri(root); write_json(root/"selection.json",selection)
    write_json(root/"source_card.json",{"source":"https://www.openslr.org/12","authors":"Panayotov, Chen, Povey & Khudanpur (2015)","license":"CC BY 4.0",
        "archives":archives,"selection_protocol":"32 dev-clean training speakers / remaining 8 validation, seed 7301; 12 test-clean speakers with >=4 complete <=5.12s clips",
        "no_external_fit":True,"new_ravdess_license":"CC BY-NC-SA 4.0, Livingstone & Russo (2018); held out only"})
    for split,rows in selection.items(): prepare_arrays(root,split,rows,cfg)


if __name__=="__main__": main()
