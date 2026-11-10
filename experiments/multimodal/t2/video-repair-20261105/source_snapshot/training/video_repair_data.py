"""Real mouth pixels and speaker-disjoint paired data for T2-V1."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error
import zipfile
import zlib
import numpy as np
import torch
from training.generation_external import RangeFile
from training.t2_prepare_generation import high_mel,packetize
from training.t2_prepare_motion import encode
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json,stable_hash

_last_public_request=0.


def public_request(request):
    """Respect source rate limits and Retry-After; never change hosts to bypass."""
    global _last_public_request
    for attempt in range(6):
        time.sleep(max(0.,1.5-(time.monotonic()-_last_public_request)))
        _last_public_request=time.monotonic()
        try: return urllib.request.urlopen(request,timeout=90)
        except urllib.error.HTTPError as exc:
            if exc.code!=429 and exc.code<500: raise
            raw=exc.headers.get("Retry-After",""); delay=float(raw) if raw.isdigit() else 30.*(attempt+1)
            print(json.dumps({"public_source_retry":exc.code,"wait_seconds":delay}),flush=True)
            time.sleep(delay)
    raise RuntimeError("Public source retry budget exhausted; resume later")


class VerifiedRanges(RangeFile):
    """Range-aware cache keys and exact Content-Range validation for public ZIPs."""
    def __init__(self,url,size):
        super().__init__(url,size); self.block_size=8*1024*1024

    def read(self,n=-1):
        n=self.size-self.position if n<0 else min(n,self.size-self.position)
        if n<=0: return b""
        pieces=[]
        while n:
            block=self.position//self.block_size; start=block*self.block_size; end=min(start+self.block_size,self.size)-1
            if block not in self.blocks:
                for attempt in range(3):
                    request=urllib.request.Request(self.url,headers={"Range":f"bytes={start}-{end}"})
                    with public_request(request) as response:
                        payload=response.read(); actual=response.headers.get("Content-Range")
                    if actual==f"bytes {start}-{end}/{self.size}" and len(payload)==end-start+1: break
                    if attempt==2: raise ValueError(f"Public server range mismatch: {actual}")
                self.blocks[block]=payload
            chunk=self.blocks[block]; offset=self.position-start; take=min(n,len(chunk)-offset)
            pieces.append(chunk[offset:offset+take]); self.position+=take; n-=take
        return b"".join(pieces)


def fetch_actor(root: Path,actor: int,full: bool,metadata: dict) -> list[dict]:
    key=f"Video_Speech_Actor_{actor:02d}.zip"; f=next(x for x in metadata["files"] if x["key"]==key)
    rows=[]
    for attempt in range(3):
        try:
            reader=VerifiedRanges(f"https://zenodo.org/records/1188976/files/{key}?download=1",f["size"])
            with zipfile.ZipFile(reader) as archive:
                selected=[]
                for member in archive.infolist():
                    parts=Path(member.filename).stem.split("-")
                    if len(parts)==7 and parts[:2]==["01","01"] and member.filename.endswith(".mp4") and (full or (parts[3]=="01" and parts[5]=="01")): selected.append(member)
                if len(selected)!=(60 if full else 16): raise ValueError("RAVDESS selection changed")
                rows=[]
                for member in sorted(selected,key=lambda m:m.filename):
                    path=root/"raw"/Path(member.filename).name; path.parent.mkdir(exist_ok=True)
                    if not path.exists() or zlib.crc32(path.read_bytes())!=member.CRC:
                        temporary=path.with_suffix(".part"); temporary.write_bytes(archive.read(member)); temporary.replace(path)
                    parts=path.stem.split("-")
                    rows.append({"sample_id":path.stem,"actor":f"RAVDESS-{actor:02d}","raw_path":str(path),"raw_sha256":sha256(path),"crc32":member.CRC,
                        "source":"RAVDESS","sentence":"RAVDESS-"+parts[4],"transcript":"Kids are talking by the door." if parts[4]=="01" else "Dogs are sitting by the door.",
                        "archive_declared_md5":f["checksum"],"license":"CC BY-NC-SA 4.0"})
            del reader
            print(json.dumps({"fetched_actor":actor,"full_training":full,"samples":len(rows)}),flush=True); return rows
        except Exception:
            if attempt==2: raise
            time.sleep(2)
    raise RuntimeError("Unreachable")


def prepare(cfg: dict,split: str,rows: list[dict],references: dict|None=None):
    root=Path(cfg["corpus"]); manifest=root/(split+"_manifest.json")
    if manifest.exists():
        existing=json.loads(manifest.read_text())
        if existing["config_hash"]!=stable_hash(cfg): raise ValueError("Use a new corpus version")
        for file in existing["arrays"].values():
            if sha256(Path(file["path"]))!=file["sha256"]: raise ValueError("Prepared data changed")
        return
    parent=json.loads(Path(cfg["restoration_config"]).read_text()); cache=root/"decoded"; cache.mkdir(exist_ok=True)
    with np.load(Path(parent["old_corpus"])/"codec.npz",allow_pickle=False) as d: old={k:d[k] for k in d.files}
    with np.load(Path(parent["generation_corpus"])/"codec100.npz",allow_pickle=False) as d: mean,scale=d["mean"],d["scale"]
    def decode(row):
        path=cache/(row["sample_id"]+".npz")
        if path.exists():
            with np.load(path,allow_pickle=False) as d: return {k:d[k] for k in d.files}
        filt=cfg["crema_crop"] if row.get("source")=="CREMA" else cfg["ravdess_crop"]
        raw=subprocess.run(["ffmpeg","-v","error","-threads","1","-i",row["raw_path"],"-an","-vf",filt,"-frames:v","128","-f","rawvideo","-pix_fmt","gray","-threads","1","-"],capture_output=True,check=True,timeout=45)
        pixels=np.frombuffer(raw.stdout,np.uint8).reshape(-1,64,96).copy(); arrays={"mouth":pixels}
        if references is None:
            mel=high_mel(Path(row["raw_path"])); length=min(len(pixels),len(mel)//4,128); packed,_=packetize(mel,length)
            # Fixed old video PCA is an ablation input, never the new CNN input.
            from PIL import Image
            coarse=np.stack([np.asarray(Image.fromarray(p).resize((48,32))) for p in pixels[:length]])
            _,video=encode(coarse,mel[:4*length:4],old)
            arrays.update({"mel100":((packed.reshape(-1,64)-mean)/scale).reshape(length,256).astype(np.float16),"video":video.astype(np.float16)})
        if len(pixels)<32: raise ValueError("Short mouth sequence")
        temporary=path.with_suffix(".part.npz"); np.savez_compressed(temporary,**arrays); temporary.replace(path); return arrays
    n=len(rows); paths={}; mouth_path=root/(split+"_mouth.npy"); mouth=np.lib.format.open_memmap(mouth_path,mode="w+",dtype=np.uint8,shape=(n,128,64,96))
    mouth[:]=0; lengths=np.zeros(n,np.int32)
    if references:
        target_lengths=np.load(references["lengths"])
    else:
        mel=np.zeros((n,128,256),np.float16); video=np.zeros((n,128,160),np.float16)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,decoded in enumerate(pool.map(decode,rows)):
            length=min(len(decoded["mouth"]),int(target_lengths[i]) if references else len(decoded["mel100"]),128)
            if length<32: raise ValueError("Short target")
            mouth[i,:length]=decoded["mouth"][:length]; lengths[i]=length; rows[i]={**rows[i],"frames":length}
            if not references: mel[i,:length]=decoded["mel100"][:length]; video[i,:length]=decoded["video"][:length]
            if (i+1)%500==0: print(json.dumps({"decoded":i+1,"split":split}),flush=True)
    mouth.flush(); del mouth
    paths["mouth"]=str(mouth_path); lengths_path=root/(split+"_lengths.npy"); np.save(lengths_path,lengths); paths["lengths"]=str(lengths_path)
    for name in ("mel100","video"):
        if references: paths[name]=references[name]
        else:
            path=root/(split+"_"+name+".npy"); np.save(path,mel if name=="mel100" else video); paths[name]=str(path)
    write_json(root/(split+".json"),rows)
    arrays={name:{"path":path,"sha256":sha256(Path(path))} for name,path in paths.items()}
    write_json(manifest,{"sprint":"T2-V1","config_hash":stable_hash(cfg),"samples":n,"people":sorted({r["actor"] for r in rows}),"arrays":arrays,
        "metadata_sha256":sha256(root/(split+".json")),"resolution":[64,96],"codec_fitting":"None; reuse CREMA-training statistics"})
    print(json.dumps({"prepared":split,"samples":n}),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--stage",choices=("crema","train_rav","val_rav","test_rav"),required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["corpus"]); root.mkdir(parents=True,exist_ok=True); torch.set_num_threads(1)
    parent=json.loads(Path(cfg["restoration_config"]).read_text())
    if args.stage=="crema":
        source=Path(parent["generation_corpus"])
        for split in ("train","validation","test"):
            rows=[{**r,"source":"CREMA"} for r in json.loads((source/(split+".json")).read_text())]
            prepare(cfg,"crema_"+split,rows,{"mel100":str(source/(split+"_mel100.npy")),"video":str(source/(split+"_video_coarse.npy")),"lengths":str(source/(split+"_lengths.npy"))})
    elif args.stage=="val_rav":
        source=Path(parent["new_corpus"]); rows=json.loads((source/"ravdess_test.json").read_text())
        prepare(cfg,"rav_validation",rows,{"mel100":str(source/"ravdess_test_mel100.npy"),"video":str(source/"ravdess_test_video.npy"),"lengths":str(source/"ravdess_test_lengths.npy")})
    else:
        full=args.stage=="train_rav"; actors=cfg["train_ravdess_actors"] if full else cfg["test_ravdess_actors"]
        with public_request("https://zenodo.org/api/records/1188976") as response: metadata=json.load(response)
        write_json(root/(args.stage+"_source.json"),metadata)
        groups=[fetch_actor(root,actor,full,metadata) for actor in actors]
        prepare(cfg,"rav_train" if full else "rav_test",[r for group in groups for r in group])


if __name__=="__main__": main()
