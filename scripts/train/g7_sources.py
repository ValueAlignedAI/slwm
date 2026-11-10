"""Acquire pinned public recordings and unpack only the verified eligible LibriSpeech members."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import zipfile


def file_sha(path,mode="sha256"):
    with path.open("rb") as f: return hashlib.file_digest(f,mode).hexdigest()


def write(path,doc):
    temp=path.with_suffix(".new"); temp.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n"); temp.replace(path)


def starss(root):
    cfg=json.loads(Path("configs/t7/study.json").read_bytes()); target=root/"starss"; target.mkdir(parents=True,exist_ok=True)
    def fetch(item):
        name,md5=item; path=target/name
        if not path.is_file():
            part=path.with_suffix(".partial")
            url=f"https://zenodo.org/records/{cfg['starss_record']}/files/{name}?download=1"
            subprocess.run(["curl","-fsSL","--retry","4","--retry-delay","8","--connect-timeout","20",
                            "--speed-time","150","--speed-limit","2048","-C","-","-o",str(part),url],check=True)
            if file_sha(part,"md5")!=md5: raise ValueError(f"Publisher checksum mismatch for {name}")
            part.replace(path)
        if file_sha(path,"md5")!=md5: raise ValueError(f"Changed source {name}")
        return name,{"bytes":path.stat().st_size,"md5":md5,"sha256":file_sha(path)}
    with ThreadPoolExecutor(max_workers=3) as pool: records=dict(pool.map(fetch,cfg["starss_md5"].items()))
    write(target/"source.json",{"doi":"10.5281/zenodo.7880637","source":"https://zenodo.org/records/7880637","files":records})
    print("Verified STARSS23",records,flush=True)


def libri(root,g5):
    source=g5/"sources/train-clean-360.tar.gz"
    expected=(g5/"sources/train-clean-360.tar.gz.sha256").read_text().split()[0]
    if file_sha(source)!=expected: raise ValueError("LibriSpeech archive checksum changed")
    folder=root/"libri";folder.mkdir(parents=True,exist_ok=True)
    with gzip.open(g5/"libri360-eligible.jsonl.gz","rt") as f: rows=[json.loads(line) for line in f]
    by_member={r["member"]:r for r in rows}; seen=set()
    with tarfile.open(source,mode="r|gz") as archive:
        for item in archive:
            row=by_member.get(item.name)
            if row is None: continue
            if not item.isfile() or row["id"]!=Path(item.name).stem: raise ValueError("Unexpected eligible member")
            file=folder/"audio"/row["speaker"]/(row["id"]+".flac"); file.parent.mkdir(exist_ok=True,parents=True)
            if not file.exists() or file.stat().st_size!=item.size:
                part=file.with_suffix(".part")
                with archive.extractfile(item) as reader,part.open("wb") as out: shutil.copyfileobj(reader,out)
                if part.stat().st_size!=item.size: raise EOFError("Truncated speech member")
                part.replace(file)
            seen.add(item.name)
    if seen!=set(by_member): raise ValueError("Eligible audio inventory incomplete")
    people=sorted({r["speaker"] for r in rows},key=lambda s:hashlib.sha256(("g7-20261110-roles-v1/libri/"+s).encode()).digest())
    n=len(people)//10
    roles={s:("validation" if i<n else "test" if i<2*n else "train") for i,s in enumerate(people)}
    with gzip.open(folder/"rows.jsonl.gz.new","wt") as out:
        for r in rows:
            r=dict(r,role=roles[r["speaker"]]);out.write(json.dumps(r,sort_keys=True)+"\n")
    (folder/"rows.jsonl.gz.new").replace(folder/"rows.jsonl.gz")
    write(folder/"source.json",{"source_sha256":expected,"utterances":len(rows),"manifest_sha256":file_sha(folder/"rows.jsonl.gz"),
        "roles":{role:{"speakers":sum(v==role for v in roles.values()),"utterances":sum(roles[r["speaker"]]==role for r in rows)} for role in ("train","validation","test")}})
    print((folder/"source.json").read_text(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("phase",choices=("libri","starss"));p.add_argument("--root",type=Path,required=True)
    p.add_argument("--g5",type=Path,default=Path("/mnt/slwm-data/SLWM/g5"));a=p.parse_args()
    if subprocess.check_output(["findmnt","-nro","TARGET","-T",a.root],text=True).strip()!="/mnt/slwm-data":
        raise RuntimeError("Use the registered study SSD")
    (libri(a.root,a.g5) if a.phase=="libri" else starss(a.root))
