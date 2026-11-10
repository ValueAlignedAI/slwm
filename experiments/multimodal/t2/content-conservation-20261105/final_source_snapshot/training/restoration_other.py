"""New official LibriSpeech test-other holdout, reserved before R2 scoring."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
from training.restoration_data import checksum,flac_duration,prepare_arrays
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json


def expanded_config(path: str) -> dict:
    extra=json.loads(Path(path).read_text()); parent=json.loads(Path(extra["parent_config"]).read_text())
    return {**parent,**extra,"training":parent["training"]}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args(); cfg=expanded_config(args.config)
    root=Path(cfg["other_corpus"]); root.mkdir(parents=True,exist_ok=True); archive=root/"test-other.tar.gz"; expected="fb5a50374b501bb3bac4815ee91d3135"
    if not archive.exists() or checksum(archive)!=expected:
        tmp=archive.with_suffix(".part")
        with urllib.request.urlopen("https://www.openslr.org/resources/12/test-other.tar.gz",timeout=90) as source,tmp.open("wb") as target: shutil.copyfileobj(source,target,1024*1024)
        if checksum(tmp)!=expected: raise ValueError("Official test-other MD5 mismatch")
        tmp.replace(archive)
    if not (root/"extracted.json").exists():
        with tarfile.open(archive,"r:gz") as tar:
            for member in tar:
                if not member.isfile(): continue
                path=(root/"raw"/member.name).resolve()
                if not path.is_relative_to((root/"raw").resolve()): raise ValueError("Unsafe archive member")
                path.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,path.open("wb") as target: shutil.copyfileobj(source,target)
        write_json(root/"extracted.json",{"official_md5":expected,"sha256":sha256(archive)})
    base=root/"raw/LibriSpeech/test-other"; text={}
    for path in base.rglob("*.trans.txt"):
        for line in path.read_text().splitlines():
            name,words=line.split(" ",1); text[name]=words
    rows=[]
    for path in sorted(base.rglob("*.flac")):
        seconds=flac_duration(path)
        if 1.28<=seconds<=5.12: rows.append({"sample_id":path.stem,"actor":"LIBRI-"+path.stem.split("-")[0],"raw_path":str(path),"duration":seconds,
            "transcript":text[path.stem],"complete_transcript":True,"source_subset":"test-other","video_present":False})
    speakers=sorted(a for a in {r["actor"] for r in rows} if sum(r["actor"]==a for r in rows)>=4)[:12]
    used=set()
    for split in ("libri_train","libri_validation","libri_test"):
        used.update(r["actor"] for r in json.loads((Path(cfg["new_corpus"])/(split+".json")).read_text()))
    if set(speakers)&used: raise ValueError("New speakers overlap prior Libri data")
    selected=[]
    for actor in speakers:
        selected.extend(sorted([r for r in rows if r["actor"]==actor],key=lambda r:hashlib.sha256(("R2-other/"+r["sample_id"]).encode()).hexdigest())[:4])
    if len(selected)!=48: raise ValueError("Unexpected eligible test-other size")
    write_json(root/"source_card.json",{"source":"https://www.openslr.org/12","license":"CC BY 4.0","authors":"Panayotov et al. (2015)","archive_md5":expected,
        "archive_sha256":sha256(archive),"purpose":"fresh R2 evaluation only; no fitting or selection","selected":48,"people":speakers})
    prepare_arrays(root,"other_test",selected,cfg)


if __name__=="__main__": main()
