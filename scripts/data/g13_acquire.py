"""Acquire full immutable real-robot shards on the data SSD, with Google object generations and MD5."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request
ROOT=Path("/mnt/slwm-data/SLWM/g13/data")
PREFIX="robotics/language_table/0.0.1/"
SPEECH="https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def write(path,doc):
    tmp=path.with_suffix(".new");tmp.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n");tmp.replace(path)


def file_hash(path,kind="sha256"):
    with path.open("rb") as f:return hashlib.file_digest(f,kind).hexdigest()


def get(url,path,size=None,md5=None):
    check=path.with_suffix(path.suffix+".json")
    if path.is_file() and check.is_file():
        meta=json.loads(check.read_bytes())
        if path.stat().st_size==meta["bytes"] and meta["url"]==url:return meta
    temp=path.with_suffix(path.suffix+".part")
    for attempt in range(5):
        try:
            offset=temp.stat().st_size if temp.exists() else 0
            if not size or offset<size:
                request=urllib.request.Request(url,headers={"Range":"bytes="+str(offset)+"-"} if offset else {})
                with urllib.request.urlopen(request,timeout=100) as stream:
                    mode="ab" if offset and stream.status==206 else "wb"
                    with temp.open(mode) as f:shutil.copyfileobj(stream,f,length=1<<20)
            if size and temp.stat().st_size!=size:raise OSError("Incomplete provider object")
            if md5 and file_hash(temp,"md5")!=base64.b64decode(md5).hex():
                temp.unlink();raise OSError("Provider MD5 mismatch")
            h=file_hash(temp);temp.replace(path)
            meta=dict(url=url,sha256=h,bytes=path.stat().st_size,provider_md5=md5)
            write(check,meta);return meta
        except (TimeoutError,OSError) as exc:
            if attempt==4:raise
            print("Retry",path.name,type(exc).__name__,flush=True);time.sleep(2**attempt)


def robot(root,extra=False,prospective=False):
    folder=root/("robot-prospective/raw" if prospective else "robot-extra/raw" if extra else "robot/raw");folder.mkdir(exist_ok=True,parents=True)
    url="https://storage.googleapis.com/storage/v1/b/gresearch/o?"+urllib.parse.urlencode(dict(prefix=PREFIX+"language_table-train",maxResults=1000,fields="items(name,size,generation,md5Hash),nextPageToken"))
    catalog=json.load(urllib.request.urlopen(url,timeout=40))
    if catalog.get("nextPageToken"):raise ValueError("Incomplete shard listing")
    items=(o for o in catalog["items"] if o["name"].split("-")[-1].isdigit() and not o["name"].endswith("-00000"))
    ranked=sorted(items,key=lambda d:hashlib.sha256(("G13/robot/"+d["name"]).encode()).hexdigest())
    files=ranked[20:24] if prospective else ranked[16:20] if extra else ranked[:16]
    if len(files)!=(4 if extra or prospective else 16):raise ValueError("Incomplete registered shard sample")
    write(folder/"selection.json",files)
    for n in ("features.json","dataset_info.json"):
        get("https://storage.googleapis.com/gresearch/"+PREFIX+n,folder/n)
    def one(o):
        path=folder/o["name"].split("/")[-1]
        res=get("https://storage.googleapis.com/gresearch/"+o["name"]+"?generation="+o["generation"],path,int(o["size"]),o["md5Hash"])
        print("Verified",path.name,res["bytes"],flush=True);return res
    with ThreadPoolExecutor(5) as workers:accepted=list(workers.map(one,files))
    write(folder/"sources.json",accepted)


def speech(root):
    target=root/"speech";target.mkdir(exist_ok=True,parents=True)
    meta=get(SPEECH,target/"speech_commands_v0.02.tar.gz")
    print("Verified speech archive",meta["sha256"],meta["bytes"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("source",choices=("robot","speech","extra","prospective"));p.add_argument("--root",type=Path,default=ROOT)
    a=p.parse_args();speech(a.root) if a.source=="speech" else robot(a.root,a.source=="extra",a.source=="prospective")
