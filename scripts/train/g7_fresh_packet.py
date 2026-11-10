"""Verify physical records and paired mask equivalence before exposing the fresh continuation packet."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from training.g4_data import digest
from training.g5_batches import JointData
from training.g7_batches import ContinuationData,FAMILIES


def prepare(root,older,role):
    base=root/"fresh"; files=[base/v/role/"card.json" for v in FAMILIES.values()]
    while not all(f.is_file() for f in files):time.sleep(12)
    card={}
    for f,p in zip(FAMILIES,files):
        d=json.loads(p.read_bytes());folder=p.parent
        for k,expected in d["sha256"].items():
            if digest(folder/(k+".npy"))!=expected:raise ValueError(f"Corrupt independent pool {f}/{k}")
        arr=np.load(folder/("group.npy" if f==8 else "uid.npy"),mmap_mode="r")
        if len(np.unique(arr))!=len(arr):raise ValueError("Duplicate independent source row")
        card[FAMILIES[f]]={"rows":len(arr),"groups":len(np.unique(np.load(folder/"group.npy",mmap_mode="r"))),"card_sha256":digest(p)}
        if (folder/"length.npy").exists():card[FAMILIES[f]]["full_64_tick_rows"]=int((np.load(folder/"length.npy")==64).sum())
        if role!="train":
            old=set(np.load(folder.parent/"train/group.npy",mmap_mode="r").tolist())
            if old.intersection(np.load(folder/"group.npy",mmap_mode="r").tolist()):raise ValueError("New training and held groups overlap")
    if role=="train":
        old=JointData(older,"train",verify=False); new=ContinuationData(old,base); counts=np.zeros(9,np.int64);visits=np.zeros(9,np.int64)
        for step in range(8192,8320):
            a=old.requests(step,256); b=new.requests(step,256)
            for k in ("source","has","family","ok","sense"):np.testing.assert_array_equal(a[k],b[k])
            changed=np.zeros(256,bool)
            for k in ("a","v","x","n","u","event","outcome"):
                changed|=(a[k]!=b[k]).reshape(256,-1).any(1)
            counts+=np.bincount(a["family"][changed],minlength=9);visits+=np.bincount(a["family"],minlength=9)
        if any(counts[k]==0 for k in FAMILIES):raise ValueError("A declared independent source was never presented")
        card["alignment_audit"]={"rows":int(sum(visits)),"changed_by_family":counts.tolist(),"visits_by_family":visits.tolist(),
                                 "family_has_source_valid_and_sense_masks_identical":True,"replay":old.identity,"fresh":new.identity}
    out=root/"packets";out.mkdir(exist_ok=True)
    meta=out/(role+".json");tmp=meta.with_suffix(".new");tmp.write_text(json.dumps(card,sort_keys=True,indent=2)+"\n");tmp.replace(meta)
    archive=out/f"g7-fresh-{role}.tar.zst"
    if not archive.exists():
        part=archive.with_suffix(".partial")
        subprocess.run(["tar","-cf",str(part),"--use-compress-program","zstd -q -1 -T2","-C",str(base),*[v+"/"+role for v in FAMILIES.values()]],check=True)
        part.replace(archive)
    sha=digest(archive)
    done=out/(role+".verified.json");tmp=done.with_suffix(".tmp");tmp.write_text(json.dumps({"file":archive.name,"sha256":sha,"bytes":archive.stat().st_size,"card":digest(meta)})+"\n");tmp.replace(done)
    print(role,archive.stat().st_size,sha,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--old",type=Path,required=True);a=p.parse_args()
    for role in ("train","dev","test"):prepare(a.root,a.old,role)
