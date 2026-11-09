"""Verify the two published primary milestones and six derived parents against their Drive and disk manifests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def read(p):
    return json.loads(p.read_bytes())


def sha(p):
    with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def remote_inventory(remote):
    values=json.loads(subprocess.check_output(["rclone","lsjson","--files-only","--hash","--hash-type","SHA-256",remote+"/objects"]))
    objects={}
    for row in values:
        digest=next((v.lower() for k,v in row.get("Hashes",{}).items() if k.lower().replace("-","")=="sha256"),None)
        if digest and row["Name"]==digest:
            objects[digest]=row["Size"]
    return objects


def verify_group(name,step,files,spool,disk,remote,objects,fill):
    markers=list((spool/"receipts"/name).glob(f"{step:012d}-*.json"))
    if len(markers)!=1: raise ValueError(f"Expected exactly one {name}/{step} receipt")
    marker=markers[0]; digest=sha(marker); payload=marker.read_bytes()
    if marker.name!=f"{step:012d}-{digest}.json": raise ValueError("Local manifest SHA mismatch")
    rel=Path("runs")/name/marker.name
    if subprocess.check_output(["rclone","cat",f"{remote}/{rel}"])!=payload: raise ValueError("Drive manifest mismatch")
    manifest=read(marker)
    if (manifest["schema"],manifest["run"],manifest["step"])!=(1,name,step): raise ValueError("Manifest identity mismatch")
    total=0
    for key,entry in manifest["files"].items():
        h=entry["sha256"]; n=entry["size"]; dest=disk/"objects"/h
        if objects.get(h)!=n: raise ValueError(f"Drive object mismatch: {name}/{key}")
        source=files.get(key)
        if source is not None and (source.stat().st_size!=n or sha(source)!=h): raise ValueError(f"Local source mismatch: {key}")
        if not dest.exists() and fill:
            dest.parent.mkdir(parents=True,exist_ok=True)
            if source is not None:
                subprocess.run(["rclone","copyto","--immutable","--checksum",source,dest],check=True)
            else:
                subprocess.run(["rclone","copyto","--immutable","--checksum",f"{remote}/objects/{h}",dest],check=True)
        if dest.stat().st_size!=n or sha(dest)!=h: raise ValueError(f"Replica mismatch: {name}/{key}")
        total+=n
    committed=disk/rel
    if not committed.exists() and fill:
        committed.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=committed.parent,delete=False) as f:
            tmp=Path(f.name); f.write(payload); f.flush(); os.fsync(f.fileno())
        tmp.replace(committed)
    if committed.read_bytes()!=payload: raise ValueError("Replica commit mismatch")
    return {"manifest_sha256":digest,"files":len(manifest["files"]),"logical_bytes":total}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("physical","spool","replica","out"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--replica-uuid",required=True);p.add_argument("--remote",default="slwm-drive:SLWM/v1")
    p.add_argument("--fill-replica",action="store_true");a=p.parse_args()
    actual=subprocess.check_output(["findmnt","-nro","UUID","-T",a.replica],text=True).strip()
    if not a.replica.is_dir() or actual!=a.replica_uuid: raise ValueError("Expected mounted replica disk")
    config=Path("configs/t4/joint_transfer.json"); primary=read(config)
    if not Path(primary["output"]).samefile(a.physical): raise ValueError("Wrong checkpoint root")
    objects=remote_inventory(a.remote); reports={}
    def check(name,step,files):
        reports[f"{name}/{step}"]=verify_group(name,step,files,a.spool,a.replica,a.remote,objects,a.fill_replica)
    for step in (100,1000):
        report=a.physical/f"summary-{step}-dev.json"
        files={"summary.json":report,"training_lock.json":a.physical/"training_lock.json","config.json":config}
        for run in read(report)["rows"]:
            files[run+".pt"]=a.physical/run/f"step-{step}.pt"
            files[run+".json"]=a.physical/run/f"step-{step}.json"
        check("g3-main-exports",step,files)
    for group,directory in (("depth6","depth_parents"),("structured","structured_parents")):
        for _,(name,expected) in read(a.physical/group/"config.json")["parents"].items():
            parent=Path(primary["data"])/directory/name
            if sha(parent/"best.pt")!=expected: raise ValueError("Registered parent changed")
            # The immutable parent package holds its original config serialization.
            check(f"g3-parent-{group}-{name}",0,{"best.pt":parent/"best.pt","best.json":parent/"best.json"})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.out.with_suffix(".tmp");tmp.write_text(json.dumps(reports,indent=2,sort_keys=True)+"\n");tmp.replace(a.out)
    print(f"verified two ten-model inference milestones and six parent archives on Drive and the replica disk ({len(reports)} manifests)",flush=True)


if __name__=="__main__": main()
