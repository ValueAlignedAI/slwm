"""Verify G3 inference milestones and six derived parents against their Drive and disk manifests."""
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


def verify_group(name,step,files,spool,disk,remote,objects,fill,exact=False):
    markers=list((spool/"receipts"/name).glob(f"{step:012d}-*.json"))
    if len(markers)!=1: raise ValueError(f"Expected exactly one {name}/{step} receipt")
    marker=markers[0]; digest=sha(marker); payload=marker.read_bytes()
    if marker.name!=f"{step:012d}-{digest}.json": raise ValueError("Local manifest SHA mismatch")
    rel=Path("runs")/name/marker.name
    if subprocess.check_output(["rclone","cat",f"{remote}/{rel}"])!=payload: raise ValueError("Drive manifest mismatch")
    manifest=read(marker)
    if (manifest["schema"],manifest["run"],manifest["step"])!=(1,name,step): raise ValueError("Manifest identity mismatch")
    if exact and set(manifest["files"])!=set(files): raise ValueError(f"Published inventory mismatch: {name}/{step}")
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


def milestone(physical,folder,config,report,step,final):
    root=physical/folder; cfg=read(config); lock=sha(root/"training_lock.json")
    runs={f"{profile}-{seed}":(profile,seed) for profile in cfg["profiles"] for seed in cfg["seeds"]}
    if set(read(report)["rows"])!=set(runs): raise ValueError(f"Incomplete comparison: {report}")
    files={"summary.json":report,"training_lock.json":root/"training_lock.json","config.json":config}
    endpoints={}; exposure={}
    for name,(profile,seed) in runs.items():
        path=root/name; card=read(path/f"step-{step}.json")
        parent=cfg["parents"][f"{cfg['profiles'][profile]['core']}-{seed}"][1]
        if (card["profile"],card["seed"],card["step"],card["lock_sha256"],card["parent_sha256"])!=(profile,seed,step,lock,parent):
            raise ValueError(f"Wrong endpoint identity: {folder}/{name}/{step}")
        baseline=read(physical/name/f"step-{step}.json")["exposure_sha256"]
        if card["exposure_sha256"]!=baseline or exposure.setdefault(seed,baseline)!=baseline:
            raise ValueError(f"Unequal training tensors: {folder}/{name}/{step}")
        files[name+".pt"]=path/f"step-{step}.pt"; files[name+".json"]=path/f"step-{step}.json"
        endpoints[name]=card["delta_sha256"]
        if sha(files[name+".pt"])!=endpoints[name]: raise ValueError(f"Delta checksum mismatch: {folder}/{name}/{step}")
        if final:
            for task in ("fsdd","fashion","banking"):
                head=path/f"eval-{step}"/f"{task}-32shot-head.npz"
                files[name+"-"+head.name]=head
    if final:
        frozen=read(root/"confirmation_lock.json")
        if frozen["endpoints"]!=endpoints or any(sha(Path(p))!=digest for p,digest in frozen["evaluation"].items()):
            raise ValueError(f"Confirmation weights or evaluation code changed: {folder}")
        if not folder and (frozen["training_lock_sha256"]!=lock or frozen["parents"]!=cfg["parents"]):
            raise ValueError("Primary training or parent lock changed")
        if folder and frozen.get("full_endpoint_lock_sha256",frozen.get("full_lock_sha256"))!=sha(physical/"confirmation_lock.json"):
            raise ValueError(f"Companion primary reference lock changed: {folder}")
    return files,{"models":len(runs),"fitted_32shot_heads":3*len(runs) if final else 0,"exposure_sha256_by_seed":exposure}


def replay_checks(physical,primary):
    spec=Path("configs/t4/joint_pusht.json"); protocol=read(spec)
    sources=[spec,Path("docs/research/joint_pusht_20261109.md"),Path("training/joint_pusht.py"),
             Path("training/joint_evaluate.py"),Path("models/joint_transfer.py")]
    expected={p.name:sha(p) for p in sources}
    if sha(Path(primary["data"])/"sources/pusht.zip")!=protocol["archive_sha256"]: raise ValueError("Push-T archive changed")
    reports={}; previous=set()
    for role in ("screen","scale"):
        output=physical/"pusht"/role; result=output/"results.json"
        if not result.exists(): continue
        p=output/"lock.json"; lock=read(p); data=read(result); step=protocol["budgets"][role]
        endpoints={f"{n}-{s}":read(physical/f"{n}-{s}"/f"step-{step}.json")["delta_sha256"]
                   for n in primary["profiles"] for s in primary["seeds"]}
        if lock["sha256"]!=protocol["archive_sha256"] or lock["endpoints"]!=endpoints or lock["sources"]!=expected:
            raise ValueError(f"Push-T source or checkpoint lock changed: {role}")
        splits={k:set(v) for k,v in lock["splits"].items()}; held=splits[role]
        if any(len(splits[k])!=v for k,v in protocol["episodes"].items()) or sum(map(len,splits.values()))!=len(set.union(*splits.values())):
            raise ValueError("Push-T episode roles overlap or changed size")
        if (data["lock_sha256"],data["role"],data["step"])!=(sha(p),role,step) or set(data["held_episode_order"])!=held or held&previous:
            raise ValueError("Push-T results differ from the locked episode roles")
        previous.update(held)
        for key,values in data["per_episode"].items():
            if len(values)!=len(held) or abs(sum(values)/len(values)-data["mean_episode_error"][key])>1e-4:
                raise ValueError(f"Push-T episode aggregation changed: {key}")
        reports[f"pusht/{role}"]={"lock_sha256":sha(p),"results_sha256":sha(result),"held_episodes":len(held),
                                  "fit_episodes":len(splits["fit"]),"held_windows":data["held_windows"],"models":len(endpoints)}
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("physical","spool","replica","out"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--replica-uuid",required=True);p.add_argument("--remote",default="slwm-drive:SLWM/v1")
    p.add_argument("--fill-replica",action="store_true")
    p.add_argument("--all-completed",action="store_true",help="Verify every completed matched stage and both available Push-T roles")
    a=p.parse_args()
    actual=subprocess.check_output(["findmnt","-nro","UUID","-T",a.replica],text=True).strip()
    if not a.replica.is_dir() or actual!=a.replica_uuid: raise ValueError("Expected mounted replica disk")
    config=Path("configs/t4/joint_transfer.json"); primary=read(config)
    if not Path(primary["output"]).samefile(a.physical): raise ValueError("Wrong checkpoint root")
    objects=remote_inventory(a.remote); reports={}
    def check(name,step,files,exact=False):
        reports[f"{name}/{step}"]=verify_group(name,step,files,a.spool,a.replica,a.remote,objects,a.fill_replica,exact)
    groups=[("","summary"),("depth6","depth"),("structured","structured")] if a.all_completed else [("","summary")]
    for folder,prefix in groups:
        for step in primary["milestones"] if a.all_completed else (100,1000):
            final=step==primary["milestones"][-1]; role="test" if final else "dev"
            group=a.physical/folder; report=group/f"{prefix}-{step}-{role}.json"
            if a.all_completed and not report.exists(): continue
            files,info=milestone(a.physical,folder,group/"config.json" if folder else config,report,step,final)
            name=f"g3-{folder or 'main'}-exports"; check(name,step,files,True)
            if a.all_completed: reports[f"{name}/{step}"].update(info)
    for group,directory in (("depth6","depth_parents"),("structured","structured_parents")):
        for _,(name,expected) in read(a.physical/group/"config.json")["parents"].items():
            parent=Path(primary["data"])/directory/name
            if sha(parent/"best.pt")!=expected: raise ValueError("Registered parent changed")
            # The immutable parent package holds its original config serialization.
            check(f"g3-parent-{group}-{name}",0,{"best.pt":parent/"best.pt","best.json":parent/"best.json"})
    if a.all_completed: reports.update(replay_checks(a.physical,primary))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    tmp=a.out.with_suffix(".tmp");tmp.write_text(json.dumps(reports,indent=2,sort_keys=True)+"\n");tmp.replace(a.out)
    print(f"verified G3 milestones and six parent archives ({len(reports)} records)",flush=True)


if __name__=="__main__": main()
