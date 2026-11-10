"""Copy finished research heads, inference examples, results and optimizer states to the second HDD."""
import hashlib
import json
from pathlib import Path
import shutil
import time
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha,lock


def copy(a,b,key,inventory):
    src=a/key;target=b/key
    target.parent.mkdir(exist_ok=True,parents=True)
    known=inventory.get(str(key))
    stat=src.stat()
    if known and known["bytes"]==stat.st_size and known["mtime_ns"]==stat.st_mtime_ns and target.is_file():return
    h=sha(src)
    if not target.is_file() or sha(target)!=h:
        tmp=target.with_name("."+target.name+".mirroring")
        shutil.copyfile(src,tmp)
        if sha(tmp)!=h:raise IOError(f"Mirror hash differs for {key}")
        tmp.replace(target)
    inventory[str(key)]={"sha256":h,"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns}


def mirror():
    a,b=roots("g7");state=a/"readouts-archive.json"
    inventory=json.loads(state.read_bytes())["files"] if state.exists() else {}
    complete=set()
    for directory,marker in (("policy","*/dev.json"),("policy-prior","*/dev.json"),("rl","*/test.json"),
        ("alignment","*/*/dev.json"),("alignment-audit","*/*/waves.json"),("camera","*/test.json"),
        ("scale-eval","*/*/*-signals.json"),("unit-renderer","complete.json"),("reacher-prior","card.json")):
        for mark in (a/directory).glob(marker):
            complete.add(str(mark.relative_to(a)))
            for f in mark.parent.rglob("*"):
                if f.is_file() and f.suffix in (".npz",".pt",".json",".wav"):
                    copy(a,b,f.relative_to(a),inventory)
    for area in ("supervised-confirmation","timing-audit","retrieval","planning"):
        for f in (a/area).rglob("*"):
            if f.is_file() and f.suffix in (".json",".npz"):copy(a,b,f.relative_to(a),inventory)
    for key in ("alignment/test-lock.json","policy-prior/sft-lock.json","camera/lock.json","scale-test-lock.json",
                "alignment-audit/crema-oracle.json","alignment-audit/libri-oracle.json","reacher-oracle.json"):
        if (a/key).exists():copy(a,b,Path(key),inventory)
    record={"files":inventory,"finished_markers":sorted(complete),"verified_files":len(inventory),
            "bytes":sum(v["bytes"] for v in inventory.values()),"separate_physical_disks":True}
    for dest in (a,b):
        tmp=dest/"readouts-archive.partial";tmp.write_text(json.dumps(record,sort_keys=True)+"\n");tmp.replace(dest/"readouts-archive.json")
    return len(complete)


if __name__=="__main__":
    deadline=time.time()+12*3600
    while time.time()<deadline:
        n=mirror()
        if n>=128:break
        time.sleep(50)
