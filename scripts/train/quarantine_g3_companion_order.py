"""Retain rejected G3 runs and the original driver code before replaying from verified parents."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def quarantine(physical, proof_file):
    cfg=read(Path("configs/t4/joint_transfer.json"))
    proof=read(proof_file)
    if not physical.samefile(Path(cfg["output"])) or sha(physical/"training_lock.json")!=proof["primary_lock_sha256"]:
        raise ValueError("Unexpected primary output or provenance")
    archive=proof_file.parent
    if archive.resolve().is_relative_to(physical.resolve()) or physical.stat().st_dev!=archive.stat().st_dev:
        raise ValueError("Choose a separate directory on the physical checkpoint filesystem")
    sources={"depth6":"training/joint_depth.py","structured":"training/joint_structured.py"}
    with ExitStack() as handles:
        for group,name in (("depth6","depth.lock"),("structured","structured.lock")):
            lock=handles.enter_context((Path(cfg["data"])/name).open("a"))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            origin=archive/group if (archive/group).exists() else physical/group
            if sha(origin/"training_lock.json")!=proof["variant_locks"][group]:
                raise ValueError(f"Unexpected training lock: {group}")
            if (origin/"confirmation_lock.json").exists() or list(origin.glob("*/step-1000.json")) or list(origin.glob("*/step-8000.json")):
                raise ValueError(f"Later artifacts exist for {group}")
            for run,stamp in proof["rejected"][group].items():
                path=origin/run
                if sha(path/"step-100.pt")!=stamp["delta_sha256"] or sha(path/"resume.pt")!=stamp["resume_sha256"]:
                    raise ValueError(f"Artifact changed: {group}/{run}")
            code=Path(sources[group]); saved=archive/(code.stem+"-original.py")
            bound=read(origin/"training_lock.json")
            expected=bound.get("sources",bound)[str(code)]
            if not saved.exists():
                if sha(code)!=expected: raise ValueError(f"Original source changed: {code}")
                shutil.copy2(code,saved)
            if sha(saved)!=expected: raise ValueError(f"Source archive mismatch: {code}")
        for group in sources:
            if not (archive/group).exists():
                (physical/group).rename(archive/group)
        # This separate kernel arithmetic test does not consume a training batch.
        structured=physical/"structured"
        structured.mkdir(exist_ok=True)
        for name in ("kernel.json","kernel_unavailable.json"):
            old=archive/"structured"/name
            if old.exists() and not (structured/name).exists(): shutil.copy2(old,structured/name)
    print("preserved eight rejected checkpoints, two source locks and both original drivers",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physical",type=Path,required=True); p.add_argument("--proof",type=Path,required=True)
    args=p.parse_args(); quarantine(args.physical,args.proof)
