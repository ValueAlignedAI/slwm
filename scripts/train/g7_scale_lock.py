"""Bind all 16 continuation endpoints and their eight zero-update controls to the fresh G7 tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import lock, sha


def cases():
    for size in ("anchor","larger"):
        for seed in (511,512):
            for core in ("hybrid","gpt2"):
                yield f"g6-v5e-{size}-{core}-{seed}",8192
                for arm in ("replay","fresh"):
                    yield f"g7-v5e-{size}-{core}-{seed}-{arm}",16384


def datasets(root, old):
    for role in ("dev","test"):
        for kind,folder in (("paired",old/"paired"/role),("impact",root/"fresh/events"/role),("pusht",root/"fresh/sim"/role),("pusht/train",old/"pusht/train")):
            dest=root/"evaluation"/kind/role if kind!="pusht/train" else root/"evaluation/pusht/train"
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                dest.symlink_to(folder, target_is_directory=True)
            if dest.resolve()!=folder.resolve():
                raise ValueError(f"Evaluation data changed at {dest}")


def signatures(path):
    return {kind:sha(path/kind/"test/card.json") for kind in ("paired","impact","pusht")}


def freeze(new,old):
    a,b=roots("g7"); datasets(new,old)
    locked={}
    for tag,step in cases():
        relative=Path(tag)/f"step-{step:06d}"
        disks=(a,b) if tag.startswith("g7-") else roots("g6")
        receipts=[json.loads((p/relative/"verified.json").read_bytes()) for p in disks]
        left,right=receipts
        if left["sha256"]!=right["sha256"] or left["step"]!=step or left["run"]!=tag:
            raise ValueError(f"Unmatched copies of {tag}")
        locked[tag]=left["sha256"]["inference.npz"]
    record={"study":"G7","models":locked,"data":signatures(new/"evaluation"),
            "paired_new":{f:sha(new/"fresh"/f/"test/card.json") for f in ("crema","libri","text")},
            "test_sources":"new G7 speakers, documents, Tokyo rooms and simulator resets"}
    lock(a/"scale-test-lock.json",record)
    lock(b/"scale-test-lock.json",record)
    return record


def verify(record,run,sha256,root):
    expected=dict(cases()); tag=run.parent.name
    if set(record["models"])!=set(expected) or record["models"].get(tag)!=sha256 or run.name!=f"step-{expected[tag]:06d}":
        raise ValueError("Outside the jointly frozen G7 confirmation")
    if signatures(root)!=record["data"]:
        raise ValueError("Fresh evaluation role changed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--new",type=Path,required=True);p.add_argument("--old",type=Path,required=True)
    a=p.parse_args();print(len(freeze(a.new,a.old)["models"]),"frozen models")
