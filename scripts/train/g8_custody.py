"""Accept a numerical bridge only when its reference is the original immutable G7 generation."""
import json
import sys
from pathlib import Path
from training.g8_models import CONFIG,MANIFEST,read
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import lock


def accept(b):
    core=b["core"];seed=b["seed"];arm=b["arm"]
    if read(CONFIG)["bridge"][str(seed)][core]!=arm:raise ValueError("Wrong counterbalanced bridge")
    a,mirror=roots("g7");tag=f"g7-v5e-anchor-{core}-{seed}-{arm}/step-010240"
    pin=read(MANIFEST)["rungs"][tag]
    for dest in (a,mirror):
        receipt=read(dest/tag/"verified.json")["sha256"]
        if (receipt["inference.npz"],receipt["resume.npz"])!=(pin["inference_sha256"],pin["optimizer_sha256"]):raise ValueError("G7 custody changed")
    if (b["expected_weights"],b["expected_optimizer"])!=(pin["inference_sha256"],pin["optimizer_sha256"]):raise ValueError("Wrong reference pin")
    matched=b["observed"]["inference.npz"]==b["expected_weights"] and b["observed"]["resume.npz"]==b["expected_optimizer"]
    if matched!=b["exact"]:raise ValueError("Malformed bridge receipt")
    for dest in roots("g8"):
        out=dest/"bridges";out.mkdir(exist_ok=True);lock(out/f"{core}-{seed}.json",b)
    print(core,seed,"exact" if matched else "same-wave controls required",flush=True)

if __name__=="__main__":accept(json.load(sys.stdin))
