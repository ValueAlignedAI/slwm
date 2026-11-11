"""Release the second TPU after its verified first stage while public training recordings are prepared."""

import argparse
import json
from pathlib import Path
import shlex
import time
from g7_inputs import FED, stream
from g9_storage import write
from g10_dispatch import dispatch


def main(ctl, rev):
    a = "slwm-g10-context-a"
    b = "slwm-g10-context-b"
    seed = 1001
    receipt_dir = ctl / "receipts"
    expected = {
        f"context-{c}-{m}-s{seed}-{step}.json"
        for c in ("hybrid", "gpt2", "gated")
        for m in ("short", "variable", "long")
        for step in (2560, 5120)
    }
    rooms = {f"align-{c}-rooms-s{seed}-1536.json" for c in ("hybrid", "gpt2")}
    while time.time() < json.loads((ctl / "bounds.json").read_text())["deadline"]:
        done = {p.name for p in receipt_dir.glob("*.json")}
        scores = ctl / (b + "-eval.json")
        if (
            expected | rooms <= done
            and scores.exists()
            and len(json.loads(scores.read_text())["files"]) >= 18
        ):
            break
        time.sleep(10)
    else:
        raise RuntimeError("First native lane did not reach the consolidation gate")
    if (ctl / (b + "-bank.json")).exists():
        return
    route = {"bank_targets": [a], "seeds": {a: [1001, 1002], b: [1001]}}
    write(ctl / "routing.json", route)
    stages = [(f"context-{c}-variable-s{seed}", 5120) for c in ("hybrid", "gpt2")] + [
        (f"align-{c}-rooms-s{seed}", 1536) for c in ("hybrid", "gpt2")
    ]
    files = []
    checks = []
    for tag, step in stages:
        rec = json.loads((receipt_dir / f"{tag}-{step}.json").read_text())
        base = f"{tag}/step-{step:05d}/"
        files += [base + "weights.npz", base + "receipt.json"]
        checks.append(
            "echo "
            + shlex.quote(rec["inference_sha256"] + "  " + base + "weights.npz")
            + " | sha256sum -c -"
        )
    stream(
        a,
        [
            *FED,
            "tar -cf - -C /mnt/slwm-checkpoints-a/SLWM/g10/runs " + shlex.join(files),
        ],
        "set -eu; mkdir -p /content/g10-out; cd /content/g10-out; tar -xf -; "
        + "; ".join(checks),
    )
    dispatch(a, seed, rev, ["mixed", "align-eval"])
    write(
        ctl / "consolidated.json",
        dict(
            source=b, target=a, verified_restored_weights=4, completed_context_scores=18
        ),
    )
    (ctl / (b + "-release")).touch()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("control", type=Path)
    ap.add_argument("revision")
    args = ap.parse_args()
    main(args.control, args.revision)
