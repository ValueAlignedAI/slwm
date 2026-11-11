"""Replay the released original waveform from CPU and GPU and compare the checked decoding."""

import json
from pathlib import Path
from scripts.infer.g14_live import active_goal, read
from scripts.train.g9_storage import write, sha

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def main():
    pkg = ROOT / "release/inference"
    man = json.loads((pkg / "manifest.json").read_bytes())
    scene = read(pkg / "cue-features.npz")
    out = {"features": active_goal(scene, pkg, man)}
    for dev in ("cpu", "cuda"):
        row = active_goal(
            scene,
            pkg,
            man,
            "wav",
            pkg / "spoken-goal.wav",
            ROOT.parent / "g11/data/whisper-small",
            dev,
        )
        out[dev] = row
    keys = ("accepted", "color", "station", "unconstrained")
    for d in out.values():
        if any(d[k] != out["features"][k] for k in keys):
            raise ValueError("Original waveform replay disagrees")
    write(
        ROOT / "work/speech-replay.json",
        dict(
            results=out,
            original_waveform=sha(pkg / "spoken-goal.wav"),
            encoder=man["encoder"],
            clip=man["example"],
        ),
    )
    print(
        "Original recording agrees on CPU, CUDA and prepared features", out, flush=True
    )


if __name__ == "__main__":
    main()
