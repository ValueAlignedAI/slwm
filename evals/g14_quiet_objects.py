"""Describe the false displacement of physically quiet distractors in the frozen forecast set."""

import json
from pathlib import Path
import numpy as np
from evals.g13_stats import Groups
from scripts.train.g9_storage import write

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def main():
    path = ROOT / "release/metrics/held-physics-predictions.npz"
    with np.load(path, allow_pickle=False) as z:
        truth = np.linalg.norm(z["actual_mm"][:, :, 1:], axis=-1)
        pred = np.linalg.norm(z["predicted_mm"][:, :, :, 1:], axis=-1)
        color = z["color"]
        world = z["world"]
        keys = z["models"].tolist()
    others = np.arange(4)[None, None, :] != color[:, None, None]
    quiet = (truth <= 1) & others
    groups = np.unique(world)
    boot = Groups(groups, groups % 2)
    out = {}
    for k in ("persistence-nominal", "raw", "gpt2", "transport"):
        ix = (
            [keys.index(k)]
            if k in ("raw", "persistence-nominal")
            else [keys.index(f"{k}-{s}") for s in (1411, 1412, 1413)]
        )
        distance = pred[ix].mean(0)
        false = (pred[ix] >= 5).mean(0)

        def group_mean(a):
            return np.array([a[world == g][quiet[world == g]].mean() for g in groups])

        out[k] = dict(
            predicted_motion_mm=boot.score(group_mean(distance)),
            predicted_at_least_5mm=boot.score(group_mean(false)),
        )
    write(
        ROOT / "release/metrics/quiet-objects.json",
        dict(
            note="Post-hoc diagnosis following the registered aggregate/moving score difference. A quiet distractor moved at most 1 mm in 300 ms. The large-motion threshold is 5 mm.",
            quiet_distractor_observations=int(quiet.sum()),
            distractor_observations=int(np.broadcast_to(others, truth.shape).sum()),
            worlds=len(groups),
            scores=out,
        ),
    )
    print(
        "Quiet distractor false motion",
        {k: round(v["predicted_at_least_5mm"]["value"], 3) for k, v in out.items()},
        flush=True,
    )


if __name__ == "__main__":
    main()
