"""Fix a train-only replacement for the 126 learned coordinates in the protected field."""

from pathlib import Path
import numpy as np
from data.g14_policy import loaded
from data.g14_signals import pack, raw_predict, attach
from models.g14_world import predict

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def means(stage):
    path = ROOT / "data/training.npz"
    with np.load(path, allow_pickle=False) as z:
        groups = z["world"]
        ids = [
            np.random.default_rng(int(i) + 1459).choice(np.flatnonzero(groups == i))
            for i in np.unique(groups)
        ]
        train = {
            k: z[k][ids]
            for k in (
                "q",
                "p",
                "seen",
                "motor",
                "u",
                "px",
                "color",
                "target",
                "choices",
            )
        }
    out = {}
    for core in ("gpt2", "transport"):
        ref, _, p, _ = loaded(stage, core, "protected", 1411)
        vec = []
        for start in range(0, len(ids), 32):
            r = pack({k: x[start : start + 32] for k, x in train.items()})
            attach(r, raw_predict(r, ref))
            _, _, field = predict(p, r, core, "protected", np)
            vec.append(field[:, 2:])
        out[core + "-1411"] = np.concatenate(vec).mean(0).astype("f4")
    file = ROOT / "work" / stage / "field-means.npz"
    tmp = file.with_suffix(".part.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(file)
    return file
