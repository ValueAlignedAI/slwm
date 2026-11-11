"""Apply a verified frozen risk readout using only the 32 observations before a proposed action start."""

import argparse
import hashlib
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from models.g10_alignment import encode_record
from scripts.infer.g10_world import read_input, SOURCES
from scripts.infer.g9_world import checkpoint


def score(weights, heads, path, onsets, features):
    spec, p, identity = checkpoint(weights)
    manifest = json.loads((heads.parent / "receipts.json").read_text())
    entry = manifest[heads.name]
    if (
        hashlib.sha256(heads.read_bytes()).hexdigest() != entry["sha256"]
        or entry["parent_sha256"] != identity
    ):
        raise ValueError("Head and immutable training parent do not match")
    if "utterance_edge" not in p:
        raise ValueError("A G10 grounded model is required")
    rows = read_input(path, SOURCES)
    last = np.floor(10 * np.asarray(onsets, float)).astype(int) - 3
    if np.any(last < 31) or np.any(last >= len(rows["u"])):
        raise ValueError(
            "The requested action needs a complete earlier 3.2-second prefix"
        )
    ix = last[:, None] - 31 + np.arange(32)[None, :]
    r = {k: jnp.asarray(v[ix]) for k, v in rows.items()}
    diff = np.abs(
        rows["v"][ix].astype(np.int16)
        - rows["v"][np.maximum(0, ix - 1)].astype(np.int16)
    )
    r["m"] = jnp.asarray(
        np.clip(diff.reshape(*ix.shape, 256, 3).mean(-1) / 64, 0, 2), jnp.float32
    )
    mask = {"all": (1, 1, 1, 1), "no_words": (1, 1, 1, 0), "words_only": (0, 0, 0, 1)}[
        features
    ]
    z = np.asarray(
        encode_record(p, spec, r, jnp.broadcast_to(np.array(mask, bool), (len(ix), 4)))[
            :, -1
        ],
        np.float32,
    )
    with np.load(heads, allow_pickle=False) as h:
        logit = (z - h[features + "_mean"]) / h[features + "_scale"] @ h[
            features + "_weight"
        ] + h[features + "_intercept"]
    prob = 1 / (1 + np.exp(-np.clip(logit, -60, 60)))
    return dict(
        parent_sha256=identity,
        features=features,
        proposed_action_seconds=list(map(float, onsets)),
        last_observed_tick=last.tolist(),
        error_probability=prob.tolist(),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("heads", type=Path)
    ap.add_argument("input", type=Path)
    ap.add_argument("--action-starts", required=True, nargs="+", type=float)
    ap.add_argument(
        "--features", choices=("all", "no_words", "words_only"), default="all"
    )
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    a.output.write_text(
        json.dumps(
            score(a.weights, a.heads, a.input, a.action_starts, a.features), indent=2
        )
        + "\n"
    )
