"""Test cross-source temporal retrieval without exposing held scene identity or fitting on held pairs."""

import argparse
import json
from functools import lru_cache
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g9_transfer import records
from models.g9_shared import Spec, encode
from scripts.train.g9_fit import numpy_weights, save_json

NAMES = ("audio", "vision", "geometry", "words")


def windows(source):
    rows = []
    for rec in source.values():
        for start in np.linspace(0, len(rec["a"]) - 32, 8, dtype=int):
            rows.append({k: r[start : start + 32] for k, r in rec.items()})
    return {k: np.stack([v[k] for v in rows]) for k in rows[0]}


@lru_cache(maxsize=12)
def encoded(spec):
    @jax.jit
    def kernel(params, r, mask):
        keep = jnp.broadcast_to(mask, (r["a"].shape[0], 4))
        z, _ = encode(params, spec, r, keep)
        return z[:, -8:].astype(jnp.float32).mean(1)

    return kernel


def fields(p, spec, bank):
    out = []
    for mask in np.eye(4, dtype=bool):
        parts = []
        for start in range(0, len(bank["a"]), 32):
            r = {k: jnp.asarray(v[start : start + 32]) for k, v in bank.items()}
            r["u"] = jnp.zeros(r["a"].shape[:2], jnp.int32)
            parts.append(np.asarray(encoded(spec)(p, r, mask)))
        out.append(np.concatenate(parts).reshape(-1, 8, spec.field).astype(np.float64))
    return out


def activity(bank):
    # Geometry and event words derive from the same annotation process. Retain a raw activity comparator.
    v = bank["v"].astype(np.float32) / 255
    motion = np.abs(np.diff(v, axis=1)).mean(-1)
    arrays = [
        bank["a"].astype(np.float32).mean(-1),
        motion,
        bank["n"][:, :, 3],
        (bank["w"][:, :, 0] == 24).astype(np.float32),
    ]
    return [v[:, -8:].reshape(-1, 8, 8).astype(np.float64) for v in arrays]


def center(a):
    v = a - a.mean(1, keepdims=True)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def matching(a, b):
    cos = np.einsum("rkd,rjd->rkj", a, b)
    winners = np.isclose(cos, cos.max(-1, keepdims=True), rtol=0, atol=1e-7)
    return (np.diagonal(winners, axis1=1, axis2=2) / winners.sum(-1)).mean(-1).tolist()


def main(data, runs, out):
    out.mkdir(parents=True, exist_ok=True)
    train = records(data / "starss-train.npz")
    held = records(data / "starss-invariance.npz")
    if set(train) & set(held) or {k.split("_")[1] for k in train} & {
        k.split("_")[1] for k in held
    }:
        raise ValueError("Train/held overlap")
    tr, te = windows(train), windows(held)
    raw = list(map(center, activity(te)))
    for field in (128, 512):
        for seed in (911, 912):
            for core in ("hybrid", "gpt2"):
                tag = f"confirm-{core}-z{field}-gelu-s{seed}-fresh"
                stage = runs / tag / "step-04096"
                dest = out / (tag + ".json")
                if dest.exists():
                    continue
                meta = json.loads((stage / "receipt.json").read_text())
                spec = Spec(**meta["spec"])
                p = jax.tree.map(jnp.asarray, numpy_weights(stage / "weights.npz"))
                training = list(map(center, fields(p, spec, tr)))
                test = list(map(center, fields(p, spec, te)))
                scores = {}
                for i in range(4):
                    for j in range(i + 1, 4):
                        x, y = (
                            training[i].reshape(-1, field),
                            training[j].reshape(-1, field),
                        )
                        left, _, right = np.linalg.svd(x.T @ y, full_matrices=False)
                        rotation = left @ right
                        for a, b, r in ((i, j, rotation), (j, i, rotation.T)):
                            scores[NAMES[a] + "_to_" + NAMES[b]] = {
                                "direct": matching(test[a], test[b]),
                                "aligned_on_train": matching(test[a] @ r, test[b]),
                                "raw_activity": matching(raw[a], raw[b]),
                            }
                save_json(
                    dest,
                    dict(
                        tag=tag,
                        parent_sha256=meta["inference_sha256"],
                        train_recordings=len(train),
                        held_recordings=list(held),
                        candidates_per_recording=8,
                        chance=0.125,
                        scores=scores,
                    ),
                )
                print(
                    tag,
                    "direct audio to video",
                    round(float(np.mean(scores["audio_to_vision"]["direct"])), 4),
                    flush=True,
                )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("runs", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    main(a.data, a.runs, a.out)
