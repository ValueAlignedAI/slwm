"""Matched post-training from hash-checked mixed-context parents, with real instruction edges."""

import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_shared import Spec
from models.g10_alignment import add_edges, objective
from scripts.train.g9_fit import numpy_weights, save_json
from scripts.train.g10_fit import LENGTHS, adam, restore, snapshot


def real_bank(starss, holo, rooms_only=False):
    chunks = {}
    offset = 0
    names = {}
    with np.load(starss) as a, np.load(starss if rooms_only else holo) as b:
        for label, db in (("star", a), ("holo", b)):
            lengths = db["lengths"].astype(np.int32)
            if lengths.min() < 8 * 33:
                raise ValueError("Cannot draw eight nonoverlapping windows")
            chunks[label + "_len"] = lengths
            chunks[label + "_off"] = (
                offset + np.cumsum(np.r_[0, lengths[:-1]])
            ).astype(np.int32)
            names[label] = list(map(str, db["names"]))
            offset += int(lengths.sum())
        for k in ("a", "v", "n", "w"):
            chunks[k] = np.concatenate([a[k], b[k]])
        chunks["x"] = np.concatenate(
            [
                np.zeros((len(a["a"]), 128), np.int8),
                np.zeros((len(b["a"]), 128), np.int8) if rooms_only else b["x"],
            ]
        )
        chunks["second_is_holo"] = np.asarray(not rooms_only)
    return jax.tree.map(jax.device_put, chunks), names


@lru_cache(maxsize=35)
def kernel(s, mode, length, total):
    def step(p, m, v, bank, i, seed):
        (loss, metrics), g = jax.value_and_grad(objective, has_aux=True)(
            p, s, bank, i, seed, mode, length
        )
        p, m, v, norm = adam(p, m, v, g, i, total, 7e-5)
        return p, m, v, jnp.r_[loss, metrics, norm]

    return jax.jit(step, donate_argnums=(0, 1, 2))


def fit(root, out, bank, names, core, seed, mode, smoke=False):
    stage = root / f"context-{core}-variable-s{seed}" / "step-05120"
    parent = json.loads((stage / "receipt.json").read_text())
    with (stage / "weights.npz").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != parent["inference_sha256"]:
            raise ValueError("Parent checksum mismatch")
    s = Spec(**parent["spec"])
    total = 4 if smoke else 1536
    tag = f"{'align-smoke' if smoke else 'align'}-{core}-{mode}-s{seed}"
    job = out / tag
    if (job / "done.json").exists():
        return
    recipe = dict(
        tag=tag,
        spec=asdict(s),
        seed=seed,
        goal_steps=total,
        phase="alignment",
        treatment=mode,
        event_source="STARSS23 only"
        if mode == "rooms"
        else "HoloAssist completed speech and IMU",
        training_recordings=names,
        parent_sha256=parent["inference_sha256"],
        backend=jax.default_backend(),
    )
    initialize = lambda: add_edges(
        jax.tree.map(jnp.asarray, numpy_weights(stage / "weights.npz")), s, seed
    )
    step, p, m, v = restore(job, initialize, recipe)
    anchor = step
    recipe = {**recipe, "trainable_parameters": sum(x.size for x in p.values())}
    start = clock = time.monotonic()
    while step < total:
        l = LENGTHS[step % 5]
        p, m, v, metric = kernel(s, mode, l, total)(
            p, m, v, bank, jnp.int32(step), jnp.int32(seed)
        )
        step += 1
        if step in (1, total) or step % 32 == 0:
            vals = np.asarray(metric)
            if not np.isfinite(vals).all():
                raise FloatingPointError((tag, step, vals))
            now = time.monotonic()
            row = dict(
                tag=tag,
                step=step,
                goal=total,
                loss=float(vals[0]),
                rehearsal=float(vals[1]),
                signal=float(vals[2]),
                contrast=float(vals[3]),
                match=float(vals[4]),
                gradient=float(vals[5]),
                seconds=now - start,
                updates_s=32 / (now - clock) if step >= 64 else None,
            )
            with (job / "train.jsonl").open("a") as log:
                log.write(json.dumps(row) + "\n")
            save_json(out / "alignment-progress.json", row)
            clock = now
            if step == 1 or step % 256 == 0 or step == total:
                print(
                    tag, step, round(row["loss"], 3), round(now - start, 1), flush=True
                )
        if step in (total // 2, total):
            snapshot(job, step, p, m, v, recipe, start, anchor)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("starss", type=Path)
    ap.add_argument("holo", type=Path)
    ap.add_argument("parents", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--cores", nargs="+", default=["hybrid", "gpt2"])
    ap.add_argument("--modes", nargs="+", default=["reconstruct", "paired", "shifted"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cache")
    args = ap.parse_args()
    if args.cache:
        jax.config.update("jax_compilation_cache_dir", args.cache)
    if "rooms" in args.modes and args.modes != ["rooms"]:
        raise ValueError("The room-only data control must run separately")
    bank, names = real_bank(args.starss, args.holo, args.modes == ["rooms"])
    for seed in args.seeds:
        for mode in args.modes:
            for core in args.cores:
                fit(args.parents, args.out, bank, names, core, seed, mode, args.smoke)
