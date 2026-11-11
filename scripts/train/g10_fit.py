"""Train the context factorial with equal information exposure and exact resume state."""

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from models.g9_shared import Spec, parameters
from models.g10_context import TICKS, continuous, objective, segment
from scripts.train.g9_fit import numpy_weights, weights, save_json

LENGTHS = (32, 64, 128, 256, 512)


def bank(path):
    with np.load(path, allow_pickle=False) as f:
        lengths = f["lengths"].astype(int)
        offsets = np.cumsum(np.r_[0, lengths[:-1]])
        starts = np.concatenate(
            [
                off + np.arange(n - TICKS, dtype=np.int32)
                for off, n in zip(offsets, lengths)
                if n > TICKS
            ]
        )
        names = [str(n) for n, l in zip(f["names"], lengths) if l > TICKS]
        return (
            {k: jax.device_put(f[k]) for k in ("a", "v", "n", "w")},
            jax.device_put(starts.astype(np.int32)),
            names,
        )


def adam(p, m, v, g, step, total, peak):
    norm = jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(g)))
    g = jax.tree.map(lambda x: x * jnp.minimum(1.0, 1.0 / (norm + 1e-8)), g)
    m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
    v = jax.tree.map(lambda a, b: 0.98 * a + 0.02 * b * b, v, g)
    t = step + 1
    warm = min(100, max(8, total // 12))
    rate = (
        peak
        * jnp.minimum(1.0, t / warm)
        * (
            0.1
            + 0.45
            * (1 + jnp.cos(jnp.pi * jnp.maximum(t - warm, 0) / max(1, total - warm)))
        )
    )
    p = jax.tree.map(
        lambda val, m0, v0: val
        - rate
        * (m0 / (1 - 0.9**t) / (jnp.sqrt(v0 / (1 - 0.98**t)) + 1e-8) + 0.01 * val),
        p,
        m,
        v,
    )
    return p, m, v, norm


@lru_cache(maxsize=30)
def update_kernel(spec, length, total, batch):
    def step(p, m, v, real, starts, i, seed):
        r = segment(continuous(real, starts, i, seed, batch), length)
        (loss, values), g = jax.value_and_grad(objective, has_aux=True)(p, spec, r)
        p, m, v, norm = adam(p, m, v, g, i, total, 3e-4)
        return p, m, v, jnp.r_[loss, values, norm]

    return jax.jit(step, donate_argnums=(0, 1, 2))


def snapshot(job, step, p, m, v, recipe, start_time, anchor):
    stage = job / f"step-{step:05d}"
    work = job / f".writing-{step}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    ws = weights(work / "weights.npz", p)
    ms = weights(
        work / "optimizer.npz",
        {**{"m_" + k: x for k, x in m.items()}, **{"v_" + k: x for k, x in v.items()}},
    )
    info = {
        **recipe,
        "step": step,
        "inference_sha256": ws,
        "optimizer_sha256": ms,
        "wall_this_process_s": time.monotonic() - start_time,
        "started_from": anchor,
    }
    save_json(work / "receipt.json", info)
    work.replace(stage)
    if step == recipe["goal_steps"]:
        save_json(job / "done.json", info)


def restore(job, initialize, recipe):
    job.mkdir(exist_ok=True, parents=True)
    if (job / "recipe.json").exists() and json.loads(
        (job / "recipe.json").read_text()
    ) != recipe:
        raise ValueError("Immutable training recipe changed")
    save_json(job / "recipe.json", recipe)
    stages = sorted(x for x in job.glob("step-*") if (x / "receipt.json").exists())
    if not stages:
        p = initialize()
        return 0, p, jax.tree.map(jnp.zeros_like, p), jax.tree.map(jnp.zeros_like, p)
    stage = stages[-1]
    rec = json.loads((stage / "receipt.json").read_text())
    for file, key in (
        ("weights.npz", "inference_sha256"),
        ("optimizer.npz", "optimizer_sha256"),
    ):
        with (stage / file).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != rec[key]:
                raise ValueError("Resume digest mismatch")
    if rec["step"] == recipe["goal_steps"] and not (job / "done.json").exists():
        save_json(job / "done.json", rec)
    p = jax.tree.map(jnp.asarray, numpy_weights(stage / "weights.npz"))
    mv = numpy_weights(stage / "optimizer.npz")
    return (
        rec["step"],
        p,
        {k: jnp.asarray(mv["m_" + k]) for k in p},
        {k: jnp.asarray(mv["v_" + k]) for k in p},
    )


def fit(real, starts, names, out, core, regime, seed, smoke=False):
    total = 4 if smoke else 5120
    spec = Spec(core, 128 if smoke else 512, 2 if smoke else 4, 128, "gelu")
    batch = 4 if smoke else 16
    tag = f"{'smoke' if smoke else 'context'}-{core}-{regime}-s{seed}"
    job = out / tag
    if (job / "done.json").exists():
        return
    recipe = dict(
        tag=tag,
        spec=asdict(spec),
        seed=seed,
        goal_steps=total,
        positions_per_update=batch * TICKS,
        train_recordings=names,
        regime=regime,
        backend=jax.default_backend(),
        phase="context",
    )
    step, p, m, v = restore(job, lambda: parameters(spec, seed), recipe)
    anchor = step
    recipe = {**recipe, "trainable_parameters": sum(x.size for x in p.values())}
    start = time.monotonic()
    clock = start
    while step < total:
        length = (
            LENGTHS[step % 5]
            if regime == "variable"
            else 32
            if regime == "short"
            else 512
        )
        p, m, v, values = update_kernel(spec, length, total, batch)(
            p, m, v, real, starts, jnp.int32(step), jnp.int32(seed)
        )
        step += 1
        if step in (1, total) or step % 64 == 0:
            metric = np.asarray(values)
            if not np.isfinite(metric).all():
                raise FloatingPointError((tag, step, metric))
            now = time.monotonic()
            row = dict(
                tag=tag,
                step=step,
                goal=total,
                loss=float(metric[0]),
                policy=float(metric[1]),
                signals=float(metric[2]),
                future=float(metric[3]),
                gradient_norm=float(metric[4]),
                seconds=now - start,
                updates_s=64 / (now - clock) if step >= 128 else None,
            )
            with (job / "train.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            save_json(out / "context-progress.json", row)
            clock = now
            if step == 1 or step % 512 == 0 or step == total:
                print(
                    tag, step, round(row["loss"], 4), round(now - start, 1), flush=True
                )
        if step in (total // 2, total):
            snapshot(job, step, p, m, v, recipe, start, anchor)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cores", nargs="+", default=["gpt2", "hybrid", "gated"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1001, 1002, 1003])
    ap.add_argument("--regimes", nargs="+", default=["variable", "short", "long"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cache")
    args = ap.parse_args()
    if args.cache:
        jax.config.update("jax_compilation_cache_dir", args.cache)
    real, starts, names = bank(args.data)
    for seed in args.seeds:
        for regime in args.regimes:
            for c in args.cores:
                fit(real, starts, names, args.out, c, regime, seed, args.smoke)
