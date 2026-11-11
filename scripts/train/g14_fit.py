"""Fit matched contact fields from physically forked interventions; select only on unseen development scenes."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from models.g14_world import initialize, objective, predict, CORES, ARMS
from data.g14_signals import (
    pack,
    raw_setup,
    raw_predict,
    attach,
    paired_regret,
    mean_group,
)
from scripts.train.g13_optim import update, save, load, complete, save_npz
from scripts.train.g9_storage import write, sha
from scripts.train.g14_cue_fit import ROOT, roots, read

CUTS = (96, 384, 1024)
SEEDS = (1411, 1412, 1413)
FEATS = ("slots", "image", "location", "phi", "command", "focus", "prior", "raw_action")


@lru_cache(maxsize=24)
def kernel(core, arm):
    @jax.jit
    def run(p, m, v, bank, indices, start):
        def one(state, d):
            pp, mm, vv = state
            ix, t = d
            b = {k: x[ix] for k, x in bank.items()}
            (loss, terms), g = jax.value_and_grad(objective, has_aux=True)(
                pp, b, core, arm
            )
            return update(pp, mm, vv, g, t, 1024, 0.00065), jnp.concatenate(
                (loss[None], terms)
            )

        return jax.lax.scan(
            one, (p, m, v), (indices, start + 1 + jnp.arange(len(indices)))
        )

    return run


@lru_cache(maxsize=24)
def eval_kernel(core, arm):
    return jax.jit(lambda p, x: predict(p, x, core, arm, jnp))


def estimates(p, r, core, arm):
    pred = []
    risk = []
    n = len(r["slots"])
    for i in range(0, n, 128):
        ix = np.minimum(np.arange(i, i + 128), n - 1)
        v, c, _ = eval_kernel(core, arm)(p, {k: jnp.asarray(r[k][ix]) for k in FEATS})
        pred.append(np.asarray(v)[: min(128, n - i)])
        risk.append(np.asarray(c)[: min(128, n - i)])
    return np.concatenate(pred), np.concatenate(risk)


def stats(pp, r):
    y = r["y"]
    groups = r["world"]
    rank = paired_regret(pp[:, :, -1], y)
    ae = np.abs(y - pp)
    return dict(
        regret_mm=mean_group(rank, groups),
        effect_mm=mean_group(
            np.linalg.norm(
                (y[:, :, :10] - pp[:, :, :10]).reshape(len(y), 8, 5, 2), axis=-1
            ).mean((1, 2))
            * 50,
            groups,
        ),
        utility_mae_mm=mean_group(ae[:, :, -1].mean(1) * 40, groups),
        top=mean_group(
            (y[:, :, -1].argmax(1) == pp[:, :, -1].argmax(1)).astype(float), groups
        ),
    )


def fit(stage):
    work = ROOT / "work" / stage
    work.mkdir(exist_ok=True, parents=True)
    data = ROOT / "data"
    raw = read(data / "training.npz")
    dev = pack(read(data / "development.npz"))
    tr = pack(raw)
    del raw
    prior, sweep = raw_setup(tr, dev)
    save_npz(work / "reference.npz", prior)
    rr = stats(dev["prior"], dev)
    write(
        work / "reference.json",
        dict(lam=float(prior["lam"]), choices=sweep, development=rr),
    )
    dim = (tr["slots"].shape[-1], tr["image"].shape[-1], tr["phi"].shape[-1])
    z = {k: jnp.asarray(tr[k]) for k in FEATS + ("y",)}
    g = tr["world"]
    _, inverse, ct = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / ct[inverse]
    w /= w.sum()
    recipes = [(c, a) for c in CORES for a in ARMS]
    folders = roots("fields")
    cfg_all = {}
    for seed in SEEDS:
        indices = (
            np.random.default_rng(seed).choice(len(g), (1024, 128), p=w).astype("i4")
        )
        for core, arm in recipes:
            tag = f"{stage}-{core}-{arm}-{seed}"
            p = jax.tree.map(jnp.asarray, initialize(dim, seed))
            m = jax.tree.map(jnp.zeros_like, p)
            v = jax.tree.map(jnp.zeros_like, p)
            curve = []
            start = 0
            cfg = dict(
                stage=stage,
                core=core,
                arm=arm,
                seed=seed,
                dim=list(dim),
                width=80,
                parameters=sum(x.size for x in p.values()),
                examples=len(g),
                batch=128,
                reference_sha=sha(work / "reference.npz"),
                data_sha=sha(data / "training.npz"),
                dev_sha=sha(data / "development.npz"),
            )
            for stop in CUTS:
                if complete(folders, tag, stop):
                    p, m, v, curve = load(folders, tag, stop, cfg)
                    start = stop
                    continue
                tic = time.monotonic()
                history = []
                for i in range(start, stop, 16):
                    (p, m, v), loss = kernel(core, arm)(
                        p, m, v, z, jnp.asarray(indices[i : i + 16]), jnp.int32(i)
                    )
                    a = np.asarray(loss)
                    if not np.isfinite(a).all():
                        raise ValueError("Unstable field update")
                    history.append(a.mean(0))
                end = time.monotonic()
                yy, risk = estimates(p, dev, core, arm)
                metric = stats(yy, dev)
                metric.update(
                    step=stop,
                    seconds=end - tic,
                    loss=np.mean(history, 0).tolist(),
                    objective=metric["regret_mm"]
                    + 0.15 * metric["effect_mm"]
                    + 0.20 * metric["utility_mae_mm"],
                )
                curve.append(metric)
                save(folders, tag, stop, p, m, v, cfg, curve)
                start = stop
            best = min(curve, key=lambda x: (x["objective"], x["step"]))
            stage_dir = folders[0] / tag / f"step-{best['step']:05d}"
            p2 = read(stage_dir / "weights.npz")
            pred, risk = estimates(jax.tree.map(jnp.asarray, p2), dev, core, arm)
            # Pick raw/neural shrinkage and a geometric fallback margin using actual fork consequences on development.
            options = []
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                for margin in (0.0, 0.04, 0.10, 0.22):
                    sc = (1 - alpha) * dev["prior"][:, :, -1] + alpha * pred[:, :, -1]
                    sel = sc.argmax(1)
                    ii = np.arange(len(sel))
                    sel = np.where(sc[ii, sel] - sc[:, 1] > margin, sel, 1)
                    regret = (dev["y"][:, :, -1].max(1) - dev["y"][ii, sel, -1]) * 40
                    options.append((mean_group(regret, dev["world"]), -alpha, margin))
            val, alpha, margin = min(options)
            cfg_all[tag] = dict(
                config=cfg,
                selected=best["step"],
                best=best,
                curve=curve,
                sha256=sha(stage_dir / "weights.npz"),
                alpha=float(-alpha),
                margin=margin,
                gated_regret_mm=val,
            )
            print(
                tag,
                best["step"],
                round(best["regret_mm"], 3),
                round(best["effect_mm"], 3),
                "alpha",
                -alpha,
                flush=True,
            )
    write(work / "lock.json", cfg_all)
    return cfg_all


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("initial",))
    a = p.parse_args()
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "cache"))
    fit(a.stage)
