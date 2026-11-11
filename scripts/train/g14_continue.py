"""Equal-exposure continuation training on teacher and learner trajectories with real forked returns."""

import argparse
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from data.g14_signals import pack, raw_setup, raw_fit, raw_predict, mean_group
from models.g14_world import initialize, predict
from scripts.train.g14_fit import FEATS, CUTS, SEEDS, kernel, stats
from scripts.train.g14_cue_fit import ROOT, roots, read
from scripts.train.g13_optim import save, load, complete, save_npz
from scripts.train.g9_storage import sha, write

SOURCES = {"expert-plus": "fresh-teacher", "visited-plus": "visited"}


def bank_lock():
    data = ROOT / "data"
    banks = {
        k: read(data / (k + ".npz")) for k in ("fresh-teacher", "visited", "value-dev")
    }
    keys = lambda b: np.stack([b[x] for x in ("world", "trial", "clock")], axis=1)
    a, b, v = (banks[k] for k in ("fresh-teacher", "visited", "value-dev"))
    if not np.array_equal(keys(a), keys(b)) or not np.array_equal(
        a["target"], b["target"]
    ):
        raise ValueError("Unmatched refresh exposure")
    if (
        len(np.unique(keys(a), axis=0)) != 2048
        or len(np.unique(keys(v), axis=0)) != 512
    ):
        raise ValueError("Incomplete refresh banks")
    if set(a["world"]) & set(v["world"]):
        raise ValueError("Development leakage")
    result = dict(
        matched_states=len(a["world"]),
        held_value_states=len(v["world"]),
        matched_scenes=len(set(a["world"])),
        value_scenes=len(set(v["world"])),
        candidates=8,
        native_ticks_per_return=15,
        hashes={k: sha(data / (k + ".npz")) for k in banks},
    )
    write(data / "aggregation-lock.json", result)
    return result


def calibrate(short, delayed, prior, value, world):
    ii = np.arange(len(world))
    best = None
    for alpha in (0.0, 0.5, 1.0):
        for beta in (0.0, 0.5, 1.0):
            scores = (1 - beta) * ((1 - alpha) * prior[0] + alpha * short) + beta * (
                (1 - alpha) * prior[1] + alpha * delayed
            )
            for margin in (0.0, 0.1, 0.3, 0.6):
                pick = scores.argmax(1)
                pick = np.where(scores[ii, pick] - scores[:, 1] > margin, pick, 1)
                regret = mean_group((value.max(1) - value[ii, pick]) * 40, world)
                key = (regret, -alpha, -beta, margin)
                if best is None or key < best:
                    best = key
    regret, neg_alpha, neg_beta, margin = best
    return dict(gated_regret_mm=regret, alpha=-neg_alpha, beta=-neg_beta, margin=margin)


def setup(stage):
    data = ROOT / "data"
    info = bank_lock()
    base = read(data / "training.npz")
    extra = read(data / (SOURCES[stage] + ".npz"))
    base["long"] = np.zeros_like(base["gain"])
    base["has_long"] = np.zeros(len(base["world"]), bool)
    extra["has_long"] = np.ones(len(extra["world"]), bool)
    tr = pack({k: np.concatenate((val, extra[k])) for k, val in base.items()})
    dv = read(data / "value-dev.npz")
    dv["has_long"] = np.ones(len(dv["world"]), bool)
    dv = pack(dv)
    prior, _ = raw_setup(tr, dv)

    def rows(r, ix):
        return dict(world=r["world"][ix], phi=r["phi"][ix], y=r["long"][ix, :, None])

    measured = tr["has_long"]
    valid = np.ones(len(dv["world"]), bool)
    Ltrain = rows(tr, measured)
    Ldev = rows(dv, valid)
    options = []
    for lam in (0.001, 0.008, 0.04, 0.2):
        p = raw_fit(Ltrain, lam)
        est = raw_predict(Ldev, p)[:, :, 0]
        v = dv["long"]
        err = mean_group(np.abs(est - v).mean(1) * 40, dv["world"])
        reg = mean_group(
            (v.max(1) - v[np.arange(len(v)), est.argmax(1)]) * 40, dv["world"]
        )
        options.append((reg + 0.25 * err, lam, p))
    _, lam, Lprior = min(options, key=lambda x: x[:2])
    tr["long_prior"] = raw_predict(tr, Lprior)[:, :, 0]
    for fold in range(5):
        fold_mask = tr["world"] % 5 == fold
        ref = raw_fit(rows(tr, measured & ~fold_mask), lam)
        tr["long_prior"][fold_mask] = raw_predict({"phi": tr["phi"][fold_mask]}, ref)[
            :, :, 0
        ]
    dv["long_prior"] = raw_predict(dv, Lprior)[:, :, 0]
    return tr, dv, prior, Lprior, info


def train(stage):
    tr, dev, prior, long_prior, info = setup(stage)
    work = ROOT / "work" / stage
    work.mkdir(parents=True, exist_ok=True)
    save_npz(work / "reference.npz", prior)
    save_npz(work / "long-reference.npz", long_prior)
    raw = (dev["prior"][:, :, -1], dev["long_prior"])
    gate = calibrate(*raw, raw, dev["long"], dev["world"])
    write(work / "reference.json", dict(short=stats(dev["prior"], dev), gate=gate))
    df = (tr["slots"].shape[-1], tr["image"].shape[-1], tr["phi"].shape[-1])
    folders = roots("fields")
    z = {k: jnp.asarray(tr[k]) for k in FEATS + ("y", "long", "long_prior", "has_long")}
    _, ix, ct = np.unique(tr["world"], return_inverse=True, return_counts=True)
    prob = 1.0 / ct[ix]
    prob /= prob.sum()
    batch = {k: jnp.asarray(dev[k]) for k in FEATS + ("long_prior",)}
    all_choices = {}
    for core in ("gpt2", "transport"):
        evaluate = jax.jit(lambda p, x: predict(p, x, core, "protected", jnp, True))
        for seed in SEEDS:
            tag = f"{stage}-{core}-protected-{seed}"
            params = initialize(df, seed)
            params["continuation"] = (
                np.random.default_rng(seed + 4800).standard_normal((80, 1)) * 0.008
            ).astype("f4")
            p = jax.tree.map(jnp.asarray, params)
            m = jax.tree.map(jnp.zeros_like, p)
            v = jax.tree.map(jnp.zeros_like, p)
            indices = (
                np.random.default_rng(seed)
                .choice(len(prob), (1024, 128), p=prob)
                .astype("i4")
            )
            cfg = dict(
                stage=stage,
                core=core,
                arm="protected",
                seed=seed,
                dim=list(df),
                parameters=sum(x.size for x in p.values()),
                batch=128,
                examples=len(prob),
                observed_continuations=int(tr["has_long"].sum()),
                data_sha=sha(ROOT / "data/training.npz"),
                augmentation_sha=info["hashes"][SOURCES[stage]],
                dev_sha=info["hashes"]["value-dev"],
                reference_sha=sha(work / "reference.npz"),
                long_reference_sha=sha(work / "long-reference.npz"),
                width=80,
            )
            start = 0
            curve = []
            for end in CUTS:
                if complete(folders, tag, end):
                    p, m, v, curve = load(folders, tag, end, cfg)
                    start = end
                    continue
                t = time.monotonic()
                loss = []
                for i in range(start, end, 16):
                    (p, m, v), hist = kernel(core, "protected")(
                        p, m, v, z, jnp.asarray(indices[i : i + 16]), jnp.int32(i)
                    )
                    h = np.asarray(hist)
                    if not np.isfinite(h).all():
                        raise ValueError("Invalid continuation gradients")
                    loss.append(h.mean(0))
                y, _, _, long = evaluate(p, batch)
                y = np.asarray(y)
                long = np.asarray(long)
                metrics = stats(y, dev)
                v0 = dev["long"]
                ii = np.arange(len(v0))
                reg = mean_group(
                    (v0.max(1) - v0[ii, long.argmax(1)]) * 40, dev["world"]
                )
                metrics.update(
                    step=end,
                    loss=np.mean(loss, 0).tolist(),
                    seconds=time.monotonic() - t,
                    long_regret_mm=reg,
                    objective=reg
                    + 0.15 * metrics["effect_mm"]
                    + 0.1 * metrics["regret_mm"],
                )
                curve.append(metrics)
                save(folders, tag, end, p, m, v, cfg, curve)
                start = end
            best = min(curve, key=lambda x: (x["objective"], x["step"]))
            file = folders[0] / tag / f"step-{best['step']:05d}" / "weights.npz"
            y, _, _, long = evaluate(jax.tree.map(jnp.asarray, read(file)), batch)
            choice = calibrate(
                np.asarray(y)[:, :, -1],
                np.asarray(long),
                raw,
                dev["long"],
                dev["world"],
            )
            all_choices[tag] = dict(
                config=cfg,
                selected=best["step"],
                best=best,
                curve=curve,
                sha256=sha(file),
                **choice,
            )
            print(tag, best["step"], choice, flush=True)
    write(work / "lock.json", all_choices)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=tuple(SOURCES))
    a = p.parse_args()
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "cache"))
    train(a.stage)
