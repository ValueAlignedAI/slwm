"""Match the training metric and preserve a causal prior in a 128-coordinate action field."""

import argparse
from functools import lru_cache
import hashlib
import json
import numpy as np
import jax
import jax.numpy as jnp
from evals.g13_predict import bank, arrays, parameters
from evals.g13_prospective import CORES, SEEDS
from evals.g13_stats import Groups
from models.g13_world import Spec, propose
from scripts.train.g13_robot_fit import ROOT, feature_keys
from scripts.train.g13_optim import save_npz
from scripts.train.g13_corpus_reference import design
from scripts.train.g9_storage import write, sha
from training.g13_physics import prepare, group_mean
from training.g13_adaptive import decorate

WORK = ROOT / "work/direct"


@lru_cache(maxsize=8)
def field_kernel(core):
    @jax.jit
    def run(p, r):
        out = propose(p, Spec(), core, "linear", r, "correct")
        z = out["field"]
        field = 0.5 * (z[:, 0] + z[:, 1:].mean(1))
        return field, out["action"].sum(1)

    return run


def data(role):
    if role != "last":
        return bank("repaired", role)
    path = ROOT / "data/robot-readout/corrected/test.npz"
    r = prepare(arrays(path), arrays(ROOT / "work/repaired/system-id.npz"))
    c = arrays(ROOT / "work/repaired/adaptive.npz")
    best = json.loads((ROOT / "work/repaired/adaptive-lock.json").read_bytes())["best"]
    return decorate(r, best, c, c["mass"])


def fields(role):
    r = data(role)
    dst = WORK / "fields" / role
    dst.mkdir(exist_ok=True, parents=True)
    for seed in SEEDS:
        for core in CORES:
            tag = f"{core}-linear-{seed}"
            file = dst / (tag + ".npz")
            if file.exists():
                continue
            d = json.loads((ROOT / "work/repaired/dev" / (tag + ".json")).read_bytes())
            p = parameters(d, "repaired")
            zs = []
            ys = []
            n = len(r["px"])
            for start in range(0, n, 256):
                ix = np.minimum(np.arange(start, start + 256), n - 1)
                row = {k: jnp.asarray(r[k][ix]) for k in feature_keys("linear")}
                z, y = field_kernel(core)(p, row)
                zs.append(np.asarray(z)[: min(256, n - start)])
                ys.append(np.asarray(y)[: min(256, n - start)])
            save_npz(file, dict(z=np.concatenate(zs), native=np.concatenate(ys)))
    (dst / "ready").touch()
    print("Cached preserved fields", role, flush=True)


def weights(g):
    _, inv, n = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / n[inv]
    return w / w.sum()


def regress(x, y, g, lam, robust):
    w = weights(g)
    x = x.astype("f8")
    y = y.astype("f8")
    mu = (x * w[:, None]).sum(0)
    sd = np.maximum(np.sqrt(((x - mu) ** 2 * w[:, None]).sum(0)), 0.04)
    xx = np.column_stack((np.ones(len(x)), (x - mu) / sd))
    pen = np.eye(xx.shape[1]) * lam
    pen[0, 0] = 0
    wi = w
    for _ in range(5 if robust else 1):
        beta = np.linalg.solve(
            xx.T @ (xx * wi[:, None]) + pen, xx.T @ (y * wi[:, None])
        )
        if robust:
            resid = np.linalg.norm(xx @ beta - y, axis=1)
            wi = w / np.maximum(0.015, resid)
            wi /= wi.sum()
    return dict(
        mu=mu,
        sd=sd,
        beta=beta[1:],
        base=beta[0],
        lam=np.array(lam),
        robust=np.array(bool(robust)),
    )


def apply(x, f):
    return (x - f["mu"]) / f["sd"] @ f["beta"] + f["base"]


def optimize(x, y, g, xd, yd, gd, prior=None):
    candidates = []
    for lam in (0.005, 0.04, 0.2, 1.0):
        for robust in (False, True):
            f = regress(x, y, g, lam, robust)
            raw = apply(xd, f)
            for alpha in (
                (1.0,) if prior is None else (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
            ):
                p = raw if prior is None else prior + alpha * raw
                value = group_mean(np.linalg.norm(p - yd, axis=1) * 1000, gd)
                candidates.append((value, alpha, lam, robust, f))
    best = min(candidates, key=lambda z: z[:4])
    score, alpha, lam, rob, f = best
    return f, dict(development_mm=score, alpha=alpha, lam=lam, robust=bool(rob))


def develop():
    tr = data("train")
    dev = data("dev")
    x = design(tr)
    xd = design(dev)
    y = tr["plan"].sum(1)
    yd = dev["plan"].sum(1)
    g = tr["group"]
    gd = dev["group"]
    folder = WORK / "probes"
    folder.mkdir(exist_ok=True, parents=True)
    raw, cfg = optimize(x, y, g, xd, yd, gd)
    path = folder / "raw.npz"
    save_npz(path, raw)
    locks = {"raw": dict(**cfg, sha256=sha(path))}
    pr_d = apply(xd, raw)
    pr_t = np.zeros_like(y)
    assignment = np.array(
        [
            int.from_bytes(
                hashlib.sha256(("G13/cross-fit/" + str(v)).encode()).digest()[:4], "big"
            )
            % 5
            for v in g
        ]
    )
    for i in range(5):
        fit = regress(
            x[assignment != i],
            y[assignment != i],
            g[assignment != i],
            cfg["lam"],
            cfg["robust"],
        )
        pr_t[assignment == i] = apply(x[assignment == i], fit)
    for seed in SEEDS:
        for c in CORES:
            tag = f"{c}-linear-{seed}"
            z = arrays(WORK / "fields/train" / (tag + ".npz"))["z"]
            v = arrays(WORK / "fields/dev" / (tag + ".npz"))["z"]
            for arm in ("field", "protected"):
                if arm == "field":
                    xx, dd, target, prior = z, v, y, None
                else:
                    xx = np.column_stack((pr_t / 0.08, z[:, :126]))
                    dd = np.column_stack((pr_d / 0.08, v[:, :126]))
                    target = y - pr_t
                    prior = pr_d
                fit, choice = optimize(xx, target, g, dd, yd, gd, prior)
                name = tag + "-" + arm
                file = folder / (name + ".npz")
                save_npz(file, fit)
                locks[name] = dict(**choice, sha256=sha(file))
            print("Development-aligned direct readouts", tag, flush=True)
    write(WORK / "lock.json", locks)


def confirm():
    lock = json.loads((WORK / "lock.json").read_bytes())
    r = data("last")
    g = r["group"]
    y = r["plan"].sum(1)
    bs = Groups(g)
    dst = ROOT / "release/metrics"
    folder = WORK / "probes"

    def fits(name):
        path = folder / (name + ".npz")
        if sha(path) != lock[name]["sha256"]:
            raise ValueError("Readout changed after development")
        return arrays(path)

    raw = apply(design(r), fits("raw"))
    err = lambda pred: 1000 * np.linalg.norm(pred - y, axis=1)
    base = err(raw)
    public = dict(
        raw=raw.astype("f4"), truth=y, group=g, episode=r["episode"], novel=r["novel"]
    )
    res = {}
    all_errors = {}
    for seed in SEEDS:
        for core in CORES:
            tag = f"{core}-linear-{seed}"
            d = arrays(WORK / "fields/last" / (tag + ".npz"))
            z = d["z"]
            probes = {}
            native = err(d["native"])
            public[tag + "_native"] = d["native"]
            for arm in ("field", "protected"):
                name = tag + "-" + arm
                pred = (
                    apply(z, fits(name))
                    if arm == "field"
                    else raw
                    + lock[name]["alpha"]
                    * apply(np.column_stack((raw / 0.08, z[:, :126])), fits(name))
                )
                ae = err(pred)
                public[name] = pred.astype("f4")
                all_errors[name] = ae
                probes[arm] = dict(
                    mm=bs.score(ae),
                    over_raw=bs.score(base - ae),
                    over_native=bs.score(native - ae),
                    over_raw_unseen_pair=bs.score(
                        np.where(r["novel"], base - ae, np.nan)
                    ),
                    alpha=lock[name]["alpha"],
                    robust=lock[name]["robust"],
                )
            res[tag] = dict(native=bs.score(native), arms=probes)
    for seed in SEEDS:
        for core in CORES:
            for arm in ("field", "protected"):
                tag = f"{core}-linear-{seed}"
                a = all_errors[tag + "-" + arm]
                b = all_errors[f"gpt2-linear-{seed}-{arm}"]
                res[tag]["arms"][arm]["over_dense"] = bs.score(b - a)
    average = np.mean(
        [base - all_errors[f"transport-linear-{s}-protected"] for s in SEEDS], axis=0
    )
    _, inv, count = np.unique(g, return_inverse=True, return_counts=True)
    v = np.bincount(inv, weights=average) / count
    rng = np.random.default_rng(411319)
    q = rng.multinomial(len(v), np.full(len(v), 1 / len(v)), size=10000) @ v / len(v)
    effect = bs.score(average)
    effect["lo99"], effect["hi99"] = map(float, np.quantile(q, [0.005, 0.995]))
    write(
        dst / "direct-action.json",
        dict(
            raw=bs.score(base),
            choice=lock["raw"],
            runs=res,
            five_seed_over_raw=effect,
            windows=len(g),
            episodes=len(np.unique(r["episode"])),
            groups=len(np.unique(g)),
            source_sha=sha(ROOT / "data/robot-readout/corrected/test.npz"),
        ),
    )
    np.savez_compressed(dst / "direct-actions.npz", **public)
    print("Direct readout", bs.score(base), effect, flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("stage", choices=("fields", "develop", "test"))
    a.add_argument("--role", choices=("train", "dev", "last"))
    p = a.parse_args()
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "cache"))
    (
        fields(p.role)
        if p.stage == "fields"
        else develop() if p.stage == "develop" else confirm()
    )
