"""Run a group-balanced development tournament, then confirm locked physical processors on new groups."""

import argparse
from dataclasses import asdict
from functools import lru_cache
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
from models.g13_world import CORES, RECIPES, Spec, initial, propose, loss
from scripts.train.g13_optim import ROOT, roots, update, complete, save, load, save_npz
from scripts.train.g9_storage import write, sha
from training.g13_physics import identify, prepare, group_mean

CUTS = (128, 512, 1024)
SEEDS = (1301, 1302, 1303)
KEYS = (
    "px",
    "geo",
    "kin",
    "plan",
    "alt_plan",
    "goal",
    "wrong",
    "bind",
    "wb",
    "ref_active",
    "ref_wrong",
    "ref_zero",
    "ref_passive",
    "ref_reset",
    "ref_pass_reset",
)
TARGET = ("mover", "future", "flow", "visible", "first")


def read(role, corrected=False, external=False):
    file = (
        ROOT
        / "data"
        / ("robot-extra" if external else "robot")
        / ("corrected" if corrected else "banks")
        / (role + ".npz")
    )
    with np.load(file) as z:
        return {k: z[k] for k in z.files}


def sample(r, seed):
    rng = np.random.default_rng(seed)

    def draw(mask, num):
        ix = np.flatnonzero(mask)
        _, inverse, counts = np.unique(
            r["group"][ix], return_inverse=True, return_counts=True
        )
        w = 1.0 / counts[inverse]
        return rng.choice(ix, (1024, num), p=w / w.sum())

    return np.concatenate(
        (
            draw(r["mover"] >= 0, 56),
            draw(r["first"], 32),
            draw(np.ones(len(r["first"]), bool), 40),
        ),
        1,
    ).astype(np.int32)


@lru_cache(maxsize=64)
def block(s, core, recipe):
    @jax.jit
    def train(p, m, v, data, indices, start):
        def step(state, a):
            par, mm, vv = state
            ix, t = a
            row = {k: d[ix] for k, d in data.items()}
            (value, terms), grad = jax.value_and_grad(loss, has_aux=True)(
                par, s, core, recipe, row
            )
            return update(par, mm, vv, grad, t, 1024, 0.0007), jnp.concatenate(
                (value[None], terms)
            )

        return jax.lax.scan(
            step, (p, m, v), (indices, start + 1 + jnp.arange(len(indices)))
        )

    return train


@lru_cache(maxsize=128)
def kernel(s, core, recipe, mode):
    @jax.jit
    def apply(p, r):
        out = propose(p, s, core, recipe, r, mode)
        return dict(
            mover=jax.nn.softmax(out["mover"]),
            active=out["active"],
            passive=out["passive"],
            flow=out["flow"],
            action=out["action"],
            confidence=jax.nn.sigmoid(out["certainty"]),
        )

    return apply


def feature_keys(recipe):
    extra = (
        (
            "adapt_active",
            "adapt_wrong",
            "adapt_zero",
            "adapt_reset",
            "p_goal",
            "p_wrong",
            "p_none",
        )
        if recipe in ("linear", "adaptive", "anchored")
        else ()
    )
    return KEYS + extra


def prediction(p, s, core, recipe, r, mode="correct"):
    result = {}
    n = len(r["px"])
    for i in range(0, n, 256):
        idx = np.minimum(np.arange(i, i + 256), n - 1)
        obj = kernel(s, core, recipe, mode)(
            p, {k: jnp.asarray(r[k][idx]) for k in feature_keys(recipe)}
        )
        for k, v in obj.items():
            result.setdefault(k, []).append(np.asarray(v)[: min(256, n - i)])
    return {k: np.concatenate(v) for k, v in result.items()}


def errors(out, r):
    y = r["mover"]
    known = y >= 0
    cls = out["mover"].argmax(1)
    p = out["mover"][np.arange(len(y)), np.maximum(y, 0)]
    return dict(
        accuracy=np.where(known, cls == y, np.nan),
        nll=np.where(known, -np.log(np.maximum(p, 1e-7)), np.nan),
        active=np.linalg.norm(out["active"][:, -1] - r["future"][:, -1], axis=-1)
        * 1000,
        passive=np.linalg.norm(out["passive"][:, -1] - r["future"][:, -1], axis=-1)
        * 1000,
        action=np.linalg.norm(out["action"].sum(1) - r["plan"].sum(1), axis=-1) * 1000,
    )


def dev_reference(tr, dev):
    y = tr["mover"]
    valid = y >= 0
    _, inv, n = np.unique(tr["group"][valid], return_inverse=True, return_counts=True)
    prior = np.bincount(y[valid], weights=1.0 / n[inv], minlength=5) + 0.01
    prior /= prior.sum()
    tar = dev["mover"]
    return dict(
        prior=prior.tolist(),
        nll=group_mean(
            np.where(tar >= 0, -np.log(prior[np.maximum(tar, 0)]), np.nan), dev["group"]
        ),
        active=group_mean(
            np.linalg.norm(dev["ref_active"][:, -1] - dev["future"][:, -1], axis=1)
            * 1000,
            dev["group"],
        ),
        passive=group_mean(
            np.linalg.norm(dev["ref_passive"][:, -1] - dev["future"][:, -1], axis=1)
            * 1000,
            dev["group"],
        ),
    )


def choices(work):
    pilot = [
        json.loads(f.read_bytes())
        for f in (work / "dev").glob("*-1301.json")
        if json.loads(f.read_bytes())["recipe"] in RECIPES
    ]
    if len(pilot) != len(CORES) * len(RECIPES):
        raise ValueError("Incomplete development tournament")
    med = {
        r: float(np.median([v["best"]["nll"] for v in pilot if v["recipe"] == r]))
        for r in ("slots", "bound")
    }
    return ["flat", min(med, key=med.get), "residual"], med


def train_phase(phase):
    repaired = phase in ("corrected", "newseeds")
    work = ROOT / "work" / ("repaired" if repaired else "robot")
    (work / "dev").mkdir(exist_ok=True, parents=True)
    folders = roots("robot-corrected" if repaired else "robot")
    s = Spec()
    tr = read("train", repaired)
    coef = identify(tr)
    save_npz(work / "system-id.npz", coef)
    dev = prepare(read("dev", repaired), coef)
    tr = prepare(tr, coef)
    ref = dev_reference(tr, dev)
    if repaired:
        from training.g13_adaptive import fit_basis, role_prior, decorate

        best, selection, coefficients = fit_basis(tr, dev)
        mass = role_prior(tr)
        save_npz(work / "adaptive.npz", {**coefficients, "mass": mass})
        write(work / "adaptive-lock.json", dict(best=best, candidates=selection))
        tr = decorate(tr, best, coefficients, mass)
        dev = decorate(dev, best, coefficients, mass)
        print("Development adaptive physical reference", best, flush=True)
    write(work / "dev-reference.json", ref)
    seed_list = (
        (1311, 1312)
        if phase == "newseeds"
        else SEEDS if repaired else (1301,) if phase == "pilot" else SEEDS[1:]
    )
    recipes = (
        ("linear",)
        if phase == "newseeds"
        else (
            ("linear", "adaptive", "anchored")
            if repaired
            else (
                RECIPES
                if phase == "pilot"
                else json.loads((work / "recipe-lock.json").read_bytes())["recipes"]
            )
        )
    )
    bank = {k: jnp.asarray(tr[k]) for k in feature_keys(recipes[0]) + TARGET}
    input_dir = ROOT / "data/robot" / ("corrected" if repaired else "banks")
    train_sha = sha(input_dir / "train.npz")
    dev_sha = sha(input_dir / "dev.npz")
    for seed in seed_list:
        idx = sample(tr, seed)
        for recipe in recipes:
            for core in (
                ("gpt2", "dense_spatial", "associative", "transport")
                if phase == "newseeds"
                else CORES
            ):
                tag = f"{core}-{recipe}-{seed}"
                file = work / "dev" / (tag + ".json")
                if file.is_file():
                    continue
                p = initial(s, seed)
                m = jax.tree.map(jnp.zeros_like, p)
                v = jax.tree.map(jnp.zeros_like, p)
                start = 0
                curve = []
                cfg = dict(
                    core=core,
                    recipe=recipe,
                    seed=seed,
                    spec=asdict(s),
                    train_sha=train_sha,
                    dev_sha=dev_sha,
                    parameters=sum(a.size for a in p.values()),
                    steps=list(CUTS),
                    batch=128,
                )
                for stop in CUTS:
                    if complete(folders, tag, stop):
                        p, m, v, curve = load(folders, tag, stop, cfg)
                        start = stop
                        continue
                    tic = time.monotonic()
                    summ = []
                    measure = tic
                    for i in range(start, stop, 16):
                        (p, m, v), val = block(s, core, recipe)(
                            p, m, v, bank, jnp.asarray(idx[i : i + 16]), jnp.int32(i)
                        )
                        z = np.asarray(val)
                        if not np.isfinite(z).all():
                            raise ValueError("Non-finite physical fit")
                        summ.append(z.mean(0))
                        if i == start:
                            measure = time.monotonic()
                    end = time.monotonic()
                    out = prediction(p, s, core, recipe, dev)
                    score = {
                        k: group_mean(v, dev["group"])
                        for k, v in errors(out, dev).items()
                    }
                    score["objective"] = (
                        0.5 * score["nll"] / ref["nll"]
                        + 0.35 * score["active"] / ref["active"]
                        + 0.15 * score["passive"] / ref["passive"]
                    )
                    score.update(
                        step=stop,
                        seconds=end - tic,
                        examples_per_second=(stop - start - 16)
                        * 128
                        / max(1e-8, end - measure),
                        loss=np.mean(summ, axis=0).tolist(),
                    )
                    curve.append(score)
                    save(folders, tag, stop, p, m, v, cfg, curve)
                    start = stop
                best = min(curve, key=lambda x: (x["objective"], x["step"]))
                write(
                    file,
                    dict(
                        tag=tag,
                        core=core,
                        recipe=recipe,
                        seed=seed,
                        cfg=cfg,
                        curve=curve,
                        best=best,
                        weights_sha=sha(
                            folders[0]
                            / tag
                            / f"step-{best['step']:05d}"
                            / "weights.npz"
                        ),
                    ),
                )
                print(
                    tag,
                    best["step"],
                    round(best["accuracy"], 3),
                    round(best["active"], 2),
                    round(best["passive"], 2),
                    flush=True,
                )
    if phase == "pilot":
        recipes, values = choices(work)
        write(
            work / "recipe-lock.json",
            dict(recipes=recipes, development_nll=values, reference=ref),
        )
    (work / ("done-" + phase)).touch()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("phase", choices=("pilot", "confirm", "corrected", "newseeds"))
    p = a.parse_args()
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "cache"))
    train_phase(p.phase)


if __name__ == "__main__":
    main()
