"""Train the direction codec on disjoint speakers; choose exposure and augmentation from development."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from models.g13_speech import init, predict, augment
from scripts.train.g13_optim import ROOT, roots, update, complete, save, load
from scripts.train.g9_storage import sha, write

STEPS = (128, 512, 1024)


@lru_cache(maxsize=4)
def kernel(mask, portable=False):
    @jax.jit
    def go(p, m, v, bank, y, idx, start, seed):
        def step(state, args):
            pp, mm, vv = state
            ix, t = args
            x = bank[ix]
            if mask:
                x = augment(x, jax.random.fold_in(jax.random.PRNGKey(seed), t))

            def loss(model):
                logits = predict(model, x, portable)[1]
                return -jnp.take_along_axis(
                    jax.nn.log_softmax(logits), y[ix, None], axis=1
                ).mean()

            obj, grad = jax.value_and_grad(loss)(pp)
            return update(pp, mm, vv, grad, t, 1024, 0.0012), obj

        return jax.lax.scan(step, (p, m, v), (idx, start + 1 + jnp.arange(len(idx))))

    return go


@lru_cache(maxsize=2)
def inference(portable):
    return jax.jit(lambda p, x: jax.nn.softmax(predict(p, x, portable)[1], -1))


def probs(p, x, portable=False):
    result = []
    for i in range(0, len(x), 128):
        sel = np.minimum(np.arange(i, i + 128), len(x) - 1)
        result.append(
            np.asarray(inference(portable)(p, jnp.asarray(x[sel])))[
                : min(128, len(x) - i)
            ]
        )
    return np.concatenate(result)


def metrics(p, y, sp):
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-7))
    hit = p.argmax(-1) == y
    return dict(
        accuracy=float(np.mean([hit[sp == g].mean() for g in np.unique(sp)])),
        nll=float(np.mean([nll[sp == g].mean() for g in np.unique(sp)])),
        raw_accuracy=float(hit.mean()),
    )


def main(stage, base=ROOT, cloud=False, seed_only=None, portable=False):
    path = base / "data/speech/directions.npz"
    z = np.load(path, allow_pickle=False)
    x, y, role, speaker = [z[k] for k in ("x", "y", "role", "speaker")]
    train = np.flatnonzero(role == "train")
    dev = np.flatnonzero(role == "dev")
    test = np.flatnonzero(role == "test")
    folders = (
        (base / "checkpoints",)
        if cloud
        else roots("speech-portable" if portable else "speech")
    )
    specs = {}
    dst = base / "work" / ("speech-portable" if portable else "speech")
    dst.mkdir(parents=True, exist_ok=True)
    for seed in ((seed_only,) if seed_only else (1301, 1302, 1303)):
        indices = np.random.default_rng(seed).integers(
            0, len(train), (1024, 128), dtype=np.int32
        )
        for mode in ("plain", "masked"):
            tag = f"{mode}-{seed}"
            params = init(seed)
            m = jax.tree.map(jnp.zeros_like, params)
            v = jax.tree.map(jnp.zeros_like, params)
            cfg = dict(
                seed=seed,
                mode=mode,
                train_sha=sha(path),
                batch=128,
                steps=list(STEPS),
                parameters=sum(t.size for t in params.values()),
            )
            if portable:
                cfg["convolution"] = "explicit_symmetric_patch_matmul_v1"
            curve = []
            start = 0
            for stop in STEPS:
                if complete(folders, tag, stop):
                    params, m, v, curve = load(folders, tag, stop, cfg)
                    start = stop
                    continue
                if stage == "seal":
                    raise ValueError("Missing registered speech checkpoint")
                bank = jnp.asarray(x[train])
                ty = jnp.asarray(y[train])
                tic = time.monotonic()
                for i in range(start, stop, 16):
                    (params, m, v), loss = kernel(mode == "masked", portable)(
                        params,
                        m,
                        v,
                        bank,
                        ty,
                        jnp.asarray(indices[i : i + 16]),
                        jnp.int32(i),
                        jnp.int32(seed),
                    )
                    if not np.isfinite(np.asarray(loss)).all():
                        raise ValueError("Invalid speech loss")
                duration = time.monotonic() - tic
                prob = probs(params, x[dev], portable)
                curve.append(
                    dict(
                        step=stop,
                        seconds=duration,
                        **metrics(prob, y[dev], speaker[dev]),
                    )
                )
                save(folders, tag, stop, params, m, v, cfg, curve)
                print(tag, stop, curve[-1], flush=True)
                start = stop
            best = min(curve, key=lambda r: (r["nll"], r["step"]))
            specs[tag] = dict(
                tag=tag,
                selected=best["step"],
                dev=best,
                curve=curve,
                parameters=cfg["parameters"],
                device_kind=jax.devices()[0].device_kind,
            )
    write(
        dst / ("choices.json" if not seed_only else f"choices-{seed_only}.json"), specs
    )
    if stage != "seal":
        return
    # Choose one common edge by development NLL before opening speaker or robot test scores.
    chosen = min(specs, key=lambda k: (specs[k]["dev"]["nll"], k))
    results = {}
    for tag, d in specs.items():
        checkpoint = folders[0] / tag / f"step-{d['selected']:05d}"
        with np.load(checkpoint / "weights.npz") as f:
            p = {k: jnp.asarray(f[k]) for k in f.files}
        result = probs(p, x[test], portable)
        results[tag] = dict(
            **d,
            test=metrics(result, y[test], speaker[test]),
            weights_sha=sha(checkpoint / "weights.npz"),
        )
        if tag == chosen:
            probability = probs(p, x, portable)
            np.savez_compressed(dst / "posterior.npz", p=probability)
            np.savez_compressed(
                dst / "selected.npz", **{k: np.asarray(v) for k, v in p.items()}
            )
    write(
        dst / "results.json",
        dict(
            selected=chosen,
            runs=results,
            feature_sha=sha(path),
            speaker_split=json.loads(path.with_suffix(".json").read_bytes())["splits"],
        ),
    )
    print("Speech release", chosen, results[chosen]["test"], flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("stage", choices=("train", "seal"))
    a.add_argument("--seed", type=int)
    a.add_argument("--portable", action="store_true")
    a.add_argument("--v6", action="store_true")
    p = a.parse_args()
    base = Path("/content/g13") if p.v6 else ROOT
    if p.v6 and not any("v6" in d.device_kind.lower() for d in jax.devices()):
        raise ValueError("Expected a native v6 assignment")
    jax.config.update("jax_compilation_cache_dir", str(base / "cache"))
    main(p.stage, base, p.v6, p.seed, p.portable)
