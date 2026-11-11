"""Train, resume and mirror matched small world processors; select checkpoints only on development."""

import argparse
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np
from models.g12_world import CORES, MODES, Spec, initial, objective, proposals
from scripts.train.g9_storage import disks, write, sha

ROOT = Path("/mnt/slwm-data/SLWM/g12")
STEPS = (256, 512, 1024)
SEEDS = (1201, 1202, 1203)
SCHEDULE = dict(
    batch=96,
    updates=1024,
    warm=256,
    peak_lr=0.0005,
    terminal_lr=0.000055,
    weight_decay=0.015,
    clip=1.0,
    sample="uniform recordings within cued, reviewed, and uncued strata",
)


def groups_for(rows, mask):
    selected = np.flatnonzero(mask)
    if not len(selected):
        raise ValueError("Empty registered sampling stratum")
    _, inverse, counts = np.unique(
        rows["group"][selected], return_inverse=True, return_counts=True
    )
    w = 1.0 / counts[inverse]
    return selected, w / w.sum()


def draws(rows, seed, joint):
    random = np.random.default_rng(seed + (20000 if joint else 0))
    cued = groups_for(rows, rows["cued"])
    other = groups_for(rows, ~rows["cued"])

    def sample(group, n):
        return random.choice(group[0], (1024, n), p=group[1])

    pieces = [sample(cued, 64 if joint else 80)]
    if joint:
        pieces.append(sample(groups_for(rows, rows["cued"] & (rows["state"] >= 0)), 16))
    pieces.append(sample(other, 16))
    return np.concatenate(pieces, axis=1).astype(np.int32)


def read(root, fold, role, states):
    file = root / "banks" / f"{fold}-{role}.npz"
    meta = json.loads(file.with_suffix(".json").read_bytes())
    with np.load(file, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    if sha(file) != meta["sha256"]:
        raise ValueError("Input bank changed")
    data["state"] = np.array(
        [states.get((r["recording"], r["id"]), -1) for r in meta["rows"]], np.float32
    )
    return meta, data


@lru_cache(maxsize=32)
def block(s, core, treat, joint):
    @jax.jit
    def run(p, m, v, data, indices, start):
        def update(carry, args):
            par, first, second = carry
            ix, step = args
            row = {k: x[ix] for k, x in data.items()}
            (loss, parts), grad = jax.value_and_grad(objective, has_aux=True)(
                par, s, core, row, treat, joint
            )
            gnorm = jnp.sqrt(sum(jnp.vdot(g, g) for g in jax.tree.leaves(grad)))
            clip = jnp.minimum(1.0, 1.0 / jnp.maximum(gnorm, 1e-8))
            grad = jax.tree.map(lambda g: g * clip, grad)
            first = jax.tree.map(lambda a, g: 0.9 * a + 0.1 * g, first, grad)
            second = jax.tree.map(lambda b, g: 0.98 * b + 0.02 * g * g, second, grad)
            t = step.astype(jnp.float32)
            cos = 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.clip((t - 48) / 976, 0.0, 1.0)))
            lr = jnp.minimum(t / 48, 1.0) * (0.000055 + (0.0005 - 0.000055) * cos)

            def adjust(w, a, b):
                val = (a / (1.0 - 0.9**t)) / (jnp.sqrt(b / (1.0 - 0.98**t)) + 1e-8)
                return w - lr * (val + (0.015 * w if w.ndim == 2 else 0.0))

            par = jax.tree.map(adjust, par, first, second)
            return (par, first, second), jnp.concatenate([loss[None], parts])

        return jax.lax.scan(
            update, (p, m, v), (indices, start + 1 + jnp.arange(len(indices)))
        )

    return run


def pack(file, arr):
    tmp = file.with_suffix(".partial.npz")
    np.savez(tmp, **{k: np.asarray(x) for k, x in arr.items()})
    tmp.replace(file)


def read_params(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: jnp.asarray(f[k]) for k in f.files}


def outputs():
    return tuple(p.parent.parent / "g12" / "runs" for p in disks())


def checkpoint(paths, tag, step, config, p, m, v, curve):
    sub = Path(tag) / f"step-{step:05d}"
    a, b = (path / sub for path in paths)
    a.mkdir(exist_ok=True, parents=True)
    b.mkdir(exist_ok=True, parents=True)
    pack(a / "weights.npz", p)
    pack(
        a / "optimizer.npz",
        {**{"m_" + k: x for k, x in m.items()}, **{"v_" + k: x for k, x in v.items()}},
    )
    write(a / "config.json", config)
    hashes = {}
    for key in ("weights.npz", "optimizer.npz", "config.json"):
        hashes[key] = sha(a / key)
        dest = b / (key + ".partial")
        shutil.copyfile(a / key, dest)
        if sha(dest) != hashes[key]:
            raise ValueError("Checkpoint readback failed")
        dest.replace(b / key)
    rec = dict(step=step, tag=tag, sha256=hashes, curve=curve)
    write(b / "verified.json", rec)
    write(a / "verified.json", rec)


def restore(paths, tag, step):
    folders = [p / tag / f"step-{step:05d}" for p in paths]
    doc = json.loads((folders[0] / "verified.json").read_bytes())
    for path in folders:
        if json.loads((path / "verified.json").read_bytes()) != doc:
            raise ValueError("Mirror receipt mismatch")
        for name, checksum in doc["sha256"].items():
            if sha(path / name) != checksum:
                raise ValueError("Mutated registered checkpoint")
    p = read_params(folders[0] / "weights.npz")
    opt = read_params(folders[0] / "optimizer.npz")
    return p, {k: opt["m_" + k] for k in p}, {k: opt["v_" + k] for k in p}, doc["curve"]


@lru_cache(maxsize=64)
def inference(s, core, mode):
    @jax.jit
    def forward(p, row):
        out = proposals(p, s, core, row, mode)
        return (
            jax.nn.softmax(out["verb"], -1),
            jax.nn.sigmoid(out["state"]),
            jax.nn.sigmoid(out["certainty"]),
        )

    return forward


FEATURES = ("scene", "ac", "wc", "words", "wrong", "human", "wrong_human")


def predict(s, core, p, data, mode):
    values = [[], [], []]
    n = len(data["scene"])
    for start in range(0, n, 96):
        idx = np.minimum(np.arange(start, start + 96), n - 1)
        values_now = inference(s, core, mode)(
            p, {k: jnp.asarray(data[k][idx]) for k in FEATURES}
        )
        for v, acc in zip(values_now, values):
            acc.append(np.asarray(v)[: min(96, n - start)])
    return tuple(np.concatenate(v) for v in values)


def per_row_nll(prob, y):
    at = np.take_along_axis(prob, np.maximum(y, 0)[:, :, None], axis=2)[:, :, 0]
    return -np.log(np.maximum(np.where(y >= 0, at, 1e-7), 1e-7)).mean(1)


def train(phase):
    root = ROOT / "data"
    paths = outputs()
    labels = {}
    state_file = root / "state-labels.json"
    if phase == "joint":
        labels = {
            (r["recording"], r["id"]): r["label"]
            for r in json.loads(state_file.read_bytes())["items"]
        }
    state_sha = sha(state_file) if labels else None
    for fold in ("A", "B"):
        meta, rows = read(root, fold, "train", labels)
        dm, dev = read(root, fold, "dev", labels)
        s = Spec(
            rows["scene"].shape[-1], rows["words"].shape[-1], len(meta["vocabulary"])
        )
        train_device = {
            k: jnp.asarray(rows[k]) for k in FEATURES + ("y", "state", "groundable")
        }
        for seed in SEEDS:
            for treat in ("paired", "deranged", "grounded"):
                if (
                    treat == "deranged"
                    and seed != 1201
                    or treat == "grounded"
                    and seed == 1203
                ):
                    continue
                indices = draws(rows, seed, phase == "joint")
                for core in CORES:
                    tag = f"{fold}-{core}-{treat}-{seed}"
                    cuts = STEPS[:1] if phase == "warm" else STEPS[1:]
                    if all(
                        (p / tag / f"step-{cuts[-1]:05d}" / "verified.json").exists()
                        for p in paths
                    ):
                        continue
                    p = initial(s, seed)
                    m = jax.tree.map(jnp.zeros_like, p)
                    v = jax.tree.map(jnp.zeros_like, p)
                    curve = []
                    start = 0
                    if phase == "joint":
                        start = 256
                        p, m, v, curve = restore(paths, tag, start)
                    for stop in cuts:
                        if all(
                            (x / tag / f"step-{stop:05d}" / "verified.json").is_file()
                            for x in paths
                        ):
                            p, m, v, curve = restore(paths, tag, stop)
                            start = stop
                            continue
                        tic = time.monotonic()
                        measured = None
                        history = []
                        for lo in range(start, stop, 16):
                            (p, m, v), parts = block(s, core, treat, phase == "joint")(
                                p,
                                m,
                                v,
                                train_device,
                                jnp.asarray(indices[lo : lo + 16]),
                                jnp.int32(lo),
                            )
                            hist = np.array(parts)
                            if not np.isfinite(hist).all():
                                raise ValueError("Nonfinite training loss")
                            history.append(hist.mean(0))
                            if lo == start:
                                measured = time.monotonic()
                        finished = time.monotonic()
                        elapsed = finished - tic
                        pp, _, _ = predict(
                            s,
                            core,
                            p,
                            dev,
                            "wrong" if treat == "deranged" else "correct",
                        )
                        nll = per_row_nll(pp, dev["y"])
                        score = float(
                            np.mean(
                                [
                                    nll[dev["group"] == g].mean()
                                    for g in np.unique(dev["group"])
                                ]
                            )
                        )
                        curve = curve + [
                            dict(
                                step=stop,
                                development_nll=score,
                                loss=np.mean(history, axis=0).tolist(),
                                seconds=elapsed,
                                examples_per_s=(stop - start - 16)
                                * 96
                                / max(1e-7, finished - measured),
                            )
                        ]
                        config = dict(
                            fold=fold,
                            core=core,
                            seed=seed,
                            treatment=treat,
                            spec=asdict(s),
                            schedule=SCHEDULE,
                            parameters=sum(a.size for a in p.values()),
                            train_sha=meta["sha256"],
                            dev_sha=dm["sha256"],
                            state_sha=state_sha,
                            hardware=str(jax.devices()[0]),
                            vocabulary=meta["vocabulary"],
                        )
                        checkpoint(paths, tag, stop, config, p, m, v, curve)
                        print(
                            tag,
                            stop,
                            "development NLL",
                            round(score, 4),
                            "seconds",
                            round(elapsed, 1),
                            flush=True,
                        )
                        start = stop
    (ROOT / ("complete-" + phase)).touch()


def evaluate():
    root = ROOT / "data"
    out = ROOT / "eval"
    a, b = outputs()
    labels = {
        (r["recording"], r["id"]): r["label"]
        for r in json.loads((root / "state-labels.json").read_bytes())["items"]
    }
    train_eval = {}
    for fold in ("A", "B"):
        datasets = {
            role: read(root, fold, role, labels)
            for role in ("train", "dev", "replica", "confirm")
        }
        meta, rows = datasets["train"]
        sel = np.flatnonzero(rows["cued"])[::4][:384]
        datasets["train"] = (meta, {k: x[sel] for k, x in rows.items()})
        for tagpath in sorted(a.glob(fold + "-*")):
            tag = tagpath.name
            fold, core, treat, seed = tag.split("-")
            last = tagpath / "step-01024"
            if not (last / "verified.json").exists():
                raise ValueError("Missing terminal state " + tag)
            dest = out / tag
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / "receipt.json").is_file():
                continue
            config = json.loads((last / "config.json").read_bytes())
            s = Spec(**config["spec"])
            curve = json.loads((last / "verified.json").read_bytes())["curve"]
            picked = min(
                (c for c in curve if c["step"] in (512, 1024)),
                key=lambda c: (c["development_nll"], c["step"]),
            )
            base = tagpath / f"step-{picked['step']:05d}"
            p = read_params(base / "weights.npz")
            modes = (
                MODES
                if treat == "paired" and seed == "1201"
                else ("correct", "wrong", "none")
            )
            thresholds = None
            for role in ("dev", "confirm", "replica", "train"):
                _, batch = datasets[role]
                arrays = {}
                for mode in modes if role != "train" else ("correct", "wrong", "none"):
                    probs, state, conf = predict(s, core, p, batch, mode)
                    confidences = np.stack(
                        (
                            0.5 * probs.max(-1).mean(-1) + 0.5 * conf[:, 0],
                            0.5 * np.maximum(state, 1 - state) + 0.5 * conf[:, 1],
                        ),
                        axis=1,
                    )
                    arrays[mode + "_verb"] = probs
                    arrays[mode + "_state"] = state
                    arrays[mode + "_confidence"] = confidences
                    if role == "dev" and mode == "correct":
                        thresholds = np.quantile(confidences, 0.60, axis=0).tolist()
                tmp = dest / (role + ".new.npz")
                np.savez_compressed(tmp, **arrays)
                tmp.replace(dest / (role + ".npz"))
            write(
                dest / "receipt.json",
                dict(
                    tag=tag,
                    selected=picked["step"],
                    thresholds=thresholds,
                    weights_sha256=sha(base / "weights.npz"),
                    config=config,
                    curve=curve,
                    pred_sha256={r: sha(dest / (r + ".npz")) for r in datasets},
                    train_indices=sel.tolist(),
                ),
            )
            print("Released predictions", tag, picked["step"], flush=True)
    (ROOT / "evaluated").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("warm", "joint", "eval"))
    args = parser.parse_args()
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "cache"))
    if args.stage == "eval":
        evaluate()
    else:
        train(args.stage)
