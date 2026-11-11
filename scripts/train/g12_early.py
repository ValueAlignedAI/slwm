"""Joint action/state learning from update one, early selection, and paired whole-modality dropout."""

import json
import time
from dataclasses import asdict
import jax
import jax.numpy as jnp
import numpy as np
from models.g12_world import CORES, Spec, initial
from scripts.train.g12_fit import (
    ROOT,
    SCHEDULE,
    SEEDS,
    FEATURES,
    outputs,
    read,
    draws,
    groups_for,
    block,
    predict,
    per_row_nll,
    checkpoint,
    restore,
    read_params,
)
from scripts.train.g9_storage import sha, write

CUTS = (32, 96, 192, 384)
ARMS = ("full", "dropout")


def macro(a, group):
    valid = np.isfinite(a)
    return float(
        np.mean([a[(group == g) & valid].mean() for g in np.unique(group[valid])])
    )


def reference(train, dev, classes):
    cu, weight = groups_for(train, train["cued"])
    y = train["y"][cu]
    freq = np.full((3, classes), 1e-5)
    for j in range(3):
        np.add.at(freq[j], y[:, j], weight)
    freq /= freq.sum(-1, keepdims=True)
    nll = macro(
        per_row_nll(np.broadcast_to(freq, (len(dev["y"]), 3, classes)), dev["y"]),
        dev["group"],
    )
    p = macro(np.where(train["state"] >= 0, train["state"], np.nan), train["group"])
    brier = macro(
        np.where(dev["state"] >= 0, (dev["state"] - p) ** 2, np.nan), dev["group"]
    )
    return nll, brier


def eval_one(path, s, core, tag, dest, datasets, idx, selection):
    config = json.loads((path / "config.json").read_bytes())
    p = read_params(path / "weights.npz")
    directory = dest / tag
    directory.mkdir(exist_ok=True, parents=True)
    thresholds = None
    for role, (_, rows) in datasets.items():
        batch = {k: x[idx] for k, x in rows.items()} if role == "train" else rows
        arr = {}
        for mode in ("correct", "wrong", "none"):
            prob, st, conf = predict(s, core, p, batch, mode)
            conf = np.stack(
                (
                    0.5 * (prob.max(-1).mean(-1) + conf[:, 0]),
                    0.5 * (np.maximum(st, 1 - st) + conf[:, 1]),
                ),
                1,
            )
            arr[mode + "_verb"] = prob
            arr[mode + "_state"] = st
            arr[mode + "_confidence"] = conf
            if role == "dev" and mode == "correct":
                thresholds = np.quantile(conf, 0.6, axis=0).tolist()
        tmp = directory / (role + ".new.npz")
        np.savez_compressed(tmp, **arr)
        tmp.replace(directory / (role + ".npz"))
    write(
        directory / "receipt.json",
        dict(
            tag=tag,
            config=config,
            selected=selection["step"],
            curve=selection["curve"],
            weights_sha256=sha(path / "weights.npz"),
            thresholds=thresholds,
            train_indices=idx.tolist(),
            pred_sha256={r: sha(directory / (r + ".npz")) for r in datasets},
        ),
    )


def main():
    roots = tuple(p.parent / "early" for p in outputs())
    root = ROOT / "data"
    labels_path = root / "state-labels.json"
    labels = {
        (r["recording"], r["id"]): r["label"]
        for r in json.loads(labels_path.read_bytes())["items"]
    }
    for fold in ("A", "B"):
        datasets = {
            r: read(root, fold, r, labels)
            for r in ("train", "dev", "confirm", "replica")
        }
        meta, rows = datasets["train"]
        dm, dev = datasets["dev"]
        s = Spec(
            rows["scene"].shape[-1], rows["words"].shape[-1], len(meta["vocabulary"])
        )
        train_device = {
            k: jnp.asarray(rows[k]) for k in FEATURES + ("y", "state", "groundable")
        }
        rn, rb = reference(rows, dev, s.verbs)
        ix = np.flatnonzero(rows["cued"])[::4][:384]
        for seed in SEEDS:
            indices = draws(rows, seed, True)
            for core in CORES:
                for arm in ARMS:
                    tag = f"{fold}-{core}-{arm}-{seed}"
                    dest = ROOT / "early-eval"
                    if (dest / tag / "receipt.json").is_file():
                        continue
                    p = initial(s, seed)
                    m = jax.tree.map(jnp.zeros_like, p)
                    v = jax.tree.map(jnp.zeros_like, p)
                    start = 0
                    curve = []
                    for stop in CUTS:
                        if all(
                            (r / tag / f"step-{stop:05d}" / "verified.json").exists()
                            for r in roots
                        ):
                            p, m, v, curve = restore(roots, tag, stop)
                            start = stop
                            continue
                        tic = time.monotonic()
                        parts = []
                        measure = tic
                        for lo in range(start, stop, 16):
                            (p, m, v), loss = block(s, core, arm, True)(
                                p,
                                m,
                                v,
                                train_device,
                                jnp.asarray(indices[lo : lo + 16]),
                                jnp.int32(lo),
                                jnp.int32(seed),
                            )
                            obs = np.asarray(loss)
                            if not np.isfinite(obs).all():
                                raise ValueError("Nonfinite early fit")
                            parts.append(obs.mean(0))
                            if lo == start:
                                measure = time.monotonic()
                        finished = time.monotonic()
                        prob, st, _ = predict(s, core, p, dev, "correct")
                        nll = macro(per_row_nll(prob, dev["y"]), dev["group"])
                        brier = macro(
                            np.where(
                                dev["state"] >= 0, (dev["state"] - st) ** 2, np.nan
                            ),
                            dev["group"],
                        )
                        curve = curve + [
                            dict(
                                step=stop,
                                development_nll=nll,
                                development_brier=brier,
                                selection=0.5 * (nll / rn + brier / rb),
                                sequence_reference=rn,
                                state_reference=rb,
                                loss=np.mean(parts, axis=0).tolist(),
                                seconds=finished - tic,
                                examples_per_s=(stop - start - 16)
                                * 96
                                / max(1e-8, finished - measure),
                            )
                        ]
                        cfg = dict(
                            fold=fold,
                            core=core,
                            seed=seed,
                            treatment=arm,
                            spec=asdict(s),
                            schedule=dict(
                                SCHEDULE, joint_from=1, updates=384, cuts=list(CUTS)
                            ),
                            parameters=sum(x.size for x in p.values()),
                            train_sha=meta["sha256"],
                            dev_sha=dm["sha256"],
                            state_sha=sha(labels_path),
                            device_kind=jax.devices()[0].device_kind,
                            vocabulary=meta["vocabulary"],
                        )
                        checkpoint(roots, tag, stop, cfg, p, m, v, curve)
                        start = stop
                    best = min(curve, key=lambda c: (c["selection"], c["step"]))
                    eval_one(
                        roots[0] / tag / f"step-{best['step']:05d}",
                        s,
                        core,
                        tag,
                        dest,
                        datasets,
                        ix,
                        dict(step=best["step"], curve=curve),
                    )
                    print(
                        tag,
                        "selected",
                        best["step"],
                        "dev sequence and state",
                        round(best["development_nll"], 3),
                        round(best["development_brier"], 3),
                        flush=True,
                    )
    (ROOT / "early-evaluated").touch()


if __name__ == "__main__":
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "cache"))
    main()
