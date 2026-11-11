"""Evaluate frozen action-conditioned effects and candidate ranking on new, physically executed forks."""

import json
import shutil
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from data.g14_policy import loaded, train_field
from data.g14_signals import pack, raw_predict, attach
from evals.g13_stats import Groups
from models.g14_world import predict
from scripts.train.g14_fit import FEATS
from scripts.train.g9_storage import write, sha

ROOT = Path("/mnt/slwm-data/SLWM/g14")
METRICS = (
    "effector_mm",
    "all_blocks_mm",
    "requested_block_mm",
    "moving_requested_block_mm",
    "short_regret_mm",
    "long_regret_mm",
)


def main():
    src = ROOT / "data/value-test-isolated.npz"
    work = ROOT / "work"
    dest = ROOT / "release/metrics"
    dest.mkdir(exist_ok=True, parents=True)
    with np.load(src, allow_pickle=False) as z:
        original = {k: z[k] for k in z.files}
    original["has_long"] = np.ones(len(original["world"]), bool)
    r = pack(original)
    world = r["world"]
    groups = np.unique(world)
    if list(groups) != list(range(286000, 286064)) or len(world) != 1024:
        raise ValueError("Inconsistent new fork bank")
    stage = json.loads((work / "final-control-lock.json").read_bytes())["stage"]
    ref, lp, _, _ = loaded(stage, "raw", "protected", 1411)
    attach(r, raw_predict(r, ref))
    r["long_prior"] = raw_predict(r, lp)[:, :, 0]
    true = original["effect"] * 1000.0
    idx = np.arange(len(world))
    focus = 1 + original["color"]
    actual_request = true[idx, :, focus, :]
    moving = np.linalg.norm(actual_request, axis=-1) >= 5
    short = r["y"][:, :, -1]
    long = r["long"]
    stores = {}
    full = {}

    def measure(name, forecast, pick1, pick2):
        predmm = 50 * np.einsum(
            "bkfi,bji->bkfj",
            forecast[:, :, :10].reshape(len(world), 8, 5, 2),
            r["basis"],
        )
        ae = np.linalg.norm(true - predmm, axis=-1)
        target = ae[idx, :, focus]
        rows = []
        for g in groups:
            m = world == g
            active = moving[m]
            rows.append(
                [
                    ae[m, :, 0].mean(),
                    ae[m, :, 1:].mean(),
                    target[m].mean(),
                    target[m][active].mean() if active.any() else np.nan,
                    ((short.max(1) - short[idx, pick1])[m] * 40).mean(),
                    ((long.max(1) - long[idx, pick2])[m] * 40).mean(),
                ]
            )
        stores[name] = np.asarray(rows)
        full[name] = dict(
            physical=predmm.astype("f4"),
            immediate=pick1.astype("i1"),
            delayed=pick2.astype("i1"),
        )

    measure(
        "persistence-nominal",
        np.zeros_like(r["prior"]),
        np.ones(len(world), int),
        np.ones(len(world), int),
    )
    measure(
        "raw", r["prior"], r["prior"][:, :, -1].argmax(1), r["long_prior"].argmax(1)
    )
    for core in ("gpt2", "transport"):
        for disable in (False, True):
            for seed in ((1411,) if disable else (1411, 1412, 1413)):
                _, _, p, _ = loaded(stage, core, "protected", seed)
                v = train_field(stage, core, seed) if disable else None
                fields = jax.jit(
                    lambda pp, b: predict(
                        pp,
                        b,
                        core,
                        "protected",
                        jnp,
                        True,
                        None if v is None else jnp.asarray(v),
                    )
                )
                ys = []
                gains = []
                params = {k: jnp.asarray(t) for k, t in p.items()}
                for i in range(0, len(world), 128):
                    b = {
                        k: jnp.asarray(r[k][i : i + 128])
                        for k in FEATS + ("long_prior",)
                    }
                    y, _, _, gain = fields(params, b)
                    ys.append(np.asarray(y))
                    gains.append(np.asarray(gain))
                y = np.concatenate(ys)
                gain = np.concatenate(gains)
                name = f"{core}-{seed}" + ("-mean" if disable else "")
                measure(name, y, y[:, :, -1].argmax(1), gain.argmax(1))
    for core in ("gpt2", "transport"):
        stores[core] = np.mean(
            [stores[f"{core}-{s}"] for s in (1411, 1412, 1413)], axis=0
        )
    boot = Groups(groups, groups % 2)
    summarize = lambda a: {k: boot.score(a[:, i]) for i, k in enumerate(METRICS)}
    comparisons = {
        name: summarize(stores[a] - stores[b])
        for name, a, b in (
            ("spatial_over_dense", "gpt2", "transport"),
            ("spatial_over_raw", "raw", "transport"),
            ("dense_over_raw", "raw", "gpt2"),
            ("spatial_over_persistence", "persistence-nominal", "transport"),
            ("dense_over_persistence", "persistence-nominal", "gpt2"),
            ("spatial_full_over_mean", "transport-1411-mean", "transport-1411"),
            ("dense_full_over_mean", "gpt2-1411-mean", "gpt2-1411"),
        )
    }
    keys = list(full)
    file = dest / "held-physics-predictions.npz"
    np.savez_compressed(
        file,
        world=world,
        color=original["color"],
        actual_mm=true,
        actual_long_mm=long * 40,
        actual_short_mm=short * 40,
        models=keys,
        predicted_mm=np.stack([full[k]["physical"] for k in keys]),
        short_choice=np.stack([full[k]["immediate"] for k in keys]),
        long_choice=np.stack([full[k]["delayed"] for k in keys]),
        metrics=METRICS,
        summary_models=list(stores),
        world_errors=np.stack(list(stores.values())),
        unique_worlds=groups,
    )
    shutil.copyfile(src, dest / "held-physics-inputs.npz")
    rec = dict(
        worlds=len(groups),
        states=len(world),
        macro_alternatives=int(moving.size),
        moving_requested_object_alternatives=int(moving.sum()),
        active_worlds=int(sum(moving[world == g].any() for g in groups)),
        metric_units="millimetres or mm-equivalent measured utility for regret. Positive paired differences favor named model.",
        scores={k: summarize(v) for k, v in stores.items()},
        comparisons=comparisons,
        input_sha256=sha(src),
        prediction_sha256=sha(file),
        registered_before_collection="ff6c949",
    )
    write(dest / "held-prediction.json", rec)
    print(
        "Independent physical forecasts",
        {
            k: (
                round(rec["scores"][k]["moving_requested_block_mm"]["value"], 2),
                round(rec["scores"][k]["long_regret_mm"]["value"], 2),
            )
            for k in ("persistence-nominal", "raw", "gpt2", "transport")
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
