"""Audit the trained one-tick sensor heads on recorded human tasks, including a persistence reference."""

import argparse
from functools import lru_cache
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g10_control import load_stage
from evals.g10_real import Bank
from models.g9_shared import decode
from models.g10_alignment import encode_record
from scripts.train.g9_fit import save_json


@lru_cache(maxsize=4)
def forecast(s):
    @jax.jit
    def apply(p, r, keep):
        z = encode_record(p, s, r, jnp.broadcast_to(keep, (len(r["a"]), 4)))
        v = decode(p, z, r["u"])
        return v["next_a"], v["next_n"]

    return apply


def errors(p, s, bank, prior):
    starts, g = bank.selection()
    results = {
        k: {} for k in ("all", "hidden_imu", "words_only", "persistence", "prior")
    }
    for i in range(0, len(starts), 64):
        subset = starts[i : i + 64]
        row = bank.samples(subset, g[i : i + 64])
        ix = subset[:, None] + np.arange(32) + 1
        gold_a = (bank.data["a"][ix].astype(np.float32) / 127.5 - 1)[:, :, :16]
        gold_n = bank.data["n"][ix]
        flags = (gold_n[:, :, 6] > 0) & (row["n"][:, :, 6] > 0)
        inputs = {}
        for name, mask in (
            ("all", [1, 1, 1, 1]),
            ("hidden_imu", [1, 1, 0, 1]),
            ("words_only", [0, 0, 0, 1]),
        ):
            a, n = forecast(s)(p, row, np.array(mask, bool))
            inputs[name] = (np.asarray(a)[:, :, :16], np.asarray(n))
        inputs["persistence"] = (
            row["a"][:, :, :16].astype(np.float32) / 127.5 - 1,
            row["n"],
        )
        inputs["prior"] = prior
        for name, (a, n) in inputs.items():
            for kind, sq, ok in (
                ("audio", ((gold_a - a) ** 2).mean(-1), np.ones_like(flags)),
                (
                    "acceleration",
                    (((gold_n[:, :, :3] - n[..., :3]) * 12) ** 2).mean(-1),
                    flags,
                ),
                (
                    "angular_velocity",
                    (((gold_n[:, :, 3:6] - n[..., 3:6]) * 4) ** 2).mean(-1),
                    flags,
                ),
            ):
                mask = ok.reshape(-1, 8, 32)
                den = mask.sum((1, 2))
                values = np.sqrt(
                    (sq.reshape(-1, 8, 32) * mask).sum((1, 2)) / np.maximum(1, den)
                )
                results[name].setdefault(kind, []).extend(
                    [float(v) if count else None for v, count in zip(values, den)]
                )
    return dict(recordings=bank.names, rmse=results)


def main(runs, data, out):
    out.mkdir(exist_ok=True, parents=True)
    with np.load(data / "holo-train.npz") as f:
        a = f["a"][:, :16].mean(0) / 127.5 - 1
        n = f["n"]
        n = n[n[:, 6] > 0].mean(0)
    banks = {
        k: Bank(data / ("holo-" + k + ".npz"), True) for k in ("familiar", "novel")
    }
    for seed in (1001, 1002, 1003):
        for mode in ("paired", "reconstruct"):
            for c in ("hybrid", "gpt2"):
                tag = f"align-{c}-{mode}-s{seed}"
                stage = runs / tag / "step-01536"
                dest = out / (tag + ".json")
                if dest.exists():
                    continue
                if not (stage / "verified.json").exists():
                    raise RuntimeError("Forecast parent has not arrived")
                rec, p, s = load_stage(stage)
                doc = dict(
                    tag=tag,
                    sha256=rec["inference_sha256"],
                    horizon_seconds=0.1,
                    units=dict(
                        audio="normalized mono log-band features",
                        acceleration="released accelerometer units",
                        angular_velocity="released gyroscope units",
                    ),
                    panels={k: errors(p, s, b, (a, n)) for k, b in banks.items()},
                )
                save_json(dest, doc)
                print(tag, "forecast", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("holo", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    main(a.runs, a.holo, a.out)
