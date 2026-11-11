"""Verify a causal last-command cache against a continuously displayed request."""

import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g10_control import live_kernel, load_stage
from scripts.train.g9_fit import save_json


def main(runs, out):
    out.mkdir(exist_ok=True, parents=True)
    for seed in (1001, 1002, 1003):
        for core in ("hybrid", "gpt2", "gated"):
            tag = f"context-{core}-variable-s{seed}"
            file = out / (tag + ".json")
            if file.exists():
                continue
            d, p, s = load_stage(runs / tag / "step-05120")
            args = (s, "full", 0, 256)
            saved = {}
            for mode in ("continuous", "brief", "latched"):
                saved[mode] = np.asarray(live_kernel(*args, mode)(p, jnp.int32(475801)))
            if not np.array_equal(saved["continuous"], saved["latched"]):
                raise ValueError(
                    "Cached observable commands failed to reproduce continuous delivery"
                )
            save_json(
                file,
                dict(
                    tag=tag,
                    sha256=d["inference_sha256"],
                    seed=475801,
                    worlds=256,
                    decisions_per_world=512,
                    causal_word_slots=8,
                    exact_control_trace=True,
                    occupancy={
                        k: v[:, :, 0].mean(0).tolist() for k, v in saved.items()
                    },
                ),
            )
            print(
                tag, "identical", round(saved["latched"][:, :, 0].mean(), 4), flush=True
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cache")
    a = ap.parse_args()
    if a.cache:
        jax.config.update("jax_compilation_cache_dir", a.cache)
    main(a.runs, a.out)
