"""Benchmark the matched, trained forecast heads at live and batched inference sizes."""

import json
import time
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from data.g14_policy import loaded
from data.g14_signals import pack, raw_predict, attach
from models.g14_world import predict
from scripts.train.g14_fit import FEATS
from scripts.train.g9_storage import write

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def run(post_experiment=False):
    stage = json.loads((ROOT / "work/final-control-lock.json").read_bytes())["stage"]
    with np.load(ROOT / "data/development.npz") as data:
        src = {
            k: data[k][:512]
            for k in (
                "q",
                "p",
                "u",
                "seen",
                "motor",
                "px",
                "choices",
                "color",
                "target",
            )
        }
    result = {}
    seed = 1411
    for core in ("gpt2", "transport"):
        ref, delayed, p, _ = loaded(stage, core, "protected", seed)
        allrows = pack(src)
        attach(allrows, raw_predict(allrows, ref))
        if delayed is not None:
            allrows["long_prior"] = raw_predict(allrows, delayed)[:, :, 0]
        keys = FEATS + (("long_prior",) if delayed is not None else ())
        batch = lambda n: {k: v[:n] for k, v in allrows.items() if k in keys}
        fn = jax.jit(
            lambda pp, r: predict(pp, r, core, "protected", jnp, delayed is not None)
        )
        gpu = {k: jnp.asarray(v) for k, v in p.items()}
        rows = {}
        for n in (1, 32, 128, 512):
            x = {k: jnp.asarray(v) for k, v in batch(n).items()}
            jax.block_until_ready(fn(gpu, x))
            if post_experiment:
                warm = time.perf_counter() + 0.35
                while time.perf_counter() < warm:
                    jax.block_until_ready(fn(gpu, x))
            vals = []
            for i in range(40):
                t = time.perf_counter()
                jax.block_until_ready(fn(gpu, x))
                vals.append(time.perf_counter() - t)
            med = np.median(vals)
            rows[n] = dict(
                milliseconds=float(1000 * med),
                p90_ms=float(1000 * np.quantile(vals, 0.9)),
                scenes_per_second=float(n / med),
            )
            if post_experiment:
                block_times = []
                for repetition in range(9):
                    start = time.perf_counter()
                    outputs = [fn(gpu, x) for _ in range(16)]
                    jax.block_until_ready(outputs)
                    block_times.append(time.perf_counter() - start)
                rows[n]["pipelined_scenes_per_second"] = float(
                    16 * n / np.median(block_times)
                )
        x = batch(1)
        predict(p, x, core, "protected", np, delayed is not None)
        vals = []
        for i in range(80):
            t = time.perf_counter()
            predict(p, x, core, "protected", np, delayed is not None)
            vals.append(time.perf_counter() - t)
        result[core] = dict(
            cuda=rows, numpy_cpu_one_median_ms=float(np.median(vals) * 1000)
        )
    write(
        ROOT
        / ("work/throughput-post.json" if post_experiment else "work/throughput.json"),
        dict(
            device=jax.devices()[0].device_kind,
            post_physical_evaluation=post_experiment,
            host_load_1m_5m_15m=list(os.getloadavg()),
            stage=stage,
            seed=seed,
            cores=result,
            note="Warm synchronized float32 processor, state, eight candidate heads and risk scores. Excludes camera, audio encoder, simulation and feature packing; 40 GPU calls and 80 single-threaded NumPy calls.",
        ),
    )
    print("Measured field throughput", result, flush=True)


if __name__ == "__main__":
    import sys

    run("--post-experiment" in sys.argv)
