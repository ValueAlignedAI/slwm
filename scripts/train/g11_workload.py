"""Measure the actual G11 optimizer workload on a fixed sample of public training decisions."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_shared import Spec, parameters
from models.g11_instruction import initialize
from scripts.train.g11_fit import read_bank, kernel
from scripts.train.g9_fit import save_json

KEYS = (
    "a",
    "v",
    "n",
    "ac",
    "wc",
    "aw",
    "ww",
    "ah",
    "wh",
    "small_words",
    "wrong_small",
    "geometry",
    "effect",
    "valid",
    "verb",
    "noun",
    "open_noun",
    "cued",
)


def prepare(banks, file):
    d, sem = read_bank(banks, "A", "train", True)
    rng = np.random.default_rng(100196)
    ix = np.unique(
        np.r_[
            rng.choice(np.flatnonzero(d["cued"]), 512, replace=False),
            rng.choice(np.flatnonzero(~d["cued"]), 384, replace=False),
            rng.choice(
                np.flatnonzero(d["valid"]), min(192, d["valid"].sum()), replace=False
            ),
        ]
    )
    meta = json.loads((banks / "A-vocabulary.json").read_text())
    np.savez_compressed(
        file,
        **{k: d[k][ix] for k in KEYS},
        sem=sem,
        nv=len(meta["verb"]),
        nn=len(meta["noun"]),
    )
    print(
        file,
        file.stat().st_size,
        hashlib.sha256(file.read_bytes()).hexdigest(),
        flush=True,
    )


def run(file, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        device=jax.devices()[0].device_kind,
        backend=jax.default_backend(),
        jax=jax.__version__,
        data_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
        cases={},
    )
    with np.load(file) as f:
        d = {k: jax.device_put(f[k]) for k in KEYS}
        sem = jnp.asarray(f["sem"])
        nv = int(f["nv"])
        nn = int(f["nn"])
    ci, bg, gi = [
        jnp.asarray(np.flatnonzero(np.asarray(x)).astype("i4"))
        for x in (d["cued"], ~d["cued"], d["valid"])
    ]
    scale = jnp.maximum(jnp.std(d["effect"][gi], axis=0), 0.065)
    for family, width, depth in (("small", 512, 4), ("large", 896, 12)):
        for c in ("gpt2", "hybrid"):
            spec = Spec(c, width, depth, 128, "gelu")
            base = parameters(spec, 911)
            for batch in (128, 256):
                p = initialize(spec, 911)
                m = jax.tree.map(jnp.zeros_like, p)
                v = jax.tree.map(jnp.zeros_like, p)
                f = kernel(spec, "correct", nv, nn, True, batch)
                start = time.monotonic()
                for i in range(8):
                    p, m, v, metric = f(
                        p,
                        m,
                        v,
                        base,
                        d,
                        ci,
                        bg,
                        gi,
                        sem,
                        scale,
                        jnp.int32(256 + i),
                        jnp.int32(911),
                    )
                metric.block_until_ready()
                warm = time.monotonic() - start
                dur = []
                for block in range(4):
                    t = time.monotonic()
                    for j in range(16):
                        p, m, v, metric = f(
                            p,
                            m,
                            v,
                            base,
                            d,
                            ci,
                            bg,
                            gi,
                            sem,
                            scale,
                            jnp.int32(264 + 16 * block + j),
                            jnp.int32(911),
                        )
                    metric.block_until_ready()
                    dur.append(time.monotonic() - t)
                if not np.isfinite(np.asarray(metric)).all():
                    raise FloatingPointError("Invalid timed workload")
                name = f"{family}-{c}-b{batch}"
                result["cases"][name] = dict(
                    batch=batch,
                    updates=64,
                    history=32,
                    warmup_s=warm,
                    block_seconds=dur,
                    positions_per_second=64 * batch * 32 / sum(dur),
                    updates_per_second=64 / sum(dur),
                    base_parameters=sum(x.size for x in base.values()),
                    trainable_parameters=sum(x.size for x in p.values()),
                )
                save_json(dest, result)
                print(
                    name,
                    round(result["cases"][name]["positions_per_second"]),
                    flush=True,
                )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=("pack", "run"))
    ap.add_argument("source", type=Path)
    ap.add_argument("dest", type=Path)
    a = ap.parse_args()
    prepare(a.source, a.dest) if a.op == "pack" else run(a.source, a.dest)
