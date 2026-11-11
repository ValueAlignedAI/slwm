"""Run G11 matched interventions on the small/large G9-lineage adapters."""

import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g11_predict import infer, MODES
from scripts.train.g11_fit import parent, read_bank
from scripts.train.g9_fit import numpy_weights, save_json


def main(a):
    a.out.mkdir(exist_ok=True, parents=True)
    cache = {}
    for tag in a.tags:
        _, fold, size, core, treat, seed = tag.split("-")
        seed = int(seed[1:])
        dest = a.out / (tag + "-1024.json")
        if dest.exists():
            continue
        if fold not in cache:
            cache[fold] = (
                {
                    r: read_bank(a.banks, fold, r, True)
                    for r in ("held", "familiar", "dev")
                },
                json.loads((a.banks / (fold + "-vocabulary.json")).read_text()),
            )
        banks, vocab = cache[fold]
        stage = a.runs / tag / "step-01024"
        rec = json.loads((stage / "receipt.json").read_text())
        meta, p, s = parent(a.parents, core, seed, size)
        if meta["inference_sha256"] != rec["parent_sha256"]:
            raise ValueError("Scale parent disagreement")
        with (stage / "weights.npz").open("rb") as file:
            if (
                hashlib.file_digest(file, "sha256").hexdigest()
                != rec["inference_sha256"]
            ):
                raise ValueError("Invalid scale adapter")
        theta = jax.tree.map(jnp.asarray, numpy_weights(stage / "weights.npz"))
        outputs = {}
        counts = {}
        for role, (rows, sem) in banks.items():
            ix = np.flatnonzero(rows["cued"])
            counts[role] = len(ix)
            for mode in MODES:
                preds = infer(
                    p,
                    theta,
                    s,
                    rows,
                    sem,
                    len(vocab["verb"]),
                    len(vocab["noun"]),
                    mode,
                    ix,
                )
                for k, v in preds.items():
                    outputs[role + "|" + mode + "|" + k] = v
        result = dest.with_suffix(".npz")
        tmp = dest.with_suffix(".new.npz")
        np.savez_compressed(tmp, **outputs)
        tmp.replace(result)
        save_json(
            dest,
            dict(
                tag=tag,
                step=1024,
                parent_sha256=rec["parent_sha256"],
                delta_sha256=rec["inference_sha256"],
                output_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
                cued=counts,
                backend=jax.default_backend(),
            ),
        )
        print(tag, "evaluated", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("banks", type=Path)
    p.add_argument("parents", type=Path)
    p.add_argument("runs", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("tags", nargs="+")
    p.add_argument("--cache")
    a = p.parse_args()
    if a.cache:
        jax.config.update("jax_compilation_cache_dir", a.cache)
    main(a)
