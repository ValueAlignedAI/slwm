"""Measure fit on a deterministic selection of optimization decisions and compare the same interventions."""

import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g11_predict import infer
from scripts.train.g11_fit import parent, read_bank
from scripts.train.g9_fit import numpy_weights, save_json


def main(root, parents, runs, out):
    b = root / "banks"
    out.mkdir(exist_ok=True, parents=True)
    for fold in ("A", "B"):
        bank, sem = read_bank(b, fold, "train", True)
        meta = json.loads((b / (fold + "-train.json")).read_text())
        vocab = json.loads((b / (fold + "-vocabulary.json")).read_text())
        candidates = np.flatnonzero(bank["cued"])
        key = lambda i: hashlib.sha256(
            (
                "G11/train-gap/"
                + meta["rows"][i]["recording"]
                + "/"
                + str(meta["rows"][i]["action_id"])
            ).encode()
        ).hexdigest()
        ids = np.asarray(sorted(candidates, key=key)[:512], int)
        for c in ("hybrid", "gpt2", "gated"):
            for seed in (1001, 1002, 1003):
                tag = f"g11-{fold}-{c}-paired-s{seed}"
                dest = out / (tag + ".json")
                if dest.exists():
                    continue
                pm, p, s = parent(parents, c, seed)
                rec = {
                    "tag": tag,
                    "rows": len(ids),
                    "parent_sha256": pm["inference_sha256"],
                    "results": {},
                }
                for step in (256, 1024):
                    stage = runs / tag / f"step-{step:05d}"
                    r = json.loads((stage / "verified.json").read_text())
                    d = jax.tree.map(jnp.asarray, numpy_weights(stage / "weights.npz"))
                    rows = (
                        bank
                        if step == 1024
                        else {**bank, "geometry": np.zeros_like(bank["geometry"])}
                    )
                    rec["results"][str(step)] = {"delta_sha256": r["inference_sha256"]}
                    for mode in ("correct", "wrong", "none"):
                        pred = infer(
                            p,
                            d,
                            s,
                            rows,
                            sem,
                            len(vocab["verb"]),
                            len(vocab["noun"]),
                            mode,
                            ids,
                        )
                        rec["results"][str(step)][mode] = dict(
                            accuracy=float(np.mean(pred["verb"] == rows["verb"][ids])),
                            nll=float(np.mean(pred["verb_nll"])),
                        )
                save_json(dest, rec)
                print(tag, rec["results"]["1024"]["correct"]["accuracy"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("parents", type=Path)
    p.add_argument("runs", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--cache")
    a = p.parse_args()
    if a.cache:
        jax.config.update("jax_compilation_cache_dir", a.cache)
    main(a.data, a.parents, a.runs, a.out)
