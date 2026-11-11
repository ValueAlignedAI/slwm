"""Check whether speech content survives in the emitted field independently of predicting a device effect."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import re
import jax
import jax.numpy as jnp
import numpy as np
from models.g11_instruction import predict
from scripts.train.g9_fit import numpy_weights, save_json
from scripts.train.g11_fit import parent, read_bank

KW = {
    "grab": "grab grasp take pick",
    "place": "place put set",
    "insert": "insert plug slot",
    "withdraw": "remove unplug withdraw",
    "screw": "screw tighten fasten",
    "unscrew": "unscrew loosen",
    "rotate": "rotate twist turn",
    "open": "open",
    "close": "close shut",
    "press": "press click",
    "lift": "lift raise",
    "slide": "slide",
    "pull": "pull",
    "push": "push",
    "align": "align",
    "flip": "flip",
    "lock": "lock",
    "unlock": "unlock",
    "pour": "pour",
}


def targets(root, banks, fold, role, cat, actions):
    meta = json.loads((banks / f"{fold}-{role}.json").read_text())
    vocab = json.loads((banks / (fold + "-vocabulary.json")).read_text())["verb"]
    mapped = {v: i for i, v in enumerate(vocab)}
    rows = [r for r in meta["rows"] if r["cue"] >= 0]
    y1 = np.zeros((len(rows), len(KW)))
    y2 = np.zeros((len(rows), len(vocab)))
    y3 = np.zeros_like(y2)
    for i, r in enumerate(rows):
        n = r["recording"]
        source = cat[n]
        speech = next(s for s in source["speech"] if s["id"] == r["cue"])
        tokens = set(re.findall(r"[a-z0-9']+", speech["text"].casefold()))
        y1[i] = [bool(tokens & set(v.split())) for v in KW.values()]
        arr, by = actions[n]
        j = by[r["action_id"]]
        start = arr[j]["start"]
        for q, e in enumerate(arr[j : j + 3]):
            v = e["attributes"].get("Verb", "").casefold()
            if v in mapped and e["start"] <= start + 12:
                y3[i, mapped[v]] = 1
                if q == 0:
                    y2[i, mapped[v]] = 1
    return [y1, y2, y3]


@lru_cache(maxsize=24)
def kernel(s, mode):
    @jax.jit
    def run(p, d, r):
        return predict(p, d, s, r, mode)["state"]

    return run


def features(s, base, delta, rows, mode):
    keys = (
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
    )
    ix = np.flatnonzero(rows["cued"])
    parts = []
    for start in range(0, len(ix), 128):
        b = ix[start : start + 128]
        r = {k: jnp.asarray(rows[k][b]) for k in keys}
        parts.append(np.asarray(kernel(s, mode)(base, delta, r)))
    return np.concatenate(parts)


def ap(y, score):
    ix = np.argsort(-score, kind="stable")
    v = y[ix]
    ss = score[ix]
    end = np.r_[np.flatnonzero(np.diff(ss) != 0), len(v) - 1]
    tp = np.cumsum(v)[end]
    p = tp / (end + 1)
    return float(np.sum(p * np.diff(tp, prepend=0)) / max(1, tp[-1]))


def score(train, variants, y, held):
    keep = (y.sum(0) >= 20) & (held.sum(0) >= 5) & ((len(held) - held.sum(0)) >= 5)
    y = y[:, keep]
    held = held[:, keep]
    mu = train.mean(0)
    sd = np.maximum(train.std(0), 0.04)
    x = (train - mu) / sd
    beta = np.linalg.solve(
        x.T @ x + 0.025 * len(x) * np.eye(x.shape[1]), x.T @ (y - y.mean(0))
    )
    result = {"labels": int(keep.sum()), "prevalence": float(held.mean(0).mean())}
    for m, v in variants.items():
        z = (v - mu) / sd @ beta + y.mean(0)
        result[m] = float(
            np.mean([ap(held[:, k], z[:, k]) for k in range(held.shape[1])])
        )
    return result


def main(root, runs, parents, out):
    out.mkdir(exist_ok=True, parents=True)
    cat = json.loads((root / "annotations.json").read_text())
    actions = {}
    for r in json.loads(
        Path("/mnt/slwm-data/SLWM/g10/data/holo/labels.json").read_text()
    ):
        if r["video_name"] not in cat:
            continue
        v = sorted(
            [e for e in r["events"] if e["label"] == "Fine grained action"],
            key=lambda e: (e["start"], e["id"]),
        )
        actions[r["video_name"]] = (v, {e["id"]: i for i, e in enumerate(v)})
    for fold in ("A", "B"):
        banks = root / "banks"
        train, _ = read_bank(banks, fold, "train")
        held, _ = read_bank(banks, fold, "held")
        yt = targets(root, banks, fold, "train", cat, actions)
        yh = targets(root, banks, fold, "held", cat, actions)
        for seed in (1001, 1002, 1003):
            for c in ("hybrid", "gpt2", "gated"):
                tag = f"g11-{fold}-{c}-paired-s{seed}"
                file = out / (tag + ".json")
                if file.exists():
                    continue
                stage = runs / tag / "step-01024"
                r = json.loads((stage / "verified.json").read_text())
                meta, p, s = parent(parents, c, seed)
                if r["parent_sha256"] != meta["inference_sha256"]:
                    raise ValueError("Probe parent mismatch")
                d = jax.tree.map(jnp.asarray, numpy_weights(stage / "weights.npz"))
                x = features(s, p, d, train, "correct")
                variants = {
                    m: features(s, p, d, held, m)
                    for m in ("correct", "wrong", "none", "human", "small")
                }
                result = {
                    k: score(x, variants, y, h)
                    for k, y, h in zip(
                        ("instruction_words", "next_verb", "next_three_verbs"), yt, yh
                    )
                }
                save_json(
                    file, dict(tag=tag, sha256=r["inference_sha256"], probe=result)
                )
                print(tag, result["instruction_words"]["correct"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("runs", type=Path)
    p.add_argument("parents", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--cache")
    a = p.parse_args()
    if a.cache:
        jax.config.update("jax_compilation_cache_dir", a.cache)
    main(a.root, a.runs, a.parents, a.out)
