"""Pair frozen G10 scores by mission or recording and retain unsigned failures."""

import argparse
import json
from pathlib import Path
import numpy as np
from g9_storage import write
from g10_store import roots

SEEDS = (1001, 1002, 1003)
CORES = ("hybrid", "gpt2", "gated")
REGIMES = ("short", "variable", "long")


def effect(left, right, seed=7310):
    l = np.asarray(left, np.float64)
    r = np.asarray(right, np.float64)
    d = l - r
    d = d[np.isfinite(d)]
    if not len(d):
        return None
    draws = np.random.default_rng(seed).integers(0, len(d), (4000, len(d)))
    lo, hi = np.quantile(d[draws].mean(1), [0.025, 0.975])
    return dict(mean=float(d.mean()), ci95=[float(lo), float(hi)], units=len(d))


def read(path):
    return json.loads(path.read_bytes())


def custody(tag, step):
    a, b = roots()
    rel = Path(tag) / f"step-{step:05d}"
    copies = [read(x / rel / "verified.json") for x in (a, b)]
    if copies[0] != copies[1] or copies[0]["copies"] != 2:
        raise ValueError("Generation custody mismatch")
    rec = read(a / rel / "receipt.json")
    if any(
        rec[k] != copies[0][k]
        for k in ("tag", "step", "inference_sha256", "optimizer_sha256")
    ):
        raise ValueError("Receipt mismatch")
    return {
        k: rec[k]
        for k in (
            "tag",
            "step",
            "inference_sha256",
            "optimizer_sha256",
            "trainable_parameters",
            "backend",
            "wall_this_process_s",
            "started_from",
        )
    }


def context(evaldir, out):
    out.mkdir(exist_ok=True, parents=True)
    rows = {}
    art = {}
    score = {}
    raw = {}
    for seed in SEEDS:
        for core in CORES:
            for regime in REGIMES:
                tag = f"context-{core}-{regime}-s{seed}"
                for step in (2560, 5120):
                    key = tag + "/" + str(step)
                    art[key] = custody(tag, step)
                    d = read(evaldir / "control" / f"{tag}-{step}.json")
                    if art[key]["inference_sha256"] != d["sha256"]:
                        raise ValueError("Wrong decision weights")
                    row = dict(backend=d["backend"], sha256=d["sha256"])
                    for cond in (
                        "development",
                        "full",
                        "block32",
                        "carry",
                        "wrong",
                        "changes",
                    ):
                        if cond in d:
                            row[cond] = {
                                k: float(np.mean(v))
                                for k, v in d[cond].items()
                                if k != "first_actions" and len(v)
                            }
                            for m in (
                                "occupancy",
                                "subgoals",
                                "tap",
                                "post_change_occupancy",
                                "post_change_probe",
                            ):
                                if m in d[cond] and d[cond][m]:
                                    raw[key + "|" + cond + "|" + m] = np.asarray(
                                        d[cond][m], "f4"
                                    )
                    if step == 5120:
                        score[tag] = d
                        row["delays"] = {
                            str(l): float(np.mean(a))
                            for l, a in zip(
                                d["delayed"]["delays"], d["delayed"]["accuracy"]
                            )
                        }
                        raw[key + "|delayed"] = np.asarray(
                            d["delayed"]["accuracy"], "f4"
                        )
                    rows[key] = row
    comparisons = {}
    for seed in SEEDS:

        def z(core, regime):
            return score[f"context-{core}-{regime}-s{seed}"]

        for label, a, b in (
            ("hybrid_minus_dense", z("hybrid", "variable"), z("gpt2", "variable")),
            ("gated_minus_hybrid", z("gated", "variable"), z("hybrid", "variable")),
            ("dense_variable_minus_short", z("gpt2", "variable"), z("gpt2", "short")),
            ("dense_variable_minus_long", z("gpt2", "variable"), z("gpt2", "long")),
        ):
            comparisons.setdefault(label, {})[str(seed)] = {
                cond: effect(a[cond]["occupancy"], b[cond]["occupancy"])
                for cond in ("full", "changes", "block32", "carry")
            }
    ref = read(evaldir / "reference.json")
    doc = dict(
        study="G10",
        fits=27,
        verified_generations=len(art),
        positions_per_fit=5120 * 8192,
        simulator_starts_per_fit=5120 * 12,
        seeds=SEEDS,
        rows=rows,
        comparisons=comparisons,
        reference={
            k: {
                j: float(np.mean(v))
                for j, v in r.items()
                if j != "first_actions" and len(v)
            }
            for k, r in ref.items()
        },
        positive_hybrid_seeds=sum(
            x["full"]["ci95"][0] > 0 for x in comparisons["hybrid_minus_dense"].values()
        ),
    )
    write(out / "context.json", doc)
    write(out / "context-artifacts.json", art)
    np.savez_compressed(out / "context-paired.npz", **raw)
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("evaldir", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    print("context gate", context(a.evaldir, a.out)["positive_hybrid_seeds"], "of 3")
