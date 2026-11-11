"""Select existing adapters using development-only criteria; preserve the fixed terminal comparison."""

import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from evals.g11_predict import infer
from scripts.train.g11_fit import parent, read_bank
from scripts.train.g9_fit import numpy_weights, save_json
from scripts.train.g11_analysis import estimate, SCORES, subgroup, metric, shift


def main(root, parents, runs, evaldir, out):
    out.mkdir(exist_ok=True, parents=True)
    banks = root / "banks"
    grades = {
        (x["recording"], x["id"]): x["grade"] == "D"
        for x in json.loads((evaldir / "device-audit.json").read_text())["rows"]
    }
    for fold in ("A", "B"):
        voc = json.loads((banks / (fold + "-vocabulary.json")).read_text())
        db = {r: read_bank(banks, fold, r, True) for r in ("held", "dev")}
        for role, (data, _) in db.items():
            meta = json.loads((banks / f"{fold}-{role}.json").read_text())
            fam = {v["recording"]: v["family"] for v in meta["rows"]}
            data["families"] = np.array([fam.get(n, "unused") for n in meta["records"]])
            data["strong_conflict"] = data["conflict"].astype(bool)
            data["device"] = np.array(
                [
                    grades.get((v["recording"], v["action_id"]), False)
                    for v in meta["rows"]
                ]
            )
        for seed in (1001, 1002, 1003):
            for core in ("hybrid", "gpt2", "gated"):
                _, p, s = parent(parents, core, seed)
                for mode in ("paired", "deranged", "absent"):
                    tag = f"g11-{fold}-{core}-{mode}-s{seed}"
                    result = out / (tag + ".json")
                    if result.exists():
                        continue
                    normal = dict(paired="correct", deranged="wrong", absent="none")[
                        mode
                    ]

                    def values(step, role, condition):
                        file = evaldir / "predictions" / (tag + f"-{step}.npz")
                        if file.exists():
                            with np.load(file) as f:
                                allowed = role + "|" + condition + "|"
                                if allowed + "verb" in f:
                                    return subgroup(f, allowed)
                        data, sem = db[role]
                        index = np.flatnonzero(data["cued"])
                        if step == 256:
                            data = {**data, "geometry": np.zeros_like(data["geometry"])}
                        delta = jax.tree.map(
                            jnp.asarray,
                            numpy_weights(
                                runs / tag / f"step-{step:05d}" / "weights.npz"
                            ),
                        )
                        return infer(
                            p,
                            delta,
                            s,
                            data,
                            sem,
                            len(voc["verb"]),
                            len(voc["noun"]),
                            condition,
                            index,
                        )

                    dev = db["dev"][0]
                    held = db["held"][0]
                    loss = {}
                    error = {}
                    for step in (256, 640, 1024):
                        pred = values(step, "dev", normal)
                        m = metric(dev, pred)
                        loss[step] = estimate(
                            m["verb_nll"], dev["group"], dev["families"], False
                        )["mean"]
                        if step > 256:
                            error[step] = estimate(
                                m["effect_mae"], dev["group"], dev["families"], False
                            )["mean"]
                    chosen = min(loss, key=lambda step: (loss[step], step))
                    physical = min(error, key=lambda step: (error[step], step))
                    states = {}
                    for cond in ("correct", "wrong", "none"):
                        pred = values(chosen, "held", cond)
                        if physical != chosen:
                            pred["effect"] = values(physical, "held", cond)["effect"]
                        states[cond] = metric(held, pred)
                    doc = dict(
                        tag=tag,
                        action_step=chosen,
                        effect_step=physical,
                        dev_nll=loss,
                        dev_effect_mae=error,
                        held={
                            k: {
                                z: estimate(
                                    v[z], held["group"], held["families"], False
                                )
                                for z in SCORES
                            }
                            for k, v in states.items()
                        },
                        correct_minus={
                            k: {
                                z: shift(states["correct"], states[k], held, z)
                                for z in ("verb", "exact", "effect_mae", "device_mae")
                            }
                            for k in ("wrong", "none")
                        },
                    )
                    save_json(result, doc)
            print(fold, seed, "development selections", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("data", type=Path)
    a.add_argument("parents", type=Path)
    a.add_argument("runs", type=Path)
    a.add_argument("eval", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--cache")
    p = a.parse_args()
    if p.cache:
        jax.config.update("jax_compilation_cache_dir", p.cache)
    main(p.data, p.parents, p.runs, p.eval, p.out)
