"""Audit all terminal branches and record source-conditioned forecasting and frozen risk artifacts."""

import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from g9_storage import write, sha
from g10_store import roots
from g10_analysis import context, effect, read
from g10_real_analysis import analyze as grounding, nonmissing
from g10_cue_analysis import analyze as commands


def run(evaldir, out):
    out.mkdir(parents=True, exist_ok=True)
    a, b = roots()
    c = context(evaldir, out)
    x = commands(evaldir, out)
    g = grounding(evaldir, out)
    expected = set()
    for file in (
        "context-artifacts.json",
        "instruction-artifacts.json",
        "grounding-artifacts.json",
    ):
        expected.update(read(out / file))
    actual = {
        f.parent.parent.name + "/" + str(int(f.parent.name.split("-")[1]))
        for f in a.glob("*/step-*/verified.json")
    }
    if expected != actual or len(expected) != 90:
        raise ValueError("Immutable G10 factorial inventory mismatch")
    frows = {}
    for seed in (1001, 1002, 1003):
        for core in ("hybrid", "gpt2"):
            for mode in ("paired", "reconstruct"):
                tag = f"align-{core}-{mode}-s{seed}"
                doc = read(evaldir / "forecast" / (tag + ".json"))
                if doc["sha256"] != g["rows"][tag]["sha256"]:
                    raise ValueError("Forecast used a different state")
                frows[tag] = {}
                for role, panel in doc["panels"].items():
                    errors = panel["rmse"]
                    frows[tag][role] = dict(
                        mean={
                            k: {
                                t: float(np.nanmean(nonmissing(v)))
                                for t, v in group.items()
                            }
                            for k, group in errors.items()
                        },
                        persistence_minus_all={
                            t: effect(
                                nonmissing(errors["persistence"][t]),
                                nonmissing(errors["all"][t]),
                            )
                            for t in errors["all"]
                        },
                        prior_minus_hidden={
                            t: effect(
                                nonmissing(errors["prior"][t]),
                                nonmissing(errors["hidden_imu"][t]),
                            )
                            for t in errors["all"]
                        },
                    )
    write(
        out / "forecasts.json",
        dict(
            horizon_seconds=0.1,
            units="Restored accelerometer and gyroscope component units; acoustic features use normalized log bands",
            rows=frows,
        ),
    )
    raw = {}
    risk_heads = {}
    for tag, row in g["rows"].items():
        f = evaldir / "real" / (tag + "-risk-heads.npz")
        digest = sha(f)
        if digest != row["readout"]["head_sha256"]:
            raise ValueError("Fitted readout identity changed")
        for volume in (a.parent, b.parent):
            target = volume / "risk-heads" / f.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha(target) != digest:
                shutil.copyfile(f, target)
            if sha(target) != digest:
                raise ValueError("Readout failed the second custody audit")
        risk_heads[f.name] = dict(
            sha256=digest, bytes=f.stat().st_size, parent_sha256=row["sha256"]
        )
        for role in ("familiar", "novel"):
            with np.load(
                evaldir / "real" / (tag + "-" + role + "-risk.npz"), allow_pickle=False
            ) as src:
                for key in src.files:
                    shared = key in (
                        "y",
                        "group",
                        "names",
                        "raw",
                        "prevalence",
                        "task_prior",
                    )
                    name = (role if shared else tag + "|" + role) + "|" + key
                    if name in raw and not np.array_equal(raw[name], src[key]):
                        raise ValueError("Mismatched outcome panel or baseline")
                    raw[name] = src[key]
    for volume in (a.parent, b.parent):
        write(volume / "risk-heads/verified.json", risk_heads)
    write(out / "risk-readouts.json", risk_heads)
    np.savez_compressed(out / "risk-predictions.npz", **raw)
    data = read(out / "data.json")
    total = sum(v["recordings"] for v in data["holo"].values())
    if total != 416 or len(read(a.parent / "inputs/verified.json")) != 15:
        raise ValueError("Source inventory incomplete")
    checked = dict(
        context_fits=c["fits"],
        alignment_fits=g["fits"],
        instruction_fits=len(x["adapted"]),
        total_fits=c["fits"] + g["fits"] + len(x["adapted"]),
        verified_generations=len(expected),
        recordings=total,
        held_action_heads=len(risk_heads),
        inference_packages=len(read(a.parent / "public-inference/verified.json")) - 1,
        physical_input_objects=len(read(a.parent / "inputs/verified.json")),
    )
    write(out / "scientific-verdict.json", checked)
    print(json.dumps(checked), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eval", type=Path)
    ap.add_argument("out", type=Path)
    p = ap.parse_args()
    run(p.eval, p.out)
