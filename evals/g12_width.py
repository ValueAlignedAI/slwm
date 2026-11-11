"""Pair equal-exposure processor widths on identical fresh cases and identical directive exchanges."""

import argparse
import json
from pathlib import Path
import numpy as np
from evals.g12_compare import RecordingBootstrap, scoring, SOFT
from scripts.train.g12_fit import ROOT, read
from scripts.train.g9_storage import write
from models.g12_world import CORES


def main(out):
    data = ROOT / "data"
    labels = {
        (r["recording"], r["id"]): r["label"]
        for r in json.loads((data / "state-labels.json").read_bytes())["items"]
    }
    results = {}
    public = {}
    counts = {}
    for fold in ("A", "B"):
        for role in ("confirm", "replica"):
            meta, row = read(data, fold, role, labels)
            bs = RecordingBootstrap(meta["rows"])
            for seed in (1201, 1202):
                for core in CORES:
                    tag = f"{fold}-{core}-paired-{seed}"
                    l = ROOT / "scale-eval" / tag
                    sm = ROOT / "eval" / tag
                    dense = ROOT / "scale-eval" / f"{fold}-gpt2-paired-{seed}"
                    d = json.loads((l / "receipt.json").read_bytes())
                    s = json.loads((sm / "receipt.json").read_bytes())
                    if (
                        d["config"]["spec"]["width"] != 256
                        or s["config"]["spec"]["width"] != 128
                        or d["config"]["train_sha"] != s["config"]["train_sha"]
                        or d["config"]["state_sha"] != s["config"]["state_sha"]
                    ):
                        raise ValueError(
                            "Width comparison requires identical independent inputs and supervision"
                        )
                    result = results.setdefault(
                        tag,
                        dict(
                            parameters=d["config"]["parameters"],
                            selected=d["selected"],
                            weights_sha256=d["weights_sha256"],
                            device_kind=d["config"]["device_kind"],
                            curve=d["curve"],
                            by_role={},
                        ),
                    )
                    with (
                        np.load(l / (role + ".npz")) as a,
                        np.load(sm / (role + ".npz")) as b,
                        np.load(dense / (role + ".npz")) as ctrl,
                    ):

                        def measure(file, mode):
                            return scoring(
                                file[mode + "_verb"], file[mode + "_state"], row
                            )

                        true, old, reference = (
                            measure(a, "correct"),
                            measure(b, "correct"),
                            measure(ctrl, "correct"),
                        )
                        wrong, blank, oldwrong = (
                            measure(a, "wrong"),
                            measure(a, "none"),
                            measure(b, "wrong"),
                        )
                        result["by_role"][role] = dict(
                            absolute={m: bs.estimates(true[m]) for m in SOFT},
                            width_gain={m: bs.gain(true[m], old[m], m) for m in SOFT},
                            against_dense={
                                m: bs.gain(true[m], reference[m], m) for m in SOFT
                            },
                            vs_wrong={m: bs.gain(true[m], wrong[m], m) for m in SOFT},
                            vs_none={m: bs.gain(true[m], blank[m], m) for m in SOFT},
                            change_in_speech_benefit={
                                m: bs.gain(true[m] - wrong[m], old[m] - oldwrong[m], m)
                                for m in SOFT
                            },
                        )
                        for mode in ("correct", "wrong", "none"):
                            prob = a[mode + "_verb"]
                            prefix = tag + "_" + role + "_" + mode
                            public[prefix + "_top"] = prob.argmax(-1).astype("i2")
                            public[prefix + "_py"] = np.where(
                                row["y"] >= 0,
                                np.take_along_axis(
                                    prob, np.maximum(row["y"], 0)[:, :, None], axis=2
                                )[:, :, 0],
                                1e-7,
                            )
                            public[prefix + "_state"] = a[mode + "_state"]
    for core in CORES:
        for endpoint in ("ordered", "brier"):
            gate = {}
            for f in ("A", "B"):
                rows = [
                    results[f"{f}-{core}-paired-{seed}"]["by_role"]["confirm"]
                    for seed in (1201, 1202)
                ]
                gate[f] = {
                    "beats_dense": (
                        sum(d["against_dense"][endpoint]["lo"] > 0 for d in rows)
                        if core != "gpt2"
                        else None
                    ),
                    "speech_pass": sum(
                        d["vs_wrong"][endpoint]["lo"] > 0
                        and d["vs_none"][endpoint]["lo"] > 0
                        for d in rows
                    ),
                }
            counts[core + "-" + endpoint] = gate
    np.savez_compressed(out / "width-predictions.npz", **public)
    write(out / "width.json", dict(runs=results, fresh_gates=counts))
    print("Matched width comparisons", len(results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    main(p.parse_args().out)
