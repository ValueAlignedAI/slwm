"""Report development-selected early checkpoints without pooling seeds or independent device panels."""

import argparse
import json
from pathlib import Path
import numpy as np
from models.g12_world import CORES
from scripts.train.g12_fit import ROOT, SEEDS, read
from scripts.train.g9_storage import write
from evals.g12_compare import ROLES, SOFT, RecordingBootstrap, scoring


def main(out):
    data = ROOT / "data"
    dest = ROOT / "early-eval"
    results = {}
    gates = {}
    public = {}
    labels = {
        (r["recording"], r["id"]): r["label"]
        for r in json.loads((data / "state-labels.json").read_bytes())["items"]
    }
    for fold in ("A", "B"):
        for role in ROLES:
            meta, rows = read(data, fold, role, labels)
            bs = RecordingBootstrap(meta["rows"])
            for seed in SEEDS:
                for core in CORES:
                    for arm in ("full", "dropout"):
                        tag = f"{fold}-{core}-{arm}-{seed}"
                        folder = dest / tag
                        rec = json.loads((folder / "receipt.json").read_bytes())
                        d = results.setdefault(
                            tag,
                            dict(
                                selected=rec["selected"],
                                parameters=rec["config"]["parameters"],
                                curve=rec["curve"],
                                weights_sha256=rec["weights_sha256"],
                                thresholds=rec["thresholds"],
                                by_role={},
                            ),
                        )
                        full = dest / f"{fold}-{core}-full-{seed}"
                        dense = dest / f"{fold}-gpt2-{arm}-{seed}"
                        original = ROOT / "eval" / f"{fold}-{core}-paired-{seed}"
                        sources = {
                            "dense": dense,
                            "full_input_training": full,
                            "original_schedule": original,
                        }
                        with np.load(folder / (role + ".npz")) as z:

                            def scores(f, m):
                                return scoring(f[m + "_verb"], f[m + "_state"], rows)

                            a = scores(z, "correct")
                            wrong = scores(z, "wrong")
                            none = scores(z, "none")
                            entry = dict(
                                absolute={m: bs.estimates(a[m]) for m in SOFT},
                                vs_wrong={m: bs.gain(a[m], wrong[m], m) for m in SOFT},
                                vs_none={m: bs.gain(a[m], none[m], m) for m in SOFT},
                                vs_no_change=bs.gain(
                                    a["brier"],
                                    np.where(
                                        rows["state"] >= 0, rows["state"] ** 2, np.nan
                                    ),
                                    "brier",
                                ),
                            )
                            for label, source in sources.items():
                                with np.load(source / (role + ".npz")) as f:
                                    b = scores(f, "correct")
                                    entry["vs_" + label] = {
                                        m: bs.gain(a[m], b[m], m) for m in SOFT
                                    }
                            confidence = z["correct_confidence"] >= np.asarray(
                                rec["thresholds"]
                            )
                            entry["policy"] = dict(
                                action_coverage=bs.estimates(confidence[:, 0]),
                                state_coverage=bs.estimates(confidence[:, 1]),
                                internal_only=bs.estimates(~confidence.any(1)),
                                conditional_action=bs.estimates(
                                    np.where(confidence[:, 0], a["ordered"], np.nan)
                                ),
                                conditional_state=bs.estimates(
                                    np.where(
                                        confidence[:, 1], a["state_accuracy"], np.nan
                                    )
                                ),
                            )
                            d["by_role"][role] = entry
                            for mode in ("correct", "wrong", "none"):
                                prob = z[mode + "_verb"]
                                prefix = tag + "_" + role + "_" + mode
                                public[prefix + "_top"] = prob.argmax(-1).astype("i2")
                                public[prefix + "_py"] = np.where(
                                    rows["y"] >= 0,
                                    np.take_along_axis(
                                        prob, np.maximum(rows["y"], 0)[:, :, None], 2
                                    )[:, :, 0],
                                    1e-7,
                                )
                                public[prefix + "_state"] = z[mode + "_state"]
    for arm in ("full", "dropout"):
        for core in CORES:
            for m in ("ordered", "brier"):
                info = {}
                for fold in ("A", "B"):
                    cells = [
                        results[f"{fold}-{core}-{arm}-{s}"]["by_role"]["confirm"]
                        for s in SEEDS
                    ]
                    info[fold] = dict(
                        speech=sum(
                            x["vs_wrong"][m]["lo"] > 0 and x["vs_none"][m]["lo"] > 0
                            for x in cells
                        ),
                        beats_dense=(
                            sum(x["vs_dense"][m]["lo"] > 0 for x in cells)
                            if core != "gpt2"
                            else None
                        ),
                        improves_on_original=sum(
                            x["vs_original_schedule"][m]["lo"] > 0 for x in cells
                        ),
                    )
                gates[arm + "-" + core + "-" + m] = info
    np.savez_compressed(out / "early-predictions.npz", **public)
    write(out / "early.json", dict(runs=results, fresh_gates=gates))
    print("Completed early comparison", len(results), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    main(p.parse_args().out)
