"""Reconstruct true, wrong and absent-cue utilities from separately executed physical branches."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from evals.g13_stats import Groups
from scripts.train.g9_storage import write

ROOT = Path("/mnt/slwm-data/SLWM/g14")
METRICS = (
    "success",
    "arrival",
    "terminal",
    "terminal_selective",
    "final_mm",
    "other_mm",
    "progress_mm",
)


def read(folder):
    return [json.loads(p.read_bytes()) for p in sorted(folder.glob("[0-9]*.json"))]


def trials(rows):
    output = defaultdict(list)
    for case in rows:
        for tag, branches in case["results"].items():
            if len({b["start"] for b in branches.values()}) != 1 or any(
                b["start"] != case["start"] for b in branches.values()
            ):
                raise ValueError("Physical state mismatch")
            conditions = {
                k: [] for k in ("actual", "wrong_color", "wrong_address", "both_wrong")
            }
            for i in range(4):
                for name, offset in (
                    ("actual", 0),
                    ("wrong_color", 2),
                    ("wrong_address", 1),
                    ("both_wrong", 3),
                ):
                    b = (
                        branches[str(i ^ offset)]
                        if len(branches) > 1
                        else branches["0"]
                    )
                    conditions[name].append(b["scores"][i])
            output[tag].append(
                dict(
                    seed=case["seed"],
                    speaker=case["speaker"],
                    shift=case["shift"],
                    occurrences=len(branches),
                    conditions={
                        n: {m: float(np.mean([r[m] for r in x])) for m in METRICS}
                        for n, x in conditions.items()
                    },
                    effort=float(np.mean([b["effort_m"] for b in branches.values()])),
                    noop=float(
                        np.mean(
                            [
                                b["chosen"][0] / sum(b["chosen"])
                                for b in branches.values()
                            ]
                        )
                    ),
                )
            )
    return output


def summary(rows, stratify=False):
    a = trials(rows)
    results = {}
    for tag, vals in a.items():
        g = [r["seed"] for r in vals]
        fam = [r["speaker"] for r in vals]
        bs = Groups(
            g,
            [f"{s}/{int(v['shift'])}" for s, v in zip(fam, vals)] if stratify else fam,
        )

        def metric(n, k):
            return np.array([r["conditions"][n][k] for r in vals])

        results[tag] = dict(
            absolute={k: bs.score(metric("actual", k)) for k in METRICS},
            correct_over={
                n: {
                    k: bs.score(
                        (metric("actual", k) - metric(n, k))
                        * (-1 if k in ("final_mm", "other_mm") else 1)
                    )
                    for k in ("success", "arrival", "final_mm")
                }
                for n in ("wrong_address", "wrong_color", "both_wrong")
            },
            by_condition={
                reg: {
                    k: bs.score(
                        np.where(
                            np.array([r["shift"] for r in vals]) == shift,
                            metric("actual", k),
                            np.nan,
                        )
                    )
                    for k in ("success", "arrival", "final_mm")
                }
                for reg, shift in (("nominal", False), ("shift", True))
            },
            by_speaker={
                str(s): float(np.mean(metric("actual", "success")[np.array(fam) == s]))
                for s in sorted(set(fam))
            },
            effort=bs.score([v["effort"] for v in vals]),
            noop=bs.score([v["noop"] for v in vals]),
        )
    return a, results


def routing():
    folder = ROOT / "work/executed/pilot/dev"
    d = json.loads((folder / "complete.json").read_bytes())
    rows = read(folder)
    if len(rows) != d["cases"]:
        raise ValueError("Incomplete pilot")
    a, s = summary(rows)
    measures = {}
    for route in ("directpath", "routed"):
        keys = [
            k
            for k in s
            if f"/{route}/" in k
            and (
                "/geometry/" in k
                or "/transport/protected/1411/" in k
                or "/gpt2/protected/1411/" in k
            )
        ]
        measures[route] = dict(
            success=float(
                np.mean([s[k]["absolute"]["success"]["value"] for k in keys])
            ),
            mm=float(np.mean([s[k]["absolute"]["final_mm"]["value"] for k in keys])),
            included=keys,
        )
    selected = min(measures, key=lambda k: (-measures[k]["success"], measures[k]["mm"]))
    write(
        ROOT / "work/routing-lock.json",
        dict(routed=selected == "routed", scores=measures, starts=len(rows)),
    )
    write(ROOT / "work/pilot-physical.json", s)
    print("Executed development routing", selected, measures, flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("stage", choices=("routing",))
    a.parse_args()
    routing()
