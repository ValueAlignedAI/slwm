"""Replay four prospectively fixed held starts and publish the native poses behind their scores."""

from pathlib import Path
import json
import numpy as np
from scripts.data.g14_collect import clean_worker
from scripts.train.g14_execute import run_one, tag, score
from scripts.train.g9_storage import write, sha

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def main():
    lock = json.loads((ROOT / "work/final-control-lock.json").read_bytes())
    variant = json.loads((ROOT / "work/geometry-choice.json").read_bytes())["variant"]
    ps = [
        p
        for p in lock["policies"]
        if p["core"] in ("geometry", "gpt2", "transport")
        and p.get("seed", 1411) == 1411
        and not any(
            p.get(k)
            for k in (
                "oracle_cue",
                "reset_memory",
                "original_edge",
                "open_gate",
                "mean_field",
            )
        )
    ]
    if len(ps) != 3:
        raise ValueError("Wrong audit panel")
    folder = ROOT / "work/audit-traces"
    folder.mkdir(exist_ok=True)
    records = []
    traces = []
    eff = []
    commands = []
    initial = []
    assigned = []
    dest = []
    metric = []
    for cohort, indices in (("held", (0, 128)), ("fresh", (0, 32))):
        cases = json.loads((ROOT / "data/cases" / (cohort + ".json")).read_bytes())
        for i in indices:
            case = cases[i]
            table = clean_worker()
            sig = table.reset(case["seed"], case["shift"])
            stations = table.stations(case["seed"])
            original = json.loads(
                (
                    ROOT
                    / "work/executed/final"
                    / cohort
                    / (str(case["seed"]) + ".json")
                ).read_bytes()
            )
            if original["start"] != sig:
                raise ValueError("Canonical replay root changed")
            for p in ps:
                for j in range(4):
                    row, raw = run_one(table, case, j, stations, p, variant, True)
                    actual = original["results"][tag(p)][str(j)]
                    x = case["commands"][j]
                    measured = score(
                        raw[0],
                        np.array(row["initial"]),
                        x["true_color"],
                        stations[x["true_station"]],
                    )
                    endpoint = float(
                        1000
                        * np.linalg.norm(
                            np.asarray(row["terminal"]) - actual["terminal"], axis=1
                        ).max()
                    )
                    records.append(
                        dict(
                            world=case["seed"],
                            cohort=cohort,
                            policy=tag(p),
                            branch=j,
                            identical_digest=row["trajectory_sha"]
                            == actual["trajectory_sha"],
                            max_object_endpoint_replay_mm=endpoint,
                            same_selective_score=measured["success"]
                            == actual["scores"][j]["success"],
                        )
                    )
                    traces.append(raw[0])
                    eff.append(raw[1])
                    commands.append(raw[2])
                    initial.append(row["initial"])
                    assigned.append(x["true_color"])
                    dest.append(stations[x["true_station"]])
                    metric.append(actual["scores"][j]["success"])
    file = folder / "native-traces.npz"
    np.savez_compressed(
        file,
        objects=traces,
        effector=eff,
        commands=commands,
        initial=initial,
        color=assigned,
        target=dest,
        original_score=metric,
        seed=[r["world"] for r in records],
        policy=[r["policy"] for r in records],
        branch=[r["branch"] for r in records],
    )
    write(
        folder / "audit.json",
        dict(
            replays=len(records),
            identical_digests=sum(r["identical_digest"] for r in records),
            same_selective_scores=sum(r["same_selective_score"] for r in records),
            max_endpoint_replay_mm=max(
                r["max_object_endpoint_replay_mm"] for r in records
            ),
            trace_sha256=sha(file),
            records=records,
        ),
    )
    print(
        "Independent physical replay",
        sum(r["same_selective_score"] for r in records),
        len(records),
        max(r["max_object_endpoint_replay_mm"] for r in records),
        flush=True,
    )


if __name__ == "__main__":
    main()
