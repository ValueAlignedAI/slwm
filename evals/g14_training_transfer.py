"""Check the chosen training regime against one-step parents on the registered, balanced held subset."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from data.g14_fork import isolated
from data.g14_table import ROOT
from data.g14_policy import loaded
from scripts.data.g14_collect import clean_worker
from scripts.train.g14_execute import run_one, tag
from scripts.train.g14_stages import await_files
from evals.g13_stats import Groups
from evals.g14_trials import read, trials
from scripts.train.g9_storage import write


def pick(role):
    seq = json.loads((ROOT / "data/cases" / (role + ".json")).read_bytes())
    take = 4 if role == "held" else 8
    count = Counter()
    out = []
    for x in seq:
        v = x["speaker"], x["shift"]
        if count[v] < take:
            out.append(x)
            count[v] += 1
    return out


def one(args):
    cohort, case, lock = args
    directory = ROOT / "work/training-transfer" / cohort
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / (str(case["seed"]) + ".json")
    if file.is_file():
        return json.loads(file.read_bytes())
    t = clean_worker()
    seed = case["seed"]
    sig = t.reset(seed, case["shift"])
    stations = t.stations(seed)
    values = {}
    for c in ("gpt2", "transport"):
        ps = dict(
            core=c,
            stage="initial",
            arm="protected",
            seed=1411,
            routed=lock["routed"],
            contact_only=lock["contact_only"],
        )
        loaded("initial", c, "protected", 1411)
        parts = [isolated(run_one, t, case, k, stations, ps, 0)[0] for k in range(4)]
        if any(v["start"] != sig for v in parts):
            raise ValueError("Original parent world diverged")
        values[c] = {
            m: np.mean([v["scores"][k][m] for k, v in enumerate(parts)]).item()
            for m in ("success", "arrival", "final_mm")
        }
    row = dict(
        seed=seed,
        start=sig,
        speaker=case["speaker"],
        shift=case["shift"],
        original=values,
    )
    write(file, row)
    return row


def development():
    rows = read(ROOT / "work/executed/verified/dev")
    bank = trials(rows)
    bs = Groups([r["seed"] for r in rows], [r["speaker"] for r in rows])
    v = {}
    for stage in ("initial", "expert-isolated", "visited-isolated"):
        vals = []
        for c in ("gpt2", "transport"):
            for s in (1411, 1412, 1413):
                k = tag(
                    dict(
                        core=c,
                        arm="protected",
                        stage=stage,
                        routed=False,
                        seed=s,
                        oracle_cue=True,
                    )
                )
                vals.append([x["conditions"]["actual"]["success"] for x in bank[k]])
        v[stage] = np.mean(vals, axis=0)
    return {
        name: bs.score(v[a] - v[b])
        for name, a, b in (
            ("teacher_over_one_step", "expert-isolated", "initial"),
            ("learner_over_one_step", "visited-isolated", "initial"),
            ("learner_over_teacher", "visited-isolated", "expert-isolated"),
        )
    }


def main():
    await_files([ROOT / "work/released-lock.json"])
    lock = json.loads((ROOT / "work/final-control-lock.json").read_bytes())
    out = {}
    for role in ("fresh", "held"):
        cases = pick(role)
        with ProcessPoolExecutor(2) as p:
            rows = list(p.map(one, [(role, r, lock) for r in cases]))
        paths = [
            ROOT / "work/executed/final" / role / (str(r["seed"]) + ".json")
            for r in rows
        ]
        await_files(paths)
        for r, path in zip(rows, paths):
            current = json.loads(path.read_bytes())
            if current["start"] != r["start"]:
                raise ValueError("Final model's initial physics differs from parent")
            r["new"] = {}
            for c in ("gpt2", "transport"):
                pol = dict(
                    core=c,
                    stage=lock["stage"],
                    arm="protected",
                    seed=1411,
                    routed=lock["routed"],
                    contact_only=lock["contact_only"],
                )
                b = current["results"][tag(pol)]
                r["new"][c] = {
                    m: float(np.mean([b[str(k)]["scores"][k][m] for k in range(4)]))
                    for m in ("success", "arrival", "final_mm")
                }
        bs = Groups(
            [r["seed"] for r in rows], [f"{r['speaker']}/{r['shift']}" for r in rows]
        )
        effects = {}
        for c in ("gpt2", "transport"):
            effects[c] = {}
            for m in ("success", "arrival", "final_mm"):
                a = np.array([r["new"][c][m] for r in rows])
                b = np.array([r["original"][c][m] for r in rows])
                effects[c][m] = dict(
                    new=bs.score(a),
                    original=bs.score(b),
                    improvement=bs.score((a - b) * (-1 if m == "final_mm" else 1)),
                )
        out[role] = dict(
            starts=len(rows),
            additional_executions=len(rows) * 8,
            effects=effects,
            rows=rows,
        )
    write(
        ROOT / "work/training-transfer.json",
        dict(
            held=out,
            development=development(),
            seed=1411,
            gate=lock["contact_only"],
            stage=lock["stage"],
        ),
    )
    print(
        "Held training-regime effects",
        {
            k: {c: s["success"]["improvement"] for c, s in v["effects"].items()}
            for k, v in out.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
