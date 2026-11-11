"""Recheck the early routing comparison after isolating the complete simulator state."""

import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from data.g14_fork import isolated
from data.g14_table import ROOT
from scripts.train.g14_execute import run_one, tag
from scripts.data.g14_collect import clean_worker
from scripts.train.g9_storage import write
from evals.g13_stats import Groups


def one(case):
    t = clean_worker()
    seed = case["seed"]
    root = t.reset(seed, case["shift"])
    stations = t.stations(seed)
    scores = {}
    for core in ("geometry", "gpt2", "transport"):
        for routed in (False, True):
            p = dict(core=core, routed=routed, oracle_cue=True)
            if core != "geometry":
                p.update(stage="initial", arm="protected", seed=1411)
            result = [
                isolated(run_one, t, case, i, stations, p, 0)[0] for i in range(4)
            ]
            if any(x["start"] != root for x in result):
                raise ValueError("Changed routing comparison root")
            scores[tag(p)] = float(
                np.mean([r["scores"][i]["success"] for i, r in enumerate(result)])
            )
    return dict(seed=seed, speaker=case["speaker"], scores=scores)


def main():
    cases = json.loads((ROOT / "data/cases-base/dev.json").read_bytes())
    with ProcessPoolExecutor(3) as pool:
        rows = list(pool.map(one, cases))
    bs = Groups([r["seed"] for r in rows], [r["speaker"] for r in rows])
    values = {
        route: np.asarray(
            [
                np.mean([x for k, x in r["scores"].items() if "/" + route + "/" in k])
                for r in rows
            ]
        )
        for route in ("directpath", "routed")
    }
    write(
        ROOT / "work/routing-audit.json",
        dict(
            isolated={k: bs.score(x) for k, x in values.items()},
            isolated_direct_over_routed=bs.score(
                values["directpath"] - values["routed"]
            ),
            early=json.loads((ROOT / "work/routing-lock.json").read_bytes()),
            policy_changed=False,
        ),
    )
    print(
        "Independent routing replication",
        {k: float(v.mean()) for k, v in values.items()},
        flush=True,
    )


if __name__ == "__main__":
    main()
