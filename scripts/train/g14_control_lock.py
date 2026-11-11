"""Choose one common physical training recipe on development, then declare the complete held panel."""

import json
import hashlib
import numpy as np
from scripts.train.g9_storage import write, sha
from scripts.train.g14_execute import tag
from evals.g14_trials import ROOT, summary, read
from pathlib import Path


def lock(final=True):
    starts = read(ROOT / "work/executed/paired/dev")
    if len(starts) != 32:
        raise ValueError("Incomplete canonical-start development comparison")
    route = json.loads((ROOT / "work/routing-lock.json").read_bytes())["routed"]
    _, scores = summary(starts)
    means = {}
    for stage in ("initial", "expert-paired", "visited-paired"):
        keys = [
            tag(
                dict(
                    stage=stage,
                    core=core,
                    arm="protected",
                    seed=s,
                    routed=route,
                    oracle_cue=True,
                )
            )
            for s in (1411, 1412, 1413)
            for core in ("gpt2", "transport")
        ]
        means[stage] = dict(
            success=float(
                np.mean([scores[k]["absolute"]["success"]["value"] for k in keys])
            ),
            final_mm=float(
                np.mean([scores[k]["absolute"]["final_mm"]["value"] for k in keys])
            ),
            policies=keys,
        )
    win = min(means, key=lambda x: (-means[x]["success"], means[x]["final_mm"]))
    if not final:
        write(
            ROOT / "work/development-selection.json",
            dict(stage=win, routed=route, selection=means),
        )
        return
    new = read(ROOT / "work/executed/gate/dev")
    if [(r["seed"], r["start"]) for r in starts] != [
        (r["seed"], r["start"]) for r in new
    ]:
        raise ValueError("Incomplete proposal-gate development comparison")
    for a, b in zip(starts, new):
        a["results"].update(b["results"])
    _, scores = summary(starts)
    gates = {}
    for guarded in (False, True):
        keys = [
            tag(
                dict(
                    core=c,
                    arm="protected",
                    seed=seed,
                    stage=win,
                    routed=route,
                    oracle_cue=True,
                    contact_only=guarded,
                )
            )
            for c in ("transport", "gpt2")
            for seed in (1411, 1412, 1413)
        ]
        gates[str(guarded)] = dict(
            success=float(
                np.mean([scores[k]["absolute"]["success"]["value"] for k in keys])
            ),
            final_mm=float(
                np.mean([scores[k]["absolute"]["final_mm"]["value"] for k in keys])
            ),
        )
    guard = (
        min(gates, key=lambda k: (-gates[k]["success"], gates[k]["final_mm"])) == "True"
    )
    ps = [
        dict(core="noop"),
        dict(core="straight"),
        dict(core="geometry", routed=route),
        dict(core="raw", stage=win, routed=route, contact_only=guard),
        dict(core="contact_speed"),
    ]
    ps.extend(
        dict(
            core=c, arm="protected", stage=win, seed=s, routed=route, contact_only=guard
        )
        for c in ("gpt2", "transport")
        for s in (1411, 1412, 1413)
    )
    ps.extend(
        dict(core=c, arm="protected", stage="initial", seed=1411, routed=route)
        for c in ("gated", "dense_spatial")
    )
    ps.extend(
        [
            dict(core="geometry", routed=route, oracle_cue=True),
            dict(core="geometry", routed=route, oracle_cue=True, oracle_pose=True),
        ]
    )
    ps.extend(
        dict(
            core=c,
            arm="protected",
            stage=win,
            seed=1411,
            routed=route,
            oracle_cue=True,
            contact_only=guard,
        )
        for c in ("gpt2", "transport")
    )
    ps.append(
        dict(
            core="transport",
            arm="protected",
            stage=win,
            seed=1411,
            routed=route,
            reset_memory=True,
            contact_only=guard,
        )
    )
    ps.extend(
        [
            dict(core="geometry", routed=route, original_edge=True),
            dict(core="geometry", routed=route, open_gate=True),
        ]
    )
    files = {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            ROOT / "work/speech-choice.json",
            ROOT / "work/routing-lock.json",
            ROOT / "data/aggregation-paired-lock.json",
            *(ROOT / "data/cases" / (r + ".json") for r in ("held", "fresh")),
            *(ROOT / "work" / s / "lock.json" for s in means),
        )
    }
    write(
        ROOT / "work/final-control-lock.json",
        dict(
            stage=win,
            routed=route,
            contact_only=guard,
            policies=ps,
            selection=means,
            contact_gates=gates,
            sha256=files,
        ),
    )
    write(ROOT / "work/development-physical.json", scores)
    print(
        "Development scene selection",
        win,
        {k: v["success"] for k, v in means.items()},
        len(ps),
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--preliminary", action="store_true")
    lock(not p.parse_args().preliminary)
