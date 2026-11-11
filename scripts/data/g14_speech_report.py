"""Compact descriptive recognition and explicit calibration counts for both speech routes."""

from pathlib import Path
import json
import numpy as np
from scripts.train.g9_storage import write, sha

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def main():
    load = lambda p: json.loads((ROOT / p).read_bytes())
    out = {}
    for source, case in (("cue", "cases-base"), ("cue-semantic", "cases")):
        fit = load("work/" + source + "/choices.json")
        sc = load("work/" + source + "/scores.json")
        tag = sc["selected"]
        out[source] = dict(
            winner=tag,
            dev=fit[tag]["dev"],
            parameters=fit[tag]["parameters"],
            scores=sc["variants"][tag],
            cases=load("data/" + case + "/design.json"),
            gate=load("data/" + case + "/gate.json"),
            extra_lip={},
        )
        for r in ("held", "fresh"):
            trials = load("data/" + case + "/" + r + ".json")
            cmds = [x for t in trials for x in t["commands"]]
            hits = np.array(
                [
                    x["heard_color"] == x["true_color"]
                    and x["heard_station"] == x["true_station"]
                    for x in cmds
                ]
            )
            keep = np.array([x["accept"] for x in cmds])
            out[source]["cases"][r].update(
                valid_scheduled=int(len(hits)),
                correct=int(hits.sum()),
                accepted=int(keep.sum()),
                rejected_correct=int((hits & ~keep).sum()),
                accepted_incorrect=int((keep & ~hits).sum()),
            )
        for k, v in fit.items():
            if v["arm"] == "audiovisual":
                out[source]["extra_lip"][k] = {
                    r: sc["variants"][k][r] for r in ("held", "fresh")
                }
    doc = dict(
        encoder=load("data/semantic-utterances/complete.json"),
        dataset=load("data/utterances/data.json"),
        fresh_sources=load("data/grid-new/sources.json"),
        results=out,
    )
    write(ROOT / "work/speech-results.json", doc)
    print("Verified G14 speech results")


if __name__ == "__main__":
    main()
