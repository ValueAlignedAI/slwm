"""Independently count original media, repeated commands, crossed goals and world-level separation."""

import json
from collections import Counter
from pathlib import Path
import numpy as np
from data.g14_grid import labels
from scripts.train.g9_storage import write, sha

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def main():
    data = ROOT / "data"
    meta = json.loads((data / "utterances/media.json").read_bytes())
    known = {m["text"] for m in meta if m["role"] in ("train", "dev", "cal")}
    used_people = {m["person"] for m in meta if m["role"] in ("train", "dev", "cal")}
    train = set()
    for n in ("training", "fresh-teacher-isolated", "visited-isolated"):
        with np.load(data / (n + ".npz")) as d:
            train.update(map(int, d["world"]))
    out = {}
    allheld = set()
    for role in ("held", "fresh"):
        population = [r for r in meta if r["role"] == role]
        people = {r["person"] for r in population}
        sentences = {r["text"] for r in population}
        tests = json.loads((data / "cases" / (role + ".json")).read_bytes())
        if sentences & known or people & used_people:
            raise ValueError("Local speech fit overlap")
        clips = Counter()
        quartets = Counter()
        shifts = Counter()
        for r in tests:
            cs = r["commands"]
            goals = [(int(x["true_color"]), int(x["true_station"])) for x in cs]
            if (
                len(set(goals)) != 4
                or goals[0][0] != goals[1][0]
                or goals[2][0] != goals[3][0]
                or goals[0][1] != goals[2][1]
                or goals[1][1] != goals[3][1]
            ):
                raise ValueError("Bad two-object/two-address cross")
            for x in cs:
                m = meta[x["clip"]]
                color, letter, digit = labels(m["text"])
                if (
                    m["role"] != role
                    or m["person"] != r["speaker"]
                    or int(color) != x["true_color"]
                    or r["addresses"][x["true_station"]] != [int(letter), int(digit)]
                ):
                    raise ValueError("Incorrect source or assigned goal")
                clips[(m["person"], m["key"])] += 1
            quartets[tuple(x["clip"] for x in cs)] += 1
            shifts[r["shift"]] += 1
        worlds = {r["seed"] for r in tests}
        if len(worlds) != len(tests) or worlds & (train | allheld):
            raise ValueError("Held simulator seed overlap")
        allheld |= worlds
        out[role] = dict(
            available_utterances=len(population),
            unique_sentences=len(sentences),
            local_fit_sentence_overlap=0,
            source_people=sorted(people),
            physical_starts=len(worlds),
            nominal=shifts[False],
            shifted=shifts[True],
            scheduled_utterances=sum(clips.values()),
            distinct_scheduled_recordings=len(clips),
            distinct_scheduled_quartets=len(quartets),
            maximum_reuses_of_one_recording=max(clips.values()),
            known_train_world_overlap=0,
            case_sha256=sha(data / "cases" / (role + ".json")),
        )
    write(
        ROOT / "work/partition-audit.json",
        dict(
            regimes=out,
            distinct_training_worlds=len(train),
            held_worlds=len(allheld),
            preprocessing="Exact evaluated sentences absent from local train, development and calibration. Frozen external encoder pretraining membership unknown.",
            shift_coverage="About one third of continuation worlds used the same randomized contact family; held shifts are new realizations.",
        ),
    )
    print("Independent source coverage", out, flush=True)


if __name__ == "__main__":
    main()
