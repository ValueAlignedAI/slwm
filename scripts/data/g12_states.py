"""Bind blind device classifications to recordings only after scoring, and check hidden duplicates."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from g12_events import ROOT, write

CODES = {
    "S": "same visible configuration",
    "U": "indeterminate",
    "O": "opening or closure",
    "A": "attachment or removal",
    "C": "configuration or load",
    "D": "display or indicator",
}


def bind(panel, index, ratings):
    if len(index) != len(ratings):
        raise ValueError("Missing blinded observations")
    cases = {p["key"]: p for p in panel}
    by_key = defaultdict(list)
    for pos, (match, rating) in enumerate(zip(index, ratings), 1):
        code = rating["code"]
        if match["number"] != pos or rating["number"] != pos or code not in CODES:
            raise ValueError("Invalid blind index")
        if code not in ("S", "U") and not rating.get("component"):
            raise ValueError("Changed state needs an identifiable component")
        by_key[match["key"]].append(rating)
    if set(cases) != set(by_key):
        raise ValueError("Missing device cases")
    rep = Counter()
    binary = Counter()
    items = []

    def state(code):
        return -1 if code == "U" else int(code != "S")

    for key, rates in by_key.items():
        r = cases[key]
        labels = [state(x["code"]) for x in rates]
        if len(labels) > 1:
            rep[(labels[0], labels[1])] += 1
            if min(labels) >= 0:
                binary[(labels[0], labels[1])] += 1
        # If ratings disagree about measurability or change, exclude this case from the physical endpoint.
        label = labels[0] if len(set(labels)) == 1 else -1
        items.append(
            dict(
                **r,
                label=label,
                ratings=[
                    dict(
                        number=q["number"],
                        code=q["code"],
                        component=q.get("component", ""),
                    )
                    for q in rates
                ],
            )
        )
    n = sum(rep.values())
    agreement = sum(v for (x, y), v in rep.items() if x == y)
    nb = sum(binary.values())
    ab = sum(v for (x, y), v in binary.items() if x == y)
    marg_a = Counter()
    marg_b = Counter()
    for (x, y), v in rep.items():
        marg_a[x] += v
        marg_b[y] += v
    chance = sum(marg_a[v] * marg_b[v] for v in (-1, 0, 1)) / max(1, n * n)
    return dict(
        items=items,
        codes=CODES,
        reviewer="single blinded visual review",
        repeats=n,
        agreement=agreement / max(1, n),
        kappa=(agreement / max(1, n) - chance) / max(1e-8, 1 - chance),
        repeats_both_readable=nb,
        readable_agreement=ab / max(1, nb),
        confusion={str(a) + "/" + str(b): v for (a, b), v in rep.items()},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ratings", type=Path)
    p.add_argument("--root", type=Path, default=ROOT)
    a = p.parse_args()
    d = bind(
        json.loads((a.root / "panel.json").read_bytes()),
        json.loads((a.root / "sheets/index.json").read_bytes()),
        json.loads(a.ratings.read_bytes()),
    )
    write(a.root / "state-labels.json", d)
    print(
        "Independent observations",
        len(d["items"]),
        "hidden repeats",
        d["repeats"],
        "kappa",
        round(d["kappa"], 3),
    )
