"""Lock causal action triplets, fresh session groups and a blinded device-change panel."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import zipfile

from g10_holo import one, session_group
from g11_events import FOLDS, G10, G11, annotate, family

ROOT = Path("/mnt/slwm-data/SLWM/g12/data")
CHANGE_VERBS = {
    "insert",
    "withdraw",
    "remove",
    "plug",
    "unplug",
    "open",
    "close",
    "lock",
    "unlock",
    "switch",
    "turn on",
    "turn off",
    "attach",
    "detach",
    "connect",
    "disconnect",
    "install",
    "slide",
}


def key(value):
    return hashlib.sha256(("G12-sequence/" + str(value)).encode()).hexdigest()


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    tmp.replace(path)


def triplet(actions, index, cut, limit):
    if index < 0 or index + 3 > len(actions):
        return None
    result = actions[index : index + 3]
    start = result[0]["start"]
    if start < cut + 0.2 - 1e-7 or start > cut + 8 or result[2]["start"] > start + 12:
        return None
    neighbors = actions[max(0, index - 1) : min(len(actions), index + 4)]
    if any(
        b["start"] - a["start"] < 0.1 - 1e-7 for a, b in zip(neighbors, neighbors[1:])
    ):
        return None
    if any(a["end"] > limit for a in result):
        return None
    return result


def actions_in(rec):
    result = []
    for e in rec["events"]:
        if e["label"] != "Fine grained action":
            continue
        attr = e["attributes"]
        verb = str(attr.get("Verb", "")).strip().casefold()
        if verb in ("", "none") or not 0 <= e["start"] < e["end"]:
            continue
        result.append(
            dict(
                id=int(e["id"]), start=float(e["start"]), end=float(e["end"]), verb=verb
            )
        )
    return sorted(result, key=lambda a: (a["start"], a["id"]))


def make_rows(rec, catalog):
    actions = actions_in(rec)
    indices = {e["id"]: i for i, e in enumerate(actions)}
    rows = []
    for ev in catalog["events"] + catalog["background"]:
        cut = (ev["cut"] + 1) / 10
        a = triplet(actions, indices.get(ev["id"], -1), cut, catalog["duration"])
        if a is None:
            continue
        prior = [v for v in actions if v["end"] <= cut]
        prior = max(prior, key=lambda e: e["end"])["verb"] if prior else ""
        rows.append(
            dict(
                id=ev["id"],
                tick=ev["cut"],
                cue=ev["cue"],
                wrong=ev["wrong"],
                verbs=[v["verb"] for v in a],
                starts=[v["start"] for v in a],
                ends=[v["end"] for v in a],
                previous=prior,
                conflict=ev.get("conflict", False),
            )
        )
    return rows


def balanced(candidates, count, used):
    tasks = defaultdict(list)
    for rec in candidates:
        tasks[rec["taskType"]].append(rec)
    for pool in tasks.values():
        pool.sort(key=lambda r: key("fresh/" + r["video_name"]), reverse=True)
    result = []
    while len(result) < count:
        before = len(result)
        for task in sorted(tasks, key=key):
            while tasks[task]:
                row = tasks[task].pop()
                group = session_group(row["video_name"])
                if group not in used:
                    used.add(group)
                    result.append(row["video_name"])
                    break
            if len(result) == count:
                break
        if before == len(result):
            raise ValueError("Insufficient disjoint sources")
    return result


def state_panel(roles, records):
    assignments = {}
    for role in ("confirm", "replica", "dev", "train"):
        names = sorted({n for fold in roles.values() for n in fold[role]})
        candidates = defaultdict(list)
        for n in names:
            if n in assignments:
                raise ValueError("Recording contamination between role panels")
            assignments[n] = role
            for row in records[n]["rows"]:
                cut = (row["tick"] + 1) / 10
                end = max(row["ends"])
                if (
                    row["cue"] < 0
                    or cut < 1
                    or end + 1.15 >= records[n]["duration"]
                    or end - cut > 35
                ):
                    continue
                tag = key(f"{n}/{row['id']}")
                likely = any(v in CHANGE_VERBS for v in row["verbs"])
                candidates[records[n]["family"]].append((n, row, likely, tag))
        target = {"train": 320, "dev": 80, "replica": 144, "confirm": 144}[role]
        picked = []
        counts = Counter()
        used = set()
        # Equal representation of available families and likelihood strata, with a per-recording cap.
        for turn in range(target):
            selected = None
            options = sorted(
                candidates,
                key=lambda f: (sum(1 for x in picked if x["family"] == f), key(f)),
            )
            for f in options:
                pool = [
                    x for x in candidates[f] if x[3] not in used and counts[x[0]] < 4
                ]
                if not pool:
                    continue
                enriched = (turn // max(1, len(options))) % 2 == 0
                pool.sort(key=lambda x: (enriched and not x[2], counts[x[0]], x[3]))
                selected = (f, pool[0])
                break
            if selected is None:
                break
            f, (n, row, likely, tag) = selected
            counts[n] += 1
            used.add(tag)
            cut = (row["tick"] + 1) / 10
            end = max(row["ends"])
            picked.append(
                dict(
                    key=tag[:14],
                    recording=n,
                    id=row["id"],
                    family=f,
                    role=role,
                    likely=likely,
                    times=[cut - 0.56, cut - 0.12, end + 0.23, end + 0.92],
                )
            )
        yield from picked


def main(root):
    root.mkdir(exist_ok=True, parents=True)
    for folder in ("fresh/media", "fresh/shards", "vision", "banks", "sheets"):
        (root / folder).mkdir(exist_ok=True, parents=True)
    full = {r["video_name"]: r for r in json.loads((G10 / "labels.json").read_bytes())}
    prior = json.loads((G11 / "groups.json").read_bytes())
    existing = json.loads((G11 / "annotations.json").read_bytes())
    oldbanks = json.loads((G10 / "banks.json").read_bytes())
    used = {session_group(n) for n in existing} | {
        session_group(v["name"]) for group in oldbanks.values() for v in group
    }
    with zipfile.ZipFile(G10 / "split.zip") as z:
        official = {
            n.decode().strip().casefold()
            for role in ("train", "val")
            for n in z.read(role + "-v1_2.txt").splitlines()
        }
    idx = {
        k: json.loads((G10 / (k + "-index.json")).read_bytes())["members"]
        for k in ("video_pitch_shifted", "imu")
    }
    files = defaultdict(dict)
    for source in idx.values():
        for p in source:
            label = {
                "video_pitchshift.mp4": "av",
                "accelerometer_sync.txt": "acc",
                "gyroscope_sync.txt": "gyro",
            }.get(p.split("/")[-1].casefold())
            if label:
                files[p.split("/")[0].casefold()][label] = p
    fresh = {}
    for fam in ("coffee", "computer", "printer", "camera"):
        eligible = [
            v
            for n, v in full.items()
            if family(v["taskType"]) == fam
            and n.casefold() in official
            and len(files[n.casefold()]) == 3
            and v["videoMetadata"]["duration"]["seconds"] >= 38
        ]
        fresh[fam] = balanced(eligible, 12, used)
    groups = {
        f: dict(
            train=prior["folds"][f]["train"],
            dev=prior["folds"][f]["dev"],
            replica=prior["folds"][f]["held"],
            confirm=sorted((n for h in fs for n in fresh[h]), key=key),
        )
        for f, fs in FOLDS.items()
    }
    record = {}
    new = {n for ns in fresh.values() for n in ns}
    freshcat = {n: annotate(full[n]) for n in new}
    write(root / "annotations.json", freshcat)
    for n in sorted({n for f in groups.values() for ns in f.values() for n in ns}):
        cat = freshcat[n] if n in new else existing[n]
        rows = make_rows(full[n], cat)
        record[n] = dict(
            family=cat["family"], duration=cat["duration"], rows=rows, task=cat["task"]
        )
    write(root / "records.json", record)
    write(root / "groups.json", groups)
    panel = list(state_panel(groups, record))
    write(root / "panel.json", panel)
    print(
        "Registered",
        len(record),
        "recordings, fresh",
        len(new),
        "visual pairs",
        Counter(p["role"] for p in panel),
        flush=True,
    )
    for f, roles in groups.items():
        print(
            f,
            {
                role: sum(sum(r["cue"] >= 0 for r in record[n]["rows"]) for n in ns)
                for role, ns in roles.items()
            },
            flush=True,
        )
    with ThreadPoolExecutor(6) as pool:
        for num, _ in enumerate(
            pool.map(lambda n: one(root / "fresh", full[n], idx, files), sorted(new)), 1
        ):
            if num % 8 == 0:
                print("New verified AV/IMU sources", num, flush=True)
    write(root / "fresh/complete.json", dict(count=len(new), recordings=sorted(new)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    main(parser.parse_args().root)
