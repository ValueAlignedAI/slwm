"""Select a motion measurement on blind development labels and apply it without test adjustment."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import struct
import numpy as np
import cv2
from g13_robot import R, episode, outcome
from g13_acquire import write, file_hash

MODES = ("original", "saturated", "sets", "corroborated")
CODES = "0RBGY"


def components(bgr, points=False):
    h, s, v = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    H, W = h.shape
    roi = np.zeros((H, W), np.uint8)
    poly = (
        np.array([[0.14, 0.035], [0.84, 0.035], [0.995, 0.90], [0.005, 0.90]])
        * np.array([W, H])
    ).astype(np.int32)
    cv2.fillPoly(roi, [poly], 1)
    base = (roi != 0) & (v > 60) & (s > 88)
    B, G, red = [a.astype(np.int16) for a in cv2.split(bgr)]
    blue = (
        (h >= 96) & (h <= 124) & (s > 164) & (B > 105) & (B - G > 27) & (B - red > 73)
    )
    hues = ((h <= 10) | (h >= 168), blue, (h >= 38) & (h <= 92), (h >= 13) & (h <= 37))
    g = np.zeros((4, 2, 6), np.float32)
    cloud = []
    for c, hue in enumerate(hues):
        m = cv2.medianBlur((base & hue).astype(np.uint8), 3)
        _, label, stats, centers = cv2.connectedComponentsWithStats(m, connectivity=8)
        kept = []
        for idx, ((x, y, w, hh, a), (cx, cy)) in enumerate(
            zip(stats[1:], centers[1:]), 1
        ):
            if (
                28 <= a <= 1450
                and 5 <= w <= 54
                and 5 <= hh <= 56
                and 0.42 <= w / hh <= 2.8
                and a / (w * hh) > 0.39
            ):
                kept.append((int(a), cx / W, cy / H, w / W, hh / H, idx))
        kept = sorted(sorted(kept, reverse=True)[:2], key=lambda x: x[1])
        for j, (a, x, y, w, hh, idx) in enumerate(kept):
            g[c, j] = (x, y, w, hh, a / (H * W), 1.0)
        if points:
            use = np.zeros_like(m, dtype=bool)
            for x in kept:
                use |= label == x[-1]
            yy, xx = np.nonzero(use)
            select = np.linspace(0, len(xx) - 1, min(160, len(xx)), dtype=int)
            cloud.append(np.stack((xx[select], yy[select]), 1).astype("f4"))
    return g, cloud


def sets(a, b):
    movement = np.zeros(4)
    direction = np.zeros((4, 2))
    valid = np.zeros(4, bool)
    for i, (x, y) in enumerate(zip(a, b)):
        if min(len(x), len(y)) < 18:
            continue
        dist = ((x[:, None] - y[None, :]) ** 2).sum(-1)
        dx = np.sqrt(dist.min(1))
        dy = np.sqrt(dist.min(0))
        movement[i] = min(np.quantile(dx, 0.86), np.quantile(dy, 0.86))
        # Symmetric gained/lost support helps distinguish actual transport from one-sided occlusion.
        left = x[dx >= max(2.8, np.quantile(dx, 0.65))]
        right = y[dy >= max(2.8, np.quantile(dy, 0.65))]
        if len(left) and len(right):
            direction[i] = right.mean(0) - left.mean(0)
        valid[i] = True
    return movement, direction, valid


def measured(frames, mode):
    from g13_robot import visual

    if mode == "original":
        geom = np.stack([visual(f)[1] for f in frames])
        return outcome(geom, 0)
    vals = [components(f, True) for f in frames]
    geom = np.stack([x[0] for x in vals])
    if mode == "saturated":
        return outcome(geom, 0)
    one, flow, ma = sets(vals[0][1], vals[-1][1])
    two, _, mb = sets(vals[0][1], vals[-2][1])
    mask = ma & mb
    strength = np.minimum(one, two)
    y = -1
    if mask.sum() >= 2:
        top = int(np.argmax(np.where(mask, strength, -1)))
        runner = np.sort(np.where(mask, strength, -1))[-2]
        if (
            strength[top] >= 4.2
            and runner <= 0.68 * strength[top]
            and np.linalg.norm(flow[top]) >= 6
        ):
            y = top + 1
        if mask.all() and np.maximum(one, two).max() <= 2.8:
            y = 0
    if mode == "corroborated":
        before, vec, m0 = outcome(geom, 0)
        if y != before:
            y = -1
        mask = mask & m0
    return y, flow.astype("f4"), mask


def get_frames(root, rec):
    p = root / "raw" / ("language_table-train.tfrecord-" + rec["shard"])
    with p.open("rb") as f:
        f.seek(rec["offset"])
        n = struct.unpack("<Q", f.read(8))[0]
        f.read(4)
        _, _, _, frames, _, _ = episode(f.read(n))
    return list(frames)


def decode(frame):
    return cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_REDUCED_COLOR_2)


def select(ratings):
    audit = json.loads((R / "audit/index.json").read_bytes())
    scores = json.loads(ratings.read_bytes())
    ep = json.loads((R / "banks/episodes.json").read_bytes())
    refs = {}
    for r, s in zip(audit, scores):
        if r["role"] == "dev":
            refs.setdefault(r["episode"], []).append((r, s["review"]))
    summary = {mode: Counter() for mode in MODES}
    examples = []
    for idx, ls in refs.items():
        r, ref = ls[0]
        if ref == "U" or any(x != ref for _, x in ls):
            continue
        frames = get_frames(R, ep[idx])
        images = [decode(frames[j]) for j in (r["tick"], -2, -1)]
        for mode in MODES:
            y = measured(images, mode)[0]
            name = "abstain" if y < 0 else "correct" if CODES[y] == ref else "incorrect"
            summary[mode][name] += 1
            if y == 2:
                summary[mode]["blue_predicted"] += 1
                summary[mode]["blue_correct"] += int(ref == "B")
            examples.append(dict(episode=idx, review=ref, recipe=mode, label=int(y)))
    # Reward an auditable correct case, penalize a wrong assignment, and give zero credit for abstention.
    for v in summary.values():
        v["utility"] = v["correct"] - v["incorrect"]
    best = max(
        MODES,
        key=lambda m: (summary[m]["utility"], summary[m]["correct"], -MODES.index(m)),
    )
    write(
        R / "audit/repair-choice.json",
        dict(
            selected=best,
            development=summary,
            rule="correct minus incorrect; then coverage",
            examples=examples,
        ),
    )
    print("Measurement choice", best, summary, flush=True)


def chunk(args):
    root, tag, mode = map(str, args)
    root = Path(root)
    ep = json.loads((root / "banks/episodes.json").read_bytes())
    dest = root / "repair-cache"
    dest.mkdir(exist_ok=True)
    file = dest / (tag + "-" + mode + ".npz")
    if file.exists():
        return str(file)
    ids = []
    gs = []
    starts = []
    labels = []
    flows = []
    masks = []
    count = 0
    for i, r in enumerate(ep):
        if r["shard"] != tag:
            continue
        frames = get_frames(root, r)
        ims = [decode(f) for f in frames]
        if mode == "original":
            from g13_robot import visual

            geo = np.stack([visual(f)[1] for f in ims])
        else:
            geo = np.stack([components(f)[0] for f in ims])
        y, flow, mask = measured([ims[2], ims[-2], ims[-1]], mode)
        ids.append(i)
        starts.append(count)
        count += len(geo)
        gs.append(geo.astype("f2"))
        labels.append(y)
        flows.append(flow)
        masks.append(mask)
    tmp = file.with_suffix(".new.npz")
    np.savez_compressed(
        tmp,
        ids=ids,
        starts=starts,
        geo=np.concatenate(gs),
        labels=labels,
        flow=flows,
        masks=masks,
    )
    tmp.replace(file)
    print("Corrected shard", tag, len(ids), flush=True)
    return str(file)


def build(root):
    lock = json.loads((R / "audit/repair-choice.json").read_bytes())
    mode = lock["selected"]
    ep = json.loads((root / "banks/episodes.json").read_bytes())
    tags = sorted({r["shard"] for r in ep})
    with ProcessPoolExecutor(6) as pool:
        files = list(pool.map(chunk, [(root, t, mode) for t in tags]))
    maps = {}
    for f in files:
        with np.load(f) as z:
            g = z["geo"]
            for i, st, y, flow, mask in zip(
                z["ids"], z["starts"], z["labels"], z["flow"], z["masks"]
            ):
                maps[int(i)] = (g[st : st + ep[i]["length"]], y, flow, mask)
    out = root / "corrected"
    out.mkdir(exist_ok=True)
    receipt = {}
    for file in sorted((root / "banks").glob("*.npz")):
        with np.load(file) as z:
            d = {k: z[k] for k in z.files}
        for j, (i, t) in enumerate(zip(d["episode"], d["tick"])):
            g, y, f, mask = maps[int(i)]
            d["geo"][j] = g[np.maximum(0, np.arange(t - 3, t + 1))]
            if d["first"][j]:
                d["mover"][j] = y
                d["flow"][j] = f
                d["visible"][j] = mask
        p = out / file.name
        tmp = p.with_suffix(".new.npz")
        np.savez_compressed(tmp, **d)
        tmp.replace(p)
        receipt[file.stem] = dict(
            sha256=file_hash(p),
            rows=len(d["tick"]),
            measured=int((d["mover"] >= 0).sum()),
            base_sha=file_hash(file),
        )
    write(out / "data.json", dict(measurement=mode, roles=receipt))
    (out / "ready").touch()
    print("Repaired", receipt, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("select", "build"))
    p.add_argument("--ratings", type=Path)
    p.add_argument("--root", type=Path, default=R)
    a = p.parse_args()
    select(a.ratings) if a.stage == "select" else build(a.root)
