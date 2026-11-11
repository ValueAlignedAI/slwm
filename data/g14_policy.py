"""Vision-based routing and a portable policy over common, physically bounded motion proposals."""

from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from data.g14_table import ROOT, TOL, clip, unit, options, geometric
from data.g14_signals import pack, raw_predict, attach
from models.g14_world import predict

CAL = ROOT / "work/geometry-choice.json"


def clearance(a, b, points):
    d = b - a
    u = np.clip(((points - a) @ d) / max(d @ d, 1e-9), 0, 1)
    return np.linalg.norm(points - (a + u[:, None] * d), axis=1)


def waypoint(start, goal, obstacles, radius):
    if len(obstacles) == 0 or clearance(start, goal, obstacles).min() >= radius:
        return goal
    ang = np.arange(8) * np.pi / 4
    around = np.stack((np.cos(ang), np.sin(ang)), 1) * (radius + 0.018)
    nodes = np.concatenate(
        (start[None], goal[None], (obstacles[:, None] + around[None]).reshape(-1, 2))
    )
    keep = (
        (nodes[:, 0] >= 0.196) & (nodes[:, 0] <= 0.55) & (np.abs(nodes[:, 1]) <= 0.258)
    )
    keep[:2] = True
    nodes = nodes[keep]
    dx = nodes[None, :, :] - nodes[:, None, :]
    sq = (dx * dx).sum(-1)
    valid = np.ones_like(sq, bool)
    for obs in obstacles:
        v = obs - nodes[:, None, :]
        p = np.clip((v * dx).sum(-1) / np.maximum(sq, 1e-9), 0, 1)
        gap = np.linalg.norm(v - p[:, :, None] * dx, axis=-1)
        valid &= gap >= radius - 1e-6
    lengths = np.where(valid, np.sqrt(sq), np.inf)
    np.fill_diagonal(lengths, np.inf)
    cost = np.full(len(nodes), np.inf)
    cost[0] = 0.0
    prev = np.full(len(nodes), -1)
    used = np.zeros(len(nodes), bool)
    for _ in nodes:
        i = int(np.argmin(np.where(used, np.inf, cost)))
        if used[i] or not np.isfinite(cost[i]) or i == 1:
            break
        used[i] = True
        new = cost[i] + lengths[i]
        mask = (new < cost) & (~used)
        prev[mask] = i
        cost[mask] = new[mask]
    if prev[1] < 0:
        return goal
    j = 1
    while prev[j] > 0:
        j = int(prev[j])
    return nodes[j]


def candidates(obs, c, target, variant, routed):
    if not routed:
        return options(obs, c, target, variant)
    q = obs["q"]
    others = np.delete(q, c, axis=0)
    sub = waypoint(q[c], target, others, 0.053)
    choices = options(obs, c, sub, variant)
    for i in range(1, len(choices)):
        dest = obs["motor"] + choices[i]
        aim = waypoint(obs["p"], dest, others, 0.033)
        choices[i] = clip(aim - obs["motor"], 0.069)
    return choices


def features(table, c, target, choices):
    h = table.history
    data = {
        k: np.stack([o[k] for o in h])[None]
        for k in ("q", "seen", "p", "motor", "px", "u")
    }
    data.update(color=np.array([c]), target=target[None], choices=choices[None])
    return pack(data)


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@lru_cache(maxsize=64)
def loaded(stage, core, arm, seed):
    from hashlib import sha256

    path = ROOT / "work" / stage
    ref = read(path / "reference.npz")
    delayed = read(path / "long-reference.npz") if stage != "initial" else None
    if core == "raw":
        gate = json.loads((path / "reference.json").read_bytes()).get(
            "gate", {"margin": 0.04, "alpha": 0.0, "beta": 0.0}
        )
        return ref, delayed, None, gate
    tag = f"{stage}-{core}-{arm}-{seed}"
    entry = json.loads((path / "lock.json").read_bytes())[tag]
    file = (
        Path("/mnt/slwm-checkpoints-a/SLWM/g14/fields")
        / tag
        / f"step-{entry['selected']:05d}"
        / "weights.npz"
    )
    if sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Unregistered policy weights")
    return ref, delayed, read(file), entry


def choose(table, obs, c, target, policy, variant, linger=0, loader=loaded):
    kind = policy["core"]
    routed = bool(policy.get("routed", False))
    distance = float(np.linalg.norm(obs["q"][c] - target)) if c >= 0 else 0.0
    if c < 0 or kind == "noop":
        return np.zeros(2, "f4"), dict(route="internal_only", chosen=0)
    if distance < 0.014:
        return np.zeros(2, "f4"), dict(route="no_op", chosen=0)
    if kind == "straight":
        return geometric(obs, c, target, False, variant), dict(route="plan", chosen=7)
    acts = candidates(obs, c, target, variant, routed)
    if kind == "geometry":
        return acts[1], dict(route="plan", chosen=1)
    contact = bool(policy.get("contact_only") or kind == "contact_speed")
    if contact:
        axis = unit(target - obs["q"][c])
        delta = obs["p"] - obs["q"][c]
        along = float(delta @ axis)
        across = float(abs(delta @ np.array([-axis[1], axis[0]])))
        clear = (
            np.linalg.norm(np.delete(obs["q"], c, axis=0) - obs["q"][c], axis=1).min()
            > 0.050
        )
        if not (-0.085 <= along <= 0.012 and across <= 0.027 and clear):
            return acts[1], dict(route="plan", chosen=1)
        if kind == "contact_speed":
            return acts[3], dict(route="plan", chosen=3)
    ref, delayed, model, entry = loader(
        policy.get("stage", "initial"),
        kind,
        policy.get("arm", "protected"),
        policy.get("seed", 1411),
    )
    row = features(table, c, target, acts)
    base = raw_predict(row, ref)
    attach(row, base)
    short = base[0, :, -1]
    long = short
    if delayed is not None:
        row["long_prior"] = raw_predict(row, delayed)[:, :, 0]
        long = row["long_prior"][0]
    if model is not None:
        alpha = entry["alpha"]
        if delayed is not None:
            vals, _, _, value = predict(model, row, kind, policy["arm"], np, True)
            long = long + alpha * (value[0] - long)
        else:
            vals, _, _ = predict(model, row, kind, policy["arm"], np)
        short = short + alpha * (vals[0, :, -1] - short)
    beta = entry.get("beta", 0.0)
    score = (1 - beta) * short + beta * long
    margin = entry["margin"]
    if contact:
        score = score.copy()
        score[[0, 7]] = -np.inf
    pick = int(score.argmax())
    if score[pick] - score[1] <= margin:
        pick = 1
    # A consequence model may wait. Record that choice and only recover if a separately enabled progress guard fires.
    if policy.get("recover", False) and linger >= 3 and pick == 0:
        pick = 1
    return acts[pick], dict(
        chosen=pick,
        route="no_op" if pick == 0 else "plan",
        advantage_mm=float((score[pick] - score[1]) * 40),
    )
