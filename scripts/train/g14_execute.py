"""Execute every paired goal and score actual contact outcomes at a fixed physical horizon."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from data.g14_table import ROOT, TOL, OTHER, LIMIT, CHUNK, evaluate_path
from data.g14_policy import choose, loaded
from data.g14_fork import isolated
from scripts.data.g14_collect import init_worker
import scripts.data.g14_collect as collector
from scripts.data.g14_runtime import json_file

SEEDS = (1411, 1412, 1413)


def policies(stage):
    if stage == "pilot":
        base = [dict(core=k) for k in ("noop", "straight", "geometry", "raw")]
        base += [dict(core="geometry", routed=True), dict(core="raw", routed=True)]
        base += [
            dict(core=c, arm=a, seed=s)
            for c in ("gpt2", "transport", "gated", "dense_spatial")
            for a in ("direct", "protected")
            for s in SEEDS
        ]
        base += [
            dict(core=c, arm="protected", seed=1411, routed=True)
            for c in ("gpt2", "transport")
        ]
        for b in base:
            b["oracle_cue"] = True
        return base
    if stage in ("refine", "paired", "isolated"):
        setting = json.loads((ROOT / "work/routing-lock.json").read_bytes())["routed"]
        base = [
            dict(
                core=c,
                arm="protected",
                seed=s,
                routed=setting,
                stage=t,
                oracle_cue=True,
            )
            for c in ("gpt2", "transport")
            for s in SEEDS
            for t in (
                ("initial", "expert-paired", "visited-paired")
                if stage in ("paired", "isolated")
                else ("expert-plus", "visited-plus")
            )
        ]
        return base
    if stage in ("gate", "isolated-gate"):
        lock = json.loads((ROOT / "work/development-selection.json").read_bytes())
        return [
            dict(
                core=c,
                arm="protected",
                seed=s,
                routed=lock["routed"],
                stage=lock["stage"],
                oracle_cue=True,
                contact_only=True,
            )
            for c in ("gpt2", "transport")
            for s in SEEDS
        ] + [dict(core="contact_speed", oracle_cue=True)]
    return json.loads((ROOT / "work/final-control-lock.json").read_bytes())["policies"]


def tag(p):
    base = "/".join(
        (
            p.get("stage", "initial"),
            p["core"],
            p.get("arm", "base"),
            str(p.get("seed", 0)),
            "routed" if p.get("routed") else "directpath",
            "oracle" if p.get("oracle_cue") else "speech",
            "truth" if p.get("oracle_pose") else "camera",
            "reset" if p.get("reset_memory") else "carry",
        )
    )

    return (
        base
        + ("/contact" if p.get("contact_only") else "")
        + ("/original_edge" if p.get("original_edge") else "")
        + ("/open_gate" if p.get("open_gate") else "")
        + ("/mean_field" if p.get("mean_field") else "")
    )


def score(traj, start, c, target):
    distance, other = evaluate_path(traj, start, c, target)
    inside = distance <= TOL
    found = np.flatnonzero(inside[:-2] & inside[1:-1] & inside[2:]) + 2
    first = int(found[0]) if len(found) else -1
    return dict(
        success=int(first >= 0 and other[: first + 1].max() <= OTHER),
        arrival=int(first >= 0),
        first=first,
        terminal=int(inside[-3:].all()),
        terminal_selective=int(inside[-3:].all() and other.max() <= OTHER),
        final_mm=float(distance[-1] * 1000),
        other_mm=float(other.max() * 1000),
        progress_mm=float((np.linalg.norm(start[c] - target) - distance[-1]) * 1000),
    )


def run_one(table, case, ci, stations, policy, variant, full=False):
    table.restore(table.root_state)
    signature = table.physical_signature()
    start = table.positions()
    cue = case["commands"][ci]
    c = cue["true_color"] if policy.get("oracle_cue") else cue["heard_color"]
    st = cue["true_station"] if policy.get("oracle_cue") else cue["heard_station"]
    if (
        not policy.get("oracle_cue")
        and not policy.get("open_gate")
        and not cue["accept"]
    ):
        c = -1
    target = stations[st]
    poses = []
    eef = []
    actions = []
    ranks = []
    model_calls = 0
    im = []
    wait = 0
    initial_img = None
    for n in range(LIMIT):
        obs = table.observation()
        if n == 0:
            initial_img = hashlib.sha256(obs["px"].tobytes()).hexdigest()[:20]
        if policy.get("oracle_pose"):
            obs = dict(**obs)
            obs["q"] = table.positions()
            obs["seen"] = np.ones(4, "f4")
        active = -1 if n > 0 and policy.get("reset_memory") else c
        cmd, info = choose(table, obs, active, target, policy, variant, wait)
        wait = wait + 1 if np.linalg.norm(cmd) < 1e-6 else 0
        q, ee = table.execute(cmd)
        poses.append(q)
        eef.append(ee)
        actions.append(cmd)
        ranks.append(info["chosen"])
        model_calls += int(info.get("model_evaluated", False))
        if full:
            im.append(table.render())
    traj = np.concatenate(poses)
    out = dict(
        start=signature,
        image=initial_img,
        executed_for=ci,
        heard_color=c,
        heard_station=st,
        scores=[
            score(traj, start, x["true_color"], stations[x["true_station"]])
            for x in case["commands"]
        ],
        chosen=np.bincount(ranks, minlength=8).tolist(),
        model_calls=model_calls,
        effort_m=float(np.linalg.norm(actions, axis=1).sum()),
        trajectory_sha=hashlib.sha256(traj.tobytes()).hexdigest(),
        initial=start.tolist(),
        terminal=traj[-1].tolist(),
    )
    return out, (
        (traj, np.concatenate(eef), np.asarray(actions), np.asarray(im))
        if full
        else None
    )


def one(args):
    role, stage, case, ps, root, trace = args
    seed = case["seed"]
    file = root / f"{seed}.json"
    if file.exists():
        return seed
    table = (
        collector.clean_worker()
        if stage in ("paired", "gate", "isolated", "isolated-gate", "final")
        else collector.ENV
    )
    signature = table.reset(seed, case["shift"])
    stations = table.stations(seed)
    variant = json.loads((ROOT / "work/geometry-choice.json").read_bytes())["variant"]
    order = [
        (i, j)
        for i, p in enumerate(ps)
        for j in (range(1) if p["core"] == "noop" else range(4))
    ]
    np.random.default_rng(seed + 941).shuffle(order)
    results = {}
    im_hash = None
    separate = stage in ("isolated", "isolated-gate", "final")
    if separate:
        for p in ps:
            if p["core"] in ("gpt2", "transport", "gated", "dense_spatial", "raw"):
                loaded(
                    p.get("stage", "initial"),
                    p["core"],
                    p.get("arm", "protected"),
                    p.get("seed", 1411),
                )
    for i, j in order:
        p = ps[i]
        trace_this = bool(trace and i == len(ps) - 1 and j == 0)
        source = (
            {**case, "commands": case["base_commands"]}
            if p.get("original_edge")
            else case
        )
        if separate:
            row, raw = isolated(
                run_one, table, source, j, stations, p, variant, trace_this
            )
        else:
            row, raw = run_one(table, source, j, stations, p, variant, trace_this)
        if row["start"] != signature or (
            im_hash is not None and row["image"] != im_hash
        ):
            raise ValueError("Paired start or camera pixels differ")
        im_hash = row["image"]
        name = tag(p)
        results.setdefault(name, {})[str(j)] = row
        if raw is not None:
            path = root / f"{seed}-trace.npz"
            np.savez_compressed(
                path, truth=raw[0], effector=raw[1], actions=raw[2], rgb=raw[3]
            )
    json_file(
        file,
        dict(
            seed=seed,
            role=role,
            stage=stage,
            isolated_branch_process=separate,
            speaker=case["speaker"],
            shift=case["shift"],
            commands=case["commands"],
            addresses=case["addresses"],
            station_xy=stations.tolist(),
            start=signature,
            results=results,
        ),
    )
    return seed


def main(role, stage, count, workers):
    query = (
        ROOT
        / "data"
        / (
            "cases-base"
            if stage
            in ("pilot", "refine", "paired", "gate", "isolated", "isolated-gate")
            else "cases"
        )
        / (role + ".json")
    )
    cases = json.loads(query.read_bytes())
    cases = cases[:count] if count else cases
    if stage == "final":
        original = {
            r["seed"]: r
            for r in json.loads(
                (ROOT / "data/cases-base" / (role + ".json")).read_bytes()
            )
        }
        for r in cases:
            other = original[r["seed"]]
            if any(
                (a["key"], a["true_color"], a["true_station"])
                != (b["key"], b["true_color"], b["true_station"])
                for a, b in zip(r["commands"], other["commands"])
            ):
                raise ValueError("Speech comparison changed the prescribed goal")
            r["base_commands"] = other["commands"]
    ps = policies(stage)
    folder = ROOT / "work/executed" / stage / role
    folder.mkdir(exist_ok=True, parents=True)
    with ProcessPoolExecutor(
        max_workers=min(workers, len(cases)), initializer=init_worker
    ) as pool:
        for i, _ in enumerate(
            pool.map(one, [(role, stage, c, ps, folder, False) for c in cases]), 1
        ):
            if i % 8 == 0:
                print(stage, role, i, len(cases), flush=True)
    json_file(
        folder / "complete.json",
        dict(cases=len(cases), policies=ps, role=role, stage=stage),
    )
    print("Finished physical rollouts", stage, role, len(cases), len(ps), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("role", choices=("dev", "held", "fresh"))
    a.add_argument(
        "stage",
        choices=(
            "pilot",
            "refine",
            "paired",
            "gate",
            "isolated",
            "isolated-gate",
            "final",
        ),
    )
    a.add_argument("--count", type=int)
    a.add_argument("--workers", type=int, default=8)
    p = a.parse_args()
    main(p.role, p.stage, p.count, p.workers)
