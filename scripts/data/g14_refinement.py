"""Collect learner-visited and demonstration controls with measured physical continuation returns."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from data.g14_table import ROOT, CHUNK, LIMIT, TOL, utility, loss_cost
from data.g14_policy import candidates, choose
from scripts.data.g14_collect import init_worker, prepare, obs_pack
import scripts.data.g14_collect as collector
from scripts.data.g14_runtime import json_file

HORIZON = 4


def continuation(t, c, target, variant, routed):
    q = t.positions()
    p = t.effector()
    # Same fixed feedback continuation follows every candidate. Future simulator poses are label-side only.
    for i in range(HORIZON):
        if np.linalg.norm(q[c] - target) < 0.014:
            action = np.zeros(2)
        else:
            obs = dict(q=q, p=p, motor=t.motor(), seen=np.ones(4), u=t.last_u)
            action = candidates(obs, c, target, variant, routed)[1]
        poses, eef = t.execute(action)
        q = poses[-1]
        p = eef[-1]
    return q, p


def one(args):
    mode, seed, paired = args
    name = mode + ("-paired" if paired else "")
    dir = ROOT / "data" / name
    f = dir / f"{seed}.npz"
    if f.is_file():
        return len(np.load(f)["world"])
    t = collector.clean_worker() if paired else collector.ENV
    conf = json.loads((ROOT / "work/routing-lock.json").read_bytes())
    routed = conf["routed"]
    variant = json.loads((ROOT / "work/geometry-choice.json").read_bytes())["variant"]
    sig, stations = prepare(seed, seed % 3 == 0, t)
    rows = []
    for trial in range(2):
        t.restore(t.root_state)
        if t.physical_signature() != sig:
            raise ValueError("Unmatched aggregation starts")
        c = (seed + trial) % 3
        goal = stations[(seed // 3 + trial) % 4]
        rng = np.random.default_rng(seed * 17 + trial)
        measure = {0, *map(int, rng.choice(np.arange(1, LIMIT), size=7, replace=False))}
        # Fixed-length visits allow late failures and settling/no-op states to enter both arms.
        policy = (
            dict(core="geometry", routed=routed)
            if mode == "fresh-teacher" or mode == "value-dev" and trial == 0
            else dict(
                core=("raw", "gpt2", "transport")[(seed + trial) % 3],
                arm="protected",
                seed=1411,
                stage="initial",
                routed=routed,
            )
        )
        for tick in range(LIMIT):
            obs = t.observation()
            opts = candidates(obs, c, goal, variant, routed)
            q0 = t.positions()
            p0 = t.effector()
            if tick in measure:
                saved = t.save()
                effects = []
                gains = []
                tails = []
                distances = []
                try:
                    for cmd in opts:
                        t.restore(saved)
                        qs, ps = t.execute(cmd)
                        q1, p1 = qs[-1], ps[-1]
                        effects.append(np.concatenate(((p1 - p0)[None], q1 - q0)))
                        gains.append(utility(q0, p0, q1, p1, c, goal, cmd))
                        distances.append(np.linalg.norm(q1[c] - goal))
                        qh, ph = continuation(t, c, goal, variant, routed)
                        # The longer return places more weight on actual target progress and penalizes unintended contact.
                        off = np.delete(np.linalg.norm(qh - q0, axis=1), c).max()
                        tails.append(
                            np.linalg.norm(q0[c] - goal)
                            - np.linalg.norm(qh[c] - goal)
                            + 0.15
                            * (loss_cost(q0, p0, c, goal) - loss_cost(qh, ph, c, goal))
                            - 0.85 * off
                        )
                finally:
                    t.restore(saved)
                    t.p.removeState(saved[0])
                rows.append(
                    dict(
                        **obs_pack(t),
                        world=seed,
                        color=c,
                        target=goal,
                        choices=opts,
                        clock=tick,
                        trial=trial,
                        effect=np.stack(effects),
                        gain=np.asarray(gains),
                        measured=np.asarray(distances),
                        long=np.asarray(tails),
                    )
                )
                if paired:
                    rows[-1]["start_digest"] = sig
            cmd, _ = choose(t, obs, c, goal, policy, variant)
            t.execute(cmd)
    d = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    tmp = f.with_suffix(".part.npz")
    np.savez_compressed(tmp, **d)
    tmp.replace(f)
    return len(rows)


def main(mode, count, workers, paired):
    name = mode + ("-paired" if paired else "")
    folder = ROOT / "data" / name
    folder.mkdir(exist_ok=True)
    with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
        for i, n in enumerate(
            pool.map(
                one,
                [
                    (mode, (156000 if mode == "value-dev" else 155000) + k, paired)
                    for k in range(count)
                ],
            ),
            1,
        ):
            if i % 16 == 0:
                print(mode, i, count, flush=True)
    docs = []
    for f in sorted(folder.glob("*.npz")):
        with np.load(f) as z:
            docs.append({k: z[k] for k in z.files})
    file = ROOT / "data" / (name + ".npz")
    temp = file.with_suffix(".new.npz")
    np.savez_compressed(
        temp, **{k: np.concatenate([x[k] for x in docs]) for k in docs[0]}
    )
    temp.replace(file)
    print(
        "Paired learner-visitation bank",
        mode,
        len(docs),
        sum(len(x["world"]) for x in docs),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("fresh-teacher", "visited", "value-dev"))
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--paired", action="store_true")
    a = p.parse_args()
    main(a.mode, a.count, a.workers, a.paired)
