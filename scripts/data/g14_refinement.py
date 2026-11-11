"""Collect learner-visited and demonstration controls with measured physical continuation returns."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from data.g14_table import ROOT, LIMIT, utility, loss_cost
from data.g14_policy import candidates, choose
from scripts.data.g14_collect import init_worker, prepare, obs_pack
import scripts.data.g14_collect as collector
from data.g14_fork import isolated

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
    mode, seed, paired, separate = args
    name = mode + ("-isolated" if separate else "-paired" if paired else "")
    dir = ROOT / "data" / name
    f = dir / f"{seed}.npz"
    if f.is_file():
        return len(np.load(f)["world"])
    t = collector.clean_worker() if paired or separate else collector.ENV
    conf = json.loads((ROOT / "work/routing-lock.json").read_bytes())
    routed = conf["routed"]
    variant = json.loads((ROOT / "work/geometry-choice.json").read_bytes())["variant"]
    shift=bool(seed%2) if mode=="value-test" else seed%3==0
    sig, stations = prepare(seed, shift, t)
    rows = []
    for trial in range(2):
        if separate and trial:
            t = collector.clean_worker()
            new_sig, new_places = prepare(seed,shift,t)
            if new_sig != sig or not np.array_equal(new_places, stations):
                raise ValueError("Goal trial root changed")
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
            if mode == "fresh-teacher" or mode in ("value-dev","value-test") and trial == 0
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

                def outcome(cmd):
                    t.restore(saved)
                    qs, ps = t.execute(cmd)
                    q1, p1 = qs[-1], ps[-1]
                    effect = np.concatenate(((p1 - p0)[None], q1 - q0))
                    immediate = utility(q0, p0, q1, p1, c, goal, cmd)
                    qh, ph = continuation(t, c, goal, variant, routed)
                    off = np.delete(np.linalg.norm(qh - q0, axis=1), c).max()
                    tail = (
                        np.linalg.norm(q0[c] - goal)
                        - np.linalg.norm(qh[c] - goal)
                        + 0.15
                        * (loss_cost(q0, p0, c, goal) - loss_cost(qh, ph, c, goal))
                        - 0.85 * off
                    )
                    return effect, immediate, tail, np.linalg.norm(q1[c] - goal)

                try:
                    if separate:
                        original = t.physical_signature()
                        results = [isolated(outcome, cmd) for cmd in opts]
                        if original != t.physical_signature():
                            raise ValueError(
                                "A physical intervention changed its parent"
                            )
                    else:
                        results = [outcome(cmd) for cmd in opts]
                    effects, gains, tails, distances = zip(*results)
                finally:
                    if not separate:
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
                if paired or separate:
                    rows[-1]["start_digest"] = sig
            cmd, _ = choose(t, obs, c, goal, policy, variant)
            t.execute(cmd)
    d = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    tmp = f.with_suffix(".part.npz")
    np.savez_compressed(tmp, **d)
    tmp.replace(f)
    return len(rows)


def main(mode, count, workers, paired, separate=False):
    name = mode + ("-isolated" if separate else "-paired" if paired else "")
    folder = ROOT / "data" / name
    folder.mkdir(exist_ok=True)
    with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
        for i, n in enumerate(
            pool.map(
                one,
                [
                    (
                        mode,
                        ({"value-dev":156000,"value-test":286000}.get(mode,155000)) + k,
                        paired,
                        separate,
                    )
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
    p.add_argument("mode", choices=("fresh-teacher","visited","value-dev","value-test"))
    p.add_argument("--count", type=int, default=128)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--paired", action="store_true")
    p.add_argument("--isolated", action="store_true")
    a = p.parse_args()
    main(a.mode, a.count, a.workers, a.paired, a.isolated)
