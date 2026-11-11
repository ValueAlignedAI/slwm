"""Measure order sensitivity of the original snapshot-based training targets at actual contact."""

import numpy as np
from scripts.data.g14_collect import clean_worker
from scripts.data.g14_refinement import continuation
from data.g14_table import ROOT, utility, loss_cost, options
from data.g14_fork import isolated
from scripts.train.g9_storage import write


def one(seed):
    t = clean_worker()
    t.reset(248600 + seed, seed % 2 == 1)
    stations = t.stations(seed + 248600)
    c = seed % 3
    goal = stations[seed % 4]
    states = []
    for tick in range(15):
        obs = t.observation()
        acts = options(obs, c, goal, 0)
        if tick in (0, 3, 6, 9, 12):
            pin = t.save()
            q0 = t.positions()
            p0 = t.effector()

            def fork(cmd):
                t.restore(pin)
                q, ee = t.execute(cmd)
                e = q[-1]
                p = ee[-1]
                qh, ph = continuation(t, c, goal, 0, False)
                off = np.delete(np.linalg.norm(qh - q0, axis=1), c).max()
                gain = (
                    np.linalg.norm(q0[c] - goal)
                    - np.linalg.norm(qh[c] - goal)
                    + 0.15 * (loss_cost(q0, p0, c, goal) - loss_cost(qh, ph, c, goal))
                    - 0.85 * off
                )
                return (
                    np.concatenate(((p - p0)[None], e - q0)),
                    utility(q0, p0, e, p, c, goal, cmd),
                    gain,
                )

            exact = [isolated(fork, a) for a in acts]

            def ordinary():
                rows = [fork(a) for a in acts]
                t.restore(pin)
                return rows

            legacy = isolated(ordinary)
            a = np.array([x[0] for x in exact])
            b = np.array([x[0] for x in legacy])
            x = np.array([v[1:] for v in exact])
            y = np.array([v[1:] for v in legacy])
            states.append(
                dict(
                    effect=list(np.linalg.norm((a - b) * 1000, axis=-1).ravel()),
                    value=list(abs(x[:, 1] - y[:, 1]) * 1000),
                    regret=(x[:, 1].max() - x[y[:, 1].argmax(), 1]) * 1000,
                )
            )
            t.p.removeState(pin[0])
        t.execute(acts[1])
    return states


def main():
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(4) as pool:
        rows = [r for group in pool.map(one, range(24)) for r in group]

    def scale(k):
        a = np.concatenate([np.atleast_1d(v[k]) for v in rows])
        return dict(
            mean=float(a.mean()),
            median=float(np.median(a)),
            p95=float(np.quantile(a, 0.95)),
            max=float(a.max()),
        )

    out = dict(
        starts=24,
        states=len(rows),
        candidates=len(rows) * 8,
        one_step_effect_difference_mm=scale("effect"),
        long_return_difference_mm=scale("value"),
        changed_best_action_regret_mm=scale("regret"),
    )
    write(ROOT / "work/training-label-audit.json", out)
    print("Training target order audit", out, flush=True)


if __name__ == "__main__":
    main()
