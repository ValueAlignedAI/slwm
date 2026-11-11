"""Check full-horizon state restoration before releasing held scenes, including intervening rollouts."""

import json
import numpy as np
from scripts.data.g14_collect import clean_worker
from data.g14_policy import choose
from data.g14_table import ROOT, LIMIT
from scripts.train.g9_storage import write


def main(isolate=False):
    from data.g14_fork import isolated

    items = []
    for i in range(8):
        seed = 247600 + i
        t = clean_worker()
        start = t.reset(seed, i % 2 == 1)
        stations = t.stations(seed)

        def roll(c, st):
            t.restore(t.root_state)
            signature = t.physical_signature()
            hist = []
            cams = []
            for tick in range(LIMIT):
                o = t.observation()
                cams.append(o["px"].copy())
                u, _ = choose(t, o, c, stations[st], {"core": "geometry"}, 0)
                hist.append(t.execute(u)[0])
            return signature, np.concatenate(hist), np.stack(cams)

        apply = isolated if isolate else lambda f, *args: f(*args)
        sig, a, im = apply(roll, i % 3, 0)
        apply(roll, (i + 1) % 3, 1)
        apply(roll, i % 3, 2)
        s2, b, jm = apply(roll, i % 3, 0)
        if start != sig or start != s2:
            raise ValueError("Restored body/actuator state changed")
        max_mm = float(np.linalg.norm(a - b, axis=-1).max() * 1000)
        items.append(
            dict(
                seed=seed,
                exact_poses=bool(np.array_equal(a, b)),
                exact_camera=bool(np.array_equal(im, jm)),
                max_mm=max_mm,
            )
        )
    write(
        ROOT / "work" / ("isolation-replay.json" if isolate else "reset-replay.json"),
        dict(worlds=8, native_steps_per_rollout=72, items=items),
    )
    if isolate and not all(x["exact_poses"] and x["exact_camera"] for x in items):
        raise ValueError("Isolated branch failed reproducibility")
    print("Full-horizon reset replay", items, flush=True)


if __name__ == "__main__":
    import sys

    main("--isolated" in sys.argv)
