"""Estimate 5 Hz response in simulation and calibrate it with small, disjoint real-robot training sets."""

from pathlib import Path
import argparse
import json
import numpy as np
from training.g13_physics import design, reference, identify
from evals.g13_stats import Groups
from scripts.train.g9_storage import write, sha

ROOT = Path("/mnt/slwm-data/SLWM/g14")
OLD = ROOT.parent / "g13/data"
KEYS = ("kin", "plan", "future", "group")


def read(file):
    with np.load(file, allow_pickle=False) as z:
        return {k: z[k] for k in KEYS}


def simulate():
    from data.g14_table import Table

    data = read(OLD / "robot/corrected/train.npz")
    table = Table(hz=5)
    rows = []
    rng = np.random.default_rng(143087)
    for world in range(192):
        table.reset(275000 + world, world % 3 == 0)
        xs = [table.effector()]
        us = []
        for chunk in range(10):
            # Match action/position support using training recordings; no recorded future is an input.
            idx = rng.integers(len(data["plan"]), size=192)
            d = np.linalg.norm(data["kin"][idx, -1, :2] - xs[-1], axis=1)
            i = idx[np.argmin(d)]
            for u in data["plan"][i]:
                table.e._step_robot_and_sim(u)
                xs.append(table.effector())
                us.append(u.copy())
        x = np.stack(xs)
        u = np.stack(us)
        for clock in range(3, len(u) - 3, 2):
            h = np.arange(clock - 3, clock + 1)
            p = x[h]
            prev = np.zeros((4, 2), "f4")
            prev[h > 0] = u[h[h > 0] - 1]
            kin = np.concatenate((p, np.diff(p, axis=0, prepend=p[:1]), prev), axis=1)
            rows.append(
                dict(
                    group=world,
                    kin=kin,
                    plan=u[clock : clock + 4],
                    future=x[clock + 1 : clock + 5] - x[clock],
                )
            )
    np.savez_compressed(
        ROOT / "data/telemetry-sim.npz",
        **{k: np.stack([r[k] for r in rows]) for k in KEYS},
    )
    print("Measured native five-Hz simulator response", len(rows), flush=True)


def adapt(r, coef):
    g = r["group"]
    _, i, n = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / n[i]
    w /= w.sum()
    out = {}
    for h in range(1, 5):
        name = "active" + str(h)
        x = design(r["kin"], r["plan"], h)
        y = r["future"][:, h - 1]
        center = coef[name]
        out[name] = (
            center
            + np.linalg.solve(
                x.T @ (w[:, None] * x) + 0.00003 * np.eye(x.shape[1]),
                x.T @ ((y - x @ center) * w[:, None]),
            )
        ).astype("f4")
    return out


def evaluate():
    train = read(OLD / "robot/corrected/train.npz")
    sim = read(ROOT / "data/telemetry-sim.npz")
    simref = identify(sim)
    full = identify(train)
    sizes = (1, 4, 16, 64)
    ids = np.unique(train["group"])
    packages = {}
    for seed in (1431, 1432, 1433, 1434, 1435):
        order = np.random.default_rng(seed).permutation(ids)
        for n in sizes:
            mask = np.isin(train["group"], order[:n])
            r = {k: x[mask] for k, x in train.items()}
            packages[(seed, n)] = (identify(r), adapt(r, simref))
    out = {}
    weights = ROOT / "work/telemetry-coefficients.npz"
    np.savez_compressed(
        weights,
        **{"sim_" + k: v for k, v in simref.items()},
        **{"real_" + k: v for k, v in full.items()},
        **{"calibrated4_" + k: v for k, v in packages[(1431, 4)][1].items()},
    )
    for cohort in ("robot-prospective", "robot-readout"):
        file = OLD / cohort / "corrected/test.npz"
        r = read(file)
        obs = r["future"]
        bs = Groups(r["group"])

        def err(v):
            return 1000 * np.linalg.norm(v - obs, axis=-1)

        def pred(p):
            return reference(r["kin"], r["plan"], p, True)

        def metrics(d):
            return [bs.score(d[:, h]) for h in range(4)]

        from_sim = err(pred(simref))
        from_real = err(pred(full))
        baseline = dict(
            still=metrics(err(np.zeros_like(obs))),
            integrate=metrics(err(np.cumsum(r["plan"], axis=1))),
            sim=metrics(from_sim),
            real_all=metrics(from_real),
        )
        cal = {}
        for n in sizes:
            cf = []
            scratch = []
            for seed in (1431, 1432, 1433, 1434, 1435):
                a, b = packages[(seed, n)]
                scratch.append(err(pred(a)))
                cf.append(err(pred(b)))
            ea = np.mean(scratch, axis=0)
            eb = np.mean(cf, axis=0)
            cal[n] = dict(
                real_only=metrics(ea),
                sim_calibrated=metrics(eb),
                advantage_over_scratch_mm=bs.score(ea[:, -1] - eb[:, -1]),
                improvement_over_sim_mm=bs.score(from_sim[:, -1] - eb[:, -1]),
                endpoint_by_calibration_seed=[bs.score(x[:, -1])["value"] for x in cf],
            )
        out[cohort] = dict(
            rows=len(obs),
            parents=len(set(r["group"])),
            source_sha256=sha(file),
            baseline=baseline,
            calibration=cal,
        )
    write(
        ROOT / "work/telemetry.json",
        dict(
            sim_worlds=len(set(sim["group"])),
            sim_windows=len(sim["group"]),
            train_groups=len(ids),
            source="Original public Language-Table captures previously released in G13",
            native_hz=5,
            endpoint_seconds=0.8,
            real_labels_for_sim_prior=0,
            control_source="real training inputs",
            cohorts=out,
        ),
    )
    for k, v in out.items():
        print(
            k,
            {x: round(z[-1]["value"], 2) for x, z in v["baseline"].items()},
            {
                n: round(z["sim_calibrated"][-1]["value"], 2)
                for n, z in v["calibration"].items()
            },
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("simulate", "evaluate"))
    a = p.parse_args()
    (simulate if a.mode == "simulate" else evaluate)()
