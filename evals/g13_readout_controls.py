"""Post-hoc attribution controls for the registered 128-coordinate action correction."""

import numpy as np
from evals.g13_direct import WORK, data, regress, optimize, apply
from evals.g13_stats import Groups
from scripts.train.g13_corpus_reference import design
from scripts.train.g13_optim import ROOT
from scripts.train.g9_storage import write
import hashlib


def main():
    tr, dev, last = [data(x) for x in ("train", "dev", "last")]
    x, xd, xt = [design(v) for v in (tr, dev, last)]
    y = tr["plan"].sum(1)
    yd = dev["plan"].sum(1)
    yt = last["plan"].sum(1)
    g = tr["group"]
    with np.load(WORK / "probes/raw.npz") as z:
        f = {k: z[k] for k in z.files}
    pd = apply(xd, f)
    pt = apply(xt, f)
    ot = np.zeros_like(y)
    assignment = np.array(
        [
            int.from_bytes(
                hashlib.sha256(("G13/cross-fit/" + str(v)).encode()).digest()[:4], "big"
            )
            % 5
            for v in g
        ]
    )
    for k in range(5):
        h = regress(
            x[assignment != k],
            y[assignment != k],
            g[assignment != k],
            float(f["lam"]),
            bool(f["robust"]),
        )
        ot[assignment == k] = apply(x[assignment == k], h)
    rng = np.random.default_rng(130886)
    projection = rng.normal(0, 1 / np.sqrt(x.shape[1]), size=(x.shape[1], 126))
    bias = rng.uniform(-np.pi, np.pi, size=126)

    def ff(a):
        return np.cos((a - f["mu"]) / f["sd"] @ projection + bias)

    results = {}
    bs = Groups(last["group"])
    pub = ROOT / "release/metrics"
    oracle = np.load(pub / "direct-actions.npz")
    mean_err = np.mean(
        [
            1000
            * np.linalg.norm(oracle[f"transport-linear-{s}-protected"] - yt, axis=1)
            for s in (1301, 1302, 1303, 1311, 1312)
        ],
        axis=0,
    )
    for arm in ("prior_alone", "untrained_random_features"):
        a, b, c = ot / 0.08, pd / 0.08, pt / 0.08
        if arm != "prior_alone":
            a, b, c = (
                np.column_stack((a, ff(x))),
                np.column_stack((b, ff(xd))),
                np.column_stack((c, ff(xt))),
            )
        q, choice = optimize(a, y - ot, g, b, yd, dev["group"], pd)
        p = pt + choice["alpha"] * apply(c, q)
        err = np.linalg.norm(p - yt, axis=1) * 1000
        results[arm] = dict(
            choice=choice, mm=bs.score(err), learned_gain=bs.score(err - mean_err)
        )
    write(pub / "readout-controls.json", dict(post_hoc=True, results=results))
    print(
        "Independent-coordinate diagnostics",
        [(k, v["mm"]["value"], v["learned_gain"]["value"]) for k, v in results.items()],
        flush=True,
    )


if __name__ == "__main__":
    main()
