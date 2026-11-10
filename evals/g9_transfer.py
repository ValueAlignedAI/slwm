"""Blind-room reconstruction, near-time controls and scene-held robot contact transfer."""

from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_shared import encode, decode

MASKS = {
    "all": [1, 1, 1, 1],
    "a": [0, 1, 1, 1],
    "v": [1, 0, 1, 1],
    "n": [1, 1, 0, 1],
    "w": [1, 1, 1, 0],
    "text_only": [0, 0, 0, 1],
    "audio_only": [1, 0, 0, 0],
    "video_only": [0, 1, 0, 0],
}


def records(path):
    with np.load(path, allow_pickle=False) as db:
        bank = {k: db[k] for k in ("a", "v", "n", "w")}
        names = list(map(str, db["names"]))
        lengths = db["lengths"]
    ends = np.cumsum(lengths)
    off = 0
    out = {}
    for name, end in zip(names, ends):
        out[name] = {k: v[off:end] for k, v in bank.items()}
        off = int(end)
    return out


@lru_cache(maxsize=32)
def predictor(spec):
    @jax.jit
    def f(p, data, mask):
        keep = jnp.broadcast_to(mask, (data["a"].shape[0], 4))
        z, _ = encode(p, spec, data, keep)
        return z, decode(p, z, data["u"])

    return f


def infer(p, spec, rows, mask):
    data = {k: np.stack([r[k] for r in rows]) for k in ("a", "v", "n", "w")}
    data["u"] = np.zeros(data["a"].shape[:2], np.int32)
    z, out = predictor(spec)(p, data, np.asarray(mask, bool))
    return np.asarray(z, dtype=np.float32), {k: np.asarray(v) for k, v in out.items()}


def labels(w):
    classes = np.array([1, 2, 3, *range(32, 45)])
    return (w[..., None] == classes).any(-2).astype(float)


def targets(row, key):
    if key in ("a", "v"):
        return row[key].astype(np.float32) / 127.5 - 1
    if key == "w":
        return labels(row["w"])
    return row["n"]


def rank(arr):
    x = np.concatenate(arr).astype(np.float64)
    x -= x.mean(0)
    eigen = np.linalg.eigvalsh(x.T @ x / max(1, len(x) - 1)).clip(min=0)
    return dict(
        participation_ratio=float(eigen.sum() ** 2 / max(1e-12, (eigen**2).sum())),
        dimensions_for_95pct=int(
            1 + np.searchsorted(np.cumsum(eigen[::-1]), 0.95 * eigen.sum())
        ),
        samples=len(x),
    )


def room(p, spec, source, train):
    rows = []
    shifted = []
    groups = []
    mean = {
        k: np.concatenate([targets(r, k) for r in train.values()]).mean(0)
        for k in ("a", "v", "n", "w")
    }
    for i, r in enumerate(source.values()):
        length = len(r["a"])
        for start in np.linspace(0, length - 33, 8, dtype=int):
            ix = start + np.arange(32)
            rows.append({k: v[ix] for k, v in r.items()})
            shifted.append({k: v[(ix + length // 2) % length] for k, v in r.items()})
            groups.append(i)
    groups = np.asarray(groups)
    out = {}
    states = []
    for mask in ("all", "a", "v", "n", "w", "audio_only", "video_only", "text_only"):
        tested = (
            ("a", "v")
            if mask == "text_only"
            else ("v",)
            if mask == "audio_only"
            else ("a",)
            if mask == "video_only"
            else (mask,)
        )
        for kind in ("aligned", "shifted") if mask != "all" else ("aligned",):
            vals = {k: [] for k in tested}
            base = {k: [] for k in tested}
            for i in range(0, len(rows), 32):
                gold = rows[i : i + 32]
                data = gold if kind == "aligned" else shifted[i : i + 32]
                z, pred = infer(p, spec, data, MASKS[mask])
                if mask == "all":
                    states.append(z[:, ::5].reshape(-1, spec.field))
                    continue
                for k in tested:
                    target = np.stack([targets(r, k) for r in gold])
                    if k == "w":
                        loss = (np.logaddexp(0, pred[k]) - target * pred[k]).mean(
                            (1, 2)
                        )
                        prob = np.clip(mean[k], 1e-6, 1 - 1e-6)
                        baseline = -(
                            target * np.log(prob) + (1 - target) * np.log(1 - prob)
                        ).mean((1, 2))
                    else:
                        loss = ((target - pred[k]) ** 2).mean((1, 2))
                        baseline = ((target - mean[k]) ** 2).mean((1, 2))
                    vals[k].extend(loss)
                    base[k].extend(baseline)
            if mask != "all":
                for k in tested:
                    by = lambda values: [
                        float(np.mean(np.asarray(values)[groups == i]))
                        for i in range(len(source))
                    ]
                    out[f"{mask}/{k}/{kind}"] = {
                        "loss": by(vals[k]),
                        "prior": by(base[k]),
                    }
    return dict(recordings=list(source), scores=out, rank=rank(states))


def contact_features(p, spec, source):
    features = []
    raw = []
    targets_y = []
    current = []
    groups = []
    for name, r in source.items():
        l = len(r["a"])
        zs = []
        for start in range(0, l, 32):
            ix = np.minimum(start + np.arange(32), l - 1)
            z, _ = infer(p, spec, [{k: v[ix] for k, v in r.items()}], MASKS["n"])
            zs.append(z[0, : min(32, l - start)])
        features.append(np.concatenate(zs)[:-1])
        # A four-frame raw ridge control has the same sources and receives the recorded action word.
        base = np.concatenate(
            [
                r["a"].astype(np.float32) / 127.5 - 1,
                r["v"].astype(np.float32) / 127.5 - 1,
                np.eye(48, dtype=np.float32)[r["w"]].sum(1),
            ],
            1,
        )
        raw.append(
            np.concatenate(
                [base[np.maximum(0, np.arange(l - 1) - lag)] for lag in range(4)],
                axis=1,
            )
        )
        targets_y.append(10 * r["n"][1:, :3])
        current.append(10 * r["n"][:-1, :3])
        groups.extend([name.split("/")[0]] * (l - 1))
    return tuple(
        np.concatenate(part) for part in (features, raw, targets_y, current)
    ), np.array(groups)


def ridge(x, y, xh):
    mu = x.mean(0)
    std = np.maximum(0.03, x.std(0))
    x = (x - mu) / std
    xh = (xh - mu) / std
    scale = np.sqrt(np.maximum(1.0, np.mean((x * x).sum(1))))
    x = x / scale
    xh = xh / scale
    intercept = y.mean(0)
    return (xh @ x.T) @ np.linalg.solve(
        x @ x.T + np.eye(len(x)), y - intercept
    ) + intercept


def robot(p, spec, cal, held):
    (x, raw, y, _), _ = contact_features(p, spec, cal)
    (xh, rh, yh, now), groups = contact_features(p, spec, held)
    values = {
        "frozen": ridge(x, y, xh),
        "raw": ridge(raw, y, rh),
        "constant": np.broadcast_to(y.mean(0), yh.shape),
        "force_persistence": now,
    }
    return dict(
        scenes=sorted(set(groups)),
        calibration_ticks=len(y),
        held_ticks=len(yh),
        rmse_newtons={
            k: [
                float(np.sqrt(np.mean((pred[groups == s] - yh[groups == s]) ** 2)))
                for s in sorted(set(groups))
            ]
            for k, pred in values.items()
        },
    )
