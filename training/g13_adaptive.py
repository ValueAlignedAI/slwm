"""Input-dependent causal response and a label-smoothed object-role prior fitted on training groups."""

import numpy as np
from training.g13_physics import design, group_mean


def regressors(kin, plan, h, kind):
    a = design(kin, plan, h)
    b = len(a)
    vel = kin[:, :, 2:4] / 0.05
    old = kin[:, :, 4:6] / 0.05
    cmd = np.concatenate((old, plan[:, :h] / 0.05), 1)
    # Past realized response measures the local delay without observing any future state.
    gain = np.sum(vel * old, axis=(1, 2)) / (0.3 + (old**2).sum((1, 2)))
    prev = np.sum(vel[:, 1:] * old[:, :-1], axis=(1, 2)) / (
        0.3 + (old[:, :-1] ** 2).sum((1, 2))
    )
    activity = np.sqrt((vel**2).mean((1, 2)))
    velocity = np.linalg.norm(vel[:, -1], axis=-1)
    signal = np.sqrt((cmd**2).sum(2, keepdims=True))
    dcmd = np.diff(cmd, axis=1, prepend=cmd[:, :1])
    future_center = kin[:, -1, :2] + (plan[:, :h].sum(1) if h else 0)
    support = (
        np.stack(
            (
                np.maximum(0.21 - future_center[:, 0], 0),
                np.maximum(future_center[:, 0] - 0.59, 0),
                np.maximum(np.abs(future_center[:, 1]) - 0.27, 0),
            ),
            1,
        )
        / 0.05
    )
    terms = [a]
    if kind in ("response", "combined"):
        regimes = np.stack(
            (
                np.clip(gain, -1, 2),
                np.clip(prev, -1, 2),
                np.clip(activity, 0, 2),
                np.clip(velocity, 0, 2),
            ),
            1,
        )
        terms += [(a[:, None, 3:] * regimes[:, :, None]).reshape(b, -1)]
    if kind in ("nonlinear", "combined"):
        terms += [
            (cmd * np.minimum(signal, 2)).reshape(b, -1),
            np.tanh(1.8 * cmd).reshape(b, -1),
            (dcmd * np.minimum(np.linalg.norm(dcmd, axis=2, keepdims=True), 2)).reshape(
                b, -1
            ),
            support,
            np.maximum(future_center[:, 0, None] - np.array([0.28, 0.4, 0.52]), 0)
            / 0.2,
            np.maximum(future_center[:, 1, None] - np.array([-0.18, 0, 0.18]), 0) / 0.2,
        ]
    return np.concatenate(terms, axis=1).astype("f8")


def fit_basis(train, dev):
    _, inv, n = np.unique(train["group"], return_inverse=True, return_counts=True)
    w = 1.0 / n[inv]
    w /= w.sum()
    candidates = []
    coefs = {}
    for kind in ("linear", "response", "nonlinear", "combined"):
        X = regressors(train["kin"], train["plan"], 4, kind)
        D = regressors(dev["kin"], dev["plan"], 4, kind)
        mat = X.T @ (X * w[:, None])
        xy = X.T @ (train["future"][:, -1] * w[:, None])
        for penalty in (0.00003, 0.0003, 0.002):
            theta = np.linalg.solve(mat + penalty * np.eye(mat.shape[0]), xy)
            err = np.linalg.norm(D @ theta - dev["future"][:, -1], axis=1) * 1000
            candidates.append(
                dict(kind=kind, penalty=penalty, dev_mm=group_mean(err, dev["group"]))
            )
    best = min(candidates, key=lambda r: (r["dev_mm"], r["kind"], r["penalty"]))
    for h in range(1, 5):
        X = regressors(train["kin"], train["plan"], h, best["kind"])
        coefs[str(h)] = np.linalg.solve(
            X.T @ (X * w[:, None]) + best["penalty"] * np.eye(X.shape[1]),
            X.T @ (train["future"][:, h - 1] * w[:, None]),
        ).astype("f4")
    return best, candidates, coefs


def response(kin, plan, kind, coefs):
    return np.stack(
        [regressors(kin, plan, h, kind) @ coefs[str(h)] for h in range(1, 5)], 1
    ).astype("f4")


def role_prior(train):
    # Within a grammar pattern the same source/destination/other categories share evidence across colors.
    # Role zero is the no-change outcome. Other colors share the remaining mass.
    mass = np.ones((2, 3, 4), np.float64) * 0.5
    ix = np.flatnonzero(train["mover"] >= 0)
    _, inv, n = np.unique(train["group"][ix], return_inverse=True, return_counts=True)
    for i, w in zip(ix, 1.0 / n[inv]):
        b = train["bind"][i]
        arm = int(b[:, 3].max() > 0.5)
        cnt = min(2, int(b[:, 2].sum()))
        y = int(train["mover"][i])
        r = 0 if y == 0 else 1 if b[y - 1, 0] > 0.5 else 2 if b[y - 1, 1] > 0.5 else 3
        mass[arm, cnt, r] += w
    return mass


def apply_prior(bind, mass):
    arm = (bind[:, :, 3].max(1) > 0.5).astype(int)
    cnt = np.minimum(2, bind[:, :, 2].sum(1).astype(int))
    m = mass[arm, cnt]
    role = np.where(bind[:, :, 0] > 0.5, 1, np.where(bind[:, :, 1] > 0.5, 2, 3))
    probs = np.zeros((len(bind), 5), np.float32)
    probs[:, 0] = m[:, 0]
    for k in (1, 2, 3):
        mask = role == k
        probs[:, 1:] += mask * (m[:, k] / np.maximum(mask.sum(1), 1))[:, None]
    probs /= probs.sum(1, keepdims=True)
    return probs


def decorate(rows, best, coefs, mass):
    rows["adapt_active"] = response(rows["kin"], rows["plan"], best["kind"], coefs)
    rows["adapt_wrong"] = response(rows["kin"], rows["alt_plan"], best["kind"], coefs)
    rows["adapt_zero"] = response(
        rows["kin"], np.zeros_like(rows["plan"]), best["kind"], coefs
    )
    reset = np.broadcast_to(rows["kin"][:, -1:], rows["kin"].shape).copy()
    reset[:, :, 2:] = 0
    rows["adapt_reset"] = response(reset, rows["plan"], best["kind"], coefs)
    for key, b in (
        ("p_goal", rows["bind"]),
        ("p_wrong", rows["wb"]),
        ("p_none", np.zeros_like(rows["bind"])),
    ):
        rows[key] = apply_prior(b, mass)
    return rows
