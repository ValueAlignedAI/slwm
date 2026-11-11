"""Matched dense and spatial contact processors with a portable NumPy/JAX forward path."""

import numpy as np

CORES = ("gpt2", "transport", "gated", "dense_spatial")
ARMS = ("direct", "protected")


def initialize(dim, seed, width=80, layers=2):
    rng = np.random.default_rng(seed)

    def mat(a, b, g=1):
        return (rng.standard_normal((a, b)) * g / np.sqrt(a)).astype("f4")

    s, im, f = dim
    p = dict(
        in_slot=mat(s, width),
        in_img=mat(im, width, 0.35),
        time=mat(4, width, 0.1),
        slot=mat(5, width, 0.1),
        emit=mat(width, 128),
        field=mat(128, width),
        candidate=mat(f, width, 0.7),
        response=mat(11, width, 0.4),
        next=mat(width, 11, 0.08),
        risk=mat(width, 1, 0.08),
        bias=np.zeros(11, "f4"),
    )
    for i in range(layers):
        for k, a, b in (
            ("qkv", width, 3 * width),
            ("out", width, width),
            ("ff", width, 3 * width),
            ("down", 3 * width, width),
        ):
            p[f"{i}_{k}"] = mat(a, b, 0.65 if k in ("out", "down") else 1)
    return p


def norm(x, xp):
    c = x - xp.mean(x, axis=-1, keepdims=True)
    return c / xp.sqrt(xp.mean(c * c, axis=-1, keepdims=True) + 1e-5)


def sigmoid(x, xp):
    return 1.0 / (1.0 + xp.exp(-xp.clip(x, -25.0, 25.0)))


def softmax(x, xp):
    e = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
    return e / xp.sum(e, axis=-1, keepdims=True)


def core_mix(q, k, v, pos, core, xp):
    b, t, o, d = q.shape
    if core == "gated":
        h = xp.zeros_like(q[:, 0])
        out = []
        for j in range(t):
            keep = sigmoid(k[:, j], xp)
            h = keep * h + (1 - keep) * v[:, j]
            out.append(sigmoid(q[:, j], xp) * h)
        return xp.stack(out, axis=1)
    heads = 4

    def split(x):
        return x.reshape(b, t * o, heads, d // heads).transpose(0, 2, 1, 3)

    q, k, v = map(split, (q, k, v))
    clock = xp.repeat(xp.arange(t), o)
    age = clock[:, None] - clock[None, :]
    bias = xp.zeros((1, 1, 1, 1), dtype=q.dtype)
    if core in ("transport", "dense_spatial"):
        locations = pos.reshape(b, t * o, 2)
        dist = xp.sum((locations[:, :, None] - locations[:, None, :]) ** 2, axis=-1)
        scale = xp.asarray([0.20, 0.45, 0.85, 3.0], dtype=q.dtype)
        bias = -0.5 * dist[:, None, :, :] / (scale[None, :, None, None] ** 2)
    if core in ("gpt2", "dense_spatial"):
        scores = (q @ k.swapaxes(-1, -2)) / np.sqrt(d // heads) + bias
        prob = softmax(xp.where(age >= 0, scores, -1e5), xp)
    elif core == "transport":
        qp = xp.where(q >= 0, q + 1, xp.exp(xp.minimum(q, 0)))
        kp = xp.where(k >= 0, k + 1, xp.exp(xp.minimum(k, 0)))
        rate = xp.asarray([0.55, 0.78, 0.93, 0.99], dtype=q.dtype)
        weight = xp.where(
            age >= 0,
            (qp @ kp.swapaxes(-1, -2))
            * xp.exp(bias)
            * rate[None, :, None, None] ** xp.maximum(age, 0),
            0,
        )
        prob = weight / xp.maximum(weight.sum(-1, keepdims=True), 1e-7)
    else:
        raise ValueError(core)
    return (prob @ v).transpose(0, 2, 1, 3).reshape(b, t, o, d)


def predict(p, r, core, arm, xp=np, include_long=False, mean_field=None):
    h = (
        r["slots"] @ p["in_slot"]
        + (r["image"] @ p["in_img"])[:, :, None, :]
        + p["time"][None, :, None, :]
        + p["slot"][None, None, :, :]
    )
    for i in range(2):
        q, k, v = xp.split(norm(h, xp) @ p[f"{i}_qkv"], 3, axis=-1)
        h = h + core_mix(q, k, v, r["location"], core, xp) @ p[f"{i}_out"]
        u = norm(h, xp) @ p[f"{i}_ff"]
        h = h + (u * sigmoid(u, xp)) @ p[f"{i}_down"]
    slots = norm(norm(h[:, -1], xp) @ p["emit"], xp)
    focus = xp.sum(slots * r["focus"][:, :, None], axis=1)
    z = norm(0.4 * slots[:, -1] + 0.4 * focus + 0.2 * slots.mean(1), xp)
    if arm == "protected":
        z = xp.concatenate((r["raw_action"], z[:, :126]), axis=1)
    elif arm != "direct":
        raise ValueError(arm)
    if mean_field is not None:
        if arm != "protected":
            raise ValueError("Mean-field intervention requires the protected topology")
        z = xp.concatenate(
            (z[:, :2], xp.broadcast_to(mean_field, z[:, 2:].shape)), axis=1
        )
    proposal = xp.tanh(
        (z @ p["field"])[:, None, :]
        + r["phi"] @ p["candidate"]
        + r["prior"] @ p["response"]
    )
    y = proposal @ p["next"] + p["bias"]
    if arm == "protected":
        y = r["prior"] + 0.45 * y
    risk = sigmoid((proposal @ p["risk"])[:, :, 0], xp)
    if include_long:
        value = r["long_prior"] + 0.65 * (proposal @ p["continuation"])[:, :, 0]
        return y, risk, z, value
    return y, risk, z


def objective(p, r, core, arm):
    import jax
    import jax.numpy as j

    long = "continuation" in p
    if long:
        y, risk, _, value = predict(p, r, core, arm, j, True)
    else:
        y, risk, _ = predict(p, r, core, arm, j)
    delta = y - r["y"]
    ae = j.abs(delta)
    smooth = j.where(ae < 0.35, 0.5 * delta * delta, 0.35 * (ae - 0.175))
    actual = r["y"][:, :, -1]
    score = y[:, :, -1]
    soft = jax.nn.softmax(actual / 0.16, axis=1)
    ranking = -j.sum(soft * jax.nn.log_softmax(score / 0.16, axis=1), 1).mean()
    target = j.argmax(r["focus"], 1) + 1
    effects = smooth[:, :, :10].reshape(len(y), y.shape[1], 5, 2)
    obj = j.take_along_axis(effects, target[:, None, None, None], axis=2).mean()
    unreliable = jax.lax.stop_gradient((ae[:, :, -1] > 0.15).astype("f4"))
    brier = j.mean((risk - unreliable) ** 2)
    loss = (
        2.3 * smooth[:, :, -1].mean()
        + effects.mean()
        + 0.8 * obj
        + 0.15 * ranking
        + 0.06 * brier
    )
    if long:
        valid = r["has_long"]
        v = value - value[:, 1:2]
        target = r["long"]
        d = j.abs(v - target)
        term = j.where(d < 0.4, 0.5 * d * d, 0.4 * (d - 0.2)).mean(1)
        pair = -(
            jax.nn.softmax(target / 0.24, axis=1) * jax.nn.log_softmax(v / 0.24, axis=1)
        ).sum(1)
        loss += 2.2 * ((term + 0.12 * pair) * valid).sum() / j.maximum(valid.sum(), 1)
    return loss, j.stack([smooth[:, :, -1].mean(), effects.mean(), obj, ranking, brier])
