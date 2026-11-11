"""Matched causal processors on a shared signal field, with prospective rollouts and an output gate."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

CORES = ("gpt2", "gated", "oscillator", "associative")
MODES = (
    "correct",
    "wrong",
    "none",
    "acoustic",
    "text",
    "human",
    "wrong_human",
    "speech_only",
    "no_vision",
    "no_imu",
)


@dataclass(frozen=True)
class Spec:
    scene: int
    words: int
    verbs: int
    width: int = 128
    layers: int = 4
    latent: int = 128
    heads: int = 4
    dtype: str = "bfloat16"


def initial(s, seed):
    rng = np.random.default_rng(seed)

    def matrix(d, k, gain=1):
        return (rng.standard_normal((d, k)) * gain / np.sqrt(d)).astype("f4")

    p = dict(
        scene=matrix(s.scene, s.width),
        ac=matrix(128, s.width),
        words=matrix(s.words, s.width, 2.0),
        position=matrix(12, s.width, 0.13),
        types=matrix(2, s.width, 0.1),
        emit=matrix(s.width, s.latent),
        rollout=matrix(s.latent, s.latent, 0.5),
        horizon=matrix(3, s.latent, 0.18),
        verb=matrix(s.latent, s.verbs, 0.15),
        vb=np.zeros(s.verbs, "f4"),
        change=matrix(s.latent, 1, 0.1),
        cb=np.zeros(1, "f4"),
        certainty=matrix(s.latent, 2, 0.1),
        ub=np.zeros(2, "f4"),
    )
    for i in range(s.layers):
        for key, ind, out, g in (
            ("qkv", s.width, 3 * s.width, 1.0),
            ("o", s.width, s.width, 0.6),
            ("ff", s.width, 4 * s.width, 1.0),
            ("down", 4 * s.width, s.width, 0.6),
        ):
            p[f"{i}_{key}"] = matrix(ind, out, g)
        for key in ("n1", "n2"):
            p[f"{i}_{key}s"] = np.ones(s.width, "f4")
            p[f"{i}_{key}b"] = np.zeros(s.width, "f4")
    return jax.tree.map(jnp.asarray, p)


def norm(x):
    xf = x.astype("float32")
    c = xf - jnp.mean(xf, -1, keepdims=True)
    return (c * jax.lax.rsqrt(jnp.mean(c * c, -1, keepdims=True) + 1e-5)).astype(
        x.dtype
    )


def mix(core, q, k, v, heads):
    b, t, d = q.shape
    if core in ("gpt2", "associative"):

        def split(x):
            return x.reshape(b, t, heads, d // heads).transpose(0, 2, 1, 3)

        q, k, v = map(split, (q, k, v))
        age = jnp.arange(t)[:, None] - jnp.arange(t)[None, :]
        if core == "gpt2":
            dots = (q @ k.swapaxes(-2, -1)).astype("float32") / (d // heads) ** 0.5
            weight = jax.nn.softmax(jnp.where(age >= 0, dots, -jnp.inf), axis=-1)
        else:
            # Equivalent to a decayed outer-product memory and a matching key normalizer.
            qp = jax.nn.elu(q.astype("float32")) + 1.0
            kp = jax.nn.elu(k.astype("float32")) + 1.0
            rates = jnp.asarray(np.geomspace(0.52, 0.995, heads), jnp.float32)
            discount = jnp.where(
                age >= 0, rates[:, None, None] ** jnp.maximum(age, 0), 0.0
            )
            weight = (qp @ kp.swapaxes(-2, -1)) * discount
            weight = weight / jnp.maximum(weight.sum(-1, keepdims=True), 1e-7)
        y = (weight.astype(v.dtype) @ v).transpose(0, 2, 1, 3).reshape(b, t, d)
        return y
    qt, kt, vt = [x.swapaxes(0, 1).astype("float32") for x in (q, k, v)]
    if core == "gated":

        def step(h, inputs):
            read, forget, value = inputs
            keep = jax.nn.sigmoid(forget)
            h = keep * h + (1.0 - keep) * value
            return h, jax.nn.sigmoid(read) * h

    elif core == "oscillator":
        periods = jnp.asarray(np.geomspace(2.5, 64, d // 2), jnp.float32)
        base = jnp.asarray(np.linspace(-0.5, 2.8, d // 2), jnp.float32)

        def step(h, inputs):
            read, forget, value = inputs
            pair = h.reshape(b, d // 2, 2)
            keep = jax.nn.sigmoid(base + 0.5 * (forget[:, 0::2] + forget[:, 1::2]))
            phase = (2 * jnp.pi / periods) * (1.0 + 0.30 * jnp.tanh(read[:, 0::2]))
            co, si = jnp.cos(phase), jnp.sin(phase)
            rot = jnp.stack(
                (
                    co * pair[:, :, 0] - si * pair[:, :, 1],
                    si * pair[:, :, 0] + co * pair[:, :, 1],
                ),
                -1,
            )
            h = (
                keep[:, :, None] * rot
                + (1.0 - keep[:, :, None]) * value.reshape(b, d // 2, 2)
            ).reshape(b, d)
            return h, jax.nn.sigmoid(read) * h

    else:
        raise ValueError(core)
    return (
        jax.lax.scan(step, jnp.zeros((b, d), jnp.float32), (qt, kt, vt))[1]
        .swapaxes(0, 1)
        .astype(q.dtype)
    )


def inputs(row, mode):
    if mode not in MODES:
        raise ValueError(mode)
    ac = row["wc"] if mode in ("wrong", "wrong_human") else row["ac"]
    words = {"wrong": "wrong", "wrong_human": "wrong_human", "human": "human"}.get(
        mode, "words"
    )
    words = row[words]
    scene = row["scene"]
    if mode in ("none", "acoustic"):
        words = jnp.zeros_like(words)
    if mode in ("none", "text"):
        ac = jnp.zeros_like(ac)
    if mode == "speech_only":
        scene = jnp.zeros_like(scene)
    if mode == "no_vision":
        scene = (
            scene.at[:, :, :512].set(0).at[:, :, 583:631].set(0).at[:, :, 631].set(0)
        )
    if mode == "no_imu":
        scene = scene.at[:, :, 576:583].set(0)
    return scene, ac, words


def proposals(p, s, core, row, mode="correct"):
    dt = getattr(jnp, s.dtype)

    def lin(x, key):
        return x.astype(dt) @ p[key].astype(dt)

    scene, ac, words = inputs(row, mode)
    field = jnp.concatenate(
        (
            lin(scene, "scene") + p["types"][0].astype(dt),
            lin(ac, "ac") + 0.95 * lin(words, "words") + p["types"][1].astype(dt),
        ),
        axis=1,
    )
    x = field + p["position"].astype(dt)
    for i in range(s.layers):
        pre = norm(x) * p[f"{i}_n1s"].astype(dt) + p[f"{i}_n1b"].astype(dt)
        q, k, v = jnp.split(lin(pre, f"{i}_qkv"), 3, -1)
        x = x + lin(mix(core, q, k, v, s.heads), f"{i}_o")
        pre = norm(x) * p[f"{i}_n2s"].astype(dt) + p[f"{i}_n2b"].astype(dt)
        x = x + lin(jax.nn.gelu(lin(pre, f"{i}_ff")), f"{i}_down")
    field = norm(lin(norm(x), "emit"))
    z = field[:, -1]
    positions = []
    for clock in range(3):
        z = norm(z + 0.5 * jnp.tanh(lin(z, "rollout") + p["horizon"][clock].astype(dt)))
        positions.append(z)
    imagined = jnp.stack(positions, axis=1)
    verb = lin(imagined, "verb").astype(jnp.float32) + p["vb"]
    state = lin(z, "change").astype(jnp.float32)[:, 0] + p["cb"][0]
    uncertainty = lin(z, "certainty").astype(jnp.float32) + p["ub"]
    return dict(
        verb=verb,
        state=state,
        certainty=uncertainty,
        field=field[:, -1].astype("float32"),
    )


def sequence_nll(logits, y):
    known = y >= 0
    lp = jax.nn.log_softmax(logits, -1)
    picked = jnp.take_along_axis(lp, jnp.maximum(y, 0)[:, :, None], axis=-1)[:, :, 0]
    return -(picked * known).sum(1) / jnp.maximum(known.sum(1), 1)


def logistic_loss(logit, target):
    return jax.nn.softplus(logit) - target * logit


def objective(p, s, core, row, treatment, joint):
    mode = "wrong" if treatment == "deranged" else "correct"
    out = proposals(p, s, core, row, mode)
    y = row["y"]
    nll = sequence_nll(out["verb"], y)
    valid = row["state"] >= 0
    state_y = jnp.maximum(row["state"], 0)
    st = (logistic_loss(out["state"], state_y) * valid).sum() / jnp.maximum(
        valid.sum(), 1
    )
    acc = ((jnp.argmax(out["verb"], -1) == y) & (y >= 0)).mean(-1)
    uc = logistic_loss(out["certainty"][:, 0], jax.lax.stop_gradient(acc)).mean()
    state_correct = (out["state"] >= 0) == (state_y >= 0.5)
    sc = (
        logistic_loss(
            out["certainty"][:, 1],
            jax.lax.stop_gradient(state_correct.astype("float32")),
        )
        * valid
    ).sum() / jnp.maximum(valid.sum(), 1)
    loss = nll.mean() + 0.10 * uc + (0.50 * st + 0.10 * sc if joint else 0.0)
    if treatment == "grounded":
        alt = sequence_nll(proposals(p, s, core, row, "wrong")["verb"], y)
        eligible = row["groundable"]
        margin = (jax.nn.softplus(0.35 + nll - alt) * eligible).sum() / jnp.maximum(
            eligible.sum(), 1
        )
        loss += 0.25 * margin
    return loss, jnp.stack([nll.mean(), st, uc, sc])


def route(verb_logits, state_probability, confidences, thresholds, vocabulary):
    """Decode accepted proposals only. A rejected pair leaves the world model in internal-only mode."""
    record = {}
    if confidences[0] >= thresholds[0]:
        record["actions"] = [
            vocabulary[int(i)] for i in np.argmax(verb_logits, axis=-1)
        ]
    if confidences[1] >= thresholds[1]:
        record["device_change"] = bool(state_probability >= 0.5)
        record["change_probability"] = float(state_probability)
    return dict(
        route="+".join(
            record_key
            for record_key in ("actions", "device_change")
            if record_key in record
        )
        or "internal_only",
        **record,
    )
