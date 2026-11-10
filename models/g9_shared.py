"""Concurrent causal world models with a single variable-width emitted state."""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from models.g4_shared import norm
from models.g5_shared import attention, retention
from models.g9_tabletop import ACTIONS, AUDIO, VISION, PHYSICS, WORDS, tag_targets


@dataclass(frozen=True)
class Spec:
    core: str
    d: int
    layers: int
    field: int
    act: str
    dtype: str = "bfloat16"


def parameters(s, seed):
    rng = np.random.default_rng(seed)
    d, f = s.d, s.field

    def w(m, n, scale=1.0):
        return (
            rng.standard_normal((m, n), dtype=np.float32) * (scale / np.sqrt(m))
        ).astype(np.float32)

    p = dict(
        audio=w(AUDIO, d),
        image=w(VISION, d),
        measure=w(PHYSICS, d),
        word=w(WORDS, d, 0.25),
        control=w(ACTIONS, d, 0.3),
        mask=w(4, d, 0.1),
        clock=w(8, d, 0.2),
        norm=np.ones(d, np.float32),
        bias=np.zeros(d, np.float32),
    )
    for i in range(s.layers):
        k = str(i) + "_"
        for name, m, n, scale in (
            ("qkv", d, d * 3, 1),
            ("o", d, d, (2 * s.layers) ** -0.5),
            ("ff", d, 4 * d, 1),
            ("out", 4 * d, d, (2 * s.layers) ** -0.5),
        ):
            p[k + name] = w(m, n, scale)
        for label in ("n1", "n2"):
            p[k + label] = np.ones(d, np.float32)
            p[k + label + "b"] = np.zeros(d, np.float32)
    # Common body and first 128 projection columns have identical starts across field widths.
    p["emit"] = w(d, 512)[:, :f].copy()
    p["read"] = w(512, AUDIO + VISION + PHYSICS + 16 + ACTIONS)[:f].copy() * np.sqrt(
        512 / f
    )
    p["future"] = w(512, AUDIO + PHYSICS)[:f].copy() * np.sqrt(512 / f)
    p["query"] = w(ACTIONS, 512, 0.25)[:, :f].copy()
    return jax.tree.map(jnp.asarray, p)


def gated(q, k, v, valid, previous):
    def update(h, t):
        value, gate, out, ok = t
        decay = jax.nn.sigmoid(gate.astype(jnp.float32) + 1)
        candidate = decay * h + (1 - decay) * jnp.tanh(value.astype(jnp.float32))
        h = jnp.where(ok[:, None], candidate, h)
        return h, h.astype(q.dtype) * jax.nn.sigmoid(out)

    initial = (
        jnp.zeros_like(q[:, 0], dtype=jnp.float32) if previous is None else previous
    )
    last, seq = jax.lax.scan(
        update, initial, (q.swapaxes(0, 1), k.swapaxes(0, 1), v.swapaxes(0, 1), valid.T)
    )
    return seq.swapaxes(0, 1), last


def extended_attention(q, k, v, valid, previous, window):
    b, t, d = q.shape
    q, k, v = (a.reshape(b, t, d // 64, 64) for a in (q, k, v))
    if previous is not None:
        k = jnp.concatenate((previous[0], k), 1)
        v = jnp.concatenate((previous[1], v), 1)
        valid = jnp.concatenate((previous[2], valid), 1)
    distance = jnp.arange(t)[:, None] + k.shape[1] - t - jnp.arange(k.shape[1])[None, :]
    allowed = (distance >= 0) & (distance < window)
    score = jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32) / 8
    prob = jax.nn.softmax(
        jnp.where(allowed[None, None] & valid[:, None, None], score, -1e9), axis=-1
    ).astype(q.dtype)
    return jnp.einsum("bhts,bshd->bthd", prob, v).reshape(b, t, d), (
        k[:, -window:],
        v[:, -window:],
        valid[:, -window:],
    )


def encode(p, s, r, keep, *, memory=None, offset=0, window=64):
    dtype = getattr(jnp, s.dtype)
    dim = r["u"].shape
    flags = keep.astype(dtype)

    def project(data, kind, div, i):
        return (
            (data.astype(dtype) / div - (1 if i < 2 else 0)) @ p[kind].astype(dtype)
        ) * flags[:, i, None, None]

    x = (
        project(r["a"], "audio", 127.5, 0)
        + project(r["v"], "image", 127.5, 1)
        + project(r["n"], "measure", 1.0, 2)
    )
    words = r["w"].astype(jnp.int32)
    text = (p["word"][words].astype(dtype) * (words > 0)[..., None]).sum(-2)
    text = text * jax.lax.rsqrt(
        jnp.maximum((words > 0).sum(-1, keepdims=True), 1).astype(jnp.float32)
    ).astype(dtype)
    x = (x + text * flags[:, 3, None, None]) * jax.lax.rsqrt(
        jnp.maximum(flags.sum(-1), 1)
    )[:, None, None]
    phase = (
        (jnp.arange(dim[1]) + offset)[:, None]
        * jnp.array([1.0, 2.0, 4.0, 8.0])
        * (2 * jnp.pi / 32)
    )
    x = (
        x
        + (
            jnp.concatenate([jnp.sin(phase), jnp.cos(phase)], -1).astype(dtype)
            @ p["clock"].astype(dtype)
        )[None]
    )
    x = (
        x
        + flags[:, None] @ p["mask"].astype(dtype)
        + p["control"][r["u"]].astype(dtype)
    )
    valid = jnp.ones(dim, bool)
    memories = []
    for i in range(s.layers):
        k = str(i) + "_"
        z = norm(x, p[k + "n1"], p[k + "n1b"], dtype)
        q, key, v = jnp.split(z @ p[k + "qkv"].astype(dtype), 3, -1)
        op = (
            gated
            if s.core == "gated"
            else attention
            if s.core == "gpt2" or (i + 1) % 4 == 0
            else retention
        )
        previous = None if memory is None else memory[i]
        if op is attention and window != 64:
            mix, carry = extended_attention(q, key, v, valid, previous, window)
        else:
            mix, carry = op(q, key, v, valid, previous)
        memories.append(carry)
        x = x + mix @ p[k + "o"].astype(dtype)
        z = norm(x, p[k + "n2"], p[k + "n2b"], dtype) @ p[k + "ff"].astype(dtype)
        z = (
            jax.nn.gelu(z)
            if s.act == "gelu"
            else z + jnp.square(jnp.sin(z.astype(jnp.float32))).astype(dtype)
        )
        x = x + z @ p[k + "out"].astype(dtype)
    z = (norm(x, p["norm"], p["bias"], dtype) @ p["emit"].astype(dtype)).astype(
        jnp.float32
    )
    z = z * jax.lax.rsqrt(jnp.mean(z * z, axis=-1, keepdims=True) + 1e-5)
    return z.astype(dtype), tuple(memories)


def decode(p, z, plan):
    raw = (z @ p["read"].astype(z.dtype)).astype(jnp.float32)
    a, v, n, w, policy = jnp.split(
        raw,
        [
            AUDIO,
            AUDIO + VISION,
            AUDIO + VISION + PHYSICS,
            AUDIO + VISION + PHYSICS + 16,
        ],
        -1,
    )
    future = (
        z * (1 + p["query"][plan].astype(z.dtype)) @ p["future"].astype(z.dtype)
    ).astype(jnp.float32)
    return dict(
        a=a,
        v=v,
        n=n,
        w=w,
        policy=policy,
        next_a=future[..., :AUDIO],
        next_n=future[..., AUDIO:],
    )


def decision_scores(z, read, head):
    base = (z @ read.astype(z.dtype)).astype(jnp.float32)
    if head is None:
        return base
    x = z.astype(jnp.float32)
    return base + jax.nn.gelu(x @ head["w"] + head["b"]) @ head["a"] + head["c"]


def errors(pred, r):
    mse = lambda a, b: jnp.mean(jnp.square(a - b), -1)
    tags = tag_targets(r["w"])
    return dict(
        a=mse(pred["a"], r["a"].astype(jnp.float32) / 127.5 - 1),
        v=mse(pred["v"], r["v"].astype(jnp.float32) / 127.5 - 1),
        n=mse(pred["n"], r["n"]),
        w=(jax.nn.softplus(pred["w"]) - tags * pred["w"]).mean(-1),
        policy=-jnp.take_along_axis(
            jax.nn.log_softmax(pred["policy"]), r["policy"][..., None], -1
        )[..., 0],
        next_a=mse(pred["next_a"], r["next_a"].astype(jnp.float32) / 127.5 - 1),
        next_n=mse(pred["next_n"], r["next_n"]),
    )


def objective(p, s, r):
    z, _ = encode(p, s, r, r["keep"])
    err = errors(decode(p, z, r["plan"]), r)
    losses = jnp.stack([err[k].mean(-1) for k in ("a", "v", "n", "w")], -1)
    weights = jnp.where(r["keep"], 0.10, 1.0)
    reconstruction = (weights * losses).sum(-1) / weights.sum(-1)
    policy = (err["policy"].mean(1) * r["is_sim"]).sum() / r["is_sim"].sum()
    future = 0.5 * (err["next_n"].mean() + err["next_a"].mean())
    return policy + reconstruction.mean() + future, jnp.stack(
        [policy, reconstruction.mean(), future]
    )
