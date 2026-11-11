"""Parameter-matched instruction adapters and prediction heads on a shared 128-D context field."""

import jax
import jax.numpy as jnp
import numpy as np
from models.g9_shared import encode

RANK = 16
VERBS = 64
NOUNS = 256


def initialize(s, seed):
    rng = np.random.default_rng(seed + 71039)

    def normal(a, b, scale=1.0):
        return jnp.asarray(
            (rng.standard_normal((a, b)) * scale / (a**0.5)).astype("f4")
        )

    d = s.d
    z = s.field
    p = {
        "ac_down": normal(128, 64),
        "ac_in": normal(256, d, 0.6),
        "text_in": normal(128, d, 1.8),
        "geo_in": normal(201, d, 0.3),
        "verb": normal(z, VERBS, 0.12),
        "noun": normal(z, NOUNS, 0.12),
        "semantic": normal(z, 128, 0.35),
        "effect": normal(z, 4, 0.05),
        "vb": jnp.zeros(VERBS),
        "nb": jnp.zeros(NOUNS),
        "eb": jnp.asarray([2.0, -1.0, -1.0, -2.0], jnp.float32),
    }
    for key, ins, outs in (
        ("qkv", d, 3 * d),
        ("o", d, d),
        ("ff", d, 4 * d),
        ("out", 4 * d, d),
        ("emit", d, z),
    ):
        p["a_" + key] = normal(ins, RANK)
        p["b_" + key] = jnp.zeros((RANK, outs))
    return p


def materialize(base, delta, spec):
    p = dict(base)
    for k in ("qkv", "o", "ff", "out", "emit"):
        name = "emit" if k == "emit" else f"{spec.layers-1}_{k}"
        p[name] = base[name] + delta["a_" + k] @ delta["b_" + k]
    return p


def carriers(r, mode):
    a = r["ac"]
    w = r["aw"]
    if mode == "wrong":
        a, w = r["wc"], r["ww"]
    elif mode == "none":
        a, w = jnp.zeros_like(a), jnp.zeros_like(w)
    elif mode == "acoustic":
        w = jnp.zeros_like(w)
    elif mode == "asr_text":
        a = jnp.zeros_like(a)
    elif mode == "human":
        w = r["ah"]
    elif mode == "wrong_human":
        a, w = r["wc"], r["wh"]
    elif mode == "small":
        w = r["small_words"]
    elif mode == "wrong_small":
        a, w = r["wc"], r["wrong_small"]
    elif mode != "correct":
        raise ValueError(mode)
    return a, w


def predict(base, delta, spec, r, mode):
    dtype = getattr(jnp, spec.dtype)
    ac, txt = carriers(r, mode)
    speech = jax.nn.gelu(ac.astype(dtype) @ delta["ac_down"].astype(dtype))
    speech = speech.reshape(len(ac), 4, 4, 64).mean(2).reshape(len(ac), 256)
    conditioned = speech @ delta["ac_in"].astype(dtype) + (
        txt.astype(dtype) / 127
    ) @ delta["text_in"].astype(dtype)
    conditioned = conditioned + r["geometry"].astype(dtype) @ delta["geo_in"].astype(
        dtype
    )
    ticks = r["a"].shape[1]
    edges = jnp.zeros((len(ac), ticks, spec.d), dtype).at[:, -1].set(conditioned)
    stream = {
        "a": r["a"],
        "v": r["v"],
        "n": r["n"],
        "u": jnp.zeros((len(ac), ticks), jnp.int32),
        "w": jnp.zeros((len(ac), ticks, 8), jnp.uint8),
    }
    keep = jnp.broadcast_to(jnp.array([1, 1, 1, 0], bool), (len(ac), 4))
    z = encode(
        materialize(base, delta, spec),
        spec,
        stream,
        keep,
        window=512,
        edge_residual=edges,
    )[0][:, -1]

    def head(k):
        return (z @ delta[k].astype(dtype)).astype(jnp.float32)

    fx = head("effect") + delta["eb"]
    fractions = jax.nn.softmax(fx[:, :3], axis=-1)
    sem = head("semantic")
    sem = sem * jax.lax.rsqrt(jnp.maximum(jnp.sum(sem * sem, -1, keepdims=True), 1e-6))
    return dict(
        state=z.astype(jnp.float32),
        verb=head("verb") + delta["vb"],
        noun=head("noun") + delta["nb"],
        semantic=sem,
        effect=jnp.concatenate([fractions[:, 1:], jax.nn.sigmoid(fx[:, 3:4])], -1),
    )


def loss(delta, base, spec, r, mode, nv, nn, semantic, scale, use_effect):
    out = predict(base, delta, spec, r, mode)

    def ce(logits, label):
        ok = label >= 0
        val = -jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1), jnp.maximum(label, 0)[:, None], -1
        )[:, 0]
        return (val * ok).sum() / jnp.maximum(ok.sum(), 1)

    verb = ce(out["verb"][:, :nv], r["verb"])
    noun = ce(out["noun"][:, :nn], r["noun"])
    target = semantic[jnp.maximum(r["open_noun"], 0)]
    target = target * jax.lax.rsqrt(
        jnp.maximum((target**2).sum(-1, keepdims=True), 1e-6)
    )
    known = r["open_noun"] >= 0
    sem = ((1 - (target * out["semantic"]).sum(-1)) * known).sum() / jnp.maximum(
        known.sum(), 1
    )
    err = (out["effect"] - r["effect"]) / scale
    ae = jnp.abs(err)
    smooth = jnp.where(ae < 0.5, err**2, 0.5 * ae * 2 - 0.25).mean(1)
    physical = (smooth * r["valid"]).sum() / jnp.maximum(r["valid"].sum(), 1)
    total = verb + 0.35 * noun + 0.15 * sem + (0.6 * physical if use_effect else 0.0)
    return total, jnp.stack([verb, noun, sem, physical])
