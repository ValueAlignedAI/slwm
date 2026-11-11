"""Open-vocabulary component proposals from the common multimodal field."""

import jax
import jax.numpy as jnp
import numpy as np
from models.g12_world import mix, norm


def initial(seed, d=128):
    rng = np.random.default_rng(seed)

    def mat(a, b, g=1):
        return jnp.asarray((rng.standard_normal((a, b)) * g / np.sqrt(a)).astype("f4"))

    p = dict(
        scene=mat(633, d),
        ac=mat(128, d),
        words=mat(384, d, 2),
        position=mat(12, d, 0.12),
        field=mat(d, 128),
        temperature=jnp.ones(1),
        gate=mat(128, 1, 0.1),
    )
    for i in range(3):
        for k, a, b in (
            ("qkv", d, 3 * d),
            ("o", d, d),
            ("ff", d, 4 * d),
            ("down", 4 * d, d),
        ):
            p[f"{i}_{k}"] = mat(a, b, 0.65 if k in ("down", "o") else 1)
    return p


def unit(x):
    return x * jax.lax.rsqrt(jnp.maximum((x * x).sum(-1, keepdims=True), 1e-8))


def forward(p, core, process, r, nouns, basis, prior, mode):
    dt = jnp.bfloat16

    def project(x, k):
        return x.astype(dt) @ p[k].astype(dt)

    scene = r["scene"]
    speech = r["wc"] if mode in ("wrong", "wrong_human") else r["ac"]
    text = r[
        {"wrong": "alt", "human": "oracle", "wrong_human": "alt_oracle"}.get(
            mode, "text"
        )
    ].astype("float32")
    if mode in ("none", "acoustic"):
        text = jnp.zeros_like(text)
    if mode in ("none", "words"):
        speech = jnp.zeros_like(speech)
    if mode == "no_scene":
        scene = jnp.zeros_like(scene)
    x = jnp.concatenate(
        (
            project(scene, "scene"),
            project(speech, "ac") + project(text, "words")[:, None],
        ),
        1,
    ) + p["position"].astype(dt)
    for i in range(3):
        q, k, v = jnp.split(project(norm(x), f"{i}_qkv"), 3, -1)
        x = x + project(mix(core, q, k, v, 4), f"{i}_o")
        x = x + project(jax.nn.gelu(project(norm(x), f"{i}_ff")), f"{i}_down")
    learned = unit(project(norm(x[:, -1]), "field").astype("float32"))
    field = (
        unit(0.55 * learned + unit(text @ basis)) if process == "semantic" else learned
    )
    logits = (
        6 + 10 * jax.nn.sigmoid(p["temperature"])
    ) * field @ nouns.T + 0.2 * jnp.log(prior)
    gate = (field @ p["gate"])[:, 0]
    return logits, gate


def loss(p, core, process, r, nouns, basis, prior):
    logits, gate = forward(p, core, process, r, nouns, basis, prior, "correct")
    y = r["y"]
    valid = y >= 0
    labels = (jax.nn.one_hot(jnp.maximum(y, 0), len(nouns)) * valid[:, :, None]).max(1)
    prob = jax.nn.softmax(logits, axis=1)
    has = valid.any(1)
    score = -jnp.log(jnp.maximum((prob * labels).sum(1), 1e-7))
    first = -jnp.take_along_axis(
        jax.nn.log_softmax(logits, axis=1), jnp.maximum(y[:, :1], 0), axis=1
    )[:, 0]
    outcome = (jnp.argmax(logits, 1)[:, None] == y).any(1)
    uc = (
        jax.nn.softplus(gate) - gate * jax.lax.stop_gradient(outcome.astype("f4"))
    ).mean()
    ce = (score * has).sum() / jnp.maximum(has.sum(), 1)
    return (
        ce
        + 0.3 * (first * valid[:, 0]).sum() / jnp.maximum(valid[:, 0].sum(), 1)
        + 0.1 * uc
    )
