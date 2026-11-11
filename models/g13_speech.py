"""A compact spectrogram codec producing a reusable direction field and calibrated word proposals."""

import jax
import jax.numpy as jnp
import numpy as np


def init(seed):
    rng = np.random.default_rng(seed)
    p = {}
    for i, (channels, out) in enumerate(((1, 24), (24, 40), (40, 64))):
        k = 5 if i == 0 else 3
        p["c" + str(i)] = jnp.asarray(
            (
                rng.standard_normal((k, k, channels, out))
                * np.sqrt(2 / (k * k * channels))
            ).astype("f4")
        )
        p["b" + str(i)] = jnp.zeros(out)
    p["field"] = jnp.asarray(
        (rng.standard_normal((320, 128)) / np.sqrt(320)).astype("f4")
    )
    p["word"] = jnp.asarray((rng.standard_normal((128, 4)) * 0.03).astype("f4"))
    p["bias"] = jnp.zeros(4)
    return p


def patch_convolution(x, weight):
    # Explicit TF-style SAME padding and cross-correlation keep single-clip CPU/CUDA semantics identical.
    h, w = x.shape[1:3]
    kh, kw = weight.shape[:2]
    oh, ow = (h + 1) // 2, (w + 1) // 2
    ph = max(0, (oh - 1) * 2 + kh - h)
    pw = max(0, (ow - 1) * 2 + kw - w)
    pad = jnp.pad(x, ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))
    patches = jnp.stack(
        [
            pad[:, i : i + 2 * oh : 2, j : j + 2 * ow : 2, :]
            for i in range(kh)
            for j in range(kw)
        ],
        axis=3,
    )
    flat = patches.reshape(x.shape[0], oh, ow, -1)
    kernel = weight.reshape(flat.shape[-1], -1)
    return jnp.einsum(
        "bhwc,cd->bhwd", flat, kernel, precision=jax.lax.Precision.HIGHEST
    )


def predict(p, x, portable=False):
    x = x[:, :, :, None].astype(jnp.float32)
    for i in range(3):
        weight = p["c" + str(i)]
        x = (
            patch_convolution(x, weight)
            if portable
            else jax.lax.conv_general_dilated(
                x, weight, (2, 2), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
            )
        ) + p["b" + str(i)]
        mu = x.mean((1, 2), keepdims=True)
        var = ((x - mu) ** 2).mean((1, 2), keepdims=True)
        x = jax.nn.gelu((x - mu) * jax.lax.rsqrt(var + 1e-4))
    z = jnp.tanh(x.mean(1).reshape(len(x), 320) @ p["field"])
    return z, z @ p["word"] + p["bias"]


def augment(x, key):
    keys = jax.random.split(key, 4)
    n = len(x)
    roll = jax.random.randint(keys[0], (n, 1), -5, 6)
    indices = jnp.arange(x.shape[1])[None, :] - roll
    shifted = (
        jnp.take_along_axis(x, jnp.clip(indices, 0, x.shape[1] - 1)[:, :, None], axis=1)
        * ((indices >= 0) & (indices < x.shape[1]))[:, :, None]
    )
    t = jax.random.randint(keys[1], (n, 1, 1), 0, 65)
    f = jax.random.randint(keys[2], (n, 1, 1), 0, 41)
    clock = jnp.arange(64)[None, :, None]
    hz = jnp.arange(40)[None, None, :]
    keep = (~((clock >= t) & (clock < t + 6))) & (~((hz >= f) & (hz < f + 3)))
    return jnp.clip(
        shifted * jnp.asarray(keep)
        + jax.random.uniform(keys[3], (n, 1, 1), minval=-0.055, maxval=0.055),
        0,
        1,
    )
