"""Unit tests for the Study X3 post-train resume checkpoint and bounded-memory 480p metric helpers.

Covers the fix for the X3 Colab kernel deaths that occurred right after the final training step:
  1. `save/load_x3_post_train_checkpoint*` round-trips exact float32 weights and fails closed on any
     binding / source / steps / truncated-train-log / weight-digest / key / shape mismatch.
  2. `_chunked_psnr_ssim_480p` is identical to the unchunked `_psnr_and_ssim_480p` per-episode metrics.
  3. `_pred480_to_u8` matches the original elementwise `[-1, 1] -> uint8` quantization.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from evals.x3_eval_480p import _chunked_psnr_ssim_480p, _pred480_to_u8, _psnr_and_ssim_480p
from scripts.train.x3_train_480p import (
    load_x3_post_train_checkpoint_meta,
    load_x3_post_train_params,
    save_x3_post_train_checkpoint,
)


def _params() -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "w_a": jnp.asarray(rng.standard_normal((8, 4)), dtype=jnp.float32),
        "b_a": jnp.asarray(rng.standard_normal((4,)), dtype=jnp.float32),
    }


def _write_log(p: Path, last_step: int) -> None:
    p.write_text("".join(json.dumps({"step": s, "loss": 1.0}) + "\n" for s in (1, last_step)))


def _setup(tmp_path: Path, steps: int = 10):
    npz, meta, log = tmp_path / "post.npz", tmp_path / "post.json", tmp_path / "train.jsonl"
    params = _params()
    _write_log(log, steps)
    save_x3_post_train_checkpoint(npz, meta, params, binding_sha="b" * 64, source_sha="s" * 64, steps=steps, train_wall_s=12.5)
    return npz, meta, log, params


def test_post_train_checkpoint_roundtrip_exact(tmp_path: Path) -> None:
    npz, meta, log, params = _setup(tmp_path)
    m = load_x3_post_train_checkpoint_meta(meta, npz, log, binding_sha="b" * 64, source_sha="s" * 64, steps=10)
    assert m is not None and m["train_wall_s"] == 12.5
    loaded = load_x3_post_train_params(npz, params)
    for k in params:
        assert loaded[k].dtype == params[k].dtype
        np.testing.assert_array_equal(np.asarray(loaded[k]), np.asarray(params[k]))


@pytest.mark.parametrize(
    "kw",
    [
        dict(binding_sha="x" * 64, source_sha="s" * 64, steps=10),
        dict(binding_sha="b" * 64, source_sha="x" * 64, steps=10),
        dict(binding_sha="b" * 64, source_sha="s" * 64, steps=11),
    ],
)
def test_post_train_checkpoint_rejects_binding_mismatch(tmp_path: Path, kw: dict) -> None:
    npz, meta, log, _ = _setup(tmp_path)
    assert load_x3_post_train_checkpoint_meta(meta, npz, log, **kw) is None


def test_post_train_checkpoint_rejects_truncated_log_and_tampered_weights(tmp_path: Path) -> None:
    npz, meta, log, _ = _setup(tmp_path)
    kw = dict(binding_sha="b" * 64, source_sha="s" * 64, steps=10)
    _write_log(log, 9)
    assert load_x3_post_train_checkpoint_meta(meta, npz, log, **kw) is None
    _write_log(log, 10)
    assert load_x3_post_train_checkpoint_meta(meta, npz, log, **kw) is not None
    with npz.open("ab") as f:
        f.write(b"\0")
    assert load_x3_post_train_checkpoint_meta(meta, npz, log, **kw) is None


def test_post_train_params_reject_shape_and_key_mismatch(tmp_path: Path) -> None:
    npz, _, _, params = _setup(tmp_path)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        load_x3_post_train_params(npz, {**params, "w_a": jnp.zeros((8, 5), dtype=jnp.float32)})
    with pytest.raises(RuntimeError, match="key mismatch"):
        load_x3_post_train_params(npz, {**params, "extra": jnp.zeros((1,), dtype=jnp.float32)})


def test_chunked_psnr_ssim_matches_unchunked() -> None:
    rng = np.random.default_rng(1)
    # 37 episodes spans multiple `_METRIC_CHUNK=16` chunks with a ragged tail; small frames keep the test fast.
    pred = rng.integers(0, 256, size=(37, 4, 48, 64, 3), dtype=np.uint8)
    targ = rng.integers(0, 256, size=(37, 4, 48, 64, 3), dtype=np.uint8)
    full = _psnr_and_ssim_480p(pred, targ)
    chunked = _chunked_psnr_ssim_480p(pred, targ)
    for a, b in zip(full, chunked):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=0.0)
    z = _chunked_psnr_ssim_480p(pred[:0], targ[:0])
    assert all(x.shape == (0,) for x in z)


def test_pred480_to_u8_matches_reference_expression() -> None:
    v = np.linspace(-1.5, 1.5, 1001, dtype=np.float32)
    ref = np.clip((v + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    np.testing.assert_array_equal(_pred480_to_u8(v), ref)
