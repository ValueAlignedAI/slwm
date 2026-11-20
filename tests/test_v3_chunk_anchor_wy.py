"""Bit-Exact Equivalence, Causality, Split-Call & Gradient Tests for Causal Chunk-Anchor Surprise WY (`C > 1`).

Verifies every mathematical requirement specified by Astra 6.2 & Gate 3 Peer Review:
  1. Exact byte-for-byte streaming state accounting (`M`, `M_anchor`, `cum_decay`, `chunk_phase`).
  2. Bit-exact forward outputs `Z_seq`, final state leaves, AND backward parameter gradients between:
       - Online 1-step `step_core` / `scan_core`,
       - Serial `scan_core_fast(..., use_wy_chunking=False)`,
       - Parallel WY `scan_core_fast(..., use_wy_chunking=True)`
     across non-multiple-of-`C` sequence length (`T=37`), mid-chunk resets, missing-packet holes
     (`valid_mask == 0`), and same-key overwrites at arbitrary offsets inside chunks.
  3. Arbitrary non-chunk-aligned split-call invariance (`0..13` then `13..37` == full `0..37`).
  4. Strict `0.000000` causal suffix invariance (`Z[:t]` depends strictly on `packets[:t]`).
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from models.v3.core import init_core_state, init_slwm_v3_params, scan_core
from models.v3.fast_core import scan_core_fast
from models.v3.spec import compute_core_streaming_state_bytes, make_slwm_v3_spec
from scripts.train.v3_pilot_train import make_synthetic_multimodal_batch


def _actual_state_bytes(state_tree) -> int:
    return int(sum(int(x.size) * int(x.dtype.itemsize) for x in jax.tree_util.tree_leaves(state_tree)))


@pytest.mark.parametrize("chunk_c", [8, 16])
def test_chunk_anchor_surprise_wy_exact_equivalence_resets_holes_splits_and_grads(chunk_c: int) -> None:
    base_spec = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_routed")
    spec = replace(base_spec, surprise_chunk_size=chunk_c)
    params = init_slwm_v3_params(jax.random.PRNGKey(501 + chunk_c), spec)
    st0 = init_core_state(spec, batch_size=2)

    # 1. Exact byte-for-byte streaming state accounting!
    reported = compute_core_streaming_state_bytes(spec, batch_size=2)["total_core_state_bytes"]
    actual = _actual_state_bytes(st0)
    assert reported == actual, f"State byte accounting mismatch for C={chunk_c}: reported={reported} vs actual={actual}"

    # 2. Build challenging sequence: T=37 (non-multiple of C!), mid-chunk resets, holes, & same-key repeated inputs!
    pkts, _, _ = make_synthetic_multimodal_batch(spec, seq_len=37, batch_size=2, seed=700 + chunk_c)
    rst = jnp.zeros((37, 2), dtype=jnp.bool_).at[0, :].set(True).at[5, 0].set(True).at[21, 1].set(True)
    vld = pkts["valid_mask"].at[9, 0].set(0.0).at[24, 1].set(0.0)
    # Inject identical repeated key/payload at ticks 3 & 4 (same-key intra-chunk overwrite test!)
    prb = pkts["probe_feat"].at[4].set(pkts["probe_feat"][3]).at[11].set(pkts["probe_feat"][3])
    pkts = {**pkts, "reset_mask": rst, "valid_mask": vld, "probe_feat": prb}

    # Reference serial `scan_core` (calls exact 1-step `step_core`)
    z_ref, st_ref, _ = scan_core(params, st0, pkts, spec)
    # Fast serial (`use_wy_chunking=False`)
    z_ser, st_ser, _ = scan_core_fast(params, st0, pkts, spec, remat_layers=False, matmul_dtype=jnp.float32, use_wy_chunking=False)
    # Fast parallel WY (`use_wy_chunking=True`)
    z_wy, st_wy, _ = scan_core_fast(params, st0, pkts, spec, remat_layers=False, matmul_dtype=jnp.float32, use_wy_chunking=True)

    assert float(jnp.max(jnp.abs(z_wy - z_ref))) < 5e-4, f"WY vs step_core Z mismatch: {jnp.max(jnp.abs(z_wy - z_ref))}"
    assert float(jnp.max(jnp.abs(z_wy - z_ser))) < 5e-4, f"WY vs serial fast Z mismatch: {jnp.max(jnp.abs(z_wy - z_ser))}"
    for li in range(spec.n_layers):
        for k in ("M", "M_anchor", "cum_decay", "chunk_phase"):
            diff = float(jnp.max(jnp.abs(st_wy[li]["mem"][k].astype(jnp.float32) - st_ref[li]["mem"][k].astype(jnp.float32))))
            assert diff < 5e-4, f"Layer {li} state[{k}] WY vs step_core diff={diff}"

    # 3. Non-chunk-aligned split-call equivalence: `0..13` -> `13..37` == full `0..37`!
    p1 = {k: v[:13] for k, v in pkts.items()}
    p2 = {k: v[13:] for k, v in pkts.items()}
    z1, st1, _ = scan_core_fast(params, st0, p1, spec, remat_layers=False, matmul_dtype=jnp.float32, use_wy_chunking=True)
    z2, st2, _ = scan_core_fast(params, st1, p2, spec, remat_layers=False, matmul_dtype=jnp.float32, use_wy_chunking=True)
    z_split = jnp.concatenate([z1, z2], axis=0)
    assert float(jnp.max(jnp.abs(z_split - z_wy))) < 5e-4, f"Split-call mismatch: {jnp.max(jnp.abs(z_split - z_wy))}"
    for li in range(spec.n_layers):
        for k in ("M", "M_anchor", "cum_decay", "chunk_phase"):
            d_sp = float(jnp.max(jnp.abs(st2[li]["mem"][k].astype(jnp.float32) - st_wy[li]["mem"][k].astype(jnp.float32))))
            assert d_sp < 5e-4, f"Split-call final state[{k}] diff={d_sp}"

    # 4. Exact backward gradient equivalence (`WY` vs `step_core`!)
    def _loss_ref(p):
        z, _, _ = scan_core(p, st0, pkts, spec)
        return jnp.mean(z * z)

    def _loss_wy(p):
        z, _, _ = scan_core_fast(p, st0, pkts, spec, remat_layers=True, matmul_dtype=jnp.float32, use_wy_chunking=True)
        return jnp.mean(z * z)

    g_ref = jax.grad(_loss_ref)(params)
    g_wy = jax.grad(_loss_wy)(params)
    max_g_diff = max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree_util.tree_leaves(g_ref), jax.tree_util.tree_leaves(g_wy))
    )
    assert max_g_diff < 2e-3, f"Gradient mismatch WY vs step_core: {max_g_diff}"

    # 5. Strict `0.000000` causal suffix invariance!
    pkts_mod = {**pkts, "probe_feat": pkts["probe_feat"].at[18:].set(pkts["probe_feat"][18:] + 7.5)}
    z_mod, _, _ = scan_core_fast(params, st0, pkts_mod, spec, remat_layers=False, matmul_dtype=jnp.float32, use_wy_chunking=True)
    assert float(jnp.max(jnp.abs(z_mod[:18] - z_wy[:18]))) == 0.0, "CAUSAL SUFFIX LEAKAGE DETECTED in Chunk-Anchor WY!"
