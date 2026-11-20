"""Unit tests verifying exact mathematical equivalence of `models/v3/fast_core.py:scan_core_fast`
and Chunkwise Parallel WY Delta-Rule against the reference O(1) kernel `scan_core` / `step_core`.

Verifies on CPU (`0.00 CU`):
  1. Forward `Z_seq` AND final persistent state equivalence across ALL 11 backbones on:
     - Dense streams (`valid_mask = 1`, non-multiple `T = 29`, `gqa_chunk_size = 8`),
     - Non-zero initial carry-in state + asynchronous mid-sequence `reset_mask`,
     - Missing-packet streams (`valid_mask = 0, dt_s = 0` holes mid-sequence).
  2. Full parameter gradient equivalence (`remat_layers=True` vs reference `scan_core`).
  3. Exact Chunkwise Parallel WY Delta-Rule (`use_wy_chunking=True, wy_chunk_size=8` on `T = 29`)
     forward, final state, and gradient equivalence across non-zero carry-in AND mid-sequence `reset_mask`.
  4. `bfloat16` MXU path finiteness and bit-exact causal suffix invariance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from models.v3.core import init_core_state, init_slwm_v3_params, scan_core
from models.v3.fast_core import scan_core_fast
from models.v3.spec import BACKBONE_TYPES, make_slwm_v3_spec


def _make_rich_packets(spec, t_len: int, batch_size: int, seed: int, *, with_resets: bool = False, with_holes: bool = False):
    rng = np.random.default_rng(seed)
    pkts = {
        "modality_id": jnp.asarray(rng.integers(0, spec.n_modalities, size=(t_len, batch_size)), dtype=jnp.int32),
        "dt_s": jnp.asarray(rng.uniform(0.02, 0.25, size=(t_len, batch_size)), dtype=jnp.float32),
        "valid_mask": jnp.ones((t_len, batch_size), dtype=jnp.float32),
        "reliability": jnp.asarray(rng.uniform(0.1, 1.0, size=(t_len, batch_size)), dtype=jnp.float32),
        "video_feat": jnp.asarray(rng.normal(0.0, 1.0, size=(t_len, batch_size, spec.video_patch_dim)), dtype=jnp.float32),
        "spatial_pos": jnp.asarray(rng.integers(0, spec.video_spatial_packets, size=(t_len, batch_size)), dtype=jnp.int32),
        "audio_codes": jnp.asarray(rng.integers(0, spec.audio_vocab_size, size=(t_len, batch_size, spec.audio_codebooks)), dtype=jnp.int32),
        "audio_cont": jnp.asarray(rng.normal(0.0, 1.0, size=(t_len, batch_size, spec.audio_cont_dim)), dtype=jnp.float32),
        "text_byte": jnp.asarray(rng.integers(0, spec.text_vocab_size, size=(t_len, batch_size)), dtype=jnp.int32),
        "proprio_feat": jnp.asarray(rng.normal(0.0, 1.0, size=(t_len, batch_size, spec.proprio_dim)), dtype=jnp.float32),
        "action_feat": jnp.asarray(rng.normal(0.0, 1.0, size=(t_len, batch_size, spec.action_dim)), dtype=jnp.float32),
        "has_true_action": jnp.asarray(rng.integers(0, 2, size=(t_len, batch_size)), dtype=jnp.float32),
        "probe_feat": jnp.asarray(rng.normal(0.0, 1.0, size=(t_len, batch_size, spec.probe_dim)), dtype=jnp.float32),
    }
    if with_resets:
        rm = np.zeros((t_len, batch_size), dtype=bool)
        rm[7, 0] = True
        rm[19, 0] = True
        if batch_size > 1:
            if 11 < t_len:
                rm[11, 1] = True
            if 24 < t_len:
                rm[24, 1] = True
        pkts["reset_mask"] = jnp.asarray(rm, dtype=jnp.bool_)
    if with_holes:
        vm = np.ones((t_len, batch_size), dtype=np.float32)
        dt = np.asarray(pkts["dt_s"]).copy()
        vm[5, 0] = 0.0
        dt[5, 0] = 0.0
        vm[14, 0] = 0.0
        dt[14, 0] = 0.0
        if batch_size > 1:
            vm[9, 1] = 0.0
            dt[9, 1] = 0.0
            vm[22, 1] = 0.0
            dt[22, 1] = 0.0
        pkts["valid_mask"] = jnp.asarray(vm, dtype=jnp.float32)
        pkts["dt_s"] = jnp.asarray(dt, dtype=jnp.float32)
    return pkts


def _assert_states_close(st_ref, st_fast, *, atol_fp32: float = 2e-4, atol_bf16: float = 2e-2) -> None:
    ref_leaves = jax.tree_util.tree_leaves(st_ref)
    fast_leaves = jax.tree_util.tree_leaves(st_fast)
    assert len(ref_leaves) == len(fast_leaves)
    for r, f in zip(ref_leaves, fast_leaves):
        assert r.shape == f.shape and r.dtype == f.dtype
        if r.dtype == jnp.bool_ or jnp.issubdtype(r.dtype, jnp.integer):
            assert bool(jnp.array_equal(r, f)), f"Exact mismatch on {r.dtype} state leaf: ref={r} vs fast={f}"
        else:
            tol = atol_bf16 if r.dtype == jnp.bfloat16 else atol_fp32
            diff = float(jnp.max(jnp.abs(r.astype(jnp.float32) - f.astype(jnp.float32))))
            assert diff <= tol, f"State diff {diff:.6g} > {tol} on dtype={r.dtype} shape={r.shape}"


def test_fast_core_matches_reference_all_11_backbones_resets_and_holes() -> None:
    """Verify exact fp32 output & final state equivalence across ALL 11 backbones with carry-in, resets, & holes."""
    for bb in BACKBONE_TYPES:
        spec = make_slwm_v3_spec("tiny_unit", backbone_type=bb)
        params = init_slwm_v3_params(jax.random.PRNGKey(101), spec)
        # Warm up a realistic non-zero carry-in state!
        pkts_warm = _make_rich_packets(spec, t_len=10, batch_size=2, seed=201)
        _, st_carry, _ = scan_core(params, init_core_state(spec, 2), pkts_warm, spec)

        # Case 1: Dense stream with asynchronous resets & non-multiple T=29, gqa_chunk_size=8
        pkts_rst = _make_rich_packets(spec, t_len=29, batch_size=2, seed=202, with_resets=True, with_holes=False)
        z_ref, st_ref, _ = scan_core(params, st_carry, pkts_rst, spec)
        z_fast, st_fast, _ = scan_core_fast(
            params, st_carry, pkts_rst, spec, remat_layers=False, matmul_dtype=jnp.float32, gqa_chunk_size=8
        )
        z_diff = float(jnp.max(jnp.abs(z_ref - z_fast)))
        assert z_diff < 5e-4, f"{bb} dense+resets z_diff={z_diff:.6g}"
        _assert_states_close(st_ref, st_fast, atol_fp32=5e-4)

        # Case 2: Missing-packet holes + resets
        pkts_hole = _make_rich_packets(spec, t_len=25, batch_size=2, seed=203, with_resets=True, with_holes=True)
        z_ref_h, st_ref_h, _ = scan_core(params, st_carry, pkts_hole, spec)
        z_fast_h, st_fast_h, _ = scan_core_fast(
            params, st_carry, pkts_hole, spec, remat_layers=False, matmul_dtype=jnp.float32, gqa_chunk_size=8
        )
        z_diff_h = float(jnp.max(jnp.abs(z_ref_h - z_fast_h)))
        assert z_diff_h < 5e-4, f"{bb} holes+resets z_diff={z_diff_h:.6g}"
        _assert_states_close(st_ref_h, st_fast_h, atol_fp32=5e-4)


def test_fast_core_gradient_equivalence_with_layer_remat() -> None:
    """Verify full trunk gradients of `scan_core_fast(remat_layers=True)` match `scan_core` in fp32."""
    for bb in ("slwm_v3_routed", "single_bank_gated_deltanet", "diag_ssm", "explicit_bounded_kv", "sliding_window_gqa"):
        spec = make_slwm_v3_spec("tiny_unit", backbone_type=bb)
        params = init_slwm_v3_params(jax.random.PRNGKey(301), spec)
        pkts = _make_rich_packets(spec, t_len=21, batch_size=2, seed=302, with_resets=True)
        st0 = init_core_state(spec, 2)

        def _loss_ref(p):
            z, _, _ = scan_core(p, st0, pkts, spec)
            return jnp.mean(z * z)

        def _loss_fast(p):
            z, _, _ = scan_core_fast(p, st0, pkts, spec, remat_layers=True, matmul_dtype=jnp.float32, gqa_chunk_size=8)
            return jnp.mean(z * z)

        l_ref, g_ref = jax.value_and_grad(_loss_ref)(params)
        l_fast, g_fast = jax.value_and_grad(_loss_fast)(params)
        assert abs(float(l_ref - l_fast)) < 2e-4, f"{bb} loss diff {float(l_ref - l_fast)}"
        max_g_diff = max(
            float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(jax.tree_util.tree_leaves(g_ref), jax.tree_util.tree_leaves(g_fast))
        )
        assert max_g_diff < 2e-3, f"{bb} grad diff={max_g_diff:.6g}"


def test_chunkwise_wy_deltanet_exact_forward_state_and_grad_equivalence() -> None:
    """Verify exact Chunkwise Parallel WY Delta-Rule (`wy_chunk_size=8` on T=29) matches `scan_core` across resets & carry-in!"""
    for bb in ("single_bank_gated_deltanet", "single_bank_spread_tau"):
        spec = make_slwm_v3_spec("tiny_unit", backbone_type=bb)
        params = init_slwm_v3_params(jax.random.PRNGKey(401), spec)
        pkts_warm = _make_rich_packets(spec, t_len=10, batch_size=2, seed=402)
        _, st_carry, _ = scan_core(params, init_core_state(spec, 2), pkts_warm, spec)

        pkts = _make_rich_packets(spec, t_len=29, batch_size=2, seed=403, with_resets=True)
        z_ref, st_ref, _ = scan_core(params, st_carry, pkts, spec)
        z_wy, st_wy, _ = scan_core_fast(
            params,
            st_carry,
            pkts,
            spec,
            remat_layers=True,
            matmul_dtype=jnp.float32,
            gqa_chunk_size=8,
            use_wy_chunking=True,
            wy_chunk_size=8,
        )
        assert float(jnp.max(jnp.abs(z_ref - z_wy))) < 5e-4, f"{bb} WY z diff={float(jnp.max(jnp.abs(z_ref - z_wy)))}"
        _assert_states_close(st_ref, st_wy, atol_fp32=5e-4)

        def _loss_ref(p):
            z, _, _ = scan_core(p, st_carry, pkts, spec)
            return jnp.mean(z * z)

        def _loss_wy(p):
            z, _, _ = scan_core_fast(
                p, st_carry, pkts, spec, remat_layers=True, matmul_dtype=jnp.float32, gqa_chunk_size=8, use_wy_chunking=True, wy_chunk_size=8
            )
            return jnp.mean(z * z)

        _, g_ref = jax.value_and_grad(_loss_ref)(params)
        _, g_wy = jax.value_and_grad(_loss_wy)(params)
        max_g_diff = max(
            float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(jax.tree_util.tree_leaves(g_ref), jax.tree_util.tree_leaves(g_wy))
        )
        assert max_g_diff < 2e-3, f"{bb} WY grad diff={max_g_diff:.6g}"


def test_fast_core_bf16_mxu_causal_suffix_invariance_and_finiteness() -> None:
    """Verify production `matmul_dtype=bfloat16` path is strictly finite & bit-exact causal (zero future leak!)."""
    for bb in ("slwm_v3_routed", "single_bank_gated_deltanet", "diag_ssm"):
        spec = make_slwm_v3_spec("tiny_unit", backbone_type=bb)
        params = init_slwm_v3_params(jax.random.PRNGKey(501), spec)
        st0 = init_core_state(spec, 2)
        pkts_a = _make_rich_packets(spec, t_len=24, batch_size=2, seed=502)
        pkts_b = dict(pkts_a)
        # Mutate strictly after t = 12!
        pkts_b["probe_feat"] = pkts_a["probe_feat"].at[12:].set(-3.0 * pkts_a["probe_feat"][12:])
        pkts_b["video_feat"] = pkts_a["video_feat"].at[12:].set(-3.0 * pkts_a["video_feat"][12:])

        za, _, _ = scan_core_fast(
            params, st0, pkts_a, spec, remat_layers=True, matmul_dtype=jnp.bfloat16, use_wy_chunking=(bb == "single_bank_gated_deltanet")
        )
        zb, _, _ = scan_core_fast(
            params, st0, pkts_b, spec, remat_layers=True, matmul_dtype=jnp.bfloat16, use_wy_chunking=(bb == "single_bank_gated_deltanet")
        )
        assert bool(jnp.all(jnp.isfinite(za))) and bool(jnp.all(jnp.isfinite(zb)))
        assert float(jnp.max(jnp.abs(za[:12] - zb[:12]))) == 0.0, f"{bb} bf16 causal prefix leak!"
        assert float(jnp.max(jnp.abs(za[12:] - zb[12:]))) > 1e-3, f"{bb} bf16 suffix unresponsive!"
