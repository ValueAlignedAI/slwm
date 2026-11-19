"""SLWM-v3 Gate 3 Unit Test Suite (Rev 2 — Strictly Non-Vacuous Causality, Equivalence, Isolation, KV, muP & Lock).

Verifies EVERY mandatory Gate 3 engineering and mathematical invariant on CPU (`0.00 CU`):
  1. Bit-exact raw-input suffix invariance, online == offline equivalence (`bf16` < 1e-3, `fp32` < 1e-5),
     and exact state-byte parity across ALL 8 Wave A backbones.
  2. Chunked `jax.checkpoint` rematerialization (`remat_step=True, remat_chunk_size=8` on non-multiple `T=25`)
     forward AND backward gradient equivalence vs unchunked scan.
  3. Exact zero streaming state growth (`T=1..256`), bit-exact `reset_mask`, AND non-vacuous cross-episode
     supervision boundary masking (`t_valid == 1.0` everywhere: mutating post-reset targets changes loss
     by `> 1e-2` WITHOUT `reset_mask` and `0.0` WITH `reset_mask`!).
  4. Non-vacuous action-ordering causality & missing-vs-zero action separation (with NON-ZERO `pred_head_w`
     proving BOTH `0.0` future leak on `[:8]` AND `> 1e-4` positive causal response on `[8:]` AND `0.0`
     dummy action leak inside `predict_future_latents`!).
  5. Non-vacuous zero decoder-to-core gradient (`probe_w2` initialized NON-ZERO -> `|grad(probe_w1)| > 1e-5`
     AND `|grad(probe_w2)| > 1e-5` AND `|grad_core| == 0.0` AND zero core weight change!).
  6. Missing-packet (`valid_mask=0, dt_s=0`) bit-exact state invariance across ALL 8 backbones +
     fair orthogonal ablation verification (`-NoWriteOption`, `-LearnedRouting` full `[0,1]` write +
     held `tau/g/read_w`, `-ZeroLagResidual`, faithful `single_bank_gated_deltanet` without router).
  7. `explicit_bounded_kv` multi-step (`T = 40`) finite gradients through fill, content-match overwrite,
     AND LRU eviction + non-zero `|grad(thresh)| > 0` + true LRU age refresh on attended read!
  8. Exact ±1.5% comparator param parity & ±5% byte parity across `pilot_d256`, `discovery_45m`, and
     `confirmation_135m` (`124M–150M` band), zero dead audio edge params across `frozen_codec_edge` vs
     `scratch_edge`, and balanced initial RMS across all 5 modalities!
  9. Exact base-shape muP (`lr_mult == 1.0` on EVERY leaf at `d == d_base` across all 8 backbones),
     strict weight-decay mask (`False` on `log_tau`, `log_a`, `norm_w`, biases, `thresh`, `gate_z0`),
     production masked `optax` AdamW 3-width coordinate check (`mup_ratio < sp_ratio`), and finite grads
     across all 8 backbones.
 10. Phase 3 frozen-codebook counterfactual generator `100%` analytic oracle + Launcher pre-flight gate-lock
     negative rejection test (empty `{}` receipt strictly rejected!).
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
import json
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from models.v3.core import (
    build_mup_lr_multipliers,
    build_weight_decay_mask,
    embed_packet_step,
    init_core_state,
    init_slwm_v3_params,
    predict_future_latents,
    scan_core,
    step_core,
)
from models.v3.memory import (
    init_explicit_kv_params,
    init_explicit_kv_state,
    init_matrix_memory_params,
    init_matrix_memory_state,
    step_explicit_kv,
    step_matrix_memory,
)
from models.v3.objectives import (
    decoupled_probe_mse_loss,
    init_decoupled_probe_params,
    masked_multi_horizon_predictive_loss,
)
from models.v3.spec import (
    BACKBONE_TYPES,
    SLWMv3Spec,
    build_wave_a_comparator_specs,
    compute_core_streaming_state_bytes,
    count_tree_params,
    make_slwm_v3_spec,
)
from scripts.train.v3_colab_pilot_launcher import (
    _run_subprocess_with_watchdog,
    projected_spend_exceeds_reserve,
    validate_phase_prerequisites,
    verified_stop_session,
    verify_gates_1_to_3_passed,
)
from scripts.train.v3_pilot_train import build_optimizer, make_synthetic_multimodal_batch
from scripts.train.v3_tpu_pilot_suite import (
    build_frozen_phase3_codebook,
    combine_phase2_verdict,
    evaluate_lr_transfer_verdict,
    generate_phase3_counterfactual_batch,
    is_tpu_v6e_device_kind,
    validate_reusable_mup_receipt,
    verify_phase3_counterfactual_oracle,
)


def _make_dummy_packets(spec: SLWMv3Spec, t_len: int = 24, batch_size: int = 2, seed: int = 0) -> dict[str, jax.Array]:
    """Generate a rich multimodal causal packet stream `[T, B, ...]` touching all 5 modalities."""
    rng = np.random.default_rng(seed)
    mod_ids = rng.integers(0, spec.n_modalities, size=(t_len, batch_size), dtype=np.int32)
    return {
        "modality_id": jnp.asarray(mod_ids, dtype=jnp.int32),
        "dt_s": jnp.asarray(rng.uniform(0.04, 0.25, size=(t_len, batch_size)), dtype=jnp.float32),
        "valid_mask": jnp.ones((t_len, batch_size), dtype=jnp.float32),
        "reliability": jnp.asarray(rng.uniform(0.6, 1.0, size=(t_len, batch_size)), dtype=jnp.float32),
        "video_feat": jnp.asarray(rng.normal(0.0, 1.0, size=(t_len, batch_size, spec.video_patch_dim)), dtype=jnp.float32),
        "spatial_pos": jnp.asarray(rng.integers(0, spec.video_spatial_packets, size=(t_len, batch_size)), dtype=jnp.int32),
        "audio_codes": jnp.asarray(rng.integers(0, spec.audio_vocab_size, size=(t_len, batch_size, spec.audio_codebooks)), dtype=jnp.int32),
        "audio_cont": jnp.asarray(rng.normal(0.0, 1.0, size=(t_len, batch_size, spec.audio_cont_dim)), dtype=jnp.float32),
        "text_byte": jnp.asarray(rng.integers(0, spec.text_vocab_size, size=(t_len, batch_size)), dtype=jnp.int32),
        "proprio_feat": jnp.asarray(rng.normal(0.0, 1.0, size=(t_len, batch_size, spec.proprio_dim)), dtype=jnp.float32),
        "action_feat": jnp.asarray(rng.normal(0.0, 1.0, size=(t_len, batch_size, spec.action_dim)), dtype=jnp.float32),
        "has_true_action": jnp.asarray((rng.random((t_len, batch_size)) < 0.5).astype(np.float32)),
        "probe_feat": jnp.asarray(rng.normal(0.0, 1.0, size=(t_len, batch_size, spec.probe_dim)), dtype=jnp.float32),
    }


def _tree_nbytes(tree) -> int:
    return sum(int(x.size) * int(x.dtype.itemsize) for x in jax.tree_util.tree_leaves(tree))


@pytest.mark.parametrize("backbone_type", BACKBONE_TYPES)
def test_strict_suffix_invariance_online_offline_and_missing_packet_invariance(backbone_type: str) -> None:
    """Verify ZERO future lookahead, online == offline (`bf16` & `fp32`), state byte parity & missing-packet invariance."""
    spec = make_slwm_v3_spec("tiny_unit", backbone_type=backbone_type)
    params = init_slwm_v3_params(jax.random.PRNGKey(11), spec)
    st0 = init_core_state(spec, batch_size=2)
    assert _tree_nbytes(st0) == compute_core_streaming_state_bytes(spec, batch_size=2)["total_core_state_bytes"]

    pkts = _make_dummy_packets(spec, t_len=24, batch_size=2, seed=101)
    pkts_alt = dict(pkts)
    rng = np.random.default_rng(999)
    for k in ("video_feat", "audio_cont", "proprio_feat", "action_feat", "probe_feat"):
        arr = np.array(pkts[k])
        arr[12:] = rng.normal(0.0, 2.0, size=arr[12:].shape).astype(np.float32)
        pkts_alt[k] = jnp.asarray(arr)
    for k, hi in (("audio_codes", spec.audio_vocab_size), ("text_byte", spec.text_vocab_size)):
        arr = np.array(pkts[k])
        arr[12:] = rng.integers(0, hi, size=arr[12:].shape, dtype=np.int32)
        pkts_alt[k] = jnp.asarray(arr)

    z_base, st_base, _ = scan_core(params, st0, pkts, spec)
    z_alt, _, _ = scan_core(params, st0, pkts_alt, spec)

    assert float(jnp.max(jnp.abs(z_base[:12] - z_alt[:12]))) == 0.0
    assert float(jnp.max(jnp.abs(z_base[12:] - z_alt[12:]))) > 1e-4

    # Online step-by-step == offline scan (`bf16` ring KV)
    cur_st = st0
    online_zs = []
    for t in range(24):
        zt, cur_st, _ = step_core(params, cur_st, {k: v[t] for k, v in pkts.items()}, spec)
        online_zs.append(zt)
    assert float(jnp.max(jnp.abs(jnp.stack(online_zs, axis=0) - z_base))) < 1e-3
    st_diffs = [
        float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))
        for a, b in zip(jax.tree_util.tree_leaves(cur_st), jax.tree_util.tree_leaves(st_base))
    ]
    assert max(st_diffs) < 2e-3

    # Exact fp32 online == offline equivalence (< 1e-5 on BOTH outputs AND final state!)
    spec_fp32 = dc_replace(spec, ring_kv_bytes_per_elem=4)
    st_fp32 = init_core_state(spec_fp32, batch_size=2)
    z_f_scan, st_f_scan, _ = scan_core(params, st_fp32, pkts, spec_fp32)
    cur_f = st_fp32
    z_f_on = []
    for t in range(24):
        zt_f, cur_f, _ = step_core(params, cur_f, {k: v[t] for k, v in pkts.items()}, spec_fp32)
        z_f_on.append(zt_f)
    assert float(jnp.max(jnp.abs(jnp.stack(z_f_on, axis=0) - z_f_scan))) < 1e-5
    st_f_diffs = [
        float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))
        for a, b in zip(jax.tree_util.tree_leaves(cur_f), jax.tree_util.tree_leaves(st_f_scan))
    ]
    assert max(st_f_diffs) < 1e-5

    # Missing packet (`valid_mask = 0, dt_s = 0`) leaves ENTIRE persistent state BIT-EXACT unchanged across ALL 8 backbones!
    _, st_after_5, _ = scan_core(params, st0, {k: v[:5] for k, v in pkts.items()}, spec)
    pkt_miss = {k: v[5] for k, v in pkts.items()}
    pkt_miss["valid_mask"] = jnp.zeros((2,), dtype=jnp.float32)
    pkt_miss["dt_s"] = jnp.zeros((2,), dtype=jnp.float32)
    _, st_after_miss, _ = step_core(params, st_after_5, pkt_miss, spec)
    for a, b in zip(jax.tree_util.tree_leaves(st_after_miss), jax.tree_util.tree_leaves(st_after_5)):
        assert float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) == 0.0, (
            f"{backbone_type} mutated state on valid_mask=0, dt_s=0"
        )


def test_chunked_rematerialization_forward_and_backward_equivalence() -> None:
    """Verify outer chunked `jax.checkpoint` remat matches unchunked scan on non-multiple T=25 in fwd & bwd."""
    spec = dc_replace(make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_routed"), ring_kv_bytes_per_elem=4)
    params = init_slwm_v3_params(jax.random.PRNGKey(15), spec)
    st0 = init_core_state(spec, batch_size=2)
    pkts = _make_dummy_packets(spec, t_len=25, batch_size=2, seed=115)
    targs = jnp.asarray(np.random.default_rng(116).normal(0.0, 1.0, size=(25, 2, spec.target_feat_dim)), dtype=jnp.float32)
    tvalid = jnp.ones((25, 2), dtype=jnp.float32)

    z_plain, st_plain, _ = scan_core(params, st0, pkts, spec, remat_step=False)
    z_chunk, st_chunk, _ = scan_core(params, st0, pkts, spec, remat_step=True, remat_chunk_size=8)
    assert float(jnp.max(jnp.abs(z_plain - z_chunk))) < 1e-5
    for a, b in zip(jax.tree_util.tree_leaves(st_plain), jax.tree_util.tree_leaves(st_chunk)):
        assert float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) < 1e-5

    # Give non-zero readout weight so full trunk gradients are exercised!
    p_nz = dict(params, pred_head_w=jax.random.normal(jax.random.PRNGKey(16), params["pred_head_w"].shape) * 0.05)
    l_plain, g_plain = jax.value_and_grad(
        lambda p: masked_multi_horizon_predictive_loss(p, st0, pkts, targs, tvalid, spec, remat_step=False)[0]
    )(p_nz)
    l_chunk, g_chunk = jax.value_and_grad(
        lambda p: masked_multi_horizon_predictive_loss(p, st0, pkts, targs, tvalid, spec, remat_step=True, remat_chunk_size=8)[0]
    )(p_nz)
    assert abs(float(l_plain) - float(l_chunk)) < 1e-5
    max_g_diff = max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree_util.tree_leaves(g_plain), jax.tree_util.tree_leaves(g_chunk))
    )
    assert max_g_diff < 1e-4


def test_zero_state_growth_bit_exact_reset_and_nonvacuous_cross_episode_mask() -> None:
    """Verify zero state growth (T=1..256), bit-exact reset, AND NON-VACUOUS `same_ep` boundary masking."""
    spec = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_routed")
    params = init_slwm_v3_params(jax.random.PRNGKey(21), spec)
    st0 = init_core_state(spec, batch_size=2)
    nbytes_t0 = _tree_nbytes(st0)

    pkts_long = _make_dummy_packets(spec, t_len=256, batch_size=2, seed=202)
    z_1, st_1, _ = scan_core(params, st0, {k: v[:1] for k, v in pkts_long.items()}, spec)
    _, st_256, _ = scan_core(params, st0, pkts_long, spec)
    assert _tree_nbytes(st_1) == nbytes_t0 and _tree_nbytes(st_256) == nbytes_t0
    assert bool(jnp.all(jnp.isfinite(z_1)))
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(st_256))

    # Bit-exact stream reset at t = 12
    pkts_pack = _make_dummy_packets(spec, t_len=24, batch_size=2, seed=303)
    reset_mask = np.zeros((24, 2), dtype=bool)
    reset_mask[12, :] = True
    pkts_pack["reset_mask"] = jnp.asarray(reset_mask)

    z_packed, _, _ = scan_core(params, st0, pkts_pack, spec)
    z_ep2_clean, _, _ = scan_core(params, st0, {k: v[12:] for k, v in pkts_pack.items() if k != "reset_mask"}, spec)
    assert float(jnp.max(jnp.abs(z_packed[12:] - z_ep2_clean))) == 0.0

    # NON-VACUOUS cross-episode supervision bleed test:
    # Keep `t_valid = ones((24, 2))` and silence ONLY `src_valid[12:] = 0` so targets at `t >= 12` are VALID
    # and ONLY `same_ep` prevents `t < 12` from predicting across the episode boundary into `t + h >= 12`!
    pkts_ep1_src = dict(pkts_pack, valid_mask=jnp.zeros((24, 2), dtype=jnp.float32).at[:12].set(1.0))
    pkts_no_reset = {k: v for k, v in pkts_ep1_src.items() if k != "reset_mask"}
    rng = np.random.default_rng(404)
    targs = jnp.asarray(rng.normal(0.0, 1.0, size=(24, 2, spec.target_feat_dim)), dtype=jnp.float32)
    targs_mut = targs.at[12:].set(99.0)
    t_valid = jnp.ones((24, 2), dtype=jnp.float32)

    loss_with_reset_a, _ = masked_multi_horizon_predictive_loss(params, st0, pkts_ep1_src, targs, t_valid, spec)
    loss_with_reset_b, _ = masked_multi_horizon_predictive_loss(params, st0, pkts_ep1_src, targs_mut, t_valid, spec)
    assert float(jnp.abs(loss_with_reset_a - loss_with_reset_b)) == 0.0

    # Positive control: WITHOUT `reset_mask`, mutating `targs[12:]` MUST change loss substantially (`> 1.0`)!
    loss_no_reset_a, _ = masked_multi_horizon_predictive_loss(params, st0, pkts_no_reset, targs, t_valid, spec)
    loss_no_reset_b, _ = masked_multi_horizon_predictive_loss(params, st0, pkts_no_reset, targs_mut, t_valid, spec)
    assert float(jnp.abs(loss_no_reset_a - loss_no_reset_b)) > 1.0


def test_nonvacuous_action_causality_and_missing_vs_zero_action_separation() -> None:
    """Verify action causality, zero dummy-action leak in BOTH core & predictor, and positive response with non-zero readout."""
    spec = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_routed")
    params = init_slwm_v3_params(jax.random.PRNGKey(31), spec)
    # Non-zero `pred_head_w` so `predict_future_latents` causality check is strictly NON-VACUOUS!
    params = dict(params, pred_head_w=jax.random.normal(jax.random.PRNGKey(32), params["pred_head_w"].shape) * 0.05)
    st0 = init_core_state(spec, batch_size=2)
    pkts = _make_dummy_packets(spec, t_len=16, batch_size=2, seed=505)
    pkts["modality_id"] = jnp.full((16, 2), 3, dtype=jnp.int32)

    # 1. Missing action (`has_true_action == 0`) inside scan_core -> 0.0 diff on dummy action mutation
    pkts_miss_a = dict(pkts, has_true_action=jnp.zeros((16, 2), dtype=jnp.float32), action_feat=jnp.zeros((16, 2, spec.action_dim)))
    pkts_miss_b = dict(pkts, has_true_action=jnp.zeros((16, 2), dtype=jnp.float32), action_feat=jnp.full((16, 2, spec.action_dim), 7.5))
    za, _, _ = scan_core(params, st0, pkts_miss_a, spec)
    zb, _, _ = scan_core(params, st0, pkts_miss_b, spec)
    assert float(jnp.max(jnp.abs(za - zb))) == 0.0

    # 2. Executed zero action vs missing action -> strictly distinct (`> 1e-4`)
    pkts_exec_zero = dict(pkts, has_true_action=jnp.ones((16, 2), dtype=jnp.float32), action_feat=jnp.zeros((16, 2, spec.action_dim)))
    zc, _, _ = scan_core(params, st0, pkts_exec_zero, spec)
    assert float(jnp.max(jnp.abs(zc - za))) > 1e-4

    # 3. Missing action inside `predict_future_latents` -> 0.0 diff on dummy action mutation!
    has_0 = jnp.zeros((16, 2), dtype=jnp.float32)
    pm1 = predict_future_latents(params, zc, jnp.zeros((16, 2, spec.action_dim)), has_0, spec)
    pm2 = predict_future_latents(params, zc, jnp.full((16, 2, spec.action_dim), 9.0), has_0, spec)
    assert float(jnp.max(jnp.abs(pm1 - pm2))) == 0.0

    # 4. Future action at t >= 8: 0.0 diff on `[:8]` AND positive causal response `> 1e-4` on `[8:]`!
    act_1 = jnp.ones((16, 2, spec.action_dim), dtype=jnp.float32)
    act_2 = act_1.at[8:].set(-5.0)
    has_1 = jnp.ones((16, 2), dtype=jnp.float32)
    p1 = predict_future_latents(params, zc, act_1, has_1, spec)
    p2 = predict_future_latents(params, zc, act_2, has_1, spec)
    assert float(jnp.max(jnp.abs(p1[:8] - p2[:8]))) == 0.0
    assert float(jnp.max(jnp.abs(p1[8:] - p2[8:]))) > 1e-4


def test_nonvacuous_zero_decoder_to_core_gradient_and_parameter_update() -> None:
    """Verify decoupled probe training produces non-zero `probe_w1` & `probe_w2` grads AND bit-exact ZERO core grad!"""
    spec = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_routed")
    core_params = init_slwm_v3_params(jax.random.PRNGKey(41), spec)
    probe_params = init_decoupled_probe_params(jax.random.PRNGKey(42), spec.d_model, out_dim=4)
    # NON-ZERO `probe_w2` so gradients genuinely propagate into `probe_w1` AND hit the `stop_gradient` wall!
    probe_params["probe_w2"] = jax.random.normal(jax.random.PRNGKey(43), probe_params["probe_w2"].shape) * 0.1
    st0 = init_core_state(spec, batch_size=2)
    pkts = _make_dummy_packets(spec, t_len=12, batch_size=2, seed=606)
    targets = jnp.ones((2, 4), dtype=jnp.float32)

    loss, (grad_core, grad_probe) = jax.value_and_grad(decoupled_probe_mse_loss, argnums=(0, 1))(
        core_params, probe_params, st0, pkts, targets, spec
    )
    assert bool(jnp.isfinite(loss))
    max_core_grad = max(float(jnp.max(jnp.abs(g))) for g in jax.tree_util.tree_leaves(grad_core))
    assert max_core_grad == 0.0, f"Core received non-zero gradient from decoupled probe: {max_core_grad}"
    assert float(jnp.max(jnp.abs(grad_probe["probe_w1"]))) > 1e-5, "Vacuous test: probe_w1 grad vanished!"
    assert float(jnp.max(jnp.abs(grad_probe["probe_w2"]))) > 1e-5, "Vacuous test: probe_w2 grad vanished!"

    new_core = jax.tree_util.tree_map(lambda p, g: p - 1e-2 * g, core_params, grad_core)
    max_core_delta = max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree_util.tree_leaves(new_core), jax.tree_util.tree_leaves(core_params))
    )
    assert max_core_delta == 0.0


def test_orthogonal_ablations_and_faithful_single_bank_contracts() -> None:
    """Verify `-NoWriteOption`, fair `-LearnedRouting`, `-ZeroLagResidual`, and faithful `single_bank_gated_deltanet`."""
    pkts = _make_dummy_packets(make_slwm_v3_spec("tiny_unit"), t_len=8, batch_size=2, seed=707)

    # 1. `-NoWriteOption`: identically 0.0 no-write probability
    spec_nw = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_no_nowrite")
    p_nw = init_slwm_v3_params(jax.random.PRNGKey(52), spec_nw)
    _, _, d_nw = scan_core(p_nw, init_core_state(spec_nw, 2), pkts, spec_nw)
    assert float(jnp.max(jnp.abs(d_nw["mean_nowrite"]))) == 0.0

    # 2. `-LearnedRouting`: holds timescales `tau_b`, decay gates `g_b`, read routing `w_read_gate`, AND the
    #    `(nb + 1)`-logit no-write calculation IDENTICAL to `slwm_v3_routed` (0.20 no-write, 0.80 write mass at init),
    #    replacing ONLY content bank selection with content-independent round-robin!
    spec_rt = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_routed")
    spec_nr = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_no_routing")
    p_rt = init_slwm_v3_params(jax.random.PRNGKey(53), spec_rt)
    p_nr = init_slwm_v3_params(jax.random.PRNGKey(53), spec_nr)
    assert count_tree_params(p_rt) == count_tree_params(p_nr)
    assert p_nr["layers"][0]["mem"]["log_tau"].shape == (spec_nr.n_banks,)
    assert "w_read_gate" in p_nr["layers"][0]["mem"]
    mem_p_rt = init_matrix_memory_params(jax.random.PRNGKey(53), spec_rt)
    mem_p_nr = init_matrix_memory_params(jax.random.PRNGKey(53), spec_nr)
    mem_st_rt = init_matrix_memory_state(spec_rt, batch_size=2)
    mem_st_nr = init_matrix_memory_state(spec_nr, batch_size=2)
    x_step = jax.random.normal(jax.random.PRNGKey(530), (2, spec_nr.d_model))
    # Verify BIT-EXACT identical `r_nowrite` and aggregate `sum_b r_b == 1 - r_nowrite` across BOTH arms at step 0!
    _, _, md_rt0 = step_matrix_memory(
        mem_p_rt, mem_st_rt, x_step, jnp.zeros((2,), dtype=jnp.int32), jnp.full((2,), 0.1), jnp.ones((2,)), jnp.ones((2,)), spec_rt, step_idx_t=jnp.zeros((2,), dtype=jnp.int32)
    )
    _, _, md_nr0 = step_matrix_memory(
        mem_p_nr, mem_st_nr, x_step, jnp.zeros((2,), dtype=jnp.int32), jnp.full((2,), 0.1), jnp.ones((2,)), jnp.ones((2,)), spec_nr, step_idx_t=jnp.zeros((2,), dtype=jnp.int32)
    )
    assert np.allclose(np.asarray(md_rt0["r_nowrite"]), np.asarray(md_nr0["r_nowrite"]), atol=1e-6)
    assert np.allclose(np.asarray(md_rt0["r_banks"]).sum(axis=-1), np.asarray(md_nr0["r_banks"]).sum(axis=-1), atol=1e-6)
    assert 0.75 < float(np.asarray(md_nr0["r_banks"]).sum(axis=-1).mean()) < 0.85

    # Verify deterministic content-independent round-robin bank selection + full [0, 1] single-bank ceiling
    mem_p_nr["router_b2"] = jnp.array([4.0, 4.0, 4.0, 4.0, -4.0], dtype=jnp.float32)
    selected_banks = []
    for step_i in range(spec_nr.n_banks):
        _, mem_st_nr, md = step_matrix_memory(
            mem_p_nr,
            mem_st_nr,
            x_step,
            jnp.zeros((2,), dtype=jnp.int32),
            jnp.full((2,), 0.1),
            jnp.ones((2,)),
            jnp.ones((2,)),
            spec_nr,
            step_idx_t=jnp.full((2,), step_i, dtype=jnp.int32),
        )
        rb = np.asarray(md["r_banks"])
        rnw = np.asarray(md["r_nowrite"])
        assert np.allclose(rb.sum(axis=-1), 1.0 - rnw, atol=1e-5)
        assert np.allclose(rb.max(axis=-1), 1.0 - rnw, atol=1e-5)
        assert float(rb.max()) > 0.99
        selected_banks.append(int(np.argmax(rb[0])))
    assert selected_banks == list(range(spec_nr.n_banks))

    # 3. `-ZeroLagResidual`: omits zero-lag bypass parameters
    spec_zl = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_no_zerolag")
    p_zl = init_slwm_v3_params(jax.random.PRNGKey(54), spec_zl)
    assert "zero_lag_proj" not in p_zl and "gate_z0" not in p_zl

    # 4. Faithful `single_bank_gated_deltanet`: NO surprise router MLP or `w_prior_proj`!
    spec_sb = make_slwm_v3_spec("tiny_unit", backbone_type="single_bank_gated_deltanet")
    p_sb = init_slwm_v3_params(jax.random.PRNGKey(55), spec_sb)
    assert "router_w1" not in p_sb["layers"][0]["mem"] and "w_prior_proj" not in p_sb["layers"][0]["mem"]


def test_explicit_bounded_kv_multistep_eviction_gradients_and_true_lru_read_refresh() -> None:
    """Verify `explicit_bounded_kv` ACTUALLY executes fill, content-match overwrite, AND LRU eviction with finite grads, zero-write key invariance, and `dt=0` LRU tie protection!"""
    spec = dc_replace(make_slwm_v3_spec("tiny_unit", backbone_type="explicit_bounded_kv"), explicit_kv_slots=4)
    params = init_slwm_v3_params(jax.random.PRNGKey(61), spec)
    params = dict(params, pred_head_w=jax.random.normal(jax.random.PRNGKey(62), params["pred_head_w"].shape) * 0.05)
    for li in range(spec.n_layers):
        params["layers"][li]["kv"]["router_b2"] = jnp.full((spec.explicit_kv_heads,), 2.0, dtype=jnp.float32)
        params["layers"][li]["kv"]["thresh"] = jnp.full((spec.explicit_kv_heads,), 0.40, dtype=jnp.float32)
    st0 = init_core_state(spec, batch_size=2)
    pkts, targs, tvalid = make_synthetic_multimodal_batch(spec, seq_len=40, batch_size=2, seed=610)
    # Repeat packet 0 at tick 6 to guarantee an exact content-match overwrite, while distinct ticks 0..39 force full fill + LRU eviction!
    pkts_ov = dict(pkts)
    for k in ("modality_id", "probe_feat", "video_feat", "audio_codes", "text_byte"):
        pkts_ov[k] = pkts[k].at[6].set(pkts[k][0])

    # Controlled full-table eviction & content-overwrite unit tests (unambiguous victim/replacement proof!)
    kv_p_ctrl = init_explicit_kv_params(jax.random.PRNGKey(65), spec)
    kv_p_ctrl["router_b2"] = jnp.full((spec.explicit_kv_heads,), 10.0, dtype=jnp.float32)  # write_prob ~ 1.0
    kv_p_ctrl["thresh"] = jnp.full((spec.explicit_kv_heads,), 0.70, dtype=jnp.float32)
    x_in = jax.random.normal(jax.random.PRNGKey(66), (1, spec.d_model))
    xn_c = x_in / jnp.sqrt(jnp.mean(x_in * x_in, axis=-1, keepdims=True) + 1e-6)
    k_in = jnp.matmul(xn_c, kv_p_ctrl["w_k"]).reshape(1, spec.explicit_kv_heads, spec.explicit_kv_dim)
    k_in = k_in / jnp.sqrt(jnp.sum(k_in * k_in, axis=-1, keepdims=True) + 1e-8)
    q_in = jnp.matmul(xn_c, kv_p_ctrl["w_q"]).reshape(1, spec.explicit_kv_heads, spec.explicit_kv_dim)
    q_in = q_in / jnp.sqrt(jnp.sum(q_in * q_in, axis=-1, keepdims=True) + 1e-8)
    v_in = jnp.matmul(xn_c, kv_p_ctrl["w_v"]).reshape(1, spec.explicit_kv_heads, spec.explicit_kv_dim)

    # Construct 4 stored keys strictly orthogonal to `k_in`, with slot 0 aligned to `q_in` (so read-before-write
    # attends to slot 0, leaving oldest victim slot 2 unrefreshed at age=99.0 across ALL 4 heads!).
    st_full = init_explicit_kv_state(spec, batch_size=1)
    q_orth_k = q_in - jnp.sum(q_in * k_in, axis=-1, keepdims=True) * k_in
    q_orth_k = q_orth_k / jnp.sqrt(jnp.sum(q_orth_k * q_orth_k, axis=-1, keepdims=True) + 1e-8)
    orth_keys = [q_orth_k]
    for s_i in range(1, 4):
        raw_r = jax.random.normal(jax.random.PRNGKey(700 + s_i), k_in.shape)
        o_k = raw_r - jnp.sum(raw_r * k_in, axis=-1, keepdims=True) * k_in
        o_k = o_k - jnp.sum(o_k * q_orth_k, axis=-1, keepdims=True) * q_orth_k
        o_k = o_k / jnp.sqrt(jnp.sum(o_k * o_k, axis=-1, keepdims=True) + 1e-8)
        orth_keys.append(o_k)
    st_full["kv_keys"] = jnp.stack(orth_keys, axis=2)  # [1, H, 4, D]
    st_full["kv_vals"] = jax.random.normal(jax.random.PRNGKey(710), st_full["kv_vals"].shape) * 5.0
    st_full["kv_valid"] = jnp.ones_like(st_full["kv_valid"])
    st_full["kv_age"] = jnp.broadcast_to(jnp.array([1.0, 2.0, 99.0, 3.0], dtype=jnp.float32)[None, None, :], (1, spec.explicit_kv_heads, 4))

    _, st_evicted, _ = step_explicit_kv(
        kv_p_ctrl, st_full, x_in, jnp.zeros((1,), dtype=jnp.int32), jnp.ones((1,)), jnp.ones((1,)), jnp.ones((1,)), spec
    )
    # Victim slot 2 MUST be cleanly replaced with `(k_in, v_in)` (zero victim value bleed, age == 0.0)!
    assert np.allclose(np.asarray(st_evicted["kv_keys"][:, :, 2, :]), np.asarray(k_in), atol=1e-4)
    assert np.allclose(np.asarray(st_evicted["kv_vals"][:, :, 2, :]), np.asarray(v_in), atol=1e-4)
    assert float(jnp.max(st_evicted["kv_age"][:, :, 2])) == 0.0
    # Non-victim slots 0, 1, 3 MUST keep their stored values cleanly (no victim bleed)!
    for s_keep in (0, 1, 3):
        assert np.allclose(np.asarray(st_evicted["kv_vals"][:, :, s_keep, :]), np.asarray(st_full["kv_vals"][:, :, s_keep, :]), atol=1e-2)

    # Controlled content-match overwrite: place `k_in` in slot 1 (NOT oldest!) -> slot 1 overwrites (>94% toward v_in), oldest slot 2 NOT evicted!
    st_match = dict(st_full, kv_keys=st_full["kv_keys"].at[:, :, 1, :].set(k_in))
    _, st_ow, _ = step_explicit_kv(
        kv_p_ctrl, st_match, x_in, jnp.zeros((1,), dtype=jnp.int32), jnp.ones((1,)), jnp.ones((1,)), jnp.ones((1,)), spec
    )
    err_before = float(jnp.max(jnp.abs(st_match["kv_vals"][:, :, 1, :] - v_in)))
    err_after = float(jnp.max(jnp.abs(st_ow["kv_vals"][:, :, 1, :] - v_in)))
    assert err_after < 0.06 * err_before, f"Expected >94% match overwrite: before={err_before:.4f}, after={err_after:.4f}"
    assert np.allclose(np.asarray(st_ow["kv_vals"][:, :, 2, :]), np.asarray(st_match["kv_vals"][:, :, 2, :]), atol=1e-2)

    loss, grads = jax.value_and_grad(
        lambda p: masked_multi_horizon_predictive_loss(p, st0, pkts_ov, targs, tvalid, spec)[0]
    )(params)
    assert bool(jnp.isfinite(loss))
    for g in jax.tree_util.tree_leaves(grads):
        assert bool(jnp.all(jnp.isfinite(g))), "Non-finite gradient in explicit_bounded_kv over T=40!"
    thresh_grad = max(float(jnp.max(jnp.abs(grads["layers"][li]["kv"]["thresh"]))) for li in range(spec.n_layers))
    assert thresh_grad > 1e-8, f"Dead thresh gradient in explicit_bounded_kv: {thresh_grad}"

    # Verify BOTH zero-write bit-exact key preservation AND `dt_s == 0.0` LRU read-refresh tie protection!
    kv_p = init_explicit_kv_params(jax.random.PRNGKey(63), spec)
    kv_p["router_b2"] = jnp.full((spec.explicit_kv_heads,), -30.0, dtype=jnp.float32)  # write_prob ~ 0!
    kv_st = init_explicit_kv_state(spec, batch_size=1)
    x_probe = jax.random.normal(jax.random.PRNGKey(64), (1, spec.d_model))
    xn = x_probe / jnp.sqrt(jnp.mean(x_probe * x_probe, axis=-1, keepdims=True) + 1e-6)
    q_dir = jnp.matmul(xn, kv_p["w_q"]).reshape(1, spec.explicit_kv_heads, spec.explicit_kv_dim)
    q_dir = q_dir / jnp.sqrt(jnp.sum(q_dir * q_dir, axis=-1, keepdims=True) + 1e-8)
    # Place non-unit perturbed fp32 key in slot 1 to verify zero-write does NOT renormalize untouched keys!
    stored_k0 = q_dir
    stored_k1 = -q_dir * 0.999123
    kv_st["kv_keys"] = kv_st["kv_keys"].at[:, :, 0, :].set(stored_k0).at[:, :, 1, :].set(stored_k1)
    kv_st["kv_valid"] = kv_st["kv_valid"].at[:, :, :2].set(True)
    # Equal starting age `0.0` with `dt_s = 0.0`! Attended slot 0 must stay strictly younger than slot 1!
    kv_st["kv_age"] = kv_st["kv_age"].at[:, :, :2].set(0.0)

    _, kv_st_after, _ = step_explicit_kv(
        kv_p, kv_st, x_probe, jnp.zeros((1,), dtype=jnp.int32), jnp.zeros((1,)), jnp.ones((1,)), jnp.ones((1,)), spec
    )
    assert float(jnp.max(jnp.abs(kv_st_after["kv_keys"][:, :, 1, :] - stored_k1))) == 0.0
    assert float(jnp.max(kv_st_after["kv_age"][0, :, 0])) == 0.0
    assert float(jnp.min(kv_st_after["kv_age"][0, :, 1])) > 0.0
    assert int(jnp.argmax(kv_st_after["kv_age"][0, 0, :2])) == 1  # Never evicts read-refreshed slot 0!


def test_comparator_param_byte_parity_zero_dead_edge_and_balanced_modality_rms() -> None:
    """Verify ±1.5% param parity & ±5% byte parity across Pilot/Discovery/Confirmation + zero dead params + balanced edge RMS."""
    for profile, lo_m, hi_m in (
        ("pilot_d256", 5_000_000, 20_000_000),
        ("discovery_45m", 30_000_000, 60_000_000),
        ("confirmation_135m", 124_000_000, 150_000_000),
    ):
        specs = build_wave_a_comparator_specs(profile, match_params_to_anchor=True)
        anchor = specs["slwm_v3_routed"]
        anchor_p = count_tree_params(jax.eval_shape(lambda k: init_slwm_v3_params(k, anchor), jax.random.PRNGKey(0)))
        anchor_b = compute_core_streaming_state_bytes(anchor, batch_size=1)
        assert lo_m <= anchor_p <= hi_m, f"{profile} anchor params={anchor_p:,} outside [{lo_m:,}, {hi_m:,}]"

        for name, sp in specs.items():
            n_p = count_tree_params(jax.eval_shape(lambda k, s=sp: init_slwm_v3_params(k, s), jax.random.PRNGKey(0)))
            p_err = abs(n_p - anchor_p) / float(anchor_p)
            assert p_err <= 0.010, f"{profile} {name} params={n_p:,} vs anchor={anchor_p:,} (err={p_err:.3%})"
            if profile == "confirmation_135m":
                assert lo_m <= n_p <= hi_m, f"Confirmation {name} params={n_p:,} outside [124M, 150M]"

            b_cmp = compute_core_streaming_state_bytes(sp, batch_size=1)
            if name in ("slwm_v3_no_routing", "slwm_v3_no_nowrite", "slwm_v3_no_zerolag", "single_bank_gated_deltanet"):
                assert b_cmp["memory_state_bytes"] == anchor_b["memory_state_bytes"]
            elif name != "sliding_window_gqa_eqparam":
                b_err = abs(b_cmp["total_core_state_bytes"] - anchor_b["total_core_state_bytes"]) / float(anchor_b["total_core_state_bytes"])
                assert b_err <= 0.05, f"{profile} {name} state bytes err={b_err:.3%}"

    # Zero dead audio edge params across frozen_codec_edge vs scratch_edge!
    p_frozen = init_slwm_v3_params(jax.random.PRNGKey(70), make_slwm_v3_spec("tiny_unit", representation_mode="frozen_codec_edge"))
    p_scratch = init_slwm_v3_params(jax.random.PRNGKey(71), make_slwm_v3_spec("tiny_unit", representation_mode="scratch_edge"))
    assert "edge_audio_rvq_emb" in p_frozen["edge"] and "edge_audio_cont_w" not in p_frozen["edge"]
    assert "edge_audio_cont_w" in p_scratch["edge"] and "edge_audio_rvq_emb" not in p_scratch["edge"]

    # Balanced initial activation RMS across ALL 5 modalities (`max / min < 2.0`)!
    sp_bal = make_slwm_v3_spec("pilot_d256", representation_mode="frozen_codec_edge")
    p_bal = init_slwm_v3_params(jax.random.PRNGKey(72), sp_bal)
    pkts_b = _make_dummy_packets(sp_bal, t_len=1, batch_size=16, seed=720)
    mod_rms = []
    for m_id in range(5):
        pkt_m = {k: v[0] for k, v in pkts_b.items()}
        pkt_m["modality_id"] = jnp.full((16,), m_id, dtype=jnp.int32)
        x_m = embed_packet_step(p_bal["edge"], pkt_m, sp_bal)
        mod_rms.append(float(jnp.sqrt(jnp.mean(x_m * x_m))))
    assert max(mod_rms) / max(min(mod_rms), 1e-8) < 2.0, f"Imbalanced modality init RMS: {mod_rms}"


def test_exact_base_shape_mup_masked_wd_and_production_adamw_width_invariance() -> None:
    """Verify exact 1.0 LR mult at d==d_base on all 8 backbones, masked WD, and production AdamW muP stability."""
    # 1. Exact `lr_mult == 1.0` on EVERY leaf at `d == d_base` across ALL 8 backbones!
    for bb in BACKBONE_TYPES:
        sp = make_slwm_v3_spec("tiny_unit", backbone_type=bb)
        p = init_slwm_v3_params(jax.random.PRNGKey(80), sp)
        mults = build_mup_lr_multipliers(p, sp)
        for m in jax.tree_util.tree_leaves(mults):
            assert float(m) == 1.0, f"{bb} at d==d_base had non-1.0 LR mult {m}"

    # 2. Strict weight-decay mask (`False` on all scalar/timescale/norm/bias leaves, `True` on 2D matrices)
    sp_r = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_routed")
    p_r = init_slwm_v3_params(jax.random.PRNGKey(81), sp_r)
    wd_m = build_weight_decay_mask(p_r)
    assert not wd_m["layers"][0]["mem"]["log_tau"]
    assert not wd_m["layers"][0]["mem"]["g_bias"]
    assert not wd_m["layers"][0]["mem"]["beta_bias"]
    assert not wd_m["layers"][0]["mem"]["router_b1"]
    assert not wd_m["layers"][0]["mem"]["router_b2"]
    assert not wd_m["layers"][0]["attn"]["norm_w"]
    assert not wd_m["layers"][0]["attn"]["rel_pos_bias"]
    assert not wd_m["gate_z0"]
    assert wd_m["layers"][0]["ffn"]["w_gate"] and wd_m["layers"][0]["ffn"]["w_down"]

    sp_kv = make_slwm_v3_spec("tiny_unit", backbone_type="explicit_bounded_kv")
    wd_kv = build_weight_decay_mask(init_slwm_v3_params(jax.random.PRNGKey(82), sp_kv))
    assert not wd_kv["layers"][0]["kv"]["thresh"]

    sp_ssm = make_slwm_v3_spec("tiny_unit", backbone_type="diag_ssm")
    wd_ssm = build_weight_decay_mask(init_slwm_v3_params(jax.random.PRNGKey(83), sp_ssm))
    assert not wd_ssm["layers"][0]["ssm"]["log_a"] and not wd_ssm["layers"][0]["ssm"]["dt_bias"]

    # 3. Real production masked `optax` AdamW 3-width (`128, 256, 512`) coordinate invariance vs SP!
    widths = (128, 256, 512)
    pkts, targs, tvalid = make_synthetic_multimodal_batch(sp_r, seq_len=24, batch_size=4, seed=808)
    z_rms_init, dz_mup, dz_sp = [], [], []
    for d in widths:
        sp = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_routed", d_model_override=d)
        p0 = init_slwm_v3_params(jax.random.PRNGKey(77), sp)
        st0 = init_core_state(sp, batch_size=4)
        z0, _, _ = scan_core(p0, st0, pkts, sp)
        z_rms_init.append(float(jnp.sqrt(jnp.mean(z0 * z0))))

        for use_mup, dest in ((True, dz_mup), (False, dz_sp)):
            tx = build_optimizer(p0, sp, base_lr=1e-3, weight_decay=0.01, use_mup_lr_scaling=use_mup)
            o_st = tx.init(p0)
            p_cur = p0
            for _ in range(5):
                loss, grads = jax.value_and_grad(
                    lambda pp: masked_multi_horizon_predictive_loss(pp, st0, pkts, targs, tvalid, sp)[0]
                )(p_cur)
                assert bool(jnp.isfinite(loss))
                assert all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads))
                upd, o_st = tx.update(grads, o_st, p_cur)
                p_cur = optax.apply_updates(p_cur, upd)
            z_end, _, _ = scan_core(p_cur, st0, pkts, sp)
            dest.append(float(jnp.sqrt(jnp.mean((z_end - z0) ** 2))))

    assert min(dz_mup) > 1e-4
    init_ratio = max(z_rms_init) / max(min(z_rms_init), 1e-8)
    mup_ratio = max(dz_mup) / max(min(dz_mup), 1e-8)
    sp_ratio = max(dz_sp) / max(min(dz_sp), 1e-8)
    assert init_ratio < 1.35
    assert mup_ratio < 1.65
    assert mup_ratio < sp_ratio


def test_phase3_observable_sensory_echo_oracle_phase2_verdict_and_launcher_safety(tmp_path: Path) -> None:
    """Verify Phase 3 real sensory echoes + 100% oracle, Phase 2 strict verdict fixtures, TPU v6 lite guard, non-deadlocking >128KiB watchdog, & strict teardown!"""
    spec = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_routed")
    cb = build_frozen_phase3_codebook(spec, vocab_size=8, codebook_seed=202601)
    for sm in ("clean", "distractor", "corrupt"):
        batch = generate_phase3_counterfactual_batch(spec, cb, batch_size=16, t_len=80, min_delay_after_corr=34, stress_mode=sm, seed=901)
        ov = verify_phase3_counterfactual_oracle(batch, cb, spec)
        assert ov["oracle_pass"] is True, f"Phase 3 oracle failed on {sm}: {ov}"
        assert ov["oracle_ab_flip_and_preserve_acc"] == 1.0
        assert ov["post_corr_pre_echo_max_abs_diff"] == 0.0
        # Verify real observable future sensory echoes ACTUALLY differ in `pkts["probe_feat"]` across A/B
        # and `targs` is purely derived from `pkts["probe_feat"]`!
        pa_pr = np.asarray(batch["pkts_a"]["probe_feat"])
        pb_pr = np.asarray(batch["pkts_b"]["probe_feat"])
        ta = np.asarray(batch["targs_a"])
        assert np.allclose(ta[:, :, 0:32], pa_pr[:, :, 0:32], atol=1e-6)
        echo_diff = np.max(np.abs(pa_pr[batch["t_pre_echo"] + 1 :] - pb_pr[batch["t_pre_echo"] + 1 :]), axis=(0, 2))
        assert bool(np.all(echo_diff > 1.0)), "Observable future sensory echo did not flip between A and B!"

    # Phase 2 strict LR transfer verdict tests (all 3 Astra counterexample fixtures!)
    lrs = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)

    def _cell(v: float, ok: bool = True) -> dict[str, object]:
        return {"val_loss_end_exact": v, "all_finite": ok}

    # Fixture 1: all optima at boundary -> MUST return `inconclusive`!
    g_bound = {
        "d256": {f"lr_{lr:.0e}": _cell(0.5 - 0.05 * i) for i, lr in enumerate(lrs)},
        "d512": {f"lr_{lr:.0e}": _cell(0.5 - 0.05 * i) for i, lr in enumerate(lrs)},
    }
    assert evaluate_lr_transfer_verdict(g_bound, ["d256", "d512"], lrs)["status"] == "inconclusive"

    # Fixture 2: interior optimum with another LR non-finite -> MUST return `inconclusive`!
    g_nf = {
        "d256": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.30), "lr_1e-03": _cell(0.20), "lr_3e-03": _cell(0.35), "lr_1e-02": _cell(float("nan"), False)},
        "d512": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.30), "lr_1e-03": _cell(0.20), "lr_3e-03": _cell(0.35), "lr_1e-02": _cell(0.50)},
    }
    assert evaluate_lr_transfer_verdict(g_nf, ["d256", "d512"], lrs)["status"] == "inconclusive"

    # Fixture 3: unrounded regret 0.0150004 > 0.015 -> MUST return `fail`!
    g_reg = {
        "d256": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.30), "lr_1e-03": _cell(0.20), "lr_3e-03": _cell(0.35), "lr_1e-02": _cell(0.50)},
        "d512": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.20), "lr_1e-03": _cell(0.2150004), "lr_3e-03": _cell(0.35), "lr_1e-02": _cell(0.50)},
    }
    assert evaluate_lr_transfer_verdict(g_reg, ["d256", "d512"], lrs)["status"] == "fail"

    # Fixture 1b: interior base optimum with wider-scale boundary optimum -> MUST ALSO return `inconclusive`!
    g_part_bound = {
        "d256": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.30), "lr_1e-03": _cell(0.20), "lr_3e-03": _cell(0.35), "lr_1e-02": _cell(0.50)},
        "d512": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.30), "lr_1e-03": _cell(0.21), "lr_3e-03": _cell(0.205), "lr_1e-02": _cell(0.20)},
    }
    assert evaluate_lr_transfer_verdict(g_part_bound, ["d256", "d512"], lrs)["status"] == "inconclusive"

    # Fixture 4: width passes at 1e-3 and depth passes at 3e-4, with 0.150 depth regret at 1e-3 -> combined MUST return `fail`!
    g_w_ok = {
        "d256": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.30), "lr_1e-03": _cell(0.20), "lr_3e-03": _cell(0.35), "lr_1e-02": _cell(0.50)},
        "d512": {"lr_1e-04": _cell(0.42), "lr_3e-04": _cell(0.31), "lr_1e-03": _cell(0.21), "lr_3e-03": _cell(0.36), "lr_1e-02": _cell(0.52)},
    }
    g_d_mismatch = {
        "L8": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.20), "lr_1e-03": _cell(0.35), "lr_3e-03": _cell(0.42), "lr_1e-02": _cell(0.50)},
        "L12": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.20), "lr_1e-03": _cell(0.35), "lr_3e-03": _cell(0.42), "lr_1e-02": _cell(0.50)},
    }
    vw = evaluate_lr_transfer_verdict(g_w_ok, ["d256", "d512"], lrs)
    vd = evaluate_lr_transfer_verdict(g_d_mismatch, ["L8", "L12"], lrs)
    assert vw["status"] == "pass" and vd["status"] == "pass"
    _, _, comb_status = combine_phase2_verdict(vw, vd, (256, 512))
    assert comb_status == "fail"

    # Positive control: clean all-finite interior optimum matching across BOTH width & depth -> `pass`!
    g_d_ok = {
        "L8": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.30), "lr_1e-03": _cell(0.20), "lr_3e-03": _cell(0.35), "lr_1e-02": _cell(0.50)},
        "L12": {"lr_1e-04": _cell(0.42), "lr_3e-04": _cell(0.31), "lr_1e-03": _cell(0.205), "lr_3e-03": _cell(0.36), "lr_1e-02": _cell(0.52)},
    }
    vd_ok = evaluate_lr_transfer_verdict(g_d_ok, ["L8", "L12"], lrs)
    assert vd_ok["status"] == "pass"
    _, _, comb_ok = combine_phase2_verdict(vw, vd_ok, (256, 512))
    assert comb_ok == "pass"

    # Fixture 1c: tie hitting upper grid endpoint `[0.40, 0.30, 0.20, 0.20, 0.20]` -> MUST return `inconclusive`!
    g_tie_bound = {
        "d256": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.30), "lr_1e-03": _cell(0.20), "lr_3e-03": _cell(0.20), "lr_1e-02": _cell(0.20)},
        "d512": {"lr_1e-04": _cell(0.40), "lr_3e-04": _cell(0.30), "lr_1e-03": _cell(0.20), "lr_3e-03": _cell(0.35), "lr_1e-02": _cell(0.50)},
    }
    assert evaluate_lr_transfer_verdict(g_tie_bound, ["d256", "d512"], lrs)["status"] == "inconclusive"

    # TPU v6e / v6 lite exact device kind check (rejects spoofed strings!)
    assert is_tpu_v6e_device_kind("TPU v6e") is True
    assert is_tpu_v6e_device_kind("TPU v6 lite") is True
    assert is_tpu_v6e_device_kind("non-TPU v6e simulator") is False
    assert is_tpu_v6e_device_kind("TPU v5 lite") is False
    assert is_tpu_v6e_device_kind("cpu") is False

    # Pre-phase projected reserve check: 41.90 CU spent at 12 CU/hr & 45s -> 42.05 >= 42.00 -> MUST trip!
    tripped_419, proj_419, _ = projected_spend_exceeds_reserve(41.90, 12.0, max_spend_cu=45.0, poll_interval_s=45)
    assert tripped_419 is True and proj_419 > 42.0

    # Watchdog file-backed logging MUST handle > 128 KiB stdout without pipe deadlock!
    rc, out_txt, _, tripped = _run_subprocess_with_watchdog(
        [sys.executable, "-c", "print('X' * 140000)"],
        env={},
        log_dir=tmp_path / "wd_logs",
        log_prefix="big_out",
        cfg_path=None,
        bal_start=100.0,
        prior_spent_cu=0.0,
        max_spend_cu=45.0,
        t_alloc_start=time.monotonic(),
        phase_timeout_s=30,
    )
    assert rc == 0 and (not tripped) and len(out_txt) >= 140_000

    # Strict `verified_stop_session` MUST reject: (a) stop rc=1 even when sessions rc=0,
    # (b) empty sessions stdout, (c) unnamed `[?]`, (d) timeout on attempt 1 recovers on attempt 2!
    def _stop_fail_sessions_ok(*args, **kwargs):
        if args[0] == "stop":
            return subprocess.CompletedProcess(args, returncode=1, stdout="", stderr="stop failed")
        return subprocess.CompletedProcess(args, returncode=0, stdout="[colab] No active sessions found on server.\n", stderr="")

    def _empty_sessions(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")

    def _unnamed_runner(*args, **kwargs):
        if args[0] == "sessions":
            return subprocess.CompletedProcess(args, returncode=0, stdout="[?] endpoint=abc | TPU | V6E1\n", stderr="")
        return subprocess.CompletedProcess(args, returncode=0, stdout="[colab] Session terminated.\n", stderr="")

    call_cnt = {"n": 0}

    def _timeout_then_ok(*args, **kwargs):
        if args[0] == "stop":
            call_cnt["n"] += 1
            if call_cnt["n"] == 1:
                raise subprocess.TimeoutExpired(cmd="colab stop", timeout=120)
            return subprocess.CompletedProcess(args, returncode=0, stdout="[colab] Session terminated.\n", stderr="")
        return subprocess.CompletedProcess(args, returncode=0, stdout="[colab] No active sessions found on server.\n", stderr="")

    def _stderr_auth_fail(*args, **kwargs):
        if args[0] == "sessions":
            return subprocess.CompletedProcess(
                args, returncode=0, stdout="[colab] No active sessions found on server.\n", stderr="Warning: credential auth failed\n"
            )
        return subprocess.CompletedProcess(args, returncode=0, stdout="[colab] Session terminated.\n", stderr="")

    assert verified_stop_session("slwm_v3_pilot", tmp_path / "c.json", runner=_stop_fail_sessions_ok, sleep_s=0.0) is False
    assert verified_stop_session("slwm_v3_pilot", tmp_path / "c.json", runner=_empty_sessions, sleep_s=0.0) is False
    assert verified_stop_session("slwm_v3_pilot", tmp_path / "c.json", runner=_stderr_auth_fail, sleep_s=0.0) is False
    assert verified_stop_session("slwm_v3_pilot", tmp_path / "c.json", runner=_unnamed_runner, sleep_s=0.0) is False
    assert verified_stop_session("slwm_v3_pilot", tmp_path / "c.json", runner=_timeout_then_ok, sleep_s=0.0) is True

    # Reusable Phase 2 receipt lock MUST reject bare `{"status": "pass"}`, bool LR `true`, off-grid LR `0.05`, and spoofed device!
    bad_mup = tmp_path / "phase_mup_check_receipt.json"
    bad_mup.write_text('{"status": "pass"}\n')
    with pytest.raises(RuntimeError, match="PHASE 2 PREREQUISITE LOCK"):
        validate_reusable_mup_receipt(bad_mup, expected_bundle_sha256="a" * 64, require_tpu_v6e=True)

    bool_lr_mup = tmp_path / "bool_lr_mup.json"
    bool_lr_mup.write_text(json.dumps({
        "status": "pass", "smoke_cpu": False, "device_kind": "TPU v6 lite",
        "widths": [256, 512, 768], "depths": [8, 12, 16], "lrs": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "pilot_bundle_sha256": "a" * 64, "recommended_base_lr": True,
    }))
    with pytest.raises(RuntimeError, match="PHASE 2 PREREQUISITE LOCK"):
        validate_reusable_mup_receipt(bool_lr_mup, expected_bundle_sha256="a" * 64, require_tpu_v6e=True)

    offgrid_mup = tmp_path / "offgrid_mup.json"
    offgrid_mup.write_text(json.dumps({
        "status": "pass", "smoke_cpu": False, "device_kind": "TPU v6 lite",
        "widths": [256, 512, 768], "depths": [8, 12, 16], "lrs": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "pilot_bundle_sha256": "a" * 64, "recommended_base_lr": 0.05,
    }))
    with pytest.raises(RuntimeError, match="PHASE 2 PREREQUISITE LOCK"):
        validate_reusable_mup_receipt(offgrid_mup, expected_bundle_sha256="a" * 64, require_tpu_v6e=True)

    with pytest.raises(RuntimeError, match="PHASE 2 PREREQUISITE LOCK"):
        validate_phase_prerequisites(["dev_ablation"], tmp_path / "no_mup", expected_bundle_sha256="a" * 64)

    fake_reports = tmp_path / "reports/v3"
    fake_reports.mkdir(parents=True)
    for fname in (
        "gate1_control_receipt.json",
        "gate1_speech_prototype_receipt.json",
        "gate2_split_ledger_receipt.json",
        "gate2_codec_ceilings_receipt.json",
        "gate3_architecture_and_mup_receipt.json",
    ):
        (fake_reports / fname).write_text("{}\n")
    with pytest.raises(RuntimeError, match="PRE-FLIGHT GATE LOCK"):
        verify_gates_1_to_3_passed(tmp_path)
