"""SLWM-v4 Gate 3 Comprehensive Unit Tests: Architecture Parity, Causality, Step==Seq, Grad Isolation & Streaming."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from data.v4_heartbeat_contract import (
    MOD_AUDIO,
    MOD_MOTOR,
    MOD_TEXT,
    MOD_VIDEO,
    SLWMv4HeartbeatSpec,
    UNIFIED_ACTION_DIM,
    UNIFIED_PROPRIO_DIM,
    build_interleaved_heartbeat_stream,
)
from data.v4_streaming_loader import (
    V4SinglePassMixtureLoader,
    dequantize_wan21_patches_int8,
    load_excluded_episode_ids,
    slice_v4_shard_window,
)
from models.v4 import (
    SLWMv4Spec,
    V4_S3_DISCOVERY_RUN_IDS,
    apply_v4_core_seq,
    build_v4_s3_discovery_specs,
    compute_v4_core_streaming_state_bytes,
    compute_v4_training_loss,
    count_state_tree_bytes,
    count_tree_params,
    estimate_v4_active_core_param_count,
    estimate_v4_core_param_count,
    init_v4_core_state,
    init_v4_model_params,
    make_slwm_v4_spec,
    make_v4_mup_optimizer,
    run_v4_live_heartbeat_rollout,
    step_v4_core,
    summarize_v4_accounting,
)
from models.v4.heads import (
    apply_audio_head_teacher_forced,
    apply_policy_commit_gate,
    compute_v4_heads_loss,
)


def _make_synthetic_quadmodal_batch(
    *,
    t_len: int = 48,
    batch_size: int = 2,
    seed: int = 0,
    with_mid_resets: bool = True,
    with_holes: bool = False,
    unlabeled_success: bool = False,
    active_act_dims: int = 14,
    active_prop_dims: int = 20,
) -> dict[str, jax.Array]:
    """Build a physically faithful quad-modal interleaved `[T, B, ...]` batch via `build_interleaved_heartbeat_stream`."""
    rng = np.random.default_rng(seed)
    hb_spec = SLWMv4HeartbeatSpec()
    n_hb = max(3, (t_len // 20) + 2)
    windows = []
    for b in range(batch_size):
        act_m = np.zeros((UNIFIED_ACTION_DIM,), dtype=np.float32)
        act_m[:active_act_dims] = 1.0
        prop_m = np.zeros((UNIFIED_PROPRIO_DIM,), dtype=np.float32)
        prop_m[:active_prop_dims] = 1.0
        st = build_interleaved_heartbeat_stream(
            video_groups=rng.normal(0.0, 0.5, size=(n_hb, 15, 1024)).astype(np.float16),
            audio_codes=rng.integers(0, 2048, size=(2 * n_hb, 16), dtype=np.int16),
            action_seq=rng.normal(0.0, 0.4, size=(4 * n_hb, UNIFIED_ACTION_DIM)).astype(np.float32),
            proprio_seq=rng.normal(0.0, 0.4, size=(4 * n_hb, UNIFIED_PROPRIO_DIM)).astype(np.float32),
            action_mask=act_m,
            proprio_mask=prop_m,
            text_by_heartbeat={h: f"hb{h}_b{b} ".encode("utf-8") for h in range(n_hb)},
            spec=hb_spec,
        )
        n_tok = int(st["modality_id"].size)
        assert n_tok >= t_len
        # Pack into shard-like dict and slice cleanly
        vid_f = np.zeros((t_len, 1024), dtype=np.float32)
        aud_c = np.zeros((t_len, 16), dtype=np.int32)
        act_f = np.zeros((t_len, UNIFIED_ACTION_DIM), dtype=np.float32)
        prop_f = np.zeros((t_len, UNIFIED_PROPRIO_DIM), dtype=np.float32)
        v_idx = st["video_idx"][:t_len]
        a_idx = st["audio_idx"][:t_len]
        m_idx = st["motor_idx"][:t_len]
        vid_f[v_idx >= 0] = st["video_feat"][v_idx[v_idx >= 0]].astype(np.float32)
        aud_c[a_idx >= 0] = st["audio_codes"][a_idx[a_idx >= 0]].astype(np.int32)
        act_f[m_idx >= 0] = st["action_feat"][m_idx[m_idx >= 0]].astype(np.float32)
        prop_f[m_idx >= 0] = st["proprio_feat"][m_idx[m_idx >= 0]].astype(np.float32)

        rst = st["reset_mask"][:t_len].astype(np.bool_)
        vld = st["valid_mask"][:t_len].astype(np.float32)
        tvld = st["target_valid_mask"][:t_len].astype(np.float32)
        if with_mid_resets and t_len >= 24:
            rst[17 + (b % 3)] = True
            if t_len >= 36:
                rst[31 + (b % 2)] = True
        if with_holes and t_len >= 24:
            vld[5 + b] = 0.0
            tvld[5 + b] = 0.0
            vld[22 + b] = 0.0
            tvld[22 + b] = 0.0

        succ_val = -1 if unlabeled_success else (b % 2)
        windows.append(
            {
                "modality_id": st["modality_id"][:t_len].astype(np.int32),
                "heartbeat_id": st["heartbeat_id"][:t_len].astype(np.int32),
                "dt_s": st["dt_s"][:t_len].astype(np.float32),
                "reset_mask": rst,
                "valid_mask": vld,
                "target_valid_mask": tvld,
                "reliability": st["reliability"][:t_len].astype(np.float32),
                "spatial_pos": st["spatial_pos"][:t_len].astype(np.int32),
                "text_byte": st["text_byte"][:t_len].astype(np.int32),
                "video_feat": vid_f,
                "audio_codes": aud_c,
                "action_feat": act_f,
                "proprio_feat": prop_f,
                "action_mask": np.broadcast_to(act_m[None, :], (t_len, UNIFIED_ACTION_DIM)).copy(),
                "proprio_mask": np.broadcast_to(prop_m[None, :], (t_len, UNIFIED_PROPRIO_DIM)).copy(),
                "episode_success": np.full((t_len,), succ_val, dtype=np.int32),
            }
        )
    return {k: jnp.asarray(np.stack([w[k] for w in windows], axis=1)) for k in windows[0]}


def test_all_17_v4s3_configs_exact_param_and_byte_parity() -> None:
    """Certify exact analytic == actual JAX param counts & state bytes across ALL 17 V4-S3 configs."""
    tiny_specs = build_v4_s3_discovery_specs("tiny_unit")
    assert tuple(tiny_specs.keys()) == V4_S3_DISCOVERY_RUN_IDS

    key = jax.random.PRNGKey(0)
    for rid, sp in tiny_specs.items():
        key, k_init = jax.random.split(key)
        params = init_v4_model_params(k_init, sp)
        state = init_v4_core_state(sp, batch_size=2)
        acct = summarize_v4_accounting(sp, params, init_v4_core_state(sp, batch_size=1))

        assert count_tree_params(params["core"]) == estimate_v4_core_param_count(sp), f"Param mismatch on {rid}"
        assert count_state_tree_bytes(state) == compute_v4_core_streaming_state_bytes(sp, batch_size=2)["total_core_state_bytes"], (
            f"State byte mismatch on {rid}"
        )
        assert acct["core_params_actual"] == acct["core_params_analytic"]
        assert acct["core_state_bytes_b1_actual"] == acct["core_state_bytes_b1_analytic"]

    # Certify exact matched core accounting & zero-M_anchor byte parity at full `discovery_45m` scale!
    specs_45m = build_v4_s3_discovery_specs("discovery_45m")
    a0_p = estimate_v4_core_param_count(specs_45m["A0_3to1_single_bank"])
    a0_b = compute_v4_core_streaming_state_bytes(specs_45m["A0_3to1_single_bank"], 1)["total_core_state_bytes"]
    assert 44_000_000 <= a0_p <= 46_500_000

    for rid, sp in specs_45m.items():
        p_act = estimate_v4_active_core_param_count(sp)
        rel_err = abs(p_act - a0_p) / float(a0_p)
        assert rel_err <= 0.005, f"{rid} active core params drift {rel_err:.4f} > 0.5%"

    b1_b = compute_v4_core_streaming_state_bytes(specs_45m["B1_sw_gqa"], 1)["total_core_state_bytes"]
    c5_b = compute_v4_core_streaming_state_bytes(specs_45m["C5_routed_2bank"], 1)["total_core_state_bytes"]
    c6_b = compute_v4_core_streaming_state_bytes(specs_45m["C6_routed_4bank"], 1)["total_core_state_bytes"]
    assert abs(b1_b - a0_b) / float(a0_b) <= 0.01, "B1_sw_gqa must match A0 streaming state bytes within 1%"
    assert c5_b / float(a0_b) <= 1.02, "C5_routed_2bank must avoid M_anchor 2x state bloat"
    assert c6_b / float(a0_b) <= 1.02, "C6_routed_4bank must avoid M_anchor 2x state bloat"


@pytest.mark.parametrize(
    "run_id",
    [
        "A0_3to1_single_bank",
        "A1_5to1_single_bank",
        "A2_1to1_single_bank",
        "B0_v3_ref",
        "B1_sw_gqa",
        "C1_no_conv1d",
        "C2_beta_0_1",
        "C3_no_ogate",
        "C4_rank128",
        "C5_routed_2bank",
        "C6_routed_4bank",
        "C7_ms_moe_ffn",
        "O1_jepa_sync",
        "H1_huber_video_abl",
        "H2_flat_rvq_abl",
        "H3_self_forcing",
        "H4_no_policy_gate",
    ],
)
def test_strict_zero_future_lookahead_causality(run_id: str) -> None:
    """Mutating ANY future packet `t > t_cut` MUST produce EXACT 0.0 change in `z_seq[:t_cut+1]`."""
    sp = make_slwm_v4_spec("tiny_unit", run_id=run_id)
    params = init_v4_model_params(jax.random.PRNGKey(11), sp)
    state0 = init_v4_core_state(sp, batch_size=2)
    batch = _make_synthetic_quadmodal_batch(t_len=36, batch_size=2, seed=7, with_mid_resets=True, with_holes=True)

    t_cut = 19
    batch_mut = dict(batch)
    batch_mut["video_feat"] = batch["video_feat"].at[t_cut + 1 :].set(batch["video_feat"][t_cut + 1 :] + 3.7)
    batch_mut["audio_codes"] = batch["audio_codes"].at[t_cut + 1 :].set((batch["audio_codes"][t_cut + 1 :] + 113) % 2048)
    batch_mut["text_byte"] = batch["text_byte"].at[t_cut + 1 :].set((batch["text_byte"][t_cut + 1 :] + 17) % 256)
    batch_mut["action_feat"] = batch["action_feat"].at[t_cut + 1 :].set(batch["action_feat"][t_cut + 1 :] - 2.1)
    batch_mut["proprio_feat"] = batch["proprio_feat"].at[t_cut + 1 :].set(batch["proprio_feat"][t_cut + 1 :] + 1.9)
    batch_mut["dt_s"] = batch["dt_s"].at[t_cut + 1 :].set(batch["dt_s"][t_cut + 1 :] + 0.5)

    z_clean, _, _ = apply_v4_core_seq(params, state0, batch, sp, matmul_dtype=jnp.float32, wy_chunk_size=8, gqa_chunk_size=8, use_remat=False)
    z_mut, _, _ = apply_v4_core_seq(params, state0, batch_mut, sp, matmul_dtype=jnp.float32, wy_chunk_size=8, gqa_chunk_size=8, use_remat=False)

    prefix_diff = float(jnp.max(jnp.abs(z_clean[: t_cut + 1] - z_mut[: t_cut + 1])))
    future_diff = float(jnp.max(jnp.abs(z_clean[t_cut + 1 :] - z_mut[t_cut + 1 :])))
    assert prefix_diff == 0.0, f"Future leak detected on {run_id}: prefix_diff={prefix_diff}"
    assert future_diff > 1e-4, f"Mutated future had unexpectedly zero effect on {run_id}"


def test_depth_transformer_intra_tick_causality() -> None:
    """Mutating RVQ codebooks `k > k_cut` MUST produce EXACT 0.0 change in logits for `k <= k_cut`."""
    sp = make_slwm_v4_spec("tiny_unit", run_id="A0_3to1_single_bank")
    params = init_v4_model_params(jax.random.PRNGKey(21), sp)
    z = jax.random.normal(jax.random.PRNGKey(22), (4, 2, sp.d_model), dtype=jnp.float32)
    codes = jax.random.randint(jax.random.PRNGKey(23), (4, 2, sp.audio_codebooks), 0, sp.audio_vocab_size)

    lg_base = apply_audio_head_teacher_forced(params["heads"]["head_audio"], z, codes, sp, matmul_dtype=jnp.float32)
    for k_cut in (0, 5, 14):
        c_mut = codes.at[..., k_cut + 1 :].set((codes[..., k_cut + 1 :] + 77) % sp.audio_vocab_size)
        lg_mut = apply_audio_head_teacher_forced(params["heads"]["head_audio"], z, c_mut, sp, matmul_dtype=jnp.float32)
        assert float(jnp.max(jnp.abs(lg_base[..., : k_cut + 1, :] - lg_mut[..., : k_cut + 1, :]))) == 0.0


@pytest.mark.parametrize(
    "run_id",
    [
        "A0_3to1_single_bank",
        "A1_5to1_single_bank",
        "A2_1to1_single_bank",
        "B0_v3_ref",
        "B1_sw_gqa",
        "C1_no_conv1d",
        "C2_beta_0_1",
        "C3_no_ogate",
        "C4_rank128",
        "C5_routed_2bank",
        "C6_routed_4bank",
        "C7_ms_moe_ffn",
    ],
)
@pytest.mark.parametrize("with_holes", [False, True])
def test_step_vs_seq_bit_exact_equivalence(run_id: str, with_holes: bool) -> None:
    """Certify `apply_v4_core_seq` matches step-by-step `step_v4_core` across resets, holes & non-zero init state."""
    sp = make_slwm_v4_spec("tiny_unit", run_id=run_id)
    params = init_v4_model_params(jax.random.PRNGKey(31), sp)
    batch_warm = _make_synthetic_quadmodal_batch(t_len=24, batch_size=2, seed=101, with_mid_resets=False, with_holes=False)
    # Warm up into realistic NON-ZERO initial state (tests cross-chunk carry!)
    _, state_warm, _ = apply_v4_core_seq(
        params, init_v4_core_state(sp, 2), batch_warm, sp, matmul_dtype=jnp.float32, wy_chunk_size=8, gqa_chunk_size=8, use_remat=False
    )

    batch = _make_synthetic_quadmodal_batch(t_len=38, batch_size=2, seed=202, with_mid_resets=True, with_holes=with_holes)
    # Ensure step 0 carries from state_warm on lane 1 (reset=False) and resets on lane 0 (reset=True)!
    batch["reset_mask"] = batch["reset_mask"].at[0, 0].set(True).at[0, 1].set(False)

    z_seq, fin_seq, _ = apply_v4_core_seq(
        params,
        state_warm,
        batch,
        sp,
        matmul_dtype=jnp.float32,
        wy_chunk_size=12,
        gqa_chunk_size=12,
        use_remat=False,
        assume_all_valid=not with_holes,
    )

    st_step = state_warm
    z_steps = []
    for t in range(batch["modality_id"].shape[0]):
        pkt_t = {k: v[t] for k, v in batch.items()}
        z_t, st_step, _ = step_v4_core(params, st_step, pkt_t, sp, matmul_dtype=jnp.float32)
        z_steps.append(z_t)
    z_step_all = jnp.stack(z_steps, axis=0)

    z_diff = float(jnp.max(jnp.abs(z_seq - z_step_all)))
    assert z_diff < 1e-4, f"step vs seq z_seq mismatch on {run_id} (holes={with_holes}): max_abs_diff={z_diff}"

    leaves_seq = jax.tree_util.tree_leaves(fin_seq)
    leaves_stp = jax.tree_util.tree_leaves(st_step)
    assert len(leaves_seq) == len(leaves_stp)
    for a, b in zip(leaves_seq, leaves_stp):
        if a.dtype == jnp.bool_ or jnp.issubdtype(a.dtype, jnp.integer):
            assert bool(jnp.array_equal(a, b)), f"Exact discrete state mismatch on {run_id} (holes={with_holes})"
        else:
            # bf16 ring KV has ~1e-2 ULP quantization vs fp32 arithmetic; float32 M/x_ema/conv_buf < 1e-4!
            tol = 2e-2 if a.dtype == jnp.bfloat16 else 1e-4
            assert float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) < tol, (
                f"State mismatch on {run_id} (holes={with_holes}, dtype={a.dtype})"
            )


def test_householder_beta_0_to_2_long_horizon_stability() -> None:
    """Verify Householder `beta in (0, 2)` remains energy-bounded with zero NaN/Inf across 256 oscillatory steps."""
    for rid in ("A0_3to1_single_bank", "C6_routed_4bank"):
        sp = make_slwm_v4_spec("tiny_unit", run_id=rid)
        assert sp.beta_max == 2.0
        params = init_v4_model_params(jax.random.PRNGKey(41), sp)
        batch = _make_synthetic_quadmodal_batch(t_len=256, batch_size=2, seed=42, with_mid_resets=False, with_holes=False)
        # Amplify input amplitude to stress-test Householder stability
        batch["video_feat"] = batch["video_feat"] * 3.0
        z_seq, fin_st, diag = apply_v4_core_seq(
            params, init_v4_core_state(sp, 2), batch, sp, matmul_dtype=jnp.float32, wy_chunk_size=64, use_remat=False
        )
        assert bool(jnp.all(jnp.isfinite(z_seq)))
        assert float(jnp.max(diag["frob_max"])) < 5e3
        for leaf in jax.tree_util.tree_leaves(fin_st):
            assert bool(jnp.all(jnp.isfinite(leaf)))


def test_exact_zero_gradient_isolation_gates() -> None:
    """Certify EXACT 0.0 gradient isolation for PolicyCommitGate, masked targets, inactive DoFs & unlabeled success."""
    sp = make_slwm_v4_spec("tiny_unit", run_id="A0_3to1_single_bank")
    params = init_v4_model_params(jax.random.PRNGKey(51), sp)
    batch = _make_synthetic_quadmodal_batch(
        t_len=48,
        batch_size=2,
        seed=52,
        with_mid_resets=True,
        active_act_dims=14,
        active_prop_dims=20,
        unlabeled_success=False,
    )

    # (a) PolicyCommitGate -> World Core & Edge MUST have EXACT 0.0 gradient!
    def _policy_only_loss(p):
        z, _, _ = apply_v4_core_seq(p, init_v4_core_state(sp, 2), batch, sp, matmul_dtype=jnp.float32, use_remat=False)
        ev = jnp.ones((48, 2, 5), dtype=jnp.float32) * 0.5
        lg, _ = apply_policy_commit_gate(p["heads"]["policy_gate"], z, ev)
        return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(lg, jnp.zeros((48, 2), dtype=jnp.int32)))

    grads_pol = jax.grad(_policy_only_loss)(params)
    core_grad_max = max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(grads_pol["core"]))
    edge_grad_max = max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(grads_pol["edge"]))
    gate_grad_max = max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(grads_pol["heads"]["policy_gate"]))
    assert core_grad_max == 0.0, f"PolicyCommitGate leaked gradient {core_grad_max} into world core!"
    assert edge_grad_max == 0.0, f"PolicyCommitGate leaked gradient {edge_grad_max} into edge encoder!"
    assert gate_grad_max > 1e-6, "PolicyCommitGate itself must receive non-zero gradient!"

    # (b) target_valid_mask == 0 -> EXACT 0.0 gradient on all predictive heads!
    batch_zero_t = dict(batch)
    batch_zero_t["target_valid_mask"] = jnp.zeros_like(batch["target_valid_mask"])

    def _pred_loss_no_pol(p, b):
        z, _, _ = apply_v4_core_seq(p, init_v4_core_state(sp, 2), b, sp, matmul_dtype=jnp.float32, use_remat=False)
        loss, _ = compute_v4_heads_loss(
            p["heads"],
            z,
            b,
            sp,
            jax.random.PRNGKey(53),
            matmul_dtype=jnp.float32,
            loss_weights={"text": 1.0, "audio": 1.0, "video": 1.0, "motor": 1.0, "success": 0.25, "jepa": 0.0, "policy": 0.0},
        )
        return loss

    grads_zero = jax.grad(_pred_loss_no_pol)(params, batch_zero_t)
    assert max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(grads_zero)) == 0.0

    # (c) Inactive action_mask[14:] == 0 and proprio_mask[20:] == 0 -> EXACT 0.0 gradient on inactive columns!
    grads_full = jax.grad(_pred_loss_no_pol)(params, batch)
    hm_g = grads_full["heads"]["head_motor"]
    assert float(jnp.max(jnp.abs(hm_g["w_act"][:, 14:]))) == 0.0, "Inactive action DoFs received non-zero gradient!"
    assert float(jnp.max(jnp.abs(hm_g["b_act"][14:]))) == 0.0
    assert float(jnp.max(jnp.abs(hm_g["w_prop"][:, 20:]))) == 0.0, "Inactive proprio channels received non-zero gradient!"
    assert float(jnp.max(jnp.abs(hm_g["b_prop"][20:]))) == 0.0
    assert float(jnp.max(jnp.abs(hm_g["w_act"][:, :14]))) > 1e-6, "Active action DoFs must receive gradient!"
    assert float(jnp.max(jnp.abs(hm_g["w_prop"][:, :20]))) > 1e-6, "Active proprio channels must receive gradient!"

    # (d) Unlabeled episode_success == -1 -> EXACT 0.0 gradient on success head!
    batch_unlab = dict(batch)
    batch_unlab["episode_success"] = jnp.full_like(batch["episode_success"], -1)
    grads_unlab = jax.grad(_pred_loss_no_pol)(params, batch_unlab)
    assert float(jnp.max(jnp.abs(grads_unlab["heads"]["head_motor"]["w_succ"]))) == 0.0
    assert float(jnp.max(jnp.abs(grads_unlab["heads"]["head_motor"]["b_succ"]))) == 0.0
    assert float(jnp.max(jnp.abs(hm_g["w_succ"]))) > 1e-6, "Labeled episode_success must receive gradient!"


@pytest.mark.parametrize(
    "run_id",
    [
        "A0_3to1_single_bank",
        "C6_routed_4bank",
        "O1_jepa_sync",
        "H1_huber_video_abl",
        "H2_flat_rvq_abl",
        "H3_self_forcing",
        "H4_no_policy_gate",
    ],
)
def test_single_batch_overfit_and_mup_optimizer(run_id: str) -> None:
    """Verify training step & muP optimizer converge cleanly across all objective & head variants."""
    sp = make_slwm_v4_spec("tiny_unit", run_id=run_id)
    params = init_v4_model_params(jax.random.PRNGKey(61), sp)
    if run_id == "H4_no_policy_gate":
        assert "policy_gate" not in params["heads"]
    else:
        assert "policy_gate" in params["heads"]

    batch = _make_synthetic_quadmodal_batch(t_len=48, batch_size=2, seed=62, with_mid_resets=False)
    opt = make_v4_mup_optimizer(params, sp, peak_lr=2e-3, total_steps=20, warmup_steps=2)
    opt_state = opt.init(params)

    @jax.jit
    def _train_step(p, ost, rng):
        def _loss_fn(pp):
            return compute_v4_training_loss(pp, batch, sp, rng, matmul_dtype=jnp.float32, wy_chunk_size=16, use_remat=False)

        (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(p)
        updates, new_ost = opt.update(grads, ost, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_ost, loss, metrics

    losses = []
    for step_i in range(8):
        params, opt_state, loss_val, metrics = _train_step(params, opt_state, jax.random.PRNGKey(100 + step_i))
        assert bool(jnp.isfinite(loss_val)), f"Non-finite loss on {run_id} step {step_i}"
        for mk, mv in metrics.items():
            assert bool(jnp.isfinite(mv)), f"Non-finite metric {mk} on {run_id}"
        losses.append(float(loss_val))

    assert losses[-1] < losses[0], f"{run_id} failed to decrease single-batch loss: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_continuous_2min_750_heartbeat_flat_o1_streaming_rollout() -> None:
    """Certify continuous 2.0-minute (750 heartbeats = 120.0 s) live streaming in strictly flat O(1) state!"""
    sp = make_slwm_v4_spec("tiny_unit", run_id="A0_3to1_single_bank")
    params = init_v4_model_params(jax.random.PRNGKey(71), sp)
    obs_v = jax.random.normal(jax.random.PRNGKey(72), (5, 15, 1024), dtype=jnp.float32) * 0.3
    obs_a = jax.random.randint(jax.random.PRNGKey(73), (10, 16), 0, 2048)

    res = run_v4_live_heartbeat_rollout(
        params,
        sp,
        jax.random.PRNGKey(74),
        n_heartbeats=750,
        generate_video=True,
        generate_audio=True,
        generate_motor=True,
        obs_video=obs_v,
        obs_audio=obs_a,
        video_flow_nfe=1,  # Keep CPU unit test < 10 s while exercising full 750-heartbeat live loop
        matmul_dtype=jnp.float32,
    )
    assert res["n_heartbeats"] == 750
    assert res["duration_sec"] == 120.0
    assert res["state_bytes_flat_o1"] is True
    assert res["init_state_bytes"] == res["final_state_bytes"]
    assert res["video_groups"].shape == (750, 15, 1024)
    assert res["audio_codes"].shape == (1500, 16)
    assert res["action_seq"].shape == (3000, UNIFIED_ACTION_DIM)
    assert res["proprio_seq"].shape == (3000, UNIFIED_PROPRIO_DIM)
    assert bool(jnp.all(jnp.isfinite(res["video_groups"])))
    assert bool(jnp.all(jnp.isfinite(res["action_seq"])))
    assert res["epistemic_labels"]["video"][:5] == ["observed"] * 5
    assert all(lbl in ("predicted", "imagined") for lbl in res["epistemic_labels"]["video"][5:])
    assert len(res["policy_actions"]) == 750


def test_streaming_loader_int8_dequant_split_isolation_and_exclusions(tmp_path: Path) -> None:
    """Certify certified int8 video dequantization, single-pass StopIteration, split isolation & exclusion blocking."""
    rng = np.random.default_rng(81)
    # 1. int8 quantization round-trip PSNR > 55 dB
    raw_v = rng.normal(0.0, 0.8, size=(30, 1024)).astype(np.float16)
    x = raw_v.astype(np.float32)
    xmin = np.min(x, axis=-1, keepdims=True)
    xmax = np.max(x, axis=-1, keepdims=True)
    scale = np.maximum((xmax - xmin) / 255.0, 1e-8)
    vq = np.clip(np.round((x - xmin) / scale), 0.0, 255.0).astype(np.uint8)
    rec_v = dequantize_wan21_patches_int8(vq, xmin.astype(np.float16), scale.astype(np.float16))
    mse = float(np.mean((x - rec_v) ** 2))
    dyn_range = float(np.max(x) - np.min(x))
    psnr = 20.0 * np.log10(dyn_range / np.sqrt(max(mse, 1e-12)))
    assert psnr > 58.0, f"int8 dequant dynamic-range PSNR too low: {psnr:.2f} dB"

    # 2. Build realistic local `.npz` + `_meta.json` shard containing one REAL excluded ID from Gate 2 manifest!
    excluded = load_excluded_episode_ids()
    assert len(excluded) == 39
    excl_corpus, excl_split, excl_eid = sorted(excluded)[0]

    cdir = tmp_path / excl_corpus
    cdir.mkdir(parents=True, exist_ok=True)
    act_m0 = np.zeros((UNIFIED_ACTION_DIM,), dtype=np.uint8)
    act_m0[:8] = 1
    prop_m0 = np.zeros((UNIFIED_PROPRIO_DIM,), dtype=np.uint8)
    prop_m0[:12] = 1
    st0 = build_interleaved_heartbeat_stream(
        video_groups=rng.normal(0.0, 0.5, size=(4, 15, 1024)).astype(np.float16),
        audio_codes=rng.integers(0, 2048, size=(8, 16), dtype=np.int16),
        action_seq=rng.normal(0.0, 0.3, size=(16, UNIFIED_ACTION_DIM)).astype(np.float32),
        proprio_seq=rng.normal(0.0, 0.3, size=(16, UNIFIED_PROPRIO_DIM)).astype(np.float32),
        action_mask=act_m0.astype(np.float32),
        proprio_mask=prop_m0.astype(np.float32),
    )
    st1 = build_interleaved_heartbeat_stream(
        video_groups=rng.normal(0.0, 0.5, size=(4, 15, 1024)).astype(np.float16),
        audio_codes=rng.integers(0, 2048, size=(8, 16), dtype=np.int16),
        action_seq=rng.normal(0.0, 0.3, size=(16, UNIFIED_ACTION_DIM)).astype(np.float32),
        proprio_seq=rng.normal(0.0, 0.3, size=(16, UNIFIED_PROPRIO_DIM)).astype(np.float32),
        action_mask=act_m0.astype(np.float32),
        proprio_mask=prop_m0.astype(np.float32),
    )

    n0 = int(st0["modality_id"].size)
    n1 = int(st1["modality_id"].size)
    v_all = np.concatenate([st0["video_feat"], st1["video_feat"]], axis=0).astype(np.float32)
    v_min = np.min(v_all, axis=-1, keepdims=True)
    v_sc = np.maximum((np.max(v_all, axis=-1, keepdims=True) - v_min) / 255.0, 1e-8)
    v_q = np.clip(np.round((v_all - v_min) / v_sc), 0.0, 255.0).astype(np.uint8)

    vi1 = st1["video_idx"].copy()
    vi1[vi1 >= 0] += st0["video_feat"].shape[0]
    ai1 = st1["audio_idx"].copy()
    ai1[ai1 >= 0] += st0["audio_codes"].shape[0]
    mi1 = st1["motor_idx"].copy()
    mi1[mi1 >= 0] += st0["action_feat"].shape[0]

    packed = {
        k: np.concatenate([st0[k], st1[k]], axis=0)
        for k in (
            "modality_id", "heartbeat_id", "capture_time_s", "last_sample_time_s",
            "wall_time_s", "dt_s", "reset_mask", "valid_mask", "target_valid_mask",
            "reliability", "spatial_pos", "text_byte",
        )
    }
    packed["episode_local_id"] = np.concatenate([np.zeros((n0,), dtype=np.int32), np.ones((n1,), dtype=np.int32)])
    packed["video_idx"] = np.concatenate([st0["video_idx"], vi1])
    packed["audio_idx"] = np.concatenate([st0["audio_idx"], ai1])
    packed["motor_idx"] = np.concatenate([st0["motor_idx"], mi1])
    packed["video_q_int8"] = v_q
    packed["video_xmin_fp16"] = v_min.astype(np.float16)
    packed["video_scale_fp16"] = v_sc.astype(np.float16)
    packed["audio_codes"] = np.concatenate([st0["audio_codes"], st1["audio_codes"]], axis=0)
    packed["action_feat"] = np.concatenate([st0["action_feat"], st1["action_feat"]], axis=0).astype(np.float16)
    packed["proprio_feat"] = np.concatenate([st0["proprio_feat"], st1["proprio_feat"]], axis=0).astype(np.float16)
    packed["episode_action_masks"] = np.stack([act_m0, act_m0], axis=0)
    packed["episode_proprio_masks"] = np.stack([prop_m0, prop_m0], axis=0)
    packed["episode_success"] = np.asarray([1, -1], dtype=np.int8)

    npz_p = cdir / f"{excl_split}_shard_000.npz"
    meta_p = cdir / f"{excl_split}_shard_000_meta.json"
    np.savez(npz_p, **packed)
    meta_p.write_text(
        json.dumps(
            {
                "corpus": excl_corpus,
                "split": excl_split,
                "n_episodes": 2,
                "episodes": [
                    {"episode_id": excl_eid, "n_packets": n0},
                    {"episode_id": "clean_non_excluded_ep", "n_packets": n1},
                ],
            }
        )
    )

    loader = V4SinglePassMixtureLoader(
        split=excl_split,
        seq_len=n0,
        batch_size=1,
        seed=7,
        local_root=tmp_path,
        mixture_weights={excl_corpus: 1.0},
    )
    batches = list(loader)
    loader.close()
    assert len(batches) == 2  # Strictly single-pass (exact 2 windows, then StopIteration!)
    # Window 0 corresponds to the excluded episode -> MUST have valid_mask == 0 & target_valid_mask == 0 everywhere!
    assert float(np.sum(batches[0]["valid_mask"])) == 0.0
    assert float(np.sum(batches[0]["target_valid_mask"])) == 0.0
    # Window 1 corresponds to the clean episode -> valid & episode_success == -1 & clean start reset_mask[0] == True!
    assert float(np.sum(batches[1]["valid_mask"])) == float(n1)
    assert bool(batches[1]["reset_mask"][0, 0]) is True
    assert float(batches[1]["dt_s"][0, 0]) == 0.0
    assert int(batches[1]["episode_success"][0, 0]) == -1
    assert float(np.sum(batches[1]["action_mask"][0, 0])) == 8.0
    assert float(np.sum(batches[1]["proprio_mask"][0, 0])) == 12.0

    # 3. Verify strict split-isolation rejection on mismatched shard filename
    from data.v4_streaming_loader import _SinglePassCorpusStream

    with pytest.raises(ValueError, match="Split isolation violation"):
        _SinglePassCorpusStream(
            corpus_name=excl_corpus,
            split="train",
            shard_specs=[{"source": "local", "path": str(npz_p)}],
            seq_len=n0,
            packet_cap=None,
            excluded_ids=excluded,
            cache_dir=None,
            delete_cached_after_consume=False,
            seed=0,
        )


def test_v4_train_grad_accum_checkpoint_and_dev_eval(tmp_path: Path) -> None:
    """Certify `scripts/train/v4_train.py` runs grad_accum, dev eval, and exact checkpoint round-trip cleanly."""
    import argparse
    from scripts.train.v4_train import load_v4_checkpoint, run_v4_training

    rng = np.random.default_rng(91)
    act_m = np.ones((UNIFIED_ACTION_DIM,), dtype=np.uint8)
    prop_m = np.ones((UNIFIED_PROPRIO_DIM,), dtype=np.uint8)
    for split in ("train", "dev"):
        cdir = tmp_path / "data" / "greatesthits_impact_av"
        cdir.mkdir(parents=True, exist_ok=True)
        st = build_interleaved_heartbeat_stream(
            video_groups=rng.normal(0.0, 0.5, (14, 15, 1024)).astype(np.float16),
            audio_codes=rng.integers(0, 2048, (28, 16), dtype=np.int16),
            action_seq=rng.normal(0.0, 0.3, (56, UNIFIED_ACTION_DIM)).astype(np.float32),
            proprio_seq=rng.normal(0.0, 0.3, (56, UNIFIED_PROPRIO_DIM)).astype(np.float32),
            text_by_heartbeat={h: b"hit " for h in range(14)},
            action_mask=act_m.astype(np.float32),
            proprio_mask=prop_m.astype(np.float32),
        )
        n = int(st["modality_id"].size)
        v = st["video_feat"].astype(np.float32)
        vmin = np.min(v, axis=-1, keepdims=True)
        vsc = np.maximum((np.max(v, axis=-1, keepdims=True) - vmin) / 255.0, 1e-8)
        vq = np.clip(np.round((v - vmin) / vsc), 0, 255).astype(np.uint8)
        packed = {
            k: st[k]
            for k in (
                "modality_id", "heartbeat_id", "capture_time_s", "last_sample_time_s",
                "wall_time_s", "dt_s", "reset_mask", "valid_mask", "target_valid_mask",
                "reliability", "spatial_pos", "text_byte", "video_idx", "audio_idx", "motor_idx", "audio_codes",
            )
        }
        packed.update(
            {
                "episode_local_id": np.zeros((n,), dtype=np.int32),
                "video_q_int8": vq,
                "video_xmin_fp16": vmin.astype(np.float16),
                "video_scale_fp16": vsc.astype(np.float16),
                "action_feat": st["action_feat"].astype(np.float16),
                "proprio_feat": st["proprio_feat"].astype(np.float16),
                "episode_action_masks": act_m[None, :],
                "episode_proprio_masks": prop_m[None, :],
                "episode_success": np.asarray([1], dtype=np.int8),
            }
        )
        np.savez(cdir / f"{split}_shard_000.npz", **packed)

    ns = argparse.Namespace(
        profile="tiny_unit",
        run_id="A0_3to1_single_bank",
        seed=42,
        seq_len=48,
        batch_size=2,
        grad_accum=2,
        target_packets=500,
        max_steps=2,
        stop_after_steps=0,
        peak_lr=1e-3,
        warmup_frac=0.5,
        weight_decay=0.05,
        grad_clip_norm=1.0,
        matmul_dtype="fp32",
        use_remat=False,
        local_data_root=str(tmp_path / "data"),
        cache_dir="",
        mixture_weights_json=json.dumps({"greatesthits_impact_av": 1.0}),
        gate2_receipt="reports/v4/gate2_v4_data_and_codec_receipt.json",
        exclusion_manifest="reports/v4/gate2_eval_exclusion_manifest.json",
        out_dir=str(tmp_path / "out"),
        resume_ckpt="",
        log_every=1,
        eval_every=1,
        eval_batches=1,
        ckpt_every=1,
    )
    summ = run_v4_training(ns)
    assert summ["steps_completed"] == 2
    assert np.isfinite(summ["final_dev_eval"]["dev_total_loss"])
    assert (tmp_path / "out/ckpt_final.npz").is_file()
    assert (tmp_path / "out/run_config.json").is_file()
    sp = make_slwm_v4_spec("tiny_unit", run_id="A0_3to1_single_bank")
    p_reload = load_v4_checkpoint(tmp_path / "out/ckpt_final.npz", init_v4_model_params(jax.random.PRNGKey(0), sp))
    assert count_tree_params(p_reload["core"]) == estimate_v4_core_param_count(sp)

    # Verify B3: lossless single-pass resume (`step 1 -> resume -> step 2` of `max_steps=2`) matches uninterrupted run!
    ns_s1 = argparse.Namespace(**vars(ns))
    ns_s1.out_dir = str(tmp_path / "out_split")
    ns_s1.max_steps = 2
    ns_s1.stop_after_steps = 1
    ns_s1.matmul_dtype = "bf16"
    ns_s1.use_remat = True
    summ_s1 = run_v4_training(ns_s1)
    assert summ_s1["steps_completed"] == 1

    ns_s2 = argparse.Namespace(**vars(ns_s1))
    ns_s2.stop_after_steps = 0
    ns_s2.resume_ckpt = str(tmp_path / "out_split/ckpt_latest.npz")
    summ_s2 = run_v4_training(ns_s2)
    assert summ_s2["steps_completed"] == 2

    ns_full = argparse.Namespace(**vars(ns_s1))
    ns_full.out_dir = str(tmp_path / "out_full")
    ns_full.stop_after_steps = 0
    ns_full.resume_ckpt = ""
    summ_full = run_v4_training(ns_full)

    # Exact zero-revisit single-pass packet count & identical weights!
    assert summ_s2["loader_stats"]["total_packets_yielded"] == summ_full["loader_stats"]["total_packets_yielded"]
    p_res = load_v4_checkpoint(tmp_path / "out_split/ckpt_final.npz", init_v4_model_params(jax.random.PRNGKey(0), sp))
    p_ful = load_v4_checkpoint(tmp_path / "out_full/ckpt_final.npz", init_v4_model_params(jax.random.PRNGKey(0), sp))
    max_p_diff = max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree_util.tree_leaves(p_res), jax.tree_util.tree_leaves(p_ful))
    )
    assert max_p_diff < 1e-5, f"Resumed training diverged from uninterrupted run: max_p_diff={max_p_diff}"


def test_b1_mup_all_1d_scalars_and_gdn_gqa_wo_parity() -> None:
    """Certify B1 fix: 100% of 0D/1D leaves classify as 'scalar' (wd=0) and mem.w_o == gqa.w_o ('hidden')."""
    from models.v4 import classify_v4_param_mup_role

    for rid in V4_S3_DISCOVERY_RUN_IDS:
        sp = make_slwm_v4_spec("tiny_unit", run_id=rid)
        params = init_v4_model_params(jax.random.PRNGKey(1), sp)

        def _walk(tree, prefix=()):
            if isinstance(tree, dict):
                for k, v in tree.items():
                    _walk(v, prefix + (str(k),))
            elif isinstance(tree, (list, tuple)):
                for i, v in enumerate(tree):
                    _walk(v, prefix + (str(i),))
            else:
                role = classify_v4_param_mup_role(prefix, param_ndim=int(tree.ndim))
                if int(tree.ndim) <= 1:
                    assert role == "scalar", f"1D/0D leaf {'.'.join(prefix)} on {rid} misclassified as {role!r}"

        _walk(params)

    assert classify_v4_param_mup_role(("core", "layers", "0", "mem", "w_o"), param_ndim=2) == "hidden"
    assert classify_v4_param_mup_role(("core", "layers", "3", "gqa", "w_o"), param_ndim=2) == "hidden"
    assert classify_v4_param_mup_role(("heads", "head_jepa", "proj_a"), param_ndim=2) == "fixed_hidden"
    assert classify_v4_param_mup_role(("heads", "policy_gate", "w2"), param_ndim=2) == "fixed_hidden"
    assert classify_v4_param_mup_role(("heads", "policy_gate", "w1"), param_ndim=2) == "readout"


def test_cross_episode_reset_mask_exact_isolation() -> None:
    """Mutating episode 0 strictly before a `reset_mask` MUST have EXACT 0.0 effect on episode 1."""
    for rid in ("A0_3to1_single_bank", "B0_v3_ref", "B1_sw_gqa", "C5_routed_2bank", "C6_routed_4bank", "C7_ms_moe_ffn"):
        sp = make_slwm_v4_spec("tiny_unit", run_id=rid)
        params = init_v4_model_params(jax.random.PRNGKey(9), sp)
        batch = _make_synthetic_quadmodal_batch(t_len=36, batch_size=2, seed=99, with_mid_resets=False, with_holes=True)
        t_rst = 18
        batch["reset_mask"] = batch["reset_mask"].at[t_rst, :].set(True)

        batch_mut = dict(batch)
        batch_mut["video_feat"] = batch["video_feat"].at[:t_rst].set(batch["video_feat"][:t_rst] + 4.5)
        batch_mut["audio_codes"] = batch["audio_codes"].at[:t_rst].set((batch["audio_codes"][:t_rst] + 99) % 2048)
        batch_mut["action_feat"] = batch["action_feat"].at[:t_rst].set(batch["action_feat"][:t_rst] - 3.0)

        z0, _, _ = apply_v4_core_seq(params, init_v4_core_state(sp, 2), batch, sp, matmul_dtype=jnp.float32, use_remat=False)
        z1, _, _ = apply_v4_core_seq(params, init_v4_core_state(sp, 2), batch_mut, sp, matmul_dtype=jnp.float32, use_remat=False)
        ep1_diff = float(jnp.max(jnp.abs(z0[t_rst:] - z1[t_rst:])))
        assert ep1_diff == 0.0, f"Cross-episode leak across reset_mask on {rid}: diff={ep1_diff}"


def test_b2_policy_gate_full_heads_loss_zero_grad_and_rvq0_parity() -> None:
    """Certify PolicyCommitGate has EXACT 0.0 grad into core/edge/other heads AND exact RVQ-0 entropy parity."""
    from models.v4.heads import compute_audio_rvq0_entropy, sample_audio_codes

    sp = make_slwm_v4_spec("tiny_unit", run_id="O1_jepa_sync")
    params = init_v4_model_params(jax.random.PRNGKey(121), sp)
    batch = _make_synthetic_quadmodal_batch(t_len=36, batch_size=2, seed=122, with_mid_resets=True)

    def _pol_only_full(p):
        z, _, _ = apply_v4_core_seq(p, init_v4_core_state(sp, 2), batch, sp, matmul_dtype=jnp.float32, use_remat=False)
        loss, _ = compute_v4_heads_loss(
            p["heads"],
            z,
            batch,
            sp,
            jax.random.PRNGKey(123),
            matmul_dtype=jnp.float32,
            loss_weights={"text": 0.0, "audio": 0.0, "video": 0.0, "motor": 0.0, "success": 0.0, "jepa": 0.0, "policy": 1.0},
        )
        return loss

    grads = jax.grad(_pol_only_full)(params)
    assert max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(grads["core"])) == 0.0
    assert max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(grads["edge"])) == 0.0
    for hk in ("head_text", "head_audio", "head_video", "head_motor", "head_jepa"):
        assert max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(grads["heads"][hk])) == 0.0
    assert max(float(jnp.max(jnp.abs(x))) for x in jax.tree_util.tree_leaves(grads["heads"]["policy_gate"])) > 1e-6

    # Exact parity between `compute_audio_rvq0_entropy`, `apply_audio_head_teacher_forced(k=0)`, and `sample_audio_codes`!
    z_test = jax.random.normal(jax.random.PRNGKey(124), (2, sp.d_model), dtype=jnp.float32)
    e_direct = compute_audio_rvq0_entropy(params["heads"]["head_audio"], z_test, sp, matmul_dtype=jnp.float32)
    _, e_samp = sample_audio_codes(params["heads"]["head_audio"], z_test, jax.random.PRNGKey(125), sp, matmul_dtype=jnp.float32)
    lg_tf = apply_audio_head_teacher_forced(
        params["heads"]["head_audio"], z_test, jnp.zeros((2, sp.audio_codebooks), dtype=jnp.int32), sp, matmul_dtype=jnp.float32
    )
    p0 = jax.nn.softmax(lg_tf[:, 0, :], axis=-1)
    e_tf = -jnp.sum(p0 * jnp.log(jnp.clip(p0, 1e-9, 1.0)), axis=-1)
    assert float(jnp.max(jnp.abs(e_direct - e_tf))) < 1e-5
    assert float(jnp.max(jnp.abs(e_direct - e_samp))) < 1e-5
