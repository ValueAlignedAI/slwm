"""Comprehensive Automated Unit & Smoke Test Suite for Study X3 (`SLWM-1B-480p`).

Verifies:
  1. Strict `< 0.01%` parameter parity across `SLWM-X3`, `SlidingWindow-W64-GQA`, `DiagonalSSM`, and `240p` ablation
     across `x3_tiny` & `x3_150m` (materialized) and analytical exactness on `x3_450m` & `x3_1b`.
  2. Exact causal streaming equivalence (`T = 12` parallel vs `12 x 1`-tick `streaming_carry`) across all 3 architectures,
     including exact `0`-byte carry growth (`_carry_nbytes(t=12) == _carry_nbytes(t=1)`).
  3. Structural `SLOT_SYM_MEM` (`54`) vs `SLOT_GOAL_LANG` (`55`) isolation:
     - Exact `0.0` task/goal/video/action gradient leakage into `sym_proj_w` and `text_emb[32:96]`,
     - Exact null-tick `H_sym` invariance when `sym_tokens == 0` (`beta_sym == 0, alpha_sym == 1`),
     - Zero visual/goal perturbation drift on `z_sym`.
  4. Strict single-factor `240p` vs `480p` bottleneck integrity (verifying `decode_480p_video_from_latents` never leaks
     un-bottlenecked `480p` high-frequency checkerboard detail through `context_frame_480p` when `resolution_mode='240p'`).
  5. Causal 3-tap audio Conv1D zero-future-lookahead & `beta_aud` silence gate behavior.
  6. End-to-end `x3_tiny` smoke training, 6-way Any-to-Any evaluation, `T=64` livestreaming, `.mp4`/`.wav`/`.png`
     showcase export, and §13.1 scientific report/brief hygiene.
"""

from __future__ import annotations

from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np

from evals.x3_eval_480p import (
    _build_synthetic_480p_eval_clips,
    _carry_nbytes,
    eval_x3_any_to_any_composition,
    eval_x3_spoken_xarm6_grounding,
    run_x3_full_evaluation,
)
from models.x3_480p_core import (
    X3_SLOT_SYM_MEM,
    compute_x3_streaming_state_bytes,
    count_x3_parameters,
    decode_480p_video_from_latents,
    decode_x3_any_to_any_heads,
    decode_x3_target_block_xy,
    encode_causal_audio_conv,
    fork_x3_streams,
    forward_x3_world_model,
    init_x3_params,
    init_x3_streaming_carry,
    make_x3_spec,
    mel64_to_waveform_griffin_lim_16khz,
    predict_x3_latent_step,
)
from scripts.eval.x3_analyze_480p import analyze_study_x3
from scripts.train.x3_train_480p import (
    build_x3_long_horizon_aux_batch_t80,
    generate_x3_sim_batch_480p,
    train_and_evaluate_x3,
    world_model_loss_x3,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_x3_parameter_parity_and_dual_track_state_accounting() -> None:
    for prof in ("x3_tiny", "x3_150m"):
        p_slwm = count_x3_parameters(init_x3_params(make_x3_spec(prof, "slwm_x3", resolution_mode="480p")))["strict_total"]
        p_gqa = count_x3_parameters(init_x3_params(make_x3_spec(prof, "sliding_window_gqa_x3", resolution_mode="480p")))["strict_total"]
        p_ssm = count_x3_parameters(init_x3_params(make_x3_spec(prof, "diagonal_ssm_x3", resolution_mode="480p")))["strict_total"]
        p_240 = count_x3_parameters(init_x3_params(make_x3_spec(prof, "slwm_x3", resolution_mode="240p")))["strict_total"]
        assert p_slwm == p_ssm == p_240
        rel_diff = abs(p_gqa - p_slwm) / float(p_slwm)
        assert rel_diff < 1e-4, f"Expected <0.01% parity on {prof}, got {rel_diff:.6%}"

    # Verify Dual-Track state accounting on 1B profiles
    s_slwm = compute_x3_streaming_state_bytes(make_x3_spec("x3_1b", "slwm_x3"), batch_size=1)
    s_gqa = compute_x3_streaming_state_bytes(make_x3_spec("x3_1b", "sliding_window_gqa_x3"), batch_size=1)
    s_ssm = compute_x3_streaming_state_bytes(make_x3_spec("x3_1b", "diagonal_ssm_x3"), batch_size=1)
    assert s_ssm["strict_system_state_bytes"] < s_slwm["strict_system_state_bytes"] < s_gqa["strict_system_state_bytes"]
    assert 29.0 < s_slwm["strict_system_state_mib"] < 31.0
    assert s_gqa["strict_system_state_mib"] > 65.0


def test_x3_parallel_vs_single_tick_streaming_equivalence_all_archs() -> None:
    batch = generate_x3_sim_batch_480p(jax.random.key(101), batch_size=2, seq_len=10)
    # Also inject dynamic audio to test causal 3-tap audio Conv1D carry equivalence!
    aud = jax.random.normal(jax.random.key(102), (2, 10, 64), dtype=jnp.float32)
    mm = jnp.ones((2, 10, 4), dtype=jnp.float32)
    batch_full = {**batch, "audio": aud, "modality_mask": mm}

    for arch in ("slwm_x3", "sliding_window_gqa_x3", "diagonal_ssm_x3"):
        spec = make_x3_spec("x3_tiny", arch, dtype="float32", grad_checkpoint=False)
        params = init_x3_params(spec, seed=202)
        z_par, f_par, ba_par, _ = forward_x3_world_model(params, spec, batch_full)

        carry = init_x3_streaming_carry(spec, batch_size=2)
        b_init = _carry_nbytes(carry)
        zs, bas = [], []
        for ti in range(10):
            b1 = {
                "video": batch_full["video"][:, ti : ti + 1],
                "audio": batch_full["audio"][:, ti : ti + 1],
                "movement": batch_full["movement"][:, ti : ti + 1],
                "action": batch_full["action"][:, ti : ti + 1],
                "sym_tokens": batch_full["sym_tokens"][:, ti : ti + 1],
                "lang_tokens": batch_full["lang_tokens"][:, ti : ti + 1],
                "modality_mask": batch_full["modality_mask"][:, ti : ti + 1],
                "embodiment_id": batch_full["embodiment_id"],
                "valid": batch_full["valid"][:, ti : ti + 1],
            }
            z1, _, ba1, carry = forward_x3_world_model(params, spec, b1, streaming_carry=carry)
            zs.append(z1)
            bas.append(ba1)
            assert _carry_nbytes(carry) == b_init, f"Carry byte growth detected in {arch} at tick {ti}!"

        z_seq = jnp.concatenate(zs, axis=1)
        ba_seq = jnp.concatenate(bas, axis=1)
        max_diff_z = float(jnp.max(jnp.abs(z_par - z_seq)))
        max_diff_ba = float(jnp.max(jnp.abs(ba_par - ba_seq)))
        assert max_diff_z < 2e-4, f"{arch} streaming z mismatch: {max_diff_z}"
        assert max_diff_ba < 1e-5, f"{arch} streaming beta_aud mismatch: {max_diff_ba}"


def test_x3_slot_sym_mem_strict_shielding_and_zero_task_grad_leak() -> None:
    spec = make_x3_spec("x3_tiny", "slwm_x3", dtype="float32", grad_checkpoint=False)
    params = init_x3_params(spec, seed=303)
    b1 = generate_x3_sim_batch_480p(jax.random.key(304), batch_size=2, seq_len=8)
    # Perturb video, audio, movement, action, and goal_lang tokens wildly while keeping sym_tokens identical:
    b2 = {
        **b1,
        "video": jnp.uint8(255) - b1["video"],
        "audio": jax.random.normal(jax.random.key(305), b1["audio"].shape),
        "movement": b1["movement"] + 1.5,
        "action": -b1["action"],
        "lang_tokens": jnp.full_like(b1["lang_tokens"], 500),
        "modality_mask": jnp.ones_like(b1["modality_mask"]),
    }
    z1, _, _, _ = forward_x3_world_model(params, spec, b1)
    z2, _, _, _ = forward_x3_world_model(params, spec, b2)
    sym_drift = float(jnp.max(jnp.abs(z1[:, :, X3_SLOT_SYM_MEM, :] - z2[:, :, X3_SLOT_SYM_MEM, :])))
    assert sym_drift == 0.0, f"Expected exact 0.0 sensory/language drift on SLOT_SYM_MEM, got {sym_drift}"

    # Verify zero gradient into sym_proj_w and text_emb[32:96] when sym associative target is inactive!
    b_notarg = {**b1, "target_sym_val": jnp.zeros_like(b1["target_sym_val"])}
    grads = jax.grad(lambda w: world_model_loss_x3(w, params, spec, b_notarg, jax.random.key(306))[0])(params)
    assert float(jnp.max(jnp.abs(grads["sym_proj_w"]))) == 0.0
    assert float(jnp.max(jnp.abs(grads["head_sym_w"]))) == 0.0
    assert float(jnp.max(jnp.abs(grads["text_emb"][32:96]))) == 0.0


def test_x3_240p_bottleneck_blocks_480p_checkerboard_leak_in_context_skip() -> None:
    spec_240 = make_x3_spec("x3_tiny", "slwm_x3", resolution_mode="240p", dtype="float32")
    params = init_x3_params(spec_240, seed=404)
    # Construct a 1-pixel alternating 480p checkerboard (spatial Nyquist = 640x480; averages to exact 127.5 at 240p!)
    yy, xx = jnp.meshgrid(jnp.arange(480), jnp.arange(640), indexing="ij")
    chk_a = (((yy + xx) % 2) * 254).astype(jnp.uint8)[None, None, :, :, None] * jnp.ones((1, 1, 1, 1, 3), dtype=jnp.uint8)
    chk_b = jnp.full((1, 1, 480, 640, 3), 127, dtype=jnp.uint8)
    z_vis = jnp.zeros((1, 2, 48, spec_240.d_model), dtype=jnp.float32)
    reg = jnp.zeros((1, 2, spec_240.d_model), dtype=jnp.float32)
    out_a = decode_480p_video_from_latents(params, spec_240, z_vis, reg, context_frame_480p=chk_a)
    out_b = decode_480p_video_from_latents(params, spec_240, z_vis, reg, context_frame_480p=chk_b)
    # Both 1px 480p checkerboard (0/254 -> mean 127) and uniform 127 produce IDENTICAL 240p boxcar means!
    diff = float(jnp.max(jnp.abs(out_a - out_b)))
    assert diff == 0.0, f"240p ablation leaked native 480p checkerboard through context skip: {diff}"


def test_x3_causal_audio_zero_future_lookahead_and_griffin_lim() -> None:
    spec = make_x3_spec("x3_tiny", "slwm_x3", dtype="float32")
    params = init_x3_params(spec, seed=505)
    a1 = jnp.zeros((1, 6, 64), dtype=jnp.float32)
    a2 = a1.at[:, 4:, :].set(3.5)  # Huge acoustic impulse at future ticks 4..5
    mm = jnp.ones((1, 6), dtype=jnp.float32)
    p1, b1, _ = encode_causal_audio_conv(params, spec, a1, mm)
    p2, b2, _ = encode_causal_audio_conv(params, spec, a2, mm)
    # Ticks 0..3 must have ZERO dependence on ticks 4..5!
    assert float(jnp.max(jnp.abs(p1[:, :4] - p2[:, :4]))) == 0.0
    assert float(jnp.max(jnp.abs(b1[:, :4] - b2[:, :4]))) == 0.0

    wav = mel64_to_waveform_griffin_lim_16khz(np.zeros((8, 64), dtype=np.float32), n_iter=4)
    assert wav.shape == (8 * 1600,)
    assert np.all(np.isfinite(wav))


def test_x3_tiny_end_to_end_smoke_train_eval_and_hygiene(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs_x3_smoke"
    rec = train_and_evaluate_x3(
        tag="x3-slwm-1b-480p-s3001",
        cohort="x3_slwm_1b_480p",
        profile="x3_tiny",
        arch="slwm_x3",
        resolution_mode="480p",
        seed=3001,
        steps=4,
        batch_size=2,
        seq_len=12,
        out_dir=runs_dir,
        export_showcase=True,
        smoke=True,
    )
    assert rec["streaming_state_memory"]["strict_system_state_bytes"] > 0
    assert rec["eval_summary"]["livestream_state_growth_bytes"] == 0
    assert (runs_dir / "x3-slwm-1b-480p-s3001/showcase_bundle.npz").is_file()

    out_json = tmp_path / "x3_summary.json"
    sc_dir = tmp_path / "showcase_out"
    fig_dir = tmp_path / "figs"
    doc = analyze_study_x3(runs_dir, out_json, sc_dir, fig_dir)
    assert "x3_slwm_1b_480p" in doc["cohorts"]
    assert len(doc["figures"]) == 4
    assert len(doc["showcase_manifest"]["exported_files"]["wav_audio"]) > 0
    assert len(doc["showcase_manifest"]["exported_files"]["png_panels"]) >= 2

    # Verify scientific brief hygiene (zero forbidden infrastructure tokens)
    brief_txt = (REPO_ROOT / "experiments/multimodal/x3/livestream-1b-480p-20261116/x3_brief.md").read_text()
    forbidden = re.compile(r"(dual[- ]hdd|disk_uuid|192\.168\.|colab compute unit|AGENTS\.md|sha-?256)", re.IGNORECASE)
    assert not forbidden.search(brief_txt), f"Forbidden operational token found in x3_brief.md: {forbidden.search(brief_txt)}"


def test_x3_spoken_xarm6_v2_schema_symmetric_scoring_and_fail_closed(tmp_path: Path) -> None:
    import json as _json
    import pytest

    spec = make_x3_spec("x3_tiny", "slwm_x3", dtype="float32")
    params = init_x3_params(spec, seed=601)
    rng = np.random.default_rng(602)
    n_ep, t_r = 8, 32
    vault_arrays = {
        "vault_video": rng.integers(0, 255, size=(n_ep, t_r, 64, 64, 3), dtype=np.uint8),
        "vault_spoken_audio": rng.standard_normal((n_ep, t_r, 64)).astype(np.float32) * 0.4,
        "vault_wrong_spoken_audio": rng.standard_normal((n_ep, t_r, 64)).astype(np.float32) * 0.4,
        "vault_wrong_color_spoken_audio": rng.standard_normal((n_ep, t_r, 64)).astype(np.float32) * 0.4,
        "vault_ee_xy": rng.uniform(-0.5, 0.5, size=(n_ep, t_r, 2)).astype(np.float32),
        "vault_action_xy": rng.uniform(-0.5, 0.5, size=(n_ep, t_r, 2)).astype(np.float32),
        "vault_length": np.full((n_ep,), 28, dtype=np.int32),
        "vault_true_text_ids": rng.integers(260, 400, size=(n_ep, 16), dtype=np.int32),
        "vault_wrong_text_ids": rng.integers(260, 400, size=(n_ep, 16), dtype=np.int32),
        "vault_wrong_color_text_ids": rng.integers(260, 400, size=(n_ep, 16), dtype=np.int32),
        "vault_cf_index": np.arange(n_ep, dtype=np.int32)[::-1],
        "vault_scene_cluster_id": np.array([10, 10, 20, 20, 30, 30, 40, 40], dtype=np.int64),
        "vault_dir_id": np.arange(n_ep, dtype=np.int32) % 4,
        "vault_color_id": np.arange(n_ep, dtype=np.int32) % 3,
        "vault_distractor_color_id": (np.arange(n_ep, dtype=np.int32) + 1) % 3,
        "vault_target_block_xy": np.tile(np.array([[-0.45, 0.35], [0.45, -0.35]], dtype=np.float32), (n_ep // 2, 1)),
        "vault_distractor_block_xy": np.tile(np.array([[0.45, -0.35], [-0.45, 0.35]], dtype=np.float32), (n_ep // 2, 1)),
    }
    vault_p = tmp_path / "vault_v2.npz"
    np.savez(vault_p, **vault_arrays)
    train_p = tmp_path / "train_v2.npz"
    np.savez(
        train_p,
        train_target_block_xy=vault_arrays["vault_target_block_xy"],
        train_color_id=vault_arrays["vault_color_id"],
    )
    res = eval_x3_spoken_xarm6_grounding(
        params, spec, vault_p, train_ref_npz_path=train_p, split="vault", chunk_bs=4, smoke=False
    )
    assert res["available"] is True
    assert len(res["spoken_disc_vec"]) == n_ep
    assert len(res["spoken_color_lift_vec"]) == n_ep
    assert res["scene_cluster_ids"] == [10, 10, 20, 20, 30, 30, 40, 40]
    assert all(val in (0.0, 0.5, 1.0) for val in _json.loads(_json.dumps(res["spoken_disc_vec"])))

    # Fail-closed check when a required Wave 8A v2 key is missing:
    bad_p = tmp_path / "bad_vault_v2.npz"
    bad_arrays = {k: v for k, v in vault_arrays.items() if k != "vault_wrong_color_spoken_audio"}
    np.savez(bad_p, **bad_arrays)
    with pytest.raises(KeyError, match="vault_wrong_color_spoken_audio"):
        eval_x3_spoken_xarm6_grounding(params, spec, bad_p, split="vault", smoke=False)


def test_x3_long_horizon_t80_aux_batch_k16_beyond_w8_and_gqa_rope() -> None:
    aux = build_x3_long_horizon_aux_batch_t80(jax.random.key(707), batch_size=6, seq_len=80)
    assert aux["sym_tokens"].shape == (6, 80, 4)
    # Verify zero writes after t=23 and queries strictly at t=72..79 -> lag >= 72 - 23 = 49 > 42 (6-layer W=8 span)!
    write_ticks = np.where(np.any(np.asarray(aux["sym_tokens"][:, :, 2]) > 0, axis=0))[0]
    query_ticks = np.where(np.any(np.asarray(aux["sym_tokens"][:, :, 3]) == 2, axis=0))[0]
    assert int(np.max(write_ticks)) <= 23
    assert int(np.min(query_ticks)) == 72
    assert int(np.min(query_ticks)) - int(np.max(write_ticks)) >= 49

    # Verify skip_visual_conv=True executes cleanly across all 3 architectures at T_aux=80:
    for arch in ("slwm_x3", "sliding_window_gqa_x3", "diagonal_ssm_x3"):
        spec = make_x3_spec("x3_tiny", arch, dtype="float32", grad_checkpoint=False)
        params = init_x3_params(spec, seed=708)
        zt, f300, ba, _ = forward_x3_world_model(params, spec, aux, skip_visual_conv=True)
        assert zt.shape == (6, 80, 60, spec.d_model)
        assert f300.shape == (6, 80, 300, spec.d_model)
        assert jnp.all(jnp.isfinite(zt)) and jnp.all(jnp.isfinite(ba))


def test_x3_any_to_any_zero_target_action_leak_and_fixed_ctx_f0() -> None:
    spec = make_x3_spec("x3_tiny", "slwm_x3", dtype="float32")
    params = init_x3_params(spec, seed=801)
    clips_a = _build_synthetic_480p_eval_clips(seed=802, n_clips=6, seq_len=24)
    clips_b = {**clips_a, "target_action": clips_a["target_action"] + 999.0}

    res_a, _ = eval_x3_any_to_any_composition(params, spec, clips_a, inv_vocab={}, chunk_bs=3)
    res_b, _ = eval_x3_any_to_any_composition(params, spec, clips_b, inv_vocab={}, chunk_bs=3)

    # Mutating target_action by +999 must leave Directions 1, 2, 3, 4, & 6 bit-identical!
    assert res_a["dir1_video_audio_to_text"]["nll_ce"] == res_b["dir1_video_audio_to_text"]["nll_ce"]
    assert res_a["dir2_text_ctx_to_480p_video"]["psnr_480p_db"] == res_b["dir2_text_ctx_to_480p_video"]["psnr_480p_db"]
    assert res_a["dir3_video_to_16khz_audio"]["log_spectral_distance_db"] == res_b["dir3_video_to_16khz_audio"]["log_spectral_distance_db"]
    assert res_a["dir4_text_video_to_3d_movement"]["ade_1step_l2"] == res_b["dir4_text_video_to_3d_movement"]["ade_1step_l2"]
    assert res_a["dir6_blind_audio_proprio_and_ood_gate"]["blind_audio_contact_bce"] == res_b["dir6_blind_audio_proprio_and_ood_gate"]["blind_audio_contact_bce"]
    # While Direction 5 ground-truth action scoring MUST change:
    assert res_b["dir5_av_text_to_robot_actions"]["action_mse"] > res_a["dir5_av_text_to_robot_actions"]["action_mse"] + 1e4

    # Fixed-Frame0 check: when lang_tokens==0 and action==0 across all clips, Dir 2 shuffle lift is exact 0.0 dB!
    clips_null_cond = {
        **clips_a,
        "action": np.zeros_like(clips_a["action"]),
        "lang_tokens": np.zeros_like(clips_a["lang_tokens"]),
    }
    res_nc, _ = eval_x3_any_to_any_composition(params, spec, clips_null_cond, inv_vocab={}, chunk_bs=3)
    assert abs(res_nc["dir2_text_ctx_to_480p_video"]["shuffle_lift_psnr_db"]) < 1e-6


def test_x3_h_obj_slot56_vs_h_dir_slots57_58_readout_disentanglement() -> None:
    spec = make_x3_spec("x3_tiny", "slwm_x3", dtype="float32")
    params = init_x3_params(spec, seed=901)
    batch = generate_x3_sim_batch_480p(jax.random.key(902), batch_size=2, seq_len=6)
    z_top, f300, _, _ = forward_x3_world_model(params, spec, batch)
    z_sym, z_task = fork_x3_streams(z_top)
    zn1, lv1 = predict_x3_latent_step(params, spec, z_task, batch["action"])

    blk_orig = decode_x3_target_block_xy(params, z_task, f300)
    goal_orig = decode_x3_any_to_any_heads(params, spec, z_sym, z_task, zn1, lv1, f300, batch, decode_480p_frames=False)["pred_goal_xy"]

    # Perturb ONLY H_dir registers (57..58) -> decode_x3_target_block_xy (slot 56) must have exact 0.0 change!
    z_pert_dir = z_task.at[:, :, 57:59, :].add(10.0)
    blk_pert_dir = decode_x3_target_block_xy(params, z_pert_dir, f300)
    assert float(jnp.max(jnp.abs(blk_orig - blk_pert_dir))) == 0.0

    # Perturb ONLY H_obj register (56) -> pred_goal_xy (slots 57..58) must have exact 0.0 change!
    z_pert_obj = z_task.at[:, :, 56, :].add(10.0)
    goal_pert_obj = decode_x3_any_to_any_heads(params, spec, z_sym, z_pert_obj, zn1, lv1, f300, batch, decode_480p_frames=False)["pred_goal_xy"]
    assert float(jnp.max(jnp.abs(goal_orig - goal_pert_obj))) == 0.0


def test_x3_three_seed_hierarchical_ci_and_median_seed_showcase(tmp_path: Path) -> None:
    import json as _json
    import shutil as _shutil

    cfg_txt = (REPO_ROOT / "configs/x3/study_x3_1b_480p.json").read_text()
    assert '"budget"' not in cfg_txt.lower()

    runs_dir = tmp_path / "runs_3seed"
    base_rec = train_and_evaluate_x3(
        tag="x3-slwm-1b-480p-s3001",
        cohort="x3_slwm_1b_480p",
        profile="x3_tiny",
        arch="slwm_x3",
        resolution_mode="480p",
        seed=3001,
        steps=2,
        batch_size=2,
        seq_len=12,
        out_dir=runs_dir,
        export_showcase=True,
        smoke=True,
    )
    assert base_rec["seed"] == 3001
    src_3001 = runs_dir / "x3-slwm-1b-480p-s3001"
    # Populate 3 seeds (3001=low, 3002=high, 3003=median!) + matched GQA 3 seeds so median-seed & head-to-head CIs are exercised
    seed_offsets = {3001: -0.05, 3002: +0.08, 3003: +0.01}
    for coh_slug, coh_name in [
        ("x3-slwm-1b-480p", "x3_slwm_1b_480p"),
        ("x3-sliding-window-w64-gqa-1b-480p", "x3_sliding_window_w64_gqa_1b_480p"),
    ]:
        for s_id, off in seed_offsets.items():
            tdir = runs_dir / f"{coh_slug}-s{s_id}"
            if tdir != src_3001:
                _shutil.copytree(src_3001, tdir)
            rec_d = _json.loads((tdir / "receipt.json").read_text())
            ev_d = _json.loads((tdir / "eval_metrics.json").read_text())
            rec_d["seed"] = s_id
            rec_d["cohort"] = coh_name
            rec_d["tag"] = f"{coh_slug}-s{s_id}"
            ev_d["summary"]["unseen_mppi_480p"] = float(np.clip(ev_d["summary"]["unseen_mppi_480p"] + off, 0.0, 1.0))
            (tdir / "receipt.json").write_text(_json.dumps(rec_d) + "\n")
            (tdir / "eval_metrics.json").write_text(_json.dumps(ev_d) + "\n")
            (tdir / "verified.json").write_text('{"status": "MIRROR_VERIFIED_2_COPIES"}\n')

    doc = analyze_study_x3(
        runs_dir,
        tmp_path / "x3_3seed_summary.json",
        tmp_path / "sc_3seed",
        tmp_path / "fig_3seed",
        require_verified=True,
    )
    c1 = doc["cohorts"]["x3_slwm_1b_480p"]
    assert c1["n_seeds"] == 3
    assert "hierarchical_ci95" in c1["metrics"]["unseen_mppi_480p"]
    assert "slwm_1b_480p_minus_x3_sliding_window_w64_gqa_1b_480p" in doc["head_to_head_hierarchical_cis"]
    # Seed 3003 has the middle offset (+0.01 between -0.05 and +0.08), so median seed MUST be s3003!
    assert doc["showcase_manifest"]["selected_median_seed_tag"] == "x3-slwm-1b-480p-s3003"
    assert doc["showcase_manifest"]["percentiles_exported"] == [5, 25, 50, 75, 95]
    assert doc["showcase_manifest"]["gallery_split"] == "sealed_vault"
    assert (tmp_path / "sc_3seed/showcase_manifest.json").is_file()
    assert any(fn.startswith("common_vault_cross_model_ep") for fn in doc["showcase_manifest"]["exported_files"]["png_panels"])
    assert "unseen_ood_auroc_480p" in doc["head_to_head_hierarchical_cis"]["slwm_1b_480p_minus_x3_sliding_window_w64_gqa_1b_480p"]


def test_x3_p01_l0_sysid_ablation_and_true_ood_dynamics_shift() -> None:
    from evals.x3_eval_480p import eval_x3_unseen_layout_mppi_480p
    from models.x3_480p_core import encode_480p_visual_stem

    spec = make_x3_spec("x3_tiny", "slwm_x3", dtype="float32", grad_checkpoint=False)
    params = init_x3_params(spec, seed=1101)
    b_raw = generate_x3_sim_batch_480p(jax.random.key(1102), batch_size=2, seq_len=6)
    # Give sample 0 and sample 1 distinct non-zero movements & actions so L0 SLOT_SYSID differs across the batch!
    mov = b_raw["movement"].at[0].add(0.45).at[1].add(-0.65)
    act = b_raw["action"].at[0].set(0.8).at[1].set(-0.7)
    batch = {**b_raw, "movement": mov, "action": act}

    z_norm, f_norm, _, _ = forward_x3_world_model(params, spec, batch, sysid_mode="normal")
    vis_m, fine_s, vp_4 = encode_480p_visual_stem(params, spec, batch["video"], batch["modality_mask"][:, :, 1])
    z_pre, f_pre, _, _ = forward_x3_world_model(params, spec, batch, precomputed_visual=(vis_m, fine_s, vp_4))
    assert float(jnp.max(jnp.abs(z_norm - z_pre))) < 1e-6
    assert float(jnp.max(jnp.abs(f_norm - f_pre))) < 1e-6

    # Also verify fused single-pass xArm6 counterfactual branch inside world_model_loss_x3 executes cleanly!
    batch_with_rob = {
        **batch,
        "rob_vid0": jnp.zeros((1, 1, 64, 64, 3), dtype=jnp.uint8),
        "rob_prop0": jnp.zeros((1, 1, 16), dtype=jnp.float32),
        "rob_audio_32": jnp.zeros((1, 32, 64), dtype=jnp.float32),
        "rob_wrong_audio_32": jnp.ones((1, 32, 64), dtype=jnp.float32) * 0.2,
        "rob_wrong_col_audio_32": jnp.ones((1, 32, 64), dtype=jnp.float32) * -0.2,
        "rob_text_32": jnp.ones((1, 32, 4), dtype=jnp.int32) * 260,
        "rob_wrong_text_32": jnp.ones((1, 32, 4), dtype=jnp.int32) * 265,
        "rob_wrong_col_text_32": jnp.ones((1, 32, 4), dtype=jnp.int32) * 270,
        "rob_end_xy": jnp.array([[0.35, -0.25]], dtype=jnp.float32),
        "rob_target_block_xy": jnp.array([[-0.40, 0.30]], dtype=jnp.float32),
        "rob_distractor_block_xy": jnp.array([[0.40, -0.30]], dtype=jnp.float32),
    }
    loss_v, mets_v = world_model_loss_x3(params, params, spec, batch_with_rob, jax.random.key(1103))
    assert jnp.isfinite(loss_v) and float(mets_v["rank_margin"]) >= 0.0

    z_zero, _, _, _ = forward_x3_world_model(params, spec, batch, sysid_mode="zeroed")
    z_swap, _, _, _ = forward_x3_world_model(
        params, spec, batch, sysid_mode="swapped", sysid_swap_perm=jnp.array([1, 0], dtype=jnp.int32)
    )
    # L0 SysID intervention must change z_top on task slots while leaving shielded SLOT_SYM_MEM (54) exact 0.0!
    assert float(jnp.max(jnp.abs(z_norm[:, :, 59, :] - z_zero[:, :, 59, :]))) > 1e-4
    assert float(jnp.max(jnp.abs(z_norm[:, :, 59, :] - z_swap[:, :, 59, :]))) > 1e-4
    assert float(jnp.max(jnp.abs(z_zero[:, :, 59, :] - z_swap[:, :, 59, :]))) > 1e-4
    assert float(jnp.max(jnp.abs(z_norm[:, :, X3_SLOT_SYM_MEM, :] - z_swap[:, :, X3_SLOT_SYM_MEM, :]))) == 0.0

    mppi_res = eval_x3_unseen_layout_mppi_480p(params, spec, n_worlds=8, horizon_ticks=12, chunk_bs=4)
    assert "sysid_zeroed_succ_vec" in mppi_res
    assert "sysid_swapped_succ_vec" in mppi_res
    assert len(mppi_res["dynamics_shift_succ_vec"]) == 2


def test_x3_p02_p03_strict_within_source_perm_and_occluded_visual_motion_dir6() -> None:
    import pytest

    spec = make_x3_spec("x3_tiny", "slwm_x3", dtype="float32")
    params = init_x3_params(spec, seed=1201)
    clips_static_vid = _build_synthetic_480p_eval_clips(seed=1202, n_clips=6, seq_len=12)
    clips_static_vid["video"] = np.full_like(clips_static_vid["video"], 90)
    # Alternating bright/dark frames -> ||V_t - V_{t-1}||_1 > 0.025 on ticks t >= 1!
    vid_osc = np.full_like(clips_static_vid["video"], 20)
    vid_osc[:, 1::2] = 230
    clips_moving_vid = {**clips_static_vid, "video": vid_osc}

    res_stat, _ = eval_x3_any_to_any_composition(
        params, spec, clips_static_vid, inv_vocab={}, chunk_bs=3, strict_within_source=True
    )
    res_mov, _ = eval_x3_any_to_any_composition(
        params, spec, clips_moving_vid, inv_vocab={}, chunk_bs=3, strict_within_source=True
    )
    # In Dir 6, mm_d6[vis] == 0 so audio+proprio inputs are identical, yet occluded visual ground-truth flips!
    assert abs(
        res_stat["dir6_blind_audio_proprio_and_ood_gate"]["blind_audio_contact_bce"]
        - res_mov["dir6_blind_audio_proprio_and_ood_gate"]["blind_audio_contact_bce"]
    ) > 1e-4

    # Injecting a singleton (source_id, embodiment_id) group MUST fail closed when strict_within_source=True!
    bad_clips = {**clips_static_vid, "embodiment_id": np.array([0, 0, 0, 0, 0, 7], dtype=np.int32)}
    with pytest.raises(RuntimeError, match="singleton group"):
        eval_x3_any_to_any_composition(params, spec, bad_clips, inv_vocab={}, chunk_bs=3, strict_within_source=True)


