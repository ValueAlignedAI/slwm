"""Unit & Integration Verification Suite for Study X2 Wave 8A (`SLWM-X2`).

Tests:
  1. `< 0.01%` strict parameter parity (`x150m`, `x450m`, `x1b`) and `core` vs `strict_system` state footprints.
  2. `fork_isolated_latent_streams` (`jax.custom_vjp`) + disjoint subspace softmax zero-leakage gradient contract.
  3. Strict `ee_only` proprioceptive de-privileging (zero object/velocity/angle leakage) + `32 -> 64` batch padding.
  4. Strict causal invariance and multi-tick (`chunk in {1, 7, 16, 48}`) cached streaming equivalence (`T = 96 > W = 64`).
  5. Shortcut-free `T = 160` world statistical independence (`N = 2048`) & `T = 64` (`lag >= 33 > 28`) token schedule.
  6. Fail-closed Gate 4 evaluator & deterministic median-seed showcase selection.
  7. End-to-end `x_tiny` joint recall + spatial pose optimization convergence, showcase gallery, and §13.1 hygiene.
"""

from __future__ import annotations

import json
from pathlib import Path
import wave

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from data.x1b_multimodal import ASSOC_VAL_OFFSET, generate_synthetic_world_batch
from evals.x1b_eval_wave8a import (
    _propose_subgoal_candidates_shared,
    _render_dynamic_visual_stream,
    _score_subgoal_candidates_shared,
    build_multi_item_overwrite_probe_batch,
    eval_beyond_window_T160_distractor_switch_v8a,
    eval_multi_item_overwrite_capacity_v8a,
    eval_unseen_ood_corruptions_v8a,
    init_shortcut_free_t160_switch_worlds,
    plan_oracle_perception_dynamics_mppi_v8a,
)
from evals.x1b_showcase_wave8a import PRE_REGISTERED_COMMON_EPISODES, render_wave8a_showcase_gallery
from evals.x1b_stats import hierarchical_paired_auroc_bootstrap_ci, hierarchical_seed_bootstrap_ci
from models.x1b_sigreg_epps_pulley import epps_pulley_sigreg_loss, vicreg_var_cov_loss
from models.x1b_wave8a_core import (
    Wave8AInterventions,
    compute_disjoint_subspace_text_losses,
    decode_goal_latch_wave8a,
    fork_isolated_latent_streams,
    forward_world_model_wave8a,
    init_decoder_streaming_carry,
    world_model_loss_wave8a,
)
from models.x1b_world import (
    SLOT_TEXT,
    SLOTS_VISUAL,
    compute_streaming_state_bytes,
    count_parameters,
    count_spec_parameters,
    init_streaming_memories,
    init_x1b_params,
    make_x1b_spec,
)
from scripts.data.x1b_prepare_fedora import (
    causal_decimate_pcm_to_16khz,
    pcm_to_causal_logmel_64,
    wav_to_logmel_64,
)
from scripts.eval.x1b_analyze_wave8a import (
    COHORT_NAMES,
    EXPECTED_SEEDS,
    HEADLINE_COHORT_METRICS,
    REQUIRED_HIER_COMPS,
    REQUIRED_HIER_METRICS,
    REQUIRED_SHOWCASE_FIGURES,
    evaluate_gate4_checks,
    select_median_seed_showcase_paths,
)
from scripts.train.x1b_train import memory_efficient_adamw_step
from scripts.train.x1b_train_wave8a import (
    build_multi_item_assoc_tokens_t64,
    generate_in_domain_deprivileged_sim_batch_v8a,
    mix_wave8a_batches,
    train_and_evaluate_wave8a,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_gqa_and_diagonal_ssm_parameter_parity_and_state_bytes():
    """Verify <0.01% parameter parity and exact core vs strict_system B=1 streaming state footprints."""
    for prof in ("x_tiny", "x150m", "x450m", "x1b"):
        sp_ref = make_x1b_spec(prof, "slwm_x1b")
        sp_gqa = make_x1b_spec(prof, "sliding_window_gqa_transformer")
        sp_dssm = make_x1b_spec(prof, "diagonal_ssm_hybrid")

        n_ref = count_spec_parameters(sp_ref)["strict_total"]
        n_gqa = count_spec_parameters(sp_gqa)["strict_total"]
        n_dssm = count_spec_parameters(sp_dssm)["strict_total"]

        assert n_dssm == n_ref, f"DiagonalSSM parameter mismatch on {prof}: {n_dssm} != {n_ref}"
        rel_delta = abs(n_gqa - n_ref) / float(n_ref)
        if prof in ("x150m", "x450m", "x1b"):
            assert rel_delta < 1e-4, f"GQA parameter parity exceeded 0.01% on {prof}: {rel_delta:.6%}"

    sp450_slwm = make_x1b_spec("x450m", "slwm_x1b")
    sp450_gqa = make_x1b_spec("x450m", "sliding_window_gqa_transformer")
    sp450_dssm = make_x1b_spec("x450m", "diagonal_ssm_hybrid")

    assert count_spec_parameters(sp450_slwm)["strict_total"] == 453_199_938
    assert count_spec_parameters(sp450_gqa)["strict_total"] == 453_224_778
    assert count_spec_parameters(sp450_dssm)["strict_total"] == 453_199_938

    m_slwm = compute_streaming_state_bytes(sp450_slwm, batch_size=1, cache_len=64)
    m_gqa = compute_streaming_state_bytes(sp450_gqa, batch_size=1, cache_len=64)
    m_dssm = compute_streaming_state_bytes(sp450_dssm, batch_size=1, cache_len=64)

    dec_carry = init_decoder_streaming_carry(sp450_slwm, batch_size=1)
    actual_dec_bytes = int(sum(int(np.prod(a.shape)) * int(a.dtype.itemsize) for a in jax.tree.leaves(dec_carry)))
    assert actual_dec_bytes == 55_397
    assert m_slwm["decoder_carry_bytes"] == actual_dec_bytes == 55_397
    assert m_slwm["controller_state_bytes"] == 292
    assert m_slwm["audio_frontend_state_bytes"] == 10876
    assert m_slwm["strict_system_state_bytes"] == m_slwm["core_state_bytes"] + 10876 + 292 + 55397
    assert abs(m_slwm["core_state_mib"] - 8.7898) < 1e-4
    assert abs(m_slwm["strict_system_state_mib"] - 8.8533) < 1e-4
    assert abs(m_gqa["core_state_mib"] - 13.5264) < 1e-4
    assert abs(m_gqa["strict_system_state_mib"] - 13.5898) < 1e-4
    assert abs(m_gqa["kv_cache_bytes"] / float(1 << 20) - 13.50) < 1e-6
    assert abs(m_dssm["core_state_mib"] - 4.9138) < 1e-4
    assert abs(m_dssm["strict_system_state_mib"] - 4.9773) < 1e-4

    for arch in ("slwm_x1b", "sliding_window_gqa_transformer", "diagonal_ssm_hybrid"):
        sp = make_x1b_spec("x_tiny", arch)
        p = init_x1b_params(sp, seed=7)
        assert count_parameters(p)["strict_total"] == count_spec_parameters(sp)["strict_total"]


def test_fork_isolated_latent_streams_and_disjoint_subspace_gradients():
    """Verify g_task and goal/LM softmaxes produce bit-for-bit zero gradient on SLOT_TEXT and head_text_w[:, 64:96]."""
    z = jax.random.normal(jax.random.key(11), (2, 5, 24, 32), dtype=jnp.float32)

    def _task_only_obj(z_in):
        zm, zt = fork_isolated_latent_streams(z_in, zmem_isolation=True)
        del zm
        return jnp.sum(zt * 3.0)

    g_iso = jax.grad(_task_only_obj)(z)
    assert float(jnp.max(jnp.abs(g_iso[:, :, SLOT_TEXT, :]))) == 0.0
    assert np.allclose(np.asarray(g_iso[:, :, SLOTS_VISUAL, :]), 3.0)

    sp = make_x1b_spec("x_tiny", "slwm_x1b")
    p = init_x1b_params(sp, seed=12)
    z_raw = jax.random.normal(jax.random.key(13), (2, 6, 24, sp.d_model), dtype=jnp.float32)
    iv_iso = Wave8AInterventions(zmem_isolation=True)
    g_latch = jax.grad(lambda zin: jnp.sum(decode_goal_latch_wave8a(p, zin, iv_iso) ** 2))(z_raw)
    assert float(jnp.max(jnp.abs(g_latch[:, :, SLOT_TEXT, :]))) == 0.0

    # Verify goal_sym_ce + lm_ce produce exact 0.0 gradient on z_raw[:, :, SLOT_TEXT, :] and head_text_w[:, 64:96]
    ttxt = jnp.array([[3600, 3605, 120, 70, 5, 0], [3710, 210, 0, 68, 7, 3590]], dtype=jnp.int32)
    txt_mask = jnp.ones((2, 6), dtype=jnp.float32)
    scen_arr = jnp.array([0, -1], dtype=jnp.int32)

    def _non_assoc_loss(z_in, hw):
        zm, zt = fork_isolated_latent_streams(z_in, zmem_isolation=True)
        p_sub = {**p, "head_text_w": hw}
        _, goal_ce, lm_ce = compute_disjoint_subspace_text_losses(
            p_sub, sp, zm, zt, {}, ttxt, txt_mask, scen_arr, iv_iso
        )
        return goal_ce + lm_ce

    gz_non_assoc, ghw_non_assoc = jax.grad(_non_assoc_loss, argnums=(0, 1))(z_raw, p["head_text_w"])
    assert float(jnp.max(jnp.abs(gz_non_assoc[:, :, SLOT_TEXT, :]))) == 0.0
    assert float(jnp.max(jnp.abs(ghw_non_assoc[:, 64:96]))) == 0.0


def test_deprivileged_proprio_zero_object_leakage_and_batch_padding():
    """Verify legacy proprio[2:10] corruption has exact 0.0 effect on loss and 32->64 padding sets valid[:, 32:]=False."""
    sp = make_x1b_spec("x_tiny", "slwm_x1b")
    p = init_x1b_params(sp, seed=19)
    iv = Wave8AInterventions(zmem_isolation=True)

    raw_leg = generate_synthetic_world_batch(jax.random.key(20), 4, seq_len=16, vocab_size=sp.vocab_size)
    corrupted_p = raw_leg["proprio"].at[:, :, 2:10].set(999.0).at[:, :, 11:].set(-777.0)
    raw_leg_corrupt = {**raw_leg, "proprio": corrupted_p}

    l_clean, _ = world_model_loss_wave8a(p, p, sp, raw_leg, iv, jax.random.key(21))
    l_corrupt, _ = world_model_loss_wave8a(p, p, sp, raw_leg_corrupt, iv, jax.random.key(21))
    assert float(jnp.abs(l_clean - l_corrupt)) == 0.0

    sim64 = generate_in_domain_deprivileged_sim_batch_v8a(jax.random.key(22), 6, seq_len=64, vocab_size=sp.vocab_size)
    pr = np.asarray(sim64["proprio"])
    # In mask_privileged_proprio, [2:4] duplicates ee_xy ([0:2]), [6:7] == 0, [7:8] == contact, [8:16] == 0
    assert float(np.max(np.abs(pr[:, :, 2:4] - pr[:, :, 0:2]))) == 0.0
    assert float(np.max(np.abs(pr[:, :, 6:7]))) == 0.0
    assert float(np.max(np.abs(pr[:, :, 8:]))) == 0.0

    # Test mix_wave8a_batches right-padding 32-tick real/robot clips to seq_len=64
    r32 = generate_synthetic_world_batch(jax.random.key(23), 2, seq_len=32, vocab_size=sp.vocab_size)
    r32 = {**r32, "scenario": jnp.full((2,), -1, dtype=jnp.int32)}
    mixed = mix_wave8a_batches(r32, None, sim64, seq_len=64)
    assert mixed["video"].shape == (8, 64, 64, 64, 3)
    assert bool(jnp.all(mixed["valid"][:2, :32]))
    assert not bool(jnp.any(mixed["valid"][:2, 32:]))
    assert float(jnp.max(jnp.abs(mixed["proprio"][:, :, 2:4] - mixed["proprio"][:, :, 0:2]))) == 0.0
    assert float(jnp.max(jnp.abs(mixed["proprio"][:, :, 8:]))) == 0.0


def test_wave8a_strict_causality_and_multitick_chunked_streaming_equivalence():
    """Verify zero future lookahead and multi-tick cached chunking equivalence (T=96 > W=64) across architectures."""
    t_seq = 12
    t_cut = 6
    batch_a = generate_in_domain_deprivileged_sim_batch_v8a(jax.random.key(31), 2, seq_len=t_seq, vocab_size=512)
    batch_b = {
        **batch_a,
        "video": batch_a["video"].at[:, t_cut:].set(255 - batch_a["video"][:, t_cut:]),
        "audio": batch_a["audio"].at[:, t_cut:].set(-3.0),
        "proprio": batch_a["proprio"].at[:, t_cut:].set(0.95),
        "text": batch_a["text"].at[:, t_cut:].set(77),
        "action": batch_a["action"].at[:, t_cut:].set(-0.8),
    }

    for arch in ("slwm_x1b", "sliding_window_gqa_transformer", "diagonal_ssm_hybrid"):
        iv = Wave8AInterventions(zmem_isolation=True)
        sp = make_x1b_spec("x_tiny", arch, dtype="float32", grad_checkpoint=False, vocab_size=512)
        p = init_x1b_params(sp, seed=32)
        za, _, _ = forward_world_model_wave8a(p, sp, batch_a, iv)
        zb, _, _ = forward_world_model_wave8a(p, sp, batch_b, iv)
        assert float(jnp.max(jnp.abs(za[:, :t_cut] - zb[:, :t_cut]))) < 1e-5

    # Test T=96 (> W=64) multi-tick cached chunking equivalence across chunk sizes {1, 7, 16, 48}
    b96 = generate_in_domain_deprivileged_sim_batch_v8a(jax.random.key(33), 2, seq_len=96, vocab_size=512)
    for arch in ("sliding_window_gqa_transformer", "slwm_x1b", "diagonal_ssm_hybrid"):
        sp = make_x1b_spec("x_tiny", arch, dtype="float32", grad_checkpoint=False, vocab_size=512)
        p = init_x1b_params(sp, seed=34)
        iv = Wave8AInterventions(zmem_isolation=True)
        z_full, _, _ = forward_world_model_wave8a(p, sp, b96, iv)
        for chunk in (1, 7, 16, 48):
            mems = init_streaming_memories(sp, 2, cache_len=64)
            z_chunks = []
            for t0 in range(0, 96, chunk):
                t1 = min(96, t0 + chunk)
                cb = {
                    "video": b96["video"][:, t0:t1],
                    "audio": b96["audio"][:, t0:t1],
                    "proprio": b96["proprio"][:, t0:t1],
                    "text": b96["text"][:, t0:t1],
                    "action": b96["action"][:, t0:t1],
                    "goal": b96["goal"],
                    "modality_mask": b96["modality_mask"][:, t0:t1],
                    "valid": b96["valid"][:, t0:t1],
                }
                zc, mems, _ = forward_world_model_wave8a(p, sp, cb, iv, memories=mems, offset=t0)
                z_chunks.append(zc)
            z_cat = jnp.concatenate(z_chunks, axis=1)
            diff = float(jnp.max(jnp.abs(z_full - z_cat)))
            assert diff < 3e-4, f"Multi-tick chunk={chunk} mismatch on {arch}: {diff}"

    # Verify decode_world_heads_wave8a chunked streaming equivalence under temporal modality dropout (Codex Round 7)
    from models.x1b_jepa import predict_latent_step
    from models.x1b_wave8a_core import (
        decode_world_heads_wave8a,
        extract_decoder_streaming_carry,
        init_decoder_streaming_carry,
    )

    b8_base = generate_in_domain_deprivileged_sim_batch_v8a(jax.random.key(35), 2, seq_len=8, vocab_size=512)
    vis_pat_a = jnp.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    vis_pat_b = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    sp_dec = make_x1b_spec("x_tiny", "slwm_x1b", dtype="float32", grad_checkpoint=False, vocab_size=512)
    p_dec = init_x1b_params(sp_dec, seed=36)
    for dec_heads_flag in (True, False):
        iv_dec = Wave8AInterventions(zmem_isolation=True, decoupled_heads=dec_heads_flag)
        for vis_pat in (vis_pat_a, vis_pat_b):
            mm8 = jnp.ones((2, 8, 4), dtype=jnp.float32).at[:, :, 1].set(vis_pat[None, :])
            b8 = {
                **b8_base,
                "video": jnp.where(mm8[:, :, 1:2, None, None] > 0.5, b8_base["video"], jnp.zeros_like(b8_base["video"])),
                "modality_mask": mm8,
            }
            z8_full, _, _ = forward_world_model_wave8a(p_dec, sp_dec, b8, iv_dec)
            zm8_full, zt8_full = fork_isolated_latent_streams(z8_full, iv_dec.zmem_isolation)
            zn8_full, lv8_full = predict_latent_step(p_dec, sp_dec, zt8_full, b8["action"], residual=True)
            hd_full = decode_world_heads_wave8a(
                p_dec,
                sp_dec,
                zm8_full,
                zt8_full,
                zn8_full,
                lv8_full,
                b8["goal"],
                iv_dec,
                obs_ee_proprio=b8["proprio"],
                obs_video=b8["video"],
                obs_modality_mask=b8["modality_mask"],
                obs_valid=b8["valid"],
            )
            checked_keys = (
                "vis_ee_curr",
                "vis_obj_curr",
                "proprio_curr",
                "proprio_next",
                "action_proposal",
                "action_proposal_trusted",
                "value_pred",
                "commit_logits",
                "innov_feats",
                "text_logits",
                "diag_video",
                "diag_audio",
                "hazard_logits_8x8",
                "pred_target_block_xy",
            )
            for chunk in (1, 2, 3, 4):
                mems_d = init_streaming_memories(sp_dec, 2, cache_len=64)
                dec_carry = init_decoder_streaming_carry(sp_dec, 2)
                hd_chunks: dict[str, list[jax.Array]] = {k: [] for k in checked_keys}
                for t0 in range(0, 8, chunk):
                    t1 = min(8, t0 + chunk)
                    cb = {
                        "video": b8["video"][:, t0:t1],
                        "audio": b8["audio"][:, t0:t1],
                        "proprio": b8["proprio"][:, t0:t1],
                        "text": b8["text"][:, t0:t1],
                        "action": b8["action"][:, t0:t1],
                        "goal": b8["goal"] if b8["goal"].ndim == 2 else b8["goal"][:, t0:t1],
                        "modality_mask": b8["modality_mask"][:, t0:t1],
                        "valid": b8["valid"][:, t0:t1],
                    }
                    zc, mems_d, _ = forward_world_model_wave8a(p_dec, sp_dec, cb, iv_dec, memories=mems_d, offset=t0)
                    zmc, ztc = fork_isolated_latent_streams(zc, iv_dec.zmem_isolation)
                    znc, lvc = predict_latent_step(p_dec, sp_dec, ztc, cb["action"], residual=True)
                    hdc = decode_world_heads_wave8a(
                        p_dec,
                        sp_dec,
                        zmc,
                        ztc,
                        znc,
                        lvc,
                        cb["goal"],
                        iv_dec,
                        obs_ee_proprio=cb["proprio"],
                        obs_video=cb["video"],
                        obs_modality_mask=cb["modality_mask"],
                        obs_valid=cb["valid"],
                        decoder_carry=dec_carry,
                    )
                    dec_carry = extract_decoder_streaming_carry(hdc)
                    for hk in checked_keys:
                        hd_chunks[hk].append(hdc[hk])
                for hk in checked_keys:
                    cat_v = jnp.concatenate(hd_chunks[hk], axis=1)
                    hdiff = float(jnp.max(jnp.abs(hd_full[hk] - cat_v)))
                    assert hdiff < 3e-4, f"Decoded head {hk} chunk={chunk} decoupled={dec_heads_flag} mismatch: {hdiff}"


def test_shortcut_free_t160_and_t64_beyond_local_rope_schedule():
    """Verify I(Goal B ; obs_{t >= 20}) == 0 on N=2048 worlds and lag >= 33 > 28 on T=64 training tokens."""
    sw = init_shortcut_free_t160_switch_worlds(jax.random.key(40), batch_size=2048)
    gb = np.asarray(sw["targ_b"])
    for key in ("ee_xy", "obj_xy", "reset_ee_xy", "reset_obj_xy", "false_distractor_xy"):
        arr = np.asarray(sw[key])
        for d_i in range(2):
            for d_j in range(2):
                corr = float(np.corrcoef(arr[:, d_i], gb[:, d_j])[0, 1])
                assert abs(corr) < 0.08, f"Spurious Goal-B correlation on {key}[{d_i},{d_j}]: {corr:.4f}"

    vid = _render_dynamic_visual_stream(41, 4)
    for horizon in (64, 256):
        for k_items in (1, 2, 4, 8, 16):
            probe = build_multi_item_overwrite_probe_batch(jax.random.key(50 + k_items), 4, horizon, k_items, vid)
            m_ov = int(probe["m_over"])
            targs = np.asarray(probe["target_text"][:, -k_items:])
            stales = np.asarray(probe["stale_text"][:, -k_items:])
            assert np.all(targs >= ASSOC_VAL_OFFSET)
            assert np.all(targs[:, :m_ov] != stales[:, :m_ov])
            if k_items > m_ov:
                assert np.all(targs[:, m_ov:] == stales[:, m_ov:])

    txt64, ttxt64 = build_multi_item_assoc_tokens_t64(jax.random.key(61), 32, 64)
    assert txt64.shape == (32, 64, 4)
    assert ttxt64.shape == (32, 64)
    # Writes/overwrites strictly at t <= 23; queries strictly at t >= 56 (lag >= 33 > 28-tick 4-layer W=8 RoPE span!)
    assert int(jnp.sum(txt64[:, 24:56, :])) == 0
    assert int(jnp.sum(ttxt64[:, :56])) == 0
    assert int(jnp.sum(ttxt64[:, 56:] >= ASSOC_VAL_OFFSET)) > 0


def test_t160_retention_trajectory_bitwise_goal_b_invariance_and_ood_auroc_decomposition():
    """Verify bitwise equality of obs_{20..88} across counterfactual Goal-B swaps and exact OOD AUROC decomposition."""
    sp = make_x1b_spec("x_tiny", "slwm_x1b", dtype="float32", grad_checkpoint=False)
    p = init_x1b_params(sp, seed=104)
    iv = Wave8AInterventions(zmem_isolation=True)

    t160 = eval_beyond_window_T160_distractor_switch_v8a(p, sp, iv, seed=202802, n_worlds=4, return_obs_trace=True)
    assert "T160_paired_causal_switch_vec" in t160
    assert len(t160["T160_paired_causal_switch_vec"]) == 4
    obs_b = t160["obs_trace_b"]
    obs_bp = t160["obs_trace_bp"]
    # Across all 69 ticks t = 20..88 (> W = 64), every sensory & action input tensor must be bitwise identical!
    for obs_key in ("video", "proprio", "action", "text"):
        assert np.array_equal(
            obs_b[obs_key][:, 20:89], obs_bp[obs_key][:, 20:89]
        ), f"Counterfactual Goal-B trajectory leaked into {obs_key} during t=20..88!"
    # Sanity check: at t = 16..19 (when Goal B vs Goal B' is instructed), text tokens DO differ
    assert not np.array_equal(obs_b["text"][:, 16:20], obs_bp["text"][:, 16:20])

    ood = eval_unseen_ood_corruptions_v8a(p, sp, iv, seed=202804, n_worlds=8)
    assert abs(float(np.mean(ood["ood_ep_auroc_vec"])) - float(ood["mean_unseen_ood_corruption_auroc"])) < 1e-12
    assert "clean_world_scores" in ood and "corrupted_world_scores" in ood
    assert np.asarray(ood["clean_world_scores"]).shape[0] == 8

    # Verify multi-item overwrite capacity exports overwritten_vec for K8_T256 and K16_T1024 CIs
    mi_cap = eval_multi_item_overwrite_capacity_v8a(
        p, sp, iv, seed=202805, n_episodes=4, k_items_list=(8, 16), horizons=(64,)
    )
    assert "K8_T64_overwritten_vec" in mi_cap and len(mi_cap["K8_T64_overwritten_vec"]) == 4
    assert "K16_T64_overwritten_vec" in mi_cap and len(mi_cap["K16_T64_overwritten_vec"]) == 4


def test_unified_mppi_functional_and_paired_world_auroc_bootstrap():
    """Verify shared subgoal MPPI objective and exact paired-world Mann-Whitney AUROC bootstrap (Codex counterexample)."""
    from data.x1b_unseen_worlds import init_unseen_layout_worlds

    obj_xy = jnp.array([[-0.3, -0.2]], dtype=jnp.float32)
    targ_xy = jnp.array([[0.4, 0.3]], dtype=jnp.float32)
    occ_cal = jnp.zeros((1, 8, 8), dtype=jnp.float32)
    v_grid = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32).reshape(1, 8, 8)
    cand_wps, v_wp_shared, occ_wp, reach_ring = _propose_subgoal_candidates_shared(
        obj_xy, targ_xy, occ_cal, v_grid, 16
    )
    assert cand_wps.shape == (1, 16, 2)
    assert v_wp_shared.shape == (1, 16)
    r_ret = jnp.linspace(-2.0, -0.5, 16, dtype=jnp.float32)[None, :]
    f_v = jnp.linspace(0.2, 0.9, 16, dtype=jnp.float32)[None, :]
    sc_roll = _score_subgoal_candidates_shared(
        obj_xy, targ_xy, cand_wps, v_wp_shared, occ_wp, reach_ring, occ_cal, r_ret, f_v, None, False, use_rollout=True
    )
    sc_noroll = _score_subgoal_candidates_shared(
        obj_xy, targ_xy, cand_wps, v_wp_shared, occ_wp, reach_ring, occ_cal, r_ret, f_v, None, False, use_rollout=False
    )
    assert np.allclose(np.asarray(sc_roll - sc_noroll), np.asarray(0.14 * r_ret - 0.12 * f_v), atol=1e-5)

    st_ora = init_unseen_layout_worlds(jax.random.key(202), 4)
    u_ora, wp_ora = plan_oracle_perception_dynamics_mppi_v8a(
        st_ora, st_ora["targ_a"], st_ora["obj_xy"], False, horizon=8, num_candidates=16
    )
    assert u_ora.shape == (4, 4) and wp_ora.shape == (4, 2)

    # Codex Round 3 exact 2-world paired-score counterexample:
    # clean_a = [-1, 11], corr_a = [0, 10] -> full-pool AUROC == 0.5, fixed-clean-pool ep_auroc == [0.5, 0.5],
    # whereas true paired-world resampling yields 1.0 on [w0, w0] and 0.0 on [w1, w1]!
    clean_a = np.array([[[-1.0], [11.0]]], dtype=np.float64)  # [S=1, W=2, T=1]
    corr_a = np.array([[[[0.0], [10.0]]]], dtype=np.float64)  # [S=1, K=1, W=2, T=1]
    clean_b = np.zeros((1, 2, 1), dtype=np.float64)
    corr_b = np.zeros((1, 1, 2, 1), dtype=np.float64)  # All ties -> 0.5 on every world draw

    single_ci = hierarchical_paired_auroc_bootstrap_ci(corr_a, clean_a, n_boot=4000, rng_seed=7)
    assert abs(float(single_ci["mean"]) - 0.5) < 1e-12
    assert single_ci["hierarchical_ci95"] == [0.0, 1.0]

    paired_ci = hierarchical_paired_auroc_bootstrap_ci(corr_a, clean_a, corr_b, clean_b, n_boot=4000, rng_seed=7)
    assert paired_ci["paired_world_auroc_recomputed"] is True
    assert abs(float(paired_ci["mean"]) - 0.0) < 1e-12
    lo, hi = paired_ci["hierarchical_ci95"]
    # Roughly 25% of draws are [w0, w0] (diff = +0.5) and 25% are [w1, w1] (diff = -0.5) -> CI must be [-0.5, +0.5]!
    assert abs(lo - (-0.5)) < 1e-9 and abs(hi - 0.5) < 1e-9


def test_wave8a_grounded_builder_color_direction_and_sync_contracts():
    """Verify exact GRID colors, yellow rejection, physical-semantic direction agreement, & synchronized frame indices."""
    import io
    from PIL import Image

    from scripts.data.x1b_prepare_wave8a_grounded_fedora import (
        COLORS,
        OPPOSITE_DIR,
        _decode_frames_synchronized,
        _strict_color_in_goal,
        _text_matches_physical_dir,
    )

    assert COLORS == ("red", "blue", "green", "white")
    assert OPPOSITE_DIR == {0: 1, 1: 0, 2: 3, 3: 2}
    assert _strict_color_in_goal("push the white star toward the left") == (3, None)
    assert _strict_color_in_goal("move the yellow hexagon up")[0] is None
    assert _strict_color_in_goal("slide the wooden block left")[0] is None

    assert _text_matches_physical_dir("push the red block to the left", 0) is True
    assert _text_matches_physical_dir("push the red block to the right", 0) is False
    assert _text_matches_physical_dir("push the red block from right to left", 0) is False

    # Build 5 JPEG frames where frame 1 duplicates frame 0 and frame 3 duplicates frame 2 -> kept_indices == [0, 2, 4]
    def _jpeg(val: int) -> memoryview:
        buf = io.BytesIO()
        Image.fromarray(np.full((16, 16, 3), val, dtype=np.uint8)).save(buf, format="JPEG", quality=95)
        return memoryview(buf.getvalue())

    jpegs = [_jpeg(20), _jpeg(20), _jpeg(120), _jpeg(120), _jpeg(220)]
    vid, kept = _decode_frames_synchronized(jpegs, state_len=5, max_t=8, img_res=16)
    assert vid.shape == (8, 16, 16, 3)
    assert kept.tolist() == [0, 2, 4]


def test_wave8a_registered_dataset_cards_and_row_level_provenance():
    """Verify frozen v2 dataset cards, 24,792-row disposition ledger, 0 cross-split overlap, & dual counterfactuals."""
    import gzip
    import hashlib

    from scripts.data.x1b_prepare_wave8a_grounded_fedora import OPPOSITE_DIR

    cards = json.loads((REPO_ROOT / "configs/x1b/wave8a_dataset_cards.json").read_text())
    prov = json.loads((REPO_ROOT / "configs/x1b/wave8a_dataset_provenance.json").read_text())
    prereg = json.loads((REPO_ROOT / "configs/x1b/wave8a_showcase_preregistration.json").read_text())
    assert prereg["schema_version"] == "wave8a_showcase_preregistration_v1"

    rb = cards["x1b_real_bank_v2"]
    assert rb["audio_frontend"]["causal"] is True
    assert rb["audio_frontend"]["universal_max_frontend_state_bytes"] == 10876
    assert rb["audio_frontend"]["future_lookahead_samples"] == 0
    assert rb["proprioception_contract"]["greatest_hits"] == "none_masked_zero"
    assert rb["proprioception_contract"]["starss23"] == "none_masked_zero_no_metadata_annotations"

    g_card = cards["x1b_wave8a_grounded_v2"]
    v_card = cards["x1b_wave8a_vault_v2"]
    assert g_card["sha256"] == prov["dataset_sha256"]["x1b_wave8a_grounded_v2"]
    assert v_card["sha256"] == prov["dataset_sha256"]["x1b_wave8a_vault_v2"]
    assert g_card["audio_frontend"]["word_alignment_metadata_used"] is False
    assert g_card["audio_frontend"]["ffmpeg_resampling_used"] is False
    assert g_card["audio_frontend"]["whole_clip_max_abs_normalization_used"] is False
    assert v_card["audio_frontend"]["word_alignment_metadata_used"] is False

    rows_by_split = prov["rows_by_split"]
    retained = sum(len(rows_by_split[s]) for s in ("train", "dev", "vault"))
    excluded = sum(int(v) for v in prov["excluded_counts_by_reason"].values())
    total_scanned = int(prov["total_source_episodes_scanned"])
    assert total_scanned == 24_792
    assert retained + excluded == total_scanned

    # Verify exhaustive 1-to-1 episode disposition ledger across all 24,792 scanned episodes
    disp_path = REPO_ROOT / "configs/x1b/wave8a_episode_dispositions.jsonl.gz"
    assert disp_path.is_file()
    disp_sha = hashlib.sha256(disp_path.read_bytes()).hexdigest()
    assert disp_sha == g_card["episode_disposition_ledger"]["sha256"]
    assert disp_sha == prov["episode_disposition_ledger"]["sha256"]
    with gzip.open(disp_path, "rt", encoding="utf-8") as gf:
        disp_rows = [json.loads(line) for line in gf]
    assert len(disp_rows) == total_scanned
    assert [int(r["scan_index"]) for r in disp_rows] == list(range(total_scanned))
    assert len({r["episode_uid"] for r in disp_rows}) == total_scanned
    assert sum(1 for r in disp_rows if r["disposition"] == "retained") == retained
    assert sum(1 for r in disp_rows if r["disposition"] == "excluded") == excluded

    # Verify zero cross-split overlap across all 5 disjointness keys directly from row-level records
    for field in ("scene_cluster_id", "connected_group_hash", "parent_hash", "color_speaker_id", "dir_speaker_id"):
        sets = {s: {r[field] for r in rows_by_split[s]} for s in ("train", "dev", "vault")}
        assert len(sets["train"] & sets["dev"]) == 0, f"train/dev overlap on {field}"
        assert len(sets["train"] & sets["vault"]) == 0, f"train/vault overlap on {field}"
        assert len(sets["dev"] & sets["vault"]) == 0, f"dev/vault overlap on {field}"

    # Verify every single row satisfies length >= 8, opposite-dir counterfactual, and separated distractor block
    for s, rows in rows_by_split.items():
        for i, r in enumerate(rows):
            assert int(r["row_index"]) == i
            assert int(r["length"]) >= 8
            assert r["strictly_increasing_timestamps"] is True
            assert r["color_name"] in ("red", "blue", "green", "white")
            assert r["distractor_color_name"] in ("red", "blue", "green", "white")
            assert r["distractor_color_name"] != r["color_name"]
            assert float(r["block_sep_l2"]) >= 0.18
            cf_r = rows[int(r["cf_index"])]
            assert int(cf_r["dir_id"]) == OPPOSITE_DIR[int(r["dir_id"])], f"Non-opposite cf in {s}[{i}]"
            assert float(r["cf_dot"]) < -0.02
            assert float(r["cf_disp_l2"]) >= 0.35


def test_raw_pcm_and_wav_causal_logmel_zero_future_leakage_and_padding_invariance(tmp_path: Path):
    """Verify strict 0.0 future leakage on raw PCM/WAV log-mel frontend and exact padding invariance of SIGReg/loss."""
    rng = np.random.default_rng(108)
    pcm_a = rng.standard_normal(51_200).astype(np.float32) * 0.25
    pcm_b = pcm_a.copy()
    pcm_b[25_600:] = rng.standard_normal(25_600).astype(np.float32) * 0.90  # Mutate t >= 1.6s (ticks 16..31)

    mel_a = pcm_to_causal_logmel_64(pcm_a, sr=16_000, n_ticks=32)
    mel_b = pcm_to_causal_logmel_64(pcm_b, sr=16_000, n_ticks=32)
    assert float(np.max(np.abs(mel_a[:16] - mel_b[:16]))) == 0.0
    assert float(np.max(np.abs(mel_a[16:] - mel_b[16:]))) > 1e-2

    # Verify 96 kHz -> 16 kHz (6:1) strict integer causal boxcar decimation has exact 0.0 future leakage
    pcm96_a = rng.standard_normal(307_200).astype(np.float32) * 0.25
    pcm96_b = pcm96_a.copy()
    pcm96_b[153_600:] = rng.standard_normal(153_600).astype(np.float32) * 0.90  # Mutate t >= 1.6s at 96 kHz
    dec_a = causal_decimate_pcm_to_16khz(pcm96_a, 96_000)
    dec_b = causal_decimate_pcm_to_16khz(pcm96_b, 96_000)
    assert dec_a.shape == (51_200,)
    assert float(np.max(np.abs(dec_a[:25_600] - dec_b[:25_600]))) == 0.0
    mel96_a = pcm_to_causal_logmel_64(pcm96_a, sr=96_000, n_ticks=32)
    mel96_b = pcm_to_causal_logmel_64(pcm96_b, sr=96_000, n_ticks=32)
    assert float(np.max(np.abs(mel96_a[:16] - mel96_b[:16]))) == 0.0
    assert float(np.max(np.abs(mel96_a[16:] - mel96_b[16:]))) > 1e-2

    # Verify 24 kHz (STARSS23 FOA W) & 44.1 kHz (GRID MP2 -> 22.05 kHz 2:1 boxcar) strict 0.0 future leakage
    for sr_test, cut_samp in ((24_000, 38_400), (44_100, 70_560)):
        pcm_sr_a = rng.standard_normal(cut_samp * 2).astype(np.float32) * 0.25
        pcm_sr_b = pcm_sr_a.copy()
        pcm_sr_b[cut_samp:] = rng.standard_normal(cut_samp).astype(np.float32) * 0.90
        mel_sr_a = pcm_to_causal_logmel_64(pcm_sr_a, sr=sr_test, n_ticks=32)
        mel_sr_b = pcm_to_causal_logmel_64(pcm_sr_b, sr=sr_test, n_ticks=32)
        assert float(np.max(np.abs(mel_sr_a[:16] - mel_sr_b[:16]))) == 0.0
        assert float(np.max(np.abs(mel_sr_a[16:] - mel_sr_b[16:]))) > 1e-2

    # Non-multiple of 16 kHz must raise on causal_decimate_pcm_to_16khz, and non-supported 22.05 kHz raw input raises
    with pytest.raises(ValueError):
        causal_decimate_pcm_to_16khz(pcm_a, 44_100)
    with pytest.raises(ValueError):
        pcm_to_causal_logmel_64(pcm_a, sr=22_050, n_ticks=32)

    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    for wp, arr in ((wav_a, pcm_a), (wav_b, pcm_b)):
        with wave.open(str(wp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16_000)
            wf.writeframes(np.clip(arr * 32767.0, -32768, 32767).astype(np.int16).tobytes())
    wmel_a = wav_to_logmel_64(wav_a, n_ticks=32)
    wmel_b = wav_to_logmel_64(wav_b, n_ticks=32)
    assert float(np.max(np.abs(wmel_a[:16] - wmel_b[:16]))) == 0.0

    # Verify exact mask-weighted SIGReg & VICReg right-padding invariance
    x32 = jax.random.normal(jax.random.key(109), (32, 24), dtype=jnp.float32)
    x64 = jnp.concatenate([x32, jnp.full((32, 24), 42.0, dtype=jnp.float32)], axis=0)
    m64 = jnp.concatenate([jnp.ones((32,), dtype=jnp.float32), jnp.zeros((32,), dtype=jnp.float32)], axis=0)

    ep32, pr32 = epps_pulley_sigreg_loss(x32, jax.random.key(110), num_slices=32)
    ep64, pr64 = epps_pulley_sigreg_loss(x64, jax.random.key(110), num_slices=32, mask=m64)
    assert abs(float(ep32 - ep64)) < 1e-5
    assert abs(float(pr32 - pr64)) < 1e-5

    vc32, vpr32 = vicreg_var_cov_loss(x32)
    vc64, vpr64 = vicreg_var_cov_loss(x64, mask=m64)
    assert abs(float(vc32 - vc64)) < 1e-5
    assert abs(float(vpr32 - vpr64)) < 1e-5

    # Verify full world_model_loss_wave8a invariance between unpadded T=32 real clip and T=64 right-padded clip
    sp = make_x1b_spec("x_tiny", "slwm_x1b", dtype="float32", grad_checkpoint=False, vocab_size=512)
    p = init_x1b_params(sp, seed=111)
    iv = Wave8AInterventions(zmem_isolation=True)
    r32 = generate_synthetic_world_batch(jax.random.key(112), 2, seq_len=32, vocab_size=512)
    r32 = {
        **r32,
        "scenario": jnp.full((2,), -1, dtype=jnp.int32),
        "ee_only_proprio": jnp.ones((2,), dtype=bool),
        "proprio": r32["proprio"].at[:, :, 2:4].set(r32["proprio"][:, :, 0:2]).at[:, :, 6:7].set(0.0).at[:, :, 8:].set(0.0),
    }
    from scripts.train.x1b_train_wave8a import _pad_temporal_batch_to_seq_len
    r64 = _pad_temporal_batch_to_seq_len(r32, 64)
    l32, met32 = world_model_loss_wave8a(p, p, sp, r32, iv, jax.random.key(113))
    l64, met64 = world_model_loss_wave8a(p, p, sp, r64, iv, jax.random.key(113))
    assert abs(float(met32["sigreg"] - met64["sigreg"])) < 1e-5
    assert abs(float(met32["pr"] - met64["pr"])) < 1e-5
    assert abs(float(l32 - l64)) < 1e-4

    # Verify crossed (Seed x World/Scene) bootstrap CI on Codex 2-scene 3-seed counterexample -> exact [0.0, 1.0]
    codex_2scene = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    ci_2scene = hierarchical_seed_bootstrap_ci(codex_2scene, n_boot=4000, rng_seed=17)
    assert ci_2scene["hierarchical_ci95"] == [0.0, 1.0]

    codex_2clust = np.array(
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    ci_2clust = hierarchical_seed_bootstrap_ci(codex_2clust, cluster_ids=[10, 10, 20, 20], n_boot=4000, rng_seed=17)
    assert ci_2clust["scene_clustered"] is True
    assert ci_2clust["n_level2_clusters"] == 2
    assert ci_2clust["hierarchical_ci95"] == [0.0, 1.0]


def test_fail_closed_gate4_checks_and_median_seed_selector(tmp_path: Path):
    """Verify Gate 4 fails closed when any cohort metric, KxT cell, CI, seed, consistency, or showcase is invalid."""
    def _mk_stat(v: float, n: int = 3) -> dict[str, object]:
        return {"mean": v, "std": 0.0, "values": [v] * n}

    def _mk_cohort(base_v: float, cap_v: float, ov_v: float, t1024_v: float = 0.98) -> dict[str, object]:
        cap_c: dict[str, object] = {}
        for hz in (64, 256, 1024):
            for k_it in (1, 2, 4, 8, 16):
                cap_c[f"K{k_it}_T{hz}_acc"] = _mk_stat(cap_v)
                cap_c[f"K{k_it}_T{hz}_overwritten_acc"] = _mk_stat(ov_v)
        coh: dict[str, object] = {
            "n_seeds": 3,
            "seeds_present": list(EXPECTED_SEEDS),
            "source_code_sha256s": ["a" * 64],
            "provenance_verified": True,
            "multi_item_overwrite": cap_c,
        }
        neg_overrides = {
            "vault_shuffled_scene_color_disc": 0.50,
            "vault_shuffled_scene_color_audio_only_disc": 0.50,
            "vault_color_prior_scene_color_disc": 0.525,
            "vault_paired_flip_dual_color_disc": 0.46,
            "vault_paired_flip_dual_audio_only_disc": 0.42,
        }
        for m_k in HEADLINE_COHORT_METRICS:
            if m_k == "dyn_assoc_T1024":
                coh[m_k] = _mk_stat(t1024_v)
            elif m_k in neg_overrides:
                coh[m_k] = _mk_stat(neg_overrides[m_k])
            else:
                coh[m_k] = _mk_stat(base_v)
        return coh

    iso_cohort = _mk_cohort(0.83, 0.91, 0.89, t1024_v=0.98)
    comp_cohort = _mk_cohort(0.82, 0.90, 0.88, t1024_v=0.97)
    cap_cells = comp_cohort["multi_item_overwrite"]
    base_cohort = comp_cohort

    full_hier: dict[str, object] = {}
    for comp in REQUIRED_HIER_COMPS:
        for mname in REQUIRED_HIER_METRICS:
            entry: dict[str, object] = {
                "mean": 0.01,
                "hierarchical_ci95": [-0.005, 0.025],
                "n_seeds": 3,
            }
            if mname.startswith("vault_"):
                entry.update({"scene_clustered": True, "n_level2_clusters": 12, "n_episodes_per_seed": 64})
            if mname == "unseen_corruption_auroc":
                entry["paired_world_auroc_recomputed"] = True
            full_hier[f"isolated_vs_{comp}_{mname}"] = entry
    sg_means = {
        "isolated_scene_vs_shuffled_joint": 0.83 - 0.50,
        "isolated_scene_vs_shuffled_audio_only": 0.83 - 0.50,
        "isolated_scene_vs_color_prior_joint": 0.83 - 0.525,
        "isolated_scene_vs_color_prior_audio_only": 0.83 - 0.525,
        "isolated_paired_flip_dual_joint": 0.46,
        "isolated_paired_flip_dual_audio_only": 0.42,
    }
    for sg_k, sg_m in sg_means.items():
        sg_entry: dict[str, object] = {
            "mean": sg_m,
            "hierarchical_ci95": [sg_m - 0.06, sg_m + 0.06],
            "n_seeds": 3,
            "scene_clustered": True,
            "n_level2_clusters": 12,
            "n_episodes_per_seed": 64,
        }
        if sg_k in ("isolated_scene_vs_shuffled_joint", "isolated_scene_vs_shuffled_audio_only"):
            sg_entry["supercluster_donor_closed"] = True
        full_hier[sg_k] = sg_entry
    full_showcase = {
        "rendered": True,
        "preregistration_verified": True,
        "cohorts_included": list(COHORT_NAMES),
        "artifacts": list(REQUIRED_SHOWCASE_FIGURES),
    }

    full_groups = {
        c: (dict(iso_cohort) if c == "w8a_slwm_x2_isolated_450m" else dict(comp_cohort))
        for c in COHORT_NAMES
    }
    g4_ok = evaluate_gate4_checks(full_groups, full_hier, full_showcase)
    assert g4_ok["gate4_overall_pass"] is True
    assert g4_ok["scene_grounding_vs_prior_and_shuffle_pass"] is True
    assert g4_ok["non_inferiority_vs_spatial_only"]["unseen_mppi"]["hierarchical_ci95"] == [-0.005, 0.025]

    # Scene-blind color-prior shortcut (0% paired-flip dual or 0 lift over shuffled scenes) MUST fail Gate 4!
    shortcut_groups = {c: dict(full_groups[c]) for c in COHORT_NAMES}
    shortcut_groups["w8a_slwm_x2_isolated_450m"] = {
        **full_groups["w8a_slwm_x2_isolated_450m"],
        "vault_shuffled_scene_color_disc": _mk_stat(0.83),
    }
    shortcut_hier = {k: dict(v) for k, v in full_hier.items()}
    shortcut_hier["isolated_scene_vs_shuffled_joint"] = {
        "mean": 0.0,
        "hierarchical_ci95": [-0.02, 0.02],
        "n_seeds": 3,
        "scene_clustered": True,
        "n_level2_clusters": 12,
        "n_episodes_per_seed": 64,
    }
    assert evaluate_gate4_checks(shortcut_groups, shortcut_hier, full_showcase)["gate4_overall_pass"] is False

    # Forged / inconsistent hierarchical CI mean (0.05 != 0.83 - 0.82) MUST fail closed!
    forged_hier = {k: dict(v) for k, v in full_hier.items()}
    forged_hier["isolated_vs_w8a_ablation_spatial_control_only_450m_unseen_mppi"] = {
        "mean": 0.05,
        "hierarchical_ci95": [0.02, 0.08],
        "n_seeds": 3,
    }
    assert evaluate_gate4_checks(full_groups, forged_hier, full_showcase)["gate4_overall_pass"] is False

    # 0% counterfactual T160_paired_causal_switch MUST fail closed even if T160_goal_b_switch == 83% and diff == 0!
    zero_paired_groups = {c: dict(full_groups[c]) for c in COHORT_NAMES}
    zero_paired_hier = {k: dict(v) for k, v in full_hier.items()}
    for c in COHORT_NAMES:
        zero_paired_groups[c]["T160_paired_causal_switch"] = _mk_stat(0.0)
    for comp in REQUIRED_HIER_COMPS:
        zero_paired_hier[f"isolated_vs_{comp}_T160_paired_causal_switch"] = {
            "mean": 0.0,
            "hierarchical_ci95": [0.0, 0.0],
            "n_seeds": 3,
        }
    assert evaluate_gate4_checks(zero_paired_groups, zero_paired_hier, full_showcase)["gate4_overall_pass"] is False

    # 1. Missing spoken audio metric on ANY cohort (including comparison baseline!) MUST fail closed
    missing_comp_metric = {c: dict(full_groups[c]) for c in COHORT_NAMES}
    missing_comp_metric["w8a_sliding_window_w64_gqa_450m"] = {
        k: v for k, v in base_cohort.items() if k != "vault_spoken_audio_only_disc"
    }
    assert evaluate_gate4_checks(missing_comp_metric, full_hier, full_showcase)["gate4_overall_pass"] is False

    # 2. Missing KxT capacity cell MUST fail closed
    missing_k_cell = {c: dict(full_groups[c]) for c in COHORT_NAMES}
    bad_cap = {k: v for k, v in cap_cells.items() if k != "K16_T1024_overwritten_acc"}
    missing_k_cell["w8a_diagonal_ssm_450m"] = {**base_cohort, "multi_item_overwrite": bad_cap}
    assert evaluate_gate4_checks(missing_k_cell, full_hier, full_showcase)["gate4_overall_pass"] is False

    # 3. Missing hierarchical CI MUST fail closed
    bad_hier = {k: v for k, v in full_hier.items() if not k.endswith("_vault_spoken_audio_only_disc")}
    assert evaluate_gate4_checks(full_groups, bad_hier, full_showcase)["gate4_overall_pass"] is False

    # 4. Wrong seed IDs, incomplete seeds (< 3), or seeds_present=None MUST fail closed without raising TypeError
    wrong_seeds = {c: dict(full_groups[c]) for c in COHORT_NAMES}
    wrong_seeds["w8a_diagonal_ssm_450m"] = {**base_cohort, "seeds_present": [2001, 2002, 2999]}
    assert evaluate_gate4_checks(wrong_seeds, full_hier, full_showcase)["gate4_overall_pass"] is False

    none_seeds = {c: dict(full_groups[c]) for c in COHORT_NAMES}
    none_seeds["w8a_diagonal_ssm_450m"] = {**base_cohort, "seeds_present": None}
    assert evaluate_gate4_checks(none_seeds, full_hier, full_showcase)["gate4_overall_pass"] is False

    # 5. Unverified dataset provenance, non-uniform source SHA, non-donor-closed shuffle CI, unclustered vault CI, non-recomputed AUROC CI, reversed CI, or NaN values MUST fail closed
    unverified_prov = {c: dict(full_groups[c]) for c in COHORT_NAMES}
    unverified_prov["w8a_slwm_x2_isolated_450m"] = {**base_cohort, "provenance_verified": False}
    assert evaluate_gate4_checks(unverified_prov, full_hier, full_showcase)["gate4_overall_pass"] is False

    mixed_src_groups = {c: dict(full_groups[c]) for c in COHORT_NAMES}
    mixed_src_groups["w8a_diagonal_ssm_450m"] = {**base_cohort, "source_code_sha256s": ["b" * 64]}
    assert evaluate_gate4_checks(mixed_src_groups, full_hier, full_showcase)["gate4_overall_pass"] is False

    unclosed_dyad_hier = {k: dict(v) for k, v in full_hier.items()}
    unclosed_dyad_hier["isolated_scene_vs_shuffled_joint"] = {
        **unclosed_dyad_hier["isolated_scene_vs_shuffled_joint"],
        "supercluster_donor_closed": False,
    }
    assert evaluate_gate4_checks(full_groups, unclosed_dyad_hier, full_showcase)["gate4_overall_pass"] is False

    unclustered_vault_hier = {k: dict(v) for k, v in full_hier.items()}
    unclustered_vault_hier["isolated_vs_w8a_ablation_spatial_control_only_450m_vault_spoken_disc"] = {
        **unclustered_vault_hier["isolated_vs_w8a_ablation_spatial_control_only_450m_vault_spoken_disc"],
        "scene_clustered": False,
    }
    assert evaluate_gate4_checks(full_groups, unclustered_vault_hier, full_showcase)["gate4_overall_pass"] is False

    unpaired_auroc_hier = {k: dict(v) for k, v in full_hier.items()}
    unpaired_auroc_hier["isolated_vs_w8a_ablation_spatial_control_only_450m_unseen_corruption_auroc"] = {
        **unpaired_auroc_hier["isolated_vs_w8a_ablation_spatial_control_only_450m_unseen_corruption_auroc"],
        "paired_world_auroc_recomputed": False,
    }
    assert evaluate_gate4_checks(full_groups, unpaired_auroc_hier, full_showcase)["gate4_overall_pass"] is False

    reversed_ci_hier = {k: dict(v) for k, v in full_hier.items()}
    reversed_ci_hier["isolated_vs_w8a_sliding_window_w64_gqa_450m_unseen_mppi"] = {
        "mean": 0.01,
        "hierarchical_ci95": [0.03, -0.01],
        "n_seeds": 3,
    }
    assert evaluate_gate4_checks(full_groups, reversed_ci_hier, full_showcase)["gate4_overall_pass"] is False

    nan_vals_groups = {c: dict(full_groups[c]) for c in COHORT_NAMES}
    nan_vals_groups["w8a_diagonal_ssm_450m"] = {
        **base_cohort,
        "unseen_mppi": {"mean": 0.82, "std": 0.0, "values": [0.82, 0.82, float("nan")]},
    }
    assert evaluate_gate4_checks(nan_vals_groups, full_hier, full_showcase)["gate4_overall_pass"] is False

    # 6. Incomplete or unverified-preregistration showcase gallery MUST fail closed
    bad_sc = {**full_showcase, "artifacts": list(REQUIRED_SHOWCASE_FIGURES[:2])}
    assert evaluate_gate4_checks(full_groups, full_hier, bad_sc)["gate4_overall_pass"] is False
    unprereg_sc = {**full_showcase, "preregistration_verified": False}
    assert evaluate_gate4_checks(full_groups, full_hier, unprereg_sc)["gate4_overall_pass"] is False

    # 7. Empty bundle dict under strict=True MUST raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        select_median_seed_showcase_paths({}, tmp_path, strict=True)

    # 8. Median seed selector picks the true median seed (2003 has median score 0.72 among {0.60, 0.85, 0.72})
    fake_bundles = {}
    for c_name in COHORT_NAMES:
        for s_id, sc in ((2001, 0.60), (2002, 0.85), (2003, 0.72)):
            d = tmp_path / f"x1b-{c_name}-s{s_id}"
            d.mkdir(parents=True)
            (d / "showcase_bundle.npz").write_bytes(b"stub")
            fake_bundles[d.name] = {
                "dir": d,
                "receipt": {"seed": s_id},
                "eval": {"unseen_layout_deprivileged_mpc": {"geometry_free_mppi_success": sc}},
            }
    picked = select_median_seed_showcase_paths(fake_bundles, tmp_path, strict=True)
    assert picked["w8a_slwm_x2_isolated_450m"].parent.name.endswith("-s2003")

    # Deleting the true median seed's showcase_bundle.npz MUST raise FileNotFoundError (no silent fallback to s2001!)
    (tmp_path / "x1b-w8a_slwm_x2_isolated_450m-s2003/showcase_bundle.npz").unlink()
    with pytest.raises(FileNotFoundError):
        select_median_seed_showcase_paths(fake_bundles, tmp_path, strict=True)


def test_wave8a_xtiny_joint_convergence_showcase_and_hygiene(tmp_path: Path):
    """Verify multi-step x_tiny joint optimization decreases BOTH assoc_ce and vis_obj_mse, plus showcase & hygiene."""
    sp = make_x1b_spec("x_tiny", "slwm_x1b")
    params = init_x1b_params(sp, seed=77)
    ema_p = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
    m_st = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), params)
    v_st = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), params)
    b_fix = generate_in_domain_deprivileged_sim_batch_v8a(jax.random.key(78), 6, seq_len=64, vocab_size=sp.vocab_size)
    iv = Wave8AInterventions(zmem_isolation=True, enable_assoc_loss=True)

    @jax.jit
    def _opt_step(p, m, v, step_i):
        (_, mets), grads = jax.value_and_grad(
            lambda w: world_model_loss_wave8a(w, ema_p, sp, b_fix, iv, jax.random.key(79)),
            has_aux=True,
        )(p)
        pn, mn, vn, _, _ = memory_efficient_adamw_step(p, m, v, grads, step_i, 24, 2.0e-3)
        return pn, mn, vn, mets

    _, _, _, m_init = _opt_step(params, m_st, v_st, jnp.int32(0))
    p_cur, m_cur, v_cur = params, m_st, v_st
    m_last = m_init
    for si in range(20):
        p_cur, m_cur, v_cur, m_last = _opt_step(p_cur, m_cur, v_cur, jnp.int32(si))

    assert float(m_last["text_ce"]) < float(m_init["text_ce"])
    assert float(m_last["vis_obj_mse"]) < float(m_init["vis_obj_mse"])

    rec = train_and_evaluate_wave8a(
        tag="x1b-w8a_slwm_x2_isolated_450m-s2001",
        cohort="w8a_slwm_x2_isolated_450m",
        profile="x_tiny",
        arch="slwm_x1b",
        regularizer="epps_pulley_sigreg",
        key_shield=True,
        zerolag_bypass=True,
        decoupled_heads=True,
        zmem_isolation=True,
        enable_assoc_loss=True,
        seed=2001,
        steps=4,
        batch_size=4,
        seq_len=32,
        out_dir=tmp_path / "runs_wave8a",
        export_showcase=True,
        smoke=True,
    )
    assert rec["showcase_bundle_bytes"] > 0
    assert rec["eval_summary"]["oracle_mppi_ceiling"] is not None
    assert rec["eval_summary"]["compiled_inference_ms_per_tick"] > 0.0
    sc_npz = tmp_path / "runs_wave8a/x1b-w8a_slwm_x2_isolated_450m-s2001/showcase_bundle.npz"
    assert sc_npz.is_file()

    gallery_dir = tmp_path / "inference_showcase/wave8a/dev"
    manifest = render_wave8a_showcase_gallery({"w8a_slwm_x2_isolated_450m": sc_npz}, gallery_dir)
    assert manifest["rendered"] is True
    assert list(PRE_REGISTERED_COMMON_EPISODES) == [0, 1, 2, 3]
    assert manifest["selection_protocol"]["common_cross_model_episode_ids"] == [0, 1, 2, 3]
    assert "ground_truth_eval" in manifest["epistemic_channel_definitions"]
    assert "abstained" in manifest["epistemic_channel_definitions"]
    assert "fallback_mitigated" in manifest["epistemic_channel_definitions"]
    assert "w8a_slwm_x2_isolated_450m" in manifest["ood_post_onset_routing_counts"]
    assert (
        manifest["ood_post_onset_routing_counts"]["w8a_slwm_x2_isolated_450m"]["gate_invoked"]
        == "route_proposals_with_commit_gate"
    )
    assert set(manifest["selection_protocol"]["selected_percentile_episodes"]) == {"P05", "P25", "P50", "P75", "P95"}
    assert (gallery_dir / "wave8a_p05_to_p95_unseen_mpc_gallery.png").is_file()
    assert (gallery_dir / "wave8a_cross_cohort_t160_and_k16_showcase.png").is_file()
    assert (gallery_dir / "wave8a_common_episodes_cross_cohort_gallery.png").is_file()

    banned_ops_terms = ("dual-hdd", "192.168.", "compute unit", "credit balance", "/mnt/slwm-checkpoints")
    brief_txt = (
        REPO_ROOT / "experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111/wave8a_brief.md"
    ).read_text().lower()
    man_txt = json.dumps(manifest).lower()
    for term in banned_ops_terms:
        assert term not in brief_txt, f"Banned ops telemetry {term!r} found in wave8a_brief.md"
        assert term not in man_txt, f"Banned ops telemetry {term!r} found in showcase_manifest.json"


def test_round4_passive_av_masking_role_bound_sha_and_scene_color_grounding(tmp_path: Path):
    """Regression suite for Codex Round 4 Findings 1, 3, & 4 (passive-AV 0-proprio, role-bound SHA, scene block xy)."""
    from evals.x1b_eval_wave8a import eval_grounded_spoken_robot_v8a
    from models.x1b_wave8a_core import decode_target_block_xy_wave8a, encode_pure_visual_grid_8x8
    from scripts.data.x1b_prepare_wave8a_grounded_fedora import extract_table_block_centroids_64
    from scripts.eval.x1b_analyze_wave8a import verify_receipt_data_provenance
    from scripts.train.x1b_train import sha256_file
    from scripts.train.x1b_train_wave8a import verify_wave8a_dataset_roles_fail_closed

    # 1. Role-bound filename-independent dataset verification (Finding 4)
    f_real = tmp_path / "renamed_real_bank.npz"
    f_grd = tmp_path / "custom_grounded.npz"
    f_vlt = tmp_path / "custom_vault.npz"
    f_real.write_bytes(b"real-v2-bytes")
    f_grd.write_bytes(b"grounded-v2-bytes")
    f_vlt.write_bytes(b"vault-v2-bytes")
    cards_p = tmp_path / "cards.json"
    cards_dict = {
        "x1b_real_bank_v2": {"sha256": sha256_file(f_real)},
        "x1b_wave8a_grounded_v2": {"sha256": sha256_file(f_grd)},
        "x1b_wave8a_vault_v2": {"sha256": sha256_file(f_vlt)},
    }
    cards_p.write_text(json.dumps(cards_dict))

    # Passing None on non-smoke MUST raise RuntimeError
    with pytest.raises(RuntimeError):
        verify_wave8a_dataset_roles_fail_closed(None, f_grd, f_vlt, smoke=False, cards_cfg_path=cards_p)

    # Passing a renamed file with mismatched SHA MUST raise RuntimeError (no filename bypass!)
    f_wrong = tmp_path / "renamed_bank.npz"
    f_wrong.write_bytes(b"tampered-bytes")
    with pytest.raises(RuntimeError):
        verify_wave8a_dataset_roles_fail_closed(f_wrong, f_grd, f_vlt, smoke=False, cards_cfg_path=cards_p)

    # Matching bytes succeed even under custom filenames, and verify_receipt_data_provenance requires smoke=False & registered_cards_verified=True
    prov_ok = verify_wave8a_dataset_roles_fail_closed(f_real, f_grd, f_vlt, smoke=False, cards_cfg_path=cards_p)
    assert prov_ok["registered_cards_verified"] is True
    assert verify_receipt_data_provenance({"data_provenance": prov_ok}, reg_cards=cards_dict) is False  # Missing smoke MUST fail!
    assert verify_receipt_data_provenance({"smoke": True, "data_provenance": prov_ok}, reg_cards=cards_dict) is False
    assert verify_receipt_data_provenance({"smoke": False, "data_provenance": prov_ok}, reg_cards=cards_dict) is True
    assert (
        verify_receipt_data_provenance(
            {"smoke": False, "data_provenance": {**prov_ok, "registered_cards_verified": False}}, reg_cards=cards_dict
        )
        is False
    )

    # 2. Exact total_loss & gradient invariance under modality_mask & has_av_pair (Codex Round 4 & Round 5)
    sp = make_x1b_spec("x_tiny", "slwm_x1b", dtype="float32", grad_checkpoint=False, vocab_size=512)
    p = init_x1b_params(sp, seed=301)
    iv = Wave8AInterventions(zmem_isolation=True)
    b_av = generate_synthetic_world_batch(jax.random.key(302), 4, seq_len=12, vocab_size=512)
    mm_av = jnp.broadcast_to(jnp.array([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32), (4, 12, 4))
    b_av_clean = {**b_av, "scenario": jnp.full((4,), -1, dtype=jnp.int32), "modality_mask": mm_av, "full_modality_mask": mm_av}
    b_av_mut_prop = {
        **b_av_clean,
        "proprio": jnp.full_like(b_av["proprio"], 42.0),
        "target_proprio": jnp.full_like(b_av["target_proprio"], -19.0),
        "target_next_proprio": jnp.full_like(b_av["target_next_proprio"], 33.0),
        "target_contacts": jnp.ones_like(b_av["target_contacts"]),
        "target_action": jnp.full_like(b_av["target_action"], 0.9),
        "target_value": jnp.full_like(b_av["target_value"], -12.0),
        "text": jnp.full_like(b_av["text"], 77),
        "target_text": jnp.full_like(b_av["target_text"], 80),
    }
    (l_av1, m_av1), g_av1 = jax.value_and_grad(
        lambda w: world_model_loss_wave8a(w, p, sp, b_av_clean, iv, jax.random.key(303)), has_aux=True
    )(p)
    (l_av2, _), g_av2 = jax.value_and_grad(
        lambda w: world_model_loss_wave8a(w, p, sp, b_av_mut_prop, iv, jax.random.key(303)), has_aux=True
    )(p)
    assert float(jnp.abs(l_av1 - l_av2)) == 0.0
    assert float(m_av1["vis_ee_mse"]) == 0.0 and float(m_av1["state_mse"]) == 0.0
    max_gdiff_av = max(float(jnp.max(jnp.abs(g_av1[k] - g_av2[k]))) for k in g_av1)
    assert max_gdiff_av == 0.0

    # Text-only `[0, 0, 0, 1]` MUST be 100% invariant in BOTH total_loss AND gradients when unavailable video/audio/proprio change!
    mm_txt = jnp.broadcast_to(jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32), (4, 12, 4))
    b_tx1 = {
        **b_av,
        "scenario": jnp.full((4,), -1, dtype=jnp.int32),
        "modality_mask": mm_txt,
        "full_modality_mask": mm_txt,
        "has_av_pair": jnp.zeros((4,)),
    }
    b_tx2 = {
        **b_tx1,
        "video": jnp.full_like(b_av["video"], 200),
        "clean_video": jnp.full_like(b_av["video"], 255),
        "audio": jnp.full_like(b_av["audio"], 7.5),
        "proprio": jnp.full_like(b_av["proprio"], -11.0),
        "target_proprio": jnp.full_like(b_av["target_proprio"], 9.0),
        "target_next_proprio": jnp.full_like(b_av["target_next_proprio"], -9.0),
    }
    (l_tx1, m_tx1), g_tx1 = jax.value_and_grad(
        lambda w: world_model_loss_wave8a(w, p, sp, b_tx1, iv, jax.random.key(304)), has_aux=True
    )(p)
    (l_tx2, m_tx2), g_tx2 = jax.value_and_grad(
        lambda w: world_model_loss_wave8a(w, p, sp, b_tx2, iv, jax.random.key(304)), has_aux=True
    )(p)
    assert float(jnp.abs(l_tx1 - l_tx2)) == 0.0
    assert float(jnp.abs(m_tx1["sigreg"] - m_tx2["sigreg"])) == 0.0
    max_gdiff_tx = max(float(jnp.max(jnp.abs(g_tx1[k] - g_tx2[k]))) for k in g_tx1)
    assert max_gdiff_tx == 0.0

    # Mixed batch with has_av_pair = [1, 1, 0, 0]: unpaired items (2..3) must never enter AV-NCE denominators!
    mm_mix = jnp.concatenate([mm_av[:2], mm_txt[2:]], axis=0)
    b_mix1 = {
        **b_av,
        "scenario": jnp.full((4,), -1, dtype=jnp.int32),
        "modality_mask": mm_mix,
        "full_modality_mask": mm_mix,
        "has_av_pair": jnp.array([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
    }
    b_mix2 = {
        **b_mix1,
        "video": b_mix1["video"].at[2:].set(240),
        "clean_video": b_mix1["clean_video"].at[2:].set(240),
        "audio": b_mix1["audio"].at[2:].set(-6.0),
    }
    (l_mx1, _), g_mx1 = jax.value_and_grad(
        lambda w: world_model_loss_wave8a(w, p, sp, b_mix1, iv, jax.random.key(309)), has_aux=True
    )(p)
    (l_mx2, _), g_mx2 = jax.value_and_grad(
        lambda w: world_model_loss_wave8a(w, p, sp, b_mix2, iv, jax.random.key(309)), has_aux=True
    )(p)
    assert float(jnp.abs(l_mx1 - l_mx2)) == 0.0
    assert float(jnp.max(jnp.abs(g_mx1["align_vis_w"] - g_mx2["align_vis_w"]))) == 0.0

    # 2b. Verify z_vis_masked in decode_world_heads_wave8a AND rob_modality_mask in auxiliary rob_* branch
    from models.x1b_wave8a_core import SLOTS_VISUAL, decode_world_heads_wave8a

    z_mem_d = jax.random.normal(jax.random.key(320), (2, 4, 24, sp.d_model))
    z_tsk_a = jax.random.normal(jax.random.key(321), (2, 4, 24, sp.d_model))
    z_tsk_b = z_tsk_a.at[:, :, SLOTS_VISUAL, :].set(jax.random.normal(jax.random.key(322), (2, 4, 16, sp.d_model)) * 50.0)
    z_nxt_a = jax.random.normal(jax.random.key(323), (2, 4, 24, sp.d_model))
    z_nxt_b = z_nxt_a.at[:, :, SLOTS_VISUAL, :].set(jax.random.normal(jax.random.key(324), (2, 4, 16, sp.d_model)) * 50.0)
    lv_d = jnp.zeros((2, 4, 24, 1), dtype=jnp.float32)
    g_d = jnp.zeros((2, 4, 4), dtype=jnp.float32)
    mm_no_vis = jnp.broadcast_to(jnp.array([[[1.0, 0.0, 1.0, 1.0]]], dtype=jnp.float32), (2, 4, 4))
    hd_a = decode_world_heads_wave8a(
        p, sp, z_mem_d, z_tsk_a, z_nxt_a, lv_d, g_d, iv,
        obs_video=jnp.ones((2, 4, 64, 64, 3), dtype=jnp.uint8) * 10,
        obs_modality_mask=mm_no_vis,
    )
    hd_b = decode_world_heads_wave8a(
        p, sp, z_mem_d, z_tsk_b, z_nxt_b, lv_d, g_d, iv,
        obs_video=jnp.ones((2, 4, 64, 64, 3), dtype=jnp.uint8) * 240,
        obs_modality_mask=mm_no_vis,
    )
    for hk in ("vis_ee_curr", "vis_obj_curr", "hazard_logits_8x8", "pred_target_block_xy", "diag_video", "commit_logits"):
        assert float(jnp.max(jnp.abs(hd_a[hk] - hd_b[hk]))) == 0.0

    b_rob_m1 = {
        **b_tx1,
        "rob_vid0": jnp.ones((4, 1, 64, 64, 3), dtype=jnp.uint8) * 15,
        "rob_prop0": jnp.ones((4, 1, sp.proprio_dim), dtype=jnp.float32) * 0.2,
        "rob_audio_32": jnp.ones((4, 32, 64), dtype=jnp.float32) * 2.0,
        "rob_wrong_audio_32": jnp.ones((4, 32, 64), dtype=jnp.float32) * -2.0,
        "rob_text_32": jnp.ones((4, 32, 4), dtype=jnp.int32) * 7,
        "rob_wrong_text_32": jnp.ones((4, 32, 4), dtype=jnp.int32) * 9,
        "rob_end_xy": jnp.ones((4, 2), dtype=jnp.float32) * 0.4,
        "rob_modality_mask": jnp.zeros((4, 32, 4), dtype=jnp.float32),
    }
    b_rob_m2 = {
        **b_rob_m1,
        "rob_vid0": jnp.ones((4, 1, 64, 64, 3), dtype=jnp.uint8) * 250,
        "rob_audio_32": jnp.ones((4, 32, 64), dtype=jnp.float32) * -99.0,
    }
    (l_rm1, _), _ = jax.value_and_grad(
        lambda w: world_model_loss_wave8a(w, p, sp, b_rob_m1, iv, jax.random.key(325)), has_aux=True
    )(p)
    (l_rm2, _), _ = jax.value_and_grad(
        lambda w: world_model_loss_wave8a(w, p, sp, b_rob_m2, iv, jax.random.key(325)), has_aux=True
    )(p)
    assert float(jnp.abs(l_rm1 - l_rm2)) == 0.0

    # 3. Unbound / mismatched checkpoint, done.json, & lane verified.json reuse MUST fail closed (Codex Round 6)
    from scripts.train.x1b_run_wave8a_lane import is_verified_wave8a_local_job
    from scripts.train.x1b_train import save_bf16_weights
    from scripts.train.x1b_train_wave8a import WAVE8A_BOUND_SOURCE_FILES, compute_wave8a_source_code_sha256

    assert len(WAVE8A_BOUND_SOURCE_FILES) >= 23
    for req_src in ("models/x1b_jepa.py", "models/x1b_sigreg_epps_pulley.py", "data/x1b_multimodal.py", "scripts/train/x1b_train.py"):
        assert req_src in WAVE8A_BOUND_SOURCE_FILES

    ckpt_out = tmp_path / "ckpt_guard_runs"
    stray_dir = ckpt_out / "x1b-stray-s2001"
    stray_dir.mkdir(parents=True)
    save_bf16_weights(stray_dir / "weights_bf16.npz", p)
    with pytest.raises(RuntimeError, match="allow_verified_resume=False"):
        train_and_evaluate_wave8a(
            tag="x1b-stray-s2001",
            cohort="w8a_slwm_x2_isolated_450m",
            profile="x_tiny",
            arch="slwm_x1b",
            regularizer="epps_pulley_sigreg",
            key_shield=True,
            zerolag_bypass=True,
            decoupled_heads=True,
            zmem_isolation=True,
            enable_assoc_loss=True,
            seed=2001,
            steps=2,
            batch_size=4,
            seq_len=32,
            out_dir=ckpt_out,
            allow_verified_resume=False,
            smoke=True,
        )
    with pytest.raises(RuntimeError, match="missing .*checkpoint_provenance"):
        train_and_evaluate_wave8a(
            tag="x1b-stray-s2001",
            cohort="w8a_slwm_x2_isolated_450m",
            profile="x_tiny",
            arch="slwm_x1b",
            regularizer="epps_pulley_sigreg",
            key_shield=True,
            zerolag_bypass=True,
            decoupled_heads=True,
            zmem_isolation=True,
            enable_assoc_loss=True,
            seed=2001,
            steps=2,
            batch_size=4,
            seq_len=32,
            out_dir=ckpt_out,
            allow_verified_resume=True,
            smoke=True,
        )

    # Complete a genuine smoke run with save_bf16=True, then verify fail-closed done.json & artifact byte checks
    done_rec = train_and_evaluate_wave8a(
        tag="x1b-bound-s2001",
        cohort="w8a_slwm_x2_isolated_450m",
        profile="x_tiny",
        arch="slwm_x1b",
        regularizer="epps_pulley_sigreg",
        key_shield=True,
        zerolag_bypass=True,
        decoupled_heads=True,
        zmem_isolation=True,
        enable_assoc_loss=True,
        seed=2001,
        steps=2,
        batch_size=4,
        seq_len=32,
        out_dir=ckpt_out,
        save_bf16=True,
        smoke=True,
    )
    assert done_rec["smoke"] is True
    # Default allow_verified_resume=False MUST refuse pre-existing done.json too!
    with pytest.raises(RuntimeError, match="allow_verified_resume=False"):
        train_and_evaluate_wave8a(
            tag="x1b-bound-s2001",
            cohort="w8a_slwm_x2_isolated_450m",
            profile="x_tiny",
            arch="slwm_x1b",
            regularizer="epps_pulley_sigreg",
            key_shield=True,
            zerolag_bypass=True,
            decoupled_heads=True,
            zmem_isolation=True,
            enable_assoc_loss=True,
            seed=2001,
            steps=2,
            batch_size=4,
            seq_len=32,
            out_dir=ckpt_out,
            save_bf16=True,
            allow_verified_resume=False,
            smoke=True,
        )
    # Exact verified resume succeeds when all artifact SHAs match
    reused_rec = train_and_evaluate_wave8a(
        tag="x1b-bound-s2001",
        cohort="w8a_slwm_x2_isolated_450m",
        profile="x_tiny",
        arch="slwm_x1b",
        regularizer="epps_pulley_sigreg",
        key_shield=True,
        zerolag_bypass=True,
        decoupled_heads=True,
        zmem_isolation=True,
        enable_assoc_loss=True,
        seed=2001,
        steps=2,
        batch_size=4,
        seq_len=32,
        out_dir=ckpt_out,
        save_bf16=True,
        allow_verified_resume=True,
        smoke=True,
    )
    assert reused_rec["portable_int8_sha256"] == done_rec["portable_int8_sha256"]
    # Requesting export_showcase=True against an export_showcase=False done.json MUST fail closed!
    with pytest.raises(RuntimeError, match="done.json"):
        train_and_evaluate_wave8a(
            tag="x1b-bound-s2001",
            cohort="w8a_slwm_x2_isolated_450m",
            profile="x_tiny",
            arch="slwm_x1b",
            regularizer="epps_pulley_sigreg",
            key_shield=True,
            zerolag_bypass=True,
            decoupled_heads=True,
            zmem_isolation=True,
            enable_assoc_loss=True,
            seed=2001,
            steps=2,
            batch_size=4,
            seq_len=32,
            out_dir=ckpt_out,
            save_bf16=True,
            export_showcase=True,
            allow_verified_resume=True,
            smoke=True,
        )
    # Tampering with portable_int8.npz bytes on disk MUST fail closed even if done.json is untouched!
    orig_int8_bytes = (ckpt_out / "x1b-bound-s2001/portable_int8.npz").read_bytes()
    (ckpt_out / "x1b-bound-s2001/portable_int8.npz").write_bytes(orig_int8_bytes + b"tamper")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        train_and_evaluate_wave8a(
            tag="x1b-bound-s2001",
            cohort="w8a_slwm_x2_isolated_450m",
            profile="x_tiny",
            arch="slwm_x1b",
            regularizer="epps_pulley_sigreg",
            key_shield=True,
            zerolag_bypass=True,
            decoupled_heads=True,
            zmem_isolation=True,
            enable_assoc_loss=True,
            seed=2001,
            steps=2,
            batch_size=4,
            seq_len=32,
            out_dir=ckpt_out,
            save_bf16=True,
            allow_verified_resume=True,
            smoke=True,
        )
    (ckpt_out / "x1b-bound-s2001/portable_int8.npz").write_bytes(orig_int8_bytes)
    with pytest.raises(RuntimeError, match="done.json"):
        train_and_evaluate_wave8a(
            tag="x1b-bound-s2001",
            cohort="w8a_slwm_x2_isolated_450m",
            profile="x_tiny",
            arch="slwm_x1b",
            regularizer="epps_pulley_sigreg",
            key_shield=True,
            zerolag_bypass=True,
            decoupled_heads=True,
            zmem_isolation=False,  # Changed intervention!
            enable_assoc_loss=True,
            seed=2001,
            steps=2,
            batch_size=4,
            seq_len=32,
            out_dir=ckpt_out,
            save_bf16=True,
            allow_verified_resume=True,
            smoke=True,
        )
    # Bare verified.json in lane runner MUST raise RuntimeError (never bypass provenance checks!)
    bare_lane_dir = ckpt_out / "x1b-bare-s2001"
    bare_lane_dir.mkdir(parents=True)
    (bare_lane_dir / "verified.json").write_text('{"tag": "x1b-bare-s2001", "verified_copies": 2, "sha256": {}}')
    with pytest.raises(RuntimeError):
        is_verified_wave8a_local_job(
            {
                "tag": "x1b-bare-s2001",
                "cohort": "w8a_slwm_x2_isolated_450m",
                "profile": "x450m",
                "arch": "slwm_x1b",
                "regularizer": "epps_pulley_sigreg",
                "key_shield": True,
                "zerolag_bypass": True,
                "decoupled_heads": True,
                "zmem_isolation": True,
                "enable_assoc_loss": True,
                "seed": 2001,
                "steps": 1800,
                "batch_size": 8,
                "seq_len": 64,
                "save_bf16": False,
                "export_showcase": True,
            },
            runs_subdir="runs_wave8a",
            source_code_sha=compute_wave8a_source_code_sha256(),
            reg_cards={},
            runs_base_dir=ckpt_out,
        )
    (ckpt_out / "x1b-bound-s2001/done.json").unlink()
    with pytest.raises(RuntimeError, match="Checkpoint provenance mismatch"):
        train_and_evaluate_wave8a(
            tag="x1b-bound-s2001",
            cohort="w8a_slwm_x2_isolated_450m",
            profile="x_tiny",
            arch="slwm_x1b",
            regularizer="epps_pulley_sigreg",
            key_shield=True,
            zerolag_bypass=True,
            decoupled_heads=True,
            zmem_isolation=False,  # Mismatched against checkpoint_provenance.json!
            enable_assoc_loss=True,
            seed=2001,
            steps=2,
            batch_size=4,
            seq_len=32,
            out_dir=ckpt_out,
            save_bf16=True,
            allow_verified_resume=True,
            smoke=True,
        )

    # 4. Symmetric 2-block grounding, 180-deg flip, shuffled-scene, & mu_train color prior (Codex Round 5 Finding 1)
    frame = np.full((64, 64, 3), 150, dtype=np.uint8)
    frame[10:18, 12:20] = np.array([220, 35, 35], dtype=np.uint8)  # Red block top-left
    frame[44:52, 42:50] = np.array([35, 75, 225], dtype=np.uint8)  # Blue block bottom-right
    c_map = extract_table_block_centroids_64(frame)
    assert 0 in c_map and 1 in c_map
    assert c_map[0][0] < -0.35 and c_map[0][1] < -0.35
    assert c_map[1][0] > 0.25 and c_map[1][1] > 0.25

    # Verify decode_target_block_xy_wave8a does NOT clamp blank video to [0, 0], yet symmetric 2-block blind_video == 0.50
    z_dummy = jax.random.normal(jax.random.key(305), (2, 1, 24, sp.d_model), dtype=jnp.float32)
    blk_unclamped = decode_target_block_xy_wave8a(p, z_dummy, iv, pure_v_8x8=None)
    assert float(jnp.max(jnp.abs(blk_unclamped))) > 0.0

    rob_npz = tmp_path / "mini_vault_v2.npz"
    frame_b = np.flip(frame, axis=(0, 1)).copy()
    vids = np.stack([frame, frame, frame_b, frame_b], axis=0)[:, None, :, :, :]
    vids = np.broadcast_to(vids, (4, 32, 64, 64, 3)).copy()
    auds = np.zeros((4, 32, 64), dtype=np.float32)
    auds[:, :16, 0] = 1.0
    wcol_auds = auds.copy()
    wcol_auds[:, :16, 0] = -1.0  # Identical t=16..31 direction audio, swapped t=0..15 color audio
    ee = np.zeros((4, 32, 2), dtype=np.float32)
    ee[:, :, 0] = np.linspace(0.0, 0.3, 32)[None, :] * np.array([1.0, -1.0, 1.0, -1.0])[:, None]
    u_xy = np.ones((4, 32, 2), dtype=np.float32) * 0.1
    t_txt = np.array([[10, 20] + [0] * 14, [10, 21] + [0] * 14, [11, 20] + [0] * 14, [11, 21] + [0] * 14], dtype=np.int32)
    cf_idx = np.array([1, 0, 3, 2], dtype=np.int64)
    tb_arr = np.array([c_map[0], c_map[0], -np.array(c_map[0]), -np.array(c_map[0])], dtype=np.float32)
    db_arr = np.array([c_map[1], c_map[1], -np.array(c_map[1]), -np.array(c_map[1])], dtype=np.float32)
    np.savez(
        rob_npz,
        vault_video=vids,
        vault_spoken_audio=auds,
        vault_wrong_spoken_audio=auds[cf_idx],
        vault_wrong_color_spoken_audio=wcol_auds,
        vault_ee_xy=ee,
        vault_action_xy=u_xy,
        vault_length=np.full((4,), 16, dtype=np.int32),
        vault_true_text_ids=t_txt,
        vault_wrong_text_ids=t_txt[cf_idx],
        vault_wrong_color_text_ids=t_txt[:, ::-1],
        vault_cf_index=cf_idx,
        vault_scene_cluster_id=np.array([100, 101, 200, 201], dtype=np.int64),
        vault_dir_id=np.array([0, 1, 0, 1], dtype=np.int32),
        vault_color_id=np.array([0, 0, 1, 1], dtype=np.int32),
        vault_distractor_color_id=np.array([1, 1, 0, 0], dtype=np.int32),
        vault_target_block_xy=tb_arr,
        vault_distractor_block_xy=db_arr,
    )
    from evals.x1b_eval_wave8a import _load_train_color_prior_means

    mu_fb = _load_train_color_prior_means(None, tmp_path / "does_not_exist.npz")
    assert float(np.max(np.abs(mu_fb))) > 0.01  # Fallback reads rows_by_split["train"] from wave8a_dataset_provenance.json
    rob_ev = eval_grounded_spoken_robot_v8a(p, sp, iv, rob_npz, split="vault")
    assert rob_ev["available"] is True
    assert rob_ev["num_shuffled_dyad_clusters"] == 2
    assert len(rob_ev["shuffled_donor_indices"]) == 4
    assert "target_block_rmse_mm" not in rob_ev
    assert abs(rob_ev["target_block_px64_err"] - rob_ev["target_block_norm_rmse"] * 32.0) < 1e-5
    assert rob_ev["color_prior_paired_flip_dual_acc"] == 0.0
    assert "shuffled_scene_color_disc_acc" in rob_ev
    assert "paired_flip_dual_scene_color_disc_acc" in rob_ev
    assert "color_prior_scene_color_disc_acc" in rob_ev
    readme_txt = (REPO_ROOT / "README.md").read_text().lower()
    assert "compute unit" not in readme_txt and "colab unit" not in readme_txt and "dual-hdd" not in readme_txt

    # 5. Codex Round 7 analyzer bundle provenance & fail-closed dyad closure regressions
    from dataclasses import asdict as _dc_asdict
    from scripts.eval.x1b_analyze_wave8a import (
        _verified_vault_shuffled_dyad_ids,
        aggregate_w8a_cohort,
        verify_wave8a_bundle_provenance,
    )
    from scripts.train.x1b_train_wave8a import compute_wave8a_binding_sha256

    study_cfg_8a = json.loads((REPO_ROOT / "configs/x1b/study_x2_wave8a.json").read_text())
    src_sha_now = compute_wave8a_source_code_sha256()
    b_tag = "x1b-w8a_ablation_no_zmem_iso_450m-s2001"
    b_dir = tmp_path / "analyzer_prov" / b_tag
    b_dir.mkdir(parents=True)
    (b_dir / "config.json").write_text('{"cfg": 1}\n')
    (b_dir / "train.jsonl").write_text('{"step": 1800}\n')
    (b_dir / "showcase_bundle.npz").write_bytes(b"showcase-bytes")
    fake_eval_doc = {
        "streaming_state_memory": {"total_state_mib": 8.7898, "strict_system_state_mib": 8.8533},
        "compiled_inference_ms_per_tick": 1.2,
        "unseen_layout_deprivileged_mpc": {
            "geometry_free_mppi_success": 0.80,
            "no_jepa_subgoal_success": 0.75,
            "explicit_latch_reactive_success": 0.40,
            "open_loop_policy_success": 0.20,
            "vis_decoded_obj_rmse": 0.03,
            "vis_decoded_ee_rmse": 0.01,
            "by_unseen_family": {
                "off_center_wall": {"mppi_success": 0.8},
                "dual_chicane": {"mppi_success": 0.8},
                "multi_block_clutter": {"mppi_success": 0.8},
                "dynamics_shift": {"mppi_success": 0.8},
            },
        },
        "beyond_window_T160_distractor_switch": {
            "T160_post_perturbation_goal_b_success": 0.80,
            "T160_paired_causal_switch_rate": 0.75,
            "T160_stale_goal_a_capture": 0.05,
            "latch_mse_inside_W64_t24_64": 0.01,
            "latch_mse_beyond_W64_t92_160": 0.01,
        },
        "symbolic_key_drift_and_recall": {
            "symbolic_key_visual_drift": 0.0,
            "dyn_visual_assoc_T64": 0.99,
            "dyn_visual_assoc_T256": 0.99,
            "dyn_visual_assoc_T512": 0.99,
            "dyn_visual_assoc_T1024": 0.99,
        },
        "unseen_ood_sensor_corruptions": {"mean_unseen_ood_corruption_auroc": 0.95},
        "grounded_spoken_robot_dev": {},
        "untouched_vault_spoken_robot": rob_ev,
        "multi_item_overwrite_capacity": {},
    }
    (b_dir / "eval_metrics.json").write_text(json.dumps(fake_eval_doc, sort_keys=True) + "\n")
    iv_no_iso = Wave8AInterventions(zmem_isolation=False)
    valid_bind = compute_wave8a_binding_sha256(
        tag=b_tag,
        cohort="w8a_ablation_no_zmem_iso_450m",
        profile="x450m",
        arch="slwm_x1b",
        interventions=iv_no_iso,
        seed=2001,
        steps=1800,
        batch_size=8,
        seq_len=64,
        smoke=False,
        save_bf16=False,
        export_showcase=True,
    )
    int8_stub_sha = "c" * 64
    valid_rec = {
        "tag": b_tag,
        "cohort": "w8a_ablation_no_zmem_iso_450m",
        "profile": "x450m",
        "arch": "slwm_x1b",
        "interventions": _dc_asdict(iv_no_iso),
        "seed": 2001,
        "steps": 1800,
        "batch_size": 8,
        "seq_len": 64,
        "smoke": False,
        "save_bf16": False,
        "export_showcase": True,
        "parameter_counts": {"strict_total": 453_199_938},
        "binding_sha256": valid_bind,
        "source_code_sha256": src_sha_now,
        "data_provenance": {**prov_ok, "sequence_tensor_budget_ticks": 1800 * 8 * 64},
        "config_file_sha256": sha256_file(b_dir / "config.json"),
        "eval_metrics_sha256": sha256_file(b_dir / "eval_metrics.json"),
        "train_log_sha256": sha256_file(b_dir / "train.jsonl"),
        "showcase_bundle_sha256": sha256_file(b_dir / "showcase_bundle.npz"),
        "portable_int8_sha256": int8_stub_sha,
    }
    (b_dir / "receipt.json").write_text(json.dumps(valid_rec, sort_keys=True) + "\n")
    (b_dir / "verified.json").write_text(
        json.dumps(
            {
                "tag": b_tag,
                "verified_copies": 2,
                "sha256": {
                    "receipt.json": sha256_file(b_dir / "receipt.json"),
                    "eval_metrics.json": valid_rec["eval_metrics_sha256"],
                    "config.json": valid_rec["config_file_sha256"],
                    "train.jsonl": valid_rec["train_log_sha256"],
                    "showcase_bundle.npz": valid_rec["showcase_bundle_sha256"],
                    "portable_int8.npz": int8_stub_sha,
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    good_bundle = {"dir": b_dir, "receipt": valid_rec, "eval": fake_eval_doc}
    assert (
        verify_wave8a_bundle_provenance(
            b_tag,
            good_bundle,
            expected_cohort="w8a_ablation_no_zmem_iso_450m",
            study_cfg=study_cfg_8a,
            reg_cards=cards_dict,
            expected_source_sha=src_sha_now,
        )
        is True
    )
    agg_good = aggregate_w8a_cohort(
        {b_tag: good_bundle},
        "w8a_ablation_no_zmem_iso_450m",
        study_cfg=study_cfg_8a,
        reg_cards=cards_dict,
        expected_source_sha=src_sha_now,
    )
    assert agg_good["provenance_verified"] is True

    # Codex Round 7 Item 1 counterexample: profile="x_tiny", steps=2, missing smoke, or bogus binding/source SHA MUST fail!
    for bad_patch in (
        {"profile": "x_tiny"},
        {"steps": 2},
        {"smoke": None},
        {"binding_sha256": "0" * 64},
        {"source_code_sha256": "f" * 64},
    ):
        bad_rec = {k: v for k, v in {**valid_rec, **bad_patch}.items() if v is not None}
        (b_dir / "receipt.json").write_text(json.dumps(bad_rec, sort_keys=True) + "\n")
        agg_bad = aggregate_w8a_cohort(
            {b_tag: {"dir": b_dir, "receipt": bad_rec, "eval": fake_eval_doc}},
            "w8a_ablation_no_zmem_iso_450m",
            study_cfg=study_cfg_8a,
            reg_cards=cards_dict,
            expected_source_sha=src_sha_now,
        )
        assert agg_bad["provenance_verified"] is False

    # Restore valid receipt, then tamper with eval_metrics.json bytes on disk -> provenance_verified MUST be False!
    (b_dir / "receipt.json").write_text(json.dumps(valid_rec, sort_keys=True) + "\n")
    (b_dir / "eval_metrics.json").write_text(json.dumps({**fake_eval_doc, "compiled_inference_ms_per_tick": 9.9}) + "\n")
    assert (
        verify_wave8a_bundle_provenance(
            b_tag,
            good_bundle,
            expected_cohort="w8a_ablation_no_zmem_iso_450m",
            study_cfg=study_cfg_8a,
            reg_cards=cards_dict,
            expected_source_sha=src_sha_now,
        )
        is False
    )

    # Codex Round 7 Item 3 counterexamples: _verified_vault_shuffled_dyad_ids MUST never silently fall back to scene_cids!
    scene_cids_4 = rob_ev["scene_cluster_ids"]
    dyad_b1 = {
        "x1b-w8a_slwm_x2_isolated_450m-s2001": {"eval": {"untouched_vault_spoken_robot": dict(rob_ev)}},
        "x1b-w8a_slwm_x2_isolated_450m-s2002": {"eval": {"untouched_vault_spoken_robot": dict(rob_ev)}},
    }
    assert _verified_vault_shuffled_dyad_ids(dyad_b1, "w8a_slwm_x2_isolated_450m", 4, scene_cids_4) == [0, 0, 1, 1]

    # Missing shuffled_dyad_cluster_ids MUST raise ValueError
    missing_dyad_b = {
        "x1b-w8a_slwm_x2_isolated_450m-s2001": {
            "eval": {"untouched_vault_spoken_robot": {k: v for k, v in rob_ev.items() if k != "shuffled_dyad_cluster_ids"}}
        }
    }
    with pytest.raises(ValueError, match="shuffled_dyad_cluster_ids"):
        _verified_vault_shuffled_dyad_ids(missing_dyad_b, "w8a_slwm_x2_isolated_450m", 4, scene_cids_4)

    # Cross-seed mismatch MUST raise ValueError
    mismatched_seed_b = {
        "x1b-w8a_slwm_x2_isolated_450m-s2001": {"eval": {"untouched_vault_spoken_robot": dict(rob_ev)}},
        "x1b-w8a_slwm_x2_isolated_450m-s2002": {
            "eval": {"untouched_vault_spoken_robot": {**rob_ev, "shuffled_dyad_cluster_ids": [1, 1, 0, 0]}}
        },
    }
    with pytest.raises(ValueError, match="mismatch across seeds"):
        _verified_vault_shuffled_dyad_ids(mismatched_seed_b, "w8a_slwm_x2_isolated_450m", 4, scene_cids_4)

    # Cross-dyad donor (violating supercluster donor-recipient closure!) MUST raise ValueError
    unclosed_donor_b = {
        "x1b-w8a_slwm_x2_isolated_450m-s2001": {
            "eval": {"untouched_vault_spoken_robot": {**rob_ev, "shuffled_donor_indices": [2, 3, 0, 1]}}
        }
    }
    with pytest.raises(ValueError, match="closure violated"):
        _verified_vault_shuffled_dyad_ids(unclosed_donor_b, "w8a_slwm_x2_isolated_450m", 4, scene_cids_4)


def test_wave8a_r2_remediations_and_capability_floor_gates():
    """Verify Wave 8A-r2 across-depth Z_mem isolation, audio-only H_goal routing, clean-tick masking, & Gate 4 floors."""
    from scripts.eval.x1b_analyze_wave8a import (
        COHORT_NAMES,
        EXPECTED_SEEDS,
        HEADLINE_COHORT_METRICS,
        REQUIRED_HIER_COMPS,
        REQUIRED_HIER_METRICS,
        REQUIRED_SHOWCASE_FIGURES,
        evaluate_gate4_checks,
    )
    from scripts.train.x1b_storage import SUBDIR_RE

    # 1. Storage subdir allowlist supports runs_wave8a_r2 and runs_x3_1b while rejecting traversal
    assert SUBDIR_RE.fullmatch("runs") is not None
    assert SUBDIR_RE.fullmatch("runs_wave8a") is not None
    assert SUBDIR_RE.fullmatch("runs_wave8a_r2") is not None
    assert SUBDIR_RE.fullmatch("runs_x3_1b_480p") is not None
    assert SUBDIR_RE.fullmatch("../runs_wave8a_r2") is None
    assert SUBDIR_RE.fullmatch("runs/sub") is None

    # 2. Across-depth Z_mem gradient isolation: Spatial-Only (enable_assoc_loss=False) produces 0.0 gradient
    # on associative rows [32:96] of text_emb and head_text_w[:, 64:96]!
    sp = make_x1b_spec("x_tiny", "slwm_x1b", dtype="float32", grad_checkpoint=False, vocab_size=512)
    p = init_x1b_params(sp, seed=401)
    b_sim = generate_in_domain_deprivileged_sim_batch_v8a(jax.random.key(402), 4, seq_len=64, vocab_size=sp.vocab_size)
    # Force pure associative tokens [32..95] onto text so any trunk task leak into SLOT_TEXT would hit text_emb[32:96]
    mi_txt, mi_ttxt = build_multi_item_assoc_tokens_t64(jax.random.key(403), 4, 64)
    # Zero out quadrant tokens in slot 0 so ONLY [32..95] and 0 appear in text
    mi_txt_pure = mi_txt.at[:, :, 0].set(0)
    b_assoc_only = {**b_sim, "text": mi_txt_pure, "target_text": mi_ttxt, "scenario": jnp.zeros((4,), dtype=jnp.int32)}
    iv_spatial_only = Wave8AInterventions(zmem_isolation=True, enable_assoc_loss=False)

    # Verify layer >= 1 _spatial_cross_modal_attn_wave8a & _gated_delta_temporal_wave8a leak 0.0 task grad into SLOT_TEXT
    from models.x1b_wave8a_core import (
        _run_block_wave8a,
        decode_goal_latch_wave8a,
        decode_target_block_xy_wave8a,
    )
    from models.x1b_world import SLOTS_REGISTER, SLOTS_VISUAL, SLOT_TEXT

    x_blk = jax.random.normal(jax.random.key(404), (2, 6, 24, sp.d_model), dtype=jnp.float32)
    val_blk = jnp.ones((2, 6), dtype=bool)
    mm_blk = jnp.ones((2, 6, 4), dtype=jnp.float32)

    def _emb_only_from_block1(x_in):
        y, _, _ = _run_block_wave8a(x_in, 1, p, sp, iv_spatial_only, val_blk, 0, None, modality_mask=mm_blk)
        return jnp.sum(y[:, :, SLOTS_REGISTER, :] ** 2) + jnp.sum(y[:, :, SLOTS_VISUAL, :] ** 2)

    g_blk1 = jax.grad(_emb_only_from_block1)(x_blk)
    assert float(jnp.max(jnp.abs(g_blk1[:, :, SLOT_TEXT, :]))) == 0.0
    _, g_params = jax.value_and_grad(
        lambda w: world_model_loss_wave8a(w, p, sp, b_assoc_only, iv_spatial_only, jax.random.key(405)),
        has_aux=True,
    )(p)
    assert float(jnp.max(jnp.abs(g_params["head_text_w"][:, 64:96]))) == 0.0

    # 3. Audio-only H_goal routing: when text is masked (mm[:, :, 3] == 0), swapping t=0..15 spoken color audio
    # while keeping t=16..31 direction audio identical strictly shifts BOTH goal latch and 2-block localization at t=31!
    mm_aud_only = jnp.ones((2, 32, 4), dtype=jnp.float32).at[:, :, 3].set(0.0)
    aud_a = jax.random.normal(jax.random.key(406), (2, 32, 64), dtype=jnp.float32)
    aud_b = aud_a.at[:, :16, :].set(-aud_a[:, :16, :])  # Identical t=16..31 direction audio!
    vid32 = jnp.broadcast_to(b_sim["video"][:2, :1], (2, 32, 64, 64, 3))
    base_ao = {
        "video": vid32,
        "audio": aud_a,
        "proprio": b_sim["proprio"][:2, :32],
        "text": jnp.zeros((2, 32, 4), dtype=jnp.int32),
        "action": jnp.zeros((2, 32, 4), dtype=jnp.float32),
        "goal": jnp.zeros((2, 32, 4), dtype=jnp.float32),
        "modality_mask": mm_aud_only,
        "valid": jnp.ones((2, 32), dtype=bool),
    }
    iv_full = Wave8AInterventions(zmem_isolation=True, enable_assoc_loss=True)
    za_raw, _, _ = forward_world_model_wave8a(p, sp, base_ao, iv_full)
    zb_raw, _, _ = forward_world_model_wave8a(p, sp, {**base_ao, "audio": aud_b}, iv_full)
    gl_a = decode_goal_latch_wave8a(p, za_raw[:, -1:], iv_full)
    gl_b = decode_goal_latch_wave8a(p, zb_raw[:, -1:], iv_full)
    blk_a = decode_target_block_xy_wave8a(p, za_raw[:, -1:], iv_full)
    blk_b = decode_target_block_xy_wave8a(p, zb_raw[:, -1:], iv_full)
    assert float(jnp.max(jnp.abs(gl_a - gl_b))) > 1e-4
    assert float(jnp.max(jnp.abs(blk_a - blk_b))) > 1e-5

    # 4. Clean-tick masking: perturbing video on faulted ticks (target_commit == 3) leaves jepa & sigreg clean!
    tc_fault = jnp.zeros((4, 16), dtype=jnp.int32).at[:, 8:].set(3)
    b16_clean = generate_in_domain_deprivileged_sim_batch_v8a(jax.random.key(407), 4, seq_len=16, vocab_size=sp.vocab_size)
    b16_f1 = {**b16_clean, "target_commit": tc_fault}
    b16_f2 = {**b16_f1, "video": b16_f1["video"].at[:, 8:].set(255 - b16_f1["video"][:, 8:])}
    _, m_f1 = world_model_loss_wave8a(p, p, sp, b16_f1, iv_full, jax.random.key(408))
    _, m_f2 = world_model_loss_wave8a(p, p, sp, b16_f2, iv_full, jax.random.key(408))
    assert abs(float(m_f1["jepa"]) - float(m_f2["jepa"])) < 1e-6

    # 5. Explicit Gate 4 capability floors reject mutual degradation (even when Spatial-Only == SLWM-X2)
    def _st(v: float) -> dict[str, object]:
        return {"mean": v, "std": 0.0, "values": [v, v, v]}

    cap_ok = {
        f"K{k}_T{hz}_{sfx}": _st(0.90)
        for hz in (64, 256, 1024)
        for k in (1, 2, 4, 8, 16)
        for sfx in ("acc", "overwritten_acc")
    }
    good_c: dict[str, object] = {
        "n_seeds": 3,
        "seeds_present": list(EXPECTED_SEEDS),
        "source_code_sha256s": ["d" * 64],
        "provenance_verified": True,
        "multi_item_overwrite": cap_ok,
    }
    neg_m = {
        "vault_shuffled_scene_color_disc": 0.50,
        "vault_shuffled_scene_color_audio_only_disc": 0.50,
        "vault_color_prior_scene_color_disc": 0.52,
        "vault_paired_flip_dual_color_disc": 0.50,
        "vault_paired_flip_dual_audio_only_disc": 0.45,
    }
    for mk in HEADLINE_COHORT_METRICS:
        good_c[mk] = _st(0.98 if mk == "dyn_assoc_T1024" else neg_m.get(mk, 0.84))
    groups_ok = {c: dict(good_c) for c in COHORT_NAMES}
    hier_ok: dict[str, object] = {}
    for comp in REQUIRED_HIER_COMPS:
        for mn in REQUIRED_HIER_METRICS:
            ent: dict[str, object] = {"mean": 0.0, "hierarchical_ci95": [-0.01, 0.01], "n_seeds": 3}
            if mn.startswith("vault_"):
                ent.update({"scene_clustered": True, "n_level2_clusters": 12, "n_episodes_per_seed": 64})
            if mn == "unseen_corruption_auroc":
                ent["paired_world_auroc_recomputed"] = True
            hier_ok[f"isolated_vs_{comp}_{mn}"] = ent
    for sgk, sgm in (
        ("isolated_scene_vs_shuffled_joint", 0.34),
        ("isolated_scene_vs_shuffled_audio_only", 0.34),
        ("isolated_scene_vs_color_prior_joint", 0.32),
        ("isolated_scene_vs_color_prior_audio_only", 0.32),
        ("isolated_paired_flip_dual_joint", 0.50),
        ("isolated_paired_flip_dual_audio_only", 0.45),
    ):
        sge: dict[str, object] = {
            "mean": sgm,
            "hierarchical_ci95": [sgm - 0.05, sgm + 0.05],
            "n_seeds": 3,
            "scene_clustered": True,
            "n_level2_clusters": 12,
            "n_episodes_per_seed": 64,
        }
        if "shuffled" in sgk:
            sge["supercluster_donor_closed"] = True
        hier_ok[sgk] = sge
    sc_ok = {
        "rendered": True,
        "preregistration_verified": True,
        "cohorts_included": list(COHORT_NAMES),
        "artifacts": list(REQUIRED_SHOWCASE_FIGURES),
    }
    assert evaluate_gate4_checks(groups_ok, hier_ok, sc_ok)["gate4_overall_pass"] is True

    # Fail closed if commit_selective_routing == 0.54 across ALL cohorts (diff == 0.0!)
    low_commit = {c: {**groups_ok[c], "commit_selective_routing": _st(0.54)} for c in COHORT_NAMES}
    assert evaluate_gate4_checks(low_commit, hier_ok, sc_ok)["gate4_overall_pass"] is False

    # Fail closed if vault_spoken_audio_only_disc == 0.51 across ALL cohorts (diff == 0.0!)
    low_aud = {c: {**groups_ok[c], "vault_spoken_audio_only_disc": _st(0.51)} for c in COHORT_NAMES}
    assert evaluate_gate4_checks(low_aud, hier_ok, sc_ok)["gate4_overall_pass"] is False

    # Fail closed if unseen_mppi (0.70) < no_jepa_subgoal (0.78) - 0.02!
    mppi_below_nojepa = {
        c: {**groups_ok[c], "unseen_mppi": _st(0.70), "no_jepa_subgoal": _st(0.78)} for c in COHORT_NAMES
    }
    assert evaluate_gate4_checks(mppi_below_nojepa, hier_ok, sc_ok)["gate4_overall_pass"] is False

    # 6. Verify scen_arr < 0 (oblique xArm6 / PyMunk) contributes exact 0.0 to commit_loss, and audio-only scen_arr == -2
    # pre-direction ticks (t < 28) contribute exact 0.0 to goal_loss!
    b_rob_neg = {
        **b16_clean,
        "scenario": jnp.full((4,), -2, dtype=jnp.int32),
        "modality_mask": jnp.ones((4, 16, 4), dtype=jnp.float32).at[:, :, 3].set(0.0),
        "target_goal_xy": jnp.ones((4, 16, 2), dtype=jnp.float32) * 0.9,
    }
    _, m_rob_neg = world_model_loss_wave8a(p, p, sp, b_rob_neg, iv_full, jax.random.key(409))
    assert float(m_rob_neg["commit_ce"]) == 0.0
    assert float(m_rob_neg["goal_mse"]) == 0.0
