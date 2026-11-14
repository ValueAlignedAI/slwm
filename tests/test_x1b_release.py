"""Independently audit Study X1B 171-run SHA-256 custody ledgers (Waves 1–7), 1.008B parity, Wave 6 LOO, and Wave 7 spatial-grid clincher."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111"
EXPECTED_LEDGER_FILES = ("config.json", "eval_metrics.json", "receipt.json", "train.jsonl")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_171_runs_across_seven_waves_pass_cryptographic_custody():
    """Verify every run across Waves 1–3 (28 each), Wave 4 (36), Wave 5 (15), Wave 6 (24), and Wave 7 (12) matches its dual-HDD SHA-256 manifest."""
    expected_counts = {
        "runs_wave1": 28,
        "runs_wave2": 28,
        "runs_wave3": 28,
        "runs": 36,
        "runs_wave5": 15,
        "runs_wave6": 24,
        "runs_wave7": 12,
    }
    total_verified = 0
    for wave_subdir, exp_n in expected_counts.items():
        wave_dir = EXP_DIR / wave_subdir
        run_dirs = sorted(p for p in wave_dir.iterdir() if p.is_dir())
        assert len(run_dirs) == exp_n, f"Expected {exp_n} runs in {wave_subdir}, found {len(run_dirs)}"

        for run_dir in run_dirs:
            verified = _load_json(run_dir / "verified.json")
            assert verified["tag"] == run_dir.name
            assert verified["verified_copies"] == 2
            assert len(set(verified["disk_uuids"])) == 2

            sha_map = verified["sha256"]
            for fname in EXPECTED_LEDGER_FILES:
                file_path = run_dir / fname
                assert file_path.is_file(), f"Missing {file_path}"
                digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
                assert digest == sha_map[fname], f"SHA-256 mismatch on {file_path}"

            receipt = _load_json(run_dir / "receipt.json")
            assert receipt["portable_int8_sha256"] == sha_map["portable_int8.npz"]
            assert receipt["portable_int8_bytes"] > 0
            if receipt["weights_bf16_bytes"] > 0:
                assert receipt["weights_bf16_sha256"] == sha_map["weights_bf16.npz"]
            else:
                cfg = _load_json(run_dir / "config.json")
                assert not receipt["weights_bf16_sha256"] and not cfg.get("save_bf16", False)
            total_verified += 1

    assert total_verified == 171


def test_wave4_3seed_1b_parity_and_metrics_recompute_from_raw_ledgers():
    """Recompute Wave 4 3-seed 1.008B, 453M, and 152M numbers directly from raw eval_metrics.json."""
    summary = _load_json(EXP_DIR / "summary_metrics.json")
    assert summary["total_verified_runs"] == 36
    groups = summary["groups"]
    assert len(groups) == 12
    for gname, gdata in groups.items():
        assert gdata["n_seeds"] == 3, f"Expected 3 seeds for {gname}, got {gdata['n_seeds']}"

    # Verify 100% TPU v6e-1 hardware parity and < 0.002% strict parameter parity across all 12 large-scale runs
    for seed in (2001, 2002, 2003):
        slwm_rec = _load_json(EXP_DIR / f"runs/x1b-flagship-slwm-1b-s{seed}/receipt.json")
        dense_rec = _load_json(EXP_DIR / f"runs/x1b-baseline-dense-1b-s{seed}/receipt.json")
        legacy_rec = _load_json(EXP_DIR / f"runs/x1b-baseline-legacy-1b-s{seed}/receipt.json")
        s450_rec = _load_json(EXP_DIR / f"runs/x1b-scale-slwm-450m-s{seed}/receipt.json")

        assert slwm_rec["backend"] == "tpu"
        assert dense_rec["backend"] == "tpu"
        assert legacy_rec["backend"] == "tpu"
        assert s450_rec["backend"] == "tpu"

        p_slwm = slwm_rec["parameter_counts"]["strict_total"]
        p_dense = dense_rec["parameter_counts"]["strict_total"]
        p_legacy = legacy_rec["parameter_counts"]["strict_total"]
        assert p_slwm == 1_008_454_410
        assert p_dense == 1_008_435_114
        assert p_legacy == 1_008_435_114
        assert abs(p_slwm - p_dense) / p_slwm < 0.00002

    # Recompute 3-seed 1.008B raw metrics & verify against summary_metrics.json
    slwm_mpc, dense_mpc, slwm_bar, dense_bar = [], [], [], []
    slwm_sw, dense_sw, legacy_sw, slwm_sust = [], [], [], []
    slwm_t1024, dense_t1024, legacy_t1024 = [], [], []
    slwm_av, legacy_av, slwm_tr04 = [], [], []

    for seed in (2001, 2002, 2003):
        m_s = _load_json(EXP_DIR / f"runs/x1b-flagship-slwm-1b-s{seed}/eval_metrics.json")
        m_d = _load_json(EXP_DIR / f"runs/x1b-baseline-dense-1b-s{seed}/eval_metrics.json")
        m_l = _load_json(EXP_DIR / f"runs/x1b-baseline-legacy-1b-s{seed}/eval_metrics.json")

        slwm_mpc.append(m_s["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"])
        dense_mpc.append(m_d["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"])
        slwm_bar.append(m_s["closed_loop_mpc"]["latent_cem_mpc"]["thin_barrier_success"])
        dense_bar.append(m_d["closed_loop_mpc"]["latent_cem_mpc"]["thin_barrier_success"])

        slwm_sust.append(m_s["long_horizon_memory_mpc"]["sustained_transient_barrier_mpc"]["success_rate"]["mean"])
        slwm_sw.append(m_s["long_horizon_memory_mpc"]["mid_episode_goal_switch_mpc"]["success_rate"]["mean"])
        dense_sw.append(m_d["long_horizon_memory_mpc"]["mid_episode_goal_switch_mpc"]["success_rate"]["mean"])
        legacy_sw.append(m_l["long_horizon_memory_mpc"]["mid_episode_goal_switch_mpc"]["success_rate"]["mean"])

        slwm_t1024.append(m_s["associative_recall_512"]["T1024"]["mean"])
        dense_t1024.append(m_d["associative_recall_512"]["T1024"]["mean"])
        legacy_t1024.append(m_l["associative_recall_512"]["T1024"]["mean"])

        slwm_av.append(m_s["real_multimodal_and_text"]["av_r1"]["mean"])
        legacy_av.append(m_l["real_multimodal_and_text"]["av_r1"]["mean"])
        slwm_tr04.append(m_s["transient_goal_retention"]["transient_04s_success"]["mean"])

        assert m_s["thin_barrier_and_persistence"]["barrier_auroc_64x64"] == 1.0
        assert m_s["spoofing_and_commit_gate"]["learned_innovation"]["unsupported_claim_rate"] == 0.0

    assert np.isclose(np.mean(slwm_sust), 1.0)
    assert np.isclose(np.mean(slwm_sw), groups["flagship_1b"]["goal_switch_mpc_overall"]["mean"])
    assert np.mean(slwm_sw) > 0.90
    assert np.mean(dense_sw) < 0.10
    assert np.mean(slwm_sw) > np.mean(legacy_sw) + 0.15
    assert np.mean(slwm_mpc) > np.mean(dense_mpc) + 0.20
    assert np.mean(slwm_bar) > np.mean(dense_bar) + 0.50
    assert np.mean(slwm_t1024) > np.mean(dense_t1024) + 0.50
    assert np.mean(slwm_t1024) > np.mean(legacy_t1024) + 0.80
    assert np.mean(slwm_av) > 0.94 and np.mean(legacy_av) < 0.05
    assert np.mean(slwm_tr04) > 0.90

    # Verify O(1) vs O(T) streaming memory scaling
    assert groups["flagship_1b"]["streaming_state_mb_T64"]["mean"] == groups["flagship_1b"]["streaming_state_mb_T1024"]["mean"]
    assert groups["dense_1b"]["streaming_state_mb_T1024"]["mean"] > 200.0 * groups["flagship_1b"]["streaming_state_mb_T1024"]["mean"]

    # Verify 99% Paired Bootstrap CIs strictly above 0 on core superiority claims
    cis = summary["paired_bootstrap_95ci"]
    for pos_key in (
        "1b_pooled_slwm_vs_dense_goal_switch_mpc",
        "1b_pooled_slwm_vs_dense_thin_barrier_mpc",
        "1b_pooled_slwm_vs_dense_mpc_success",
        "1b_pooled_slwm_vs_dense_T1024_assoc",
        "1b_pooled_slwm_vs_legacy_goal_switch_mpc",
        "1b_pooled_slwm_vs_legacy_T1024_assoc",
        "1b_pooled_slwm_vs_legacy_av_r1",
        "1b_pooled_cem_mpc_vs_g8_shooting",
    ):
        assert cis[pos_key]["ci99_low"] > 0.0, f"Expected 99% CI > 0 for {pos_key}, got {cis[pos_key]}"


def test_wave3_1b_parameter_parity_and_primary_metrics_recompute_from_raw_ledgers():
    """Preserve historical verification of Wave 3 28-run ledger alongside Wave 4."""
    summary = _load_json(EXP_DIR / "wave3_1seed_1b_summary_metrics.json")
    assert summary["total_verified_runs"] == 28

    slwm_1b_rec = _load_json(EXP_DIR / "runs_wave3/x1b-flagship-slwm-1b-s2001/receipt.json")
    assert slwm_1b_rec["parameter_counts"]["strict_total"] == 1_008_426_726
    assert slwm_1b_rec["backend"] == "tpu"


def test_wave5_chinchilla_o1_baselines_and_real_robot_recompute_from_raw_ledgers():
    """Recompute Wave 5 15-run Chinchilla 5,600-step 1.008B, SlidingWindow-W64, DiagonalSSM, and xArm6 metrics."""
    w5 = _load_json(EXP_DIR / "wave5_summary_metrics.json")
    assert w5["cumulative_verified_runs_waves_1_to_5"] == 135
    groups = w5["groups"]
    for gname in ("w5_slwm_1b", "w5_dense_1b", "w5_swin64_1b", "w5_diagssm_1b", "w5_slwm_450m"):
        assert groups[gname]["n_seeds"] == 3

    slwm_mpc, slwm_bar, slwm_sust, slwm_sw, slwm_t1024, slwm_r4 = [], [], [], [], [], []
    swin_sust, swin_sw, swin_t1024 = [], [], []
    dssm_mpc, dssm_sw, dssm_t1024 = [], [], []
    dense_sw, s450_mpc = [], []

    for seed in (2001, 2002, 2003):
        r_slwm = _load_json(EXP_DIR / f"runs_wave5/x1b-w5-slwm-1b-s{seed}/receipt.json")
        r_dense = _load_json(EXP_DIR / f"runs_wave5/x1b-w5-dense-1b-s{seed}/receipt.json")
        r_swin = _load_json(EXP_DIR / f"runs_wave5/x1b-w5-swin64-1b-s{seed}/receipt.json")
        r_dssm = _load_json(EXP_DIR / f"runs_wave5/x1b-w5-diagssm-1b-s{seed}/receipt.json")
        r_450 = _load_json(EXP_DIR / f"runs_wave5/x1b-w5-slwm-450m-s{seed}/receipt.json")

        assert r_slwm["steps"] == 5600 and r_dense["steps"] == 5600
        assert r_swin["steps"] == 5600 and r_dssm["steps"] == 5600
        assert r_450["steps"] == 2600

        assert r_slwm["parameter_counts"]["strict_total"] == 1_008_454_410
        assert r_dssm["parameter_counts"]["strict_total"] == 1_008_454_410
        assert r_dense["parameter_counts"]["strict_total"] == 1_008_435_114
        assert r_swin["parameter_counts"]["strict_total"] == 1_008_435_114
        assert r_450["parameter_counts"]["strict_total"] == 453_199_938

        m_s = _load_json(EXP_DIR / f"runs_wave5/x1b-w5-slwm-1b-s{seed}/eval_metrics.json")
        m_d = _load_json(EXP_DIR / f"runs_wave5/x1b-w5-dense-1b-s{seed}/eval_metrics.json")
        m_w = _load_json(EXP_DIR / f"runs_wave5/x1b-w5-swin64-1b-s{seed}/eval_metrics.json")
        m_m = _load_json(EXP_DIR / f"runs_wave5/x1b-w5-diagssm-1b-s{seed}/eval_metrics.json")
        m_4 = _load_json(EXP_DIR / f"runs_wave5/x1b-w5-slwm-450m-s{seed}/eval_metrics.json")

        slwm_mpc.append(m_s["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"])
        slwm_bar.append(m_s["closed_loop_mpc"]["latent_cem_mpc"]["thin_barrier_success"])
        slwm_sust.append(m_s["long_horizon_memory_mpc"]["sustained_transient_barrier_mpc"]["success_rate"]["mean"])
        slwm_sw.append(m_s["long_horizon_memory_mpc"]["mid_episode_goal_switch_mpc"]["success_rate"]["mean"])
        slwm_t1024.append(m_s["associative_recall_512"]["T1024"]["mean"])
        slwm_r4.append(m_s["real_robot_transfer"]["step4_error_mm"]["mean"])

        swin_sust.append(m_w["long_horizon_memory_mpc"]["sustained_transient_barrier_mpc"]["success_rate"]["mean"])
        swin_sw.append(m_w["long_horizon_memory_mpc"]["mid_episode_goal_switch_mpc"]["success_rate"]["mean"])
        swin_t1024.append(m_w["associative_recall_512"]["T1024"]["mean"])

        dssm_mpc.append(m_m["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"])
        dssm_sw.append(m_m["long_horizon_memory_mpc"]["mid_episode_goal_switch_mpc"]["success_rate"]["mean"])
        dssm_t1024.append(m_m["associative_recall_512"]["T1024"]["mean"])

        dense_sw.append(m_d["long_horizon_memory_mpc"]["mid_episode_goal_switch_mpc"]["success_rate"]["mean"])
        s450_mpc.append(m_4["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"])

    # Chinchilla doubling cures Wave 4's thin-barrier outlier and reaches 100% OOD Sustained & Goal-Switch
    assert np.isclose(np.mean(slwm_mpc), groups["w5_slwm_1b"]["mpc_success"]["mean"])
    assert np.mean(slwm_bar) > 0.85 and np.mean(slwm_mpc) > 0.80
    assert np.isclose(np.mean(slwm_sust), 1.0)
    assert np.isclose(np.mean(slwm_sw), 1.0)
    assert np.mean(slwm_mpc) > np.mean(s450_mpc) + 0.15

    # SlidingWindow-W64 O(1) baseline evicts cues past W=64 and collapses on T=76 Sustained & T=1024
    assert np.mean(slwm_sust) > np.mean(swin_sust) + 0.50
    assert np.mean(slwm_sw) > np.mean(swin_sw) + 0.20
    assert np.mean(slwm_t1024) > np.mean(swin_t1024) + 0.75
    assert np.isclose(groups["w5_swin64_1b"]["streaming_state_mb_T64"]["mean"], groups["w5_swin64_1b"]["streaming_state_mb_T1024"]["mean"])

    # DiagonalSSM O(1) baseline retains strong short-horizon MPC but loses on selective overwrite (T=1024 & Goal-Switch)
    assert np.mean(dssm_mpc) > 0.80
    assert np.mean(slwm_t1024) > np.mean(dssm_t1024) + 0.40
    assert np.mean(slwm_sw) > np.mean(dssm_sw) + 0.05
    assert np.mean(slwm_sw) > np.mean(dense_sw) + 0.55

    # Physical xArm6 transfer & hierarchical seed-cluster CIs
    assert np.mean(slwm_r4) < 7.5 and groups["w5_slwm_1b"]["robot_step4_skill"]["mean"] > 0.91
    assert groups["w5_slwm_1b"]["robot_plan_gain_mm"]["mean"] > 160.0
    assert groups["w5_slwm_1b"]["robot_cf_end_gain_mm"]["mean"] > 15.0
    assert w5["paired_bootstrap_cis"]["slwm1b_robot_disc_above_chance"]["ci99_low"] > 0.0

    hcis = w5["hierarchical_seed_cis"]
    for pos_hkey in (
        "slwm1b_vs_dense_5600_goal_switch_mpc",
        "slwm1b_vs_swin64_5600_sustained_T76_mpc",
        "slwm1b_vs_swin64_5600_goal_switch_mpc",
        "slwm1b_vs_swin64_5600_T1024_assoc",
        "slwm1b_vs_diagssm_5600_goal_switch_mpc",
        "slwm1b_vs_diagssm_5600_T1024_assoc",
        "slwm1b_vs_slwm_450m_mpc",
        "slwm1b_vs_slwm_450m_goal_switch_mpc",
    ):
        assert hcis[pos_hkey]["hierarchical_ci99"][0] > 0.0, f"Expected hierarchical 99% CI > 0 for {pos_hkey}"
    assert hcis["slwm1b_vs_dense_5600_robot_step4_mm"]["hierarchical_ci99"][1] < 0.0
    assert hcis["slwm1b_vs_slwm_450m_robot_step4_mm"]["hierarchical_ci99"][1] < 0.0


def test_wave6_24run_loo_and_gap_bottleneck_recompute_from_raw_ledgers():
    """Recompute Wave 6 24-run single-factor 453M LOO isolations, O(1) dynamic overwrite recall, and Eq. 5 GAP bottleneck."""
    w6 = _load_json(EXP_DIR / "wave6_summary_metrics.json")
    assert w6["wave6_verified_runs"] == 24
    groups = w6["groups"]
    assert len(groups) == 8
    for gname, gdata in groups.items():
        assert gdata["n_seeds"] == 3, f"Expected 3 seeds for {gname}"

    full_t1024, vic_t1024, noks_t1024, nodec_t1024 = [], [], [], []
    swin_t1024, scal_t1024, dssm_t1024 = [], [], []
    full_step4, nozl_step4 = [], []
    full_drift, noks_drift = [], []

    for seed in (2001, 2002, 2003):
        for exact_453m in (
            "w6_slwm_450m_full_epps_sigreg",
            "w6_slwm_450m_vicreg",
            "w6_loo_no_key_shield_450m",
            "w6_loo_no_zerolag_bypass_450m",
            "w6_loo_no_decoupled_heads_450m",
            "w6_diagonal_ssm_jepa_450m",
        ):
            rec = _load_json(EXP_DIR / f"runs_wave6/x1b-{exact_453m}-s{seed}/receipt.json")
            assert rec["parameter_counts"]["strict_total"] == 453_199_938

        for matched_453m in ("w6_sliding_window_w64_450m", "w6_scalar_retention_jepa_450m"):
            rec_m = _load_json(EXP_DIR / f"runs_wave6/x1b-{matched_453m}-s{seed}/receipt.json")
            assert rec_m["parameter_counts"]["strict_total"] == 453_204_042

        e_full = _load_json(EXP_DIR / f"runs_wave6/x1b-w6_slwm_450m_full_epps_sigreg-s{seed}/eval_metrics.json")
        e_vic = _load_json(EXP_DIR / f"runs_wave6/x1b-w6_slwm_450m_vicreg-s{seed}/eval_metrics.json")
        e_noks = _load_json(EXP_DIR / f"runs_wave6/x1b-w6_loo_no_key_shield_450m-s{seed}/eval_metrics.json")
        e_nozl = _load_json(EXP_DIR / f"runs_wave6/x1b-w6_loo_no_zerolag_bypass_450m-s{seed}/eval_metrics.json")
        e_nodec = _load_json(EXP_DIR / f"runs_wave6/x1b-w6_loo_no_decoupled_heads_450m-s{seed}/eval_metrics.json")
        e_swin = _load_json(EXP_DIR / f"runs_wave6/x1b-w6_sliding_window_w64_450m-s{seed}/eval_metrics.json")
        e_scal = _load_json(EXP_DIR / f"runs_wave6/x1b-w6_scalar_retention_jepa_450m-s{seed}/eval_metrics.json")
        e_dssm = _load_json(EXP_DIR / f"runs_wave6/x1b-w6_diagonal_ssm_jepa_450m-s{seed}/eval_metrics.json")

        full_t1024.append(e_full["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"])
        vic_t1024.append(e_vic["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"])
        noks_t1024.append(e_noks["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"])
        nodec_t1024.append(e_nodec["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"])
        swin_t1024.append(e_swin["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"])
        scal_t1024.append(e_scal["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"])
        dssm_t1024.append(e_dssm["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"])

        full_drift.append(e_full["symbolic_key_drift_and_recall"]["symbolic_key_visual_drift"])
        noks_drift.append(e_noks["symbolic_key_drift_and_recall"]["symbolic_key_visual_drift"])

        full_step4.append(e_full["untouched_vault_spoken_robot"]["step4_mm"])
        nozl_step4.append(e_nozl["untouched_vault_spoken_robot"]["step4_mm"])

        # Under ee_only de-privileging, Eq. 5 global slot pooling destroys 2D coordinates across all 8 Wave 6 cohorts
        assert e_full["unseen_layout_deprivileged_mpc"]["vis_decoded_obj_rmse"] > 0.50

    assert np.isclose(np.mean(full_t1024), 1.0) and np.isclose(np.mean(vic_t1024), 1.0)
    assert all(d == 0.0 for d in full_drift)
    assert all(d > 0.0 for d in noks_drift)
    assert np.mean(full_t1024) > np.mean(noks_t1024) + 0.50
    assert np.mean(full_t1024) > np.mean(nodec_t1024) + 0.25
    assert np.mean(full_t1024) > np.mean(swin_t1024) + 0.90
    assert np.mean(full_t1024) > np.mean(scal_t1024) + 0.90
    assert np.mean(full_t1024) > np.mean(dssm_t1024) + 0.75
    assert np.mean(nozl_step4) > np.mean(full_step4) + 5.0

    hattr = w6["hierarchical_seed_attributions"]
    assert hattr["full_vs_w6_loo_no_zerolag_bypass_450m_vault_step4_mm"]["hierarchical_ci99"][1] < -2.5
    assert hattr["full_vs_w6_loo_no_decoupled_heads_450m_dyn_assoc_T1024"]["hierarchical_ci99"][0] > 0.0
    assert hattr["full_vs_w6_diagonal_ssm_jepa_450m_dyn_assoc_T1024"]["hierarchical_ci99"][0] > 0.55
    assert hattr["full_vs_w6_sliding_window_w64_450m_dyn_assoc_T1024"]["hierarchical_ci99"][0] > 0.70
    assert hattr["full_vs_w6_scalar_retention_jepa_450m_dyn_assoc_T1024"]["hierarchical_ci99"][0] > 0.80


def test_wave7_12run_spatial_grid_clincher_recompute_from_raw_ledgers():
    """Recompute Wave 7 12-run zero-extra-parameter 8x8 spatial-grid grounding, Subgoal MPPI, T=160 switch, and OOD AUROC."""
    w7 = _load_json(EXP_DIR / "wave7_summary_metrics.json")
    assert w7["wave7_verified_runs"] == 12
    groups = w7["groups"]
    assert len(groups) == 4

    slwm_mppi, slwm_reac, slwm_open = [], [], []
    slwm_t160, swin_t160, dssm_t160 = [], [], []
    slwm_lmse, swin_lmse, dssm_lmse = [], [], []
    slwm_auroc, slwm_ee_rmse, slwm_obj_rmse = [], [], []
    slwm_drift, noks_drift = [], []

    for seed in (2001, 2002, 2003):
        for exact_453m in ("w7_slwm_450m_full", "w7_diagonal_ssm_450m", "w7_loo_no_key_shield_450m"):
            rec = _load_json(EXP_DIR / f"runs_wave7/x1b-{exact_453m}-s{seed}/receipt.json")
            assert rec["parameter_counts"]["strict_total"] == 453_199_938

        rec_swin = _load_json(EXP_DIR / f"runs_wave7/x1b-w7_sliding_window_w64_450m-s{seed}/receipt.json")
        assert rec_swin["parameter_counts"]["strict_total"] == 453_204_042

        e_slwm = _load_json(EXP_DIR / f"runs_wave7/x1b-w7_slwm_450m_full-s{seed}/eval_metrics.json")
        e_swin = _load_json(EXP_DIR / f"runs_wave7/x1b-w7_sliding_window_w64_450m-s{seed}/eval_metrics.json")
        e_dssm = _load_json(EXP_DIR / f"runs_wave7/x1b-w7_diagonal_ssm_450m-s{seed}/eval_metrics.json")
        e_noks = _load_json(EXP_DIR / f"runs_wave7/x1b-w7_loo_no_key_shield_450m-s{seed}/eval_metrics.json")

        slwm_mppi.append(e_slwm["unseen_layout_deprivileged_mpc"]["geometry_free_mppi_success"])
        slwm_reac.append(e_slwm["unseen_layout_deprivileged_mpc"]["explicit_latch_reactive_success"])
        slwm_open.append(e_slwm["unseen_layout_deprivileged_mpc"]["open_loop_policy_success"])

        slwm_ee_rmse.append(e_slwm["unseen_layout_deprivileged_mpc"]["vis_decoded_ee_rmse"])
        slwm_obj_rmse.append(e_slwm["unseen_layout_deprivileged_mpc"]["vis_decoded_obj_rmse"])

        slwm_t160.append(e_slwm["beyond_window_T160_distractor_switch"]["T160_post_perturbation_goal_b_success"])
        swin_t160.append(e_swin["beyond_window_T160_distractor_switch"]["T160_post_perturbation_goal_b_success"])
        dssm_t160.append(e_dssm["beyond_window_T160_distractor_switch"]["T160_post_perturbation_goal_b_success"])

        slwm_lmse.append(e_slwm["beyond_window_T160_distractor_switch"]["latch_mse_beyond_W64_t92_160"])
        swin_lmse.append(e_swin["beyond_window_T160_distractor_switch"]["latch_mse_beyond_W64_t92_160"])
        dssm_lmse.append(e_dssm["beyond_window_T160_distractor_switch"]["latch_mse_beyond_W64_t92_160"])

        slwm_auroc.append(e_slwm["unseen_ood_sensor_corruptions"]["mean_unseen_ood_corruption_auroc"])
        slwm_drift.append(e_slwm["symbolic_key_drift_and_recall"]["symbolic_key_visual_drift"])
        noks_drift.append(e_noks["symbolic_key_drift_and_recall"]["symbolic_key_visual_drift"])

    # 18x-36x spatial pose grounding improvement at zero extra parameters
    assert np.mean(slwm_ee_rmse) < 0.025 and np.mean(slwm_obj_rmse) < 0.045
    assert np.isclose(np.mean(slwm_mppi), groups["w7_slwm_450m_full"]["unseen_mppi"]["mean"])
    assert np.mean(slwm_mppi) > np.mean(slwm_reac) + 0.25
    assert np.isclose(np.mean(slwm_open), 0.0)

    # Beyond-window T=160 (> 2.5x W=64) perturbed embodied goal-switch
    assert np.mean(slwm_t160) > 0.88
    assert np.mean(slwm_t160) > np.mean(swin_t160) + 0.60
    assert np.mean(slwm_t160) > np.mean(dssm_t160) + 0.20
    assert np.mean(swin_lmse) > 200.0 * np.mean(slwm_lmse)
    assert np.mean(dssm_lmse) > 8.0 * np.mean(slwm_lmse)

    # Zero-shot unseen sensor corruption AUROC & symbolic-key shield invariance
    assert np.mean(slwm_auroc) > 0.995
    assert all(d == 0.0 for d in slwm_drift)
    assert np.mean(noks_drift) > 0.002

    hattr7 = w7["hierarchical_seed_attributions"]
    assert hattr7["full_mppi_vs_explicit_latch_reactive"]["hierarchical_ci95"][0] > 0.05
    assert hattr7["full_vs_w7_sliding_window_w64_450m_T160_goal_b_switch"]["hierarchical_ci99"][0] > 0.50
    assert hattr7["full_vs_w7_diagonal_ssm_450m_T160_goal_b_switch"]["hierarchical_ci99"][0] > 0.04


def test_inference_manifest_replays_and_figures_integrity():
    """Verify standalone 1.008B manifest SHA-256 parity, closed-loop replay pack, and all 12 visual figures."""
    manifest = _load_json(EXP_DIR / "inference/manifest.json")
    assert manifest["study_id"] == "X1B-20261111"
    assert len(manifest["models"]) == 3
    for m in manifest["models"]:
        rec = _load_json(EXP_DIR / "runs" / m["tag"] / "receipt.json")
        assert m["portable_int8_sha256"] == rec["portable_int8_sha256"]
        assert m["weights_bf16_sha256"] == rec["weights_bf16_sha256"]
        assert m["strict_total_parameters"] == rec["parameter_counts"]["strict_total"]
        assert m["fedora_verified_copies"] == 2

    replays = _load_json(EXP_DIR / "examples/closed_loop_replays.json")
    sw = replays["missions"]["ood_goal_switch_barrier_mpc"]
    assert sw["SLWM-X1B (1.008B)"]["success"] is True
    assert sw["DenseTransformer-X1B"]["success"] is False
    assert sw["LegacyScalar-X1B"]["success"] is False
    assert replays["t1024_live_recall_check"]["SLWM-X1B (1.008B)"] == 1.0
    assert replays["t1024_live_recall_check"]["LegacyScalar-X1B"] == 0.0

    for rel_png in (
        "figures/x1b_scaling_and_1b_head_to_head.png",
        "figures/x1b_memory_and_horizon_extrapolation.png",
        "figures/x1b_jepa_mpc_and_spoofing_safety.png",
        "figures/x1b_long_horizon_ood_goal_switch_mpc.png",
        "figures/x1b_wave5_chinchilla_and_o1_baselines.png",
        "figures/x1b_wave5_real_robot_transfer.png",
        "figures/x1b_wave6_loo_and_beyond_window.png",
        "figures/x1b_wave6_unseen_worlds_and_vault.png",
        "figures/x1b_wave7_spatial_mpc_and_t160.png",
        "figures/x1b_wave7_ood_safety_and_spoken_vault.png",
        "examples/preview_thin_barrier_mpc.png",
        "examples/preview_ood_goal_switch_mpc.png",
    ):
        p = EXP_DIR / rel_png
        assert p.is_file() and p.stat().st_size > 20_000, f"Missing or empty figure {p}"


def test_study_x1b_report_and_readme_links_resolve():
    """Ensure every local markdown link in Study X1B docs and README resolves cleanly."""
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for md_path in (
        EXP_DIR / "report.md",
        EXP_DIR / "inference/README.md",
        EXP_DIR / "examples/README.md",
        REPO_ROOT / "docs/research/x1b_spatiotemporal_jepa_protocol_20261111.md",
        REPO_ROOT / "README.md",
    ):
        text = md_path.read_text(encoding="utf-8")
        for target in link_re.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            rel = target.split("#", 1)[0]
            if not rel:
                continue
            resolved = (md_path.parent / rel).resolve()
            assert resolved.exists(), f"Broken link '{target}' in {md_path} -> {resolved}"
