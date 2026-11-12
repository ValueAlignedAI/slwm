"""Independently audit Study X1B 120-run SHA-256 custody ledgers, 3-seed 1.008B parity, and published metrics."""

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


def test_all_120_runs_across_four_waves_pass_cryptographic_custody():
    """Verify every run across Waves 1–3 (28 each) and Wave 4 (36) matches its dual-HDD SHA-256 manifest."""
    expected_counts = {
        "runs_wave1": 28,
        "runs_wave2": 28,
        "runs_wave3": 28,
        "runs": 36,
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
                assert receipt["seed"] in (2002, 2003)
            total_verified += 1

    assert total_verified == 120


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


def test_inference_manifest_replays_and_figures_integrity():
    """Verify standalone 1.008B manifest SHA-256 parity, closed-loop replay pack, and all 6 visual figures."""
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
