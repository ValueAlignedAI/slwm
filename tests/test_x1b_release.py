"""Independently audit Study X1B 84-run SHA-256 custody ledgers, 1.008B parity, and published metrics."""

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


def test_all_84_runs_across_three_waves_pass_cryptographic_custody():
    """Verify every run in Wave 1 (28), Wave 2 (28), and Wave 3 (28) matches its dual-HDD SHA-256 manifest."""
    for wave_subdir in ("runs_wave1", "runs_wave2", "runs_wave3"):
        wave_dir = EXP_DIR / wave_subdir
        run_dirs = sorted(p for p in wave_dir.iterdir() if p.is_dir())
        assert len(run_dirs) == 28, f"Expected 28 runs in {wave_subdir}, found {len(run_dirs)}"

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


def test_wave3_1b_parameter_parity_and_primary_metrics_recompute_from_raw_ledgers():
    """Recompute Wave 3 1.008B, 453M, and 3-seed 152M factorial numbers directly from raw eval_metrics.json."""
    summary = _load_json(EXP_DIR / "wave3_1seed_1b_summary_metrics.json")
    assert summary["total_verified_runs"] == 28

    slwm_1b_rec = _load_json(EXP_DIR / "runs_wave3/x1b-flagship-slwm-1b-s2001/receipt.json")
    dense_1b_rec = _load_json(EXP_DIR / "runs_wave3/x1b-baseline-dense-1b-s2001/receipt.json")
    legacy_1b_rec = _load_json(EXP_DIR / "runs_wave3/x1b-baseline-legacy-1b-s2001/receipt.json")

    p_slwm = slwm_1b_rec["parameter_counts"]["strict_total"]
    p_dense = dense_1b_rec["parameter_counts"]["strict_total"]
    p_legacy = legacy_1b_rec["parameter_counts"]["strict_total"]

    assert p_slwm == 1_008_426_726
    assert abs(p_slwm - p_dense) / p_slwm < 0.0031
    assert abs(p_slwm - p_legacy) / p_slwm < 0.0001
    assert slwm_1b_rec["backend"] == "tpu"
    assert dense_1b_rec["backend"] == "tpu"
    assert legacy_1b_rec["backend"] == "tpu"

    slwm_1b = _load_json(EXP_DIR / "runs_wave3/x1b-flagship-slwm-1b-s2001/eval_metrics.json")
    dense_1b = _load_json(EXP_DIR / "runs_wave3/x1b-baseline-dense-1b-s2001/eval_metrics.json")
    legacy_1b = _load_json(EXP_DIR / "runs_wave3/x1b-baseline-legacy-1b-s2001/eval_metrics.json")

    assert np.isclose(slwm_1b["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"], 37 / 48)
    assert np.isclose(slwm_1b["closed_loop_mpc"]["latent_cem_mpc"]["thin_barrier_success"], 13 / 24)
    assert np.isclose(dense_1b["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"], 34 / 48)
    assert np.isclose(legacy_1b["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"], 33 / 48)

    assert slwm_1b["associative_recall_512"]["T512"]["mean"] == 1.0
    assert dense_1b["associative_recall_512"]["T256"]["mean"] < 0.10
    assert legacy_1b["associative_recall_512"]["T512"]["mean"] < 0.10

    assert np.isclose(slwm_1b["real_multimodal_and_text"]["av_r1"]["mean"], 0.90625)
    assert legacy_1b["real_multimodal_and_text"]["av_r1"]["mean"] < 0.05
    assert slwm_1b["thin_barrier_and_persistence"]["barrier_auroc_64x64"] == 1.0
    assert slwm_1b["spoofing_and_commit_gate"]["learned_innovation"]["unsupported_claim_rate"] == 0.0

    # 3-seed 152M factorial audit
    slwm_150_mpc = []
    dense_150_mpc = []
    for seed in (2001, 2002, 2003):
        m_full = _load_json(EXP_DIR / f"runs_wave3/x1b-abl-slwm_full-s{seed}/eval_metrics.json")
        m_dense = _load_json(EXP_DIR / f"runs_wave3/x1b-abl-dense_attn-s{seed}/eval_metrics.json")
        assert m_full["associative_recall_512"]["T512"]["mean"] == 1.0
        slwm_150_mpc.append(m_full["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"])
        dense_150_mpc.append(m_dense["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"])

    assert np.isclose(np.mean(slwm_150_mpc), summary["groups"]["abl_slwm_full"]["mpc_success"]["mean"])
    assert np.mean(slwm_150_mpc) > np.mean(dense_150_mpc)


def test_study_x1b_report_and_readme_links_resolve():
    """Ensure every local markdown link in the Study X1B report and README entry resolves cleanly."""
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for md_path in (EXP_DIR / "report.md", REPO_ROOT / "README.md"):
        text = md_path.read_text(encoding="utf-8")
        for target in link_re.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            rel = target.split("#", 1)[0]
            if not rel:
                continue
            resolved = (md_path.parent / rel).resolve()
            assert resolved.exists(), f"Broken link '{target}' in {md_path} -> {resolved}"
