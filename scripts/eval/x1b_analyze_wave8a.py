"""Aggregate Study X2 Wave 8A runs, compute Hierarchical Seed CIs, check Gate 4 KPIs, & render Showcase Gallery."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from evals.x1b_showcase_wave8a import render_wave8a_showcase_gallery
from evals.x1b_stats import (
    hierarchical_paired_auroc_bootstrap_ci,
    hierarchical_seed_bootstrap_ci,
)
from models.x1b_wave8a_core import Wave8AInterventions
from models.x1b_world import count_spec_parameters, make_x1b_spec
from scripts.train.x1b_train import sha256_file
from scripts.train.x1b_train_wave8a import (
    compute_wave8a_binding_sha256,
    compute_wave8a_source_code_sha256,
)


def _get_plt():
    if "MPLCONFIGDIR" not in os.environ:
        try:
            os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="slwm_mpl_")
        except OSError:
            pass
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


COHORT_NAMES = (
    "w8a_slwm_x2_isolated_450m",
    "w8a_ablation_no_zmem_iso_450m",
    "w8a_ablation_spatial_control_only_450m",
    "w8a_sliding_window_w64_gqa_450m",
    "w8a_diagonal_ssm_450m",
)
EXPECTED_SEEDS = (2001, 2002, 2003)
OOD_CORRUPTION_KINDS = (
    "frozen_camera_frame",
    "cross_episode_replay_swap",
    "unseen_90deg_rotation",
    "half_occluded_camera",
)
HEADLINE_COHORT_METRICS = (
    "unseen_mppi",
    "no_jepa_subgoal",
    "T160_goal_b_switch",
    "T160_paired_causal_switch",
    "dyn_assoc_T64",
    "dyn_assoc_T256",
    "dyn_assoc_T512",
    "dyn_assoc_T1024",
    "unseen_corruption_auroc",
    "commit_selective_routing",
    "vault_step4_mm",
    "vault_spoken_disc",
    "vault_spoken_audio_only_disc",
    "vault_scene_color_disc",
    "vault_scene_color_audio_only_disc",
    "vault_shuffled_scene_color_disc",
    "vault_shuffled_scene_color_audio_only_disc",
    "vault_color_prior_scene_color_disc",
    "vault_paired_flip_dual_color_disc",
    "vault_paired_flip_dual_audio_only_disc",
)
REQUIRED_SCENE_GROUNDING_CIS = (
    "isolated_scene_vs_shuffled_joint",
    "isolated_scene_vs_shuffled_audio_only",
    "isolated_scene_vs_color_prior_joint",
    "isolated_scene_vs_color_prior_audio_only",
    "isolated_paired_flip_dual_joint",
    "isolated_paired_flip_dual_audio_only",
)
REQUIRED_HIER_COMPS = (
    "w8a_ablation_no_zmem_iso_450m",
    "w8a_ablation_spatial_control_only_450m",
    "w8a_sliding_window_w64_gqa_450m",
    "w8a_diagonal_ssm_450m",
)
REQUIRED_HIER_METRICS = (
    "unseen_mppi",
    "T160_goal_b_switch",
    "T160_paired_causal_switch",
    "dyn_assoc_T1024",
    "K8_T256_overwrite",
    "K16_T1024_overwrite",
    "unseen_corruption_auroc",
    "commit_selective_routing",
    "vault_step4_mm",
    "vault_spoken_disc",
    "vault_spoken_audio_only_disc",
    "vault_scene_color_disc",
    "vault_scene_color_audio_only_disc",
)
REQUIRED_SHOWCASE_FIGURES = (
    "wave8a_p05_to_p95_unseen_mpc_gallery.png",
    "wave8a_cross_cohort_t160_and_k16_showcase.png",
    "wave8a_common_episodes_cross_cohort_gallery.png",
)


def _is_sha256_hex(val: Any) -> bool:
    return isinstance(val, str) and len(val) == 64 and all(ch in "0123456789abcdef" for ch in val)


def verify_receipt_data_provenance(receipt: dict[str, Any], reg_cards: dict[str, Any] | None = None) -> bool:
    """Verify that a run's `data_provenance` matches the frozen registered `wave8a_dataset_cards.json` SHAs (`smoke is False`)."""
    if receipt.get("smoke") is not False:
        return False
    if reg_cards is None:
        cards_p = REPO_ROOT / "configs/x1b/wave8a_dataset_cards.json"
        if not cards_p.is_file():
            return False
        reg_cards = json.loads(cards_p.read_text())
    prov = receipt.get("data_provenance")
    if not isinstance(prov, dict):
        return False
    if prov.get("registered_cards_verified") is not True:
        return False
    checks = (
        ("real_bank_sha256", "x1b_real_bank_v2"),
        ("grounded_robot_sha256", "x1b_wave8a_grounded_v2"),
        ("vault_robot_sha256", "x1b_wave8a_vault_v2"),
    )
    for prov_k, card_k in checks:
        if card_k not in reg_cards:
            return False
        if prov.get(prov_k) != reg_cards[card_k].get("sha256"):
            return False
    return True


def verify_wave8a_bundle_provenance(
    bundle_name: str,
    bundle: dict[str, Any],
    *,
    expected_cohort: str,
    study_cfg: dict[str, Any] | None = None,
    reg_cards: dict[str, Any] | None = None,
    expected_source_sha: str | None = None,
) -> bool:
    """Cryptographically verify a completed Wave 8A run bundle against frozen study config, source SHA, and on-disk digests."""
    if study_cfg is None:
        cfg_p = REPO_ROOT / "configs/x1b/study_x2_wave8a.json"
        if not cfg_p.is_file():
            return False
        study_cfg = json.loads(cfg_p.read_text())
    if reg_cards is None:
        cards_p = REPO_ROOT / "configs/x1b/wave8a_dataset_cards.json"
        if not cards_p.is_file():
            return False
        reg_cards = json.loads(cards_p.read_text())
    if expected_source_sha is None:
        try:
            expected_source_sha = compute_wave8a_source_code_sha256()
        except OSError:
            return False

    cohort_specs = {
        str(c["cohort"]): c
        for c in study_cfg.get("cohorts_453m_3seed", ())
        if isinstance(c, dict) and "cohort" in c
    }
    if expected_cohort not in cohort_specs:
        return False
    c_spec = cohort_specs[expected_cohort]

    local_dir = bundle.get("dir")
    rec_doc = bundle.get("receipt")
    eval_doc = bundle.get("eval")
    if not isinstance(local_dir, Path) or not local_dir.is_dir() or not isinstance(rec_doc, dict) or not isinstance(eval_doc, dict):
        return False

    ver_p = local_dir / "verified.json"
    rec_p = local_dir / "receipt.json"
    ev_p = local_dir / "eval_metrics.json"
    if not ver_p.is_file() or not rec_p.is_file() or not ev_p.is_file():
        return False

    try:
        ver_doc = json.loads(ver_p.read_text())
        disk_rec = json.loads(rec_p.read_text())
        disk_eval = json.loads(ev_p.read_text())
    except (OSError, ValueError):
        return False
    if disk_rec != rec_doc or disk_eval != eval_doc:
        return False

    raw_seed = rec_doc.get("seed")
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        return False
    allowed_seeds = [int(s) for s in c_spec.get("seeds", ())]
    if raw_seed not in allowed_seeds:
        return False
    expected_tag = f"x1b-{expected_cohort}-s{raw_seed}"
    if (
        bundle_name != expected_tag
        or rec_doc.get("tag") != expected_tag
        or rec_doc.get("cohort") != expected_cohort
        or ver_doc.get("tag") != expected_tag
        or int(ver_doc.get("verified_copies", 0)) < 2
    ):
        return False

    sched = study_cfg.get("training_schedule", {})
    exp_profile = str(c_spec["profile"])
    exp_arch = str(c_spec["arch"])
    exp_reg = str(c_spec["regularizer"])
    exp_iv_cfg = c_spec.get("interventions", {})
    exp_steps = int(c_spec.get("steps", sched.get("steps", 1800)))
    exp_bs = int(c_spec.get("batch_size", sched.get("batch_size", 8)))
    exp_seq = int(c_spec.get("seq_len", sched.get("seq_len", 64)))
    exp_save_bf16 = bool(expected_cohort == "w8a_slwm_x2_isolated_450m" and raw_seed == 2001)
    exp_export_showcase = True

    iv = Wave8AInterventions(
        key_shield=bool(exp_iv_cfg.get("key_shield")),
        zerolag_bypass=bool(exp_iv_cfg.get("zerolag_bypass")),
        decoupled_heads=bool(exp_iv_cfg.get("decoupled_heads")),
        zmem_isolation=bool(exp_iv_cfg.get("zmem_isolation")),
        enable_assoc_loss=bool(exp_iv_cfg.get("enable_assoc_loss")),
        regularizer=exp_reg,
    )
    exp_spec = make_x1b_spec(profile=exp_profile, arch=exp_arch)
    exp_strict_params = int(count_spec_parameters(exp_spec)["strict_total"])
    exp_binding_sha = compute_wave8a_binding_sha256(
        tag=expected_tag,
        cohort=expected_cohort,
        profile=exp_profile,
        arch=exp_arch,
        interventions=iv,
        seed=raw_seed,
        steps=exp_steps,
        batch_size=exp_bs,
        seq_len=exp_seq,
        smoke=False,
        save_bf16=exp_save_bf16,
        export_showcase=exp_export_showcase,
    )
    prov = rec_doc.get("data_provenance", {})
    if (
        rec_doc.get("smoke") is not False
        or rec_doc.get("profile") != exp_profile
        or rec_doc.get("arch") != exp_arch
        or rec_doc.get("interventions") != asdict(iv)
        or int(rec_doc.get("steps", -1)) != exp_steps
        or int(rec_doc.get("batch_size", -1)) != exp_bs
        or int(rec_doc.get("seq_len", -1)) != exp_seq
        or rec_doc.get("save_bf16") is not exp_save_bf16
        or rec_doc.get("export_showcase") is not exp_export_showcase
        or int(rec_doc.get("parameter_counts", {}).get("strict_total", -1)) != exp_strict_params
        or rec_doc.get("binding_sha256") != exp_binding_sha
        or not _is_sha256_hex(expected_source_sha)
        or rec_doc.get("source_code_sha256") != expected_source_sha
        or not verify_receipt_data_provenance(rec_doc, reg_cards)
        or int(prov.get("sequence_tensor_budget_ticks", -1)) != (exp_steps * exp_bs * exp_seq)
    ):
        return False

    ver_digests = ver_doc.get("sha256")
    if not isinstance(ver_digests, dict):
        return False

    local_files = (
        ("receipt.json", ver_digests.get("receipt.json")),
        ("eval_metrics.json", rec_doc.get("eval_metrics_sha256")),
        ("config.json", rec_doc.get("config_file_sha256")),
        ("train.jsonl", rec_doc.get("train_log_sha256")),
        ("showcase_bundle.npz", rec_doc.get("showcase_bundle_sha256")),
    )
    for fname, exp_sha in local_files:
        fp = local_dir / fname
        if not fp.is_file() or not _is_sha256_hex(exp_sha):
            return False
        got_sha = sha256_file(fp)
        if got_sha != exp_sha or ver_digests.get(fname) != got_sha:
            return False

    int8_sha = rec_doc.get("portable_int8_sha256")
    if not _is_sha256_hex(int8_sha) or ver_digests.get("portable_int8.npz") != int8_sha:
        return False
    int8_local = local_dir / "portable_int8.npz"
    if int8_local.is_file() and sha256_file(int8_local) != int8_sha:
        return False

    if exp_save_bf16:
        bf16_sha = rec_doc.get("weights_bf16_sha256")
        if not _is_sha256_hex(bf16_sha) or ver_digests.get("weights_bf16.npz") != bf16_sha:
            return False
        bf16_local = local_dir / "weights_bf16.npz"
        if bf16_local.is_file() and sha256_file(bf16_local) != bf16_sha:
            return False
    return True


def load_wave8a_bundles(w8a_dir: Path) -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    if not w8a_dir.is_dir():
        return bundles
    for d in sorted(w8a_dir.iterdir()):
        if d.is_dir() and (d / "receipt.json").is_file() and (d / "eval_metrics.json").is_file():
            bundles[d.name] = {
                "dir": d,
                "receipt": json.loads((d / "receipt.json").read_text()),
                "eval": json.loads((d / "eval_metrics.json").read_text()),
            }
    return bundles


def aggregate_w8a_cohort(
    bundles: dict[str, dict[str, Any]],
    cohort: str,
    *,
    study_cfg: dict[str, Any] | None = None,
    reg_cards: dict[str, Any] | None = None,
    expected_source_sha: str | None = None,
) -> dict[str, Any]:
    pref = f"x1b-{cohort}-s"
    member_items = [(k, v) for k, v in sorted(bundles.items()) if k.startswith(pref)]
    members = [v for _, v in member_items]
    if not members:
        return {}

    def _stat(fn) -> dict[str, Any]:
        vals: list[float] = []
        for m in members:
            raw = fn(m["eval"])
            vals.append(float("nan") if raw is None else float(raw))
        if not vals or not all(np.isfinite(x) for x in vals):
            return {"values": vals, "complete": False}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals,
            "complete": True,
        }

    cap_curves: dict[str, Any] = {}
    for horizon in (64, 256, 1024):
        for k_items in (1, 2, 4, 8, 16):
            key_acc = f"K{k_items}_T{horizon}_acc"
            key_ov = f"K{k_items}_T{horizon}_overwritten_acc"
            st_acc = _stat(lambda e, ka=key_acc: e.get("multi_item_overwrite_capacity", {}).get(ka))
            st_ov = _stat(lambda e, ko=key_ov: e.get("multi_item_overwrite_capacity", {}).get(ko))
            cap_curves[key_acc] = st_acc
            cap_curves[key_ov] = st_ov

    mem_info = members[0]["eval"].get("streaming_state_memory", {})
    seeds_present = [int(m["receipt"]["seed"]) for m in members if "seed" in m.get("receipt", {})]
    source_shas = sorted(
        {
            str(m.get("receipt", {}).get("source_code_sha256", ""))
            for m in members
            if isinstance(m.get("receipt"), dict)
        }
    )
    prov_ok = bool(member_items) and all(
        verify_wave8a_bundle_provenance(
            k,
            v,
            expected_cohort=cohort,
            study_cfg=study_cfg,
            reg_cards=reg_cards,
            expected_source_sha=expected_source_sha,
        )
        for k, v in member_items
    )
    return {
        "n_seeds": len(members),
        "seeds_present": sorted(seeds_present),
        "source_code_sha256s": source_shas,
        "provenance_verified": prov_ok,
        "strict_params": int(members[0]["receipt"]["parameter_counts"]["strict_total"]),
        "streaming_state_mib": float(mem_info.get("total_state_mib", 0.0)),
        "core_state_mib": float(mem_info.get("core_state_mib", mem_info.get("total_state_mib", 0.0))),
        "strict_system_state_mib": float(
            mem_info.get("strict_system_state_mib", mem_info.get("total_state_mib", 0.0))
        ),
        "audio_frontend_state_bytes": int(mem_info.get("audio_frontend_state_bytes", 0)),
        "controller_state_bytes": int(mem_info.get("controller_state_bytes", 0)),
        "decoder_carry_bytes": int(mem_info.get("decoder_carry_bytes", 0)),
        "kv_cache_mib": round(float(mem_info.get("kv_cache_bytes", 0)) / float(1 << 20), 4),
        "recurrent_state_mib": round(float(mem_info.get("recurrent_state_bytes", 0)) / float(1 << 20), 4),
        "compiled_inference_ms_per_tick": _stat(lambda e: e.get("compiled_inference_ms_per_tick")),
        "oracle_mppi_ceiling": _stat(
            lambda e: e["unseen_layout_deprivileged_mpc"].get("oracle_perception_dynamics_mppi_ceiling")
        ),
        "oracle_no_jepa_subgoal_ceiling": _stat(
            lambda e: e["unseen_layout_deprivileged_mpc"].get("oracle_no_jepa_subgoal_ceiling")
        ),
        "oracle_terminal16_hold_ceiling": _stat(
            lambda e: e["unseen_layout_deprivileged_mpc"].get("oracle_terminal16_all_ticks_hold_success")
        ),
        "oracle_any_tick_ceiling": _stat(
            lambda e: e["unseen_layout_deprivileged_mpc"].get("oracle_any_tick_success")
        ),
        "unseen_mppi": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["geometry_free_mppi_success"]),
        "unseen_mppi_terminal16_hold": _stat(
            lambda e: e["unseen_layout_deprivileged_mpc"].get("geometry_free_mppi_terminal16_all_ticks_hold_success")
        ),
        "no_jepa_subgoal": _stat(lambda e: e["unseen_layout_deprivileged_mpc"].get("no_jepa_subgoal_success")),
        "mppi_over_no_jepa_gain_pp": _stat(
            lambda e: e["unseen_layout_deprivileged_mpc"].get("mppi_over_no_jepa_gain_pp")
        ),
        "explicit_latch_reactive": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["explicit_latch_reactive_success"]),
        "open_loop_policy": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["open_loop_policy_success"]),
        "vis_obj_rmse": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["vis_decoded_obj_rmse"]),
        "vis_ee_rmse": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["vis_decoded_ee_rmse"]),
        "unseen_off_center": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["by_unseen_family"]["off_center_wall"]["mppi_success"]),
        "unseen_dual_chicane": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["by_unseen_family"]["dual_chicane"]["mppi_success"]),
        "unseen_multi_clutter": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["by_unseen_family"]["multi_block_clutter"]["mppi_success"]),
        "unseen_dynamics_shift": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["by_unseen_family"]["dynamics_shift"]["mppi_success"]),
        "T160_goal_b_switch": _stat(lambda e: e["beyond_window_T160_distractor_switch"]["T160_post_perturbation_goal_b_success"]),
        "T160_counterfactual_goal_b_prime": _stat(
            lambda e: e["beyond_window_T160_distractor_switch"].get("T160_counterfactual_goal_b_prime_success")
        ),
        "T160_paired_causal_switch": _stat(
            lambda e: e["beyond_window_T160_distractor_switch"].get("T160_paired_causal_switch_rate")
        ),
        "T160_uninstructed_leak": _stat(
            lambda e: e["beyond_window_T160_distractor_switch"].get("T160_uninstructed_goal_b_leak_rate")
        ),
        "T160_stale_goal_a": _stat(lambda e: e["beyond_window_T160_distractor_switch"]["T160_stale_goal_a_capture"]),
        "latch_mse_inside_W64": _stat(lambda e: e["beyond_window_T160_distractor_switch"]["latch_mse_inside_W64_t24_64"]),
        "latch_mse_beyond_W64": _stat(lambda e: e["beyond_window_T160_distractor_switch"]["latch_mse_beyond_W64_t92_160"]),
        "symbolic_key_drift": _stat(lambda e: e["symbolic_key_drift_and_recall"]["symbolic_key_visual_drift"]),
        "dyn_assoc_T64": _stat(lambda e: e["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T64"]),
        "dyn_assoc_T256": _stat(lambda e: e["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T256"]),
        "dyn_assoc_T512": _stat(lambda e: e["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T512"]),
        "dyn_assoc_T1024": _stat(lambda e: e["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"]),
        "unseen_corruption_auroc": _stat(lambda e: e["unseen_ood_sensor_corruptions"]["mean_unseen_ood_corruption_auroc"]),
        "commit_selective_routing": _stat(lambda e: e["unseen_ood_sensor_corruptions"].get("commit_selective_routing_acc")),
        "commit_binary_ece": _stat(lambda e: e["unseen_ood_sensor_corruptions"].get("commit_binary_ece")),
        "ood_abstain_rate": _stat(lambda e: e["unseen_ood_sensor_corruptions"].get("ood_abstain_rate")),
        "ood_fallback_rate": _stat(lambda e: e["unseen_ood_sensor_corruptions"].get("ood_fallback_mitigated_rate")),
        "dev_step4_mm": _stat(lambda e: e["grounded_spoken_robot_dev"].get("step4_mm")),
        "dev_spoken_disc": _stat(lambda e: e["grounded_spoken_robot_dev"].get("spoken_16tok_discrimination_acc")),
        "dev_spoken_audio_only_disc": _stat(
            lambda e: e["grounded_spoken_robot_dev"].get("spoken_audio_only_discrimination_acc")
        ),
        "dev_scene_color_disc": _stat(lambda e: e["grounded_spoken_robot_dev"].get("spoken_scene_color_disc_acc")),
        "dev_scene_color_audio_only_disc": _stat(
            lambda e: e["grounded_spoken_robot_dev"].get("spoken_audio_only_scene_color_disc_acc")
        ),
        "dev_shuffled_scene_color_disc": _stat(
            lambda e: e["grounded_spoken_robot_dev"].get("shuffled_scene_color_disc_acc")
        ),
        "dev_paired_flip_dual_color_disc": _stat(
            lambda e: e["grounded_spoken_robot_dev"].get("paired_flip_dual_scene_color_disc_acc")
        ),
        "dev_text_only_disc": _stat(lambda e: e["grounded_spoken_robot_dev"].get("text_16tok_only_discrimination_acc")),
        "vault_step4_mm": _stat(lambda e: e["untouched_vault_spoken_robot"].get("step4_mm")),
        "vault_plan_gain_mm": _stat(lambda e: e["untouched_vault_spoken_robot"].get("causal_plan_gain_mm")),
        "vault_spoken_disc": _stat(lambda e: e["untouched_vault_spoken_robot"].get("spoken_16tok_discrimination_acc")),
        "vault_spoken_audio_only_disc": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("spoken_audio_only_discrimination_acc")
        ),
        "vault_scene_color_disc": _stat(lambda e: e["untouched_vault_spoken_robot"].get("spoken_scene_color_disc_acc")),
        "vault_scene_color_audio_only_disc": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("spoken_audio_only_scene_color_disc_acc")
        ),
        "vault_shuffled_scene_color_disc": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("shuffled_scene_color_disc_acc")
        ),
        "vault_shuffled_scene_color_audio_only_disc": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("shuffled_scene_color_audio_only_disc_acc")
        ),
        "vault_color_prior_scene_color_disc": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("color_prior_scene_color_disc_acc")
        ),
        "vault_mirrored_180_scene_color_disc": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("mirrored_180_scene_color_disc_acc")
        ),
        "vault_paired_flip_dual_color_disc": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("paired_flip_dual_scene_color_disc_acc")
        ),
        "vault_paired_flip_dual_audio_only_disc": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("paired_flip_dual_audio_only_disc_acc")
        ),
        "vault_blind_video_scene_color_disc": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("blind_video_scene_color_disc_acc")
        ),
        "vault_target_block_norm_rmse": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("target_block_norm_rmse")
        ),
        "vault_target_block_px64_err": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("target_block_px64_err")
        ),
        "vault_text_only_disc": _stat(
            lambda e: e["untouched_vault_spoken_robot"].get("text_16tok_only_discrimination_acc")
        ),
        "multi_item_overwrite": cap_curves,
    }


def _seed_matrix(bundles: dict[str, dict[str, Any]], cohort: str, fn) -> np.ndarray | None:
    pref = f"x1b-{cohort}-s"
    members = [v for k, v in sorted(bundles.items()) if k.startswith(pref)]
    if not members:
        return None
    rows: list[np.ndarray] = []
    for v in members:
        raw = fn(v["eval"])
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        rows.append(arr)
    if len({r.shape for r in rows}) != 1:
        return None
    return np.stack(rows, axis=0)


def _ood_score_tensors(bundles: dict[str, dict[str, Any]], cohort: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract `(seed_corr_scores [S, K, W, T], seed_clean_scores [S, W, T])` across seeds of `cohort`."""
    pref = f"x1b-{cohort}-s"
    members = [v for k, v in sorted(bundles.items()) if k.startswith(pref)]
    if not members:
        return None
    corr_list: list[np.ndarray] = []
    clean_list: list[np.ndarray] = []
    for m in members:
        ood = m["eval"].get("unseen_ood_sensor_corruptions", {})
        cl = ood.get("clean_world_scores")
        co_map = ood.get("corrupted_world_scores")
        if not isinstance(cl, list) or not isinstance(co_map, dict):
            return None
        if not all(k in co_map for k in OOD_CORRUPTION_KINDS):
            return None
        cl_arr = np.asarray(cl, dtype=np.float64)
        co_arr = np.stack([np.asarray(co_map[k], dtype=np.float64) for k in OOD_CORRUPTION_KINDS], axis=0)
        if not (np.all(np.isfinite(cl_arr)) and np.all(np.isfinite(co_arr))):
            return None
        clean_list.append(cl_arr)
        corr_list.append(co_arr)
    if len({a.shape for a in clean_list}) != 1 or len({a.shape for a in corr_list}) != 1:
        return None
    return np.stack(corr_list, axis=0), np.stack(clean_list, axis=0)


def _verified_vault_cluster_ids(
    bundles: dict[str, dict[str, Any]],
    cohort_a: str,
    cohort_b: str,
    expected_len: int,
) -> list[int]:
    """Extract & verify identical, non-trivial recording-session cluster IDs across every seed of both cohorts."""
    canonical: list[int] | None = None
    for c in (cohort_a, cohort_b):
        pref = f"x1b-{c}-s"
        members = [v for k, v in sorted(bundles.items()) if k.startswith(pref)]
        if not members:
            raise ValueError(f"Missing evaluated runs for cohort {c!r} during vault cluster verification")
        for m in members:
            cids = m["eval"].get("untouched_vault_spoken_robot", {}).get("scene_cluster_ids")
            if not isinstance(cids, list) or len(cids) != expected_len:
                raise ValueError(
                    f"Missing or length-mismatched vault scene_cluster_ids in {c!r}: got {None if cids is None else len(cids)}, expected {expected_len}"
                )
            int_cids = [int(x) for x in cids]
            n_uniq = len(set(int_cids))
            if n_uniq <= 1 or n_uniq >= expected_len:
                raise ValueError(
                    f"Invalid trivial vault scene_cluster_ids in {c!r}: {n_uniq} unique clusters across {expected_len} episodes"
                )
            if canonical is None:
                canonical = int_cids
            elif int_cids != canonical:
                raise ValueError(f"Vault scene_cluster_ids mismatch across seeds/cohorts ({cohort_a!r} vs {cohort_b!r})")
    assert canonical is not None
    return canonical


def _verified_vault_shuffled_dyad_ids(
    bundles: dict[str, dict[str, Any]],
    cohort: str,
    expected_len: int,
    scene_cids: list[int],
) -> list[int]:
    """Fail-closed verification of donor-recipient-closed supercluster IDs for disjoint-scene shuffle CIs."""
    pref = f"x1b-{cohort}-s"
    members = [v for k, v in sorted(bundles.items()) if k.startswith(pref)]
    if not members:
        raise ValueError(f"Missing evaluated runs for cohort {cohort!r} during shuffled dyad verification")
    if len(scene_cids) != expected_len:
        raise ValueError(f"scene_cids length {len(scene_cids)} != expected_len {expected_len}")

    canonical_dyads: list[int] | None = None
    canonical_donors: list[int] | None = None
    for m in members:
        vlt = m["eval"].get("untouched_vault_spoken_robot", {})
        dyads = vlt.get("shuffled_dyad_cluster_ids")
        donors = vlt.get("shuffled_donor_indices")
        if not isinstance(dyads, list) or len(dyads) != expected_len:
            raise ValueError(
                f"Missing or length-mismatched shuffled_dyad_cluster_ids in {cohort!r}: "
                f"got {None if dyads is None else len(dyads)}, expected {expected_len}"
            )
        if not isinstance(donors, list) or len(donors) != expected_len:
            raise ValueError(
                f"Missing or length-mismatched shuffled_donor_indices in {cohort!r}: "
                f"got {None if donors is None else len(donors)}, expected {expected_len}"
            )
        int_dyads = [int(x) for x in dyads]
        int_donors = [int(x) for x in donors]
        if canonical_dyads is None:
            canonical_dyads = int_dyads
            canonical_donors = int_donors
        elif int_dyads != canonical_dyads or int_donors != canonical_donors:
            raise ValueError(f"Vault shuffled_dyad_cluster_ids or shuffled_donor_indices mismatch across seeds in {cohort!r}")

    assert canonical_dyads is not None and canonical_donors is not None
    n_scene_uniq = len(set(scene_cids))
    n_dyad_uniq = len(set(canonical_dyads))
    if not (1 < n_dyad_uniq < n_scene_uniq):
        raise ValueError(
            f"Invalid non-coarsened shuffled_dyad_cluster_ids in {cohort!r}: "
            f"n_dyad_uniq={n_dyad_uniq}, n_scene_uniq={n_scene_uniq}"
        )
    if len({(scene_cids[i], canonical_dyads[i]) for i in range(expected_len)}) != n_scene_uniq:
        raise ValueError(f"shuffled_dyad_cluster_ids in {cohort!r} splits episodes of a single scene_cluster_id across dyads")

    for i in range(expected_len):
        don_i = canonical_donors[i]
        if not (0 <= don_i < expected_len):
            raise ValueError(f"Out-of-bounds shuffled_donor_indices[{i}]={don_i} in {cohort!r}")
        if scene_cids[don_i] == scene_cids[i]:
            raise ValueError(
                f"Shuffled donor {don_i} shares scene_cluster_id {scene_cids[i]} with recipient {i} in {cohort!r}"
            )
        if canonical_dyads[don_i] != canonical_dyads[i]:
            raise ValueError(
                f"Donor-recipient supercluster closure violated at episode {i} -> donor {don_i} in {cohort!r}: "
                f"dyad {canonical_dyads[i]} != {canonical_dyads[don_i]}"
            )
    return canonical_dyads


def compute_wave8a_hierarchical_attributions(bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ref_c = "w8a_slwm_x2_isolated_450m"
    vec_specs = (
        ("unseen_mppi", lambda e: e["unseen_layout_deprivileged_mpc"]["geometry_free_mppi_succ_vec"]),
        ("T160_goal_b_switch", lambda e: e["beyond_window_T160_distractor_switch"]["T160_succ_vec"]),
        ("T160_paired_causal_switch", lambda e: e["beyond_window_T160_distractor_switch"].get("T160_paired_causal_switch_vec")),
        ("dyn_assoc_T1024", lambda e: e["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024_vec"]),
        ("K8_T256_overwrite", lambda e: e["multi_item_overwrite_capacity"].get("K8_T256_overwritten_vec")),
        ("K16_T1024_overwrite", lambda e: e["multi_item_overwrite_capacity"].get("K16_T1024_overwritten_vec")),
        ("commit_selective_routing", lambda e: e["unseen_ood_sensor_corruptions"].get("commit_selective_routing_vec")),
        ("vault_step4_mm", lambda e: e["untouched_vault_spoken_robot"].get("step4_mm_vec")),
        ("vault_spoken_disc", lambda e: e["untouched_vault_spoken_robot"].get("disc_vec")),
        ("vault_spoken_audio_only_disc", lambda e: e["untouched_vault_spoken_robot"].get("disc_audio_only_vec")),
        ("vault_scene_color_disc", lambda e: e["untouched_vault_spoken_robot"].get("scene_color_disc_vec")),
        ("vault_scene_color_audio_only_disc", lambda e: e["untouched_vault_spoken_robot"].get("scene_color_audio_only_vec")),
    )
    out: dict[str, Any] = {}
    ood_ref = _ood_score_tensors(bundles, ref_c)

    for comp in REQUIRED_HIER_COMPS:
        for mname, fn in vec_specs:
            a = _seed_matrix(bundles, ref_c, fn)
            b = _seed_matrix(bundles, comp, fn)
            if a is not None and b is not None and a.shape == b.shape:
                cids = _verified_vault_cluster_ids(bundles, ref_c, comp, a.shape[1]) if mname.startswith("vault_") else None
                out[f"isolated_vs_{comp}_{mname}"] = hierarchical_seed_bootstrap_ci(a, b, cluster_ids=cids)

        ood_comp = _ood_score_tensors(bundles, comp)
        if ood_ref is not None and ood_comp is not None and ood_ref[0].shape == ood_comp[0].shape:
            out[f"isolated_vs_{comp}_unseen_corruption_auroc"] = hierarchical_paired_auroc_bootstrap_ci(
                ood_ref[0], ood_ref[1], ood_comp[0], ood_comp[1]
            )

    sc_joint = _seed_matrix(bundles, ref_c, lambda e: e["untouched_vault_spoken_robot"].get("scene_color_disc_vec"))
    sc_aud = _seed_matrix(
        bundles, ref_c, lambda e: e["untouched_vault_spoken_robot"].get("scene_color_audio_only_vec")
    )
    shuf_joint = _seed_matrix(
        bundles, ref_c, lambda e: e["untouched_vault_spoken_robot"].get("shuffled_scene_color_disc_vec")
    )
    shuf_aud = _seed_matrix(
        bundles, ref_c, lambda e: e["untouched_vault_spoken_robot"].get("shuffled_scene_color_audio_only_vec")
    )
    prior_vec = _seed_matrix(
        bundles, ref_c, lambda e: e["untouched_vault_spoken_robot"].get("color_prior_scene_color_vec")
    )
    pflip_joint = _seed_matrix(
        bundles, ref_c, lambda e: e["untouched_vault_spoken_robot"].get("paired_flip_dual_scene_color_vec")
    )
    pflip_aud = _seed_matrix(
        bundles, ref_c, lambda e: e["untouched_vault_spoken_robot"].get("paired_flip_dual_audio_only_vec")
    )
    if sc_joint is not None:
        iso_cids = _verified_vault_cluster_ids(bundles, ref_c, ref_c, sc_joint.shape[1])
        shuf_cids = _verified_vault_shuffled_dyad_ids(bundles, ref_c, sc_joint.shape[1], iso_cids)
        if shuf_joint is None or sc_joint.shape != shuf_joint.shape:
            raise ValueError(f"Missing or mismatched shuffled_scene_color_disc_vec in {ref_c!r}")
        ci_shuf_j = hierarchical_seed_bootstrap_ci(sc_joint, shuf_joint, cluster_ids=shuf_cids)
        ci_shuf_j["supercluster_donor_closed"] = True
        ci_shuf_j["n_scene_clusters"] = len(set(iso_cids))
        out["isolated_scene_vs_shuffled_joint"] = ci_shuf_j

        if sc_aud is None or shuf_aud is None or sc_aud.shape != shuf_aud.shape or sc_aud.shape != sc_joint.shape:
            raise ValueError(f"Missing or mismatched shuffled_scene_color_audio_only_vec in {ref_c!r}")
        ci_shuf_a = hierarchical_seed_bootstrap_ci(sc_aud, shuf_aud, cluster_ids=shuf_cids)
        ci_shuf_a["supercluster_donor_closed"] = True
        ci_shuf_a["n_scene_clusters"] = len(set(iso_cids))
        out["isolated_scene_vs_shuffled_audio_only"] = ci_shuf_a

        if prior_vec is not None and sc_joint.shape == prior_vec.shape:
            out["isolated_scene_vs_color_prior_joint"] = hierarchical_seed_bootstrap_ci(
                sc_joint, prior_vec, cluster_ids=iso_cids
            )
        if sc_aud is not None and prior_vec is not None and sc_aud.shape == prior_vec.shape:
            out["isolated_scene_vs_color_prior_audio_only"] = hierarchical_seed_bootstrap_ci(
                sc_aud, prior_vec, cluster_ids=iso_cids
            )
        if pflip_joint is not None and pflip_joint.shape == sc_joint.shape:
            out["isolated_paired_flip_dual_joint"] = hierarchical_seed_bootstrap_ci(
                pflip_joint, np.zeros_like(pflip_joint), cluster_ids=iso_cids
            )
        if pflip_aud is not None and pflip_aud.shape == sc_joint.shape:
            out["isolated_paired_flip_dual_audio_only"] = hierarchical_seed_bootstrap_ci(
                pflip_aud, np.zeros_like(pflip_aud), cluster_ids=iso_cids
            )
    return out


def select_median_seed_showcase_paths(
    bundles: dict[str, dict[str, Any]],
    w8a_dir: Path,
    *,
    strict: bool = True,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
) -> dict[str, Path]:
    """Deterministically select the true median seed (ranked by `unseen_mppi`) per cohort without silent fallback."""
    del w8a_dir
    if not bundles:
        if strict:
            raise FileNotFoundError("Cannot select median-seed showcase bundles from an empty bundle dictionary.")
        return {}

    exp_sorted = sorted(int(s) for s in expected_seeds)
    sc_paths: dict[str, Path] = {}
    for c in COHORT_NAMES:
        pref = f"x1b-{c}-s"
        cands: list[tuple[float, int, Path, Any]] = []
        for k, v in sorted(bundles.items()):
            if k.startswith(pref):
                raw_sc = v["eval"].get("unseen_layout_deprivileged_mpc", {}).get("geometry_free_mppi_success")
                score = float("nan") if raw_sc is None else float(raw_sc)
                seed_num = int(v["receipt"]["seed"])
                sc_p = v["dir"] / "showcase_bundle.npz"
                cands.append((score, seed_num, sc_p, v["receipt"].get("showcase_bundle_sha256")))
        if not cands:
            if strict:
                raise FileNotFoundError(f"No evaluated runs found for required cohort {c!r}")
            continue
        if strict:
            got_seeds = sorted(t[1] for t in cands)
            if got_seeds != exp_sorted:
                raise ValueError(
                    f"Cohort {c!r} seeds {got_seeds} do not match exact registered seeds {exp_sorted}"
                )
            if not all(np.isfinite(t[0]) for t in cands):
                raise ValueError(f"Cohort {c!r} contains non-finite geometry_free_mppi_success in {[t[0] for t in cands]}")
        cands.sort(key=lambda t: (t[0], t[1]))
        _, med_seed, med_sc_p, exp_sc_sha = cands[len(cands) // 2]
        if not med_sc_p.is_file():
            if strict:
                raise FileNotFoundError(
                    f"True median seed {med_seed} of cohort {c!r} is missing showcase bundle: {med_sc_p}"
                )
            continue
        if exp_sc_sha is not None and strict:
            if not _is_sha256_hex(exp_sc_sha) or sha256_file(med_sc_p) != exp_sc_sha:
                raise ValueError(
                    f"Showcase bundle SHA-256 mismatch on median seed {med_seed} of cohort {c!r}: {med_sc_p}"
                )
        sc_paths[c] = med_sc_p
    return sc_paths


def evaluate_gate4_checks(
    groups: dict[str, dict[str, Any]],
    hier: dict[str, Any] | None = None,
    showcase_manifest: dict[str, Any] | None = None,
    *,
    min_seeds: int = 3,
    expected_seeds: tuple[int, ...] | None = EXPECTED_SEEDS,
) -> dict[str, Any]:
    """Fail-closed Gate 4 evaluator: enforces registered seeds, provenance, finite values, valid CIs, & showcase."""
    iso = groups.get("w8a_slwm_x2_isolated_450m", {})
    ctrl = groups.get("w8a_ablation_spatial_control_only_450m", {})
    if not iso or not ctrl:
        return {"evaluated": False, "gate4_overall_pass": False, "reason": "missing_required_cohort"}

    exp_sorted = sorted(int(s) for s in expected_seeds) if expected_seeds is not None else None
    req_n = len(exp_sorted) if exp_sorted is not None else int(min_seeds)
    missing_seed_cohorts: list[str] = []
    unverified_provenance_cohorts: list[str] = []
    cohort_source_shas: set[str] = set()
    source_sha_valid = True

    for c in COHORT_NAMES:
        g_c = groups.get(c, {})
        n_s = int(g_c.get("n_seeds", 0))
        s_pres = g_c.get("seeds_present")
        if not isinstance(s_pres, list) or not all(isinstance(x, (int, np.integer)) for x in s_pres):
            seeds_ok = False
        elif exp_sorted is not None:
            seeds_ok = (len(s_pres) == req_n) and (sorted(int(x) for x in s_pres) == exp_sorted)
        else:
            seeds_ok = (len(s_pres) >= req_n) and (len(set(int(x) for x in s_pres)) == len(s_pres))
        if n_s != req_n or not seeds_ok:
            missing_seed_cohorts.append(c)
        src_list = g_c.get("source_code_sha256s")
        if not isinstance(src_list, list) or len(src_list) != 1 or not _is_sha256_hex(src_list[0]):
            source_sha_valid = False
        else:
            cohort_source_shas.add(src_list[0])
        if g_c.get("provenance_verified") is not True:
            unverified_provenance_cohorts.append(c)

    all_source_sha_uniform = bool(source_sha_valid and len(cohort_source_shas) == 1)
    if not all_source_sha_uniform:
        for c in COHORT_NAMES:
            if c not in unverified_provenance_cohorts:
                unverified_provenance_cohorts.append(c)

    all_cohorts_complete = len(missing_seed_cohorts) == 0
    all_provenance_verified = len(unverified_provenance_cohorts) == 0 and all_source_sha_uniform

    def _valid_stat_mean(d: Any) -> float | None:
        if not isinstance(d, dict) or "mean" not in d:
            return None
        vals = d.get("values")
        if not isinstance(vals, (list, tuple)) or len(vals) != req_n:
            return None
        try:
            f_vals = [float(x) for x in vals]
            mean_v = float(d["mean"])
        except (TypeError, ValueError):
            return None
        if not (np.isfinite(mean_v) and all(np.isfinite(x) for x in f_vals)):
            return None
        if abs(float(np.mean(f_vals)) - mean_v) > 1e-5:
            return None
        return mean_v

    def _m(g: dict[str, Any], k: str) -> float | None:
        return _valid_stat_mean(g.get(k))

    missing_cohort_metrics: list[str] = []
    for c in COHORT_NAMES:
        g_c = groups.get(c, {})
        for m_k in HEADLINE_COHORT_METRICS:
            if _m(g_c, m_k) is None:
                missing_cohort_metrics.append(f"{c}.{m_k}")
        mi_c = g_c.get("multi_item_overwrite", {})
        for hz in (64, 256, 1024):
            for k_it in (1, 2, 4, 8, 16):
                for sfx in ("acc", "overwritten_acc"):
                    ckey = f"K{k_it}_T{hz}_{sfx}"
                    if _valid_stat_mean(mi_c.get(ckey)) is None:
                        missing_cohort_metrics.append(f"{c}.multi_item_overwrite.{ckey}")
    all_cohort_metrics_complete = len(missing_cohort_metrics) == 0

    def _hier_expected_mean(g_a: dict[str, Any], g_b: dict[str, Any], mname: str) -> float | None:
        if mname == "K8_T256_overwrite":
            va = _valid_stat_mean(g_a.get("multi_item_overwrite", {}).get("K8_T256_overwritten_acc"))
            vb = _valid_stat_mean(g_b.get("multi_item_overwrite", {}).get("K8_T256_overwritten_acc"))
        elif mname == "K16_T1024_overwrite":
            va = _valid_stat_mean(g_a.get("multi_item_overwrite", {}).get("K16_T1024_overwritten_acc"))
            vb = _valid_stat_mean(g_b.get("multi_item_overwrite", {}).get("K16_T1024_overwritten_acc"))
        else:
            va = _m(g_a, mname)
            vb = _m(g_b, mname)
        return (va - vb) if (va is not None and vb is not None) else None

    invalid_hier_cis: list[str] = []
    hier_dict = hier if isinstance(hier, dict) else {}
    for comp in REQUIRED_HIER_COMPS:
        g_comp = groups.get(comp, {})
        for mname in REQUIRED_HIER_METRICS:
            hkey = f"isolated_vs_{comp}_{mname}"
            hval = hier_dict.get(hkey)
            if not isinstance(hval, dict) or "mean" not in hval:
                invalid_hier_cis.append(hkey)
                continue
            ci95 = hval.get("hierarchical_ci95")
            if not isinstance(ci95, (list, tuple)) or len(ci95) != 2:
                invalid_hier_cis.append(hkey)
                continue
            try:
                lo, hi = float(ci95[0]), float(ci95[1])
                h_mean = float(hval["mean"])
                h_n_seeds = int(hval.get("n_seeds", 0))
            except (TypeError, ValueError):
                invalid_hier_cis.append(hkey)
                continue
            exp_diff = _hier_expected_mean(iso, g_comp, mname)
            bounds_ok = (
                np.isfinite(lo)
                and np.isfinite(hi)
                and np.isfinite(h_mean)
                and (lo <= hi)
                and (lo <= h_mean + 1e-6)
                and (h_mean - 1e-6 <= hi)
                and (h_n_seeds == req_n)
                and (exp_diff is not None)
                and (abs(h_mean - exp_diff) <= 1e-4)
            )
            if not bounds_ok:
                invalid_hier_cis.append(hkey)
                continue
            if mname.startswith("vault_"):
                n_cl = int(hval.get("n_level2_clusters", 0))
                n_ep = int(hval.get("n_episodes_per_seed", 0))
                if hval.get("scene_clustered") is not True or not (1 < n_cl < n_ep):
                    invalid_hier_cis.append(hkey)
                    continue
            if mname == "unseen_corruption_auroc" and hval.get("paired_world_auroc_recomputed") is not True:
                invalid_hier_cis.append(hkey)
                continue

    scene_grounding_spec = (
        ("isolated_scene_vs_shuffled_joint", "vault_scene_color_disc", "vault_shuffled_scene_color_disc", 0.10),
        (
            "isolated_scene_vs_shuffled_audio_only",
            "vault_scene_color_audio_only_disc",
            "vault_shuffled_scene_color_audio_only_disc",
            0.10,
        ),
        ("isolated_scene_vs_color_prior_joint", "vault_scene_color_disc", "vault_color_prior_scene_color_disc", 0.10),
        (
            "isolated_scene_vs_color_prior_audio_only",
            "vault_scene_color_audio_only_disc",
            "vault_color_prior_scene_color_disc",
            0.10,
        ),
        ("isolated_paired_flip_dual_joint", "vault_paired_flip_dual_color_disc", None, 0.25),
        ("isolated_paired_flip_dual_audio_only", "vault_paired_flip_dual_audio_only_disc", None, 0.20),
    )
    scene_grounding_checks: dict[str, Any] = {}
    scene_grounding_pass_all = True
    for sg_key, pos_k, neg_k, min_margin in scene_grounding_spec:
        hval = hier_dict.get(sg_key)
        v_pos = _m(iso, pos_k)
        v_neg = 0.0 if neg_k is None else _m(iso, neg_k)
        exp_sg = (v_pos - v_neg) if (v_pos is not None and v_neg is not None) else None
        ci_valid = False
        lo_sg, hi_sg = float("nan"), float("nan")
        if isinstance(hval, dict) and "mean" in hval and isinstance(hval.get("hierarchical_ci95"), (list, tuple)) and len(hval["hierarchical_ci95"]) == 2:
            try:
                lo_sg, hi_sg = float(hval["hierarchical_ci95"][0]), float(hval["hierarchical_ci95"][1])
                hm_sg = float(hval["mean"])
                ns_sg = int(hval.get("n_seeds", 0))
                n_cl_sg = int(hval.get("n_level2_clusters", 0))
                n_ep_sg = int(hval.get("n_episodes_per_seed", 0))
                needs_dyad_closure = sg_key in (
                    "isolated_scene_vs_shuffled_joint",
                    "isolated_scene_vs_shuffled_audio_only",
                )
                ci_valid = bool(
                    np.isfinite(lo_sg)
                    and np.isfinite(hi_sg)
                    and np.isfinite(hm_sg)
                    and (lo_sg <= hi_sg)
                    and (lo_sg <= hm_sg + 1e-6)
                    and (hm_sg - 1e-6 <= hi_sg)
                    and (ns_sg == req_n)
                    and (exp_sg is not None)
                    and (abs(hm_sg - exp_sg) <= 1e-4)
                    and (hval.get("scene_clustered") is True)
                    and (1 < n_cl_sg < n_ep_sg)
                    and (not needs_dyad_closure or (hval.get("supercluster_donor_closed") is True))
                )
            except (TypeError, ValueError):
                ci_valid = False
        if not ci_valid:
            invalid_hier_cis.append(sg_key)
            scene_grounding_pass_all = False
            scene_grounding_checks[sg_key] = {
                "margin_or_score": exp_sg,
                "min_required": min_margin,
                "hierarchical_ci95": None,
                "pass": False,
            }
        else:
            assert exp_sg is not None
            sg_ok = bool(exp_sg >= (min_margin - 1e-6) and lo_sg > 0.0)
            scene_grounding_pass_all = scene_grounding_pass_all and sg_ok
            scene_grounding_checks[sg_key] = {
                "margin_or_score": round(exp_sg, 4),
                "min_required": min_margin,
                "hierarchical_ci95": [lo_sg, hi_sg],
                "pass": sg_ok,
            }

    all_hier_complete = len(invalid_hier_cis) == 0

    sc_dict = showcase_manifest if isinstance(showcase_manifest, dict) else {}
    showcase_complete = bool(
        sc_dict.get("rendered") is True
        and sc_dict.get("preregistration_verified") is True
        and set(sc_dict.get("cohorts_included", ())) == set(COHORT_NAMES)
        and set(REQUIRED_SHOWCASE_FIGURES).issubset(set(sc_dict.get("artifacts", ())))
    )

    t1024 = _m(iso, "dyn_assoc_T1024")
    t160_paired = _m(iso, "T160_paired_causal_switch")
    unseen_mppi = _m(iso, "unseen_mppi")
    no_jepa_iso = _m(iso, "no_jepa_subgoal")
    commit_sr = _m(iso, "commit_selective_routing")
    aud_disc = _m(iso, "vault_spoken_audio_only_disc")
    unseen_mppi_ge_65 = bool(unseen_mppi is not None and unseen_mppi >= 0.65)
    mppi_vs_nojepa_ok = bool(
        unseen_mppi is not None and no_jepa_iso is not None and unseen_mppi >= (no_jepa_iso - 0.0200001)
    )
    commit_sr_ge_80 = bool(commit_sr is not None and commit_sr >= 0.80)
    aud_disc_ge_70 = bool(aud_disc is not None and aud_disc >= 0.70)
    checks: dict[str, Any] = {
        "evaluated": True,
        "min_seeds_required": req_n,
        "expected_seeds": exp_sorted,
        "all_cohorts_seed_complete": bool(all_cohorts_complete),
        "incomplete_seed_cohorts": missing_seed_cohorts,
        "all_provenance_verified": bool(all_provenance_verified),
        "all_source_code_sha_uniform": bool(all_source_sha_uniform),
        "unverified_provenance_cohorts": unverified_provenance_cohorts,
        "all_cohort_metrics_complete": bool(all_cohort_metrics_complete),
        "missing_cohort_metrics": missing_cohort_metrics[:25],
        "missing_cohort_metrics_count": len(missing_cohort_metrics),
        "all_hierarchical_cis_complete": bool(all_hier_complete),
        "missing_hierarchical_cis": invalid_hier_cis,
        "showcase_gallery_complete": showcase_complete,
        "unseen_mppi_mean": unseen_mppi,
        "no_jepa_subgoal_mean": no_jepa_iso,
        "unseen_mppi_pass_ge_65pct": unseen_mppi_ge_65,
        "mppi_non_inferior_to_no_jepa_subgoal": mppi_vs_nojepa_ok,
        "commit_selective_routing_mean": commit_sr,
        "commit_selective_routing_pass_ge_80pct": commit_sr_ge_80,
        "vault_spoken_audio_only_disc_mean": aud_disc,
        "vault_spoken_audio_only_pass_ge_70pct": aud_disc_ge_70,
        "dyn_assoc_T1024_mean": t1024,
        "dyn_assoc_T1024_pass_ge_95pct": bool(t1024 is not None and t1024 >= 0.95),
        "T160_paired_causal_switch_mean": t160_paired,
        "T160_paired_causal_pass_ge_50pct": bool(t160_paired is not None and t160_paired >= 0.50),
        "scene_grounding_vs_prior_and_shuffle_pass": bool(scene_grounding_pass_all),
        "scene_grounding_checks": scene_grounding_checks,
        "non_inferiority_vs_spatial_only": {},
    }
    ni_pass_all = True
    required_ni_metrics = (
        "unseen_mppi",
        "T160_goal_b_switch",
        "T160_paired_causal_switch",
        "unseen_corruption_auroc",
        "commit_selective_routing",
        "vault_spoken_disc",
        "vault_spoken_audio_only_disc",
        "vault_scene_color_disc",
        "vault_scene_color_audio_only_disc",
    )
    for m_key in required_ni_metrics:
        v_iso = _m(iso, m_key)
        v_ctrl = _m(ctrl, m_key)
        hier_key = f"isolated_vs_w8a_ablation_spatial_control_only_450m_{m_key}"
        ci_info = hier_dict.get(hier_key)
        ci95 = ci_info.get("hierarchical_ci95") if (isinstance(ci_info, dict) and hier_key not in invalid_hier_cis) else None
        if v_iso is None or v_ctrl is None or ci95 is None:
            ni_pass_all = False
            checks["non_inferiority_vs_spatial_only"][m_key] = {
                "isolated_mean": v_iso,
                "spatial_only_mean": v_ctrl,
                "delta_pp": None,
                "hierarchical_ci95": ci95,
                "missing_or_incomplete_seeds": True,
                "pass_le_2pp_drop": False,
            }
        else:
            diff = v_iso - v_ctrl
            ok = diff >= -0.0200001
            ni_pass_all = ni_pass_all and ok
            checks["non_inferiority_vs_spatial_only"][m_key] = {
                "isolated_mean": round(v_iso, 4),
                "spatial_only_mean": round(v_ctrl, 4),
                "delta_pp": round(diff * 100.0, 2),
                "hierarchical_ci95": [float(ci95[0]), float(ci95[1])],
                "missing_or_incomplete_seeds": False,
                "pass_le_2pp_drop": bool(ok),
            }
    checks["gate4_overall_pass"] = bool(
        all_cohorts_complete
        and all_provenance_verified
        and all_cohort_metrics_complete
        and all_hier_complete
        and showcase_complete
        and unseen_mppi_ge_65
        and mppi_vs_nojepa_ok
        and commit_sr_ge_80
        and aud_disc_ge_70
        and checks["dyn_assoc_T1024_pass_ge_95pct"]
        and checks["T160_paired_causal_pass_ge_50pct"]
        and scene_grounding_pass_all
        and ni_pass_all
    )
    return checks


def render_wave8a_figures(
    groups: dict[str, dict[str, Any]],
    fig_dir: Path,
    *,
    fig_prefix: str = "x1b_wave8a",
) -> list[Path]:
    def _mib_tag(ck: str, default_mib: float) -> str:
        g = groups.get(ck, {})
        val = float(g.get("strict_system_state_mib", g.get("streaming_state_mib", default_mib)) or default_mib)
        return f"{val:.2f} MiB"

    order = [
        (f"SLWM-X2\n(Z_mem, {_mib_tag('w8a_slwm_x2_isolated_450m', 8.85)})", "w8a_slwm_x2_isolated_450m", "#117A65"),
        ("No-Zmem-Iso\n(Shared Trunk)", "w8a_ablation_no_zmem_iso_450m", "#C0392B"),
        ("Spatial-Only\n(No Assoc CE)", "w8a_ablation_spatial_control_only_450m", "#2874A6"),
        (
            f"SWin-W64-GQA\n(9:1, {_mib_tag('w8a_sliding_window_w64_gqa_450m', 13.59)})",
            "w8a_sliding_window_w64_gqa_450m",
            "#D35400",
        ),
        (
            f"DiagonalSSM\n(Vec, {_mib_tag('w8a_diagonal_ssm_450m', 4.98)})",
            "w8a_diagonal_ssm_450m",
            "#8E44AD",
        ),
    ]
    pres = [(l, k, c) for l, k, c in order if k in groups and groups[k]]
    if not pres:
        return []

    plt = _get_plt()
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9.2,
            "axes.titlesize": 10.2,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "figure.dpi": 170,
        }
    )
    eb = {"capsize": 3.0, "elinewidth": 1.1, "ecolor": "#1B2631"}
    saved: list[Path] = []

    def _ms(gk: str, mk: str, sc: float = 100.0) -> tuple[float, float]:
        d = groups.get(gk, {}).get(mk, {})
        return (float(d.get("mean", 0.0)) * sc, float(d.get("std", 0.0)) * sc) if "mean" in d else (0.0, 0.0)

    xp = np.arange(len(pres))
    lbls = [p[0] for p in pres]
    cols = [p[2] for p in pres]

    # Figure 11: Joint Recall Cure + Spatial Control Non-Inferiority
    fig1, axes1 = plt.subplots(1, 3, figsize=(16.8, 5.2))

    mp_m, mp_s = zip(*[_ms(k, "unseen_mppi") for _, k, _ in pres])
    sw_m, sw_s = zip(*[_ms(k, "T160_paired_causal_switch") for _, k, _ in pres])
    axes1[0].bar(xp - 0.18, mp_m, 0.35, yerr=mp_s, error_kw=eb, color="#117A65", edgecolor="black", label="Unseen-Layout MPPI (%)")
    axes1[0].bar(xp + 0.18, sw_m, 0.35, yerr=sw_s, error_kw=eb, color="#2E86C1", edgecolor="black", label="T=160 Paired Causal Switch (%)")
    axes1[0].set_xticks(xp)
    axes1[0].set_xticklabels(lbls, fontsize=7.6, rotation=12, ha="right")
    axes1[0].set_ylim(0, 118)
    axes1[0].set_ylabel("Closed-Loop Success (%)")
    axes1[0].set_title("A. Spatial Control & T=160 Causal Switch")
    axes1[0].legend(fontsize=7.4, loc="upper right", framealpha=0.92)

    r64_m, r64_s = zip(*[_ms(k, "dyn_assoc_T64") for _, k, _ in pres])
    r1k_m, r1k_s = zip(*[_ms(k, "dyn_assoc_T1024") for _, k, _ in pres])
    axes1[1].bar(xp - 0.18, r64_m, 0.35, yerr=r64_s, error_kw=eb, color=cols, alpha=0.52, edgecolor="black", label="Dynamic-Visual Recall T=64 (%)")
    axes1[1].bar(xp + 0.18, r1k_m, 0.35, yerr=r1k_s, error_kw=eb, color=cols, hatch="//", edgecolor="black", label="Dynamic-Visual Recall T=1024 (%)")
    axes1[1].axhline(95.0, color="#117A65", linestyle="--", linewidth=1.1, label="Gate 4 Target (95%)")
    axes1[1].set_xticks(xp)
    axes1[1].set_xticklabels(lbls, fontsize=7.6, rotation=12, ha="right")
    axes1[1].set_ylim(0, 122)
    axes1[1].set_ylabel("Exact Overwrite Recall (%)")
    axes1[1].set_title("B. Dynamic-Visual Associative Overwrite (T=64 vs T=1024)")
    axes1[1].legend(fontsize=7.2, loc="upper right", framealpha=0.92)

    au_m, au_s = zip(*[_ms(k, "commit_selective_routing") for _, k, _ in pres])
    vd_m, vd_s = zip(*[_ms(k, "vault_spoken_audio_only_disc") for _, k, _ in pres])
    axes1[2].bar(xp - 0.18, au_m, 0.35, yerr=au_s, error_kw=eb, color="#D68910", edgecolor="black", label="Commit Selective Routing (%)")
    axes1[2].bar(xp + 0.18, vd_m, 0.35, yerr=vd_s, error_kw=eb, color="#1F618D", edgecolor="black", label="Audio-Only xArm6 Grounding (%)")
    axes1[2].set_xticks(xp)
    axes1[2].set_xticklabels(lbls, fontsize=7.6, rotation=12, ha="right")
    axes1[2].set_ylim(30, 118)
    axes1[2].set_ylabel("Routing / Grounding Accuracy (%)")
    axes1[2].set_title("C. Commit Routing & Audio-Only xArm6 Grounding")
    axes1[2].legend(fontsize=7.4, loc="upper right", framealpha=0.92)

    fig1.tight_layout()
    f11 = fig_dir / f"{fig_prefix}_joint_recall_and_non_inferiority.png"
    fig1.savefig(f11, bbox_inches="tight")
    plt.close(fig1)
    saved.append(f11)

    # Figure 12: Multi-Item Overwrite Capacity Curves (K in {1, 2, 4, 8, 16}) at T=64 (In-Window) vs T=1024 (Beyond-Window)
    fig2, axes2 = plt.subplots(1, 3, figsize=(16.2, 4.9))
    ks = [1, 2, 4, 8, 16]
    for h_i, (hz, ttl) in enumerate(((64, "A. Inside-Window Capacity (T=64)"), (1024, "B. Long-Horizon Capacity (T=1024)"))):
        ax = axes2[h_i]
        for lbl, ck, col in pres:
            mi = groups[ck].get("multi_item_overwrite", {})
            ys = [float(mi.get(f"K{k}_T{hz}_acc", {}).get("mean", np.nan)) * 100.0 for k in ks]
            es = [float(mi.get(f"K{k}_T{hz}_acc", {}).get("std", 0.0)) * 100.0 for k in ks]
            if not all(np.isnan(ys)):
                ax.errorbar(ks, ys, yerr=es, marker="o", linewidth=1.9, capsize=3.0, color=col, label=lbl.replace("\n", " "))
        ax.axhline(6.25, color="#7F8C8D", linestyle=":", label="16-Value Chance (6.25%)")
        ax.set_xticks(ks)
        ax.set_ylim(0, 108)
        ax.set_xlabel("Concurrent Associative Bindings K")
        ax.set_ylabel("Selective Overwrite + Retention Accuracy (%)")
        ax.set_title(ttl)
        if h_i == 0:
            ax.legend(fontsize=6.9, loc="lower left", framealpha=0.92)

    # Panel C: Live Streaming State Footprint (MiB) vs Beyond-Window Recall Pareto
    ax_p = axes2[2]
    for lbl, ck, col in pres:
        mib = float(groups[ck].get("strict_system_state_mib", groups[ck].get("streaming_state_mib", 0.0)))
        r1k, r1k_e = _ms(ck, "dyn_assoc_T1024")
        ax_p.errorbar([mib], [r1k], yerr=[r1k_e], fmt="s", markersize=9, capsize=3.5, color=col, label=lbl.replace("\n", " "))
    ax_p.set_xlabel("Live B=1 Strict System State Footprint (MiB)")
    ax_p.set_ylabel("Dynamic-Visual Recall at T=1024 (%)")
    ax_p.set_ylim(0, 108)
    ax_p.set_title("C. Streaming State Footprint vs Long-Horizon Recall")
    ax_p.legend(fontsize=6.9, loc="best", framealpha=0.92)

    fig2.tight_layout()
    f12 = fig_dir / f"{fig_prefix}_k_capacity_and_gqa_tradeoff.png"
    fig2.savefig(f12, bbox_inches="tight")
    plt.close(fig2)
    saved.append(f12)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--exp-dir",
        type=Path,
        default=REPO_ROOT / "experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111",
    )
    ap.add_argument(
        "--study-config",
        type=Path,
        default=REPO_ROOT / "configs/x1b/study_x2_wave8a.json",
    )
    ap.add_argument("--runs-subdir", type=str, default=None)
    ap.add_argument("--summary-out", type=Path, default=None)
    ap.add_argument("--showcase-out", type=Path, default=None)
    ap.add_argument("--fig-dir", type=Path, default=None)
    args = ap.parse_args()

    study_cfg = json.loads(args.study_config.read_text())
    runs_subdir = args.runs_subdir or str(study_cfg.get("runs_subdir", "runs_wave8a"))
    w8a_dir = args.exp_dir / runs_subdir
    bundles = load_wave8a_bundles(w8a_dir)
    groups = {c: aggregate_w8a_cohort(bundles, c, study_cfg=study_cfg) for c in COHORT_NAMES}
    hier = compute_wave8a_hierarchical_attributions(bundles)
    fig_dir = args.fig_dir or (args.exp_dir / "figures")
    fig_prefix = "x1b_wave8a" if runs_subdir == "runs_wave8a" else f"x1b_{runs_subdir.removeprefix('runs_')}"
    figs = render_wave8a_figures(groups, fig_dir, fig_prefix=fig_prefix)

    sc_paths = select_median_seed_showcase_paths(bundles, w8a_dir, strict=True) if bundles else {}
    default_sc = (
        args.exp_dir / "inference_showcase/wave8a/dev"
        if runs_subdir == "runs_wave8a"
        else args.exp_dir / f"inference_showcase/{runs_subdir.removeprefix('runs_')}/dev"
    )
    showcase_dir = args.showcase_out or default_sc
    showcase_manifest = render_wave8a_showcase_gallery(sc_paths, showcase_dir) if sc_paths else {"rendered": False}
    gate4 = evaluate_gate4_checks(groups, hier, showcase_manifest)

    out = {
        "study_id": str(study_cfg.get("study_id", "spatiotemporal-jepa-x2-wave8a-trunk-isolation-and-gqa-parity")),
        "runs_subdir": runs_subdir,
        "wave8a_completed_runs": len(bundles),
        "groups": groups,
        "hierarchical_seed_attributions": hier,
        "gate4_non_inferiority_and_recall_checks": gate4,
        "figures": [
            str(f.relative_to(REPO_ROOT)) if f.is_relative_to(REPO_ROOT) else str(f)
            for f in figs
        ],
        "showcase_manifest": showcase_manifest,
    }
    default_sum = (
        args.exp_dir / "wave8a_summary_metrics.json"
        if runs_subdir == "runs_wave8a"
        else args.exp_dir / f"{runs_subdir.removeprefix('runs_')}_summary_metrics.json"
    )
    out_p = args.summary_out or default_sum
    out_p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        f"Wrote {out_p} ({len(bundles)} runs from {runs_subdir}, gate4_overall_pass={gate4.get('gate4_overall_pass')}), "
        f"{len(figs)} figures, showcase={bool(sc_paths)}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
