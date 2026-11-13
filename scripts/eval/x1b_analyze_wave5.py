"""Aggregate verified Study X1B Wave 5 results, compute pooled 3-seed paired bootstrap 95%/99% CIs, and render Wave 5 publication figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evals.x1b_stats import hierarchical_seed_bootstrap_ci
from scripts.eval.x1b_analyze_and_plot import (
    aggregate_seed_group,
    load_run_bundle,
    paired_bootstrap_95_99,
)


def aggregate_wave5_group(bundles: dict[str, dict[str, Any]], prefix: str) -> dict[str, Any]:
    """Extend aggregate_seed_group with Panel 9 physical xArm6 robot millimeter & counterfactual metrics."""
    base = aggregate_seed_group(bundles, prefix)
    if not base:
        return {}
    members = [v for k, v in sorted(bundles.items()) if k.startswith(prefix)]

    def _collect_rob(fn) -> list[float]:
        vals = []
        for m in members:
            rob = m["eval"].get("real_robot_transfer", {})
            if rob.get("available"):
                val = fn(rob)
                if val is not None:
                    vals.append(float(val))
        return vals

    rob_metrics = {
        "robot_step1_mm": _collect_rob(lambda r: r["step1_error_mm"]["mean"]),
        "robot_step1_persist_mm": _collect_rob(lambda r: r["step1_persistence_mm"]),
        "robot_step1_skill": _collect_rob(lambda r: r["step1_skill_vs_persistence"]),
        "robot_step4_mm": _collect_rob(lambda r: r["step4_error_mm"]["mean"]),
        "robot_step4_persist_mm": _collect_rob(lambda r: r["step4_persistence_mm"]),
        "robot_step4_cf_plan_mm": _collect_rob(lambda r: r["step4_counterfactual_plan_mm"]),
        "robot_step4_skill": _collect_rob(lambda r: r["step4_skill_vs_persistence"]),
        "robot_plan_gain_mm": _collect_rob(lambda r: r["causal_plan_gain_mm"]["mean"]),
        "robot_end_true_mm": _collect_rob(lambda r: r["endpoint_true_t3_mm"]["mean"]),
        "robot_end_wrong_mm": _collect_rob(lambda r: r["endpoint_wrong_t3_mm"]["mean"]),
        "robot_cf_end_gain_mm": _collect_rob(lambda r: r["counterfactual_endpoint_gain_mm"]["mean"]),
        "robot_cmd_disc_acc": _collect_rob(lambda r: r["command_discrimination_acc"]["mean"]),
        "robot_tran04_disc_acc": _collect_rob(lambda r: r["transient_04s_discrimination_acc"]["mean"]),
        "robot_tran04_end_mm": _collect_rob(lambda r: r["transient_04s_late_endpoint_mm"]["mean"]),
        "robot_cos_true": _collect_rob(lambda r: r["policy_cosine_true"]),
        "robot_cos_wrong": _collect_rob(lambda r: r["policy_cosine_wrong"]),
        "robot_cos_gain": _collect_rob(lambda r: r["policy_cosine_gain"]["mean"]),
        "robot_novel_step4_mm": _collect_rob(lambda r: r["novel_pair_step4_mm"]["mean"]),
        "robot_novel_disc_acc": _collect_rob(lambda r: r["novel_pair_discrimination_acc"]["mean"]),
    }
    for k, v in rob_metrics.items():
        if v:
            base[k] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
    return base


def _pooled_vec(bundles: dict[str, dict[str, Any]], prefix: str, fn) -> np.ndarray:
    parts = []
    for k, v in sorted(bundles.items()):
        if k.startswith(prefix):
            arr = fn(v["eval"])
            if arr is not None and len(arr):
                parts.append(np.asarray(arr, dtype=np.float64))
    return np.concatenate(parts, axis=0) if parts else np.array([], dtype=np.float64)


def _seed_mat(bundles: dict[str, dict[str, Any]], prefix: str, fn) -> np.ndarray | None:
    rows = []
    for k, v in sorted(bundles.items()):
        if k.startswith(prefix):
            arr = fn(v["eval"])
            if arr is not None and len(arr):
                rows.append(np.asarray(arr, dtype=np.float64))
    if not rows:
        return None
    if len({r.shape for r in rows}) != 1:
        return None
    return np.stack(rows, axis=0)


def compute_wave5_hierarchical_cis(
    w5_bundles: dict[str, dict[str, Any]],
    w4_bundles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute two-level seed-aware hierarchical cluster bootstrap CIs alongside episode-pooled CIs."""
    hier: dict[str, Any] = {}
    metrics_spec = (
        ("mpc", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"]),
        ("thin_barrier_mpc", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"].get("thin_barrier_vec")),
        ("sustained_T76_mpc", lambda e: e["long_horizon_memory_mpc"]["sustained_transient_barrier_mpc"]["succ_vec"]),
        ("goal_switch_mpc", lambda e: e["long_horizon_memory_mpc"]["mid_episode_goal_switch_mpc"]["succ_vec"]),
        ("T1024_assoc", lambda e: e["associative_recall_512"].get("T1024_vec")),
        ("robot_step4_mm", lambda e: e.get("real_robot_transfer", {}).get("step4_mm_vec")),
        ("robot_disc_acc", lambda e: e.get("real_robot_transfer", {}).get("disc_vec")),
    )

    slwm_w5_mpc = _seed_mat(w5_bundles, "x1b-w5-slwm-1b-", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    slwm_w4_mpc = _seed_mat(w4_bundles, "x1b-flagship-slwm-1b-", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    if slwm_w5_mpc is not None and slwm_w4_mpc is not None and slwm_w5_mpc.shape == slwm_w4_mpc.shape:
        hier["chinchilla_slwm1b_5600_vs_2800_mpc"] = hierarchical_seed_bootstrap_ci(slwm_w5_mpc, slwm_w4_mpc)

    for bname, bpref in (
        ("dense_5600", "x1b-w5-dense-1b-"),
        ("swin64_5600", "x1b-w5-swin64-1b-"),
        ("diagssm_5600", "x1b-w5-diagssm-1b-"),
        ("slwm_450m", "x1b-w5-slwm-450m-"),
    ):
        for mname, mfn in metrics_spec:
            s_mat = _seed_mat(w5_bundles, "x1b-w5-slwm-1b-", mfn)
            b_mat = _seed_mat(w5_bundles, bpref, mfn)
            if s_mat is not None and b_mat is not None and s_mat.shape == b_mat.shape:
                hier[f"slwm1b_vs_{bname}_{mname}"] = hierarchical_seed_bootstrap_ci(s_mat, b_mat)
    return hier


def compute_wave5_paired_cis(
    w5_bundles: dict[str, dict[str, Any]],
    w4_bundles: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute pooled 3-seed paired bootstrap 95% & 99% CIs across Wave 5 & Wave 4 cohorts."""
    stats: dict[str, Any] = {}

    slwm_w5_mpc = _pooled_vec(w5_bundles, "x1b-w5-slwm-1b-", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    slwm_w4_mpc = _pooled_vec(w4_bundles, "x1b-flagship-slwm-1b-", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    if len(slwm_w5_mpc) == len(slwm_w4_mpc) and len(slwm_w5_mpc) > 0:
        stats["chinchilla_slwm1b_5600_vs_2800_mpc"] = paired_bootstrap_95_99(slwm_w5_mpc, slwm_w4_mpc)

    for bname, bpref in (
        ("dense_5600", "x1b-w5-dense-1b-"),
        ("swin64_5600", "x1b-w5-swin64-1b-"),
        ("diagssm_5600", "x1b-w5-diagssm-1b-"),
        ("slwm_450m", "x1b-w5-slwm-450m-"),
    ):
        b_mpc = _pooled_vec(w5_bundles, bpref, lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
        if len(slwm_w5_mpc) == len(b_mpc) and len(b_mpc) > 0:
            stats[f"slwm1b_vs_{bname}_mpc"] = paired_bootstrap_95_99(slwm_w5_mpc, b_mpc)

        s_sust = _pooled_vec(
            w5_bundles, "x1b-w5-slwm-1b-", lambda e: e["long_horizon_memory_mpc"]["sustained_transient_barrier_mpc"]["succ_vec"]
        )
        b_sust = _pooled_vec(
            w5_bundles, bpref, lambda e: e["long_horizon_memory_mpc"]["sustained_transient_barrier_mpc"]["succ_vec"]
        )
        if len(s_sust) == len(b_sust) and len(b_sust) > 0:
            stats[f"slwm1b_vs_{bname}_sustained_T76_mpc"] = paired_bootstrap_95_99(s_sust, b_sust)

        s_sw = _pooled_vec(
            w5_bundles, "x1b-w5-slwm-1b-", lambda e: e["long_horizon_memory_mpc"]["mid_episode_goal_switch_mpc"]["succ_vec"]
        )
        b_sw = _pooled_vec(
            w5_bundles, bpref, lambda e: e["long_horizon_memory_mpc"]["mid_episode_goal_switch_mpc"]["succ_vec"]
        )
        if len(s_sw) == len(b_sw) and len(b_sw) > 0:
            stats[f"slwm1b_vs_{bname}_goal_switch_mpc"] = paired_bootstrap_95_99(s_sw, b_sw)

        s_t1024 = _pooled_vec(w5_bundles, "x1b-w5-slwm-1b-", lambda e: e["associative_recall_512"].get("T1024_vec"))
        b_t1024 = _pooled_vec(w5_bundles, bpref, lambda e: e["associative_recall_512"].get("T1024_vec"))
        if len(s_t1024) == len(b_t1024) and len(b_t1024) > 0:
            stats[f"slwm1b_vs_{bname}_T1024_assoc"] = paired_bootstrap_95_99(s_t1024, b_t1024)

        s_rob4 = _pooled_vec(w5_bundles, "x1b-w5-slwm-1b-", lambda e: e.get("real_robot_transfer", {}).get("step4_mm_vec"))
        b_rob4 = _pooled_vec(w5_bundles, bpref, lambda e: e.get("real_robot_transfer", {}).get("step4_mm_vec"))
        if len(s_rob4) == len(b_rob4) and len(b_rob4) > 0:
            stats[f"slwm1b_vs_{bname}_robot_step4_mm_diff"] = paired_bootstrap_95_99(s_rob4, b_rob4)

        s_disc = _pooled_vec(w5_bundles, "x1b-w5-slwm-1b-", lambda e: e.get("real_robot_transfer", {}).get("disc_vec"))
        b_disc = _pooled_vec(w5_bundles, bpref, lambda e: e.get("real_robot_transfer", {}).get("disc_vec"))
        if len(s_disc) == len(b_disc) and len(b_disc) > 0:
            stats[f"slwm1b_vs_{bname}_robot_disc_acc"] = paired_bootstrap_95_99(s_disc, b_disc)

    # Test SLWM-X1B command discrimination strictly above 50% chance across N=768 pooled physical robot trajectories
    s_disc_all = _pooled_vec(w5_bundles, "x1b-w5-slwm-1b-", lambda e: e.get("real_robot_transfer", {}).get("disc_vec"))
    if len(s_disc_all) > 0:
        stats["slwm1b_robot_disc_above_chance"] = paired_bootstrap_95_99(s_disc_all - 0.50)

    return stats


def render_wave5_figures(
    groups: dict[str, dict[str, Any]],
    fig_dir: Path,
) -> list[Path]:
    """Render Wave 5 Chinchilla/O(1) Baselines figure + Real Physical xArm6 Robot Transfer figure."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9.6,
        "axes.titlesize": 10.6,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.28,
        "figure.dpi": 170,
    })
    eb_kw = {"capsize": 3.0, "elinewidth": 1.1, "ecolor": "#1B2631"}
    saved: list[Path] = []

    def _ms(gkey: str, mkey: str, scale: float = 100.0) -> tuple[float, float]:
        d = groups.get(gkey, {}).get(mkey)
        if not d:
            return 0.0, 0.0
        return float(d["mean"]) * scale, float(d["std"]) * scale

    # -------------------------------------------------------------------------
    # Figure 5: Chinchilla Token-Matched 1.008B Scaling & Modern O(1) Baselines
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7))
    cohorts = [
        ("453M SLWM\n(2.6k steps)", "w5_slwm_450m", "#2B5C8F"),
        ("1.008B SLWM\n(Wave4 2.8k)", "w4_slwm_1b", "#7FB3D5"),
        ("1.008B SLWM\n(Chinchilla 5.6k)", "w5_slwm_1b", "#117A65"),
        ("1.008B Diag-SSM\n(Mamba-Vec 5.6k)", "w5_diagssm_1b", "#8E44AD"),
        ("1.008B SWin-64\n(O(1) KV 5.6k)", "w5_swin64_1b", "#D35400"),
        ("1.008B Dense-TF\n(Full KV 5.6k)", "w5_dense_1b", "#E67E22"),
    ]
    pres = [(lbl, k, c) for lbl, k, c in cohorts if k in groups and groups[k]]
    if pres:
        xp = np.arange(len(pres))
        lbls = [p[0] for p in pres]
        cols = [p[2] for p in pres]
        mpc_m, mpc_s = zip(*[_ms(k, "mpc_success") for _, k, _ in pres])
        bar_m, bar_s = zip(*[_ms(k, "mpc_thin_barrier") for _, k, _ in pres])
        axes[0].bar(xp - 0.18, mpc_m, 0.35, yerr=mpc_s, error_kw=eb_kw, color=cols, edgecolor="black", linewidth=0.7, label="Overall MPC (N=96/seed)")
        axes[0].bar(xp + 0.18, bar_m, 0.35, yerr=bar_s, error_kw=eb_kw, color=cols, alpha=0.55, hatch="//", edgecolor="black", linewidth=0.7, label="1-px Thin-Barrier MPC")
        axes[0].set_xticks(xp)
        axes[0].set_xticklabels(lbls, fontsize=8.1)
        axes[0].set_ylabel("Closed-Loop Manipulation Success (%)")
        axes[0].set_title("A. Chinchilla 1.008B Scaling & O(1) MPC")
        axes[0].set_ylim(0, 108)
        axes[0].legend(fontsize=7.6, loc="lower right")

    horizons = [32, 128, 256, 512, 1024]
    curve_defs = [
        ("SLWM-X1B (GatedDelta Matrix, 5.6k)", "w5_slwm_1b", "#117A65", "-o"),
        ("SLWM-X450M (GatedDelta Matrix, 2.6k)", "w5_slwm_450m", "#2B5C8F", "--s"),
        ("DiagonalSSM-X1B (Selective Vec, 5.6k)", "w5_diagssm_1b", "#8E44AD", "-.d"),
        ("SlidingWindow-X1B (W=64 KV, 5.6k)", "w5_swin64_1b", "#D35400", ":v"),
        ("DenseTransformer-X1B (Full KV, 5.6k)", "w5_dense_1b", "#E67E22", "--^"),
    ]
    for clbl, gk, col, fmt in curve_defs:
        if gk in groups and groups[gk]:
            hs = [h for h in horizons if f"assoc_T{h}" in groups[gk]]
            if hs:
                ys = [groups[gk][f"assoc_T{h}"]["mean"] * 100.0 for h in hs]
                es = [groups[gk][f"assoc_T{h}"]["std"] * 100.0 for h in hs]
                axes[1].errorbar(hs, ys, yerr=es, fmt=fmt, color=col, linewidth=2.0, markersize=5.5, capsize=2.5, label=clbl)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(horizons)
    axes[1].set_xticklabels([str(h) for h in horizons])
    axes[1].set_xlabel("Sequence Horizon T (trained at T=32)")
    axes[1].set_ylabel("Associative Key-Value Overwrite Recall (%)")
    axes[1].set_title("B. Matrix Erase-Write vs Diagonal SSM & SWin-64")
    axes[1].set_ylim(-3, 108)
    axes[1].legend(fontsize=7.0, loc="lower left")

    mem_cohorts = [
        ("SLWM-X1B\n(14.9 MB)", "w5_slwm_1b"),
        ("Diag-SSM\n(7.1 MB)", "w5_diagssm_1b"),
        ("SWin-64\n(207 MB)", "w5_swin64_1b"),
        ("Dense-TF\n(3,312 MB)", "w5_dense_1b"),
        ("SLWM-450M\n(8.6 MB)", "w5_slwm_450m"),
    ]
    mpres = [(l, k) for l, k in mem_cohorts if k in groups and groups[k]]
    if mpres:
        xm = np.arange(len(mpres))
        su_m, su_s = zip(*[_ms(k, "sustained_mpc_overall") for _, k in mpres])
        sw_m, sw_s = zip(*[_ms(k, "goal_switch_mpc_overall") for _, k in mpres])
        axes[2].bar(xm - 0.18, su_m, 0.35, yerr=su_s, error_kw=eb_kw, color="#1ABC9C", edgecolor="black", label="T=76 Sustained 0.4s Goal MPC")
        axes[2].bar(xm + 0.18, sw_m, 0.35, yerr=sw_s, error_kw=eb_kw, color="#117A65", edgecolor="black", label="T=76 Mid-Ep Goal-Switch MPC")
        axes[2].set_xticks(xm)
        axes[2].set_xticklabels([p[0] for p in mpres], fontsize=8.3)
        axes[2].set_ylabel("OOD Horizon T=76 Success (%)")
        axes[2].set_title("C. T=76 Streaming Memory & Goal Overwrite")
        axes[2].set_ylim(0, 108)
        axes[2].legend(fontsize=7.6, loc="lower left")

    fig.tight_layout()
    f5 = fig_dir / "x1b_wave5_chinchilla_and_o1_baselines.png"
    fig.savefig(f5, bbox_inches="tight")
    plt.close(fig)
    saved.append(f5)

    # -------------------------------------------------------------------------
    # Figure 6: Panel 9 Zero-Leakage Physical xArm6 Robot Transfer (Millimeters)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.7))
    rob_cohorts = [
        ("SLWM-X1B\n(1.008B)", "w5_slwm_1b", "#117A65"),
        ("Diag-SSM\n(1.008B)", "w5_diagssm_1b", "#8E44AD"),
        ("SWin-64\n(1.008B)", "w5_swin64_1b", "#D35400"),
        ("Dense-TF\n(1.008B)", "w5_dense_1b", "#E67E22"),
        ("SLWM-450M\n(453M)", "w5_slwm_450m", "#2B5C8F"),
    ]
    rpres = [(l, k, c) for l, k, c in rob_cohorts if k in groups and "robot_step4_mm" in groups[k]]
    if rpres:
        xr = np.arange(len(rpres))
        rlbls = [p[0] for p in rpres]
        rcols = [p[2] for p in rpres]

        s1_m, s1_s = zip(*[_ms(k, "robot_step1_mm", 1.0) for _, k, _ in rpres])
        s4_m, s4_s = zip(*[_ms(k, "robot_step4_mm", 1.0) for _, k, _ in rpres])
        cf_m, cf_s = zip(*[_ms(k, "robot_step4_cf_plan_mm", 1.0) for _, k, _ in rpres])
        pers4 = groups[rpres[0][1]]["robot_step4_persist_mm"]["mean"]

        axes[0].bar(xr - 0.25, s1_m, 0.24, yerr=s1_s, error_kw=eb_kw, color="#5DADE2", edgecolor="black", label="1-Step Rollout Error (mm)")
        axes[0].bar(xr, s4_m, 0.24, yerr=s4_s, error_kw=eb_kw, color=rcols, edgecolor="black", label="4-Step True-Plan Error (mm)")
        axes[0].bar(xr + 0.25, cf_m, 0.24, yerr=cf_s, error_kw=eb_kw, color="#E74C3C", alpha=0.65, hatch="//", edgecolor="black", label="4-Step Negated-Plan (-u) (mm)")
        axes[0].axhline(pers4, color="#34495E", linestyle="--", linewidth=1.2, label=f"4-Step Zero-Vel Persistence ({pers4:.1f} mm)")
        axes[0].set_xticks(xr)
        axes[0].set_xticklabels(rlbls, fontsize=8.4)
        axes[0].set_ylabel("Physical xArm6 Cartesian Error (mm, lower is better)")
        axes[0].set_title("A. Physical xArm6 Multi-Step Rollout (mm)")
        axes[0].legend(fontsize=7.3, loc="upper right")

        et_m, et_s = zip(*[_ms(k, "robot_end_true_mm", 1.0) for _, k, _ in rpres])
        ew_m, ew_s = zip(*[_ms(k, "robot_end_wrong_mm", 1.0) for _, k, _ in rpres])
        axes[1].bar(xr - 0.18, et_m, 0.35, yerr=et_s, error_kw=eb_kw, color="#117A65", edgecolor="black", label="True Human Instruction (t=3)")
        axes[1].bar(xr + 0.18, ew_m, 0.35, yerr=ew_s, error_kw=eb_kw, color="#C0392B", alpha=0.72, hatch="//", edgecolor="black", label="Counterfactual Wrong Instruction")
        axes[1].set_xticks(xr)
        axes[1].set_xticklabels(rlbls, fontsize=8.4)
        axes[1].set_ylabel("Predicted Endpoint Error at t=0.4s (mm, lower is better)")
        axes[1].set_title("B. Counterfactual Language Grounding (mm)")
        axes[1].legend(fontsize=7.6, loc="upper right")

        dc_m, dc_s = zip(*[_ms(k, "robot_cmd_disc_acc") for _, k, _ in rpres])
        nv_m, nv_s = zip(*[_ms(k, "robot_novel_disc_acc") for _, k, _ in rpres])
        tr_m, tr_s = zip(*[_ms(k, "robot_tran04_disc_acc") for _, k, _ in rpres])
        axes[2].bar(xr - 0.25, dc_m, 0.24, yerr=dc_s, error_kw=eb_kw, color="#117A65", edgecolor="black", label="All Paired Test (N=256/seed)")
        axes[2].bar(xr, nv_m, 0.24, yerr=nv_s, error_kw=eb_kw, color="#2980B9", edgecolor="black", label="Novel Color Pairs (N=64/seed)")
        axes[2].bar(xr + 0.25, tr_m, 0.24, yerr=tr_s, error_kw=eb_kw, color="#1ABC9C", edgecolor="black", label="Transient 0.4s Late-Ep Retention")
        axes[2].axhline(50.0, color="#7F8C8D", linestyle="--", linewidth=1.2, label="Random / Language-Blind Chance (50%)")
        axes[2].set_xticks(xr)
        axes[2].set_xticklabels(rlbls, fontsize=8.4)
        axes[2].set_ylabel("Pairwise Instruction Discrimination (%)")
        axes[2].set_title("C. Physical Robot Command & Novel-Pair Accuracy")
        axes[2].set_ylim(35, 102)
        axes[2].legend(fontsize=7.3, loc="lower right")

    fig.tight_layout()
    f6 = fig_dir / "x1b_wave5_real_robot_transfer.png"
    fig.savefig(f6, bbox_inches="tight")
    plt.close(fig)
    saved.append(f6)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--exp-dir",
        type=Path,
        default=REPO_ROOT / "experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111",
    )
    args = ap.parse_args()

    w5_dir = args.exp_dir / "runs_wave5"
    w4_dir = args.exp_dir / "runs"
    w5_bundles = load_run_bundle(w5_dir)
    w4_bundles = load_run_bundle(w4_dir)

    groups = {
        "w5_slwm_1b": aggregate_wave5_group(w5_bundles, "x1b-w5-slwm-1b-"),
        "w5_dense_1b": aggregate_wave5_group(w5_bundles, "x1b-w5-dense-1b-"),
        "w5_swin64_1b": aggregate_wave5_group(w5_bundles, "x1b-w5-swin64-1b-"),
        "w5_diagssm_1b": aggregate_wave5_group(w5_bundles, "x1b-w5-diagssm-1b-"),
        "w5_slwm_450m": aggregate_wave5_group(w5_bundles, "x1b-w5-slwm-450m-"),
        "w4_slwm_1b": aggregate_wave5_group(w4_bundles, "x1b-flagship-slwm-1b-"),
        "w4_dense_1b": aggregate_wave5_group(w4_bundles, "x1b-baseline-dense-1b-"),
        "w4_legacy_1b": aggregate_wave5_group(w4_bundles, "x1b-baseline-legacy-1b-"),
        "w4_slwm_450m": aggregate_wave5_group(w4_bundles, "x1b-scale-slwm-450m-"),
    }
    paired_cis = compute_wave5_paired_cis(w5_bundles, w4_bundles)
    hier_cis = compute_wave5_hierarchical_cis(w5_bundles, w4_bundles)
    figs = render_wave5_figures(groups, args.exp_dir / "figures")

    out_doc = {
        "study_id": "spatiotemporal-jepa-1b-20261111-wave5",
        "wave5_verified_runs": len(w5_bundles),
        "cumulative_verified_runs_waves_1_to_5": 120 + len(w5_bundles),
        "groups": groups,
        "paired_bootstrap_cis": paired_cis,
        "hierarchical_seed_cis": hier_cis,
        "figures": [str(f.relative_to(REPO_ROOT)) for f in figs],
    }
    out_path = args.exp_dir / "wave5_summary_metrics.json"
    out_path.write_text(json.dumps(out_doc, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_path} ({len(w5_bundles)} Wave 5 runs) and {len(figs)} figures.", flush=True)


if __name__ == "__main__":
    main()
