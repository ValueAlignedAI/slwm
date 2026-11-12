"""Aggregate verified Study X1B Wave 4 results, compute pooled 3-seed paired bootstrap 95%/99% CIs, and render 4 publication figures."""

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


def paired_bootstrap_95_99(
    values_a: np.ndarray,
    values_b: np.ndarray | None = None,
    *,
    n_boot: int = 8000,
    seed: int = 20261111,
) -> dict[str, float | int]:
    """Compute mean, 95% CI, and 99% CI for values_a (or paired difference values_a - values_b)."""
    arr = np.asarray(values_a, dtype=np.float64)
    if values_b is not None:
        arr = arr - np.asarray(values_b, dtype=np.float64)
    arr = arr.reshape(-1)
    n = int(len(arr))
    if n == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "ci99_low": 0.0, "ci99_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.mean(arr[idx], axis=1)
    lo95, hi95, lo99, hi99 = np.percentile(means, [2.5, 97.5, 0.5, 99.5])
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": float(lo95),
        "ci95_high": float(hi95),
        "ci99_low": float(lo99),
        "ci99_high": float(hi99),
        "n": n,
    }


def load_run_bundle(runs_dir: Path) -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    for run_path in sorted(runs_dir.glob("x1b-*")):
        rec_p = run_path / "receipt.json"
        ev_p = run_path / "eval_metrics.json"
        ver_p = run_path / "verified.json"
        log_p = run_path / "train.jsonl"
        if not (rec_p.is_file() and ev_p.is_file()):
            continue
        logs = [json.loads(line) for line in log_p.read_text().splitlines() if line.strip()] if log_p.is_file() else []
        bundles[run_path.name] = {
            "receipt": json.loads(rec_p.read_text()),
            "eval": json.loads(ev_p.read_text()),
            "verified": json.loads(ver_p.read_text()) if ver_p.is_file() else None,
            "logs": logs,
        }
    return bundles


def aggregate_seed_group(bundles: dict[str, dict[str, Any]], prefix: str) -> dict[str, Any]:
    members = [v for k, v in sorted(bundles.items()) if k.startswith(prefix)]
    if not members:
        return {}

    def _collect(fn) -> list[float]:
        vals = []
        for m in members:
            val = fn(m["eval"])
            if val is not None:
                vals.append(float(val))
        return vals

    metrics = {
        "n_seeds": len(members),
        "strict_params": members[0]["receipt"]["parameter_counts"]["strict_total"],
        "av_r1": _collect(lambda e: e["real_multimodal_and_text"].get("av_r1", {}).get("mean")),
        "av_r5": _collect(lambda e: e["real_multimodal_and_text"].get("av_r5", {}).get("mean")),
        "contact_auroc": _collect(lambda e: e["real_multimodal_and_text"].get("contact_auroc")),
        "fineweb_ppl": _collect(lambda e: e["real_multimodal_and_text"].get("fineweb_ppl")),
        "pr": _collect(lambda e: e["real_multimodal_and_text"].get("participation_ratio")),
        "persist_skill_1": _collect(lambda e: e["thin_barrier_and_persistence"]["step1_skill_vs_persistence"]),
        "persist_skill_4": _collect(lambda e: e["thin_barrier_and_persistence"]["step4_skill_vs_persistence"]),
        "barrier_auroc_64": _collect(lambda e: e["thin_barrier_and_persistence"]["barrier_auroc_64x64"]),
        "barrier_auroc_16": _collect(lambda e: e["thin_barrier_and_persistence"]["barrier_auroc_16x16_subsampled"]),
        "pymunk_barrier_auroc": _collect(lambda e: e["thin_barrier_and_persistence"].get("pymunk_puck_barrier_auroc_64")),
        "assoc_T32": _collect(lambda e: e["associative_recall_512"]["T32"]["mean"]),
        "assoc_T128": _collect(lambda e: e["associative_recall_512"]["T128"]["mean"]),
        "assoc_T256": _collect(lambda e: e["associative_recall_512"]["T256"]["mean"]),
        "assoc_T512": _collect(lambda e: e["associative_recall_512"]["T512"]["mean"]),
        "assoc_T1024": _collect(lambda e: e["associative_recall_512"].get("T1024", {}).get("mean")),
        "transient_cont": _collect(lambda e: e["transient_goal_retention"]["continuous_success"]["mean"]),
        "transient_04s": _collect(lambda e: e["transient_goal_retention"]["transient_04s_success"]["mean"]),
        "transient_drop_pp": _collect(lambda e: e["transient_goal_retention"]["transient_drop_pp"]["mean"]),
        "mpc_success": _collect(lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"]),
        "mpc_pusht_open": _collect(lambda e: e["closed_loop_mpc"]["latent_cem_mpc"].get("pusht_open_success")),
        "mpc_thin_barrier": _collect(lambda e: e["closed_loop_mpc"]["latent_cem_mpc"].get("thin_barrier_success")),
        "g8_shoot_success": _collect(lambda e: e["closed_loop_mpc"]["g8_random_shooting_3"]["success_rate"]["mean"]),
        "open_loop_success": _collect(lambda e: e["closed_loop_mpc"]["open_loop_policy"]["success_rate"]["mean"]),
        "sustained_mpc_overall": _collect(
            lambda e: e.get("long_horizon_memory_mpc", {}).get("sustained_transient_barrier_mpc", {}).get("success_rate", {}).get("mean")
        ),
        "sustained_mpc_open": _collect(
            lambda e: e.get("long_horizon_memory_mpc", {}).get("sustained_transient_barrier_mpc", {}).get("open_success")
        ),
        "sustained_mpc_barrier": _collect(
            lambda e: e.get("long_horizon_memory_mpc", {}).get("sustained_transient_barrier_mpc", {}).get("thin_barrier_success")
        ),
        "goal_switch_mpc_overall": _collect(
            lambda e: e.get("long_horizon_memory_mpc", {}).get("mid_episode_goal_switch_mpc", {}).get("success_rate", {}).get("mean")
        ),
        "goal_switch_mpc_open": _collect(
            lambda e: e.get("long_horizon_memory_mpc", {}).get("mid_episode_goal_switch_mpc", {}).get("open_success")
        ),
        "goal_switch_mpc_barrier": _collect(
            lambda e: e.get("long_horizon_memory_mpc", {}).get("mid_episode_goal_switch_mpc", {}).get("thin_barrier_success")
        ),
        "stale_goal_a_trap": _collect(
            lambda e: e.get("long_horizon_memory_mpc", {}).get("mid_episode_goal_switch_mpc", {}).get("stale_goal_a_trap_rate")
        ),
        "spoof_regret_learned": _collect(lambda e: e["spoofing_and_commit_gate"]["learned_innovation"]["spoofed_action_regret"]),
        "spoof_regret_always": _collect(lambda e: e["spoofing_and_commit_gate"]["always_commit"]["spoofed_action_regret"]),
        "spoof_detect_learned": _collect(lambda e: e["spoofing_and_commit_gate"]["learned_innovation"].get("spoof_detection_rate")),
        "hidden_probe_learned": _collect(lambda e: e["spoofing_and_commit_gate"]["learned_innovation"].get("hidden_barrier_probe_rate")),
        "mode_acc_learned": _collect(lambda e: e["spoofing_and_commit_gate"]["learned_innovation"].get("mode_accuracy")),
        "clean_usefulness_learned": _collect(lambda e: e["spoofing_and_commit_gate"]["learned_innovation"]["clean_usefulness"]),
        "clean_usefulness_always": _collect(lambda e: e["spoofing_and_commit_gate"]["always_commit"]["clean_usefulness"]),
        "unsupp_claim_learned": _collect(lambda e: e["spoofing_and_commit_gate"]["learned_innovation"]["unsupported_claim_rate"]),
        "unsupp_claim_always": _collect(lambda e: e["spoofing_and_commit_gate"]["always_commit"]["unsupported_claim_rate"]),
        "ece_learned": _collect(lambda e: e["spoofing_and_commit_gate"]["learned_innovation"]["ece"]),
        "t32_ms": _collect(lambda e: e.get("throughput_scaling", {}).get("T32_ms")),
        "t128_ms": _collect(lambda e: e.get("throughput_scaling", {}).get("T128_ms")),
        "t256_ms": _collect(lambda e: e.get("throughput_scaling", {}).get("T256_ms")),
        "t512_ms": _collect(lambda e: e.get("throughput_scaling", {}).get("T512_ms")),
        "t256_slots_s": _collect(lambda e: e.get("throughput_scaling", {}).get("T256_slots_per_s")),
        "t512_slots_s": _collect(lambda e: e.get("throughput_scaling", {}).get("T512_slots_per_s")),
        "streaming_state_mb_T64": _collect(lambda e: e.get("throughput_scaling", {}).get("streaming_state_mb_T64")),
        "streaming_state_mb_T1024": _collect(lambda e: e.get("throughput_scaling", {}).get("streaming_state_mb_T1024")),
    }
    summary: dict[str, Any] = {"n_seeds": metrics["n_seeds"], "strict_params": metrics["strict_params"]}
    for k, v in metrics.items():
        if isinstance(v, list) and v:
            summary[k] = {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "values": v,
            }
    return summary


def generate_figures(
    bundles: dict[str, dict[str, Any]],
    summary_table: dict[str, Any],
    paired_stats: dict[str, Any],
    fig_dir: Path,
) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9.8,
        "axes.titlesize": 10.8,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.28,
        "figure.dpi": 160,
    })
    saved: list[Path] = []
    eb_kw = {"capsize": 3.0, "elinewidth": 1.1, "ecolor": "#1B2631"}

    def _m_s(key: str, metric: str, scale: float = 100.0) -> tuple[float, float]:
        d = summary_table.get(key, {}).get(metric)
        if not d:
            return 0.0, 0.0
        return float(d["mean"]) * scale, float(d["std"]) * scale

    # -------------------------------------------------------------------------
    # Figure 1: 3-Seed 1.008B Head-to-Head & Scaling Law (152M -> 453M -> 1.008B)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))
    scale_tags = [
        ("152M\nSLWM", "abl_slwm_full"),
        ("453M\nSLWM", "scale_450m"),
        ("1.008B\nSLWM-X1B", "flagship_1b"),
        ("1.008B\nDense-TF", "dense_1b"),
        ("1.008B\nLegacy-G10", "legacy_1b"),
    ]
    present = [(lbl, key) for lbl, key in scale_tags if key in summary_table and summary_table[key]]
    if present:
        labels = [p[0] for p in present]
        colors = ["#4C78A8", "#2B5C8F", "#117A65", "#E67E22", "#C0392B"][: len(present)]
        xp = np.arange(len(present))

        mpc_m, mpc_s = zip(*[_m_s(k, "mpc_success") for _, k in present])
        bar_m, bar_s = zip(*[_m_s(k, "mpc_thin_barrier") for _, k in present])
        axes[0].bar(
            xp - 0.18, mpc_m, 0.35, yerr=mpc_s, error_kw=eb_kw,
            label="Overall Closed-Loop MPC (N=96/seed)", color=colors, edgecolor="black", linewidth=0.7,
        )
        axes[0].bar(
            xp + 0.18, bar_m, 0.35, yerr=bar_s, error_kw=eb_kw,
            label="1-px Thin-Barrier MPC (N=48/seed)", color=colors, alpha=0.55, hatch="//", edgecolor="black", linewidth=0.7,
        )
        axes[0].set_xticks(xp)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("Closed-Loop Manipulation Success (%)")
        axes[0].set_title("A. 3-Seed 1.008B Closed-Loop & Thin-Barrier MPC")
        axes[0].set_ylim(0, 108)
        axes[0].legend(fontsize=7.8, loc="upper right")

        pr_m, pr_s = zip(*[_m_s(k, "pr") for _, k in present])
        p1_m, p1_s = zip(*[_m_s(k, "persist_skill_1") for _, k in present])
        p4_m, p4_s = zip(*[_m_s(k, "persist_skill_4") for _, k in present])
        axes[1].bar(xp - 0.25, pr_m, 0.24, yerr=pr_s, error_kw=eb_kw, label="Participation Ratio (%)", color="#5DADE2", edgecolor="black", linewidth=0.7)
        axes[1].bar(xp, p1_m, 0.24, yerr=p1_s, error_kw=eb_kw, label="1-Step Skill vs Persist (%)", color="#27AE60", edgecolor="black", linewidth=0.7)
        axes[1].bar(xp + 0.25, p4_m, 0.24, yerr=p4_s, error_kw=eb_kw, label="4-Step Skill vs Persist (%)", color="#117A65", edgecolor="black", linewidth=0.7)
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        axes[1].set_xticks(xp)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_title("B. Representation Rank & Multi-Step Skill")
        axes[1].legend(fontsize=7.6, loc="upper left")

        av_m, av_s = zip(*[_m_s(k, "av_r1") for _, k in present])
        axes[2].bar(xp, av_m, 0.56, yerr=av_s, error_kw=eb_kw, color=colors, edgecolor="black", linewidth=0.7)
        axes[2].set_xticks(xp)
        axes[2].set_xticklabels(labels)
        axes[2].set_ylabel("Real Audio-Visual Retrieval R@1 (%)")
        axes[2].set_title("C. Real GreatestHits + STARSS23 Grounding")
        axes[2].set_ylim(0, 108)

    fig.tight_layout()
    p1 = fig_dir / "x1b_scaling_and_1b_head_to_head.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # -------------------------------------------------------------------------
    # Figure 2: T=32..1024 Associative Overwrite, Transient Retention & O(1) State
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.6))
    horizons = [32, 128, 256, 512, 1024]
    arch_curves = [
        ("SLWM-X1B (1.008B, 3 seeds)", "flagship_1b", "#117A65", "-o"),
        ("SLWM-X450M (453M, 3 seeds)", "scale_450m", "#1F618D", "--s"),
        ("SLWM-Full (152M, 3 seeds)", "abl_slwm_full", "#5DADE2", ":d"),
        ("Dense-Transformer (1.008B)", "dense_1b", "#E67E22", "--^"),
        ("No-Registers (152M)", "abl_no_registers", "#8E44AD", "-.x"),
        ("Legacy Scalar-Hybrid (1.008B)", "legacy_1b", "#C0392B", ":v"),
    ]
    for title_lbl, k, col, style in arch_curves:
        if k in summary_table and summary_table[k]:
            avail_h = [h for h in horizons if f"assoc_T{h}" in summary_table[k]]
            if avail_h:
                ys = [summary_table[k][f"assoc_T{h}"]["mean"] * 100.0 for h in avail_h]
                es = [summary_table[k][f"assoc_T{h}"]["std"] * 100.0 for h in avail_h]
                axes[0].errorbar(
                    avail_h, ys, yerr=es, fmt=style, color=col, linewidth=2.0, markersize=5.5, capsize=2.5, label=title_lbl
                )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(horizons)
    axes[0].set_xticklabels([str(h) for h in horizons])
    axes[0].set_xlabel("Evaluation Sequence Horizon T (trained at T=32; T=1024 = 24,576 slots)")
    axes[0].set_ylabel("Associative Overwrite Recall (%)")
    axes[0].set_title("A. Key-Value Overwrite Extrapolation (T=32 -> 1024)")
    axes[0].set_ylim(-3, 108)
    axes[0].legend(fontsize=6.9, loc="lower left", bbox_to_anchor=(0.02, 0.14), framealpha=0.92)

    mem_keys = [
        ("SLWM-Full\n(152M)", "abl_slwm_full"),
        ("SLWM-X450M\n(453M)", "scale_450m"),
        ("SLWM-X1B\n(1.008B)", "flagship_1b"),
        ("Dense-TF\n(1.008B)", "dense_1b"),
        ("Legacy-G10\n(1.008B)", "legacy_1b"),
    ]
    mem_pres = [(l, k) for l, k in mem_keys if k in summary_table and summary_table[k]]
    if mem_pres:
        x = np.arange(len(mem_pres))
        w = 0.36
        c_m, c_s = zip(*[_m_s(k, "transient_cont") for _, k in mem_pres])
        t_m, t_s = zip(*[_m_s(k, "transient_04s") for _, k in mem_pres])
        axes[1].bar(x - w / 2, c_m, w, yerr=c_s, error_kw=eb_kw, label="Continuous Goal Cue", color="#3498DB", edgecolor="black")
        axes[1].bar(x + w / 2, t_m, w, yerr=t_s, error_kw=eb_kw, label="Transient 0.4s Cue (Decoupled Head)", color="#1ABC9C", edgecolor="black")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([p[0] for p in mem_pres])
        axes[1].set_ylabel("Closed-Loop Policy Success (%)")
        axes[1].set_title("B. Resolving Transient-Command Amnesia")
        axes[1].set_ylim(0, 108)
        axes[1].legend(fontsize=7.8, loc="lower right")

    st_keys = [
        ("SLWM-X1B\n(3:1 Ring)", "flagship_1b"),
        ("Dense-TF\n(Full KV)", "dense_1b"),
        ("Legacy-G10\n(Scalar+Ring)", "legacy_1b"),
    ]
    st_pres = [(l, k) for l, k in st_keys if k in summary_table and summary_table[k] and "streaming_state_mb_T1024" in summary_table[k]]
    if st_pres:
        xs = np.arange(len(st_pres))
        mb64 = [_m_s(k, "streaming_state_mb_T64", scale=1.0)[0] for _, k in st_pres]
        mb1024 = [_m_s(k, "streaming_state_mb_T1024", scale=1.0)[0] for _, k in st_pres]
        axes[2].bar(xs - 0.18, mb64, 0.35, label="Streaming State at T=64 (MB)", color="#5499C7", edgecolor="black")
        axes[2].bar(xs + 0.18, mb1024, 0.35, label="Streaming State at T=1024 (MB)", color="#E74C3C", edgecolor="black")
        axes[2].set_yscale("log")
        axes[2].set_xticks(xs)
        axes[2].set_xticklabels([p[0] for p in st_pres])
        axes[2].set_ylabel("Recurrent / KV Cache State Size (MB, log scale)")
        axes[2].set_title("C. O(1) Ring-Buffer vs O(T) Dense KV State (1B)")
        axes[2].legend(fontsize=7.8, loc="upper left")

    fig.tight_layout()
    p2 = fig_dir / "x1b_memory_and_horizon_extrapolation.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    # -------------------------------------------------------------------------
    # Figure 3: Objective Factorial, Planner Comparison & Spoofing CommitGate
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))
    obj_keys = [
        ("Res-JEPA\n+SIGReg", "abl_slwm_full"),
        ("Direct-JEPA\nNo-Res", "abl_direct_jepa_no_res"),
        ("JEPA\nNo-SIGReg", "abl_jepa_no_sigreg"),
        ("Raw-Pixel\nLinear-MSE", "abl_raw_pixel_mse"),
    ]
    obj_pres = [(l, k) for l, k in obj_keys if k in summary_table and summary_table[k]]
    if obj_pres:
        xl = [p[0] for p in obj_pres]
        pr_m, pr_s = zip(*[_m_s(k, "pr") for _, k in obj_pres])
        s1_m, s1_s = zip(*[_m_s(k, "persist_skill_1") for _, k in obj_pres])
        s4_m, s4_s = zip(*[_m_s(k, "persist_skill_4") for _, k in obj_pres])
        xx = np.arange(len(obj_pres))
        axes[0].bar(xx - 0.25, pr_m, 0.24, yerr=pr_s, error_kw=eb_kw, label="Participation Ratio (%)", color="#5DADE2", edgecolor="black")
        axes[0].bar(xx, s1_m, 0.24, yerr=s1_s, error_kw=eb_kw, label="1-Step Skill vs Persist (%)", color="#27AE60", edgecolor="black")
        axes[0].bar(xx + 0.25, s4_m, 0.24, yerr=s4_s, error_kw=eb_kw, label="4-Step Skill vs Persist (%)", color="#117A65", edgecolor="black")
        axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axes[0].set_xticks(xx)
        axes[0].set_xticklabels(xl)
        axes[0].set_title("A. 3-Seed Objective Ablation (Rank & Multi-Step)")
        axes[0].legend(fontsize=7.6)

    if "flagship_1b" in summary_table and summary_table["flagship_1b"]:
        pl_names = ["Latent-CEM-MPC\n(H=6, 3x48)", "Decoupled Policy\n(Open-Loop Head)", "G8 1-Step\nRandom Shoot (3)"]
        pl_m = [_m_s("flagship_1b", m)[0] for m in ("mpc_success", "open_loop_success", "g8_shoot_success")]
        pl_s = [_m_s("flagship_1b", m)[1] for m in ("mpc_success", "open_loop_success", "g8_shoot_success")]
        axes[1].bar(np.arange(3), pl_m, 0.56, yerr=pl_s, error_kw=eb_kw, color=["#117A65", "#5499C7", "#CD6155"], edgecolor="black")
        axes[1].set_xticks(np.arange(3))
        axes[1].set_xticklabels(pl_names)
        axes[1].set_ylabel("Closed-Loop Success Rate (%)")
        axes[1].set_title("B. 3-Seed Planner Comparison on SLWM-X1B (1B)")
        axes[1].set_ylim(0, 108)

        g_labels = ["Spoofed Action\nRegret (lower=better)", "Unsupported Claim\nRate (lower=better)", "Clean Task\nUsefulness (higher)"]
        l_m = [_m_s("flagship_1b", m, scale=1.0)[0] for m in ("spoof_regret_learned", "unsupp_claim_learned", "clean_usefulness_learned")]
        l_s = [_m_s("flagship_1b", m, scale=1.0)[1] for m in ("spoof_regret_learned", "unsupp_claim_learned", "clean_usefulness_learned")]
        a_m = [_m_s("flagship_1b", m, scale=1.0)[0] for m in ("spoof_regret_always", "unsupp_claim_always", "clean_usefulness_always")]
        a_s = [_m_s("flagship_1b", m, scale=1.0)[1] for m in ("spoof_regret_always", "unsupp_claim_always", "clean_usefulness_always")]
        xg = np.arange(3)
        axes[2].bar(xg - 0.18, a_m, 0.35, yerr=a_s, error_kw=eb_kw, label="Always-Commit (G6/G8 Stub)", color="#E74C3C", edgecolor="black")
        axes[2].bar(xg + 0.18, l_m, 0.35, yerr=l_s, error_kw=eb_kw, label="Innovation PolicyCommitGate", color="#16A085", edgecolor="black")
        axes[2].set_xticks(xg)
        axes[2].set_xticklabels(g_labels, fontsize=8.4)
        axes[2].set_title("C. Defeating G6 Camera Spoofing & Hallucination")
        axes[2].legend(fontsize=7.8)

    fig.tight_layout()
    p3 = fig_dir / "x1b_jepa_mpc_and_spoofing_safety.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved.append(p3)

    # -------------------------------------------------------------------------
    # Figure 4: OOD Long-Horizon (T=76) Transient Barrier & Goal-Switch Overwrite MPC
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8))
    p8_keys = [
        ("SLWM\n1.008B", "flagship_1b", "#117A65"),
        ("SLWM\n453M", "scale_450m", "#1F618D"),
        ("SLWM\n152M", "abl_slwm_full", "#4C78A8"),
        ("Dense-TF\n1.008B", "dense_1b", "#E67E22"),
        ("No-Regs\n152M", "abl_no_registers", "#8E44AD"),
        ("Legacy\n1.008B", "legacy_1b", "#C0392B"),
    ]
    p8_pres = [(l, k, c) for l, k, c in p8_keys if k in summary_table and "sustained_mpc_overall" in summary_table[k]]
    if p8_pres:
        xp8 = np.arange(len(p8_pres))
        cols8 = [c for _, _, c in p8_pres]
        su_m, su_s = zip(*[_m_s(k, "sustained_mpc_overall") for _, k, _ in p8_pres])
        sb_m, sb_s = zip(*[_m_s(k, "sustained_mpc_barrier") for _, k, _ in p8_pres])
        axes[0].bar(xp8 - 0.18, su_m, 0.35, yerr=su_s, error_kw=eb_kw, label="Overall OOD Sustained MPC", color=cols8, edgecolor="black")
        axes[0].bar(xp8 + 0.18, sb_m, 0.35, yerr=sb_s, error_kw=eb_kw, label="1-px Thin-Barrier OOD MPC", color=cols8, alpha=0.55, hatch="//", edgecolor="black")
        axes[0].set_xticks(xp8)
        axes[0].set_xticklabels([l for l, _, _ in p8_pres], fontsize=8.2)
        axes[0].set_ylabel("Closed-Loop Success at T=76 (%)")
        axes[0].set_title("A. T=76 OOD Sustained 0.4s Transient-Goal MPC")
        axes[0].set_ylim(0, 108)
        axes[0].legend(fontsize=7.6, loc="lower left", framealpha=0.92)

        sw_m, sw_s = zip(*[_m_s(k, "goal_switch_mpc_overall") for _, k, _ in p8_pres])
        tr_m, tr_s = zip(*[_m_s(k, "stale_goal_a_trap") for _, k, _ in p8_pres])
        axes[1].bar(xp8 - 0.18, sw_m, 0.35, yerr=sw_s, error_kw=eb_kw, label="Goal-B Overwrite Success (%)", color=cols8, edgecolor="black")
        axes[1].bar(xp8 + 0.18, tr_m, 0.35, yerr=tr_s, error_kw=eb_kw, label="Stale Goal-A Trap Rate (%)", color="#95A5A6", hatch="xx", edgecolor="black")
        axes[1].set_xticks(xp8)
        axes[1].set_xticklabels([l for l, _, _ in p8_pres], fontsize=8.2)
        axes[1].set_ylabel("Rate at T=76 (%)")
        axes[1].set_title("B. Mid-Episode Goal-B Overwrite (t=36..39 > T_train)")
        axes[1].set_ylim(0, 108)
        axes[1].legend(fontsize=7.6, loc="upper right")

    forest_items = [
        ("vs Dense-1B: T=1024 Recall", "1b_pooled_slwm_vs_dense_T1024_assoc"),
        ("vs Dense-1B: OOD Goal-Switch", "1b_pooled_slwm_vs_dense_goal_switch_mpc"),
        ("vs Dense-1B: OOD Sustained", "1b_pooled_slwm_vs_dense_sustained_mpc"),
        ("vs Dense-1B: Thin-Barrier MPC", "1b_pooled_slwm_vs_dense_thin_barrier_mpc"),
        ("vs Dense-1B: Overall MPC", "1b_pooled_slwm_vs_dense_mpc_success"),
        ("vs Legacy-1B: OOD Goal-Switch", "1b_pooled_slwm_vs_legacy_goal_switch_mpc"),
        ("vs Legacy-1B: Overall MPC", "1b_pooled_slwm_vs_legacy_mpc_success"),
        ("vs Legacy-1B: Real AV R@1", "1b_pooled_slwm_vs_legacy_av_r1"),
    ]
    f_pres = [(lbl, paired_stats[k]) for lbl, k in forest_items if k in paired_stats]
    if f_pres:
        yf = np.arange(len(f_pres))
        axes[2].axvline(0.0, color="#C0392B", linestyle="--", linewidth=1.2)
        for idx_f, (_, st_ci) in enumerate(f_pres):
            m_pp = st_ci["mean"] * 100.0
            lo99, hi99 = st_ci["ci99_low"] * 100.0, st_ci["ci99_high"] * 100.0
            lo95, hi95 = st_ci["ci95_low"] * 100.0, st_ci["ci95_high"] * 100.0
            axes[2].plot([lo99, hi99], [idx_f, idx_f], color="#7FB3D5", linewidth=2.2, label="99% Paired CI" if idx_f == 0 else None)
            axes[2].plot([lo95, hi95], [idx_f, idx_f], color="#117A65", linewidth=4.2, label="95% Paired CI" if idx_f == 0 else None)
            axes[2].plot([m_pp], [idx_f], "o", color="#0B5345", markersize=5.5)
        axes[2].set_yticks(yf)
        axes[2].set_yticklabels([lbl for lbl, _ in f_pres], fontsize=8.1)
        axes[2].set_xlabel("SLWM-X1B Advantage (Delta percentage points)")
        axes[2].set_title("C. Pooled 3-Seed 1.008B Paired Bootstrap CIs")
        axes[2].legend(fontsize=7.6, loc="lower right")

    fig.tight_layout()
    p4 = fig_dir / "x1b_long_horizon_ood_goal_switch_mpc.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved.append(p4)

    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--study-dir",
        type=Path,
        default=REPO_ROOT / "experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111",
    )
    args = ap.parse_args()

    bundles = load_run_bundle(args.study_dir / "runs")
    groups = {
        "flagship_1b": "x1b-flagship-slwm-1b",
        "scale_450m": "x1b-scale-slwm-450m",
        "dense_1b": "x1b-baseline-dense-1b",
        "legacy_1b": "x1b-baseline-legacy-1b",
        "abl_slwm_full": "x1b-abl-slwm_full",
        "abl_no_registers": "x1b-abl-no_registers",
        "abl_scalar_retention": "x1b-abl-scalar_retention",
        "abl_dense_attn": "x1b-abl-dense_attn",
        "abl_separate_cores": "x1b-abl-separate_cores",
        "abl_raw_pixel_mse": "x1b-abl-raw_pixel_mse",
        "abl_direct_jepa_no_res": "x1b-abl-direct_jepa_no_res",
        "abl_jepa_no_sigreg": "x1b-abl-jepa_no_sigreg",
    }
    summary_table = {name: aggregate_seed_group(bundles, pfx) for name, pfx in groups.items()}

    def _pooled_vecs(pfx: str, extractor) -> np.ndarray | None:
        arrs = []
        for s in (2001, 2002, 2003):
            tag = f"{pfx}-s{s}"
            if tag in bundles:
                val = extractor(bundles[tag]["eval"])
                if val is not None:
                    arrs.append(np.asarray(val, dtype=np.float64))
        return np.concatenate(arrs, axis=0) if arrs else None

    def _thin_barrier_vec(ev: dict[str, Any]) -> np.ndarray:
        sv = np.asarray(ev["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"], dtype=np.float64)
        return sv[0::2]  # Even-indexed episodes in Panel 5 have has_bar == True

    paired_stats: dict[str, Any] = {}

    # 3-Seed Pooled 1.008B Paired Bootstrap 95% & 99% CIs (N=288 Panel 5, N=144 Thin-Barrier, N=192 Panel 8, N=48 Recall)
    f1b_mpc = _pooled_vecs("x1b-flagship-slwm-1b", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    f1b_bar = _pooled_vecs("x1b-flagship-slwm-1b", _thin_barrier_vec)
    f1b_g8 = _pooled_vecs("x1b-flagship-slwm-1b", lambda e: e["closed_loop_mpc"]["g8_random_shooting_3"]["succ_vec"])
    f1b_ol = _pooled_vecs("x1b-flagship-slwm-1b", lambda e: e["closed_loop_mpc"]["open_loop_policy"]["succ_vec"])
    f1b_t512 = _pooled_vecs("x1b-flagship-slwm-1b", lambda e: e["associative_recall_512"]["T512_vec"])
    f1b_t1024 = _pooled_vecs("x1b-flagship-slwm-1b", lambda e: e["associative_recall_512"].get("T1024_vec"))
    f1b_sust = _pooled_vecs("x1b-flagship-slwm-1b", lambda e: e.get("long_horizon_memory_mpc", {}).get("sustained_transient_barrier_mpc", {}).get("succ_vec"))
    f1b_sw = _pooled_vecs("x1b-flagship-slwm-1b", lambda e: e.get("long_horizon_memory_mpc", {}).get("mid_episode_goal_switch_mpc", {}).get("succ_vec"))
    f1b_av = _pooled_vecs("x1b-flagship-slwm-1b", lambda e: e["real_multimodal_and_text"].get("per_item_r1"))

    d1b_mpc = _pooled_vecs("x1b-baseline-dense-1b", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    d1b_bar = _pooled_vecs("x1b-baseline-dense-1b", _thin_barrier_vec)
    d1b_t512 = _pooled_vecs("x1b-baseline-dense-1b", lambda e: e["associative_recall_512"]["T512_vec"])
    d1b_t1024 = _pooled_vecs("x1b-baseline-dense-1b", lambda e: e["associative_recall_512"].get("T1024_vec"))
    d1b_sust = _pooled_vecs("x1b-baseline-dense-1b", lambda e: e.get("long_horizon_memory_mpc", {}).get("sustained_transient_barrier_mpc", {}).get("succ_vec"))
    d1b_sw = _pooled_vecs("x1b-baseline-dense-1b", lambda e: e.get("long_horizon_memory_mpc", {}).get("mid_episode_goal_switch_mpc", {}).get("succ_vec"))

    l1b_mpc = _pooled_vecs("x1b-baseline-legacy-1b", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    l1b_bar = _pooled_vecs("x1b-baseline-legacy-1b", _thin_barrier_vec)
    l1b_t512 = _pooled_vecs("x1b-baseline-legacy-1b", lambda e: e["associative_recall_512"]["T512_vec"])
    l1b_t1024 = _pooled_vecs("x1b-baseline-legacy-1b", lambda e: e["associative_recall_512"].get("T1024_vec"))
    l1b_sust = _pooled_vecs("x1b-baseline-legacy-1b", lambda e: e.get("long_horizon_memory_mpc", {}).get("sustained_transient_barrier_mpc", {}).get("succ_vec"))
    l1b_sw = _pooled_vecs("x1b-baseline-legacy-1b", lambda e: e.get("long_horizon_memory_mpc", {}).get("mid_episode_goal_switch_mpc", {}).get("succ_vec"))
    l1b_av = _pooled_vecs("x1b-baseline-legacy-1b", lambda e: e["real_multimodal_and_text"].get("per_item_r1"))

    def _add_pair(key: str, a: np.ndarray | None, b: np.ndarray | None) -> None:
        if a is not None and b is not None and len(a) == len(b) and len(a) > 0:
            paired_stats[key] = paired_bootstrap_95_99(a, b)

    _add_pair("1b_pooled_slwm_vs_dense_mpc_success", f1b_mpc, d1b_mpc)
    _add_pair("1b_pooled_slwm_vs_dense_thin_barrier_mpc", f1b_bar, d1b_bar)
    _add_pair("1b_pooled_slwm_vs_dense_T512_assoc", f1b_t512, d1b_t512)
    _add_pair("1b_pooled_slwm_vs_dense_T1024_assoc", f1b_t1024, d1b_t1024)
    _add_pair("1b_pooled_slwm_vs_dense_sustained_mpc", f1b_sust, d1b_sust)
    _add_pair("1b_pooled_slwm_vs_dense_goal_switch_mpc", f1b_sw, d1b_sw)

    _add_pair("1b_pooled_slwm_vs_legacy_mpc_success", f1b_mpc, l1b_mpc)
    _add_pair("1b_pooled_slwm_vs_legacy_thin_barrier_mpc", f1b_bar, l1b_bar)
    _add_pair("1b_pooled_slwm_vs_legacy_T512_assoc", f1b_t512, l1b_t512)
    _add_pair("1b_pooled_slwm_vs_legacy_T1024_assoc", f1b_t1024, l1b_t1024)
    _add_pair("1b_pooled_slwm_vs_legacy_sustained_mpc", f1b_sust, l1b_sust)
    _add_pair("1b_pooled_slwm_vs_legacy_goal_switch_mpc", f1b_sw, l1b_sw)
    _add_pair("1b_pooled_slwm_vs_legacy_av_r1", f1b_av, l1b_av)

    _add_pair("1b_pooled_cem_mpc_vs_g8_shooting", f1b_mpc, f1b_g8)
    _add_pair("1b_pooled_cem_mpc_vs_open_loop", f1b_mpc, f1b_ol)

    # 3-Seed Pooled 152M Factorial Paired Bootstrap CIs
    v_full_mpc = _pooled_vecs("x1b-abl-slwm_full", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    v_full_g8 = _pooled_vecs("x1b-abl-slwm_full", lambda e: e["closed_loop_mpc"]["g8_random_shooting_3"]["succ_vec"])
    v_dir_mpc = _pooled_vecs("x1b-abl-direct_jepa_no_res", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    v_sep_mpc = _pooled_vecs("x1b-abl-separate_cores", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    v_dense_mpc = _pooled_vecs("x1b-abl-dense_attn", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    v_full_sw = _pooled_vecs("x1b-abl-slwm_full", lambda e: e.get("long_horizon_memory_mpc", {}).get("mid_episode_goal_switch_mpc", {}).get("succ_vec"))
    v_noreg_sw = _pooled_vecs("x1b-abl-no_registers", lambda e: e.get("long_horizon_memory_mpc", {}).get("mid_episode_goal_switch_mpc", {}).get("succ_vec"))
    v_dense_sw = _pooled_vecs("x1b-abl-dense_attn", lambda e: e.get("long_horizon_memory_mpc", {}).get("mid_episode_goal_switch_mpc", {}).get("succ_vec"))
    v_full_t1024 = _pooled_vecs("x1b-abl-slwm_full", lambda e: e["associative_recall_512"].get("T1024_vec"))
    v_scal_t1024 = _pooled_vecs("x1b-abl-scalar_retention", lambda e: e["associative_recall_512"].get("T1024_vec"))
    v_full_av = _pooled_vecs("x1b-abl-slwm_full", lambda e: e["real_multimodal_and_text"].get("per_item_r1"))
    v_sep_av = _pooled_vecs("x1b-abl-separate_cores", lambda e: e["real_multimodal_and_text"].get("per_item_r1"))
    v_raw_av = _pooled_vecs("x1b-abl-raw_pixel_mse", lambda e: e["real_multimodal_and_text"].get("per_item_r1"))

    _add_pair("152m_pooled_cem_mpc_vs_g8_shooting", v_full_mpc, v_full_g8)
    _add_pair("152m_pooled_slwm_vs_direct_jepa_mpc", v_full_mpc, v_dir_mpc)
    _add_pair("152m_pooled_slwm_vs_separate_cores_mpc", v_full_mpc, v_sep_mpc)
    _add_pair("152m_pooled_slwm_vs_dense_attn_mpc", v_full_mpc, v_dense_mpc)
    _add_pair("152m_pooled_slwm_vs_no_registers_goal_switch_mpc", v_full_sw, v_noreg_sw)
    _add_pair("152m_pooled_slwm_vs_dense_attn_goal_switch_mpc", v_full_sw, v_dense_sw)
    _add_pair("152m_pooled_slwm_vs_scalar_T1024_assoc", v_full_t1024, v_scal_t1024)
    _add_pair("152m_pooled_slwm_vs_separate_cores_av_r1", v_full_av, v_sep_av)
    _add_pair("152m_pooled_slwm_vs_raw_pixel_av_r1", v_full_av, v_raw_av)

    out_summary = {
        "total_verified_runs": len(bundles),
        "groups": summary_table,
        "paired_bootstrap_95ci": paired_stats,
    }
    (args.study_dir / "summary_metrics.json").write_text(json.dumps(out_summary, indent=2, sort_keys=True) + "\n")
    figs = generate_figures(bundles, summary_table, paired_stats, args.study_dir / "figures")
    print("ANALYSIS_COMPLETE", len(bundles), [str(p) for p in figs], flush=True)


if __name__ == "__main__":
    main()
