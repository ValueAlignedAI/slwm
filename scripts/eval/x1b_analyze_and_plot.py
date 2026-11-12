"""Aggregate verified Study X1B results, compute paired bootstrap CIs, and generate publication figures."""

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

from evals.x1b_eval_suite import paired_bootstrap_ci


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
        "transient_cont": _collect(lambda e: e["transient_goal_retention"]["continuous_success"]["mean"]),
        "transient_04s": _collect(lambda e: e["transient_goal_retention"]["transient_04s_success"]["mean"]),
        "transient_drop_pp": _collect(lambda e: e["transient_goal_retention"]["transient_drop_pp"]["mean"]),
        "mpc_success": _collect(lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["success_rate"]["mean"]),
        "mpc_pusht_open": _collect(lambda e: e["closed_loop_mpc"]["latent_cem_mpc"].get("pusht_open_success")),
        "mpc_thin_barrier": _collect(lambda e: e["closed_loop_mpc"]["latent_cem_mpc"].get("thin_barrier_success")),
        "g8_shoot_success": _collect(lambda e: e["closed_loop_mpc"]["g8_random_shooting_3"]["success_rate"]["mean"]),
        "open_loop_success": _collect(lambda e: e["closed_loop_mpc"]["open_loop_policy"]["success_rate"]["mean"]),
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
        "t256_slots_s": _collect(lambda e: e.get("throughput_scaling", {}).get("T256_slots_per_s")),
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


def generate_figures(bundles: dict[str, dict[str, Any]], summary_table: dict[str, Any], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.28,
        "figure.dpi": 160,
    })
    saved: list[Path] = []

    # Figure 1: 1.008B Head-to-Head & Scaling Law (152M -> 453M -> 1.008B)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
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

        mpc_vals = [summary_table[k]["mpc_success"]["mean"] * 100.0 for _, k in present]
        bar_vals = [summary_table[k]["mpc_thin_barrier"]["mean"] * 100.0 for _, k in present]
        axes[0].bar(xp - 0.18, mpc_vals, 0.35, label="Overall Closed-Loop MPC", color=colors, edgecolor="black", linewidth=0.7)
        axes[0].bar(xp + 0.18, bar_vals, 0.35, label="1-px Thin-Barrier MPC", color=colors, alpha=0.55, hatch="//", edgecolor="black", linewidth=0.7)
        axes[0].set_xticks(xp)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("Closed-Loop Manipulation Success (%)")
        axes[0].set_title("A. 1B Closed-Loop & Thin-Barrier MPC")
        axes[0].set_ylim(0, 104)
        axes[0].legend(fontsize=8, loc="upper right")

        pr_vals = [summary_table[k]["pr"]["mean"] * 100.0 for _, k in present]
        pers_vals = [summary_table[k]["persist_skill_1"]["mean"] * 100.0 for _, k in present]
        axes[1].bar(xp - 0.18, pr_vals, 0.35, label="Latent Participation Ratio (%)", color="#5DADE2", edgecolor="black", linewidth=0.7)
        axes[1].bar(xp + 0.18, pers_vals, 0.35, label="1-Step Skill vs Persistence (%)", color="#27AE60", edgecolor="black", linewidth=0.7)
        axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.9)
        axes[1].set_xticks(xp)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_title("B. Representation Rank & Forecast Skill")
        axes[1].legend(fontsize=8, loc="upper left")

        av_vals = [summary_table[k].get("av_r1", {"mean": 0.0})["mean"] * 100.0 for _, k in present]
        axes[2].bar(labels, av_vals, color=colors, edgecolor="black", linewidth=0.7)
        axes[2].set_ylabel("Real Audio-Visual Retrieval R@1 (%)")
        axes[2].set_title("C. Real GreatestHits + STARSS23 Grounding")
        axes[2].set_ylim(0, 104)

    fig.tight_layout()
    p1 = fig_dir / "x1b_scaling_and_1b_head_to_head.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # Figure 2: Associative Horizon Extrapolation (T=32 -> 512) & Transient 0.4s Command Retention
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5))
    horizons = [32, 128, 256, 512]
    arch_curves = [
        ("SLWM-X1B (1.008B)", "flagship_1b", "#117A65", "-o"),
        ("SLWM-X450M (453M)", "scale_450m", "#1F618D", "--s"),
        ("SLWM-Full (152M, 3 seeds)", "abl_slwm_full", "#5DADE2", ":d"),
        ("Dense-Transformer (1.008B)", "dense_1b", "#E67E22", "--^"),
        ("No-Registers (152M)", "abl_no_registers", "#8E44AD", "-.x"),
        ("Legacy Scalar-Hybrid (1.008B)", "legacy_1b", "#C0392B", ":v"),
    ]
    for title_lbl, k, col, style in arch_curves:
        if k in summary_table and summary_table[k]:
            ys = [summary_table[k][f"assoc_T{h}"]["mean"] * 100.0 for h in horizons]
            axes[0].plot(horizons, ys, style, color=col, linewidth=2.1, markersize=6, label=title_lbl)
    axes[0].set_xlabel("Evaluation Sequence Horizon T (trained at T=32)")
    axes[0].set_ylabel("Associative Overwrite Recall (%)")
    axes[0].set_title("A. Key-Value Overwrite Extrapolation (T=32 -> 512)")
    axes[0].set_ylim(-2, 106)
    axes[0].legend(fontsize=8, loc="center right")

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
        c_v = [summary_table[k]["transient_cont"]["mean"] * 100.0 for _, k in mem_pres]
        t_v = [summary_table[k]["transient_04s"]["mean"] * 100.0 for _, k in mem_pres]
        axes[1].bar(x - w / 2, c_v, w, label="Continuous Goal Cue", color="#3498DB", edgecolor="black")
        axes[1].bar(x + w / 2, t_v, w, label="Transient 0.4s Cue (No Latching)", color="#1ABC9C", edgecolor="black")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([p[0] for p in mem_pres])
        axes[1].set_ylabel("Closed-Loop Goal Success (%)")
        axes[1].set_title("B. Resolving Transient-Command Amnesia")
        axes[1].set_ylim(0, 105)
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    p2 = fig_dir / "x1b_memory_and_horizon_extrapolation.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    # Figure 3: Objective Factorial, CEM-MPC vs G8 Shooting, and Spoofing CommitGate Safety
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    obj_keys = [
        ("Res-JEPA\n+SIGReg", "abl_slwm_full"),
        ("Direct-JEPA\nNo-Res", "abl_direct_jepa_no_res"),
        ("JEPA\nNo-SIGReg", "abl_jepa_no_sigreg"),
        ("Raw-Pixel\nLinear-MSE", "abl_raw_pixel_mse"),
    ]
    obj_pres = [(l, k) for l, k in obj_keys if k in summary_table and summary_table[k]]
    if obj_pres:
        xl = [p[0] for p in obj_pres]
        pr_v = [summary_table[k]["pr"]["mean"] * 100.0 for _, k in obj_pres]
        sk_v = [summary_table[k]["persist_skill_1"]["mean"] * 100.0 for _, k in obj_pres]
        xx = np.arange(len(obj_pres))
        axes[0].bar(xx - 0.18, pr_v, 0.35, label="Latent Participation Ratio (%)", color="#5DADE2", edgecolor="black")
        axes[0].bar(xx + 0.18, sk_v, 0.35, label="Forecast Skill vs Persistence (%)", color="#27AE60", edgecolor="black")
        axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axes[0].set_xticks(xx)
        axes[0].set_xticklabels(xl)
        axes[0].set_title("A. Objective Ablation (Anti-Collapse & Skill)")
        axes[0].legend(fontsize=8)

    if "flagship_1b" in summary_table and summary_table["flagship_1b"]:
        f1b = summary_table["flagship_1b"]
        pl_names = ["Latent-CEM-MPC\n(H=6, 3x48)", "Open-Loop\nPolicy Head", "G8 1-Step\nRandom Shoot (3)"]
        pl_vals = [
            f1b["mpc_success"]["mean"] * 100.0,
            f1b["open_loop_success"]["mean"] * 100.0,
            f1b["g8_shoot_success"]["mean"] * 100.0,
        ]
        axes[1].bar(pl_names, pl_vals, color=["#117A65", "#5499C7", "#CD6155"], edgecolor="black")
        axes[1].set_ylabel("Closed-Loop Success Rate (%)")
        axes[1].set_title("B. Planner Comparison on SLWM-X1B (1.008B)")
        axes[1].set_ylim(0, 102)

        # Spoofing & Unsupported Claim Comparison
        g_labels = ["Spoofed Action\nRegret (lower=better)", "Unsupported Claim\nRate (lower=better)", "Clean Task\nUsefulness (higher)"]
        learned_g = [
            f1b["spoof_regret_learned"]["mean"],
            f1b["unsupp_claim_learned"]["mean"],
            f1b["clean_usefulness_learned"]["mean"],
        ]
        always_g = [
            f1b["spoof_regret_always"]["mean"],
            f1b["unsupp_claim_always"]["mean"],
            f1b["clean_usefulness_always"]["mean"],
        ]
        xg = np.arange(3)
        axes[2].bar(xg - 0.18, always_g, 0.35, label="Always-Commit (G6/G8 Stub)", color="#E74C3C", edgecolor="black")
        axes[2].bar(xg + 0.18, learned_g, 0.35, label="Innovation PolicyCommitGate", color="#16A085", edgecolor="black")
        axes[2].set_xticks(xg)
        axes[2].set_xticklabels(g_labels, fontsize=8.5)
        axes[2].set_title("C. Defeating G6 Camera Spoofing & Hallucination")
        axes[2].legend(fontsize=8)

    fig.tight_layout()
    p3 = fig_dir / "x1b_jepa_mpc_and_spoofing_safety.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    saved.append(p3)

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

    # Compute exact paired bootstrap 95% CIs on 1.008B Flagship vs Baselines + Pooled 3-Seed 152M Factorials
    paired_stats: dict[str, Any] = {}
    if "x1b-flagship-slwm-1b-s2001" in bundles and "x1b-baseline-dense-1b-s2001" in bundles:
        f_ev = bundles["x1b-flagship-slwm-1b-s2001"]["eval"]
        d_ev = bundles["x1b-baseline-dense-1b-s2001"]["eval"]
        paired_stats["1b_slwm_vs_dense_mpc_success"] = paired_bootstrap_ci(
            np.asarray(f_ev["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"]),
            np.asarray(d_ev["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"]),
        )
        paired_stats["1b_slwm_vs_dense_T512_assoc"] = paired_bootstrap_ci(
            np.asarray(f_ev["associative_recall_512"]["T512_vec"]),
            np.asarray(d_ev["associative_recall_512"]["T512_vec"]),
        )
        paired_stats["1b_cem_mpc_vs_g8_shooting"] = paired_bootstrap_ci(
            np.asarray(f_ev["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"]),
            np.asarray(f_ev["closed_loop_mpc"]["g8_random_shooting_3"]["succ_vec"]),
        )
        paired_stats["1b_cem_mpc_vs_open_loop"] = paired_bootstrap_ci(
            np.asarray(f_ev["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"]),
            np.asarray(f_ev["closed_loop_mpc"]["open_loop_policy"]["succ_vec"]),
        )
    if "x1b-flagship-slwm-1b-s2001" in bundles and "x1b-baseline-legacy-1b-s2001" in bundles:
        f_ev = bundles["x1b-flagship-slwm-1b-s2001"]["eval"]
        l_ev = bundles["x1b-baseline-legacy-1b-s2001"]["eval"]
        paired_stats["1b_slwm_vs_legacy_mpc_success"] = paired_bootstrap_ci(
            np.asarray(f_ev["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"]),
            np.asarray(l_ev["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"]),
        )
        paired_stats["1b_slwm_vs_legacy_T512_assoc"] = paired_bootstrap_ci(
            np.asarray(f_ev["associative_recall_512"]["T512_vec"]),
            np.asarray(l_ev["associative_recall_512"]["T512_vec"]),
        )
        if "per_item_r1" in f_ev["real_multimodal_and_text"] and "per_item_r1" in l_ev["real_multimodal_and_text"]:
            paired_stats["1b_slwm_vs_legacy_av_r1"] = paired_bootstrap_ci(
                np.asarray(f_ev["real_multimodal_and_text"]["per_item_r1"]),
                np.asarray(l_ev["real_multimodal_and_text"]["per_item_r1"]),
            )

    def _pooled_vecs(pfx: str, extractor) -> np.ndarray | None:
        arrs = [
            np.asarray(extractor(bundles[f"{pfx}-s{s}"]["eval"]))
            for s in (2001, 2002, 2003)
            if f"{pfx}-s{s}" in bundles
        ]
        return np.concatenate(arrs, axis=0) if len(arrs) == 3 else None

    v_full_mpc = _pooled_vecs("x1b-abl-slwm_full", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    v_full_g8 = _pooled_vecs("x1b-abl-slwm_full", lambda e: e["closed_loop_mpc"]["g8_random_shooting_3"]["succ_vec"])
    v_dir_mpc = _pooled_vecs("x1b-abl-direct_jepa_no_res", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    v_sep_mpc = _pooled_vecs("x1b-abl-separate_cores", lambda e: e["closed_loop_mpc"]["latent_cem_mpc"]["succ_vec"])
    v_full_t512 = _pooled_vecs("x1b-abl-slwm_full", lambda e: e["associative_recall_512"]["T512_vec"])
    v_scal_t512 = _pooled_vecs("x1b-abl-scalar_retention", lambda e: e["associative_recall_512"]["T512_vec"])
    v_full_av = _pooled_vecs("x1b-abl-slwm_full", lambda e: e["real_multimodal_and_text"]["per_item_r1"])
    v_sep_av = _pooled_vecs("x1b-abl-separate_cores", lambda e: e["real_multimodal_and_text"]["per_item_r1"])
    v_raw_av = _pooled_vecs("x1b-abl-raw_pixel_mse", lambda e: e["real_multimodal_and_text"]["per_item_r1"])

    if v_full_mpc is not None and v_full_g8 is not None:
        paired_stats["152m_pooled_cem_mpc_vs_g8_shooting"] = paired_bootstrap_ci(v_full_mpc, v_full_g8)
    if v_full_mpc is not None and v_dir_mpc is not None:
        paired_stats["152m_pooled_slwm_vs_direct_jepa_mpc"] = paired_bootstrap_ci(v_full_mpc, v_dir_mpc)
    if v_full_mpc is not None and v_sep_mpc is not None:
        paired_stats["152m_pooled_slwm_vs_separate_cores_mpc"] = paired_bootstrap_ci(v_full_mpc, v_sep_mpc)
    if v_full_t512 is not None and v_scal_t512 is not None:
        paired_stats["152m_pooled_slwm_vs_scalar_T512_assoc"] = paired_bootstrap_ci(v_full_t512, v_scal_t512)
    if v_full_av is not None and v_sep_av is not None:
        paired_stats["152m_pooled_slwm_vs_separate_cores_av_r1"] = paired_bootstrap_ci(v_full_av, v_sep_av)
    if v_full_av is not None and v_raw_av is not None:
        paired_stats["152m_pooled_slwm_vs_raw_pixel_av_r1"] = paired_bootstrap_ci(v_full_av, v_raw_av)

    out_summary = {
        "total_verified_runs": len(bundles),
        "groups": summary_table,
        "paired_bootstrap_95ci": paired_stats,
    }
    (args.study_dir / "summary_metrics.json").write_text(json.dumps(out_summary, indent=2, sort_keys=True) + "\n")
    figs = generate_figures(bundles, summary_table, args.study_dir / "figures")
    print("ANALYSIS_COMPLETE", len(bundles), [str(p) for p in figs], flush=True)


if __name__ == "__main__":
    main()
