"""Aggregate verified Wave 7 Spatial-Grid Grounding runs, compute Seed-Aware Hierarchical CIs, & plot Figures 9-10."""

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


def load_wave7_bundles(w7_dir: Path) -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    if not w7_dir.is_dir():
        return bundles
    for d in sorted(w7_dir.iterdir()):
        if d.is_dir() and (d / "verified.json").is_file() and (d / "eval_metrics.json").is_file():
            bundles[d.name] = {
                "receipt": json.loads((d / "receipt.json").read_text()),
                "eval": json.loads((d / "eval_metrics.json").read_text()),
                "verified": json.loads((d / "verified.json").read_text()),
            }
    return bundles


def aggregate_w7_cohort(bundles: dict[str, dict[str, Any]], cohort: str) -> dict[str, Any]:
    pref = f"x1b-{cohort}-s"
    members = [v for k, v in sorted(bundles.items()) if k.startswith(pref)]
    if not members:
        return {}

    def _stat(fn) -> dict[str, Any]:
        vals = [float(fn(m["eval"])) for m in members if fn(m["eval"]) is not None]
        if not vals:
            return {}
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals,
        }

    return {
        "n_seeds": len(members),
        "strict_params": int(members[0]["receipt"]["parameter_counts"]["strict_total"]),
        "unseen_mppi": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["geometry_free_mppi_success"]),
        "explicit_latch_reactive": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["explicit_latch_reactive_success"]),
        "open_loop_policy": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["open_loop_policy_success"]),
        "vis_obj_rmse": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["vis_decoded_obj_rmse"]),
        "vis_ee_rmse": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["vis_decoded_ee_rmse"]),
        "unseen_off_center": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["by_unseen_family"]["off_center_wall"]["mppi_success"]),
        "unseen_dual_chicane": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["by_unseen_family"]["dual_chicane"]["mppi_success"]),
        "unseen_multi_clutter": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["by_unseen_family"]["multi_block_clutter"]["mppi_success"]),
        "unseen_dynamics_shift": _stat(lambda e: e["unseen_layout_deprivileged_mpc"]["by_unseen_family"]["dynamics_shift"]["mppi_success"]),
        "T160_goal_b_switch": _stat(lambda e: e["beyond_window_T160_distractor_switch"]["T160_post_perturbation_goal_b_success"]),
        "T160_stale_goal_a": _stat(lambda e: e["beyond_window_T160_distractor_switch"]["T160_stale_goal_a_capture"]),
        "latch_mse_inside_W64": _stat(lambda e: e["beyond_window_T160_distractor_switch"]["latch_mse_inside_W64_t24_64"]),
        "latch_mse_beyond_W64": _stat(lambda e: e["beyond_window_T160_distractor_switch"]["latch_mse_beyond_W64_t92_160"]),
        "symbolic_key_drift": _stat(lambda e: e["symbolic_key_drift_and_recall"]["symbolic_key_visual_drift"]),
        "dyn_assoc_T64": _stat(lambda e: e["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T64"]),
        "dyn_assoc_T256": _stat(lambda e: e["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T256"]),
        "dyn_assoc_T512": _stat(lambda e: e["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T512"]),
        "dyn_assoc_T1024": _stat(lambda e: e["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"]),
        "unseen_corruption_auroc": _stat(lambda e: e["unseen_ood_sensor_corruptions"]["mean_unseen_ood_corruption_auroc"]),
        "frozen_camera_auroc": _stat(lambda e: e["unseen_ood_sensor_corruptions"]["frozen_camera_frame_auroc"]),
        "replay_swap_auroc": _stat(lambda e: e["unseen_ood_sensor_corruptions"]["cross_episode_replay_swap_auroc"]),
        "rot90_auroc": _stat(lambda e: e["unseen_ood_sensor_corruptions"]["unseen_90deg_rotation_auroc"]),
        "half_occluded_auroc": _stat(lambda e: e["unseen_ood_sensor_corruptions"]["half_occluded_camera_auroc"]),
        "dev_step4_mm": _stat(lambda e: e["grounded_spoken_robot_dev"].get("step4_mm")),
        "dev_spoken_disc": _stat(lambda e: e["grounded_spoken_robot_dev"].get("spoken_16tok_discrimination_acc")),
        "vault_step4_mm": _stat(lambda e: e["untouched_vault_spoken_robot"].get("step4_mm")),
        "vault_step4_mm_norm250": _stat(lambda e: e["untouched_vault_spoken_robot"].get("step4_mm_norm250")),
        "vault_plan_gain_mm": _stat(lambda e: e["untouched_vault_spoken_robot"].get("causal_plan_gain_mm")),
        "vault_spoken_disc": _stat(lambda e: e["untouched_vault_spoken_robot"].get("spoken_16tok_discrimination_acc")),
    }


def _seed_matrix(bundles: dict[str, dict[str, Any]], cohort: str, fn) -> np.ndarray | None:
    pref = f"x1b-{cohort}-s"
    rows = [
        np.asarray(fn(v["eval"]), dtype=np.float64)
        for k, v in sorted(bundles.items())
        if k.startswith(pref) and fn(v["eval"]) is not None
    ]
    if not rows or len({r.shape for r in rows}) != 1:
        return None
    return np.stack(rows, axis=0)


def compute_wave7_hierarchical_attributions(bundles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ref_c = "w7_slwm_450m_full"
    comps = (
        "w7_sliding_window_w64_450m",
        "w7_diagonal_ssm_450m",
        "w7_loo_no_key_shield_450m",
    )
    vec_specs = (
        ("unseen_mppi", lambda e: e["unseen_layout_deprivileged_mpc"]["geometry_free_mppi_succ_vec"]),
        ("T160_goal_b_switch", lambda e: e["beyond_window_T160_distractor_switch"]["T160_succ_vec"]),
        ("dyn_assoc_T1024", lambda e: e["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024_vec"]),
        ("vault_step4_mm", lambda e: e["untouched_vault_spoken_robot"].get("step4_mm_vec")),
        ("vault_spoken_disc", lambda e: e["untouched_vault_spoken_robot"].get("disc_vec")),
    )
    out: dict[str, Any] = {}
    for comp in comps:
        for mname, fn in vec_specs:
            a = _seed_matrix(bundles, ref_c, fn)
            b = _seed_matrix(bundles, comp, fn)
            if a is not None and b is not None and a.shape == b.shape:
                out[f"full_vs_{comp}_{mname}"] = hierarchical_seed_bootstrap_ci(a, b)

    mppi_m = _seed_matrix(bundles, ref_c, lambda e: e["unseen_layout_deprivileged_mpc"]["geometry_free_mppi_succ_vec"])
    reac_m = _seed_matrix(bundles, ref_c, lambda e: e["unseen_layout_deprivileged_mpc"]["explicit_latch_reactive_succ_vec"])
    if mppi_m is not None and reac_m is not None and mppi_m.shape == reac_m.shape:
        out["full_mppi_vs_explicit_latch_reactive"] = hierarchical_seed_bootstrap_ci(mppi_m, reac_m)
    return out


def render_wave7_figures(groups: dict[str, dict[str, Any]], fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9.4,
            "axes.titlesize": 10.4,
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
        return (float(d.get("mean", 0.0)) * sc, float(d.get("std", 0.0)) * sc) if d else (0.0, 0.0)

    order = [
        ("Full SLWM-v2\n(453M Spatial)", "w7_slwm_450m_full", "#117A65"),
        ("SWin-W64\n(O(1) KV)", "w7_sliding_window_w64_450m", "#D35400"),
        ("DiagSSM\n(Mamba-JEPA)", "w7_diagonal_ssm_450m", "#8E44AD"),
        ("-KeyShield\n(LOO)", "w7_loo_no_key_shield_450m", "#C0392B"),
    ]
    pres = [(l, k, c) for l, k, c in order if k in groups and groups[k]]
    if not pres:
        return saved

    xp = np.arange(len(pres))
    lbls = [p[0] for p in pres]
    cols = [p[2] for p in pres]

    # Figure 9: De-Privileged Unseen-Layout MPC, Beyond-Window T=160 Switch, & Latch MSE
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    mp_m, mp_s = zip(*[_ms(k, "unseen_mppi") for _, k, _ in pres])
    re_m, re_s = zip(*[_ms(k, "explicit_latch_reactive") for _, k, _ in pres])
    op_m, op_s = zip(*[_ms(k, "open_loop_policy") for _, k, _ in pres])
    axes[0].bar(xp - 0.24, mp_m, 0.24, yerr=mp_s, error_kw=eb, color="#117A65", edgecolor="black", label="Geometry-Free Subgoal MPPI")
    axes[0].bar(xp, re_m, 0.24, yerr=re_s, error_kw=eb, color="#7F8C8D", edgecolor="black", label="Explicit-Latch Reactive")
    axes[0].bar(xp + 0.24, op_m, 0.24, yerr=op_s, error_kw=eb, color="#BDC3C7", edgecolor="black", label="Open-Loop Policy")
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(lbls, fontsize=8.2)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Unseen Blocked Layout Success (%)")
    axes[0].set_title("A. De-Privileged Unseen Layouts (4 Families)")
    axes[0].legend(fontsize=7.4, loc="upper right")

    sw_m, sw_s = zip(*[_ms(k, "T160_goal_b_switch") for _, k, _ in pres])
    as_m, as_s = zip(*[_ms(k, "dyn_assoc_T1024") for _, k, _ in pres])
    axes[1].bar(xp - 0.18, sw_m, 0.35, yerr=sw_s, error_kw=eb, color=cols, edgecolor="black", label="T=160 (>2.5x W=64) Embodied Switch (%)")
    axes[1].bar(xp + 0.18, as_m, 0.35, yerr=as_s, error_kw=eb, color=cols, alpha=0.50, hatch="//", edgecolor="black", label="Dynamic-Visual Recall T=1024 (%)")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(lbls, fontsize=8.2)
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("Success / Accuracy (%)")
    axes[1].set_title("B. Beyond-Window (T=160) Embodied Switch")
    axes[1].legend(fontsize=7.3, loc="upper right")

    in_m, in_s = zip(*[_ms(k, "latch_mse_inside_W64", 1.0) for _, k, _ in pres])
    by_m, by_s = zip(*[_ms(k, "latch_mse_beyond_W64", 1.0) for _, k, _ in pres])
    axes[2].bar(xp - 0.18, in_m, 0.35, yerr=in_s, error_kw=eb, color="#2E86C1", edgecolor="black", label="Inside Window (t=24..64)")
    axes[2].bar(xp + 0.18, by_m, 0.35, yerr=by_s, error_kw=eb, color="#CB4335", edgecolor="black", label="Beyond Window (t=92..160)")
    axes[2].set_xticks(xp)
    axes[2].set_xticklabels(lbls, fontsize=8.2)
    axes[2].set_ylabel("Decoded Goal-B Latch MSE (lower is better)")
    axes[2].set_title("C. Working-Memory Latch Error Inside vs Beyond W=64")
    axes[2].legend(fontsize=7.6, loc="upper left")

    fig.tight_layout()
    f9 = fig_dir / "x1b_wave7_spatial_mpc_and_t160.png"
    fig.savefig(f9, bbox_inches="tight")
    plt.close(fig)
    saved.append(f9)

    # Figure 10: Zero-Shot Unseen OOD Camera Corruptions & Grounded Spoken Physical xArm6 Vault
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))
    au_m, au_s = zip(*[_ms(k, "unseen_corruption_auroc") for _, k, _ in pres])
    axes[0].bar(xp, au_m, 0.48, yerr=au_s, error_kw=eb, color=cols, edgecolor="black")
    axes[0].axhline(50.0, color="#7F8C8D", linestyle="--", linewidth=1.1, label="Chance (50%)")
    axes[0].set_xticks(xp)
    axes[0].set_xticklabels(lbls, fontsize=8.4)
    axes[0].set_ylim(40, 104)
    axes[0].set_ylabel("Zero-Shot Detection AUROC (%)")
    axes[0].set_title("A. Unseen OOD Camera Corruptions (4 Families)")
    axes[0].legend(fontsize=8.0, loc="lower right")

    dd_m, dd_s = zip(*[_ms(k, "dev_spoken_disc") for _, k, _ in pres])
    vd_m, vd_s = zip(*[_ms(k, "vault_spoken_disc") for _, k, _ in pres])
    axes[1].bar(xp - 0.18, dd_m, 0.35, yerr=dd_s, error_kw=eb, color="#2980B9", edgecolor="black", label="Dev Split (Speaker-Disjoint)")
    axes[1].bar(xp + 0.18, vd_m, 0.35, yerr=vd_s, error_kw=eb, color="#117A65", edgecolor="black", label="Untouched Vault (00147, 00419)")
    axes[1].axhline(50.0, color="#7F8C8D", linestyle="--", linewidth=1.1, label="Chance (50%)")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(lbls, fontsize=8.4)
    axes[1].set_ylim(35, 102)
    axes[1].set_ylabel("Initial-Scene Spoken+16-Token Grounding (%)")
    axes[1].set_title("B. Grounded Human Speech + 16-Token Physical xArm6")
    axes[1].legend(fontsize=7.6, loc="lower right")

    fig.tight_layout()
    f10 = fig_dir / "x1b_wave7_ood_safety_and_spoken_vault.png"
    fig.savefig(f10, bbox_inches="tight")
    plt.close(fig)
    saved.append(f10)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--exp-dir",
        type=Path,
        default=REPO_ROOT / "experiments/multimodal/x1b/spatiotemporal-jepa-1b-20261111",
    )
    args = ap.parse_args()

    bundles = load_wave7_bundles(args.exp_dir / "runs_wave7")
    cohort_names = [
        "w7_slwm_450m_full",
        "w7_sliding_window_w64_450m",
        "w7_diagonal_ssm_450m",
        "w7_loo_no_key_shield_450m",
    ]
    groups = {c: aggregate_w7_cohort(bundles, c) for c in cohort_names}
    hier = compute_wave7_hierarchical_attributions(bundles)
    figs = render_wave7_figures(groups, args.exp_dir / "figures")

    out = {
        "study_id": "spatiotemporal-jepa-wave7-spatial-grid-clincher",
        "wave7_verified_runs": len(bundles),
        "groups": groups,
        "hierarchical_seed_attributions": hier,
        "figures": [str(f.relative_to(REPO_ROOT)) for f in figs],
    }
    out_p = args.exp_dir / "wave7_summary_metrics.json"
    out_p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_p} ({len(bundles)} Wave 7 runs) and {len(figs)} figures.", flush=True)


if __name__ == "__main__":
    main()
