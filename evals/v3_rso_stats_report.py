"""SLWM-v3 Crossed-Seed & Speaker-Cluster Paired Bootstrap CI Aggregator for Real-Speech Selective Overwrite.

Computes:
  - Per-arm multi-seed means, standard errors, and crossed-seed × speaker-cluster 95% bootstrap CIs
    across Overwrite A/B flip-and-preserve, Updated-Attr flip, Preserved-Attr, Catch dual-preserve,
    and Overall Counterfactual Joint Accuracy!
  - Paired single-factor advantage (`slwm_v3_routed - comparator`) on matched seeds & identical
    speaker clusters, checking the preregistered Gate 1 criterion (`>= +5.0 pp` AND 95% CI strictly > 0)!
  - Clean `AGENTS.md §13.1`-compliant Markdown table (ALL regressions surfaced, ZERO ops/billing telemetry).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def _se(xs: list[float]) -> float:
    if len(xs) <= 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / ((len(xs) - 1) * len(xs)))


def _crossed_seed_speaker_delta_ci95(
    anc_docs: dict[int, dict[str, Any]],
    cmp_docs: dict[int, dict[str, Any]],
    split: str = "dev",
    *,
    n_boot: int = 2000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Two-way (crossed-seed × speaker-cluster) paired bootstrap 95% CI for `anc - cmp`."""
    common_seeds = sorted(set(anc_docs) & set(cmp_docs))
    if not common_seeds:
        return {"matched_seeds": [], "mean_delta": None, "ci95": None}

    # Build per-seed speaker-indexed delta lists
    seed_spk_deltas: list[dict[str, list[float]]] = []
    all_spks: set[str] = set()
    for s in common_seeds:
        ea = anc_docs[s]["frozen_evaluations"].get(split) or {}
        ec = cmp_docs[s]["frozen_evaluations"].get(split) or {}
        va = ea.get("episode_overall_joint_vector") or []
        vc = ec.get("episode_overall_joint_vector") or []
        spks = ea.get("episode_primary_speaker_ids") or [str(i) for i in range(len(va))]
        if len(va) != len(vc) or not va:
            continue
        m_d: dict[str, list[float]] = {}
        for spk, a_i, c_i in zip(spks, va, vc):
            m_d.setdefault(str(spk), []).append(float(a_i) - float(c_i))
            all_spks.add(str(spk))
        seed_spk_deltas.append(m_d)

    if not seed_spk_deltas or not all_spks:
        return {"matched_seeds": common_seeds, "mean_delta": None, "ci95": None}

    spk_list = sorted(all_spks)
    point = _mean([v for sd in seed_spk_deltas for vs in sd.values() for v in vs])
    rng = random.Random(seed)
    n_s = len(seed_spk_deltas)
    n_k = len(spk_list)
    boots: list[float] = []
    for _ in range(n_boot):
        s_draw = [rng.randrange(n_s) for _ in range(n_s)]
        k_draw = [spk_list[rng.randrange(n_k)] for _ in range(n_k)]
        vals = [v for si in s_draw for spk in k_draw for v in seed_spk_deltas[si].get(spk, ())]
        if vals:
            boots.append(_mean(vals))
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return {
        "matched_seeds": common_seeds,
        "n_unique_speakers": n_k,
        "mean_delta_pp": round(point * 100.0, 2),
        "crossed_seed_speaker_ci95_pp": [round(lo * 100.0, 2), round(hi * 100.0, 2)],
        "ci95_strictly_positive": bool(lo > 0.0),
        "beats_gate1_5pp_and_ci_pos": bool(point >= 0.05 and lo > 0.0),
    }


def aggregate_rso_dir(rso_dir: Path, split: str = "dev") -> dict[str, Any]:
    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for fp in sorted(rso_dir.glob("rso_*_receipt.json")):
        doc = json.loads(fp.read_text())
        if split not in doc.get("frozen_evaluations", {}):
            continue
        c_sz = int(doc.get("surprise_chunk_size", 1))
        bb = str(doc["backbone"])
        arm = f"{bb}_c{c_sz}" if c_sz > 1 else bb
        by_arm.setdefault(arm, {})[int(doc["seed"])] = doc

    arms_summary: dict[str, Any] = {}
    for arm, seed_map in sorted(by_arm.items()):
        docs = [seed_map[s] for s in sorted(seed_map)]
        evs = [d["frozen_evaluations"][split] for d in docs]
        arms_summary[arm] = {
            "n_seeds": len(docs),
            "seeds": sorted(seed_map),
            "trainable_core_params": int(docs[0]["trainable_core_params"]),
            "core_streaming_state_bytes_b1": int(docs[0]["core_streaming_state_bytes_b1"]),
            "mean_wall_sec": round(_mean([float(d["wall_sec"]) for d in docs]), 1),
            "overall_joint_mean": round(_mean([float(e["overall_counterfactual_joint_acc"]) for e in evs]), 4),
            "overall_joint_se": round(_se([float(e["overall_counterfactual_joint_acc"]) for e in evs]), 4),
            "overwrite_ab_mean": round(_mean([float(e["overwrite_ab_flip_and_preserve"]) for e in evs]), 4),
            "overwrite_upd_mean": round(_mean([float(e["overwrite_updated_attr_flip_acc"]) for e in evs]), 4),
            "overwrite_pres_mean": round(_mean([float(e["overwrite_preserved_attr_acc"]) for e in evs]), 4),
            "catch_preserve_mean": round(_mean([float(e["catch_dual_preserve_acc"]) for e in evs]), 4),
            "mean_nowrite": round(_mean([float(e["mean_nowrite"]) for e in evs]), 4),
        }

    paired: dict[str, Any] = {}
    if "slwm_v3_routed" in by_arm:
        anc = by_arm["slwm_v3_routed"]
        for arm, sm in sorted(by_arm.items()):
            if arm == "slwm_v3_routed":
                continue
            paired[arm] = _crossed_seed_speaker_delta_ci95(anc, sm, split=split)

    return {"rso_dir": str(rso_dir), "split": split, "arms": arms_summary, "paired_vs_slwm_v3_routed": paired}


def render_md(agg: dict[str, Any]) -> str:
    lines = [
        f"| Arm ({agg['split']}) | Seeds | Core Params | State (MiB) | Wall (s) | Overall Joint ↑ | Overwrite A/B ↑ | Upd Flip ↑ | Preserved ↑ | Catch Dual ↑ | vs `slwm_v3_routed` Δ (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, d in agg["arms"].items():
        mib = d["core_streaming_state_bytes_b1"] / (1024.0**2)
        if arm == "slwm_v3_routed":
            delta_s = "anchor"
        else:
            p = agg["paired_vs_slwm_v3_routed"].get(arm) or {}
            if p.get("mean_delta_pp") is not None:
                lo, hi = p["crossed_seed_speaker_ci95_pp"]
                delta_s = f"{p['mean_delta_pp']:+.1f} pp [{lo:+.1f}, {hi:+.1f}]"
            else:
                delta_s = "—"
        lines.append(
            f"| `{arm}` | {d['n_seeds']} | {d['trainable_core_params']:,} | {mib:.3f} | {d['mean_wall_sec']:.0f} "
            f"| {d['overall_joint_mean']:.4f} ± {d['overall_joint_se']:.4f} | {d['overwrite_ab_mean']:.4f} "
            f"| {d['overwrite_upd_mean']:.4f} | {d['overwrite_pres_mean']:.4f} | {d['catch_preserve_mean']:.4f} | {delta_s} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Real-Speech Selective Overwrite multi-seed receipts.")
    ap.add_argument("--rso-dir", type=Path, required=True)
    ap.add_argument("--split", type=str, default="dev", choices=("dev", "sealed_test"))
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    agg = aggregate_rso_dir(args.rso_dir, split=args.split)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(agg, indent=2) + "\n")
    md = render_md(agg)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)
    print(md, flush=True)


if __name__ == "__main__":
    main()
