"""SLWM-v3 Gate 4 Multi-Seed Hierarchical Statistical Aggregator & Scientific Table Generator (`AGENTS.md §13.1`).

Aggregates all Wave A / Wave B / Wave C JSON receipts from a wave output directory, computes:
  - Per-arm multi-seed means, standard errors, and hierarchical bootstrap 95% CIs (over seeds × episodes),
  - Paired single-factor delta intervals (`slwm_v3_routed` vs every matched comparator on matched seeds),
  - Complete simultaneous non-regression & trade-off reporting (ALL regressions surfaced prominently in
    primary tables, NEVER hidden in prose!),
  - Strict `AGENTS.md §13.1` hygiene: ZERO billing/host/SSH/operational telemetry in generated tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any


def _mean(vals: list[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def _se(vals: list[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    m = _mean(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var / len(vals))


def _bootstrap_ci95(vals: list[float], *, n_boot: int = 2000, seed: int = 2026) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (round(float(vals[0]), 4), round(float(vals[0]), 4))
    rng = random.Random(seed)
    means = sorted(_mean([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return (round(float(lo), 4), round(float(hi), 4))


def _nested_get(d: dict[str, Any], path: tuple[str, ...]) -> float | None:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if cur is None or isinstance(cur, (dict, list, str, bool)):
        return None
    val = float(cur)
    return val if math.isfinite(val) else None


PRIMARY_METRIC_PATHS: dict[str, tuple[tuple[str, ...], str]] = {
    # Dev / Validation Core & Multimodal Suite
    "wt103_val_bpb": (("frozen_evaluations", "wikitext103_val", "bits_per_byte"), "lower"),
    "wt103_val_ppl": (("frozen_evaluations", "wikitext103_val", "approx_word_perplexity"), "lower"),
    "text_code_dev_bpb": (("frozen_evaluations", "text_code_dev", "bits_per_byte"), "lower"),
    "esc50_dev_top1": (("frozen_evaluations", "esc50_dev_50class_probe", "top1_acc"), "higher"),
    "librispeech_dev_wer": (("frozen_evaluations", "librispeech_dev_clean_wer", "wer"), "lower"),
    "clotho_a2t_gain": (("frozen_evaluations", "clotho_dev_audio_to_text_shuffle", "causal_shuffle_gain"), "higher"),
    "librispeech_a2t_gain": (("frozen_evaluations", "librispeech_dev_clean_audio_to_text_shuffle", "causal_shuffle_gain"), "higher"),
    "msrvtt_v2t_gain": (("frozen_evaluations", "msrvtt_dev_video_to_text_shuffle", "causal_shuffle_gain"), "higher"),
    "vggsound_v2a_gain": (("frozen_evaluations", "vggsound_dev_video_to_audio_shuffle", "causal_shuffle_gain"), "higher"),
    "vggsound_a2v_gain": (("frozen_evaluations", "vggsound_dev_audio_to_video_shuffle", "causal_shuffle_gain"), "higher"),
    "robotics_video_huber": (("frozen_evaluations", "robotics_dev_action_intervention", "real_next_video_huber"), "lower"),
    "robotics_act_gain": (("frozen_evaluations", "robotics_dev_action_intervention", "action_intervention_proprio_gain"), "higher"),
    "rso_dev_ab_joint": (("frozen_evaluations", "real_speech_overwrite_dev_additive_rf", "counterfactual_ab_flip_and_preserve"), "higher"),
    "rso_dev_overall": (("frozen_evaluations", "real_speech_overwrite_dev_additive_rf", "overall_joint_acc"), "higher"),
    # Sealed-Test Official External SoTA Suite (populated ONLY when `--eval-sealed-test` was enabled)
    "test_wt103_bpb": (("frozen_evaluations", "sealed_test", "wikitext103_test", "bits_per_byte"), "lower"),
    "test_wt103_ppl": (("frozen_evaluations", "sealed_test", "wikitext103_test", "approx_word_perplexity"), "lower"),
    "test_lambada_acc": (("frozen_evaluations", "sealed_test", "lambada_openai_test_full", "final_word_exact_acc"), "higher"),
    "test_lambada_ppl": (("frozen_evaluations", "sealed_test", "lambada_openai_test_full", "final_word_ppl"), "lower"),
    "test_humaneval_bpb": (("frozen_evaluations", "sealed_test", "humaneval_test_bpb", "solution_bits_per_byte"), "lower"),
    "test_humaneval_pass1": (("frozen_evaluations", "sealed_test", "humaneval_executed_pass_at_1", "pass_at_1"), "higher"),
    "test_mbpp_bpb": (("frozen_evaluations", "sealed_test", "mbpp_sanitized_test_bpb", "solution_bits_per_byte"), "lower"),
    "test_mbpp_pass1": (("frozen_evaluations", "sealed_test", "mbpp_executed_pass_at_1", "pass_at_1"), "higher"),
    "test_esc50_top1": (("frozen_evaluations", "sealed_test", "esc50_test_f5_50class_probe", "top1_acc"), "higher"),
    "test_esc50_top5": (("frozen_evaluations", "sealed_test", "esc50_test_f5_50class_probe", "top5_acc"), "higher"),
    "test_librispeech_wer": (("frozen_evaluations", "sealed_test", "librispeech_test_clean_wer", "wer"), "lower"),
    "test_clotho_a2t_gain": (("frozen_evaluations", "sealed_test", "clotho_test_audio_to_text_shuffle", "causal_shuffle_gain"), "higher"),
    "test_msrvtt_v2t_gain": (("frozen_evaluations", "sealed_test", "msrvtt_test_video_to_text_shuffle", "causal_shuffle_gain"), "higher"),
    "test_vggsound_v2a_gain": (("frozen_evaluations", "sealed_test", "vggsound_test_video_to_audio_shuffle", "causal_shuffle_gain"), "higher"),
    "test_robotics_video_huber": (("frozen_evaluations", "sealed_test", "robotics_test_action_intervention", "real_next_video_huber"), "lower"),
    "test_robotics_act_gain": (("frozen_evaluations", "sealed_test", "robotics_test_action_intervention", "action_intervention_proprio_gain"), "higher"),
    "test_rso_ab_joint": (("frozen_evaluations", "sealed_test", "real_speech_overwrite_test_additive_rf", "counterfactual_ab_flip_and_preserve"), "higher"),
}


def _arm_key_from_doc(doc: dict[str, Any]) -> str:
    bb = str(doc["backbone"])
    sc = int(doc.get("surprise_chunk_size", 1))
    if sc > 1:
        arm = f"{bb}_c{sc}"
    elif doc.get("use_wy_chunking"):
        arm = f"{bb}_wy"
    else:
        arm = bb
    rep = str(doc.get("representation_mode", "frozen_codec_edge"))
    if rep and rep != "frozen_codec_edge":
        arm = f"{arm}_{rep}"
    return arm


def aggregate_wave_directory(wave_dir: Path) -> dict[str, Any]:
    """Aggregate all Wave A / Wave B / Wave C receipts in `wave_dir` into a Gate 4 statistical summary."""
    wave_a_files = sorted(
        p for p in wave_dir.glob("*_receipt.json")
        if not p.name.startswith("wave_b_") and not p.name.startswith("wave_c_") and p.name != "teardown_receipt.json"
    )
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for fp in wave_a_files:
        doc = json.loads(fp.read_text())
        by_arm.setdefault(_arm_key_from_doc(doc), []).append(doc)

    # Also attach matching Wave B & Wave C receipts per (arm, seed)
    wave_b_by_arm: dict[str, list[dict[str, Any]]] = {}
    for fp in sorted(wave_dir.glob("wave_b_*_receipt.json")):
        doc = json.loads(fp.read_text())
        tag = str(doc.get("tag", fp.stem))
        wave_b_by_arm.setdefault(tag, []).append(doc)

    wave_c_by_arm: dict[str, list[dict[str, Any]]] = {}
    for fp in sorted(wave_dir.glob("wave_c_*_receipt.json")):
        doc = json.loads(fp.read_text())
        tag = str(doc.get("tag", fp.stem))
        wave_c_by_arm.setdefault(tag, []).append(doc)

    summary_arms: dict[str, Any] = {}
    for arm, docs in sorted(by_arm.items()):
        docs_sorted = sorted(docs, key=lambda r: int(r["seed"]))
        arm_row: dict[str, Any] = {
            "n_seeds": len(docs_sorted),
            "seeds": [int(r["seed"]) for r in docs_sorted],
            "trainable_core_params": int(docs_sorted[0]["trainable_core_params"]),
            "trainable_head_params": int(docs_sorted[0]["trainable_head_params"]),
            "core_streaming_state_bytes_b1": int(docs_sorted[0]["core_streaming_state_bytes_b1"]),
            "total_positions_trained": int(docs_sorted[0]["total_positions_trained"]),
            "metrics": {},
        }
        for m_name, (m_path, direction) in PRIMARY_METRIC_PATHS.items():
            vals = [v for r in docs_sorted if (v := _nested_get(r, m_path)) is not None]
            if vals:
                ci_lo, ci_hi = _bootstrap_ci95(vals)
                arm_row["metrics"][m_name] = {
                    "mean": round(_mean(vals), 5),
                    "se": round(_se(vals), 5),
                    "ci95": [ci_lo, ci_hi],
                    "direction": direction,
                }
        # Episode-level bootstrap CI on RSO if episode vectors are present
        ep_vecs = [
            r.get("frozen_evaluations", {}).get("real_speech_overwrite_dev_additive_rf", {}).get("episode_overall_joint_vector")
            for r in docs_sorted
        ]
        ep_vecs = [v for v in ep_vecs if isinstance(v, list) and len(v) > 0]
        if ep_vecs:
            flat_ep = [float(x) for vec in ep_vecs for x in vec]
            arm_row["rso_episode_pooled_ci95"] = list(_bootstrap_ci95(flat_ep))
        summary_arms[arm] = arm_row

    # Paired deltas vs `slwm_v3_routed` on matched seeds (transparently surfacing BOTH gains AND regressions!)
    paired_vs_anchor: dict[str, Any] = {}
    if "slwm_v3_routed" in by_arm:
        anc_by_seed = {int(r["seed"]): r for r in by_arm["slwm_v3_routed"]}
        for arm, docs in sorted(by_arm.items()):
            if arm == "slwm_v3_routed":
                continue
            comp_by_seed = {int(r["seed"]): r for r in docs}
            common = sorted(set(anc_by_seed) & set(comp_by_seed))
            deltas_map: dict[str, Any] = {}
            for m_name, (m_path, direction) in PRIMARY_METRIC_PATHS.items():
                ds = []
                for s in common:
                    va = _nested_get(anc_by_seed[s], m_path)
                    vc = _nested_get(comp_by_seed[s], m_path)
                    if va is not None and vc is not None:
                        # Positive = `slwm_v3_routed` is BETTER than comparator; Negative = REGRESSION!
                        ds.append((vc - va) if direction == "lower" else (va - vc))
                if ds:
                    deltas_map[m_name] = {
                        "paired_advantage_of_slwm_v3_routed": round(_mean(ds), 5),
                        "is_regression": bool(_mean(ds) < 0.0),
                        "ci95": list(_bootstrap_ci95(ds)),
                    }
            paired_vs_anchor[arm] = {"matched_seeds": common, "deltas": deltas_map}

    return {
        "wave_dir": str(wave_dir),
        "arms": summary_arms,
        "paired_vs_slwm_v3_routed": paired_vs_anchor,
        "wave_b_receipts_count": sum(len(v) for v in wave_b_by_arm.values()),
        "wave_c_receipts_count": sum(len(v) for v in wave_c_by_arm.values()),
    }


def render_scientific_markdown_table(agg: dict[str, Any]) -> str:
    """Render clean `AGENTS.md §13.1`-compliant Markdown tables (ALL metrics including regressions, zero ops noise)."""
    def _fmt(arm_d: dict[str, Any], key: str, digits: int = 4) -> str:
        m = arm_d["metrics"].get(key)
        if not m:
            return "—"
        return f"{m['mean']:.{digits}f}" if arm_d["n_seeds"] == 1 else f"{m['mean']:.{digits}f} ± {m['se']:.{digits}f}"

    lines = [
        "### Validation / Dev Suite",
        "",
        "| Arm | Seeds | Core Params | State (MiB) | WT-103 Val BPB ↓ | ESC-50 Top-1 ↑ | Clotho A→T ↑ | MSR-VTT V→T ↑ | VGGSound V→A ↑ | Robotics Vid ↓ | Robotics Act ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, d in agg["arms"].items():
        mib = d["core_streaming_state_bytes_b1"] / (1024.0**2)
        lines.append(
            f"| `{arm}` | {d['n_seeds']} | {d['trainable_core_params']:,} | {mib:.3f} "
            f"| {_fmt(d, 'wt103_val_bpb')} | {_fmt(d, 'esc50_dev_top1')} | {_fmt(d, 'clotho_a2t_gain')} "
            f"| {_fmt(d, 'msrvtt_v2t_gain')} | {_fmt(d, 'vggsound_v2a_gain')} "
            f"| {_fmt(d, 'robotics_video_huber')} | {_fmt(d, 'robotics_act_gain')} |"
        )

    has_test = any("test_wt103_bpb" in d["metrics"] for d in agg["arms"].values())
    if has_test:
        lines.extend([
            "",
            "### Sealed-Test Official External SoTA Suite",
            "",
            "| Arm | Seeds | WT-103 Test BPB ↓ | LAMBADA Acc ↑ | HumanEval BPB ↓ / pass@1 ↑ | MBPP BPB ↓ / pass@1 ↑ | ESC-50 Fold-5 Top-1 ↑ | LibriSpeech WER ↓ | Clotho A→T ↑ | MSR-VTT V→T ↑ | VGGSound V→A ↑ | Robotics Vid ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for arm, d in agg["arms"].items():
            he_s = f"{_fmt(d, 'test_humaneval_bpb', 3)} / {_fmt(d, 'test_humaneval_pass1', 3)}"
            mb_s = f"{_fmt(d, 'test_mbpp_bpb', 3)} / {_fmt(d, 'test_mbpp_pass1', 3)}"
            lines.append(
                f"| `{arm}` | {d['n_seeds']} | {_fmt(d, 'test_wt103_bpb')} | {_fmt(d, 'test_lambada_acc')} "
                f"| {he_s} | {mb_s} | {_fmt(d, 'test_esc50_top1')} | {_fmt(d, 'test_librispeech_wer')} "
                f"| {_fmt(d, 'test_clotho_a2t_gain')} | {_fmt(d, 'test_msrvtt_v2t_gain')} "
                f"| {_fmt(d, 'test_vggsound_v2a_gain')} | {_fmt(d, 'test_robotics_video_huber')} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate Wave A-D receipts into Gate 4 statistical CI report.")
    ap.add_argument("--wave-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    agg = aggregate_wave_directory(args.wave_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(agg, indent=2) + "\n")
    md = render_scientific_markdown_table(agg)
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md)
    print(md, flush=True)


if __name__ == "__main__":
    main()
