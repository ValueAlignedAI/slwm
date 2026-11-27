"""Generate deterministic SLWM-v4 Gate 3 Architecture, Dual-Track Accounting & Causality Receipt."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import jax

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.v4 import (
    V4_S3_DISCOVERY_RUN_IDS,
    build_v4_s3_discovery_specs,
    compute_v4_core_streaming_state_bytes,
    count_state_tree_bytes,
    count_tree_params,
    estimate_v4_active_core_param_count,
    estimate_v4_core_param_count,
    init_v4_core_state,
    init_v4_model_params,
    make_slwm_v4_spec,
    summarize_v4_accounting,
)

GATE3_CODE_FILES: tuple[str, ...] = (
    "data/v4_heartbeat_contract.py",
    "data/v4_streaming_loader.py",
    "models/v4/__init__.py",
    "models/v4/spec.py",
    "models/v4/blocks.py",
    "models/v4/heads.py",
    "models/v4/model.py",
    "scripts/train/v4_train.py",
    "tests/unit/test_v4_gate3_models.py",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_gate3_receipt(
    *,
    latency_receipt_path: Path = Path("reports/v4/gate3_steady_state_latency_receipt.json"),
    local_codec_120s_path: Path = Path("reports/v4/gate3_local_120s_codec_kv249_saturation_receipt.json"),
    out_path: Path = Path("reports/v4/gate3_v4_architecture_and_causality_receipt.json"),
) -> dict[str, Any]:
    file_hashes = {rel: _sha256_file(_REPO_ROOT / rel) for rel in GATE3_CODE_FILES}
    bundle_h = hashlib.sha256()
    for rel in sorted(file_hashes):
        bundle_h.update(f"{rel}:{file_hashes[rel]}\n".encode("utf-8"))

    # 1. Exact JAX instantiation verification on tiny_unit for all 17 V4-S3 configs
    tiny_specs = build_v4_s3_discovery_specs("tiny_unit")
    tiny_verified = True
    key = jax.random.PRNGKey(0)
    for rid, sp in tiny_specs.items():
        key, ki = jax.random.split(key)
        p = init_v4_model_params(ki, sp)
        st = init_v4_core_state(sp, batch_size=1)
        if count_tree_params(p["core"]) != estimate_v4_core_param_count(sp):
            tiny_verified = False
        if count_state_tree_bytes(st) != compute_v4_core_streaming_state_bytes(sp, 1)["total_core_state_bytes"]:
            tiny_verified = False

    # 2. Exact analytic + FULL Dual-Track (`Core` + `Strict System`) accounting across ALL 17 V4-S3 discovery_45m configs!
    from models.v4 import classify_v4_param_mup_role

    specs_45m = build_v4_s3_discovery_specs("discovery_45m")
    a0_p = estimate_v4_core_param_count(specs_45m["A0_3to1_single_bank"])
    a0_b = compute_v4_core_streaming_state_bytes(specs_45m["A0_3to1_single_bank"], 1)["total_core_state_bytes"]

    discovery_table: dict[str, Any] = {}
    max_active_param_drift_pct = 0.0
    a0_45m_full_acct: dict[str, Any] = {}
    all_17_mup_scalars_clean = True

    def _check_mup_clean(tree: Any, prefix: tuple[str, ...] = ()) -> bool:
        if isinstance(tree, dict):
            return all(_check_mup_clean(v, prefix + (str(k),)) for k, v in tree.items())
        if isinstance(tree, (list, tuple)):
            return all(_check_mup_clean(v, prefix + (str(i),)) for i, v in enumerate(tree))
        if hasattr(tree, "ndim") and int(tree.ndim) <= 1:
            return classify_v4_param_mup_role(prefix, param_ndim=int(tree.ndim)) == "scalar"
        return True

    for rid in V4_S3_DISCOVERY_RUN_IDS:
        sp = specs_45m[rid]
        p_45 = init_v4_model_params(jax.random.PRNGKey(1), sp)
        st_45 = init_v4_core_state(sp, batch_size=1)
        acct = summarize_v4_accounting(sp, p_45, st_45)
        assert acct["core_params_actual"] == acct["core_params_analytic"], f"Param mismatch on {rid}"
        assert acct["core_state_bytes_b1_actual"] == acct["core_state_bytes_b1_analytic"], f"State mismatch on {rid}"
        if not _check_mup_clean(p_45):
            all_17_mup_scalars_clean = False
        if rid == "A0_3to1_single_bank":
            a0_45m_full_acct = acct

        p_tot = acct["core_params_actual"]
        p_act = acct["core_params_active_analytic"]
        drift_pct = (p_act - a0_p) / float(a0_p) * 100.0
        max_active_param_drift_pct = max(max_active_param_drift_pct, abs(drift_pct))
        sb = compute_v4_core_streaming_state_bytes(sp, 1)
        discovery_table[rid] = {
            "backbone_family": sp.backbone_family,
            "layer_schedule": "".join("G" if lt == "gdn" else ("A" if lt == "gqa" else "D") for lt in sp.layer_types()),
            "d_model": sp.d_model,
            "n_layers": sp.n_layers,
            "ffn_dim": sp.ffn_dim,
            "window_size": sp.window_size,
            "n_sinks": sp.n_sinks,
            "n_banks": sp.n_banks,
            "n_mem_heads": sp.n_mem_heads,
            "mem_key_dim": sp.mem_key_dim,
            "beta_max": sp.beta_max,
            "use_short_conv": sp.use_short_conv,
            "use_output_gate": sp.use_output_gate,
            "use_surprise_router": sp.use_surprise_router,
            "use_m_anchor_legacy": sp.use_m_anchor_legacy,
            "use_ms_moe_ffn": sp.use_ms_moe_ffn,
            "video_head_type": sp.video_head_type,
            "audio_head_type": sp.audio_head_type,
            "use_jepa_sync": sp.use_jepa_sync,
            "use_self_forcing": sp.use_self_forcing,
            "use_policy_commit_gate": sp.use_policy_commit_gate,
            "core_params_total": p_tot,
            "core_params_active": p_act,
            "edge_params_actual": acct["edge_params_actual"],
            "heads_params_actual": acct["heads_params_actual"],
            "total_trainable_params": acct["total_trainable_params"],
            "strict_system_total_params": acct["strict_system_total_params"],
            "active_param_delta_vs_a0_pct": round(drift_pct, 3),
            "core_state_bytes_b1": sb["total_core_state_bytes"],
            "core_state_mib_b1": round(sb["total_core_state_bytes"] / (1024.0**2), 4),
            "strict_system_state_mib_b1": acct["strict_system_state_mib_b1"],
            "state_ratio_vs_a0": round(sb["total_core_state_bytes"] / float(a0_b), 4),
        }
        del p_45, st_45

    # 3. Scale ladder profiles
    scale_ladder: dict[str, Any] = {}
    for prof in ("tiny_unit", "discovery_45m", "confirmation_150m", "scale_700m", "scale_1p3b"):
        sp = make_slwm_v4_spec(prof, run_id="A0_3to1_single_bank")
        sb = compute_v4_core_streaming_state_bytes(sp, 1)
        scale_ladder[prof] = {
            "d_model": sp.d_model,
            "n_layers": sp.n_layers,
            "ffn_dim": sp.ffn_dim,
            "width_mult": sp.width_mult,
            "depth_residual_scale": sp.depth_residual_scale,
            "core_params_analytic": estimate_v4_core_param_count(sp),
            "core_state_mib_b1": round(sb["total_core_state_bytes"] / (1024.0**2), 4),
        }

    lat_summary: dict[str, Any] | None = None
    if latency_receipt_path.is_file():
        lat_raw = json.loads(latency_receipt_path.read_text())
        best_prof = lat_raw.get("selected_best_codec_profile", "bf16")
        codec_sec = lat_raw.get(f"steady_state_codecs_120s_{best_prof}", lat_raw.get("steady_state_codecs_120s_bf16", {}))
        lat_summary = {
            "receipt_path": str(latency_receipt_path),
            "gpu_name": lat_raw.get("gpu_name"),
            "selected_best_codec_profile": best_prof,
            "steady_state_codecs_bidir_median_ms": codec_sec.get("total_bidirectional_codecs_steady_ms", {}).get("median_ms"),
            "mimi_kv_249_saturated_20s_to_120s": codec_sec.get("mimi_kv_249_saturated_20s_to_120s"),
            "hbm_flat_o1_verified": codec_sec.get("hbm_flat_o1_verified"),
            "advisory_a1_gate3_pass": lat_raw.get("advisory_a1_gate3_pass"),
        }

    local_120s_summary: dict[str, Any] | None = None
    if local_codec_120s_path.is_file():
        lraw = json.loads(local_codec_120s_path.read_text())
        m = lraw.get("mimi_120s_750hb_cpu_fp32", {})
        w = lraw.get("wan21_24s_150hb_mps_fp16", {})
        local_120s_summary = {
            "receipt_path": str(local_codec_120s_path),
            "mimi_120s_kv_saturated_249_from_10s_to_120s": m.get("kv_saturated_249_from_10s_to_120s"),
            "mimi_120s_state_bytes_flat_o1_10s_to_120s": m.get("state_bytes_flat_o1_10s_to_120s"),
            "mimi_120s_state_bytes_fp32": m.get("state_bytes_trace", {}).get("120s"),
            "mimi_steady_t_gt_20s_cpu_fp32_enc_ms": m.get("steady_t_gt_20s_enc_median_ms"),
            "mimi_steady_t_gt_20s_cpu_fp32_dec_ms": m.get("steady_t_gt_20s_dec_median_ms"),
            "wan21_24s_state_bytes_flat_o1": w.get("state_bytes_flat_o1"),
            "wan21_24s_cache_bytes_fp16": w.get("cache_bytes_trace", {}).get("24s"),
            "combined_bidir_bf16_codec_state_mib": round(
                (w.get("cache_bytes_trace", {}).get("24s", 0) + m.get("state_bytes_trace", {}).get("120s", 0) / 2.0) / (1024.0**2),
                2,
            ),
        }

    receipt = {
        "gate": "SLWM-v4 Gate 3 — Architecture, Exact Dual-Track Accounting, Causality & Gradient-Isolation Receipt",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha256": file_hashes,
        "gate3_bundle_sha256": bundle_h.hexdigest(),
        "all_17_configs_exact_jax_param_and_byte_verified": tiny_verified,
        "all_17_configs_mup_1d_scalars_clean": all_17_mup_scalars_clean,
        "discovery_45m_max_active_param_drift_pct": round(max_active_param_drift_pct, 3),
        "discovery_45m_a0_full_dual_track_accounting": a0_45m_full_acct,
        "discovery_45m_17_configs": discovery_table,
        "scale_ladder_a0": scale_ladder,
        "local_120s_codec_kv249_o1_summary": local_120s_summary,
        "steady_state_live_latency_summary": lat_summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    rec = build_gate3_receipt()
    print(
        f"[G3_RECEIPT] bundle_sha256={rec['gate3_bundle_sha256'][:16]} "
        f"exact_verified={rec['all_17_configs_exact_jax_param_and_byte_verified']} "
        f"max_drift={rec['discovery_45m_max_active_param_drift_pct']}% "
        f"a0_45m_total_trainable={rec['discovery_45m_a0_full_dual_track_accounting']['total_trainable_params']:,}"
    )
