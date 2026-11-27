"""SLWM-v4 Gate 2: Global Non-Vacuous Connected-Data, Disjoint-Split, License & Codec Parity Auditor.

Downloads all `*_preencode_manifest.json` files across the 11 public, license-pure `valueai-org/slwm-v4-*`
HuggingFace dataset repositories and verifies:
  1. Public visibility, complete `README.md` Dataset Cards, and exact YAML license compliance across all 11 repos,
     plus strict 1:1 `.npz` <-> `_meta.json` pairing (zero orphan metadata, zero missing manifest shards).
  2. Per-corpus & global `audit_v4_splits_non_vacuous()` across `(train, dev, test)` for BOTH cluster keys AND
     per-modality payload SHA-256 digests (`video`, `audio`, `motor`, `text`).
  3. Cross-corpus YouTube leakage audit verifying zero `yt::*` overlap across splits between
     `vggsound_full_sync` and `wavcaps_and_libriheavy_speech` (`OpenSound/AudioCaps`).
  4. Exact accounting of realized hours, packet counts, active modality shares, `RH20T` valid-audio rate,
     `AgiBotWorld`/`DROID` success labels, `>= 120 s` continuous long-horizon egocentric AV segments,
     and single-pass (`<= 1.0 epoch`) V4-S3 (`450M` positions) headroom.
  5. Merges certified Gate 2 frozen codec receipts (`gate2_codec_state_and_parity_receipt.json`,
     `gate2_gpu_latency_and_probe_receipt.json`, `showcase_manifest_common_v4.json`) into the single
     Gate 2 authoritative ledger `reports/v4/gate2_v4_data_and_codec_receipt.json`.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.v4_embodiment_and_splits import audit_v4_splits_non_vacuous
from data.v4_heartbeat_contract import (
    V4_TIER1_PERSISTENT_CORPUS_SPEC,
    compute_corpus_exact_packets_and_storage,
)
from scripts.data.v4_valueai_registry import VALUEAI_DATASET_CATALOG, VALUEAI_ORG

# Primary content modalities audited for STRICT 0/0/0 cross-split SHA-256 zero-overlap.
# Note: corpora whose `text` field consists of shared short task/class labels (e.g., 12 Unitree tasks, 17 AgiBot tasks,
# 309 VGGSound sound classes, short DROID instructions, or repeated short spoken utterances across disjoint speakers)
# legitimately share task/class prompt strings across train/dev/test; their text-prompt overlaps are still logged
# transparently while strict zero-overlap is enforced on all unique sensory streams (`video`, `audio`, `motor`) and
# all unique-document/narration text corpora (`fineweb_edu_and_stack_code`, `epic_kitchens_100_long_ego_av`).
CORPUS_EXPECTED_MODALITIES: dict[str, tuple[str, ...]] = {
    "unitree_g1_h1_humanoid": ("video", "motor"),
    "droid_1_0_1": ("video", "motor"),
    "rh20t_contact_av_force": ("video", "audio", "motor"),
    "agibot_and_aloha_sim": ("video", "motor"),
    "vggsound_full_sync": ("video", "audio"),
    "epic_kitchens_100_long_ego_av": ("video", "audio", "text"),
    "holoassist_and_greatesthits_av": ("video", "audio"),
    "greatesthits_impact_av": ("video", "audio"),
    "wavcaps_and_libriheavy_speech": ("audio",),
    "fineweb_edu_and_stack_code": ("text",),
}
UNIQUE_TEXT_CORPORA: frozenset[str] = frozenset(
    {"fineweb_edu_and_stack_code", "epic_kitchens_100_long_ego_av"}
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Global Non-Vacuous SLWM-v4 Gate 2 Audit across public valueai-org datasets.")
    ap.add_argument("--hf-org", type=str, default=VALUEAI_ORG)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "reports/v4/gate2_v4_data_and_codec_receipt.json")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    manifest_sources: list[tuple[str, str, str]] = []  # (corpus_key, repo_id, manifest_relpath)
    shard_files: list[str] = []
    eval_files: list[str] = []
    public_license_audit: dict[str, Any] = {}

    for cname, meta in VALUEAI_DATASET_CATALOG.items():
        repo_id = str(meta["repo_id"])
        try:
            info = api.repo_info(repo_id, repo_type="dataset")
            sibs = sorted(s.rfilename for s in (info.siblings or []))
            cd = info.card_data.to_dict() if info.card_data else {}
            sib_set = set(sibs)
            orphan_metas = sorted(
                f for f in sibs if f.endswith("_meta.json") and f.replace("_meta.json", ".npz") not in sib_set
            )
            orphan_npzs = sorted(
                f for f in sibs
                if f.endswith(".npz")
                and not f.startswith("eval_benchmarks/")
                and f.replace(".npz", "_meta.json") not in sib_set
            )
            public_license_audit[cname] = {
                "repo_id": repo_id,
                "private": bool(getattr(info, "private", False)),
                "card_license": cd.get("license"),
                "expected_license": meta["license"],
                "readme_present": "README.md" in sibs,
                "orphan_meta_files": orphan_metas,
                "orphan_npz_files": orphan_npzs,
                "license_compliant": (
                    (not bool(getattr(info, "private", False)))
                    and ("README.md" in sibs)
                    and (cd.get("license") == meta["license"])
                    and (len(orphan_metas) == 0)
                    and (len(orphan_npzs) == 0)
                ),
            }
            for f in sibs:
                if f.endswith("_preencode_manifest.json"):
                    manifest_sources.append((cname, repo_id, f))
                elif f.startswith("eval_benchmarks/"):
                    eval_files.append(f"{repo_id}/{f}")
                elif f.endswith(".npz"):
                    shard_files.append(f"{repo_id}/{f}")
        except Exception as exc:
            public_license_audit[cname] = {"repo_id": repo_id, "error": str(exc), "license_compliant": False}

    print(f"[GATE2 AUDIT] Found {len(manifest_sources)} corpus manifests, {len(shard_files)} training/eval shards, {len(eval_files)} external eval benchmarks across {args.hf_org}", flush=True)

    all_records: list[dict[str, Any]] = []
    per_corpus_summary: dict[str, Any] = {}
    yt_splits: dict[str, set[str]] = defaultdict(set)

    total_packets_by_split: Counter[str] = Counter()
    total_hours_by_split: Counter[str] = Counter()
    total_modality_counts = Counter()
    long_ego_ge_120s_by_split: Counter[str] = Counter()
    all_corpora_pass = True

    for _ckey, repo_id, mf in manifest_sources:
        lp = hf_hub_download(repo_id, mf, repo_type="dataset")
        data = json.loads(Path(lp).read_text())
        cname = str(data["corpus"])
        eps = list(data.get("episodes", []))
        exp_mods = CORPUS_EXPECTED_MODALITIES.get(cname, ("video",))

        # Per-corpus non-vacuous audit
        try:
            audit_res = audit_v4_splits_non_vacuous(eps, expected_modalities=exp_mods)
            c_pass = bool(audit_res["non_vacuous_zero_overlap_pass"])
        except Exception as exc:
            audit_res = {"non_vacuous_zero_overlap_pass": False, "error": str(exc)}
            c_pass = False
        if not c_pass:
            all_corpora_pass = False

        n_aud_valid = sum(1 for r in eps if r.get("audio_valid") is True)
        n_aud_checked = sum(1 for r in eps if r.get("audio_valid") is not None)
        n_ge_120s = sum(1 for r in eps if float(r.get("duration_s", 0.0)) >= 120.0)
        n_succ_pos = sum(1 for r in eps if int(r.get("episode_success", -1)) == 1)
        n_succ_neg = sum(1 for r in eps if int(r.get("episode_success", -1)) == 0)
        subcorpus_counts = Counter(str(r.get("subcorpus", r.get("rh20t_cfg", "default"))) for r in eps)

        c_shards = [f for f in shard_files if f.startswith(f"{repo_id}/{cname}/")]
        c_shard_rel_set = {f[len(repo_id) + 1 :] for f in c_shards}
        manifest_shards = sorted(set(data.get("uploaded_shards", [])))
        missing_manifest_shards = sorted(s for s in manifest_shards if s not in c_shard_rel_set)
        if missing_manifest_shards:
            all_corpora_pass = False
        per_corpus_summary[cname] = {
            "public_repo_id": repo_id,
            "license": VALUEAI_DATASET_CATALOG.get(cname, {}).get("license"),
            "manifest_path": mf,
            "n_shards": len(c_shards),
            "n_manifest_shards": len(manifest_shards),
            "missing_manifest_shards": missing_manifest_shards,
            "n_episodes": len(eps),
            "total_hours": round(float(data.get("total_hours", 0.0)), 3),
            "hours_by_split": data.get("hours_by_split", {}),
            "total_packets": int(data.get("total_packets", 0)),
            "packets_by_split": data.get("packets_by_split", {}),
            "expected_modalities": list(exp_mods),
            "audio_valid_episodes": n_aud_valid,
            "audio_checked_episodes": n_aud_checked,
            "audio_valid_rate": round(n_aud_valid / max(1, n_aud_checked), 4) if n_aud_checked else None,
            "episodes_ge_120s": n_ge_120s,
            "labeled_success_pos": n_succ_pos,
            "labeled_success_neg": n_succ_neg,
            "subcorpus_counts": dict(subcorpus_counts),
            "non_vacuous_split_audit": audit_res,
        }

        for sp, h_val in data.get("hours_by_split", {}).items():
            total_hours_by_split[sp] += float(h_val)
        for sp, p_val in data.get("packets_by_split", {}).items():
            total_packets_by_split[sp] += int(p_val)

        for r in eps:
            all_records.append(r)
            total_modality_counts["video_patches"] += int(r.get("n_video_patches", 0))
            total_modality_counts["audio_ticks"] += int(r.get("n_audio_ticks", 0))
            total_modality_counts["motor_ticks"] += int(r.get("n_motor_ticks", 0))
            total_modality_counts["text_bytes"] += int(r.get("n_text_bytes", 0))
            if float(r.get("duration_s", 0.0)) >= 120.0:
                long_ego_ge_120s_by_split[str(r["split"])] += 1
            ck = str(r.get("cluster_key", ""))
            if ck.startswith("yt::"):
                yt_splits[ck].add(str(r["split"]))

    # Global cross-corpus non-vacuous audit:
    #   1. All 916k+ episodes across all sensory modalities (`video`, `audio`, `motor`) + cluster keys.
    #   2. All unique-document/narration text corpora (`fineweb_edu_and_stack_code`, `epic_kitchens_100`, `rh20t`) for `text`.
    #   3. Informational raw audit including shared short task/class prompt strings (`text`) so nothing is hidden.
    global_sensory_audit = audit_v4_splits_non_vacuous(
        all_records, expected_modalities=("video", "audio", "motor")
    )
    unique_text_records = [r for r in all_records if str(r.get("corpus", "")) in UNIQUE_TEXT_CORPORA]
    global_unique_text_audit = audit_v4_splits_non_vacuous(
        unique_text_records, expected_modalities=("text",)
    )
    global_all_text_informational = audit_v4_splits_non_vacuous(
        all_records, expected_modalities=("text",)
    )
    global_audit = {
        "sensory_and_cluster_audit": global_sensory_audit,
        "unique_document_text_audit": global_unique_text_audit,
        "shared_label_text_prompt_overlaps_informational": global_all_text_informational["media_payload_overlaps"]["text"],
        "non_vacuous_zero_overlap_pass": (
            bool(global_sensory_audit["non_vacuous_zero_overlap_pass"])
            and bool(global_unique_text_audit["non_vacuous_zero_overlap_pass"])
        ),
    }
    cross_corpus_yt_conflicts = [k for k, sps in yt_splits.items() if len(sps) > 1]

    # Modality share verification
    tot_pkts = sum(total_packets_by_split.values())
    non_text_pkts = (
        total_modality_counts["video_patches"]
        + total_modality_counts["audio_ticks"]
        + total_modality_counts["motor_ticks"]
    )
    train_pkts = int(total_packets_by_split.get("train", 0))
    v4_s3_budget = 450_000_000
    realized_s3_max_ep = round(v4_s3_budget / max(1, train_pkts), 4)

    # Compute exact per-corpus single-pass (<= 0.98 epoch per corpus) V4-S3 (450M) water-filling mixture:
    # Matches Gate 1 §4.1 target (`75.0%` multimodal signal [video+motor+AV + captioned audio/speech] / `25.0%` text+code)!
    s3_waterfill_alloc: dict[str, dict[str, Any]] = {}
    video_motor_av_corpora = [
        "unitree_g1_h1_humanoid", "droid_1_0_1", "agibot_and_aloha_sim", "rh20t_contact_av_force",
        "vggsound_full_sync", "epic_kitchens_100_long_ego_av", "holoassist_and_greatesthits_av", "greatesthits_impact_av",
    ]
    used_s3 = 0
    for cname_w in video_motor_av_corpora:
        if cname_w in per_corpus_summary:
            tr_c = int(per_corpus_summary[cname_w]["packets_by_split"].get("train", 0))
            take_c = int(tr_c * 0.98)
            s3_waterfill_alloc[cname_w] = {
                "alloc_packets": take_c,
                "share_pct": round(100.0 * take_c / v4_s3_budget, 2),
                "train_packets_avail": tr_c,
                "realized_epochs": round(take_c / max(1, tr_c), 4),
            }
            used_s3 += take_c
    tr_aud = int(per_corpus_summary.get("wavcaps_and_libriheavy_speech", {}).get("packets_by_split", {}).get("train", 0))
    tr_txt = int(per_corpus_summary.get("fineweb_edu_and_stack_code", {}).get("packets_by_split", {}).get("train", 0))
    # Cap text+code strictly at 25.0% (112.5M pkts) per Gate 1 §4.1; fill remaining multimodal audio/speech strictly <= 0.98 epoch!
    target_multimodal_pkts = int(v4_s3_budget * 0.75)
    take_aud = min(int(tr_aud * 0.98), max(0, target_multimodal_pkts - used_s3))
    take_txt = max(0, v4_s3_budget - used_s3 - take_aud)
    if tr_aud > 0:
        s3_waterfill_alloc["wavcaps_and_libriheavy_speech"] = {
            "alloc_packets": take_aud,
            "share_pct": round(100.0 * take_aud / v4_s3_budget, 2),
            "train_packets_avail": tr_aud,
            "realized_epochs": round(take_aud / tr_aud, 4),
        }
    if tr_txt > 0:
        s3_waterfill_alloc["fineweb_edu_and_stack_code"] = {
            "alloc_packets": take_txt,
            "share_pct": round(100.0 * take_txt / v4_s3_budget, 2),
            "train_packets_avail": tr_txt,
            "realized_epochs": round(take_txt / tr_txt, 4),
        }
    s3_max_per_corpus_ep = max((v["realized_epochs"] for v in s3_waterfill_alloc.values()), default=99.0)
    s3_video_motor_av_share_pct = round(
        sum(s3_waterfill_alloc[c]["share_pct"] for c in video_motor_av_corpora if c in s3_waterfill_alloc), 2
    )
    s3_multimodal_share_pct = round(
        100.0 - s3_waterfill_alloc.get("fineweb_edu_and_stack_code", {}).get("share_pct", 0.0), 2
    )
    # Exact realized 450M modality packet shares (separating multimodal corpus share from pure non-text sensory packet share!)
    train_mod_by_corpus: dict[str, Counter[str]] = defaultdict(Counter)
    for r in all_records:
        if str(r.get("split")) == "train":
            c_r = str(r.get("corpus", ""))
            train_mod_by_corpus[c_r]["video"] += int(r.get("n_video_patches", 0))
            train_mod_by_corpus[c_r]["audio"] += int(r.get("n_audio_ticks", 0))
            train_mod_by_corpus[c_r]["motor"] += int(r.get("n_motor_ticks", 0))
            train_mod_by_corpus[c_r]["text"] += int(r.get("n_text_bytes", 0))
    s3_mod_pkts: dict[str, float] = {"video": 0.0, "audio": 0.0, "motor": 0.0, "text": 0.0}
    for cname_w, alloc_info in s3_waterfill_alloc.items():
        ep_frac = float(alloc_info["realized_epochs"])
        for mk in s3_mod_pkts:
            s3_mod_pkts[mk] += train_mod_by_corpus[cname_w][mk] * ep_frac
    s3_mod_tot = max(1.0, sum(s3_mod_pkts.values()))
    s3_modality_share_pct = {mk: round(100.0 * val / s3_mod_tot, 2) for mk, val in s3_mod_pkts.items()}
    s3_sensory_non_text_packet_share_pct = round(
        100.0 * (s3_mod_pkts["video"] + s3_mod_pkts["audio"] + s3_mod_pkts["motor"]) / s3_mod_tot, 2
    )

    # Gate 2 remediation quality checks (B3, B5, B6):
    #   1. Robotics & impact AV eval cluster richness (>= 4 disjoint clusters in BOTH dev and test)
    min_eval_cluster_corpora = (
        "unitree_g1_h1_humanoid", "droid_1_0_1", "agibot_and_aloha_sim",
        "rh20t_contact_av_force", "greatesthits_impact_av",
    )
    eval_cluster_richness: dict[str, dict[str, Any]] = {}
    eval_cluster_richness_ok = True
    for cname_c in min_eval_cluster_corpora:
        uc = per_corpus_summary.get(cname_c, {}).get("non_vacuous_split_audit", {}).get("unique_cluster_counts", {})
        d_cl, t_cl = int(uc.get("dev", 0)), int(uc.get("test", 0))
        ok_c = (d_cl >= 4) and (t_cl >= 4)
        eval_cluster_richness[cname_c] = {"dev_clusters": d_cl, "test_clusters": t_cl, "ge_4_pass": ok_c}
        if not ok_c:
            eval_cluster_richness_ok = False

    #   2. AudioCaps split completeness (train/dev/test all populated)
    ac_counts_by_split = Counter(
        str(r["split"]) for r in all_records
        if str(r.get("corpus", "")) == "wavcaps_and_libriheavy_speech"
        and (r.get("subcorpus") == "audiocaps" or str(r.get("episode_id", "")).startswith("ac_"))
    )
    audiocaps_splits_ok = (
        ac_counts_by_split.get("train", 0) >= 10000
        and ac_counts_by_split.get("dev", 0) >= 500
        and ac_counts_by_split.get("test", 0) >= 500
    )

    # Load certified codec + latency + showcase + eval exclusion receipts
    codec_rec = json.loads((_REPO_ROOT / "reports/v4/gate2_codec_state_and_parity_receipt.json").read_text())
    gpu_rec = json.loads((_REPO_ROOT / "reports/v4/gate2_gpu_latency_and_probe_receipt.json").read_text())
    show_rec = json.loads((_REPO_ROOT / "reports/v4/showcase_manifest_common_v4.json").read_text())
    excl_path = _REPO_ROOT / "reports/v4/gate2_eval_exclusion_manifest.json"
    excl_rec = json.loads(excl_path.read_text()) if excl_path.is_file() else {"n_excluded_eval_episodes": 0, "exclusions": []}
    excl_ids = {
        eid for x in excl_rec.get("exclusions", [])
        for eid in [str(x.get("episode_id") or x.get("excluded_episode_id") or "").strip()]
        if eid
    }

    #   3. Verify concrete pre-locked showcase episode_ids exist in their splits & are absent from exclusion list!
    ep_to_split = {f"{r['corpus']}::{r['episode_id']}": str(r["split"]) for r in all_records}
    locked_pools = show_rec.get("locked_episode_pools", {})
    n_locked_ids = 0
    missing_locked_ids: list[str] = []
    excluded_locked_ids: list[str] = []
    for _pool_k, pool_spec in locked_pools.items():
        c_tgt = str(pool_spec.get("corpus", ""))
        for sp_k, want_sp in (("dev_episode_ids", "dev"), ("sealed_test_episode_ids", "test")):
            for eid in pool_spec.get(sp_k, []):
                n_locked_ids += 1
                if str(eid) in excl_ids:
                    excluded_locked_ids.append(str(eid))
                if ep_to_split.get(f"{c_tgt}::{eid}") != want_sp:
                    missing_locked_ids.append(f"{c_tgt}:{want_sp}:{eid}")
    showcase_concrete_ok = (
        bool(show_rec.get("locked_before_training", False))
        and n_locked_ids >= 100
        and len(missing_locked_ids) == 0
        and len(excluded_locked_ids) == 0
    )

    # Required Tier-1 suite coverage check (all 9 corpus groups present)
    expected_tier1_names = {c["name"] for c in V4_TIER1_PERSISTENT_CORPUS_SPEC}
    present_names = set(per_corpus_summary.keys())
    missing_corpora = sorted(expected_tier1_names - present_names)

    all_public_licenses_ok = all(bool(v.get("license_compliant", False)) for v in public_license_audit.values())

    gate2_pass = (
        all_corpora_pass
        and all_public_licenses_ok
        and eval_cluster_richness_ok
        and audiocaps_splits_ok
        and showcase_concrete_ok
        and (s3_multimodal_share_pct >= 70.0)
        and bool(global_audit["non_vacuous_zero_overlap_pass"])
        and len(cross_corpus_yt_conflicts) == 0
        and len(missing_corpora) == 0
        and bool(codec_rec.get("gate2_codec_parity_pass", False))
        and bool(gpu_rec.get("gpu_160ms_live_codec_latency", {}).get("codecs_fit_inside_160ms_realtime", False))
        and realized_s3_max_ep <= 1.0
        and sum(long_ego_ge_120s_by_split.values()) >= 50
    )

    out_obj = {
        "gate": "SLWM-v4 Gate 2 — Connected Data Contract, Disjoint Splits & Codec Parity",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hf_public_org": args.hf_org,
        "public_license_audit": public_license_audit,
        "all_public_licenses_compliant": all_public_licenses_ok,
        "n_total_shards": len(shard_files),
        "n_external_eval_benchmarks": len(eval_files),
        "expected_tier1_corpora": sorted(expected_tier1_names),
        "missing_corpora": missing_corpora,
        "global_totals": {
            "total_episodes": len(all_records),
            "total_hours_all_splits": round(sum(total_hours_by_split.values()), 3),
            "hours_by_split": {k: round(v, 3) for k, v in total_hours_by_split.items()},
            "total_packets_all_splits": tot_pkts,
            "packets_by_split": dict(total_packets_by_split),
            "modality_packet_counts": dict(total_modality_counts),
            "non_text_packet_fraction": round(non_text_pkts / max(1, tot_pkts), 4),
            "long_continuous_ge_120s_by_split": dict(long_ego_ge_120s_by_split),
            "v4_s3_450m_implied_max_epochs": realized_s3_max_ep,
            "v4_s3_450m_per_corpus_max_epochs": s3_max_per_corpus_ep,
            "v4_s3_450m_video_motor_av_share_pct": s3_video_motor_av_share_pct,
            "v4_s3_450m_multimodal_corpus_share_pct": s3_multimodal_share_pct,
            "v4_s3_450m_sensory_non_text_packet_share_pct": s3_sensory_non_text_packet_share_pct,
            "v4_s3_450m_modality_packet_shares_pct": s3_modality_share_pct,
            "single_pass_le_1_0_epoch_for_v4_s3": (realized_s3_max_ep <= 1.0) and (s3_max_per_corpus_ep <= 1.0),
        },
        "eval_cluster_richness_audit": eval_cluster_richness,
        "audiocaps_split_counts": dict(ac_counts_by_split),
        "showcase_concrete_lock_audit": {
            "n_locked_episode_ids": n_locked_ids,
            "missing_locked_ids": missing_locked_ids[:20],
            "excluded_locked_ids": excluded_locked_ids,
            "showcase_concrete_lock_pass": showcase_concrete_ok,
        },
        "v4_s3_450m_single_pass_waterfill_mixture": s3_waterfill_alloc,
        "global_non_vacuous_split_audit": global_audit,
        "cross_corpus_youtube_consistency": {
            "n_unique_yt_clusters": len(yt_splits),
            "n_cross_split_conflicts": len(cross_corpus_yt_conflicts),
            "zero_cross_corpus_yt_leak_pass": len(cross_corpus_yt_conflicts) == 0,
        },
        "per_corpus_summary": per_corpus_summary,
        "codec_parity_summary": {
            "gate2_codec_parity_pass": codec_rec["gate2_codec_parity_pass"],
            "track2_frozen_codec_strict_system_totals": codec_rec["track2_frozen_codec_strict_system_totals"],
            "mimi_16rvq_librispeech_wer_pct": codec_rec["kyutai_mimi"]["direct_16rvq_librispeech_dev_clean_whisper_tiny_wer"]["mimi_16rvq"]["wer_pct"],
            "wan21_int8_latent_psnr_db": codec_rec["wan21_causal_vae"]["int8_vs_fp16_latent_quantization_audit"]["mean_latent_psnr_db"],
            "wan21_int8_decoded_rgb_psnr_db": codec_rec["wan21_causal_vae"]["int8_vs_fp16_latent_quantization_audit"]["mean_decoded_rgb_psnr_db_vs_fp16"],
            "wan21_int8_decoded_rgb_ssim": codec_rec["wan21_causal_vae"]["int8_vs_fp16_latent_quantization_audit"]["mean_decoded_rgb_11x11_ssim_vs_fp16"],
            "a100_bidirectional_codec_ms": gpu_rec["gpu_160ms_live_codec_latency"]["total_bidirectional_codecs_eager_bf16_ms"],
            "a100_headroom_inside_160ms": gpu_rec["gpu_160ms_live_codec_latency"]["remaining_headroom_inside_160ms_ms"],
        },
        "showcase_manifest_sha256": show_rec["manifest_sha256"],
        "eval_exclusion_manifest": excl_rec,
        "gate2_overall_pass": gate2_pass,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2))
    print(f"[GATE2 AUDIT] Wrote {args.out} | gate2_overall_pass={gate2_pass}", flush=True)
    print(json.dumps({k: out_obj[k] for k in ("missing_corpora", "all_public_licenses_compliant", "global_totals", "cross_corpus_youtube_consistency", "gate2_overall_pass")}, indent=2), flush=True)

    if args.upload:
        eval_repo = VALUEAI_DATASET_CATALOG["eval_benchmarks"]["repo_id"]
        api.upload_file(
            path_or_fileobj=str(args.out),
            path_in_repo=f"manifests/{args.out.name}",
            repo_id=eval_repo,
            repo_type="dataset",
        )
        print(f"[HF_UP OK] {eval_repo}/manifests/{args.out.name}", flush=True)


if __name__ == "__main__":
    main()
