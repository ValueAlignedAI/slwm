"""SLWM-v4 Gate 2: Pre-Registered Cross-Model Qualitative Showcase Manifest Generator (AGENTS.md §13.2).

Locks concrete common cross-model `dev` and `sealed_test` `episode_id` pools directly from the finalized public
`valueai-org/slwm-v4-*` manifests BEFORE any SLWM-v4 model is trained, alongside deterministic quantile selectors,
trivial copy/retrieval baselines, fixed decoding seeds (`n_samples = 1`, zero best-of-N), epistemic channel labels,
cross-modal causal intervention checks, and strict exclusion of every ID in `gate2_eval_exclusion_manifest.json`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.data.v4_valueai_registry import VALUEAI_DATASET_CATALOG


def _det_rank(seed_tag: str, eid: str) -> str:
    return hashlib.sha256(f"slwm-v4-showcase-lock::{seed_tag}::{eid}".encode("utf-8")).hexdigest()


def _select_stratified_episode_ids(
    records: list[dict[str, Any]],
    *,
    split: str,
    n_target: int,
    seed_tag: str,
    excluded_ids: set[str],
    min_duration_s: float = 0.0,
    require_audio_valid: bool = False,
) -> list[str]:
    """Deterministically select up to `n_target` concrete `episode_id`s stratified round-robin across clusters."""
    by_cluster: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if str(r.get("split")) != split:
            continue
        eid = str(r.get("episode_id", ""))
        if not eid or eid in excluded_ids:
            continue
        if float(r.get("duration_s", 0.0)) < min_duration_s:
            continue
        if require_audio_valid and (r.get("audio_valid") is not True):
            continue
        ck = str(r.get("cluster_key", "default"))
        by_cluster[ck].append(eid)

    for ck in by_cluster:
        by_cluster[ck].sort(key=lambda e: _det_rank(seed_tag, e))
    clusters_sorted = sorted(by_cluster.keys(), key=lambda c: _det_rank(f"{seed_tag}::cl", c))

    picked: list[str] = []
    depth = 0
    while len(picked) < n_target:
        added_any = False
        for ck in clusters_sorted:
            if depth < len(by_cluster[ck]):
                picked.append(by_cluster[ck][depth])
                added_any = True
                if len(picked) >= n_target:
                    break
        if not added_any:
            break
        depth += 1
    return picked


def _load_corpus_manifest_episodes(corpus_key: str) -> list[dict[str, Any]]:
    repo_id = str(VALUEAI_DATASET_CATALOG[corpus_key]["repo_id"])
    mf = f"{corpus_key}/{corpus_key}_preencode_manifest.json"
    lp = hf_hub_download(repo_id, mf, repo_type="dataset")
    data = json.loads(Path(lp).read_text())
    return list(data.get("episodes", []))


def build_showcase_manifest_v4(
    *,
    excl_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Construct immutable pre-registered `showcase_manifest_common_v4.json` with concrete locked `episode_id`s."""
    excl_path = excl_manifest_path or (_REPO_ROOT / "reports/v4/gate2_eval_exclusion_manifest.json")
    excluded_ids: set[str] = set()
    if excl_path.is_file():
        excl_data = json.loads(excl_path.read_text())
        excluded_ids = {
            eid for x in excl_data.get("exclusions", [])
            for eid in [str(x.get("episode_id") or x.get("excluded_episode_id") or "").strip()]
            if eid
        }

    needed_corpora = (
        "greatesthits_impact_av",
        "rh20t_contact_av_force",
        "vggsound_full_sync",
        "epic_kitchens_100_long_ego_av",
        "holoassist_and_greatesthits_av",
        "droid_1_0_1",
        "unitree_g1_h1_humanoid",
        "agibot_and_aloha_sim",
    )
    recs_by_corpus = {c: _load_corpus_manifest_episodes(c) for c in needed_corpora}

    def _lock_pool(
        pool_name: str,
        corpus_key: str,
        n_dev: int,
        n_test: int,
        *,
        min_duration_s: float = 0.0,
        require_audio_valid: bool = False,
    ) -> dict[str, Any]:
        dev_ids = _select_stratified_episode_ids(
            recs_by_corpus[corpus_key],
            split="dev",
            n_target=n_dev,
            seed_tag=f"{pool_name}::dev",
            excluded_ids=excluded_ids,
            min_duration_s=min_duration_s,
            require_audio_valid=require_audio_valid,
        )
        test_ids = _select_stratified_episode_ids(
            recs_by_corpus[corpus_key],
            split="test",
            n_target=n_test,
            seed_tag=f"{pool_name}::test",
            excluded_ids=excluded_ids,
            min_duration_s=min_duration_s,
            require_audio_valid=require_audio_valid,
        )
        assert len(dev_ids) > 0, f"Pool {pool_name} ({corpus_key}) yielded 0 dev episodes!"
        assert len(test_ids) > 0, f"Pool {pool_name} ({corpus_key}) yielded 0 test episodes!"
        assert len(set(dev_ids) & set(test_ids)) == 0, f"Overlap inside showcase pool {pool_name}!"
        return {
            "corpus": corpus_key,
            "public_repo_id": VALUEAI_DATASET_CATALOG[corpus_key]["repo_id"],
            "min_duration_s": min_duration_s,
            "require_audio_valid": require_audio_valid,
            "n_dev_locked": len(dev_ids),
            "n_sealed_test_locked": len(test_ids),
            "dev_episode_ids": dev_ids,
            "sealed_test_episode_ids": test_ids,
        }

    locked_pools: dict[str, dict[str, Any]] = {
        # Track A: Video -> Audio impact & open-domain Foley
        "track_a_greatesthits": _lock_pool("track_a_gh", "greatesthits_impact_av", 64, 64, min_duration_s=4.8, require_audio_valid=True),
        "track_a_rh20t_contact_av": _lock_pool("track_a_rh20t", "rh20t_contact_av_force", 64, 64, min_duration_s=3.2, require_audio_valid=True),
        "track_a_vggsound": _lock_pool("track_a_vgg", "vggsound_full_sync", 64, 64, min_duration_s=4.8, require_audio_valid=True),
        # Track B: Audio/Text -> Video generation
        "track_b_vggsound": _lock_pool("track_b_vgg", "vggsound_full_sync", 64, 64, min_duration_s=4.8, require_audio_valid=True),
        "track_b_epic100": _lock_pool("track_b_epic", "epic_kitchens_100_long_ego_av", 64, 64, min_duration_s=10.0, require_audio_valid=True),
        # Track C: 2+ min continuous live AV streaming (12 s prompt + 120 s rollout = 132.0 s <= 135.84 s max chunk!)
        "track_c_epic100_2min": _lock_pool("track_c_epic2m", "epic_kitchens_100_long_ego_av", 5, 5, min_duration_s=132.0, require_audio_valid=True),
        "track_c_holoassist_2min": _lock_pool("track_c_holo2m", "holoassist_and_greatesthits_av", 5, 5, min_duration_s=132.0, require_audio_valid=True),
        # Track D: Real robot & humanoid held-out building / session rollouts
        "track_d_droid_heldout": _lock_pool("track_d_droid", "droid_1_0_1", 64, 64, min_duration_s=4.8),
        "track_d_unitree_g1_h1_heldout": _lock_pool("track_d_unitree", "unitree_g1_h1_humanoid", 64, 64, min_duration_s=4.8),
        "track_d_agibot_heldout": _lock_pool("track_d_agibot", "agibot_and_aloha_sim", 64, 64, min_duration_s=4.8),
        # Track E: Blind AV desync & PolicyCommitGate calibration
        "track_e_vggsound_desync": _lock_pool("track_e_vgg", "vggsound_full_sync", 100, 100, min_duration_s=6.4, require_audio_valid=True),
        "track_e_greatesthits_desync": _lock_pool("track_e_gh", "greatesthits_impact_av", 64, 64, min_duration_s=4.8, require_audio_valid=True),
    }

    manifest: dict[str, Any] = {
        "schema_version": "slwm_v4_showcase_manifest_1.1",
        "gate": "SLWM-v4 Gate 2 — Pre-Registered Cross-Model Showcase Manifest (AGENTS.md §13.2)",
        "locked_before_training": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "excluded_eval_episode_ids_checked": sorted(excluded_ids),
        "global_protocol": {
            "weights_strictly_frozen": True,
            "online_weight_updates_allowed": False,
            "fixed_decoding_seed": 4201,
            "samples_per_episode": 1,
            "best_of_n_regeneration_forbidden": True,
            "dev_vs_sealed_test_separation": "dev gallery used during V4-S3/V4-S4 iteration; sealed_test gallery generated ONCE after checkpoint and decoding hyperparameters are frozen.",
            "required_side_by_side_columns": [
                "Ground Truth",
                "Trivial Copy/Retrieval Baseline",
                "3:1 Single-Bank Gated DeltaNet (A0)",
                "Equal-Byte SW-GQA (B1)",
                "SLWM-v3 Reference (B0)",
                "SLWM-v4",
            ],
            "required_epistemic_channel_labels": [
                "observed",
                "reconstructed",
                "predicted",
                "imagined",
                "abstained",
            ],
            "separate_forced_diagnostic_vs_policy_routed": True,
            "deterministic_quantile_selector": {
                "quantiles": ["P05_failure", "P25", "P50_median", "P75", "P95_best"],
                "seed_selection_rule": "Rank episodes on the median training seed by each track's pre-registered physical grounding metric; never drop P05 failures.",
            },
            "causal_conditioning_interventions": {
                "av_condition_shuffling": "Pair identical video with symmetric +-0.96 s interior-cropped audio from same clip (zero splice seam, MOD_TEXT stripped) to verify causal AV conditioning vs unconditional prior hallucination.",
                "robotics_action_intervention": "Compare two valid joint-limit-respecting in-distribution motor programs (0.0% joint-limit violation, NEVER OOD negated -a) to verify causal action-to-video controllability.",
                "generative_head_non_inferiority_gate": "Adding Flow-Video + Depth-Audio heads must NOT regress closed-loop ALOHA Stage 2/3, >30s overwrite/re-ID recall, or PolicyCommitGate ECE beyond pre-registered 1.0% / 0.015 bounds.",
            },
        },
        "locked_episode_pools": locked_pools,
        "tracks": {
            "track_a_video_to_audio_impact_and_foley": {
                "description": "Causal Video -> Audio contact & open-domain Foley synthesis (160 ms interleaved, 16-RVQ Mimi).",
                "pool_refs": ["track_a_greatesthits", "track_a_rh20t_contact_av", "track_a_vggsound"],
                "ranking_metric": "onset_f1_20ms_and_desync_sec",
                "trivial_baseline": "last_observed_audio_tick_repeat_and_nearest_neighbor_train_audio_retrieval",
                "prompt_sec": 1.6,
                "continuation_sec": 3.2,
                "render_quantile_count_per_corpus": 5,
            },
            "track_b_audio_and_text_to_video_generation": {
                "description": "Causal Audio -> Video and Text -> Video Flow-Matching generation (192x320 @ 25 fps, 4-8 NFE).",
                "pool_refs": ["track_b_vggsound", "track_b_epic100"],
                "ranking_metric": "fvd_and_lpips_and_dino_sim",
                "trivial_baseline": "static_last_observed_frame_copy_and_huber_mean_regression",
                "prompt_sec": 1.6,
                "continuation_sec": 3.2,
                "render_quantile_count_per_corpus": 5,
            },
            "track_c_2min_plus_live_continuous_av_action_streaming": {
                "description": "Sustainable >= 120.0 s (750 heartbeats) continuous live simultaneous Audio+Video+Action generation in flat O(1) VRAM.",
                "pool_refs": ["track_c_epic100_2min", "track_c_holoassist_2min"],
                "ranking_metric": "bucket_60_120s_dinov2_drift_and_gt30s_object_permanence_reid",
                "trivial_baseline": "static_prompt_loop_and_30s_truncated_window_copy",
                "prompt_sec": 12.0,
                "rollout_sec": 120.0,
                "min_episode_duration_sec": 132.0,
                "rollout_heartbeats": 750,
                "dev_episodes": [
                    {"episode_id": eid, "corpus": "epic_kitchens_100_long_ego_av", "prompt_sec": 12.0, "rollout_sec": 120.0, "heartbeats": 750}
                    for eid in locked_pools["track_c_epic100_2min"]["dev_episode_ids"]
                ] + [
                    {"episode_id": eid, "corpus": "holoassist_and_greatesthits_av", "prompt_sec": 12.0, "rollout_sec": 120.0, "heartbeats": 750}
                    for eid in locked_pools["track_c_holoassist_2min"]["dev_episode_ids"]
                ],
                "sealed_test_episodes": [
                    {"episode_id": eid, "corpus": "epic_kitchens_100_long_ego_av", "prompt_sec": 12.0, "rollout_sec": 120.0, "heartbeats": 750}
                    for eid in locked_pools["track_c_epic100_2min"]["sealed_test_episode_ids"]
                ] + [
                    {"episode_id": eid, "corpus": "holoassist_and_greatesthits_av", "prompt_sec": 12.0, "rollout_sec": 120.0, "heartbeats": 750}
                    for eid in locked_pools["track_c_holoassist_2min"]["sealed_test_episode_ids"]
                ],
                "aloha_2min_continuous_seeds": {"dev": [3500, 3501, 3502], "sealed_test": [4500, 4501, 4502]},
            },
            "track_d_closed_loop_humanoid_and_robotics": {
                "description": "Closed-loop MuJoCo ALOHA manipulation (TransferCube + Insertion) + DROID held-out building + Unitree G1/H1 held-out sessions + AgiBot held-out task/scenes.",
                "pool_refs": ["track_d_droid_heldout", "track_d_unitree_g1_h1_heldout", "track_d_agibot_heldout"],
                "ranking_metric": "anti_knockoff_3stage_physical_score_and_kendall_tau",
                "trivial_baseline": "hold_pose_null_and_open_loop_nominal_replay",
                "aloha_transfer_cube_seeds": {
                    "dev": list(range(3000, 3032)),
                    "sealed_test": list(range(4000, 4032)),
                },
                "aloha_insertion_seeds": {
                    "dev": list(range(3100, 3132)),
                    "sealed_test": list(range(4100, 4132)),
                },
            },
            "track_e_blind_av_desync_and_policy_commit_gate": {
                "description": "5-action PolicyCommitGate calibration (ECE < 0.05), clean grounded accuracy (>= 85%), clean abstention (<= 10%), and blind-desync/OOD abstention (>= 70%).",
                "pool_refs": ["track_e_vggsound_desync", "track_e_greatesthits_desync"],
                "ranking_metric": "calibrated_ece_and_abstain_f1_at_clean_utility",
                "trivial_baseline": "always_emit_forced_decode_and_always_abstain_null",
                "desync_protocol": "symmetric_plus_minus_6hb_0p96s_seam_free_interior_crop_text_stripped",
            },
        },
    }
    canonical = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate pre-locked SLWM-v4 cross-model showcase manifest from finalized valueai-org manifests.")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "reports/v4/showcase_manifest_common_v4.json")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    m = build_showcase_manifest_v4()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(m, indent=2))
    tot_dev = sum(p["n_dev_locked"] for p in m["locked_episode_pools"].values())
    tot_test = sum(p["n_sealed_test_locked"] for p in m["locked_episode_pools"].values())
    print(
        f"Wrote concrete locked showcase manifest -> {args.out} "
        f"(pools={len(m['locked_episode_pools'])}, dev_ids={tot_dev}, test_ids={tot_test}, sha256={m['manifest_sha256'][:16]}...)"
    )
    if args.upload:
        eval_repo = str(VALUEAI_DATASET_CATALOG["eval_benchmarks"]["repo_id"])
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(args.out),
            path_in_repo=f"manifests/{args.out.name}",
            repo_id=eval_repo,
            repo_type="dataset",
            commit_message="Lock concrete pre-training SLWM-v4 showcase_manifest_common_v4.json (v1.1)",
        )
        print(f"[HF_UP OK] {eval_repo}/manifests/{args.out.name}")


if __name__ == "__main__":
    main()
