"""SLWM-v4 Gate 2: Atomic Binary Excision of Upstream Cross-Split Duplicate Eval Episodes (SHA-256 Provenance).

Excises upstream dataset duplicate recordings that crossed cluster boundaries into `dev`/`test`
(e.g., identical episodes duplicated inside upstream `unitreerobotics/G1_*` at shifted episode indices,
and 1 `OpenSound/AudioCaps` YouTube re-upload under a different YouTube ID), re-packs ONLY the affected
`dev`/`test` `.npz` shards with exact offset re-indexing, updates paired `*_meta.json` and corpus manifests
atomically, and appends full SHA-256 provenance to `reports/v4/gate2_eval_exclusion_manifest.json`.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.v4_embodiment_and_splits import audit_v4_splits_non_vacuous
from data.v4_heartbeat_contract import UNIFIED_ACTION_DIM, UNIFIED_PROPRIO_DIM
from scripts.data.v4_preencode_tier1 import _hf_commit_with_retry, cleanup_hf_file
from scripts.data.v4_valueai_registry import VALUEAI_DATASET_CATALOG

STREAM_KEYS = (
    "modality_id", "heartbeat_id", "capture_time_s", "last_sample_time_s",
    "wall_time_s", "dt_s", "reset_mask", "valid_mask", "target_valid_mask",
    "reliability", "spatial_pos", "text_byte",
)


def _filter_shard_npz(src_npz: Path, dst_npz: Path, keep_local_ids: list[int]) -> None:
    """Repack `.npz` shard keeping strictly `keep_local_ids` with exact contiguous offset re-indexing."""
    with np.load(src_npz) as z:
        ep_loc_all = z["episode_local_id"]
        v_idx_all = z["video_idx"]
        a_idx_all = z["audio_idx"]
        m_idx_all = z["motor_idx"]
        vq_all = z["video_q_int8"]
        vxmin_all = z["video_xmin_fp16"]
        vsc_all = z["video_scale_fp16"]
        aud_all = z["audio_codes"]
        act_all = z["action_feat"]
        prop_all = z["proprio_feat"]
        act_m_all = z["episode_action_masks"]
        prop_m_all = z["episode_proprio_masks"]
        succ_all = z["episode_success"]

        out_stream: dict[str, list[np.ndarray]] = {k: [] for k in (*STREAM_KEYS, "video_idx", "audio_idx", "motor_idx", "episode_local_id")}
        vq_l, vxmin_l, vsc_l, aud_l, act_l, prop_l = [], [], [], [], [], []
        act_m_l, prop_m_l, succ_l = [], [], []
        vid_off = aud_off = mot_off = 0

        for new_i, old_i in enumerate(keep_local_ids):
            mask = ep_loc_all == old_i
            n_tok = int(np.sum(mask))
            assert n_tok > 0, f"Empty episode old_local_id={old_i}"
            for k in STREAM_KEYS:
                out_stream[k].append(z[k][mask])
            out_stream["episode_local_id"].append(np.full((n_tok,), new_i, dtype=np.int32))

            vi = v_idx_all[mask].copy()
            ai = a_idx_all[mask].copy()
            mi = m_idx_all[mask].copy()
            vp = vi[vi >= 0]
            ap = ai[ai >= 0]
            mp = mi[mi >= 0]
            if vp.size > 0:
                v0, v1 = int(vp.min()), int(vp.max()) + 1
                vi = np.where(vi >= 0, vi - v0 + vid_off, -1).astype(np.int32)
                vq_l.append(vq_all[v0:v1])
                vxmin_l.append(vxmin_all[v0:v1])
                vsc_l.append(vsc_all[v0:v1])
                vid_off += (v1 - v0)
            if ap.size > 0:
                a0, a1 = int(ap.min()), int(ap.max()) + 1
                ai = np.where(ai >= 0, ai - a0 + aud_off, -1).astype(np.int32)
                aud_l.append(aud_all[a0:a1])
                aud_off += (a1 - a0)
            if mp.size > 0:
                m0, m1 = int(mp.min()), int(mp.max()) + 1
                mi = np.where(mi >= 0, mi - m0 + mot_off, -1).astype(np.int32)
                act_l.append(act_all[m0:m1])
                prop_l.append(prop_all[m0:m1])
                mot_off += (m1 - m0)
            out_stream["video_idx"].append(vi)
            out_stream["audio_idx"].append(ai)
            out_stream["motor_idx"].append(mi)
            act_m_l.append(act_m_all[old_i])
            prop_m_l.append(prop_m_all[old_i])
            succ_l.append(succ_all[old_i])

        packed = {k: np.concatenate(v, axis=0) for k, v in out_stream.items()}
        packed["video_q_int8"] = np.concatenate(vq_l, axis=0) if vid_off else np.zeros((0, 1024), dtype=np.uint8)
        packed["video_xmin_fp16"] = np.concatenate(vxmin_l, axis=0) if vid_off else np.zeros((0, 1), dtype=np.float16)
        packed["video_scale_fp16"] = np.concatenate(vsc_l, axis=0) if vid_off else np.zeros((0, 1), dtype=np.float16)
        packed["audio_codes"] = np.concatenate(aud_l, axis=0) if aud_off else np.zeros((0, 16), dtype=np.int16)
        packed["action_feat"] = np.concatenate(act_l, axis=0) if mot_off else np.zeros((0, UNIFIED_ACTION_DIM), dtype=np.float16)
        packed["proprio_feat"] = np.concatenate(prop_l, axis=0) if mot_off else np.zeros((0, UNIFIED_PROPRIO_DIM), dtype=np.float16)
        packed["episode_action_masks"] = np.stack(act_m_l, axis=0)
        packed["episode_proprio_masks"] = np.stack(prop_m_l, axis=0)
        packed["episode_success"] = np.asarray(succ_l, dtype=np.int8)
        np.savez(dst_npz, **packed)


def excise_corpus_duplicates(
    corpus_key: str,
    modalities: tuple[str, ...],
    work_dir: Path,
) -> list[dict[str, Any]]:
    api = HfApi()
    repo_id = str(VALUEAI_DATASET_CATALOG[corpus_key]["repo_id"])
    mf_rel = f"{corpus_key}/{corpus_key}_preencode_manifest.json"
    lp_mf = hf_hub_download(repo_id, mf_rel, repo_type="dataset")
    mdata = json.loads(Path(lp_mf).read_text())
    eps = list(mdata["episodes"])

    # Index train hashes per modality (excise strictly from dev/test to protect eval integrity!)
    train_hashes: dict[str, dict[str, str]] = {m: {} for m in modalities}
    for r in eps:
        if r["split"] == "train":
            mh = r.get("media_hashes", {})
            for m in modalities:
                h = mh.get(m)
                if h and h not in train_hashes[m]:
                    train_hashes[m][h] = str(r["episode_id"])

    # Also check dev vs test collisions (excise from test if colliding with dev)
    dev_hashes: dict[str, dict[str, str]] = {m: {} for m in modalities}
    excise_map: dict[str, dict[str, Any]] = {}
    for sp_target in ("dev", "test"):
        for r in eps:
            if r["split"] != sp_target:
                continue
            eid = str(r["episode_id"])
            mh = r.get("media_hashes", {})
            matched = None
            for m in modalities:
                h = mh.get(m)
                if h and h in train_hashes[m]:
                    matched = ("train", train_hashes[m][h], m, h)
                    break
                if sp_target == "test" and h and h in dev_hashes[m]:
                    matched = ("dev", dev_hashes[m][h], m, h)
                    break
            if matched is not None:
                ref_sp, ref_eid, mod, hval = matched
                excise_map[eid] = {
                    "corpus": corpus_key,
                    "public_repo_id": repo_id,
                    "excluded_split": sp_target,
                    "episode_id": eid,
                    "cluster_key": str(r["cluster_key"]),
                    "matched_split": ref_sp,
                    "matched_episode_id": ref_eid,
                    "matched_modality": mod,
                    "payload_sha256": hval,
                    "reason": "upstream_duplicate_recording_crossing_split_cluster_boundary",
                }
            elif sp_target == "dev":
                for m in modalities:
                    h = mh.get(m)
                    if h and h not in dev_hashes[m]:
                        dev_hashes[m][h] = eid

    if not excise_map:
        print(f"[EXCISE] {corpus_key}: 0 duplicates found", flush=True)
        return []

    print(f"[EXCISE] {corpus_key}: excising {len(excise_map)} duplicate eval episodes...", flush=True)
    # Inspect ONLY dev/test _meta.json shards to locate exact affected shards!
    info = api.repo_info(repo_id, repo_type="dataset")
    eval_metas = sorted(
        s.rfilename for s in (info.siblings or [])
        if s.rfilename.startswith(f"{corpus_key}/")
        and (s.rfilename.startswith(f"{corpus_key}/dev_") or s.rfilename.startswith(f"{corpus_key}/test_"))
        and s.rfilename.endswith("_meta.json")
    )
    c_dir = work_dir / corpus_key
    c_dir.mkdir(parents=True, exist_ok=True)
    ops: list[Any] = []
    removed_shards: set[str] = set()

    for meta_rel in eval_metas:
        lp_m = hf_hub_download(repo_id, meta_rel, repo_type="dataset")
        sm = json.loads(Path(lp_m).read_text())
        s_eps = list(sm.get("episodes", []))
        bad_idxs = [i for i, r in enumerate(s_eps) if str(r["episode_id"]) in excise_map]
        if not bad_idxs:
            continue
        keep_idxs = [i for i in range(len(s_eps)) if i not in set(bad_idxs)]
        npz_rel = meta_rel.replace("_meta.json", ".npz")
        print(f"  [SHARD] {npz_rel}: keeping {len(keep_idxs)}/{len(s_eps)} eps (removing {len(bad_idxs)})", flush=True)
        assert len(keep_idxs) > 0, f"Entire shard {npz_rel} would become empty; handle delete explicitly!"
        lp_npz = hf_hub_download(repo_id, npz_rel, repo_type="dataset")
        out_npz = c_dir / Path(npz_rel).name
        out_meta = c_dir / Path(meta_rel).name
        _filter_shard_npz(Path(lp_npz), out_npz, keep_idxs)
        cleanup_hf_file(lp_npz)

        new_s_eps = [s_eps[i] for i in keep_idxs]
        sm["n_episodes"] = len(new_s_eps)
        sm["episodes"] = new_s_eps
        out_meta.write_text(json.dumps(sm))
        ops.append(CommitOperationAdd(path_in_repo=npz_rel, path_or_fileobj=str(out_npz)))
        ops.append(CommitOperationAdd(path_in_repo=meta_rel, path_or_fileobj=str(out_meta)))

    # Rebuild finalized manifest strictly without excised episodes
    new_eps = [r for r in eps if str(r["episode_id"]) not in excise_map]
    hrs_by_sp = {"train": 0.0, "dev": 0.0, "test": 0.0}
    pkts_by_sp = {"train": 0, "dev": 0, "test": 0}
    for r in new_eps:
        sp = str(r["split"])
        hrs_by_sp[sp] += float(r.get("duration_s", 0.0)) / 3600.0
        pkts_by_sp[sp] += int(r.get("n_packets", 0))
    mdata["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    mdata["n_episodes"] = len(new_eps)
    mdata["hours_by_split"] = {k: round(v, 3) for k, v in hrs_by_sp.items()}
    mdata["total_hours"] = round(sum(hrs_by_sp.values()), 3)
    mdata["packets_by_split"] = pkts_by_sp
    mdata["total_packets"] = sum(pkts_by_sp.values())
    mdata["episodes"] = new_eps

    # Assert 0/0/0 non-vacuous overlap BEFORE committing to HF!
    aud = audit_v4_splits_non_vacuous(new_eps, expected_modalities=modalities)
    assert aud["non_vacuous_zero_overlap_pass"] is True, f"Post-excision audit failed for {corpus_key}: {aud}"
    out_mf = c_dir / Path(mf_rel).name
    out_mf.write_text(json.dumps(mdata, indent=2))
    ops.append(CommitOperationAdd(path_in_repo=mf_rel, path_or_fileobj=str(out_mf)))

    _hf_commit_with_retry(
        api,
        repo_id,
        ops,
        f"Gate 2: atomically excise {len(excise_map)} upstream cross-split duplicate eval episodes (0/0/0 SHA-256 verified)",
    )
    print(f"[EXCISE OK] {repo_id}: committed {len(ops)} files; post-excision clusters={aud['unique_cluster_counts']} pass=True", flush=True)
    gc.collect()
    return sorted(excise_map.values(), key=lambda x: (x["corpus"], x["excluded_split"], x["episode_id"]))


def main() -> None:
    work_dir = _REPO_ROOT / ".ops/v4_excise_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)
    excl_path = _REPO_ROOT / "reports/v4/gate2_eval_exclusion_manifest.json"
    existing = json.loads(excl_path.read_text()) if excl_path.is_file() else {"exclusions": []}
    excl_list = list(existing.get("exclusions", []))
    seen_eids = {
        (str(x.get("corpus", "")), str(x.get("episode_id") or x.get("excluded_episode_id") or "").strip())
        for x in excl_list
    }

    new_excls: list[dict[str, Any]] = []
    new_excls.extend(excise_corpus_duplicates("unitree_g1_h1_humanoid", ("video", "motor"), work_dir))
    new_excls.extend(excise_corpus_duplicates("wavcaps_and_libriheavy_speech", ("audio",), work_dir))

    for ex in new_excls:
        key = (ex["corpus"], ex["episode_id"])
        if key not in seen_eids:
            excl_list.append(ex)
            seen_eids.add(key)

    out_excl = {
        "gate": "SLWM-v4 Gate 2 — Pre-Training Cross-Split Upstream Duplicate Eval Exclusion Manifest",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Upstream duplicate recordings crossing disjoint split cluster boundaries are atomically excised strictly from dev/test shards prior to training and permanently excluded from showcase pools and evaluation.",
        "n_excluded_eval_episodes": len(excl_list),
        "exclusions": excl_list,
    }
    excl_path.write_text(json.dumps(out_excl, indent=2))
    api = HfApi()
    eval_repo = str(VALUEAI_DATASET_CATALOG["eval_benchmarks"]["repo_id"])
    api.upload_file(
        path_or_fileobj=str(excl_path),
        path_in_repo=f"manifests/{excl_path.name}",
        repo_id=eval_repo,
        repo_type="dataset",
        commit_message=f"Update gate2_eval_exclusion_manifest.json ({len(excl_list)} total upstream duplicates excised)",
    )
    print(f"[EXCLUSION MANIFEST OK] {excl_path} ({len(excl_list)} total exclusions) uploaded to {eval_repo}", flush=True)


if __name__ == "__main__":
    main()
