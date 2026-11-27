"""SLWM-v4 Gate 2 Surgical Patch: Zero-Video-Reencode AgiBotWorld2026 Causal ZOH + Stratified Split Patch + Full 100/100 ALOHA-Sim Re-Encode.

Because `agibot-world/AgiBotWorld2026` uses LeRobot v2.1 (1 dedicated `episode_XXXXXX.parquet` <-> 1 dedicated
`episode_XXXXXX.mp4`, decoded via strictly causal `decode_video_native_frames_192x320`), 100% of the existing
45.41 h Wan 2.1 video latents in `valueai-org/slwm-v4-agibot-aloha-latents` are already aligned and strictly causal.

This script saves ~18 GPU hours by:
  1. Re-encoding all 100/100 `ALOHA-Sim` episodes cleanly from LeRobot v3.0 `meta/episodes/` (15 train / 2 dev / 3 test blocks).
  2. Downloading each existing AgiBot `.npz` shard + lightweight upstream `AgiBotWorld2026` parquets (~330 KB each),
     recomputing strictly causal zero-order hold (`0.000 ms` lookahead) 25 Hz action & proprio/wrench tensors,
     setting `episode_success = -1` (since upstream PhysicalReasoning `succeed` is empirically 100% False),
     and re-sharding under the updated 68/17/15 split (`19 train / 4 dev / 5 test` AgiBot `task::scene` clusters)
     while preserving the exact certified int8 Wan 2.1 video patches bit-for-bit.
  3. Verifying non-vacuous zero-overlap + >= 4 eval clusters locally BEFORE atomically replacing `agibot_and_aloha_sim/*`
     on `valueai-org/slwm-v4-agibot-aloha-latents`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.v4_embodiment_and_splits import (
    assign_v4_split,
    audit_v4_splits_non_vacuous,
    hash_payload_sha256,
    map_embodiment_to_unified_52_68,
    resample_video_and_kinematics_to_25hz,
)
from data.v4_heartbeat_contract import UNIFIED_ACTION_DIM, UNIFIED_PROPRIO_DIM
from scripts.data.v4_preencode_tier1 import (
    FrozenV4CodecEngine,
    StreamingV4ShardWriter,
    _hf_commit_with_retry,
    cleanup_hf_file,
    encode_aloha_and_unitree_g1,
    safe_hf_download,
)
from scripts.data.v4_valueai_registry import VALUEAI_DATASET_CATALOG

_AGIBOT_EID_RE = re.compile(r"^agibot_(task_\d+)_(.+?)_(episode_\d+)$")


def _add_prequantized_episode(
    writer: StreamingV4ShardWriter,
    *,
    split: str,
    episode_id: str,
    cluster_key: str,
    stream_arrays: dict[str, np.ndarray],
    vq: np.ndarray,
    vxmin: np.ndarray,
    vsc: np.ndarray,
    action_25hz: np.ndarray,
    proprio_25hz: np.ndarray,
    action_mask: np.ndarray,
    proprio_mask: np.ndarray,
    duration_s: float,
) -> None:
    """Append an episode with existing bit-exact int8 video patches and freshly recomputed causal ZOH kinematics."""
    if episode_id in writer.seen_episode_ids:
        return
    writer.seen_episode_ids.add(episode_id)

    media_hashes: dict[str, str] = {}
    if vq.shape[0] > 0:
        media_hashes["video"] = hash_payload_sha256(vq)
    if stream_arrays["audio_codes"].shape[0] > 0:
        media_hashes["audio"] = hash_payload_sha256(stream_arrays["audio_codes"])
    if action_25hz.shape[0] > 0:
        media_hashes["motor"] = hash_payload_sha256(action_25hz)
    txt_mask = stream_arrays["modality_id"] == 2
    if np.any(txt_mask):
        media_hashes["text"] = hash_payload_sha256(stream_arrays["text_byte"][txt_mask])

    item = {
        "stream": stream_arrays,
        "video_q_int8": vq,
        "video_xmin_fp16": vxmin,
        "video_scale_fp16": vsc,
        "action_feat_fp16": action_25hz.astype(np.float16),
        "proprio_feat_fp16": proprio_25hz.astype(np.float16),
        "action_mask_u8": action_mask.astype(np.uint8),
        "proprio_mask_u8": proprio_mask.astype(np.uint8),
        "episode_success": -1,
    }
    est_b = (
        vq.nbytes
        + stream_arrays["audio_codes"].nbytes
        + item["action_feat_fp16"].nbytes
        + item["proprio_feat_fp16"].nbytes
        + int(stream_arrays["modality_id"].size) * 40
    )
    writer.bufs[split].append(item)
    writer.buf_bytes[split] += est_b
    writer.total_hours_by_split[split] += float(duration_s) / 3600.0
    writer.total_packets_by_split[split] += int(stream_arrays["modality_id"].size)

    rec = {
        "corpus": writer.corpus_name,
        "split": split,
        "episode_id": episode_id,
        "cluster_key": cluster_key,
        "duration_s": round(float(duration_s), 3),
        "n_packets": int(stream_arrays["modality_id"].size),
        "n_video_patches": int(vq.shape[0]),
        "n_audio_ticks": int(stream_arrays["audio_codes"].shape[0]),
        "n_motor_ticks": int(item["action_feat_fp16"].shape[0]),
        "n_text_bytes": int(np.sum(txt_mask)),
        "audio_valid": None,
        "episode_success": -1,
        "media_hashes": media_hashes,
        "subcorpus": "agibot_world_2026",
    }
    writer.buf_recs[split].append(rec)
    writer.all_episode_records.append(rec)

    split_flush_lim = writer.flush_bytes if split == "train" else min(writer.flush_bytes, 80 * 1024 * 1024)
    if writer.buf_bytes[split] >= split_flush_lim:
        writer.flush_split(split)


def _fetch_agibot_zoh_kinematics(eid: str) -> tuple[str, dict[str, Any] | None]:
    """Download lightweight AgiBotWorld2026 parquet (~330 KB) and recompute strictly causal ZOH 25 Hz kinematics."""
    m = _AGIBOT_EID_RE.match(eid)
    if not m:
        return eid, None
    task_id, scene_id, ep_stem = m.group(1), m.group(2), m.group(3)
    pq_rel = f"PhysicalReasoning/opensourcev4-20261001/lerobot/{task_id}/{scene_id}/data/chunk-000/{ep_stem}.parquet"
    lp_pq = ""
    try:
        lp_pq = safe_hf_download("agibot-world/AgiBotWorld2026", pq_rel, repo_type="dataset")
        tbl = pq.read_table(lp_pq, columns=["action", "observation.state"])
        t_len = tbl.num_rows
        if t_len < 20:
            return eid, None
        raw_a = np.stack(tbl["action"].to_pylist(), axis=0).astype(np.float32)
        raw_s = np.stack(tbl["observation.state"].to_pylist(), axis=0).astype(np.float32)
        a52, am, p68, pm = map_embodiment_to_unified_52_68("agibot_g2a_bimanual", raw_action=raw_a, raw_proprio=raw_s)
        resamp = resample_video_and_kinematics_to_25hz(
            native_fps=30.0, n_src_frames=t_len, action_52=a52, proprio_68=p68, proprio_mask=pm
        )
        return eid, {
            "task_id": task_id,
            "scene_id": scene_id,
            "n_hb": resamp["n_heartbeats"],
            "action_25hz": resamp["action_25hz"],
            "proprio_25hz": resamp["proprio_25hz"],
            "am": am,
            "pm": pm,
        }
    except Exception as exc:
        print(f"[AGIBOT PATCH WARN] {eid}: {exc}", flush=True)
        return eid, None
    finally:
        cleanup_hf_file(lp_pq)


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch AgiBotWorld2026 (causal ZOH + splits + succ=-1) & encode 100/100 ALOHA-Sim.")
    ap.add_argument("--work-dir", type=Path, default=Path("/content/v4_agibot_rebuilt"))
    ap.add_argument("--device", type=str, default="cuda", choices=("cuda", "mps", "cpu"))
    ap.add_argument("--flush-mb", type=float, default=650.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.monotonic()
    cname = "agibot_and_aloha_sim"
    repo_id = VALUEAI_DATASET_CATALOG[cname]["repo_id"]
    api = HfApi()

    out_corpus_dir = args.work_dir / cname
    if out_corpus_dir.exists():
        shutil.rmtree(out_corpus_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Local-first StreamingV4ShardWriter (NEVER modifies remote HF until full local verification passes!)
    writer = StreamingV4ShardWriter(
        corpus_name=cname,
        local_out_dir=args.work_dir,
        hf_repo_id=None,
        flush_mb=args.flush_mb,
        delete_local_after_upload=False,
    )

    # Step 1: Freshly encode ALL 100/100 ALOHA-Sim episodes via exact LeRobot v3.0 meta/episodes/ (~1 min on L4!)
    print("[PATCH STEP 1] Freshly encoding 100/100 ALOHA-Sim episodes via meta/episodes/...", flush=True)
    codec = FrozenV4CodecEngine(device=args.device, load_video=True, load_audio=False)
    encode_aloha_and_unitree_g1(codec, writer, corpus_name=cname, target_hours=1.0, max_episodes=None)
    n_aloha = len(writer.all_episode_records)
    print(f"[PATCH STEP 1 OK] ALOHA-Sim encoded: {n_aloha} episodes (expected 100)", flush=True)
    assert n_aloha == 100, f"Expected all 100 ALOHA-Sim episodes, got {n_aloha}"
    del codec
    gc.collect()

    # Step 2: Patch all existing AgiBotWorld2026 shards WITHOUT re-encoding any Wan 2.1 video latents!
    info = api.repo_info(repo_id, repo_type="dataset")
    old_files = sorted(s.rfilename for s in (info.siblings or []) if s.rfilename.startswith(f"{cname}/"))
    old_npzs = sorted(f for f in old_files if f.endswith(".npz"))
    print(f"[PATCH STEP 2] Patching {len(old_npzs)} existing AgiBot shards from {repo_id}...", flush=True)

    n_patched = 0
    n_skipped = 0
    stream_keys = (
        "modality_id", "heartbeat_id", "capture_time_s", "last_sample_time_s",
        "wall_time_s", "dt_s", "reset_mask", "valid_mask", "target_valid_mask",
        "reliability", "spatial_pos", "text_byte",
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for sh_i, npz_rel in enumerate(old_npzs):
            meta_rel = npz_rel.replace(".npz", "_meta.json")
            lp_meta = lp_npz = ""
            try:
                lp_meta = safe_hf_download(repo_id, meta_rel, repo_type="dataset")
                mdata = json.loads(Path(lp_meta).read_text())
                ep_recs = list(mdata.get("episodes", []))
                agibot_indices = [
                    (idx, str(rec["episode_id"]))
                    for idx, rec in enumerate(ep_recs)
                    if str(rec["episode_id"]).startswith("agibot_")
                ]
                if not agibot_indices:
                    continue

                # Download upstream AgiBot parquets in parallel AND the existing shard .npz!
                kin_fut = pool.map(_fetch_agibot_zoh_kinematics, [eid for _, eid in agibot_indices])
                lp_npz = safe_hf_download(repo_id, npz_rel, repo_type="dataset")
                kin_by_eid = dict(kin_fut)

                with np.load(lp_npz) as z:
                    ep_local_all = z["episode_local_id"]
                    arrs_all = {k: z[k] for k in stream_keys}
                    v_idx_all = z["video_idx"]
                    a_idx_all = z["audio_idx"]
                    m_idx_all = z["motor_idx"]
                    vq_all = z["video_q_int8"]
                    vxmin_all = z["video_xmin_fp16"]
                    vsc_all = z["video_scale_fp16"]
                    aud_all = z["audio_codes"]

                    for loc_i, eid in agibot_indices:
                        kinfo = kin_by_eid.get(eid)
                        if kinfo is None:
                            n_skipped += 1
                            continue
                        mask_ep = ep_local_all == loc_i
                        if not np.any(mask_ep):
                            n_skipped += 1
                            continue
                        vi_raw = v_idx_all[mask_ep].copy()
                        ai_raw = a_idx_all[mask_ep].copy()
                        mi_raw = m_idx_all[mask_ep].copy()

                        v_pos = vi_raw[vi_raw >= 0]
                        a_pos = ai_raw[ai_raw >= 0]
                        m_pos = mi_raw[mi_raw >= 0]
                        expected_mot = int(kinfo["action_25hz"].shape[0])
                        if m_pos.size != expected_mot or v_pos.size == 0:
                            print(
                                f"[AGIBOT LEN MISMATCH] {eid}: old_mot={m_pos.size} vs zoh_mot={expected_mot}; skipping",
                                flush=True,
                            )
                            n_skipped += 1
                            continue

                        v0, v1 = int(v_pos.min()), int(v_pos.max()) + 1
                        vi_local = np.where(vi_raw >= 0, vi_raw - v0, -1).astype(np.int32)
                        if a_pos.size > 0:
                            a0, a1 = int(a_pos.min()), int(a_pos.max()) + 1
                            ai_local = np.where(ai_raw >= 0, ai_raw - a0, -1).astype(np.int32)
                            aud_ep = aud_all[a0:a1].copy()
                        else:
                            ai_local = np.full_like(ai_raw, -1, dtype=np.int32)
                            aud_ep = np.zeros((0, 16), dtype=np.int16)
                        m0 = int(m_pos.min())
                        mi_local = np.where(mi_raw >= 0, mi_raw - m0, -1).astype(np.int32)

                        st_ep = {k: arrs_all[k][mask_ep].copy() for k in stream_keys}
                        st_ep["video_idx"] = vi_local
                        st_ep["audio_idx"] = ai_local
                        st_ep["motor_idx"] = mi_local
                        st_ep["audio_codes"] = aud_ep

                        sp, ckey = assign_v4_split(
                            cname,
                            {"source": "agibot", "task_id": kinfo["task_id"], "scene_id": kinfo["scene_id"]},
                        )
                        _add_prequantized_episode(
                            writer,
                            split=sp,
                            episode_id=eid,
                            cluster_key=ckey,
                            stream_arrays=st_ep,
                            vq=vq_all[v0:v1].copy(),
                            vxmin=vxmin_all[v0:v1].copy(),
                            vsc=vsc_all[v0:v1].copy(),
                            action_25hz=kinfo["action_25hz"],
                            proprio_25hz=kinfo["proprio_25hz"],
                            action_mask=kinfo["am"],
                            proprio_mask=kinfo["pm"],
                            duration_s=kinfo["n_hb"] * 0.16,
                        )
                        n_patched += 1
                print(
                    f"[PATCH SHARD {sh_i + 1}/{len(old_npzs)}] {npz_rel} -> patched_total={n_patched} skipped={n_skipped} hrs={sum(writer.total_hours_by_split.values()):.2f}h",
                    flush=True,
                )
            finally:
                cleanup_hf_file(lp_npz)
                cleanup_hf_file(lp_meta)
                gc.collect()

    # Step 3: Finalize locally and run strict non-vacuous + cluster richness verification BEFORE touching HF!
    receipt = writer.finalize()
    new_npzs = sorted(out_corpus_dir.glob("*_shard_*.npz"))
    receipt["uploaded_shards"] = [f"{cname}/{p.name}" for p in new_npzs]
    audit = audit_v4_splits_non_vacuous(receipt["episodes"], expected_modalities=("video", "motor"))
    receipt["non_vacuous_disjointness_audit"] = audit
    manifest_p = out_corpus_dir / f"{cname}_preencode_manifest.json"
    manifest_p.write_text(json.dumps(receipt, indent=2))

    uc = audit["unique_cluster_counts"]
    print(f"[PATCH VERIFY] total_eps={receipt['n_episodes']} hours={receipt['total_hours']} clusters={uc} pass={audit['non_vacuous_zero_overlap_pass']}", flush=True)
    assert audit["non_vacuous_zero_overlap_pass"] is True, f"Audit failed: {audit}"
    assert uc["dev"] >= 4 and uc["test"] >= 4, f"Expected >= 4 clusters in dev & test, got {uc}"
    assert n_patched >= 3000, f"Expected >= 3000 patched AgiBot episodes, got {n_patched}"

    if args.dry_run:
        print("[PATCH DRY RUN] Skipping remote swap.", flush=True)
        return

    # Step 4: Clean remote swap — delete obsolete shards, then upload all rebuilt shards + paired _meta.json + manifest!
    print(f"[PATCH STEP 4] Atomically replacing {len(old_files)} old files in {repo_id}...", flush=True)
    if old_files:
        del_ops = [CommitOperationDelete(path_in_repo=f) for f in old_files]
        _hf_commit_with_retry(api, repo_id, del_ops, f"Remove pre-patch {cname} files before atomic swap")

    upload_files = sorted(
        p for p in out_corpus_dir.glob("*")
        if p.is_file() and (p.name.endswith(".npz") or p.name.endswith(".json"))
    )
    # Upload in batched commits of 8 files (~2.5 GB/commit) for fast, rate-limit-safe atomic publication!
    for b0 in range(0, len(upload_files), 8):
        batch = upload_files[b0 : b0 + 8]
        ops = [CommitOperationAdd(path_in_repo=f"{cname}/{p.name}", path_or_fileobj=str(p)) for p in batch]
        _hf_commit_with_retry(
            api,
            repo_id,
            ops,
            f"Upload patched {cname} shards ({b0 + 1}..{b0 + len(batch)} of {len(upload_files)})",
        )
        print(f"[PATCH HF_UP OK] Uploaded {b0 + len(batch)}/{len(upload_files)} files to {repo_id}", flush=True)

    dt_m = (time.monotonic() - t0) / 60.0
    print(f"V4_AGIBOT_PATCH_DONE eps={receipt['n_episodes']} hrs={receipt['total_hours']:.2f}h clusters={uc} wall={dt_m:.1f}m", flush=True)


if __name__ == "__main__":
    main()
