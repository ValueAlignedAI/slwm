"""SLWM-v4 Gate 2: Local Zero-CU Migration of Date-Disjoint GreatestHits (977 eps) into 160 ms Interleaved v4 Shards + Frozen External Eval Benchmark Upload.

Converts the verified date-disjoint `GreatestHits` (`458 train / 215 dev / 304 test = 977 eps`, `16 / 6 / 6 = 28`
true MIT CSAIL `YYYY-MM-DD` recording dates recovered from the official archive central directory; 1+31 Wan 2.1 groups
@ 25 fps + 62 Mimi 16-RVQ ticks @ 12.5 Hz) into SLWM-v4's strictly causal 160 ms interleaved heartbeat format
(certified int8 Wan 2.1), verifies non-vacuous SHA-256 zero-overlap across train/dev/test, and atomically publishes
all `.npz` shards + paired `*_meta.json` + manifest to `valueai-org/slwm-v4-greatesthits-latents`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import struct
import sys
import urllib.request

from huggingface_hub import CommitOperationAdd, HfApi
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.v4_embodiment_and_splits import audit_v4_splits_non_vacuous
from data.v4_heartbeat_contract import build_interleaved_heartbeat_stream
from scripts.data.v4_preencode_tier1 import StreamingV4ShardWriter, _hf_commit_with_retry
from scripts.data.v4_valueai_registry import VALUEAI_DATASET_CATALOG


def fetch_greatesthits_official_stems_by_split() -> dict[str, list[tuple[str, str]]]:
    """Read official MIT CSAIL `vis-sfs.zip` Central Directory via HTTP Range (< 256 KB) to recover all 977 exact clip stems & 28 true YYYY-MM-DD dates."""
    url = "https://andrewowens.com/vis/vis-sfs.zip"
    req_head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req_head, timeout=30) as resp:
        total_len = int(resp.headers["Content-Length"])

    tail_sz = min(total_len, 65536 + 22)
    req_tail = urllib.request.Request(url, headers={"Range": f"bytes={total_len - tail_sz}-{total_len - 1}"})
    with urllib.request.urlopen(req_tail, timeout=30) as resp:
        tail = resp.read()
    eocd_pos = tail.rfind(b"PK\x05\x06")
    if eocd_pos < 0:
        raise RuntimeError("Could not find ZIP EOCD in official GreatestHits archive")
    cd_size, cd_offset = struct.unpack_from("<II", tail, eocd_pos + 12)

    req_cd = urllib.request.Request(url, headers={"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"})
    with urllib.request.urlopen(req_cd, timeout=30) as resp:
        cd = resp.read()

    stems_set: set[str] = set()
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        fn_len, ex_len, cm_len = struct.unpack_from("<HHH", cd, pos + 28)
        name = cd[pos + 46 : pos + 46 + fn_len].decode("utf-8", errors="ignore")
        pos += 46 + fn_len + ex_len + cm_len
        bn = Path(name).name
        if "_sf." in bn:
            stem = bn.split("_sf.", 1)[0]
            if re.match(r"^\d{4}-\d{2}-\d{2}", stem):
                stems_set.add(stem)

    stems = sorted(stems_set)
    dates = sorted({s[:10] for s in stems})
    if len(dates) != 28 or len(stems) != 977:
        raise RuntimeError(f"Expected 977 stems across 28 dates from GreatestHits, got {len(stems)} stems / {len(dates)} dates")

    date_to_split = {d: ("train" if i < 16 else ("dev" if i < 22 else "test")) for i, d in enumerate(dates)}
    by_split: dict[str, list[tuple[str, str]]] = {"train": [], "dev": [], "test": []}
    for s in stems:
        d = s[:10]
        by_split[date_to_split[d]].append((s, d))
    return by_split


def migrate_greatesthits(shards_root: Path, out_dir: Path, hf_repo: str) -> dict:
    stems_by_split = fetch_greatesthits_official_stems_by_split()
    corpus_dir = out_dir / "greatesthits_impact_av"
    if corpus_dir.exists():
        for f in corpus_dir.glob("*"):
            f.unlink(missing_ok=True)

    # Write locally first, verify non-vacuous disjointness & exact 16/6/6 date clusters, then commit atomically to HF!
    writer = StreamingV4ShardWriter(
        corpus_name="greatesthits_impact_av",
        local_out_dir=out_dir,
        hf_repo_id=None,
        flush_mb=500.0,
        delete_local_after_upload=False,
    )
    for sp in ("train", "dev", "test"):
        p = shards_root / "av" / f"greatesthits_{sp}.npz"
        d = np.load(p, allow_pickle=True)
        reset_all = d["reset_mask"]
        mods_all = d["modality_id"]
        v_idx_all = d["video_idx"]
        a_idx_all = d["audio_idx"]
        v_feat_all = d["video_feat"]
        a_codes_all = d["audio_codes"]
        resets = np.where(reset_all == 1)[0]
        ends = np.append(resets[1:], len(reset_all))
        expected_stems = stems_by_split[sp]
        assert len(resets) == len(expected_stems), f"Split {sp}: npz has {len(resets)} eps vs {len(expected_stems)} official stems!"

        for ep_i, (s0, e0) in enumerate(zip(resets, ends)):
            stem, rec_date = expected_stems[ep_i]
            mods = mods_all[s0:e0]
            v_idx = v_idx_all[s0:e0][mods == 0]
            a_idx = a_idx_all[s0:e0][mods == 1]
            n_grps = int(v_idx.size // 15)
            if n_grps < 2 or a_idx.size < 4:
                continue
            v_all = v_feat_all[v_idx[: n_grps * 15]].reshape(n_grps, 15, 1024)
            f0 = v_all[0]
            v_rest = v_all[1:]
            a16 = a_codes_all[a_idx, :16].astype(np.int16)
            n_hb = min(int(v_rest.shape[0]), int(a16.shape[0] // 2))
            if n_hb < 4:
                continue
            stream = build_interleaved_heartbeat_stream(
                video_init_frame0=f0,
                video_groups=v_rest[:n_hb],
                audio_codes=a16[: n_hb * 2],
            )
            writer.add_episode(
                split=sp,
                episode_id=f"gh_{stem}",
                cluster_key=f"gh::{rec_date}",
                stream=stream,
                duration_s=n_hb * 0.16,
                audio_valid=True,
                extra_meta={"recording_date": rec_date, "source_stem": stem},
            )
    receipt = writer.finalize()
    # Populate uploaded_shards accurately before saving final manifest and committing atomically
    local_npzs = sorted(corpus_dir.glob("*_shard_*.npz"))
    receipt["uploaded_shards"] = [f"greatesthits_impact_av/{p.name}" for p in local_npzs]
    audit = audit_v4_splits_non_vacuous(receipt["episodes"], expected_modalities=("video", "audio"))
    receipt["non_vacuous_disjointness_audit"] = audit
    assert audit["non_vacuous_zero_overlap_pass"] is True, f"GreatestHits overlap detected: {audit}"
    assert audit["unique_cluster_counts"] == {"train": 16, "dev": 6, "test": 6}, f"Unexpected date cluster counts: {audit['unique_cluster_counts']}"

    manifest_path = corpus_dir / "greatesthits_impact_av_preencode_manifest.json"
    manifest_path.write_text(json.dumps(receipt, indent=2))

    # Atomic 1-commit upload of ALL 3 shards + 3 paired _meta.json + 1 finalized manifest to valueai-org!
    api = HfApi()
    ops = []
    for p_f in sorted(corpus_dir.glob("*")):
        if p_f.is_file() and (p_f.name.endswith(".npz") or p_f.name.endswith(".json")):
            ops.append(CommitOperationAdd(path_in_repo=f"greatesthits_impact_av/{p_f.name}", path_or_fileobj=str(p_f)))
    _hf_commit_with_retry(
        api,
        hf_repo,
        ops,
        f"Publish date-disjoint GreatestHits (16/6/6 true YYYY-MM-DD clusters, {len(ops)} paired files)",
    )
    print(f"[GH_UP OK] Atomically committed {len(ops)} files to {hf_repo}", flush=True)
    return receipt


def upload_frozen_eval_benchmarks(shards_root: Path, hf_repo: str) -> list[str]:
    api = HfApi()
    eval_files = [
        ("text_code/wikitext103_validation.npz", "eval_benchmarks/wikitext103_validation.npz"),
        ("text_code/wikitext103_test.npz", "eval_benchmarks/wikitext103_test.npz"),
        ("text_code/lambada_openai_test.npz", "eval_benchmarks/lambada_openai_test.npz"),
        ("text_code/humaneval_test.npz", "eval_benchmarks/humaneval_test.npz"),
        ("text_code/humaneval_exec.jsonl", "eval_benchmarks/humaneval_exec.jsonl"),
        ("text_code/mbpp_sanitized_test.npz", "eval_benchmarks/mbpp_sanitized_test.npz"),
        ("text_code/mbpp_exec.jsonl", "eval_benchmarks/mbpp_exec.jsonl"),
        ("audio_text/librispeech_dev_clean.npz", "eval_benchmarks/librispeech_dev_clean.npz"),
        ("audio_text/librispeech_test_clean.npz", "eval_benchmarks/librispeech_test_clean.npz"),
        ("audio_text/clotho_dev.npz", "eval_benchmarks/clotho_dev.npz"),
        ("audio_text/clotho_test.npz", "eval_benchmarks/clotho_test.npz"),
        ("audio_text/esc50_test_f5.npz", "eval_benchmarks/esc50_test_f5.npz"),
        ("real_speech_overwrite/real_speech_overwrite_dev_L16_Weq128_additive.npz", "eval_benchmarks/rso_dev_L16_Weq128.npz"),
        ("real_speech_overwrite/real_speech_overwrite_test_L16_Weq128_additive.npz", "eval_benchmarks/rso_test_L16_Weq128.npz"),
    ]
    ops = []
    uploaded = []
    for rel_src, rel_dst in eval_files:
        fp = shards_root / rel_src
        if fp.is_file():
            ops.append(CommitOperationAdd(path_in_repo=rel_dst, path_or_fileobj=str(fp)))
            uploaded.append(rel_dst)
    if ops:
        _hf_commit_with_retry(api, hf_repo, ops, f"Sync {len(ops)} frozen external eval benchmarks")
    return uploaded


def main() -> None:
    ap = argparse.ArgumentParser(description="Migrate GreatestHits (real 16/6/6 dates) & optional eval benchmarks to public valueai-org.")
    ap.add_argument("--skip-eval-benchmarks", action="store_true")
    args = ap.parse_args()

    shards_root = _REPO_ROOT / ".ops/v2_shards"
    out_dir = _REPO_ROOT / ".ops/v4_local_migrated"
    gh_repo = VALUEAI_DATASET_CATALOG["greatesthits_impact_av"]["repo_id"]
    eval_repo = VALUEAI_DATASET_CATALOG["eval_benchmarks"]["repo_id"]

    gh_rec = migrate_greatesthits(shards_root, out_dir, gh_repo)
    ev_up = [] if args.skip_eval_benchmarks else upload_frozen_eval_benchmarks(shards_root, eval_repo)
    summary = {
        "greatesthits_repo": gh_repo,
        "greatesthits_n_episodes": gh_rec["n_episodes"],
        "greatesthits_hours": gh_rec["total_hours"],
        "greatesthits_date_clusters": gh_rec["non_vacuous_disjointness_audit"]["unique_cluster_counts"],
        "greatesthits_non_vacuous_pass": gh_rec["non_vacuous_disjointness_audit"]["non_vacuous_zero_overlap_pass"],
        "eval_benchmarks_uploaded": ev_up,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
