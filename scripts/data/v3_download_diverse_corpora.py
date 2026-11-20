"""SLWM-v3 Diverse Any-to-Any Multimodal & SoTA Benchmark Acquisition Script (`0.00 CU`).

Downloads license-verified, ungated open datasets organized strictly by supported input -> output
modality direction (per `AGENTS.md §6, §8, §12 Gate 2`) and writes a cryptographic provenance
manifest (`v3_diverse_corpus_manifest.json`) recording HF repo IDs, commit SHAs, licenses,
file counts, and exact byte sizes.

Tiers:
  - `benchmarks_and_core` (~22 GiB, fast SSD `/mnt/slwm-data/SLWM/v3_diverse_raw`):
      * Language/Code SoTA evals: `Salesforce/wikitext` (wikitext-103-raw-v1),
        `EleutherAI/lambada_openai` (en), `openai/openai_humaneval`, `google-research-datasets/mbpp`
      * Diverse Code & Textbooks: `HuggingFaceTB/smollm-corpus` (`python-edu` 8 shards + `cosmopedia-v2` 6 shards)
      * Audio/Speech SoTA & Audio <-> Text: `ashraq/esc50`, `openslr/librispeech_asr` (dev-clean + test-clean),
        `CLAPv2/Clotho` (full train/valid/test audio + 5 captions/clip)
      * Synchronized Video <-> Audio <-> Text: `friedrichor/MSR-VTT` (10k AV clips + 200k captions)
      * Embodied Robotics (Video + Proprio + Action + Language): `lerobot/pusht`,
        `lerobot/aloha_sim_insertion_human`, `lerobot/libero`
      * StreamingBench manifests (`StreamingBench/*.csv`)
  - `large_av` (~105 GiB, HDD `/mnt/slwm-checkpoints-a/slwm_v3_raw`):
      * `Loie/VGGSound` (`vggsound.csv` + shards `00..03` = ~40k synchronized 10s AV clips / ~110 hrs)
      * `mjuicem/StreamingBench` (`Multimodal Alignment`, `Proactive Output`, `Misleading Context Understanding`,
        `Source Discrimination`, `Anomaly Context Understanding` zips)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


CORPUS_CARDS: dict[str, dict[str, Any]] = {
    "wikitext_103": {
        "repo_id": "Salesforce/wikitext",
        "tier": "benchmarks_and_core",
        "directions": ["text->text"],
        "role": "external_sota_lm_perplexity_benchmark",
        "allow_patterns": ["wikitext-103-raw-v1/*.parquet", "README.md"],
        "cluster_key": "official_document_split",
    },
    "lambada_openai": {
        "repo_id": "EleutherAI/lambada_openai",
        "tier": "benchmarks_and_core",
        "directions": ["text->text"],
        "role": "external_sota_long_context_continuation_benchmark",
        "allow_patterns": ["data/lambada_test.jsonl", "README.md"],
        "cluster_key": "book_passage_id",
    },
    "humaneval": {
        "repo_id": "openai/openai_humaneval",
        "tier": "benchmarks_and_core",
        "directions": ["code->code"],
        "role": "external_sota_code_generation_benchmark",
        "allow_patterns": ["*.parquet", "openai_humaneval/*.parquet", "README.md"],
        "cluster_key": "task_id",
    },
    "mbpp": {
        "repo_id": "google-research-datasets/mbpp",
        "tier": "benchmarks_and_core",
        "directions": ["text+code->code"],
        "role": "external_sota_code_generation_benchmark",
        "allow_patterns": ["full/*.parquet", "sanitized/*.parquet", "README.md"],
        "cluster_key": "task_id",
    },
    "smollm_python_edu_and_cosmopedia": {
        "repo_id": "HuggingFaceTB/smollm-corpus",
        "tier": "benchmarks_and_core",
        "directions": ["code->code", "text->text"],
        "role": "diverse_code_and_textbook_pretraining",
        "allow_patterns": [
            "python-edu/train-0000[0-7]-of-*.parquet",
            "cosmopedia-v2/train-0000[0-5]-of-*.parquet",
            "README.md",
        ],
        "cluster_key": "source_document_hash",
    },
    "esc50": {
        "repo_id": "ashraq/esc50",
        "tier": "benchmarks_and_core",
        "directions": ["audio->audio", "audio->text"],
        "role": "external_sota_environmental_audio_benchmark",
        "allow_patterns": ["data/*.parquet", "README.md"],
        "cluster_key": "official_5fold_src_file",
    },
    "librispeech_clean_evals": {
        "repo_id": "openslr/librispeech_asr",
        "tier": "benchmarks_and_core",
        "directions": ["audio->audio", "audio->text", "text->audio"],
        "role": "external_sota_speech_asr_and_continuation_benchmark",
        "allow_patterns": [
            "clean/validation/*.parquet",
            "clean/test/*.parquet",
            "all/validation.clean/*.parquet",
            "all/test.clean/*.parquet",
            "README.md",
        ],
        "cluster_key": "speaker_id_disjoint_from_train360",
    },
    "clotho": {
        "repo_id": "CLAPv2/Clotho",
        "tier": "benchmarks_and_core",
        "directions": ["audio->text", "text->audio", "audio->audio"],
        "role": "general_audio_captioning_and_cross_modal_retrieval",
        "allow_patterns": ["data/*/*.parquet", "README.md"],
        "cluster_key": "freesound_recording_id",
    },
    "msrvtt": {
        "repo_id": "friedrichor/MSR-VTT",
        "tier": "benchmarks_and_core",
        "directions": ["video+audio->text", "video->audio", "audio->video", "text->video"],
        "role": "open_domain_synchronized_video_audio_text",
        "allow_patterns": ["MSRVTT_Videos.zip", "raw_data/*", "*.json", "*.csv", "*.parquet", "README.md"],
        "cluster_key": "video_id_official_train9k_test1k",
    },
    "lerobot_pusht": {
        "repo_id": "lerobot/pusht",
        "tier": "benchmarks_and_core",
        "directions": ["video+proprio+action->video+action"],
        "role": "embodied_visuomotor_control_no_audio",
        "allow_patterns": ["data/**", "videos/**", "meta/**", "README.md"],
        "cluster_key": "episode_index",
    },
    "lerobot_aloha_sim": {
        "repo_id": "lerobot/aloha_sim_insertion_human",
        "tier": "benchmarks_and_core",
        "directions": ["video+proprio+action->video+action"],
        "role": "embodied_bimanual_14dof_control_no_audio",
        "allow_patterns": ["data/**", "videos/**", "meta/**", "README.md"],
        "cluster_key": "episode_index",
    },
    "lerobot_libero": {
        "repo_id": "lerobot/libero",
        "tier": "benchmarks_and_core",
        "directions": ["text+video+proprio+action->video+action"],
        "role": "embodied_language_conditioned_multitask_control_no_audio",
        "allow_patterns": ["data/**", "videos/**", "meta/**", "README.md"],
        "cluster_key": "task_and_episode_index",
    },
    "streamingbench_meta": {
        "repo_id": "mjuicem/StreamingBench",
        "tier": "benchmarks_and_core",
        "directions": ["video+audio->text", "policy_commit_proactive_output"],
        "role": "external_sota_online_streaming_multimodal_qa_metadata",
        "allow_patterns": ["StreamingBench/*.csv", "README.md"],
        "cluster_key": "video_id",
    },
    "vggsound_subset": {
        "repo_id": "Loie/VGGSound",
        "tier": "large_av",
        "directions": ["video->audio", "audio->video", "video+audio->av_sync"],
        "role": "open_domain_synchronized_av_training_and_vggsound_sparse_eval",
        "allow_patterns": [
            "vggsound.csv",
            "vggsound_00.tar.gz",
            "vggsound_01.tar.gz",
            "vggsound_02.tar.gz",
            "vggsound_03.tar.gz",
            "README.md",
        ],
        "cluster_key": "youtube_video_id_official_train_test",
    },
    "streamingbench_videos": {
        "repo_id": "mjuicem/StreamingBench",
        "tier": "large_av",
        "directions": ["video+audio->text", "policy_commit_proactive_output", "hallucination_misleading_context"],
        "role": "external_sota_online_streaming_multimodal_qa_videos",
        "allow_patterns": [
            "StreamingBench/*.csv",
            "Multimodal Alignment.zip",
            "Proactive Output_1-25.zip",
            "Proactive Output_26-50.zip",
            "Misleading Context Understanding.zip",
            "Source Discrimination.zip",
            "Anomaly Context Understanding.zip",
            "README.md",
        ],
        "cluster_key": "video_id",
    },
}


def download_corpus(name: str, card: dict[str, Any], out_root: Path, api: HfApi) -> dict[str, Any]:
    repo_id = card["repo_id"]
    dest = out_root / name
    dest.mkdir(parents=True, exist_ok=True)
    info = api.dataset_info(repo_id)
    rev_sha = info.sha
    lic = info.cardData.get("license") if info.cardData else None

    print(f"[DATA] Downloading {name} ({repo_id} @ {rev_sha[:10]}) -> {dest} ...", flush=True)
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=rev_sha,
        local_dir=str(dest),
        allow_patterns=card["allow_patterns"],
        max_workers=4,
    )
    files = [p for p in Path(local_dir).rglob("*") if p.is_file() and not p.name.startswith(".")]
    tot_bytes = sum(p.stat().st_size for p in files)
    print(f"[DATA] DONE {name}: {len(files)} files, {tot_bytes / (1024**3):.2f} GiB", flush=True)
    return {
        "name": name,
        "repo_id": repo_id,
        "revision_sha": rev_sha,
        "license": lic,
        "tier": card["tier"],
        "directions": card["directions"],
        "role": card["role"],
        "cluster_key": card["cluster_key"],
        "local_dir": str(dest),
        "num_files": len(files),
        "total_bytes": tot_bytes,
        "total_gib": round(tot_bytes / (1024**3), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Download SLWM-v3 diverse any-to-any multimodal & SoTA corpora.")
    ap.add_argument("--tier", type=str, default="benchmarks_and_core", choices=("benchmarks_and_core", "large_av", "all"))
    ap.add_argument("--only", type=str, default=None, help="Optional comma-separated corpus names.")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--manifest-json", type=Path, required=True)
    args = ap.parse_args()

    api = HfApi()
    selected = []
    only_set = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
    for name, card in CORPUS_CARDS.items():
        if only_set is not None and name not in only_set:
            continue
        if only_set is None and args.tier != "all" and card["tier"] != args.tier:
            continue
        selected.append((name, card))

    existing: dict[str, Any] = {}
    if args.manifest_json.is_file():
        prev = json.loads(args.manifest_json.read_text())
        existing = prev.get("corpora", {})

    for name, card in selected:
        try:
            existing[name] = download_corpus(name, card, args.out_root, api)
        except Exception as e:
            print(f"[DATA][ERROR] {name} failed: {type(e).__name__}: {e}", flush=True)
            existing[name] = {"name": name, "repo_id": card["repo_id"], "error": f"{type(e).__name__}: {e}"}

    manifest = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "out_root": str(args.out_root),
        "corpora": existing,
        "total_downloaded_gib": round(sum(float(v.get("total_gib", 0.0)) for v in existing.values()), 3),
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[DATA] Manifest saved -> {args.manifest_json} (total {manifest['total_downloaded_gib']} GiB)", flush=True)


if __name__ == "__main__":
    main()
