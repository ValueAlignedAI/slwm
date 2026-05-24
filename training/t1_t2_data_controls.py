"""Manifest-level T1/T2 data-control reports.

These checks are intentionally lightweight and reproducible.  They verify split
accounting, manifest hashes, duplicate IDs/hashes, and obvious cross-split
leakage in the prepared manifests.  They do **not** replace full near-duplicate
deduplication, PII review, code secret scanning over raw text/code, or external
benchmark contamination checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            yield row


def write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def split_leakage(keys_to_splits: Mapping[str, set[str]], *, max_examples: int = 20) -> dict[str, Any]:
    leaked = {key: sorted(splits) for key, splits in keys_to_splits.items() if len(splits) > 1}
    return {
        "leaked_key_count": len(leaked),
        "examples": dict(list(leaked.items())[:max_examples]),
    }


def t1_controls(manifest_path: Path, dataset_card_path: Path) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    text_hash_counts: Counter[str] = Counter()
    sample_id_counts: Counter[str] = Counter()
    text_hash_splits: dict[str, set[str]] = defaultdict(set)
    sample_id_splits: dict[str, set[str]] = defaultdict(set)
    suspicious_sample_ids = 0
    rows = 0
    for row in read_jsonl(manifest_path):
        rows += 1
        split = str(row.get("split", "unknown"))
        category = str(row.get("category", "unknown"))
        source = str(row.get("source_name", row.get("dataset", "unknown")))
        sample_id = str(row.get("sample_id", ""))
        text_sha = str(row.get("text_sha256", ""))
        split_counts[split] += 1
        category_counts[category] += 1
        source_counts[source] += 1
        token_counts[split] += int(row.get("token_count", 0) or 0)
        if sample_id:
            sample_id_counts[sample_id] += 1
            sample_id_splits[sample_id].add(split)
            if EMAIL_RE.search(sample_id):
                suspicious_sample_ids += 1
        if text_sha:
            text_hash_counts[text_sha] += 1
            text_hash_splits[text_sha].add(split)
    duplicate_text_hashes = sum(1 for count in text_hash_counts.values() if count > 1)
    duplicate_sample_ids = sum(1 for count in sample_id_counts.values() if count > 1)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_card_path": str(dataset_card_path),
        "dataset_card_sha256": sha256_file(dataset_card_path),
        "rows": rows,
        "split_document_counts": dict(sorted(split_counts.items())),
        "split_token_counts": dict(sorted(token_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "duplicate_text_hash_count": duplicate_text_hashes,
        "duplicate_sample_id_count": duplicate_sample_ids,
        "text_hash_cross_split_leakage": split_leakage(text_hash_splits),
        "sample_id_cross_split_leakage": split_leakage(sample_id_splits),
        "suspicious_sample_id_email_pattern_count": suspicious_sample_ids,
        "claim_scope": "manifest-level exact duplicate/split checks only; not full near-dedup/PII/secrets/eval-contamination closure",
    }


def t2_controls(name: str, manifest_path: Path, dataset_card_path: Path) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    sample_ids: Counter[str] = Counter()
    sample_id_splits: dict[str, set[str]] = defaultdict(set)
    source_key_splits: dict[str, set[str]] = defaultdict(set)
    natural_av_counts: Counter[str] = Counter()
    caption_use_counts: Counter[str] = Counter()
    rows = 0
    for row in read_jsonl(manifest_path):
        rows += 1
        split = str(row.get("split", "unknown"))
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
        sample_id = str(row.get("sample_id", ""))
        split_counts[split] += 1
        dataset_counts[str(row.get("dataset", "unknown"))] += 1
        if sample_id:
            sample_ids[sample_id] += 1
            sample_id_splits[sample_id].add(split)
        source_key = str(metadata.get("source_id_hash") or metadata.get("youtube_id") or sample_id)
        if source_key:
            source_key_splits[source_key].add(split)
        natural_av_counts[str(metadata.get("natural_av_correspondence", "unknown"))] += 1
        caption_use_counts[str(metadata.get("caption_use", "unknown"))] += 1
    return {
        "name": name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_card_path": str(dataset_card_path),
        "dataset_card_sha256": sha256_file(dataset_card_path),
        "rows": rows,
        "split_counts": dict(sorted(split_counts.items())),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "duplicate_sample_id_count": sum(1 for count in sample_ids.values() if count > 1),
        "sample_id_cross_split_leakage": split_leakage(sample_id_splits),
        "source_key_cross_split_leakage": split_leakage(source_key_splits),
        "natural_av_correspondence_counts": dict(sorted(natural_av_counts.items())),
        "caption_use_counts": dict(sorted(caption_use_counts.items())),
        "claim_scope": "manifest-level split/source checks only; not full media near-duplicate or benchmark contamination closure",
    }


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_controls(out_dir: Path) -> dict[str, Any]:
    t1_root = Path("artifacts/t1_text_code/gpt2_bpe_1b_v0_exact_dedup")
    if not t1_root.exists():
        t1_root = Path("artifacts/t1_text_code/gpt2_bpe_1b_v0")
    t1_manifest = t1_root / "manifest.jsonl"
    t1_card = t1_root / "dataset_card.json"
    external_manifest = Path("/data/slwm/t2_signal_latents/manifests/external_t2_control_v0/manifest.jsonl")
    external_card = Path("artifacts/t2_audio_visual/external_t2_control_v0/dataset_card.json")
    vgg_manifest = Path("/data/slwm/t2_signal_latents/manifests/vggsound_5k_v0/latent_manifest.jsonl")
    vgg_card = Path("artifacts/t2_audio_visual/vggsound_5k_available_v0/dataset_card.json")
    report = {
        "status": "controls_partial_manifest_level",
        "t1": t1_controls(t1_manifest, t1_card),
        "t2_external_control": t2_controls("external_librispeech_coco_control_v0", external_manifest, external_card),
        "t2_vggsound_available": t2_controls("vggsound_5k_available_v0", vgg_manifest, vgg_card),
        "vggsound_media_availability": load_optional_json(Path("/data/slwm/t2_signal_latents/manifests/vggsound_5k_v0/media_availability.summary.json")),
        "vggsound_latent_extraction": load_optional_json(Path("/data/slwm/t2_signal_latents/manifests/vggsound_5k_v0/latent_extraction_report.json")),
        "unclosed_controls": [
            "near-duplicate deduplication over raw text/code is not complete",
            "T1 raw-content PII/secrets scan is not complete",
            "external eval contamination checks (for example LAMBADA/HumanEval/MBPP overlap) are not complete",
            "VGGSound validation/test splits are too small for evidence-bearing natural-AV evaluation",
        ],
    }
    write_json(report, out_dir / "t1_t2_manifest_controls.json")
    md = [
        "# T1/T2 manifest-level data controls",
        "",
        f"Status: `{report['status']}`",
        "",
        "## T1",
        f"- Rows: {report['t1']['rows']}",
        f"- Duplicate text hashes: {report['t1']['duplicate_text_hash_count']}",
        f"- Text-hash cross-split leakage keys: {report['t1']['text_hash_cross_split_leakage']['leaked_key_count']}",
        f"- Sample-id cross-split leakage keys: {report['t1']['sample_id_cross_split_leakage']['leaked_key_count']}",
        "",
        "## T2 external control",
        f"- Rows: {report['t2_external_control']['rows']}",
        f"- Source-key cross-split leakage keys: {report['t2_external_control']['source_key_cross_split_leakage']['leaked_key_count']}",
        f"- Natural AV counts: `{report['t2_external_control']['natural_av_correspondence_counts']}`",
        "",
        "## T2 VGGSound available",
        f"- Rows: {report['t2_vggsound_available']['rows']}",
        f"- Source-key cross-split leakage keys: {report['t2_vggsound_available']['source_key_cross_split_leakage']['leaked_key_count']}",
        f"- Natural AV counts: `{report['t2_vggsound_available']['natural_av_correspondence_counts']}`",
        "",
        "## Not closed by this report",
    ]
    md += [f"- {item}" for item in report["unclosed_controls"]]
    (out_dir / "t1_t2_manifest_controls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run manifest-level T1/T2 data controls")
    parser.add_argument("--out-dir", default="/data/slwm/manifests/controls/t1_t2_manifest_controls_v0")
    args = parser.parse_args()
    print(json.dumps(run_controls(Path(args.out_dir)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
