"""Filter exact cross-split duplicate hashes from a prepared T1 token corpus.

The T1 preparation manifest records document token counts and text SHA-256 values.
This script reconstructs split token arrays while removing exact text-hash or
sample-id duplicates that appear in more than one split.  It is not a full
near-duplicate, PII, secrets, or benchmark-contamination filter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


SPLITS = ("train", "validation", "test")
SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def canonical_splits(rows: list[Mapping[str, Any]], key: str) -> dict[str, str]:
    splits_by_key: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = str(row.get(key, ""))
        if value:
            splits_by_key[value].add(str(row.get("split", "unknown")))
    return {
        value: sorted(splits, key=lambda split: SPLIT_PRIORITY.get(split, 99))[0]
        for value, splits in splits_by_key.items()
        if len(splits) > 1
    }


def filter_corpus(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_in = input_dir / "manifest.jsonl"
    rows = read_jsonl(manifest_in)
    canonical_text = canonical_splits(rows, "text_sha256")
    canonical_sample = canonical_splits(rows, "sample_id")
    split_arrays = {split: np.load(input_dir / f"{split}.tokens.npy", mmap_mode="r") for split in SPLITS}
    cursors = {split: 0 for split in SPLITS}
    raw_paths = {split: output_dir / f"{split}.tokens.uint32.tmp" for split in SPLITS}
    handles = {split: raw_paths[split].open("wb") for split in SPLITS}
    kept_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    split_token_counts: Counter[str] = Counter()
    split_doc_counts: Counter[str] = Counter()
    try:
        for row in rows:
            split = str(row["split"])
            token_count = int(row.get("token_count", 0) or 0)
            start = cursors[split]
            end = start + token_count
            cursors[split] = end
            text_sha = str(row.get("text_sha256", ""))
            sample_id = str(row.get("sample_id", ""))
            keep = True
            reasons = []
            if text_sha in canonical_text and split != canonical_text[text_sha]:
                keep = False
                reasons.append("cross_split_text_sha256_duplicate")
            if sample_id in canonical_sample and split != canonical_sample[sample_id]:
                keep = False
                reasons.append("cross_split_sample_id_duplicate")
            if keep:
                np.asarray(split_arrays[split][start:end], dtype=np.uint32).tofile(handles[split])
                kept_rows.append(dict(row))
                split_doc_counts[split] += 1
                split_token_counts[split] += token_count
            else:
                dropped = dict(row)
                dropped["drop_reasons"] = reasons
                dropped_rows.append(dropped)
    finally:
        for handle in handles.values():
            handle.close()
    token_files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        if cursors[split] != int(split_arrays[split].shape[0]):
            raise ValueError(
                f"Manifest token accounting mismatch for {split}: "
                f"consumed {cursors[split]} tokens but array has {int(split_arrays[split].shape[0])}"
            )
        token_path = output_dir / f"{split}.tokens.npy"
        token_count = int(split_token_counts[split])
        mmap = np.lib.format.open_memmap(token_path, mode="w+", dtype=np.uint32, shape=(token_count,))
        copied = 0
        with raw_paths[split].open("rb") as handle:
            while copied < token_count:
                chunk = np.fromfile(handle, dtype=np.uint32, count=min(1_000_000, token_count - copied))
                if chunk.size == 0:
                    break
                mmap[copied : copied + int(chunk.size)] = chunk
                copied += int(chunk.size)
        del mmap
        raw_paths[split].unlink(missing_ok=True)
        token_files[split] = {"path": str(token_path), "sha256": sha256_file(token_path), "tokens": token_count}
    manifest_out = write_jsonl(kept_rows, output_dir / "manifest.jsonl")
    write_jsonl(dropped_rows, output_dir / "dropped_exact_cross_split_duplicates.jsonl")
    card = json.loads((input_dir / "dataset_card.json").read_text(encoding="utf-8"))
    card.update(
        {
            "status": "prepared_exact_split_deduped",
            "source_prepared_corpus_dir": str(input_dir),
            "exact_split_dedup_filter": {
                "method": "drop non-canonical split rows for text_sha256/sample_id values appearing in multiple splits; train split priority is preferred",
                "input_rows": len(rows),
                "kept_rows": len(kept_rows),
                "dropped_rows": len(dropped_rows),
                "cross_split_text_sha256_keys": len(canonical_text),
                "cross_split_sample_id_keys": len(canonical_sample),
                "limitations": [
                    "exact duplicate/split leakage only",
                    "does not remove near duplicates",
                    "does not scan raw text for PII/secrets",
                    "does not check external benchmark contamination",
                ],
            },
            "split_counts": {split: {"documents": int(split_doc_counts[split]), "tokens": int(split_token_counts[split])} for split in SPLITS},
            "token_files": token_files,
            "manifest_path": str(manifest_out),
            "manifest_sha256": sha256_file(manifest_out),
            "dataset_card_path": str(output_dir / "dataset_card.json"),
            "wall_clock_filter_seconds": time.perf_counter() - started,
        }
    )
    shutil.copy2(input_dir / "prepare_config.json", output_dir / "prepare_config.json")
    write_json(card, output_dir / "dataset_card.json")
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter exact cross-split leakage from T1 prepared corpus")
    parser.add_argument("--input-dir", default="artifacts/t1_text_code/gpt2_bpe_1b_v0")
    parser.add_argument("--output-dir", default="artifacts/t1_text_code/gpt2_bpe_1b_v0_exact_dedup")
    args = parser.parse_args()
    print(json.dumps(filter_corpus(Path(args.input_dir), Path(args.output_dir)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
