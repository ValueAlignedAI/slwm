"""Prepare Sprint T2 audio/visual latent corpora.

T2 uses frozen or precomputed features/codecs.  This script standardizes local
latent arrays into a manifest-backed corpus that the T2 training runner can load.
It can also generate a tiny deterministic latent fixture for smoke tests; that
fixture is explicitly **not** dataset-quality evidence.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from data.audio_visual_latents import (
    DEFAULT_T2_SCHEMA_VERSION,
    T2_REQUIRED_MODALITIES,
    assign_splits,
    records_from_source,
    sha256_json,
    standardize_record,
    validate_t2_audio_visual_data_config,
    write_jsonl,
)
from utils.config import config_hash, load_config, write_config


ARTIFACT_ROOT = Path("artifacts/t2_audio_visual")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_output_dir(raw_path: str | Path) -> Path:
    """Require T2 prepared corpora to live under ``artifacts/t2_audio_visual``."""

    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("T2 output_dir must be relative")
    repo_root = Path.cwd().resolve()
    artifact_root = (repo_root / ARTIFACT_ROOT).resolve()
    resolved = (repo_root / path).resolve()
    if not _is_relative_to(resolved, artifact_root):
        raise ValueError(f"T2 prepared latents must be written under {ARTIFACT_ROOT}")
    if resolved.exists() and resolved.is_symlink():
        raise ValueError("T2 output_dir must not be a symlink")
    return path


def _data_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}


def _runtime_seed(config: Mapping[str, Any]) -> int:
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    return int(runtime.get("seed", 0))


def _write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _collect_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    data_cfg = _data_config(config)
    sources = data_cfg.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("T2 prepare config requires data.sources")
    records: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("T2 data.sources entries must be objects")
        source_records = records_from_source(source, default_seed=_runtime_seed(config))
        source_name = str(source.get("name", source.get("dataset", source.get("type", "unknown_source"))))
        for record in source_records:
            enriched = dict(record)
            enriched.setdefault("source", source_name)
            enriched.setdefault("dataset", str(source.get("dataset", source_name)))
            enriched.setdefault("license", str(source.get("license", record.get("license", "unknown"))))
            records.append(enriched)
    if not records:
        raise ValueError("T2 sources produced no records")
    return records


def prepare_t2_audio_visual_latents(config_path: str | Path) -> dict[str, Any]:
    """Prepare a manifest-backed T2 latent corpus and return its dataset card."""

    path = Path(config_path)
    config = load_config(path)
    data_cfg = _data_config(config)
    validate_t2_audio_visual_data_config(data_cfg)
    output_dir = _safe_output_dir(str(data_cfg.get("output_dir", "artifacts/t2_audio_visual/t2_av_latents_v0")))
    seed = _runtime_seed(config)
    split_policy = data_cfg.get("split_policy", {}) if isinstance(data_cfg.get("split_policy", {}), Mapping) else {}
    audio_length = int(data_cfg.get("max_audio_length", data_cfg.get("audio_length", 64)))
    visual_length = int(data_cfg.get("max_visual_length", data_cfg.get("visual_length", 64)))
    if audio_length <= 0 or visual_length <= 0:
        raise ValueError("T2 max_audio_length and max_visual_length must be positive")

    records = _collect_records(config)
    splits = assign_splits(records, split_policy=split_policy, seed=seed)
    manifest_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    dataset_names: set[str] = set()
    license_notes: set[str] = set()
    split_counts: dict[str, dict[str, Any]] = {}
    for split_name, split_records in splits.items():
        split_dir = output_dir / "latents" / split_name
        for index, record in enumerate(split_records):
            sample_id = str(record.get("sample_id", f"{split_name}-{index:06d}"))
            safe_sample = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in sample_id)
            prepared = standardize_record(
                record,
                output_path=split_dir / f"{safe_sample}.npz",
                split=split_name,
                audio_length=audio_length,
                visual_length=visual_length,
            )
            row = prepared.__dict__
            manifest_rows[split_name].append(row)
            dataset_names.add(prepared.dataset)
            license_notes.add(prepared.license)
        write_jsonl(manifest_rows[split_name], output_dir / "manifests" / f"{split_name}.jsonl")
        split_counts[split_name] = {
            "samples": len(manifest_rows[split_name]),
            "audio_frames": sum(int(row["audio_valid_length"]) for row in manifest_rows[split_name]),
            "visual_frames": sum(int(row["visual_valid_length"]) for row in manifest_rows[split_name]),
        }

    manifest_hash_payload = {
        "splits": manifest_rows,
        "config_hash": config_hash(config),
        "schema": data_cfg.get("sample_schema_version", DEFAULT_T2_SCHEMA_VERSION),
    }
    dataset_card: dict[str, Any] = {
        "status": "prepared",
        "sprint": "T2",
        "created_at": date.today().isoformat(),
        "config_path": str(path),
        "config_hash": config_hash(config),
        "manifest_sha256": sha256_json(manifest_hash_payload),
        "modalities": list(T2_REQUIRED_MODALITIES),
        "sample_schema_version": data_cfg.get("sample_schema_version", DEFAULT_T2_SCHEMA_VERSION),
        "output_dir": str(output_dir),
        "split_counts": split_counts,
        "feature_spec": {
            "audio_codec_or_features": data_cfg.get("audio_codec_or_features", "precomputed_audio_latents"),
            "visual_codec_or_features": data_cfg.get("visual_codec_or_features", "precomputed_visual_latents"),
            "audio_length": audio_length,
            "visual_length": visual_length,
            "audio_feature_dim": manifest_rows["train"][0]["audio_shape"][1] if manifest_rows["train"] else None,
            "visual_feature_dim": manifest_rows["train"][0]["visual_shape"][1] if manifest_rows["train"] else None,
            "dtype": "float32_npz",
        },
        "dataset_mix": data_cfg.get("dataset_mix", {"audio": 0.5, "visual_video": 0.5}),
        "datasets": [
            {
                "name": name,
                "version_or_snapshot": data_cfg.get("dataset_versions", {}).get(name, "recorded_in_source_manifest")
                if isinstance(data_cfg.get("dataset_versions", {}), Mapping)
                else "recorded_in_source_manifest",
                "license_notes": "; ".join(sorted(license_notes)) or "unknown",
                "leakage_checks": data_cfg.get(
                    "leakage_checks",
                    "stable split before training; duplicate/dead-media checks must be recorded for external corpora",
                ),
            }
            for name in sorted(dataset_names)
        ],
        "source_notes": data_cfg.get("source_notes", []),
        "claim_language_allowed": (
            "Prepared T2 latent corpus can support loader/training mechanics only until external dataset versions, "
            "feature extractors, split hashes, and shuffled/null controls are registered."
        ),
    }
    _write_json(dataset_card, output_dir / "dataset_card.json")
    write_config(config, output_dir / "prepare_config.json")
    return dataset_card


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Sprint T2 audio/visual latent corpus")
    parser.add_argument("--config", required=True, help="Path to a T2 prepare JSON config")
    args = parser.parse_args(argv)
    card = prepare_t2_audio_visual_latents(args.config)
    print(json.dumps(card, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())


__all__ = ["prepare_t2_audio_visual_latents"]
