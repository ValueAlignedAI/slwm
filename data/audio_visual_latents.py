"""Sprint T2 audio/visual latent dataset helpers.

This module keeps Sprint T2 data handling focused on **precomputed or frozen**
audio and visual/video latent features.  It deliberately does not download raw
media, train codecs, decode waveforms, generate video, or introduce text
generation objectives.

Prepared latent files use this per-sample NPZ contract::

    audio_features: FloatTensor[T_audio,A]
    visual_features: FloatTensor[T_visual,V]
    audio_mask: BoolTensor[T_audio]
    visual_mask: BoolTensor[T_visual]

Batch helpers return padded tensors compatible with the project data contract:
audio modality id ``2`` and visual/video modality id ``3``.  Downstream adapters
map these arrays into the shared ``Z: FloatTensor[B,T,D]`` field.
"""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from data.contract import MODALITY_IDS, SOURCE_TAGS


T2_REQUIRED_MODALITIES: tuple[str, str] = ("audio", "visual_video")
DEFAULT_T2_SCHEMA_VERSION = "t2.audio_visual_latents_v0"
T2_PREPARED_ROOT = Path("artifacts/t2_audio_visual")
MAX_LATENT_FILE_BYTES = 512 * 1024 * 1024


def file_sha256(path: str | Path) -> str:
    """Return ``sha256:<hex>`` for a file path."""

    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def sha256_json(payload: Any) -> str:
    """Return a stable SHA-256 digest over JSON-serializable content."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def validate_t2_audio_visual_data_config(data_cfg: Mapping[str, Any]) -> None:
    """Validate Sprint T2 data-scope fields.

    Raises:
        ValueError: if text/code or unsupported modalities are requested, or if
            one of the required audio/visual modalities is missing.
    """

    modalities = data_cfg.get("modalities", list(T2_REQUIRED_MODALITIES))
    if not isinstance(modalities, list):
        raise ValueError("T2 data.modalities must be a list")
    normalized = [str(modality) for modality in modalities]
    unsupported = [modality for modality in normalized if modality not in T2_REQUIRED_MODALITIES]
    if unsupported:
        raise ValueError(f"T2 supports only audio and visual_video modalities; unsupported: {unsupported}")
    missing = [modality for modality in T2_REQUIRED_MODALITIES if modality not in normalized]
    if missing:
        raise ValueError(f"T2 requires both audio and visual_video modalities; missing: {missing}")


def _as_2d_float_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D latent array [T,D], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def pad_or_truncate_features(features: np.ndarray, *, length: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Pad/truncate ``FloatTensor[T,D]`` to a fixed temporal length.

    Returns:
        ``(padded_features, mask, valid_length)`` where ``mask`` is a
        ``BoolTensor[length]`` marking real positions.
    """

    array = _as_2d_float_array(features, name="features")
    if length <= 0:
        raise ValueError("length must be positive")
    valid_length = min(int(array.shape[0]), int(length))
    output = np.zeros((int(length), int(array.shape[1])), dtype=np.float32)
    mask = np.zeros((int(length),), dtype=bool)
    if valid_length:
        output[:valid_length] = array[:valid_length]
        mask[:valid_length] = True
    return output, mask, valid_length


def _load_feature_array(record: Mapping[str, Any], *, key: str, path_key: str) -> np.ndarray:
    if key in record:
        return _as_2d_float_array(record[key], name=key)
    if path_key not in record:
        raise ValueError(f"T2 record requires either {key!r} or {path_key!r}")
    path = Path(str(record[path_key]))
    if not path.exists():
        raise FileNotFoundError(f"Missing latent feature file: {path}")
    if path.stat().st_size > MAX_LATENT_FILE_BYTES:
        raise ValueError(f"Latent feature file is too large for T2 preparation: {path}")
    if path.suffix == ".npz":
        _validate_npz_member_sizes(path)
        loaded = np.load(path, allow_pickle=False)
        candidate_keys = [key, key.replace("_features", ""), "features", "latents"]
        for candidate in candidate_keys:
            if candidate in loaded:
                return _as_2d_float_array(loaded[candidate], name=f"{path}:{candidate}")
        raise ValueError(f"NPZ file {path} did not contain one of {candidate_keys}")
    return _as_2d_float_array(np.load(path, allow_pickle=False), name=str(path))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        loaded = json.loads(stripped)
        if not isinstance(loaded, dict):
            raise ValueError(f"Manifest line {line_no} in {path} is not an object")
        records.append(loaded)
    return records


def generated_audio_visual_records(
    *,
    sample_count: int,
    audio_length: int,
    visual_length: int,
    audio_feature_dim: int,
    visual_feature_dim: int,
    seed: int,
    dataset: str = "project_generated_t2_latent_fixture",
    license_note: str = "project-authored synthetic latent fixture for pipeline validation only",
    label_count: int = 4,
) -> list[dict[str, Any]]:
    """Create a deterministic paired audio/visual latent fixture.

    The generated fixture is useful only for local smoke tests.  It creates
    class-conditioned temporal patterns so tiny training and correspondence
    metrics can be exercised without downloading external media.
    """

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if label_count <= 0:
        raise ValueError("label_count must be positive")
    rng = np.random.default_rng(int(seed))
    records: list[dict[str, Any]] = []
    t_audio = np.linspace(0.0, 1.0, int(audio_length), endpoint=False, dtype=np.float32)
    t_visual = np.linspace(0.0, 1.0, int(visual_length), endpoint=False, dtype=np.float32)
    for index in range(int(sample_count)):
        label_id = int(index % label_count)
        frequency = float(label_id + 1)
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        audio_channels = []
        visual_channels = []
        for channel in range(int(audio_feature_dim)):
            amp = 0.6 + 0.1 * channel + 0.05 * label_id
            audio_channels.append(amp * np.sin(2.0 * math.pi * frequency * t_audio + phase + 0.13 * channel))
        for channel in range(int(visual_feature_dim)):
            amp = 0.55 + 0.08 * channel + 0.04 * label_id
            visual_channels.append(amp * np.cos(2.0 * math.pi * frequency * t_visual + phase + 0.11 * channel))
        audio = np.stack(audio_channels, axis=-1).astype(np.float32)
        visual = np.stack(visual_channels, axis=-1).astype(np.float32)
        audio += rng.normal(0.0, 0.015, size=audio.shape).astype(np.float32)
        visual += rng.normal(0.0, 0.015, size=visual.shape).astype(np.float32)
        records.append(
            {
                "sample_id": f"t2-generated-{index:06d}",
                "audio_features": audio.tolist(),
                "visual_features": visual.tolist(),
                "dataset": dataset,
                "license": license_note,
                "label": f"latent_pattern_{label_id}",
                "metadata": {"generated_fixture": True, "label_id": label_id, "source_tag": "observed"},
            }
        )
    return records


def records_from_source(source: Mapping[str, Any], *, default_seed: int) -> list[dict[str, Any]]:
    """Resolve one preparation source into in-memory latent records."""

    source_type = str(source.get("type", ""))
    if source_type == "inline_latent_records":
        records = source.get("records", [])
        if not isinstance(records, list):
            raise ValueError("inline_latent_records source requires a records list")
        return [dict(record) for record in records if isinstance(record, Mapping)]
    if source_type == "manifest_jsonl":
        return _read_jsonl(str(source.get("manifest_path")))
    if source_type == "generated_fixture":
        return generated_audio_visual_records(
            sample_count=int(source.get("sample_count", 16)),
            audio_length=int(source.get("audio_length", source.get("max_audio_length", 8))),
            visual_length=int(source.get("visual_length", source.get("max_visual_length", 8))),
            audio_feature_dim=int(source.get("audio_feature_dim", 4)),
            visual_feature_dim=int(source.get("visual_feature_dim", 5)),
            seed=int(source.get("seed", default_seed)),
            dataset=str(source.get("dataset", source.get("name", "project_generated_t2_latent_fixture"))),
            license_note=str(source.get("license", "project-authored synthetic latent fixture for pipeline validation only")),
            label_count=int(source.get("label_count", 4)),
        )
    raise ValueError(f"Unsupported T2 latent source type {source_type!r}")


def assign_splits(
    records: list[dict[str, Any]],
    *,
    split_policy: Mapping[str, Any],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Assign records to train/validation/test splits deterministically."""

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    explicit = [record for record in records if record.get("split") in splits]
    implicit = [record for record in records if record.get("split") not in splits]
    for record in explicit:
        splits[str(record["split"])].append(record)
    if implicit:
        rng = np.random.default_rng(int(split_policy.get("seed", seed)))
        indices = np.arange(len(implicit))
        rng.shuffle(indices)
        train_ratio = float(split_policy.get("train", 0.8))
        val_ratio = float(split_policy.get("validation", 0.1))
        if train_ratio < 0.0 or val_ratio < 0.0 or train_ratio + val_ratio > 1.0:
            raise ValueError("split_policy train/validation ratios must be non-negative and sum to <= 1")
        train_end = int(round(len(indices) * train_ratio))
        val_end = train_end + int(round(len(indices) * val_ratio))
        assignments = {
            "train": indices[:train_end],
            "validation": indices[train_end:val_end],
            "test": indices[val_end:],
        }
        for split_name, split_indices in assignments.items():
            for index in split_indices:
                record = dict(implicit[int(index)])
                record["split"] = split_name
                splits[split_name].append(record)
    return splits


@dataclass(frozen=True)
class T2ManifestRecord:
    """One prepared T2 latent sample manifest row."""

    sample_id: str
    split: str
    latent_path: str
    audio_shape: list[int]
    visual_shape: list[int]
    audio_valid_length: int
    visual_valid_length: int
    dataset: str
    license: str
    label: str | None
    sha256: str
    metadata: dict[str, Any]


def standardize_record(
    record: Mapping[str, Any],
    *,
    output_path: Path,
    split: str,
    audio_length: int,
    visual_length: int,
) -> T2ManifestRecord:
    """Write one standardized NPZ latent file and return its manifest row."""

    sample_id = str(record.get("sample_id", output_path.stem))
    audio_raw = _load_feature_array(record, key="audio_features", path_key="audio_path")
    visual_raw = _load_feature_array(record, key="visual_features", path_key="visual_path")
    audio, audio_mask, audio_valid = pad_or_truncate_features(audio_raw, length=audio_length)
    visual, visual_mask, visual_valid = pad_or_truncate_features(visual_raw, length=visual_length)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp.npz")
    np.savez_compressed(
        tmp,
        audio_features=audio,
        visual_features=visual,
        audio_mask=audio_mask,
        visual_mask=visual_mask,
        sample_id=np.asarray(sample_id),
    )
    tmp.replace(output_path)
    metadata = dict(record.get("metadata", {})) if isinstance(record.get("metadata", {}), Mapping) else {}
    metadata.setdefault("source_tag", "observed")
    if metadata["source_tag"] not in SOURCE_TAGS:
        raise ValueError(f"Unsupported source_tag {metadata['source_tag']!r}")
    return T2ManifestRecord(
        sample_id=sample_id,
        split=split,
        latent_path=str(output_path),
        audio_shape=[int(audio.shape[0]), int(audio.shape[1])],
        visual_shape=[int(visual.shape[0]), int(visual.shape[1])],
        audio_valid_length=int(audio_valid),
        visual_valid_length=int(visual_valid),
        dataset=str(record.get("dataset", record.get("source", "unknown"))),
        license=str(record.get("license", record.get("license_notes", "unknown"))),
        label=None if record.get("label") is None else str(record.get("label")),
        sha256=file_sha256(output_path),
        metadata=metadata,
    )


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write JSONL records atomically."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")
    tmp.replace(output_path)
    return output_path


@dataclass(frozen=True)
class T2PreparedLatentDataset:
    """Prepared audio/visual latent split loader.

    Shape contract for :meth:`batch`:
        Returns ``audio_features: FloatTensor[B,T_audio,A]``,
        ``visual_features: FloatTensor[B,T_visual,V]``, masks with shape
        ``BoolTensor[B,T_*]``, targets with matching feature shapes, and loss
        masks selecting future/missing positions.
    """

    prepared_dir: Path
    split: str
    dataset_card: Mapping[str, Any]
    records: list[Mapping[str, Any]]

    @classmethod
    def load(cls, prepared_dir: str | Path, *, split: str) -> "T2PreparedLatentDataset":
        root = _safe_prepared_dir(prepared_dir)
        card_path = root / "dataset_card.json"
        if not card_path.exists():
            raise FileNotFoundError(f"Missing T2 dataset_card.json: {card_path}")
        card = json.loads(card_path.read_text(encoding="utf-8"))
        _validate_dataset_card_manifest_hash(root, card)
        manifest_path = root / "manifests" / f"{split}.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing T2 split manifest: {manifest_path}")
        records = _read_jsonl(manifest_path)
        if not records:
            raise ValueError(f"T2 split {split!r} has no records")
        return cls(prepared_dir=root, split=split, dataset_card=card, records=records)

    @property
    def sample_count(self) -> int:
        return len(self.records)

    def _load_record_arrays(self, record: Mapping[str, Any]) -> dict[str, np.ndarray]:
        path = _resolve_manifest_latent_path(self.prepared_dir, record)
        expected_sha = record.get("sha256")
        if expected_sha is None:
            raise ValueError(f"Manifest row for {record.get('sample_id')} is missing required sha256")
        actual_sha = file_sha256(path)
        if str(expected_sha) != actual_sha:
            raise ValueError(f"Latent file hash mismatch for {path}: expected {expected_sha}, got {actual_sha}")
        if path.stat().st_size > MAX_LATENT_FILE_BYTES:
            raise ValueError(f"Latent file is too large for T2 loading: {path}")
        if path.suffix == ".npz":
            _validate_npz_member_sizes(path)
        loaded = np.load(path, allow_pickle=False)
        audio = np.asarray(loaded["audio_features"], dtype=np.float32)
        visual = np.asarray(loaded["visual_features"], dtype=np.float32)
        audio_mask = np.asarray(loaded["audio_mask"], dtype=bool)
        visual_mask = np.asarray(loaded["visual_mask"], dtype=bool)
        if audio.ndim != 2 or visual.ndim != 2:
            raise ValueError(f"T2 latent file arrays must be 2D [T,D], got audio={audio.shape}, visual={visual.shape}")
        if audio_mask.shape != audio.shape[:1] or visual_mask.shape != visual.shape[:1]:
            raise ValueError(f"T2 latent masks must match time dimensions for {path}")
        if record.get("audio_shape") is not None and list(record["audio_shape"]) != [int(audio.shape[0]), int(audio.shape[1])]:
            raise ValueError(f"Audio shape mismatch for {path}: manifest {record['audio_shape']} vs file {list(audio.shape)}")
        if record.get("visual_shape") is not None and list(record["visual_shape"]) != [int(visual.shape[0]), int(visual.shape[1])]:
            raise ValueError(f"Visual shape mismatch for {path}: manifest {record['visual_shape']} vs file {list(visual.shape)}")
        return {
            "audio_features": audio,
            "visual_features": visual,
            "audio_mask": audio_mask,
            "visual_mask": visual_mask,
        }

    def batch(
        self,
        *,
        batch_size: int,
        seed: int,
        step: int = 0,
        context_fraction: float = 0.5,
        missing_span_fraction: float = 0.0,
        shuffle_visual: bool = False,
        sequential: bool = False,
    ) -> dict[str, Any]:
        """Return a deterministic T2 latent training/eval batch."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 < context_fraction <= 1.0:
            raise ValueError("context_fraction must be in (0,1]")
        rng = np.random.default_rng(int(seed) + int(step))
        if sequential:
            indices = (np.arange(int(batch_size)) + int(step) * int(batch_size)) % len(self.records)
        else:
            indices = rng.integers(0, len(self.records), size=int(batch_size), endpoint=False)
        selected = [self.records[int(index)] for index in indices]
        arrays = [self._load_record_arrays(record) for record in selected]
        audio_target = np.stack([item["audio_features"] for item in arrays]).astype(np.float32)
        visual_target = np.stack([item["visual_features"] for item in arrays]).astype(np.float32)
        audio_valid_mask = np.stack([item["audio_mask"] for item in arrays]).astype(bool)
        visual_valid_mask = np.stack([item["visual_mask"] for item in arrays]).astype(bool)

        audio_input = audio_target.copy()
        visual_input = visual_target.copy()
        audio_loss_mask = np.zeros(audio_valid_mask.shape, dtype=bool)
        visual_loss_mask = np.zeros(visual_valid_mask.shape, dtype=bool)
        audio_observed_mask = audio_valid_mask.copy()
        visual_observed_mask = visual_valid_mask.copy()

        def apply_masks(features: np.ndarray, valid_mask: np.ndarray, observed_mask: np.ndarray, loss_mask: np.ndarray) -> None:
            for row in range(features.shape[0]):
                valid_positions = np.flatnonzero(valid_mask[row])
                if valid_positions.size == 0:
                    continue
                cutoff_index = max(1, int(math.ceil(valid_positions.size * float(context_fraction))))
                future_positions = valid_positions[cutoff_index:]
                if future_positions.size:
                    observed_mask[row, future_positions] = False
                    loss_mask[row, future_positions] = True
                    features[row, future_positions, :] = 0.0
                if missing_span_fraction > 0.0 and valid_positions.size >= 4:
                    span = max(1, int(round(valid_positions.size * missing_span_fraction)))
                    max_start = max(1, cutoff_index - span)
                    start = int(rng.integers(0, max_start, endpoint=False))
                    span_positions = valid_positions[start : start + span]
                    observed_mask[row, span_positions] = False
                    loss_mask[row, span_positions] = True
                    features[row, span_positions, :] = 0.0
                if not np.any(loss_mask[row]):
                    # Keep at least one target position for loss accounting.
                    last = int(valid_positions[-1])
                    observed_mask[row, last] = False
                    loss_mask[row, last] = True
                    features[row, last, :] = 0.0

        apply_masks(audio_input, audio_valid_mask, audio_observed_mask, audio_loss_mask)
        apply_masks(visual_input, visual_valid_mask, visual_observed_mask, visual_loss_mask)

        visual_targets_for_alignment = visual_target
        visual_pair_indices = np.arange(int(batch_size))
        if shuffle_visual and batch_size > 1:
            visual_pair_indices = np.roll(visual_pair_indices, 1)
            visual_input = visual_input[visual_pair_indices]
            visual_target = visual_target[visual_pair_indices]
            visual_valid_mask = visual_valid_mask[visual_pair_indices]
            visual_observed_mask = visual_observed_mask[visual_pair_indices]
            visual_loss_mask = visual_loss_mask[visual_pair_indices]

        return {
            "audio_features": audio_input,
            "visual_features": visual_input,
            "audio_targets": audio_target,
            "visual_targets": visual_target,
            "visual_targets_unshuffled": visual_targets_for_alignment,
            "audio_mask": audio_observed_mask,
            "visual_mask": visual_observed_mask,
            "audio_valid_mask": audio_valid_mask,
            "visual_valid_mask": visual_valid_mask,
            "audio_loss_mask": audio_loss_mask,
            "visual_loss_mask": visual_loss_mask,
            "sample_ids": [str(record["sample_id"]) for record in selected],
            "labels": [record.get("label") for record in selected],
            "visual_pair_indices": visual_pair_indices.tolist(),
            "metadata": {
                "split": self.split,
                "source_tags": ["observed", "predicted", "reconstructed"],
                "modalities": list(T2_REQUIRED_MODALITIES),
                "modality_ids": {"audio": MODALITY_IDS["audio"], "visual_video": MODALITY_IDS["visual_video"]},
            },
        }


def t2_batch_contract_summary(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, test-friendly summary of a T2 batch contract."""

    audio = np.asarray(batch["audio_features"])
    visual = np.asarray(batch["visual_features"])
    audio_mask = np.asarray(batch["audio_mask"], dtype=bool)
    visual_mask = np.asarray(batch["visual_mask"], dtype=bool)
    if audio.ndim != 3 or visual.ndim != 3:
        raise ValueError("T2 batch features must have shape [B,T,D]")
    if audio_mask.shape != audio.shape[:2] or visual_mask.shape != visual.shape[:2]:
        raise ValueError("T2 batch masks must match feature [B,T] shapes")
    return {
        "audio_shape": list(audio.shape),
        "visual_shape": list(visual.shape),
        "audio_mask_shape": list(audio_mask.shape),
        "visual_mask_shape": list(visual_mask.shape),
        "modality_ids": {"audio": MODALITY_IDS["audio"], "visual_video": MODALITY_IDS["visual_video"]},
        "source_tags_valid": all(tag in SOURCE_TAGS for tag in batch.get("metadata", {}).get("source_tags", [])),
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_prepared_dir(prepared_dir: str | Path) -> Path:
    """Resolve a trusted prepared T2 corpus directory under the artifact root."""

    raw = Path(prepared_dir)
    if raw.is_absolute():
        raise ValueError("T2 prepared_corpus_dir must be relative")
    repo_root = Path.cwd().resolve()
    root = (repo_root / raw).resolve()
    allowed_root = (repo_root / T2_PREPARED_ROOT).resolve()
    if not _is_relative_to(root, allowed_root):
        raise ValueError(f"T2 prepared_corpus_dir must be under {T2_PREPARED_ROOT}")
    if root.exists() and root.is_symlink():
        raise ValueError("T2 prepared_corpus_dir must not be a symlink")
    return raw


def _resolve_manifest_latent_path(prepared_dir: Path, record: Mapping[str, Any]) -> Path:
    raw = Path(str(record["latent_path"]))
    if raw.is_absolute():
        raise ValueError("T2 manifest latent_path must be relative")
    repo_root = Path.cwd().resolve()
    prepared_root = (repo_root / prepared_dir).resolve()
    candidate = (repo_root / raw).resolve()
    if not _is_relative_to(candidate, prepared_root):
        alternate = (prepared_root / raw).resolve()
        if _is_relative_to(alternate, prepared_root):
            candidate = alternate
        else:
            raise ValueError(f"T2 manifest latent_path must stay under prepared corpus: {raw}")
    if candidate.exists() and candidate.is_symlink():
        raise ValueError(f"T2 manifest latent_path must not be a symlink: {raw}")
    return candidate


def _validate_dataset_card_manifest_hash(prepared_dir: Path, card: Mapping[str, Any]) -> None:
    """Recompute and validate the dataset-card manifest hash."""

    expected = card.get("manifest_sha256")
    if not expected:
        raise ValueError("T2 dataset_card.json is missing manifest_sha256")
    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "validation", "test"):
        manifest_path = prepared_dir / "manifests" / f"{split_name}.jsonl"
        splits[split_name] = _read_jsonl(manifest_path) if manifest_path.exists() else []
    payload = {
        "splits": splits,
        "config_hash": card.get("config_hash"),
        "schema": card.get("sample_schema_version", DEFAULT_T2_SCHEMA_VERSION),
    }
    actual = sha256_json(payload)
    if str(expected) != actual:
        raise ValueError(f"T2 manifest hash mismatch: expected {expected}, got {actual}")


def _validate_npz_member_sizes(path: Path) -> None:
    """Reject compressed NPZ files whose declared uncompressed arrays are huge."""

    with zipfile.ZipFile(path) as archive:
        total = sum(int(info.file_size) for info in archive.infolist())
    if total > MAX_LATENT_FILE_BYTES:
        raise ValueError(f"T2 NPZ uncompressed size is too large: {path}")


__all__ = [
    "DEFAULT_T2_SCHEMA_VERSION",
    "T2PreparedLatentDataset",
    "T2_REQUIRED_MODALITIES",
    "assign_splits",
    "file_sha256",
    "generated_audio_visual_records",
    "pad_or_truncate_features",
    "records_from_source",
    "sha256_json",
    "standardize_record",
    "t2_batch_contract_summary",
    "validate_t2_audio_visual_data_config",
    "write_jsonl",
]
