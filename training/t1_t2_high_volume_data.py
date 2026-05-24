"""Plan and safely bootstrap high-volume Sprint T1/T2 data under ``/data``.

This module is intentionally conservative.  It creates a reproducible dataset
acquisition/preparation plan, directory layout, source manifest, storage budget
report, and blocker list for high-volume data work.  It does **not** silently
download external corpora: network acquisition must be explicitly confirmed and
dataset-specific terms/authentication must be recorded first.

Sprint boundaries enforced here:

* T1 sources are text/code only and must preserve GPT-2-BPE-compatible split
  metadata for later baseline-vs-SLWM comparisons.
* T2 sources are audio/visual/video latent candidates only; captions may be
  carried as metadata, but not as text-generation targets.

The output files are small metadata artifacts suitable for ``/data``.  Prepared
training corpora still flow through the existing T1/T2 preparation runners once
source data or precomputed latents are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from utils.config import config_hash, load_config, write_config


STAGE_ORDER: tuple[str, ...] = ("smoke", "pilot", "quality", "high_volume")
PLAN_SUBDIR = Path("manifests/t1_t2_high_volume")
T2_REQUIRED_MODALITIES: tuple[str, str] = ("audio", "visual_video")
PLAN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_t1_text_only_data_config(data_cfg: Mapping[str, Any]) -> None:
    """Dependency-light T1 scope guard mirroring ``data.text_code`` rules."""

    dataset_mix = data_cfg.get("dataset_mix", {}) if isinstance(data_cfg.get("dataset_mix", {}), Mapping) else {}
    for forbidden in ("audio", "visual_video"):
        value = dataset_mix.get(forbidden)
        if value not in (None, 0, 0.0, "", "none", "None"):
            raise ValueError(f"Sprint T1 must not include {forbidden} data; got dataset_mix[{forbidden!r}]={value!r}")
    modalities = data_cfg.get("modalities", ["text_code"])
    if any(str(modality) != "text_code" for modality in modalities):
        raise ValueError(f"Sprint T1 modalities must be ['text_code']; got {modalities!r}")


def _validate_t2_audio_visual_data_config(data_cfg: Mapping[str, Any]) -> None:
    """Dependency-light T2 scope guard mirroring ``data.audio_visual_latents``."""

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


@dataclass(frozen=True)
class PlannedSource:
    """One planned source in the high-volume data ladder.

    Shape / contract note:
        The source is metadata-only.  It does not contain tensor data.  T1
        downstream preparation converts selected text/code rows into
        ``IntTensor[T]`` token streams.  T2 downstream preparation converts
        selected media/features into ``FloatTensor[T,D]`` audio/visual latents.
    """

    sprint: str
    name: str
    dataset: str
    stage: str
    modality: str
    category: str
    enabled: bool
    estimated_bytes: int
    target_subdir: str
    license_notes: str
    source_url: str | None
    version_or_snapshot: str | None
    requires_auth: bool
    requires_terms: bool
    accepted_terms: bool
    required_tools: tuple[str, ...]
    download_kind: str
    blockers: tuple[str, ...]
    notes: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable source manifest row."""

        return {
            "sprint": self.sprint,
            "name": self.name,
            "dataset": self.dataset,
            "stage": self.stage,
            "modality": self.modality,
            "category": self.category,
            "enabled": self.enabled,
            "estimated_bytes": self.estimated_bytes,
            "target_subdir": self.target_subdir,
            "license_notes": self.license_notes,
            "source_url": self.source_url,
            "version_or_snapshot": self.version_or_snapshot,
            "requires_auth": self.requires_auth,
            "requires_terms": self.requires_terms,
            "accepted_terms": self.accepted_terms,
            "required_tools": list(self.required_tools),
            "download_kind": self.download_kind,
            "blockers": list(self.blockers),
            "notes": self.notes,
        }


def _write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _as_list(value: Any, *, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _stage_rank(stage: str) -> int:
    normalized = str(stage)
    if normalized not in STAGE_ORDER:
        raise ValueError(f"Unknown data stage {stage!r}; expected one of {STAGE_ORDER}")
    return STAGE_ORDER.index(normalized)


def _source_is_selected(source: Mapping[str, Any], *, active_stage: str) -> bool:
    if not bool(source.get("enabled", True)):
        return False
    source_stage = str(source.get("stage", "pilot"))
    return _stage_rank(source_stage) <= _stage_rank(active_stage)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_data_root(raw_root: str | Path, *, allow_non_data_root: bool = False) -> Path:
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        raise ValueError("storage.data_root must be an absolute path such as /data/slwm")
    if root.exists() and root.is_symlink():
        raise ValueError("storage.data_root must not be a symlink")
    resolved = root.resolve(strict=False)
    data_parent = Path("/data").resolve(strict=False)
    if not allow_non_data_root and not _is_relative_to(resolved, data_parent):
        raise ValueError("storage.data_root must be under /data unless storage.allow_non_data_root=true is set for tests")
    return resolved


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _disk_report(data_root: Path) -> dict[str, int | None | str]:
    parent = _nearest_existing_parent(data_root)
    try:
        usage = shutil.disk_usage(parent)
    except OSError:
        return {"path_checked": str(parent), "total_bytes": None, "used_bytes": None, "free_bytes": None}
    return {
        "path_checked": str(parent),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _ensure_writable_data_parent(data_root: Path) -> None:
    parent = _nearest_existing_parent(data_root)
    if not os.access(parent, os.W_OK):
        raise PermissionError(
            f"Cannot write high-volume data manifests under {data_root}; nearest existing parent {parent} is not writable. "
            "Choose a writable /data subdirectory or create/chown it before running the planner."
        )


def _target_subdir_for_source(sprint: str, source: Mapping[str, Any]) -> str:
    explicit = source.get("target_subdir")
    if explicit:
        return str(explicit)
    if sprint == "T1":
        category = str(source.get("category", "text_english"))
        return f"t1_text_code/raw/{category}/{source.get('name', 'unnamed_source')}"
    modality = str(source.get("modality", "audio_visual_pair"))
    if modality == "audio":
        return f"t2_signal_latents/raw_cache/audio/{source.get('name', 'unnamed_source')}"
    if modality == "visual_video":
        return f"t2_signal_latents/raw_cache/visual_video/{source.get('name', 'unnamed_source')}"
    return f"t2_signal_latents/raw_cache/audio_visual_pair/{source.get('name', 'unnamed_source')}"


def _resolve_under_data_root(data_root: Path, relative_subdir: str) -> Path:
    candidate = Path(relative_subdir)
    if candidate.is_absolute():
        raise ValueError(f"source target_subdir must be relative, got {relative_subdir!r}")
    resolved = (data_root / candidate).resolve(strict=False)
    if not _is_relative_to(resolved, data_root):
        raise ValueError(f"source target_subdir escapes data_root: {relative_subdir!r}")
    return resolved


def _validate_source_target_subdirs(data_root: Path, planned_sources: Iterable[PlannedSource]) -> None:
    for source in planned_sources:
        _resolve_under_data_root(data_root, source.target_subdir)


def _resolve_plan_dir(data_root: Path, plan_id: str) -> Path:
    if not PLAN_ID_PATTERN.fullmatch(plan_id):
        raise ValueError("storage.plan_id must contain only letters, numbers, '.', '_', and '-' and must not contain path separators")
    manifest_root = (data_root / PLAN_SUBDIR).resolve(strict=False)
    resolved = (manifest_root / plan_id).resolve(strict=False)
    if not _is_relative_to(resolved, manifest_root):
        raise ValueError(f"storage.plan_id escapes manifest root: {plan_id!r}")
    return resolved


def _source_blockers(source: Mapping[str, Any], *, dry_run: bool, confirm_download: bool) -> list[str]:
    blockers: list[str] = []
    if bool(source.get("requires_auth", False)):
        auth_hint = str(source.get("auth_hint", "record required credentials/token in the local run environment"))
        blockers.append(f"requires_auth: {auth_hint}")
    if bool(source.get("requires_terms", False)) and not bool(source.get("accepted_terms", False)):
        blockers.append("requires_terms_acceptance: set accepted_terms=true only after terms/license review")
    for tool in _as_list(source.get("required_tools", []), name="source.required_tools"):
        blockers.append(f"requires_tool:{tool}")
    if not dry_run and not confirm_download:
        blockers.append("requires_cli_confirm_download")
    if not dry_run and str(source.get("download_kind", "manual_or_external")) != "local_fixture":
        blockers.append("network_download_not_implemented_by_this_safety_planner")
    return blockers


def _planned_source_from_config(
    sprint: str,
    source: Mapping[str, Any],
    *,
    active_stage: str,
    dry_run: bool,
    confirm_download: bool,
) -> PlannedSource | None:
    if not _source_is_selected(source, active_stage=active_stage):
        return None
    name = str(source.get("name", source.get("dataset", "unnamed_source")))
    dataset = str(source.get("dataset", name))
    stage = str(source.get("stage", "pilot"))
    modality = str(source.get("modality", "text_code" if sprint == "T1" else "audio_visual_pair"))
    blockers = _source_blockers(source, dry_run=dry_run, confirm_download=confirm_download)
    return PlannedSource(
        sprint=sprint,
        name=name,
        dataset=dataset,
        stage=stage,
        modality=modality,
        category=str(source.get("category", modality)),
        enabled=bool(source.get("enabled", True)),
        estimated_bytes=int(source.get("estimated_bytes", 0)),
        target_subdir=_target_subdir_for_source(sprint, source),
        license_notes=str(source.get("license_notes", "license/terms must be recorded before evidence-bearing use")),
        source_url=None if source.get("source_url") is None else str(source.get("source_url")),
        version_or_snapshot=None if source.get("version_or_snapshot") is None else str(source.get("version_or_snapshot")),
        requires_auth=bool(source.get("requires_auth", False)),
        requires_terms=bool(source.get("requires_terms", False)),
        accepted_terms=bool(source.get("accepted_terms", False)),
        required_tools=tuple(str(tool) for tool in _as_list(source.get("required_tools", []), name="source.required_tools")),
        download_kind=str(source.get("download_kind", "manual_or_external")),
        blockers=tuple(blockers),
        notes=str(source.get("notes", "")),
    )


def _validate_t1_scope(t1_cfg: Mapping[str, Any]) -> None:
    data_cfg = {
        "modalities": t1_cfg.get("modalities", ["text_code"]),
        "dataset_mix": t1_cfg.get("dataset_mix", {"text_code": 1.0, "audio": None, "visual_video": None}),
    }
    _validate_t1_text_only_data_config(data_cfg)
    for source in _as_list(t1_cfg.get("sources", []), name="t1.sources"):
        if not isinstance(source, Mapping):
            raise ValueError("t1.sources entries must be objects")
        modality = str(source.get("modality", "text_code"))
        if modality != "text_code":
            raise ValueError(f"T1 high-volume sources must be text_code only; got {modality!r}")


def _validate_t2_scope(t2_cfg: Mapping[str, Any]) -> None:
    data_cfg = {
        "modalities": t2_cfg.get("modalities", ["audio", "visual_video"]),
        "dataset_mix": t2_cfg.get("dataset_mix", {"text_code": None, "audio": 0.5, "visual_video": 0.5}),
    }
    _validate_t2_audio_visual_data_config(data_cfg)
    for source in _as_list(t2_cfg.get("sources", []), name="t2.sources"):
        if not isinstance(source, Mapping):
            raise ValueError("t2.sources entries must be objects")
        modality = str(source.get("modality", "audio_visual_pair"))
        if modality not in {"audio", "visual_video", "audio_visual_pair"}:
            raise ValueError(f"T2 sources must be audio/visual/video only; got {modality!r}")
        caption_use = str(source.get("caption_use", "none"))
        if caption_use not in {"none", "metadata_only"}:
            raise ValueError("T2 captions/text labels may be metadata_only only; no text-generation targets in T2")
        if bool(source.get("text_generation_objective", False)):
            raise ValueError("T2 must not define text_generation_objective=true")


def _directory_layout() -> dict[str, list[str]]:
    return {
        "manifests": [
            "manifests/t1_t2_high_volume/<plan_id>/plan.json",
            "manifests/t1_t2_high_volume/<plan_id>/sources.jsonl",
            "manifests/t1_t2_high_volume/<plan_id>/blockers.json",
            "manifests/t1_t2_high_volume/<plan_id>/directory_layout.json",
        ],
        "t1_text_code": [
            "t1_text_code/raw/",
            "t1_text_code/normalized_jsonl/",
            "t1_text_code/tokenized_gpt2/",
            "t1_text_code/dedup_index/",
            "t1_text_code/leakage_reports/",
        ],
        "t2_signal_latents": [
            "t2_signal_latents/raw_cache/",
            "t2_signal_latents/manifests/",
            "t2_signal_latents/audio_latents/",
            "t2_signal_latents/visual_latents/",
            "t2_signal_latents/av_pair_index/",
            "t2_signal_latents/feature_extraction_logs/",
        ],
    }


def _bootstrap_directories(data_root: Path, plan_dir: Path, planned_sources: list[PlannedSource]) -> list[str]:
    created: list[str] = []
    base_dirs = [
        plan_dir,
        data_root / "t1_text_code/raw",
        data_root / "t1_text_code/normalized_jsonl",
        data_root / "t1_text_code/tokenized_gpt2",
        data_root / "t1_text_code/dedup_index",
        data_root / "t1_text_code/leakage_reports",
        data_root / "t2_signal_latents/raw_cache",
        data_root / "t2_signal_latents/manifests",
        data_root / "t2_signal_latents/audio_latents",
        data_root / "t2_signal_latents/visual_latents",
        data_root / "t2_signal_latents/av_pair_index",
        data_root / "t2_signal_latents/feature_extraction_logs",
    ]
    for source in planned_sources:
        base_dirs.append(_resolve_under_data_root(data_root, source.target_subdir))
    for directory in base_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        created.append(str(directory))
    return created


def _quality_gates(config: Mapping[str, Any]) -> dict[str, Any]:
    configured = _as_mapping(config.get("quality_checks", {}), name="quality_checks")
    return {
        "required_before_training_claims": [
            "source_manifest_with_license_and_version",
            "deterministic_document_or_media_split_hashes",
            "raw_and_processed_byte_counts",
            "exact_duplicate_report",
            "near_duplicate_or_similarity_report",
            "eval_contamination_report_for_text_code",
            "secrets_or_pii_filter_report_for_code_and_web_text",
            "media_availability_and_corruption_report_for_t2",
            "feature_extractor_version_and_shape_report_for_t2",
            "entropy_or_diversity_summary_by_source_and_split",
        ],
        "configured": configured,
        "status": "planned_not_completed_by_this_script",
    }


def _acceptance_gates() -> dict[str, Any]:
    return {
        "this_script": [
            "dry_run_plan_written",
            "storage_budget_checked",
            "t1_text_code_scope_enforced",
            "t2_audio_visual_latent_scope_enforced",
            "blockers_reported_before_download",
            "dataset_cards_and_split_hashes_required_for_later_training",
        ],
        "not_satisfied_by_this_script": [
            "external_high_volume_download_complete",
            "dedup_and_leakage_filters_complete",
            "T1_or_T2_training_success",
            "model_quality_or_grounding_claims",
        ],
        "expected_artifacts_before_training": [
            "source_manifest_with_versions_and_licenses.jsonl",
            "split_hashes.json",
            "dedup_and_leakage_report.json",
            "t1_tokenizer_metadata.json",
            "t1_token_counts_by_split.json",
            "t2_feature_extractor_report.json",
            "t2_media_availability_report.json",
            "t2_latent_shape_report.json",
        ],
    }


def plan_t1_t2_high_volume_data(
    config_path: str | Path,
    *,
    data_root_override: str | Path | None = None,
    stage_override: str | None = None,
    dry_run_override: bool | None = None,
    confirm_download: bool = False,
) -> dict[str, Any]:
    """Create a high-volume T1/T2 data plan under the configured data root.

    Args:
        config_path: JSON config describing storage, source ladder, and checks.
        data_root_override: Optional absolute path replacing ``storage.data_root``.
        stage_override: Optional active stage from ``smoke/pilot/quality/high_volume``.
        dry_run_override: Optional dry-run override.  ``None`` uses config.
        confirm_download: Required for any future non-dry-run network mode.

    Returns:
        A JSON-serializable summary including written manifest paths and blockers.
    """

    start = time.perf_counter()
    path = Path(config_path)
    config = load_config(path)
    storage_cfg = _as_mapping(config.get("storage", {}), name="storage")
    active_stage = str(stage_override or config.get("active_stage", storage_cfg.get("active_stage", "pilot")))
    _stage_rank(active_stage)
    dry_run = bool(storage_cfg.get("dry_run", True)) if dry_run_override is None else bool(dry_run_override)
    data_root = _resolve_data_root(
        data_root_override or storage_cfg.get("data_root", "/data/slwm"),
        allow_non_data_root=bool(storage_cfg.get("allow_non_data_root", False)),
    )
    plan_id = str(storage_cfg.get("plan_id", f"t1_t2_{active_stage}_{date.today().isoformat()}"))
    plan_dir = _resolve_plan_dir(data_root, plan_id)

    if not dry_run and not confirm_download:
        raise ValueError("Non-dry-run high-volume data mode requires --confirm-download")
    if not dry_run and not bool(storage_cfg.get("allow_network", False)):
        raise ValueError("Non-dry-run mode also requires storage.allow_network=true in the config")

    t1_cfg = _as_mapping(config.get("t1", {}), name="t1")
    t2_cfg = _as_mapping(config.get("t2", {}), name="t2")
    if bool(t1_cfg.get("enabled", True)):
        _validate_t1_scope(t1_cfg)
    if bool(t2_cfg.get("enabled", True)):
        _validate_t2_scope(t2_cfg)

    planned_sources: list[PlannedSource] = []
    for sprint, sprint_cfg in (("T1", t1_cfg), ("T2", t2_cfg)):
        if not bool(sprint_cfg.get("enabled", True)):
            continue
        for raw_source in _as_list(sprint_cfg.get("sources", []), name=f"{sprint.lower()}.sources"):
            if not isinstance(raw_source, Mapping):
                raise ValueError(f"{sprint.lower()}.sources entries must be objects")
            planned = _planned_source_from_config(
                sprint,
                raw_source,
                active_stage=active_stage,
                dry_run=dry_run,
                confirm_download=confirm_download,
            )
            if planned is not None:
                planned_sources.append(planned)

    if not planned_sources:
        raise ValueError("No enabled T1/T2 sources selected for the requested stage")

    source_rows = [source.as_dict() for source in planned_sources]
    total_estimated_bytes = int(sum(source.estimated_bytes for source in planned_sources))
    max_total_bytes = int(storage_cfg.get("max_total_bytes", 20_000_000_000_000))
    budget_status = "within_budget" if total_estimated_bytes <= max_total_bytes else "over_budget"
    blockers = [
        {"source": source.name, "sprint": source.sprint, "blocker": blocker}
        for source in planned_sources
        for blocker in source.blockers
    ]
    if not dry_run and blockers:
        blocker_text = "; ".join(f"{row['source']}:{row['blocker']}" for row in blockers)
        raise ValueError(f"Cannot run non-dry-run data acquisition while blockers remain: {blocker_text}")
    _validate_source_target_subdirs(data_root, planned_sources)

    disk_report = _disk_report(data_root)
    free_bytes = disk_report.get("free_bytes")
    if isinstance(free_bytes, int):
        disk_free_status = "fits_current_free_space" if total_estimated_bytes <= free_bytes else "exceeds_current_free_space"
    else:
        disk_free_status = "unknown_current_free_space"
    layout = _directory_layout()
    created_dirs: list[str] = []
    if bool(storage_cfg.get("bootstrap_directories", True)):
        _ensure_writable_data_parent(data_root)
        created_dirs = _bootstrap_directories(data_root, plan_dir, planned_sources)
    else:
        _ensure_writable_data_parent(plan_dir)
        plan_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "status": "planned" if dry_run else "blocked_or_manual_execution_required",
        "dry_run": dry_run,
        "active_stage": active_stage,
        "claim_scope": "Dataset acquisition/preparation planning only; not T1/T2 model-quality evidence.",
        "config_path": str(path),
        "config_hash": config_hash(config),
        "data_root": str(data_root),
        "plan_id": plan_id,
        "plan_dir": str(plan_dir),
        "sprints": ["T1", "T2"],
        "stage_order": list(STAGE_ORDER),
        "estimated_bytes": total_estimated_bytes,
        "max_total_bytes": max_total_bytes,
        "storage_budget_status": budget_status,
        "disk_free_status": disk_free_status,
        "disk_report": disk_report,
        "selected_source_count": len(planned_sources),
        "selected_sources_by_sprint": {
            "T1": [source.name for source in planned_sources if source.sprint == "T1"],
            "T2": [source.name for source in planned_sources if source.sprint == "T2"],
        },
        "blockers": blockers,
        "directory_layout": layout,
        "created_directories": created_dirs,
        "quality_gates": _quality_gates(config),
        "acceptance_gates": _acceptance_gates(),
        "runtime_seconds": time.perf_counter() - start,
    }

    plan_path = _write_json(summary, plan_dir / "plan.json")
    sources_path = _write_jsonl(source_rows, plan_dir / "sources.jsonl")
    blockers_path = _write_json({"blockers": blockers}, plan_dir / "blockers.json")
    layout_path = _write_json(layout, plan_dir / "directory_layout.json")
    copied_config_path = write_config(config, plan_dir / "source_config.json")
    deterministic_plan_payload = {
        "active_stage": active_stage,
        "config_hash": config_hash(config),
        "selected_sources": source_rows,
        "estimated_bytes": total_estimated_bytes,
        "max_total_bytes": max_total_bytes,
        "quality_gates": summary["quality_gates"],
        "acceptance_gates": summary["acceptance_gates"],
    }
    summary.update(
        {
            "plan_path": str(plan_path),
            "sources_path": str(sources_path),
            "blockers_path": str(blockers_path),
            "directory_layout_path": str(layout_path),
            "source_config_copy_path": str(copied_config_path),
            "sources_sha256": _sha256_path(sources_path),
            "source_plan_sha256": _sha256_json(deterministic_plan_payload),
        }
    )
    _write_json(summary, plan_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan high-volume SLWM T1/T2 data under /data. Dry-run writes small manifests/directories but never downloads data."
    )
    parser.add_argument("--config", required=True, help="Path to the high-volume T1/T2 data-plan JSON config")
    parser.add_argument("--data-root", default=None, help="Override absolute data root, e.g. /data/projects/slwm")
    parser.add_argument("--stage", default=None, choices=STAGE_ORDER, help="Override active data ladder stage")
    parser.add_argument("--no-dry-run", action="store_true", help="Require explicit confirmation; network downloads are still safety-gated")
    parser.add_argument("--confirm-download", action="store_true", help="Confirm the operator intends to run non-dry-run acquisition")
    args = parser.parse_args(argv)
    summary = plan_t1_t2_high_volume_data(
        args.config,
        data_root_override=args.data_root,
        stage_override=args.stage,
        dry_run_override=False if args.no_dry_run else None,
        confirm_download=bool(args.confirm_download),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())


__all__ = ["plan_t1_t2_high_volume_data"]
