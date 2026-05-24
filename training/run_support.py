"""Reusable training-run support utilities.

This module intentionally contains only generic mechanics that every training
runner needs: safe artifact path validation, atomic writes, JSONL progress
events, memory telemetry, and checkpoint payload validation/loading helpers.
Sprint-specific runners remain responsible for model/data/loss/eval semantics
and claim language.
"""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


try:  # Optional for docs/tests that do not install torch.
    import torch
except Exception:  # pragma: no cover - backend-dependent optional import
    torch = None  # type: ignore[assignment]


def is_relative_to(path: Path, parent: Path) -> bool:
    """Compatibility helper for checking path containment."""

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class SafePathPolicy:
    """Repository-relative path allowlist for run artifacts and inputs."""

    root: Path
    label: str

    def validate(self, raw_path: str | Path, *, must_exist: bool = False) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            raise ValueError(f"{self.label} paths must be repository-relative")
        repo_root = Path.cwd().resolve()
        root = (repo_root / self.root).resolve()
        candidate = repo_root / path
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"{self.label} path must not be a symlink")
        resolved = candidate.resolve()
        if not is_relative_to(resolved, root):
            raise ValueError(f"{self.label} paths must be under {self.root}")
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"{self.label} path does not exist: {path}")
        return path


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> Path:
    """Atomically write a JSON mapping beside the target file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def atomic_write_text(payload: str, path: Path) -> Path:
    """Atomically write text beside the target file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def append_jsonl(path: Path, payload: Mapping[str, Any], *, echo_stdout: bool = True) -> None:
    """Append one JSONL event and optionally echo it for nohup/system logs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    if echo_stdout:
        print(line, flush=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def max_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def accelerator_memory_stats_mb() -> dict[str, float | None]:
    """Return generic accelerator memory telemetry in MiB."""

    if torch is None:
        return {"allocated_mb": None, "reserved_mb": None, "max_allocated_mb": None, "max_reserved_mb": None}
    if torch.cuda.is_available():
        return {
            "allocated_mb": float(torch.cuda.memory_allocated() / (1024.0 * 1024.0)),
            "reserved_mb": float(torch.cuda.memory_reserved() / (1024.0 * 1024.0)),
            "max_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
            "max_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)),
        }
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            current = float(torch.mps.current_allocated_memory() / (1024.0 * 1024.0))
        except Exception:  # pragma: no cover - backend-dependent
            current = None
        return {"allocated_mb": current, "reserved_mb": None, "max_allocated_mb": current, "max_reserved_mb": None}
    return {"allocated_mb": None, "reserved_mb": None, "max_allocated_mb": None, "max_reserved_mb": None}


def accelerator_peak_memory_mb() -> float | None:
    return accelerator_memory_stats_mb()["max_allocated_mb"]


def torch_save_atomic(payload: Mapping[str, Any], path: Path) -> Path:
    if torch is None:  # pragma: no cover - optional dependency guard
        raise ImportError("torch_save_atomic requires PyTorch")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)
    return path


def copy_atomic(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, tmp)
    tmp.replace(target)
    return target


def torch_load_checkpoint(path: Path, device: Any) -> Mapping[str, Any]:
    if torch is None:  # pragma: no cover - optional dependency guard
        raise ImportError("torch_load_checkpoint requires PyTorch")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older torch compatibility
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    return payload


@dataclass(frozen=True)
class CheckpointSpec:
    """Generic metadata contract for fail-closed checkpoint loading."""

    schema_version: str
    required_keys: frozenset[str]
    expected_values: Mapping[str, Any]
    allow_mismatch_keys: frozenset[str] = field(default_factory=frozenset)


def validate_checkpoint_payload(payload: Mapping[str, Any], spec: CheckpointSpec) -> None:
    missing = sorted(set(spec.required_keys).difference(payload.keys()))
    if missing:
        raise ValueError(f"checkpoint missing required keys: {missing}")
    if payload.get("schema_version") != spec.schema_version:
        raise ValueError("checkpoint schema_version is not supported")
    for key, expected in spec.expected_values.items():
        if key in spec.allow_mismatch_keys:
            continue
        if expected is None:
            raise ValueError(f"current expected checkpoint value is missing for {key}")
        if key not in payload:
            raise ValueError(f"checkpoint missing expected metadata key: {key}")
        if str(payload.get(key)) != str(expected):
            raise ValueError(f"checkpoint {key} does not match current run")


__all__ = [
    "CheckpointSpec",
    "SafePathPolicy",
    "accelerator_memory_stats_mb",
    "accelerator_peak_memory_mb",
    "append_jsonl",
    "atomic_write_json",
    "atomic_write_text",
    "copy_atomic",
    "file_sha256",
    "is_relative_to",
    "max_rss_mb",
    "torch_load_checkpoint",
    "torch_save_atomic",
    "validate_checkpoint_payload",
]
