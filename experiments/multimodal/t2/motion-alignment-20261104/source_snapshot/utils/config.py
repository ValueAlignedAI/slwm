"""Dependency-free JSON config loader for Sprint I0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _path(path: str | Path) -> Path:
    return Path(path)


def canonical_json(config: Mapping[str, Any]) -> str:
    """Return stable JSON for hashing and round-trip tests."""

    return json.dumps(config, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def config_hash(config: Mapping[str, Any]) -> str:
    """Return ``sha256:<hex>`` hash over canonical JSON."""

    digest = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config file.

    I0 uses JSON to avoid adding undocumented YAML dependencies. JSON is also a
    valid subset of YAML for registry/config interchange.
    """

    with _path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Top-level config must be an object")
    return loaded


def write_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write a canonical JSON config and return the output path."""

    output_path = _path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(canonical_json(config), encoding="utf-8")
    return output_path
