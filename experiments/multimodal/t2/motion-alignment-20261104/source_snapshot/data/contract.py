"""Unified multimodal sample contract for SLWM-124M.

I0 deliberately stores schema and validation helpers only. It does not load
datasets, preprocess signals, or create tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


REQUIRED_MODALITIES: tuple[str, ...] = ("text_code", "audio", "visual_video")
"""Minimum modalities required by the project contract."""

MODALITY_IDS: dict[str, int] = {
    "noop": 0,
    "text_code": 1,
    "audio": 2,
    "visual_video": 3,
}
"""Stable modality IDs used in adapter metadata and tests."""

SOURCE_TAGS: tuple[str, ...] = (
    "observed",
    "reconstructed",
    "predicted",
    "inferred",
    "imagined",
    "unknown",
    "unsupported",
)
"""Required source/uncertainty labels for future heads and probes."""


@dataclass(frozen=True)
class SignalStreamRef:
    """Reference to one modality stream in a unified sample.

    Shape contract: this object is pre-tensor. Adapters later map each stream
    to ``z: FloatTensor[B,T,D]`` and ``mask: BoolTensor[B,T]``.
    """

    modality: str
    data: Any = None
    path: str | None = None
    start: float | None = None
    end: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.modality not in MODALITY_IDS:
            known = ", ".join(sorted(MODALITY_IDS))
            raise ValueError(f"Unknown modality {self.modality!r}; expected one of: {known}")


@dataclass(frozen=True)
class SignalSample:
    """Unified sample passed to data/adapters in later sprints.

    Shape contract after adapters:
        streams[modality] -> {"z": FloatTensor[B,T,D],
                              "mask": BoolTensor[B,T],
                              "metadata": dict}
    """

    sample_id: str
    streams: dict[str, SignalStreamRef]
    targets: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_sample_contract(sample: Mapping[str, Any]) -> None:
    """Validate the minimal dictionary form of the I0 sample contract.

    Args:
        sample: Mapping with ``sample_id``, ``streams``, ``targets`` and
            ``metadata`` keys. This is schema validation only; no dataset IO is
            performed.

    Raises:
        ValueError: if required fields or known modality identifiers are absent.
    """

    required_keys = ("sample_id", "streams", "targets", "metadata")
    missing = [key for key in required_keys if key not in sample]
    if missing:
        raise ValueError(f"Sample is missing required keys: {missing}")

    streams = sample["streams"]
    if not isinstance(streams, Mapping):
        raise ValueError("sample['streams'] must be a mapping")

    for modality in streams:
        if modality not in MODALITY_IDS:
            known = ", ".join(sorted(MODALITY_IDS))
            raise ValueError(f"Unknown stream modality {modality!r}; expected one of: {known}")
