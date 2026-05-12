"""Text/code dataset helpers for Sprint T1.

The helpers here intentionally do not download external corpora.  They provide a
deterministic, documented path for tiny local pilots and a stable interface that
full FineWeb/The Stack style preprocessing can later target.  All emitted data is
text-only (`text_code` modality id 1); audio and visual streams are rejected in
T1 configs to preserve sprint scope.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from data.contract import MODALITY_IDS, validate_sample_contract
from data.tokenizer import TextTokenizer


ALLOWED_SPLITS: tuple[str, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class TextCodeRecord:
    """One pre-tokenization text/code sample for T1.

    Shape contract after tokenization:
        ``tokens`` is a one-dimensional integer sequence.  Windowing converts it
        into language-model pairs ``input_ids: IntTensor[B,T]`` and
        ``target_ids: IntTensor[B,T]``.
    """

    sample_id: str
    text: str
    split: str
    dataset: str
    license_notes: str = "local_or_configured"
    language: str = "en"

    def contract_sample(self) -> dict[str, Any]:
        """Return the high-level data-contract dictionary for validation."""

        return {
            "sample_id": self.sample_id,
            "streams": {"text_code": {"data": self.text, "start": 0.0, "end": None}},
            "targets": {"future_text": None, "future_audio": None, "future_video": None, "caption": None, "answerability": None},
            "metadata": {"dataset": self.dataset, "license": self.license_notes, "language": self.language, "split": self.split},
        }


@dataclass(frozen=True)
class TokenWindowDataset:
    """Token windows for one split of next-token language modeling.

    Shape contract:
        ``windows`` has shape ``IntTensor[N,T+1]``.  Batches use
        ``windows[:, :-1]`` as input IDs and ``windows[:, 1:]`` as targets.
    """

    split: str
    windows: np.ndarray
    source_sample_ids: tuple[str, ...]
    token_count: int
    repeated_to_minimum: bool
    digest: str

    @property
    def sample_count(self) -> int:
        return int(self.windows.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.windows.shape[1] - 1)

    def batch(self, *, batch_size: int, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Return a deterministic cycling batch for ``step``.

        Returns:
            ``(input_ids, target_ids)`` each with shape ``IntTensor[B,T]``.
        """

        if self.sample_count <= 0:
            raise ValueError(f"Split {self.split!r} has no windows")
        indices = (np.arange(int(batch_size), dtype=np.int64) + int(step) * int(batch_size)) % self.sample_count
        batch = self.windows[indices]
        return batch[:, :-1].astype(np.int64), batch[:, 1:].astype(np.int64)


@dataclass(frozen=True)
class TextCodeDatasetBundle:
    """Train/validation/test token-window bundle for T1."""

    train: TokenWindowDataset
    validation: TokenWindowDataset
    test: TokenWindowDataset | None
    records: tuple[TextCodeRecord, ...]
    tokenizer_metadata: Mapping[str, Any]
    dataset_mix: Mapping[str, Any]
    datasets_config: Sequence[Mapping[str, Any]]
    sample_schema_version: str

    def registry_datasets(self) -> list[dict[str, Any]]:
        """Return dataset rows compatible with ``docs/experiments/experiment_registry.md``."""

        by_dataset: dict[tuple[str, str], dict[str, Any]] = {}
        for record in self.records:
            key = (record.dataset, record.split)
            if key not in by_dataset:
                by_dataset[key] = {
                    "name": record.dataset,
                    "version_or_snapshot": "inline_or_prepared_t1_config",
                    "split": record.split,
                    "sample_count": 0,
                    "tokens": 0,
                    "audio_hours": None,
                    "video_hours": None,
                    "license_notes": record.license_notes,
                    "leakage_checks": "document-level split by configured split field; split digests recorded in metrics",
                }
            by_dataset[key]["sample_count"] += 1
        split_tokens = {"train": self.train.token_count, "validation": self.validation.token_count}
        if self.test is not None:
            split_tokens["test"] = self.test.token_count
        for (dataset, split), row in by_dataset.items():
            row["tokens"] = int(split_tokens.get(split, 0)) if dataset == "t1_inline_text_code_pilot" else row["tokens"]
        return list(by_dataset.values())

    def split_digests(self) -> dict[str, str | None]:
        """Return stable split digests for registry/metrics."""

        return {
            "train": self.train.digest,
            "validation": self.validation.digest,
            "test": None if self.test is None else self.test.digest,
        }


def _sha256_int_sequence(values: Sequence[int]) -> str:
    digest = hashlib.sha256(",".join(str(int(v)) for v in values).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _records_from_inline_config(data_cfg: Mapping[str, Any]) -> list[TextCodeRecord]:
    raw_records = data_cfg.get("inline_records", [])
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("T1 text/code data config requires non-empty inline_records for dependency-free pilot runs")

    records: list[TextCodeRecord] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ValueError("Each inline_records item must be an object")
        split = str(raw.get("split", "train"))
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Unknown T1 split {split!r}; expected one of {ALLOWED_SPLITS}")
        text = str(raw.get("text", ""))
        if not text:
            raise ValueError("T1 inline text/code records must contain non-empty text")
        records.append(
            TextCodeRecord(
                sample_id=str(raw.get("sample_id", f"inline-{index:04d}")),
                text=text,
                split=split,
                dataset=str(raw.get("dataset", "t1_inline_text_code_pilot")),
                license_notes=str(raw.get("license_notes", "local synthetic/configured text for smoke testing")),
                language=str(raw.get("language", "en")),
            )
        )
    return records


def validate_t1_text_only_data_config(data_cfg: Mapping[str, Any]) -> None:
    """Reject audio/visual data in Sprint T1 configs."""

    dataset_mix = data_cfg.get("dataset_mix", {}) if isinstance(data_cfg.get("dataset_mix", {}), Mapping) else {}
    for forbidden in ("audio", "visual_video"):
        value = dataset_mix.get(forbidden)
        if value not in (None, 0, 0.0, "", "none", "None"):
            raise ValueError(f"Sprint T1 must not include {forbidden} data; got dataset_mix[{forbidden!r}]={value!r}")
    modalities = data_cfg.get("modalities", ["text_code"])
    if any(str(modality) != "text_code" for modality in modalities):
        raise ValueError(f"Sprint T1 modalities must be ['text_code']; got {modalities!r}")
    if MODALITY_IDS["text_code"] != 1:
        raise ValueError("data contract modality id for text_code must remain 1")


def _token_stream_for_split(records: Sequence[TextCodeRecord], tokenizer: TextTokenizer, split: str) -> tuple[list[int], tuple[str, ...]]:
    selected = [record for record in records if record.split == split]
    if not selected:
        return [], ()
    tokens: list[int] = []
    sample_ids: list[str] = []
    for record in selected:
        validate_sample_contract(record.contract_sample())
        tokens.extend(tokenizer.encode(record.text, add_eos=True))
        sample_ids.append(record.sample_id)
    return tokens, tuple(sample_ids)


def _make_windows(tokens: Sequence[int], *, sequence_length: int, split: str, source_sample_ids: tuple[str, ...]) -> TokenWindowDataset:
    if int(sequence_length) <= 0:
        raise ValueError("sequence_length must be positive")
    token_list = [int(token) for token in tokens]
    repeated = False
    needed = int(sequence_length) + 1
    if not token_list:
        raise ValueError(f"Split {split!r} has no tokens")
    while len(token_list) < needed:
        token_list.extend(token_list)
        repeated = True
    stride = max(1, int(sequence_length))
    rows: list[list[int]] = []
    for start in range(0, max(1, len(token_list) - needed + 1), stride):
        window = token_list[start : start + needed]
        if len(window) == needed:
            rows.append(window)
    if not rows:
        rows.append(token_list[:needed])
    windows = np.asarray(rows, dtype=np.int64)
    return TokenWindowDataset(
        split=split,
        windows=windows,
        source_sample_ids=source_sample_ids,
        token_count=len(token_list),
        repeated_to_minimum=repeated,
        digest=_sha256_int_sequence(token_list),
    )


def build_text_code_lm_datasets(config: Mapping[str, Any], tokenizer: TextTokenizer) -> TextCodeDatasetBundle:
    """Build deterministic language-model datasets from a T1 config.

    Returns:
        A ``TextCodeDatasetBundle`` whose train/validation windows can be fed to
        GPT-2 and SLWM text-only variants under identical tokenization/splits.
    """

    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}
    train_cfg = config.get("training", {}) if isinstance(config.get("training", {}), Mapping) else {}
    validate_t1_text_only_data_config(data_cfg)
    records = tuple(_records_from_inline_config(data_cfg))
    sequence_length = int(train_cfg.get("sequence_length", data_cfg.get("sequence_length", config.get("model", {}).get("context_length", 16))))

    train_tokens, train_ids = _token_stream_for_split(records, tokenizer, "train")
    val_tokens, val_ids = _token_stream_for_split(records, tokenizer, "validation")
    test_tokens, test_ids = _token_stream_for_split(records, tokenizer, "test")
    if not train_tokens:
        raise ValueError("T1 requires at least one train split text_code record")
    if not val_tokens:
        raise ValueError("T1 requires at least one validation split text_code record")

    train = _make_windows(train_tokens, sequence_length=sequence_length, split="train", source_sample_ids=train_ids)
    validation = _make_windows(val_tokens, sequence_length=sequence_length, split="validation", source_sample_ids=val_ids)
    test = None if not test_tokens else _make_windows(test_tokens, sequence_length=sequence_length, split="test", source_sample_ids=test_ids)
    return TextCodeDatasetBundle(
        train=train,
        validation=validation,
        test=test,
        records=records,
        tokenizer_metadata=tokenizer.metadata(),
        dataset_mix=data_cfg.get("dataset_mix", {"text_code": 1.0, "audio": None, "visual_video": None}),
        datasets_config=data_cfg.get("datasets", []),
        sample_schema_version=str(data_cfg.get("sample_schema_version", "t1.text_code_v0")),
    )


__all__ = [
    "TextCodeDatasetBundle",
    "TextCodeRecord",
    "TokenWindowDataset",
    "build_text_code_lm_datasets",
    "validate_t1_text_only_data_config",
]
