"""Prepare GPT-2-BPE text/code corpora for Sprint T1 full-stack runs.

This script is intentionally separate from the dependency-light
``training.t1_text_baseline`` pilot runner.  It prepares text/code-only token
streams and a document-level manifest for runs that need the T1 acceptance
criteria beyond inline smoke data:

* GPT-2 BPE tokenizer metadata,
* pinned source names/configs/splits,
* stable document-level train/validation/test assignment,
* split/token hashes for registry traceability,
* no audio or visual data.

The output is a local artifact directory (ignored by git) that full T1 trainers
can consume without re-downloading or re-splitting data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from data.text_code import validate_t1_text_only_data_config
from data.tokenizer import TextTokenizer, build_text_tokenizer
from utils.config import config_hash, load_config, write_config


ARTIFACT_ROOT = Path("artifacts/t1_text_code")
ALLOWED_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class PreparedDocument:
    """Tokenized text/code document assigned later to a T1 split.

    Shape contract:
        ``tokens`` is a one-dimensional GPT-2-BPE ID sequence.  The training
        runner forms ``IntTensor[B,T]`` windows from concatenated split streams.
    """

    sample_id: str
    dataset: str
    source_name: str
    category: str
    language: str
    license_notes: str
    tokens: tuple[int, ...]
    text_sha256: str
    char_count: int

    @property
    def token_count(self) -> int:
        return len(self.tokens)


class _FormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:  # pragma: no cover - stdlib callback
        return ""


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_output_dir(raw_path: str | Path) -> Path:
    """Validate corpus output paths before writing large local artifacts."""

    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("T1 prepared corpus output_dir must be relative")
    repo_root = Path.cwd().resolve()
    artifact_root = (repo_root / ARTIFACT_ROOT).resolve()
    resolved = (repo_root / path).resolve()
    if not _is_relative_to(resolved, artifact_root):
        raise ValueError(f"T1 prepared corpora must be written under {ARTIFACT_ROOT}")
    if resolved.exists() and resolved.is_symlink():
        raise ValueError("T1 prepared corpus output_dir must not be a symlink")
    return path


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
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _stable_split(sample_id: str, *, seed: int, train_fraction: float, validation_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big") / float(2**64 - 1)
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def _extract_by_template(row: Mapping[str, Any], template: str) -> str:
    values = _FormatDict({key: _stringify_field(value) for key, value in row.items()})
    # Validate format fields early so typos produce deterministic empty strings
    # rather than surprising exceptions inside ``str.format_map``.
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name and field_name not in values:
            values[field_name] = ""
    return template.format_map(values)


def _stringify_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify_field(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _extract_text(row: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    template = source.get("text_template")
    if isinstance(template, str) and template.strip():
        return _extract_by_template(row, template)
    fields = source.get("text_fields")
    if fields is None and source.get("text_field") is not None:
        fields = [source.get("text_field")]
    if isinstance(fields, list) and fields:
        return "\n\n".join(_stringify_field(row.get(str(field), "")) for field in fields).strip()
    for default_field in ("text", "content", "code", "solution", "completion", "prompt"):
        value = row.get(default_field)
        if isinstance(value, str) and value.strip():
            return value
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _sample_id(row: Mapping[str, Any], *, source_name: str, index: int, id_fields: list[str]) -> str:
    parts = [source_name]
    for field in id_fields:
        value = row.get(field)
        if value not in (None, ""):
            parts.append(_stringify_field(value)[:120])
    if len(parts) == 1:
        parts.append(f"row-{index:08d}")
    return "::".join(parts)


def _document_from_text(
    *,
    tokenizer: TextTokenizer,
    text: str,
    sample_id: str,
    dataset: str,
    source_name: str,
    category: str,
    language: str,
    license_notes: str,
    min_chars: int,
) -> PreparedDocument | None:
    text = str(text).strip()
    if len(text) < int(min_chars):
        return None
    tokens = tuple(int(token_id) for token_id in tokenizer.encode(text, add_eos=True))
    if len(tokens) < 2:
        return None
    return PreparedDocument(
        sample_id=sample_id,
        dataset=dataset,
        source_name=source_name,
        category=category,
        language=language,
        license_notes=license_notes,
        tokens=tokens,
        text_sha256=_sha256_bytes(text.encode("utf-8", errors="replace")),
        char_count=len(text),
    )


def _iter_inline_records(source: Mapping[str, Any], *, tokenizer: TextTokenizer) -> tuple[list[PreparedDocument], dict[str, Any]]:
    docs: list[PreparedDocument] = []
    report = _stream_inline_records(source, tokenizer=tokenizer, consume_doc=docs.append)
    return docs, report


def _stream_inline_records(
    source: Mapping[str, Any],
    *,
    tokenizer: TextTokenizer,
    consume_doc: Callable[[PreparedDocument], None],
) -> dict[str, Any]:
    records = source.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("inline_records source requires a non-empty records list")
    source_name = str(source.get("name", "inline_records"))
    document_count = 0
    token_total = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("inline_records entries must be objects")
        text = _extract_text(record, source)
        doc = _document_from_text(
            tokenizer=tokenizer,
            text=text,
            sample_id=str(record.get("sample_id", f"{source_name}::{index:08d}")),
            dataset=str(record.get("dataset", source.get("dataset", source_name))),
            source_name=source_name,
            category=str(record.get("category", source.get("category", "text_english"))),
            language=str(record.get("language", source.get("language", "en"))),
            license_notes=str(record.get("license_notes", source.get("license_notes", "local inline text/code"))),
            min_chars=int(source.get("min_chars", 1)),
        )
        if doc is not None:
            consume_doc(doc)
            document_count += 1
            token_total += doc.token_count
    return {
        "name": source_name,
        "type": "inline_records",
        "status": "loaded",
        "documents": document_count,
        "tokens": int(token_total),
        "category": str(source.get("category", "text_english")),
        "license_notes": str(source.get("license_notes", "local inline text/code")),
    }


def _iter_local_text_files(source: Mapping[str, Any], *, tokenizer: TextTokenizer) -> tuple[list[PreparedDocument], dict[str, Any]]:
    docs: list[PreparedDocument] = []
    report = _stream_local_text_files(source, tokenizer=tokenizer, consume_doc=docs.append)
    return docs, report


def _stream_local_text_files(
    source: Mapping[str, Any],
    *,
    tokenizer: TextTokenizer,
    consume_doc: Callable[[PreparedDocument], None],
) -> dict[str, Any]:
    paths = source.get("paths", [])
    if not isinstance(paths, list) or not paths:
        raise ValueError("local_text_files source requires non-empty paths")
    source_name = str(source.get("name", "local_text_files"))
    document_count = 0
    token_total = 0
    for raw_path in paths:
        path = Path(str(raw_path))
        if path.is_absolute():
            raise ValueError("local_text_files paths must be relative to the repository root")
        text = path.read_text(encoding="utf-8", errors="replace")
        doc = _document_from_text(
            tokenizer=tokenizer,
            text=text,
            sample_id=f"{source_name}::{path.as_posix()}",
            dataset=str(source.get("dataset", source_name)),
            source_name=source_name,
            category=str(source.get("category", "markdown_docs")),
            language=str(source.get("language", "en")),
            license_notes=str(source.get("license_notes", "local repository file")),
            min_chars=int(source.get("min_chars", 1)),
        )
        if doc is not None:
            consume_doc(doc)
            document_count += 1
            token_total += doc.token_count
    return {
        "name": source_name,
        "type": "local_text_files",
        "status": "loaded",
        "documents": document_count,
        "tokens": int(token_total),
        "category": str(source.get("category", "markdown_docs")),
        "license_notes": str(source.get("license_notes", "local repository file")),
    }


def _iter_hf_dataset(source: Mapping[str, Any], *, tokenizer: TextTokenizer) -> tuple[list[PreparedDocument], dict[str, Any]]:
    docs: list[PreparedDocument] = []
    report = _stream_hf_dataset(source, tokenizer=tokenizer, consume_doc=docs.append)
    return docs, report


def _stream_hf_dataset(
    source: Mapping[str, Any],
    *,
    tokenizer: TextTokenizer,
    consume_doc: Callable[[PreparedDocument], None],
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("Hugging Face dataset sources require the optional 'datasets' dependency") from exc

    source_name = str(source.get("name", source.get("path", "hf_dataset")))
    path = str(source.get("path"))
    if not path or path == "None":
        raise ValueError("hf_dataset source requires a path")
    config_name = source.get("config_name", source.get("subset"))
    split = str(source.get("split", "train"))
    streaming = bool(source.get("streaming", True))
    trust_remote_code = bool(source.get("trust_remote_code", False))
    max_documents = int(source.get("max_documents", 1_000))
    max_tokens = int(source.get("max_tokens", 250_000))
    min_chars = int(source.get("min_chars", 40))
    id_fields = [str(field) for field in source.get("id_fields", ["id", "sha", "sha1", "url"]) if field]

    kwargs: dict[str, Any] = {"split": split, "streaming": streaming, "trust_remote_code": trust_remote_code}
    if source.get("data_dir"):
        kwargs["data_dir"] = source.get("data_dir")
    if source.get("data_files"):
        kwargs["data_files"] = source.get("data_files")
    if source.get("revision"):
        kwargs["revision"] = source.get("revision")

    dataset = load_dataset(path, config_name, **kwargs) if config_name else load_dataset(path, **kwargs)
    document_count = 0
    token_total = 0
    for index, row in enumerate(dataset):
        if document_count >= max_documents or token_total >= max_tokens:
            break
        if not isinstance(row, Mapping):
            continue
        text = _extract_text(row, source)
        sample_id = _sample_id(row, source_name=source_name, index=index, id_fields=id_fields)
        doc = _document_from_text(
            tokenizer=tokenizer,
            text=text,
            sample_id=sample_id,
            dataset=str(source.get("dataset", path)),
            source_name=source_name,
            category=str(source.get("category", "text_english")),
            language=str(row.get("language", source.get("language", "en"))),
            license_notes=str(source.get("license_notes", "license/terms recorded in source config")),
            min_chars=min_chars,
        )
        if doc is None:
            continue
        consume_doc(doc)
        document_count += 1
        token_total += doc.token_count
    return {
        "name": source_name,
        "type": "hf_dataset",
        "path": path,
        "config_name": config_name,
        "data_dir": source.get("data_dir"),
        "data_files": source.get("data_files"),
        "revision": source.get("revision"),
        "split": split,
        "streaming": streaming,
        "trust_remote_code": trust_remote_code,
        "status": "loaded",
        "documents": document_count,
        "tokens": int(token_total),
        "max_documents": max_documents,
        "max_tokens": max_tokens,
        "category": str(source.get("category", "text_english")),
        "language": str(source.get("language", "en")),
        "license_notes": str(source.get("license_notes", "license/terms recorded in source config")),
    }


def _load_source(source: Mapping[str, Any], *, tokenizer: TextTokenizer) -> tuple[list[PreparedDocument], dict[str, Any]]:
    source_type = str(source.get("type", "hf_dataset"))
    try:
        if source_type == "inline_records":
            return _iter_inline_records(source, tokenizer=tokenizer)
        if source_type == "local_text_files":
            return _iter_local_text_files(source, tokenizer=tokenizer)
        if source_type == "hf_dataset":
            return _iter_hf_dataset(source, tokenizer=tokenizer)
    except Exception as exc:
        if bool(source.get("optional", False)):
            return [], {"name": str(source.get("name", source_type)), "type": source_type, "status": "failed_optional", "error": repr(exc)}
        raise
    raise ValueError(f"Unsupported T1 source type {source_type!r}")


def _stream_source(
    source: Mapping[str, Any],
    *,
    tokenizer: TextTokenizer,
    consume_doc: Callable[[PreparedDocument], None],
) -> dict[str, Any]:
    source_type = str(source.get("type", "hf_dataset"))
    try:
        if source_type == "inline_records":
            return _stream_inline_records(source, tokenizer=tokenizer, consume_doc=consume_doc)
        if source_type == "local_text_files":
            return _stream_local_text_files(source, tokenizer=tokenizer, consume_doc=consume_doc)
        if source_type == "hf_dataset":
            return _stream_hf_dataset(source, tokenizer=tokenizer, consume_doc=consume_doc)
    except Exception as exc:
        if bool(source.get("optional", False)):
            return {"name": str(source.get("name", source_type)), "type": source_type, "status": "failed_optional", "error": repr(exc)}
        raise
    raise ValueError(f"Unsupported T1 source type {source_type!r}")


def _ensure_minimum_splits(assignments: dict[str, list[PreparedDocument]]) -> None:
    """Keep validation/test non-empty for small prepared smoke corpora."""

    total_docs = sum(len(value) for value in assignments.values())
    if total_docs < 3:
        return
    for split in ("validation", "test"):
        if assignments[split]:
            continue
        donor = "train" if len(assignments["train"]) > 1 else "validation"
        if assignments[donor]:
            assignments[split].append(assignments[donor].pop())


class _SplitTokenWriter:
    """Incrementally write one split token stream without Python token accumulation.

    Output contract after ``finalize``:
        ``<split>.tokens.npy`` is a one-dimensional ``uint32`` array that the
        T1 training runner can memory-map as ``IntTensor`` windows.
    """

    def __init__(self, *, split: str, output_dir: Path, copy_chunk_tokens: int = 1_000_000) -> None:
        self.split = split
        self.output_dir = output_dir
        self.raw_path = output_dir / f"{split}.tokens.uint32.tmp"
        self.token_path = output_dir / f"{split}.tokens.npy"
        self.copy_chunk_tokens = int(copy_chunk_tokens)
        self.documents = 0
        self.tokens = 0
        self._handle = self.raw_path.open("wb")

    def append(self, doc: PreparedDocument) -> None:
        array = np.asarray(doc.tokens, dtype=np.uint32)
        array.tofile(self._handle)
        self.documents += 1
        self.tokens += int(array.size)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def finalize(self) -> dict[str, Any]:
        self.close()
        mmap = np.lib.format.open_memmap(self.token_path, mode="w+", dtype=np.uint32, shape=(self.tokens,))
        copied = 0
        with self.raw_path.open("rb") as handle:
            while copied < self.tokens:
                count = min(self.copy_chunk_tokens, self.tokens - copied)
                chunk = np.fromfile(handle, dtype=np.uint32, count=count)
                if chunk.size == 0:
                    break
                mmap[copied : copied + int(chunk.size)] = chunk
                copied += int(chunk.size)
        del mmap
        if copied != self.tokens:
            raise IOError(f"Only copied {copied} of {self.tokens} tokens for split {self.split}")
        self.raw_path.unlink(missing_ok=True)
        return {"path": str(self.token_path), "sha256": _sha256_path(self.token_path), "tokens": int(self.tokens)}


def _manifest_row(doc: PreparedDocument, *, split: str) -> dict[str, Any]:
    return {
        "sample_id": doc.sample_id,
        "dataset": doc.dataset,
        "source_name": doc.source_name,
        "category": doc.category,
        "language": doc.language,
        "license_notes": doc.license_notes,
        "split": split,
        "token_count": doc.token_count,
        "char_count": doc.char_count,
        "text_sha256": doc.text_sha256,
    }


def _dataset_card_payload(
    *,
    start: float,
    config: Mapping[str, Any],
    config_path: Path,
    config_copy_path: Path,
    data_cfg: Mapping[str, Any],
    tokenizer: TextTokenizer,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    split_seed: int,
    source_reports: list[dict[str, Any]],
    split_counts: dict[str, dict[str, int]],
    token_files: dict[str, dict[str, Any]],
    manifest_path: Path,
    preparation_mode: str,
) -> dict[str, Any]:
    return {
        "sprint": "T1",
        "status": "prepared",
        "claim_scope": "Prepared GPT-2-BPE text/code corpus; training evidence requires registered model runs.",
        "prepared_at_unix": time.time(),
        "source_config_path": str(config_path),
        "source_config_hash": config_hash(config),
        "config_copy_path": str(config_copy_path),
        "tokenizer": tokenizer.metadata(),
        "modalities": ["text_code"],
        "dataset_mix": data_cfg.get("dataset_mix", {"text_code": 1.0, "audio": None, "visual_video": None}),
        "target_tokens": data_cfg.get("target_tokens"),
        "preparation_mode": preparation_mode,
        "split_policy": {"train": train_fraction, "validation": validation_fraction, "test": test_fraction, "seed": split_seed},
        "sources": source_reports,
        "split_counts": split_counts,
        "token_files": token_files,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_path(manifest_path),
        "wall_clock_time_seconds": time.perf_counter() - start,
        "limitations": [
            "Dataset quality depends on source configs and filters recorded here.",
            "Code datasets may require authenticated access for The Stack v2/StarCoderData; any fallback source is labeled in source reports.",
            "No audio or visual/video samples are included in T1 prepared corpora.",
            "The preparation script records stable document splits and hashes but does not by itself prove deduplication, PII filtering, or eval contamination removal.",
        ],
    }


def _prepare_t1_text_code_corpus_streaming(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    data_cfg: Mapping[str, Any],
    output_dir: Path,
    tokenizer: TextTokenizer,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    split_seed: int,
    sources: list[Any],
    start: float,
) -> dict[str, Any]:
    """Prepare a large corpus in one pass using raw split files + final .npy conversion."""

    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {split: _SplitTokenWriter(split=split, output_dir=output_dir) for split in ALLOWED_SPLITS}
    source_reports: list[dict[str, Any]] = []
    manifest_path = output_dir / "manifest.jsonl"
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    total_documents = 0

    try:
        with manifest_tmp.open("w", encoding="utf-8") as manifest_handle:

            def consume_doc(doc: PreparedDocument) -> None:
                nonlocal total_documents
                split = _stable_split(doc.sample_id, seed=split_seed, train_fraction=train_fraction, validation_fraction=validation_fraction)
                writers[split].append(doc)
                manifest_handle.write(json.dumps(_manifest_row(doc, split=split), sort_keys=True) + "\n")
                total_documents += 1

            for source in sources:
                if not isinstance(source, Mapping):
                    raise ValueError("data.sources entries must be objects")
                source_reports.append(_stream_source(source, tokenizer=tokenizer, consume_doc=consume_doc))
    finally:
        for writer in writers.values():
            writer.close()

    if total_documents == 0:
        raise ValueError("No T1 text/code documents were prepared")
    if writers["train"].documents == 0 or writers["validation"].documents == 0:
        raise ValueError("Streaming preparation produced an empty train or validation split; increase source document counts")

    manifest_tmp.replace(manifest_path)
    token_files: dict[str, dict[str, Any]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for split, writer in writers.items():
        token_files[split] = writer.finalize()
        split_counts[split] = {"documents": int(writer.documents), "tokens": int(writer.tokens)}

    config_copy_path = write_config(config, output_dir / "prepare_config.json")
    dataset_card = _dataset_card_payload(
        start=start,
        config=config,
        config_path=config_path,
        config_copy_path=config_copy_path,
        data_cfg=data_cfg,
        tokenizer=tokenizer,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
        source_reports=source_reports,
        split_counts=split_counts,
        token_files=token_files,
        manifest_path=manifest_path,
        preparation_mode="streaming",
    )
    card_path = _write_json(dataset_card, output_dir / "dataset_card.json")
    dataset_card["dataset_card_path"] = str(card_path)
    _write_json(dataset_card, card_path)
    return dataset_card


def prepare_t1_text_code_corpus(config_path: str | Path) -> dict[str, Any]:
    """Prepare a tokenized text/code-only corpus for full-stack T1 runs."""

    start = time.perf_counter()
    path = Path(config_path)
    config = load_config(path)
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}
    validate_t1_text_only_data_config(data_cfg)
    output_dir = _safe_output_dir(data_cfg.get("output_dir", "artifacts/t1_text_code/default"))
    tokenizer = build_text_tokenizer(config.get("tokenizer", config.get("model", {})))
    split_cfg = data_cfg.get("split_policy", {}) if isinstance(data_cfg.get("split_policy", {}), Mapping) else {}
    train_fraction = float(split_cfg.get("train", 0.98))
    validation_fraction = float(split_cfg.get("validation", 0.01))
    test_fraction = float(split_cfg.get("test", 0.01))
    split_seed = int(split_cfg.get("seed", config.get("runtime", {}).get("seed", 0) if isinstance(config.get("runtime", {}), Mapping) else 0))
    if not (0.0 < train_fraction < 1.0) or validation_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("split fractions must be positive and train must be in (0,1)")
    if abs((train_fraction + validation_fraction + test_fraction) - 1.0) > 1e-6:
        raise ValueError("train/validation/test split fractions must sum to 1.0")

    sources = data_cfg.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("T1 corpus preparation requires at least one data source")
    preparation_mode = str(data_cfg.get("preparation_mode", "in_memory"))
    if preparation_mode not in {"in_memory", "streaming"}:
        raise ValueError("data.preparation_mode must be either 'in_memory' or 'streaming'")
    if preparation_mode == "streaming":
        return _prepare_t1_text_code_corpus_streaming(
            config=config,
            config_path=path,
            data_cfg=data_cfg,
            output_dir=output_dir,
            tokenizer=tokenizer,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            split_seed=split_seed,
            sources=sources,
            start=start,
        )

    docs: list[PreparedDocument] = []
    source_reports: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("data.sources entries must be objects")
        loaded_docs, report = _load_source(source, tokenizer=tokenizer)
        docs.extend(loaded_docs)
        source_reports.append(report)
    if not docs:
        raise ValueError("No T1 text/code documents were prepared")

    assignments: dict[str, list[PreparedDocument]] = {split: [] for split in ALLOWED_SPLITS}
    for doc in docs:
        split = _stable_split(doc.sample_id, seed=split_seed, train_fraction=train_fraction, validation_fraction=validation_fraction)
        assignments[split].append(doc)
    _ensure_minimum_splits(assignments)

    output_dir.mkdir(parents=True, exist_ok=True)
    token_files: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}
    for split, split_docs in assignments.items():
        tokens: list[int] = []
        for doc in split_docs:
            tokens.extend(doc.tokens)
            manifest_rows.append(_manifest_row(doc, split=split))
        array = np.asarray(tokens, dtype=np.uint32)
        token_path = output_dir / f"{split}.tokens.npy"
        np.save(token_path, array)
        token_files[split] = {"path": str(token_path), "sha256": _sha256_path(token_path), "tokens": int(array.size)}
        split_counts[split] = {"documents": len(split_docs), "tokens": int(array.size)}

    manifest_path = _write_jsonl(manifest_rows, output_dir / "manifest.jsonl")
    config_copy_path = write_config(config, output_dir / "prepare_config.json")
    dataset_card = _dataset_card_payload(
        start=start,
        config=config,
        config_path=path,
        config_copy_path=config_copy_path,
        data_cfg=data_cfg,
        tokenizer=tokenizer,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        split_seed=split_seed,
        source_reports=source_reports,
        split_counts=split_counts,
        token_files=token_files,
        manifest_path=manifest_path,
        preparation_mode="in_memory",
    )
    card_path = _write_json(dataset_card, output_dir / "dataset_card.json")
    dataset_card["dataset_card_path"] = str(card_path)
    _write_json(dataset_card, card_path)
    return dataset_card


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a GPT-2-BPE text/code corpus for Sprint T1")
    parser.add_argument("--config", required=True, help="Path to a T1 corpus preparation JSON config")
    args = parser.parse_args(argv)
    card = prepare_t1_text_code_corpus(args.config)
    print(json.dumps(card, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())


__all__ = ["prepare_t1_text_code_corpus"]
