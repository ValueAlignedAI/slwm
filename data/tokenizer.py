"""Text/code tokenizers for Sprint T1.

Sprint T1 requires the GPT-2 BPE tokenizer for fair text/code baseline
comparisons when a full training stack is available.  The dependency-light tiny
pilot still uses a deterministic byte fallback tokenizer and records that
fallback explicitly in configs/registries.  The full-T1 path can use
``transformers`` to load GPT-2 BPE lazily, while keeping the same tiny-pilot
imports dependency-light for tests and offline shape checks.

Both GPT-2 and SLWM variants consume the same tokenizer object, preserving the
same-tokenizer guardrail for pilot, smoke, and full-stack evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class TextTokenizer(Protocol):
    """Minimal tokenizer protocol used by T1 text/code runners.

    Shape contract:
        ``encode(text)`` returns a list of integer IDs with values in
        ``[0, vocab_size)``.  ``decode(ids)`` returns a UTF-8 string for sample
        generation reports.
    """

    name: str
    vocab_size: int
    pad_token_id: int
    eos_token_id: int

    def encode(self, text: str, *, add_eos: bool = True) -> list[int]:
        """Encode text/code into integer IDs."""

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        """Decode integer IDs into UTF-8 text."""

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable tokenizer metadata."""


@dataclass(frozen=True)
class ByteFallbackTokenizer:
    """Deterministic byte-level fallback tokenizer for local T1 pilots.

    The tokenizer reserves ``0`` for padding, ``1`` for EOS, and maps raw bytes
    ``0..255`` to IDs ``2..257``.  ``vocab_size`` must therefore be at least
    ``258``.  It is not a GPT-2 BPE implementation; configs must label runs that
    use it as pilot/smoke runs rather than GPT-2-comparable evidence.
    """

    name: str = "byte_fallback_t1_smoke"
    vocab_size: int = 260
    pad_token_id: int = 0
    eos_token_id: int = 1
    byte_offset: int = 2
    intended_tokenizer: str = "gpt2_bpe"
    claim_scope: str = "local_pilot_not_gpt2_bpe_evidence"

    def __post_init__(self) -> None:
        if int(self.vocab_size) < self.byte_offset + 256:
            raise ValueError("ByteFallbackTokenizer requires vocab_size >= 258")
        if self.pad_token_id == self.eos_token_id:
            raise ValueError("pad_token_id and eos_token_id must differ")

    def encode(self, text: str, *, add_eos: bool = True) -> list[int]:
        """Encode a string as UTF-8 byte IDs.

        Args:
            text: English text/code string.
            add_eos: Whether to append the EOS token.

        Returns:
            A list of IDs in ``[0, vocab_size)``.
        """

        ids = [int(byte) + self.byte_offset for byte in str(text).encode("utf-8", errors="replace")]
        if add_eos:
            ids.append(int(self.eos_token_id))
        return ids

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        """Decode byte IDs, skipping padding/EOS and unknown control IDs."""

        raw = bytearray()
        for token_id in token_ids:
            token = int(token_id)
            if token in {self.pad_token_id, self.eos_token_id}:
                continue
            byte_value = token - self.byte_offset
            if 0 <= byte_value <= 255:
                raw.append(byte_value)
        return raw.decode("utf-8", errors="replace")

    def metadata(self) -> dict[str, Any]:
        """Return metadata recorded in T1 configs, metrics, and registry."""

        return {
            "type": self.name,
            "effective_type": "byte_fallback",
            "intended_tokenizer": self.intended_tokenizer,
            "vocab_size": int(self.vocab_size),
            "pad_token_id": int(self.pad_token_id),
            "eos_token_id": int(self.eos_token_id),
            "byte_offset": int(self.byte_offset),
            "claim_scope": self.claim_scope,
            "notes": "Dependency-light local tokenizer; replace with GPT-2 BPE before full T1 evidence claims.",
        }


class GPT2BPETokenizer:
    """GPT-2 BPE tokenizer wrapper used by the full Sprint T1 path.

    The wrapper intentionally follows the small ``TextTokenizer`` protocol used
    by the repository instead of exposing the entire Hugging Face tokenizer API.
    It appends EOS explicitly so text/code corpus preparation can create stable
    document boundaries.

    Shape contract:
        ``encode(text)`` returns ``list[int]`` with values in
        ``[0, vocab_size)``.  Batched training code later forms
        ``IntTensor[B,T]`` windows from the resulting one-dimensional token
        stream.
    """

    def __init__(
        self,
        *,
        pretrained_name: str = "gpt2",
        cache_dir: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        claim_scope: str = "full_t1_gpt2_bpe_text_code_evidence_requires_registered_data_controls",
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - exercised only without optional dependency
            raise ImportError(
                "GPT-2 BPE tokenizer requires the optional 'transformers' dependency. "
                "Install the T1 full-stack dependencies or use byte_fallback for smoke tests."
            ) from exc

        kwargs: dict[str, Any] = {"use_fast": True, "local_files_only": bool(local_files_only)}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if revision:
            kwargs["revision"] = revision
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_name, **kwargs)
        if self._tokenizer.eos_token_id is None:
            raise ValueError(f"Tokenizer {pretrained_name!r} must expose an EOS token for T1 document boundaries")
        if self._tokenizer.pad_token_id is None:
            # GPT-2 has no native pad token.  T1 causal LM batching uses fixed
            # windows, so no padding is needed during training; recording EOS as
            # the pad ID keeps metadata complete without changing tokenization.
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self.name = "gpt2_bpe"
        self.pretrained_name = pretrained_name
        self.revision = revision
        self.cache_dir = cache_dir
        self.claim_scope = claim_scope
        self.vocab_size = int(len(self._tokenizer))
        self.pad_token_id = int(self._tokenizer.pad_token_id)
        self.eos_token_id = int(self._tokenizer.eos_token_id)

    def encode(self, text: str, *, add_eos: bool = True) -> list[int]:
        """Encode text/code with GPT-2 BPE and optional EOS boundary."""

        ids = [int(token_id) for token_id in self._tokenizer.encode(str(text), add_special_tokens=False)]
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        """Decode GPT-2 BPE IDs for sample-generation reports."""

        return str(self._tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=True))

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable tokenizer metadata for registry artifacts."""

        return {
            "type": self.name,
            "effective_type": "gpt2_bpe",
            "intended_tokenizer": "gpt2_bpe",
            "pretrained_name": self.pretrained_name,
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "vocab_size": int(self.vocab_size),
            "pad_token_id": int(self.pad_token_id),
            "eos_token_id": int(self.eos_token_id),
            "claim_scope": self.claim_scope,
            "notes": "GPT-2 BPE edge codec for full Sprint T1 text/code comparisons.",
        }


def build_text_tokenizer(config: Mapping[str, Any]) -> TextTokenizer:
    """Build the tokenizer requested by a T1 config.

    Supported tokenizer types:
        - ``byte_fallback_t1_smoke`` / ``byte_fallback``
        - ``gpt2_bpe`` / ``hf_gpt2_bpe`` (requires optional ``transformers``)

    The config may still record ``intended_tokenizer: gpt2_bpe`` so the dataset
    decision record is explicit about the intended full-run tokenizer.
    """

    tokenizer_cfg = config.get("tokenizer", config) if isinstance(config.get("tokenizer", config), Mapping) else config
    tokenizer_type = str(tokenizer_cfg.get("type", "byte_fallback_t1_smoke"))
    if tokenizer_type in {"byte_fallback_t1_smoke", "byte_fallback", "dependency_free_byte_fallback"}:
        return ByteFallbackTokenizer(
            name=tokenizer_type,
            vocab_size=int(tokenizer_cfg.get("vocab_size", 260)),
            pad_token_id=int(tokenizer_cfg.get("pad_token_id", 0)),
            eos_token_id=int(tokenizer_cfg.get("eos_token_id", 1)),
            byte_offset=int(tokenizer_cfg.get("byte_offset", 2)),
            intended_tokenizer=str(tokenizer_cfg.get("intended_tokenizer", "gpt2_bpe")),
            claim_scope=str(tokenizer_cfg.get("claim_scope", "local_pilot_not_gpt2_bpe_evidence")),
        )
    if tokenizer_type in {"gpt2_bpe", "gpt2", "hf_gpt2_bpe"}:
        return GPT2BPETokenizer(
            pretrained_name=str(tokenizer_cfg.get("pretrained_name", tokenizer_cfg.get("name", "gpt2"))),
            cache_dir=tokenizer_cfg.get("cache_dir"),
            revision=tokenizer_cfg.get("revision"),
            local_files_only=bool(tokenizer_cfg.get("local_files_only", False)),
            claim_scope=str(
                tokenizer_cfg.get(
                    "claim_scope",
                    "full_t1_gpt2_bpe_text_code_evidence_requires_registered_data_controls",
                )
            ),
        )
    raise ValueError(
        f"Unsupported tokenizer type {tokenizer_type!r}. "
        "Sprint T1 supports byte_fallback_t1_smoke for smoke tests and gpt2_bpe for the full text/code path."
    )


__all__ = ["ByteFallbackTokenizer", "GPT2BPETokenizer", "TextTokenizer", "build_text_tokenizer"]
