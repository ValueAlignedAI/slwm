"""Dependency-light text/code tokenizers for Sprint T1.

Sprint T1 requires the GPT-2 BPE tokenizer for fair text/code baseline
comparisons when a full training stack is available.  The current repository is
still dependency-light, so the runnable tiny pilot uses a deterministic byte
fallback tokenizer and records that fallback explicitly in configs/registries.
Both GPT-2 and SLWM variants consume the same tokenizer object, preserving the
same-tokenizer guardrail for pilot/smoke evidence.
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


def build_text_tokenizer(config: Mapping[str, Any]) -> TextTokenizer:
    """Build the tokenizer requested by a T1 config.

    Supported dependency-free tokenizer types:
        - ``byte_fallback_t1_smoke`` / ``byte_fallback``

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
    raise ValueError(
        f"Unsupported tokenizer type {tokenizer_type!r}. "
        "Dependency-free Sprint T1 supports byte_fallback_t1_smoke; use a documented external stack for GPT-2 BPE."
    )


__all__ = ["ByteFallbackTokenizer", "TextTokenizer", "build_text_tokenizer"]
