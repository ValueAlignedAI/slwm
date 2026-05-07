"""Parameter accounting helpers for Sprint I1 baselines.

The functions in this module are formula-based so the repository can report
GPT-2-small-style parameter counts without instantiating a 124M-parameter model
during smoke tests. Counts are exact for the baseline shapes/configs represented
here and are used by configs, tests, smoke logs, and registry entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class GPT2ParameterBreakdown:
    """Exact trainable parameter count for a GPT-2-style decoder.

    Shape contract:
        token ids: ``IntTensor[B,T]`` where ``T <= context_length``.
        hidden states: ``FloatTensor[B,T,D]`` where ``D = n_embd``.
        logits: ``FloatTensor[B,T,V]`` where ``V = vocab_size``.
    """

    token_embeddings: int
    position_embeddings: int
    transformer_blocks: int
    final_layer_norm: int
    lm_head: int

    @property
    def total(self) -> int:
        """Total trainable parameters for strict baseline accounting."""

        return self.token_embeddings + self.position_embeddings + self.transformer_blocks + self.final_layer_norm + self.lm_head

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-serializable count dictionary."""

        return {
            "token_embeddings": self.token_embeddings,
            "position_embeddings": self.position_embeddings,
            "transformer_blocks": self.transformer_blocks,
            "final_layer_norm": self.final_layer_norm,
            "lm_head": self.lm_head,
            "total": self.total,
        }


def gpt2_parameter_breakdown(
    *,
    vocab_size: int,
    context_length: int,
    n_layer: int,
    n_embd: int,
    intermediate_size: int | None = None,
    tie_embeddings: bool = True,
) -> GPT2ParameterBreakdown:
    """Compute exact GPT-2-style decoder parameter counts.

    The default with ``vocab_size=50257``, ``context_length=1024``,
    ``n_layer=12``, ``n_embd=768``, ``intermediate_size=3072``, and tied
    token/language-model embeddings is 124,439,808 parameters, matching the
    standard GPT-2 small accounting used as the SLWM reference baseline.
    """

    d_model = int(n_embd)
    d_ff = int(intermediate_size if intermediate_size is not None else 4 * d_model)
    token_embeddings = int(vocab_size) * d_model
    position_embeddings = int(context_length) * d_model

    # Per block: ln1 + qkv + attn output + ln2 + MLP up/down projections.
    layer_norms = 4 * d_model
    qkv = d_model * (3 * d_model) + (3 * d_model)
    attn_out = d_model * d_model + d_model
    mlp_up = d_model * d_ff + d_ff
    mlp_down = d_ff * d_model + d_model
    per_block = layer_norms + qkv + attn_out + mlp_up + mlp_down
    transformer_blocks = int(n_layer) * per_block
    final_layer_norm = 2 * d_model
    lm_head = int(vocab_size) * d_model + int(vocab_size) if not tie_embeddings else 0
    return GPT2ParameterBreakdown(
        token_embeddings=token_embeddings,
        position_embeddings=position_embeddings,
        transformer_blocks=transformer_blocks,
        final_layer_norm=final_layer_norm,
        lm_head=lm_head,
    )


def gpt2_module_counts_for_registry(breakdown: GPT2ParameterBreakdown) -> dict[str, int]:
    """Map GPT-2-style counts onto the shared registry module buckets."""

    embeddings = breakdown.token_embeddings + breakdown.position_embeddings
    return {
        "adapters": 0,
        "processor": breakdown.transformer_blocks + breakdown.final_layer_norm,
        "heads": breakdown.lm_head,
        "policy": 0,
        "decoders": embeddings,
        "embeddings": embeddings,
        "total": breakdown.total,
    }


@dataclass(frozen=True)
class MultimodalParameterBreakdown:
    """Parameter count for the vanilla multimodal Transformer baseline.

    Shape contract:
        text tokens: ``IntTensor[B,T_text]``.
        audio features: ``FloatTensor[B,T_audio,A]``.
        visual features: ``FloatTensor[B,T_visual,V]``.
        concatenated hidden field: ``FloatTensor[B,T_total,D]``.
        per-position logits: ``FloatTensor[B,T_total,target_vocab_size]``.
    """

    text_embeddings: int
    position_embeddings: int
    modality_embeddings: int
    audio_projection: int
    visual_projection: int
    transformer_blocks: int
    final_layer_norm: int
    output_head: int

    @property
    def total(self) -> int:
        """Total trainable parameters for strict baseline accounting."""

        return sum(self.as_dict(include_total=False).values())

    def as_dict(self, *, include_total: bool = True) -> dict[str, int]:
        """Return a JSON-serializable count dictionary."""

        values = {
            "text_embeddings": self.text_embeddings,
            "position_embeddings": self.position_embeddings,
            "modality_embeddings": self.modality_embeddings,
            "audio_projection": self.audio_projection,
            "visual_projection": self.visual_projection,
            "transformer_blocks": self.transformer_blocks,
            "final_layer_norm": self.final_layer_norm,
            "output_head": self.output_head,
        }
        if include_total:
            values["total"] = sum(values.values())
        return values


def vanilla_multimodal_parameter_breakdown(
    *,
    text_vocab_size: int,
    target_vocab_size: int,
    context_length: int,
    n_layer: int,
    n_embd: int,
    audio_feature_dim: int,
    visual_feature_dim: int,
    modality_count: int = 3,
    intermediate_size: int | None = None,
) -> MultimodalParameterBreakdown:
    """Compute vanilla multimodal Transformer baseline parameter counts."""

    d_model = int(n_embd)
    d_ff = int(intermediate_size if intermediate_size is not None else 4 * d_model)
    text_embeddings = int(text_vocab_size) * d_model
    position_embeddings = int(context_length) * d_model
    modality_embeddings = int(modality_count) * d_model
    audio_projection = int(audio_feature_dim) * d_model + d_model
    visual_projection = int(visual_feature_dim) * d_model + d_model

    layer_norms = 4 * d_model
    qkv = d_model * (3 * d_model) + (3 * d_model)
    attn_out = d_model * d_model + d_model
    mlp_up = d_model * d_ff + d_ff
    mlp_down = d_ff * d_model + d_model
    per_block = layer_norms + qkv + attn_out + mlp_up + mlp_down

    return MultimodalParameterBreakdown(
        text_embeddings=text_embeddings,
        position_embeddings=position_embeddings,
        modality_embeddings=modality_embeddings,
        audio_projection=audio_projection,
        visual_projection=visual_projection,
        transformer_blocks=int(n_layer) * per_block,
        final_layer_norm=2 * d_model,
        output_head=d_model * int(target_vocab_size) + int(target_vocab_size),
    )


def multimodal_module_counts_for_registry(breakdown: MultimodalParameterBreakdown) -> dict[str, int]:
    """Map vanilla multimodal counts onto shared registry buckets."""

    adapters = breakdown.audio_projection + breakdown.visual_projection
    decoders = breakdown.text_embeddings + breakdown.position_embeddings + breakdown.modality_embeddings
    processor = breakdown.transformer_blocks + breakdown.final_layer_norm
    return {
        "adapters": adapters,
        "processor": processor,
        "heads": breakdown.output_head,
        "policy": 0,
        "decoders": decoders,
        "embeddings": decoders,
        "total": breakdown.total,
    }


def count_numpy_parameters(parameters: Mapping[str, object]) -> int:
    """Count elements in a mapping of NumPy-like arrays or Parameter objects."""

    total = 0
    for value in parameters.values():
        array = getattr(value, "value", value)
        size = getattr(array, "size", None)
        if size is None:
            raise TypeError(f"Cannot count parameter object {type(value)!r}")
        total += int(size)
    return total


__all__ = [
    "GPT2ParameterBreakdown",
    "MultimodalParameterBreakdown",
    "count_numpy_parameters",
    "gpt2_module_counts_for_registry",
    "gpt2_parameter_breakdown",
    "multimodal_module_counts_for_registry",
    "vanilla_multimodal_parameter_breakdown",
]
