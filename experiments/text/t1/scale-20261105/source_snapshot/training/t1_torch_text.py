"""PyTorch/MPS Sprint T1 text/code training runner.

This runner is the non-pilot path for T1 text/code experiments.  It keeps the
existing NumPy runner unchanged for dependency-light smoke tests, while adding a
registered GPT-2-BPE + prepared-corpus workflow that can run GPT-2-size models on
Apple Silicon/MPS when PyTorch is available.

Scope boundaries:
    * text/code only (`text_code` modality id 1),
    * next-token cross-entropy only,
    * no audio, visual/video, policy, hallucination, or grounding claims,
    * all results must be interpreted through registered loss/PPL/throughput
      metrics and explicit train-token budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data.text_code import validate_t1_text_only_data_config
from data.tokenizer import TextTokenizer, build_text_tokenizer
from utils.config import config_hash, load_config, write_config
from utils.experiment_registry import make_t1_text_registry_entry, write_registry_entry


try:  # Optional dependency by design; dependency-light tests can still import other T1 modules.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - exercised only without optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


ARTIFACT_ROOT = Path("experiments/text/t1")


def _require_torch() -> None:
    if torch is None or nn is None or F is None:  # pragma: no cover - optional dependency guard
        raise ImportError("training.t1_torch_text requires PyTorch; install the T1 full-stack dependency") from _TORCH_IMPORT_ERROR


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_artifact_path(raw_path: str) -> Path:
    """Validate T1 artifact paths before writing model artifacts."""

    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("T1 artifact paths must be relative")
    repo_root = Path.cwd().resolve()
    artifact_root = (repo_root / ARTIFACT_ROOT).resolve()
    resolved = (repo_root / path).resolve()
    if not _is_relative_to(resolved, artifact_root):
        raise ValueError(f"T1 artifacts must be written under {ARTIFACT_ROOT}")
    if resolved.exists() and resolved.is_symlink():
        raise ValueError("T1 artifact path must not be a symlink")
    return path


def _registry_paths(config: Mapping[str, Any], experiment_id: str) -> dict[str, Path]:
    registry = config.get("registry", {}) if isinstance(config.get("registry", {}), Mapping) else {}
    return {
        "artifact_dir": _safe_artifact_path(str(registry.get("artifact_dir", f"experiments/text/t1/{experiment_id}"))),
        "registry": _safe_artifact_path(str(registry.get("path", f"experiments/text/t1/{experiment_id}/registry.json"))),
        "metrics": _safe_artifact_path(str(registry.get("metrics_path", f"experiments/text/t1/{experiment_id}/metrics.json"))),
        "samples": _safe_artifact_path(str(registry.get("samples_path", f"experiments/text/t1/{experiment_id}/samples.json"))),
        "report": _safe_artifact_path(str(registry.get("report_path", f"experiments/text/t1/{experiment_id}/report.md"))),
        "checkpoint": _safe_artifact_path(str(registry.get("checkpoint_path", f"experiments/text/t1/{experiment_id}/checkpoint.pt"))),
        "config_copy": _safe_artifact_path(str(registry.get("config_copy_path", f"experiments/text/t1/{experiment_id}/config.json"))),
    }


def _write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _write_text(payload: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _max_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _mps_memory_mb() -> float | None:
    if torch is None or not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        return None
    try:
        return float(torch.mps.current_allocated_memory() / (1024.0 * 1024.0))
    except Exception:  # pragma: no cover - backend-dependent
        return None


def _device_from_config(config: Mapping[str, Any]) -> "torch.device":
    _require_torch()
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    requested = str(runtime.get("device", "auto"))
    if requested == "auto":
        if torch.backends.mps.is_available():
            requested = "mps"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"
    return torch.device(requested)


def _runtime_seed(config: Mapping[str, Any]) -> int:
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    model = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    return int(runtime.get("seed", model.get("seed", 0)))


@dataclass(frozen=True)
class PreparedCorpus:
    """Memory-mapped token streams for one prepared T1 corpus.

    Shape contract:
        Each split is a one-dimensional ``np.ndarray`` of token IDs.  Batches are
        converted to ``torch.LongTensor[B,T]`` inputs and targets.
    """

    train_tokens: np.ndarray
    validation_tokens: np.ndarray
    test_tokens: np.ndarray | None
    dataset_card: Mapping[str, Any]
    dataset_card_path: str


def _load_prepared_corpus(data_cfg: Mapping[str, Any]) -> PreparedCorpus:
    validate_t1_text_only_data_config(data_cfg)
    raw_corpus_dir = str(data_cfg.get("prepared_corpus_dir", ""))
    if not raw_corpus_dir:
        raise ValueError("PyTorch T1 runner requires data.prepared_corpus_dir from training.t1_prepare_text_code")
    corpus_dir = Path(raw_corpus_dir)
    if corpus_dir.is_absolute():
        raise ValueError("data.prepared_corpus_dir must be relative")
    card_path = corpus_dir / "dataset_card.json"
    if not card_path.exists():
        raise FileNotFoundError(f"Missing prepared corpus dataset card: {card_path}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    token_files = card.get("token_files", {}) if isinstance(card.get("token_files", {}), Mapping) else {}
    train_path = Path(str(token_files.get("train", {}).get("path", corpus_dir / "train.tokens.npy")))
    validation_path = Path(str(token_files.get("validation", {}).get("path", corpus_dir / "validation.tokens.npy")))
    test_path = Path(str(token_files.get("test", {}).get("path", corpus_dir / "test.tokens.npy")))
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Prepared corpus must include train and validation token files")
    train = np.load(train_path, mmap_mode="r")
    validation = np.load(validation_path, mmap_mode="r")
    test = np.load(test_path, mmap_mode="r") if test_path.exists() else None
    if int(train.size) < 2 or int(validation.size) < 2:
        raise ValueError("Prepared corpus train/validation splits must each contain at least two tokens")
    return PreparedCorpus(train_tokens=train, validation_tokens=validation, test_tokens=test, dataset_card=card, dataset_card_path=str(card_path))


class TokenStreamBatcher:
    """Deterministic token-window sampler for prepared T1 streams.

    Shape contract:
        ``batch(step)`` returns ``(input_ids, target_ids)`` with shape
        ``LongTensor[B,T]`` on the configured torch device.
    """

    def __init__(self, tokens: np.ndarray, *, sequence_length: int, batch_size: int, seed: int, device: "torch.device") -> None:
        _require_torch()
        self.tokens = tokens
        self.sequence_length = int(sequence_length)
        self.batch_size = int(batch_size)
        self.device = device
        self.rng = np.random.default_rng(int(seed))
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if int(tokens.size) <= self.sequence_length + 1:
            raise ValueError(f"Token stream has {tokens.size} tokens, not enough for sequence_length={sequence_length}")

    @property
    def token_count(self) -> int:
        return int(self.tokens.size)

    def batch(self, *, step: int | None = None, sequential: bool = False) -> tuple["torch.Tensor", "torch.Tensor"]:
        limit = int(self.tokens.size) - self.sequence_length - 1
        if sequential:
            base = ((0 if step is None else int(step)) * self.batch_size * self.sequence_length) % max(1, limit)
            starts = (base + np.arange(self.batch_size) * self.sequence_length) % max(1, limit)
        else:
            starts = self.rng.integers(0, max(1, limit), size=self.batch_size, endpoint=False)
        rows = np.stack([np.asarray(self.tokens[int(start) : int(start) + self.sequence_length + 1], dtype=np.int64) for start in starts])
        batch = torch.from_numpy(rows).to(self.device, dtype=torch.long)
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


class CausalSelfAttention(nn.Module):
    """GPT-style causal self-attention.

    Shape contract:
        Input/output ``x`` is ``FloatTensor[B,T,D]``.  The module never attends
        to future positions because ``scaled_dot_product_attention`` is called
        with ``is_causal=True``.
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_embd = int(n_embd)
        self.n_head = int(n_head)
        self.head_dim = int(n_embd // n_head)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.dropout = float(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        bsz, seq_len, _ = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_embd)
        return self.c_proj(out)


class MLP(nn.Module):
    """Transformer feed-forward block preserving ``FloatTensor[B,T,D]``."""

    def __init__(self, n_embd: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(n_embd * mlp_ratio)
        self.fc = nn.Linear(n_embd, hidden)
        self.proj = nn.Linear(hidden, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class GPTBlock(nn.Module):
    """Causal GPT block preserving ``FloatTensor[B,T,D]``."""

    def __init__(self, n_embd: int, n_head: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout=dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TorchGPT2LM(nn.Module):
    """GPT-2-small-style causal decoder language model.

    Shape contract:
        ``input_ids``: ``LongTensor[B,T]``.
        Returns logits ``FloatTensor[B,T,V]`` over the text/code BPE vocabulary.
    """

    def __init__(self, *, vocab_size: int, context_length: int, n_layer: int, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(context_length, n_embd)
        self.blocks = nn.ModuleList([GPTBlock(n_embd, n_head, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        bsz, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError(f"sequence length {seq_len} exceeds context_length {self.context_length}")
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return F.linear(x, self.token_embedding.weight)


class CausalDepthwiseConv(nn.Module):
    """Causal depthwise temporal mixer preserving ``FloatTensor[B,T,D]``."""

    def __init__(self, n_embd: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(n_embd, n_embd, kernel_size=self.kernel_size, groups=n_embd)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        y = x.transpose(1, 2)
        y = F.pad(y, (self.kernel_size - 1, 0))
        y = self.conv(y)
        return y.transpose(1, 2)


class CausalSpectralMixer(nn.Module):
    """Causal frequency-parameterized depthwise temporal mixer.

    The kernel is parameterized in the frequency domain and transformed with
    ``irfft`` before causal convolution.  This keeps T1 language-model training
    free of future-token leakage while preserving an independently disableable
    spectral path for the text guardrail run.

    Shape contract:
        Input/output ``x`` is ``FloatTensor[B,T,D]``.
    """

    def __init__(self, n_embd: int, kernel_size: int) -> None:
        super().__init__()
        self.n_embd = int(n_embd)
        self.kernel_size = int(kernel_size)
        freq_bins = self.kernel_size // 2 + 1
        self.real = nn.Parameter(torch.randn(n_embd, freq_bins) * 0.01)
        self.imag = nn.Parameter(torch.randn(n_embd, freq_bins) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        freq = torch.complex(self.real, self.imag)
        kernel = torch.fft.irfft(freq, n=self.kernel_size, dim=-1).to(dtype=x.dtype)
        y = x.transpose(1, 2)
        y = F.pad(y, (self.kernel_size - 1, 0))
        y = F.conv1d(y, kernel.unsqueeze(1), bias=self.bias.to(dtype=x.dtype), groups=self.n_embd)
        return y.transpose(1, 2)


class SLWMTextBlock(nn.Module):
    """Causal text-only SLWM processor block for the T1 guardrail.

    Shape contract:
        Input/output ``z`` is ``FloatTensor[B,T,D]``.  All temporal branches are
        causal in T1 so validation loss cannot be improved by future-token
        leakage.
    """

    def __init__(
        self,
        *,
        n_embd: int,
        n_head: int,
        use_local_temporal_mixer: bool,
        use_spectral_mixer: bool,
        use_long_conv: bool,
        use_attention_binding: bool,
        use_gated_mlp: bool,
        local_kernel_size: int,
        spectral_kernel_size: int,
        long_kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.use_local_temporal_mixer = bool(use_local_temporal_mixer)
        self.use_spectral_mixer = bool(use_spectral_mixer)
        self.use_long_conv = bool(use_long_conv)
        self.use_attention_binding = bool(use_attention_binding)
        self.use_gated_mlp = bool(use_gated_mlp)
        self.norm_local = nn.LayerNorm(n_embd)
        self.local = CausalDepthwiseConv(n_embd, local_kernel_size) if self.use_local_temporal_mixer else None
        self.norm_spectral = nn.LayerNorm(n_embd)
        self.spectral = CausalSpectralMixer(n_embd, spectral_kernel_size) if self.use_spectral_mixer else None
        self.norm_long = nn.LayerNorm(n_embd)
        self.long_conv = CausalDepthwiseConv(n_embd, long_kernel_size) if self.use_long_conv else None
        self.norm_attn = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout=dropout) if self.use_attention_binding else None
        self.norm_mlp = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, mlp_ratio=4, dropout=dropout) if self.use_gated_mlp else None

    def forward(self, z: "torch.Tensor") -> "torch.Tensor":
        if self.local is not None:
            z = z + self.local(self.norm_local(z))
        if self.spectral is not None:
            z = z + self.spectral(self.norm_spectral(z))
        if self.long_conv is not None:
            z = z + self.long_conv(self.norm_long(z))
        if self.attn is not None:
            z = z + self.attn(self.norm_attn(z))
        if self.mlp is not None:
            z = z + self.mlp(self.norm_mlp(z))
        return z


class TorchSLWMTextLM(nn.Module):
    """Text-only SLWM adapter → causal processor → text head model.

    Shape contract:
        ``input_ids``: ``LongTensor[B,T]`` text/code BPE tokens.
        Returns logits ``FloatTensor[B,T,V]``.  The text head is tied to the text
        adapter embedding for strict GPT-2-style parameter accounting.
    """

    def __init__(self, *, vocab_size: int, context_length: int, n_layer: int, n_embd: int, n_head: int, flags: Mapping[str, Any], dropout: float = 0.0) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(context_length, n_embd)
        self.blocks = nn.ModuleList(
            [
                SLWMTextBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    use_local_temporal_mixer=bool(flags.get("use_local_temporal_mixer", True)),
                    use_spectral_mixer=bool(flags.get("use_spectral_mixer", True)),
                    use_long_conv=bool(flags.get("use_long_conv", True)),
                    use_attention_binding=bool(flags.get("use_attention_binding", True)),
                    use_gated_mlp=bool(flags.get("use_gated_mlp", True)),
                    local_kernel_size=int(flags.get("local_kernel_size", 3)),
                    spectral_kernel_size=int(flags.get("spectral_kernel_size", 31)),
                    long_kernel_size=int(flags.get("long_kernel_size", 31)),
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        bsz, seq_len = input_ids.shape
        if seq_len > self.context_length:
            raise ValueError(f"sequence length {seq_len} exceeds context_length {self.context_length}")
        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        z = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            z = block(z)
        z = self.ln_f(z)
        return F.linear(z, self.token_embedding.weight)


def _model_variant(config: Mapping[str, Any]) -> str:
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    return str(model_cfg.get("variant", ""))


def _build_model(config: Mapping[str, Any], tokenizer: TextTokenizer) -> tuple[nn.Module, str]:
    _require_torch()
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    variant = _model_variant(config)
    n_layer = int(model_cfg.get("n_layer", model_cfg.get("layers", 12)))
    n_embd = int(model_cfg.get("n_embd", model_cfg.get("latent_dim", 768)))
    n_head = int(model_cfg.get("n_head", model_cfg.get("attention_heads", 12)))
    context_length = int(model_cfg.get("context_length", model_cfg.get("latent_length", 1024)))
    dropout = float(model_cfg.get("dropout", 0.0))
    if variant == "gpt2_baseline":
        return TorchGPT2LM(vocab_size=tokenizer.vocab_size, context_length=context_length, n_layer=n_layer, n_embd=n_embd, n_head=n_head, dropout=dropout), str(model_cfg.get("name", "gpt2_torch_t1_text"))
    if variant in {"slwm_text_only", "slwm_text_only_no_spectral", "slwm_ablation"}:
        flags = model_cfg.get("architecture_flags", {}) if isinstance(model_cfg.get("architecture_flags", {}), Mapping) else {}
        if variant == "slwm_text_only_no_spectral":
            flags = dict(flags)
            flags["use_spectral_mixer"] = False
        return TorchSLWMTextLM(vocab_size=tokenizer.vocab_size, context_length=context_length, n_layer=n_layer, n_embd=n_embd, n_head=n_head, flags=flags, dropout=dropout), str(model_cfg.get("name", "slwm_torch_t1_text"))
    raise ValueError(f"Unsupported PyTorch T1 model variant {variant!r}")


def _module_parameter_counts(model: nn.Module, variant: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        n = int(parameter.numel())
        if name.startswith("token_embedding") or name.startswith("position_embedding"):
            key = "adapters" if variant.startswith("slwm") else "embeddings"
        elif name.startswith("blocks") or name.startswith("ln_f"):
            key = "processor" if variant.startswith("slwm") else "transformer_blocks"
        else:
            key = "heads"
        counts[key] = counts.get(key, 0) + n
    if variant.startswith("slwm"):
        counts.setdefault("policy", 0)
        counts.setdefault("text_code_adapter", counts.get("adapters", 0))
        counts.setdefault("inactive_audio_adapter", 0)
        counts.setdefault("inactive_visual_video_adapter", 0)
        counts.setdefault("inactive_adapter_parameters", 0)
    else:
        counts.setdefault("separate_lm_head", 0)
    counts["total"] = sum(int(parameter.numel()) for parameter in model.parameters())
    return counts


def _loss_for_batch(model: nn.Module, input_ids: "torch.Tensor", target_ids: "torch.Tensor") -> "torch.Tensor":
    logits = model(input_ids)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))


@torch.no_grad() if torch is not None else (lambda fn: fn)
def _evaluate_loss(model: nn.Module, batcher: TokenStreamBatcher, *, batches: int) -> float:
    model.eval()
    losses: list[float] = []
    for step in range(max(1, int(batches))):
        input_ids, target_ids = batcher.batch(step=step, sequential=True)
        loss = _loss_for_batch(model, input_ids, target_ids)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _learning_rate(step: int, *, base_lr: float, warmup_steps: int, total_steps: int, schedule: str) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if schedule == "cosine":
        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr


def _apply_lr(optimizer: "torch.optim.Optimizer", lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


@torch.no_grad() if torch is not None else (lambda fn: fn)
def _generate_samples(*, model: nn.Module, tokenizer: TextTokenizer, config: Mapping[str, Any], device: "torch.device") -> list[dict[str, Any]]:
    model.eval()
    gen_cfg = config.get("generation", {}) if isinstance(config.get("generation", {}), Mapping) else {}
    prompts = gen_cfg.get("prompts", ["The text model"])
    if not isinstance(prompts, list) or not prompts:
        prompts = ["The text model"]
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
    temperature = float(gen_cfg.get("temperature", 0.0))
    top_k = gen_cfg.get("top_k")
    top_p = gen_cfg.get("top_p")
    stop_on_eos = bool(gen_cfg.get("stop_on_eos", True))
    seed = int(gen_cfg.get("seed", _runtime_seed(config) + 10_000))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    context_length = int(config.get("model", {}).get("context_length", config.get("training", {}).get("sequence_length", 1024)))
    samples: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_text = str(prompt)
        token_ids = tokenizer.encode(prompt_text, add_eos=False) or [tokenizer.eos_token_id]
        generated: list[int] = []
        for _ in range(max_new_tokens):
            context = token_ids[-context_length:]
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(input_ids)[0, -1, :].float().detach().cpu()
            if temperature <= 0.0:
                next_token = int(torch.argmax(logits).item())
            else:
                scaled = logits / max(temperature, 1e-8)
                if top_k is not None and int(top_k) > 0 and int(top_k) < scaled.numel():
                    keep = torch.topk(scaled, int(top_k)).indices
                    mask = torch.full_like(scaled, -1e12)
                    mask[keep] = scaled[keep]
                    scaled = mask
                probs = torch.softmax(scaled, dim=-1)
                if top_p is not None and 0.0 < float(top_p) < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=0)
                    keep_count = int(torch.searchsorted(cumulative, torch.tensor(float(top_p))).item()) + 1
                    keep_idx = sorted_idx[: max(1, keep_count)]
                    filtered = torch.zeros_like(probs)
                    filtered[keep_idx] = probs[keep_idx]
                    probs = filtered / torch.clamp(filtered.sum(), min=1e-12)
                next_token = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
            token_ids.append(next_token)
            generated.append(next_token)
            if stop_on_eos and next_token == tokenizer.eos_token_id:
                break
        samples.append(
            {
                "prompt": prompt_text,
                "prompt_token_count": len(tokenizer.encode(prompt_text, add_eos=False)),
                "generated_token_ids": generated,
                "generated_text": tokenizer.decode(generated),
                "full_text": tokenizer.decode(token_ids),
                "decoding_settings": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "seed": seed,
                    "stop_on_eos": stop_on_eos,
                },
            }
        )
    return samples


def _baseline_text_loss_from_refs(config: Mapping[str, Any]) -> tuple[float | None, str | None]:
    refs = config.get("baselines_compared", [])
    if not isinstance(refs, list):
        return None, None
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        experiment_id = str(ref.get("experiment_id", ""))
        if not experiment_id:
            continue
        metrics_path = Path("experiments/text/t1") / experiment_id / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if metrics.get("model_variant") == "gpt2_baseline" and metrics.get("validation_loss") is not None:
            return float(metrics["validation_loss"]), experiment_id
    return None, None


def _report_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        f"# Sprint T1 PyTorch Report — {metrics['experiment_id']}",
        "",
        f"- Model variant: `{metrics['model_variant']}`",
        f"- Tokenizer: `{metrics['tokenizer']['effective_type']}`",
        f"- Prepared corpus: `{metrics.get('prepared_corpus_dir')}`",
        f"- Validation loss: `{metrics['validation_loss']}`",
        f"- Validation perplexity: `{metrics['validation_perplexity']}`",
        f"- Validation tokens evaluated: `{metrics.get('validation_tokens_evaluated')}` of `{metrics.get('validation_tokens_available')}` available",
        f"- Throughput tokens/s: `{metrics['throughput_tokens_per_second']}`",
        f"- Max RSS memory MB: `{metrics['max_memory_mb']}`",
        f"- MPS allocated memory MB: `{metrics.get('mps_memory_mb')}`",
        f"- Checkpoint: `{metrics['checkpoint_path']}`",
        "",
        "## Scope",
        "Text/code only. No audio or visual data was loaded or trained in this sprint run.",
        "",
        "## Claim limits",
        metrics.get("claim_language_allowed", "Only registered T1 metrics may be reported."),
        "",
    ]
    return "\n".join(lines)


def run_t1_torch_text_training(config_path: str | Path) -> dict[str, Any]:
    """Run one registered PyTorch/MPS Sprint T1 text/code training job."""

    _require_torch()
    path = Path(config_path)
    config = load_config(path)
    seed = _runtime_seed(config)
    torch.manual_seed(seed)
    np.random.seed(seed)
    variant = _model_variant(config)
    registry_cfg = config.get("registry", {}) if isinstance(config.get("registry", {}), Mapping) else {}
    experiment_id = str(registry_cfg.get("experiment_id", "EXP-T1-101"))
    paths = _registry_paths(config, experiment_id)
    tokenizer = build_text_tokenizer(config.get("tokenizer", config.get("model", {})))
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}
    corpus = _load_prepared_corpus(data_cfg)
    tokenizer_metadata = tokenizer.metadata()
    corpus_tokenizer = corpus.dataset_card.get("tokenizer", {}) if isinstance(corpus.dataset_card.get("tokenizer", {}), Mapping) else {}
    if corpus_tokenizer:
        for key in ("effective_type", "vocab_size", "eos_token_id"):
            if corpus_tokenizer.get(key) != tokenizer_metadata.get(key):
                raise ValueError(
                    f"Configured tokenizer {key}={tokenizer_metadata.get(key)!r} does not match "
                    f"prepared corpus tokenizer {key}={corpus_tokenizer.get(key)!r}"
                )
    train_cfg = config.get("training", {}) if isinstance(config.get("training", {}), Mapping) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    device = _device_from_config(config)
    sequence_length = int(train_cfg.get("sequence_length", model_cfg.get("context_length", 1024)))
    batch_size = int(train_cfg.get("batch_size", 1))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    steps = int(train_cfg.get("steps", 100))
    val_batches = int(train_cfg.get("validation_batches", 8))
    learning_rate = float(train_cfg.get("learning_rate", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.1))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    schedule = str(train_cfg.get("learning_rate_schedule", "cosine"))
    grad_clip = train_cfg.get("grad_clip_norm", 1.0)
    grad_clip_norm = None if grad_clip is None else float(grad_clip)

    model, model_name = _build_model(config, tokenizer)
    model.to(device)
    module_counts = _module_parameter_counts(model, variant)
    parameter_count = int(module_counts["total"])
    if sequence_length > int(model_cfg.get("context_length", model_cfg.get("latent_length", 1024))):
        raise ValueError("training.sequence_length must not exceed model.context_length")
    train_batcher = TokenStreamBatcher(corpus.train_tokens, sequence_length=sequence_length, batch_size=batch_size, seed=seed, device=device)
    validation_batcher = TokenStreamBatcher(corpus.validation_tokens, sequence_length=sequence_length, batch_size=batch_size, seed=seed + 1, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    initial_train_loss = _evaluate_loss(model, train_batcher, batches=min(2, val_batches))
    initial_validation_loss = _evaluate_loss(model, validation_batcher, batches=val_batches)
    losses: list[float] = [float(initial_train_loss)]
    grad_norms: list[float] = []
    nan_or_inf = not (math.isfinite(initial_train_loss) and math.isfinite(initial_validation_loss))
    loss_explosion = False
    tokens_seen = 0
    start = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(steps):
        lr = _learning_rate(step, base_lr=learning_rate, warmup_steps=warmup_steps, total_steps=steps, schedule=schedule)
        _apply_lr(optimizer, lr)
        accumulated_loss = 0.0
        for _ in range(max(1, grad_accum)):
            input_ids, target_ids = train_batcher.batch()
            loss = _loss_for_batch(model, input_ids, target_ids) / max(1, grad_accum)
            if not torch.isfinite(loss).all():
                nan_or_inf = True
                break
            loss.backward()
            accumulated_loss += float(loss.detach().cpu())
            tokens_seen += int(input_ids.numel())
        if nan_or_inf:
            break
        if grad_clip_norm is not None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm).detach().cpu())
        else:
            total = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total += float(torch.sum(param.grad.detach().float() ** 2).cpu())
            grad_norm = math.sqrt(total)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step_loss = float(accumulated_loss)
        losses.append(step_loss)
        grad_norms.append(grad_norm)
        if step_loss > max(initial_train_loss, 1e-12) * 10.0:
            loss_explosion = True
    wall_clock = time.perf_counter() - start
    train_loss = _evaluate_loss(model, train_batcher, batches=min(4, val_batches))
    validation_loss = _evaluate_loss(model, validation_batcher, batches=val_batches)
    validation_tokens_evaluated = int(max(1, val_batches) * batch_size * sequence_length)
    validation_perplexity = float(math.exp(min(validation_loss, 50.0)))
    samples = _generate_samples(model=model, tokenizer=tokenizer, config=config, device=device)
    checkpoint_metadata = {
        "experiment_id": experiment_id,
        "model_variant": variant,
        "config_hash": config_hash(config),
        "parameter_count": parameter_count,
        "tokenizer": tokenizer.metadata(),
        "prepared_corpus": corpus.dataset_card_path,
    }
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "metadata": checkpoint_metadata}, paths["checkpoint"])
    write_config(config, paths["config_copy"])
    throughput = float(tokens_seen / wall_clock) if wall_clock > 0.0 else None

    registry_datasets = []
    split_counts = corpus.dataset_card.get("split_counts", {}) if isinstance(corpus.dataset_card.get("split_counts", {}), Mapping) else {}
    source_names = [
        str(source.get("path", source.get("name")))
        for source in corpus.dataset_card.get("sources", [])
        if isinstance(source, Mapping) and source.get("status") == "loaded"
    ]
    source_summary = ", ".join(source_names)
    for split_name in ("train", "validation", "test"):
        split_row = split_counts.get(split_name, {}) if isinstance(split_counts.get(split_name, {}), Mapping) else {}
        registry_datasets.append(
            {
                "name": "prepared_t1_text_code_mix",
                "version_or_snapshot": corpus.dataset_card.get("manifest_sha256", "recorded_in_dataset_card"),
                "split": split_name,
                "sample_count": split_row.get("documents"),
                "tokens": split_row.get("tokens"),
                "audio_hours": None,
                "video_hours": None,
                "license_notes": f"sources: {source_summary}; see prepared corpus dataset_card sources and manifest",
                "leakage_checks": "document-level stable hash split; manifest SHA recorded",
            }
        )
    token_files = corpus.dataset_card.get("token_files", {}) if isinstance(corpus.dataset_card.get("token_files", {}), Mapping) else {}
    split_digests = {split: token_files.get(split, {}).get("sha256") for split in ("train", "validation", "test") if isinstance(token_files.get(split, {}), Mapping)}
    claim_scope = str(train_cfg.get("claim_scope", "gpt2_bpe_prepared_corpus_limited_training_budget"))
    metrics: dict[str, Any] = {
        "experiment_id": experiment_id,
        "sprint": "T1",
        "model_variant": variant,
        "model_name": model_name,
        "initial_train_loss": float(initial_train_loss),
        "initial_validation_loss": float(initial_validation_loss),
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "validation_perplexity": validation_perplexity,
        "losses": losses,
        "grad_norms": grad_norms,
        "nan_or_inf": bool(nan_or_inf),
        "loss_explosion": bool(loss_explosion),
        "parameter_count": parameter_count,
        "module_parameter_counts": dict(module_counts),
        "tokenizer": tokenizer_metadata,
        "split_digests": split_digests,
        "registry_datasets": registry_datasets,
        "prepared_corpus_dir": str(data_cfg.get("prepared_corpus_dir")),
        "prepared_corpus_dataset_card": corpus.dataset_card_path,
        "prepared_corpus_manifest_sha256": corpus.dataset_card.get("manifest_sha256"),
        "train_tokens_available": int(corpus.train_tokens.size),
        "validation_tokens_available": int(corpus.validation_tokens.size),
        "validation_batches": int(val_batches),
        "validation_tokens_evaluated": validation_tokens_evaluated,
        "train_tokens_seen": int(tokens_seen),
        "throughput_tokens_per_second": throughput,
        "wall_clock_time_seconds": float(wall_clock),
        "max_memory_mb": float(_max_rss_mb()),
        "mps_memory_mb": _mps_memory_mb(),
        "hardware": f"torch:{torch.__version__}:{device}:{platform.platform()}",
        "config_hash": config_hash(config),
        "eval_script": "training/t1_torch_text.py",
        "eval_script_hash": _file_sha256(Path(__file__).resolve()),
        "checkpoint_path": str(paths["checkpoint"]),
        "config_copy_path": str(paths["config_copy"]),
        "samples_path": str(paths["samples"]),
        "claim_language_allowed": (
            "GPT-2-BPE prepared-corpus T1 run: report validation loss/PPL, samples, throughput, memory, and exact settings only. "
            f"Claim scope is {claim_scope!r}; do not claim converged GPT-2-scale quality unless the registered train-token budget supports it."
        ),
        "limitations": [
            "Prepared text/code corpus only; no audio or visual/video data used.",
            "Validation loss/PPL reflects the registered train-token budget and should not be compared to fully trained GPT-2 unless budgets match.",
            "No LAMBADA, HumanEval, MBPP, hallucination, grounding, or policy claim is supported by this T1 runner.",
        ],
        "result_summary": f"Sprint T1 {variant} PyTorch text/code run completed with validation loss {validation_loss:.6f}.",
        "hypothesis_decision": "guardrail_pass" if variant == "gpt2_baseline" else "untested",
        "next_allowed_step": "Run companion T1 variants on the same prepared corpus/tokenizer/optimizer family before updating G-R0-1.",
        "failure_modes_observed": [],
        "baselines_compared": config.get("baselines_compared", []),
    }
    tolerance = float(train_cfg.get("guardrail_tolerance_percent", 20.0))
    metrics["guardrail_tolerance_percent"] = tolerance
    if variant == "gpt2_baseline":
        metrics["gpt2_baseline_text_validation_loss"] = float(validation_loss)
        metrics["text_loss_relative_delta_percent"] = 0.0
        metrics["guardrail_status"] = "baseline_anchor"
    else:
        baseline_loss, baseline_experiment_id = _baseline_text_loss_from_refs(config)
        metrics["gpt2_baseline_text_validation_loss"] = baseline_loss
        metrics["gpt2_baseline_experiment_id"] = baseline_experiment_id
        if baseline_loss is not None and baseline_loss > 0.0:
            delta = 100.0 * (float(validation_loss) - baseline_loss) / baseline_loss
            metrics["text_loss_relative_delta_percent"] = float(delta)
            guardrail_pass = delta <= tolerance
            metrics["guardrail_status"] = "guardrail_pass" if guardrail_pass else "guardrail_fail"
            metrics["hypothesis_decision"] = metrics["guardrail_status"]
            if not guardrail_pass:
                metrics["failure_modes_observed"].append("text_loss_exceeds_registered_tolerance")
                metrics["result_summary"] += f" Text tradeoff recorded: {delta:.2f}% loss delta vs {baseline_experiment_id}."
                metrics["next_allowed_step"] = "Record the T1 text tradeoff; do not claim text improvement unless a matched rerun beats the GPT-2 baseline."
        else:
            metrics["text_loss_relative_delta_percent"] = None
            metrics["guardrail_status"] = "baseline_missing"
    if nan_or_inf:
        metrics["failure_modes_observed"].append("nan_or_inf")
    if loss_explosion:
        metrics["failure_modes_observed"].append("loss_explosion")

    metrics["registry_path"] = str(paths["registry"])
    metrics["metrics_path"] = str(paths["metrics"])
    metrics["report_path"] = str(paths["report"])
    _write_json({"samples": samples, "decoding_settings": config.get("generation", {})}, paths["samples"])
    _write_json(metrics, paths["metrics"])
    entry = make_t1_text_registry_entry(
        experiment_id=experiment_id,
        config_path=str(path),
        config=config,
        metrics=metrics,
        model_name=model_name,
        model_variant=variant,
        parameter_count=parameter_count,
        module_parameter_counts=module_counts,
        training_steps=steps,
        train_tokens=tokens_seen,
        checkpoint_path=str(paths["checkpoint"]),
        working_tree_state="dirty",
    )
    write_registry_entry(entry, paths["registry"])
    _write_text(_report_markdown(metrics), paths["report"])
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PyTorch/MPS Sprint T1 text/code training")
    parser.add_argument("--config", required=True, help="Path to a T1 PyTorch JSON config")
    args = parser.parse_args(argv)
    metrics = run_t1_torch_text_training(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())


__all__ = ["run_t1_torch_text_training"]
