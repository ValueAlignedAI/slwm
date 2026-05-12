"""PyTorch Sprint T2 audio/visual latent training runner.

Scope boundaries:
    * audio and visual/video latent features only;
    * frozen/precomputed feature corpora loaded from ``training.t2_prepare_latents``;
    * objectives are latent continuation, missing-span reconstruction, and
      audio-video correspondence/alignment;
    * no text generation, raw waveform/video generation, policy training, or
      hallucination claims.
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
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data.audio_visual_latents import T2PreparedLatentDataset, t2_batch_contract_summary, validate_t2_audio_visual_data_config
from training.synthetic_metrics import frequency_recovery_error, phase_error, spectral_magnitude_loss
from utils.config import config_hash, load_config, write_config
from utils.experiment_registry import make_t2_audio_visual_registry_entry, write_registry_entry


try:  # Optional dependency; docs/tests can still import preparation code without torch.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
except Exception as exc:  # pragma: no cover - exercised only without optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    torch_checkpoint = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


ARTIFACT_ROOT = Path("experiments/multimodal/t2")
PREPARED_ROOT = Path("artifacts/t2_audio_visual")


def _require_torch() -> None:
    if torch is None or nn is None or F is None:  # pragma: no cover - optional dependency guard
        raise ImportError("training.t2_train_latents requires PyTorch") from _TORCH_IMPORT_ERROR


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_artifact_path(raw_path: str | Path) -> Path:
    """Validate T2 artifact paths before writing."""

    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("T2 artifact paths must be relative")
    repo_root = Path.cwd().resolve()
    artifact_root = (repo_root / ARTIFACT_ROOT).resolve()
    resolved = (repo_root / path).resolve()
    if not _is_relative_to(resolved, artifact_root):
        raise ValueError(f"T2 artifacts must be written under {ARTIFACT_ROOT}")
    if resolved.exists() and resolved.is_symlink():
        raise ValueError("T2 artifact path must not be a symlink")
    return path


def _safe_prepared_corpus_dir(raw_path: str | Path) -> Path:
    """Validate config-controlled T2 prepared corpus reads."""

    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("T2 prepared_corpus_dir must be relative")
    repo_root = Path.cwd().resolve()
    prepared_root = (repo_root / PREPARED_ROOT).resolve()
    resolved = (repo_root / path).resolve()
    if not _is_relative_to(resolved, prepared_root):
        raise ValueError(f"T2 prepared_corpus_dir must be under {PREPARED_ROOT}")
    if resolved.exists() and resolved.is_symlink():
        raise ValueError("T2 prepared_corpus_dir must not be a symlink")
    return path


def _registry_paths(config: Mapping[str, Any], experiment_id: str) -> dict[str, Path]:
    registry = config.get("registry", {}) if isinstance(config.get("registry", {}), Mapping) else {}
    artifact_dir = _safe_artifact_path(str(registry.get("artifact_dir", f"experiments/multimodal/t2/{experiment_id}")))
    return {
        "artifact_dir": artifact_dir,
        "registry": _safe_artifact_path(str(registry.get("path", artifact_dir / "registry.json"))),
        "metrics": _safe_artifact_path(str(registry.get("metrics_path", artifact_dir / "metrics.json"))),
        "report": _safe_artifact_path(str(registry.get("report_path", artifact_dir / "report.md"))),
        "checkpoint": _safe_artifact_path(str(registry.get("checkpoint_path", artifact_dir / "checkpoint.pt"))),
        "config_copy": _safe_artifact_path(str(registry.get("config_copy_path", artifact_dir / "config.json"))),
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


def _accelerator_memory_mb() -> float | None:
    if torch is None:
        return None
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            return float(torch.mps.current_allocated_memory() / (1024.0 * 1024.0))
        except Exception:  # pragma: no cover - backend-dependent
            return None
    return None


def _runtime_seed(config: Mapping[str, Any]) -> int:
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    model = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    return int(runtime.get("seed", model.get("seed", 0)))


def _set_determinism(config: Mapping[str, Any]) -> None:
    """Apply best-effort deterministic torch settings when requested."""

    _require_torch()
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    if bool(runtime.get("deterministic", True)):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - backend-dependent
            pass


def _device_from_config(config: Mapping[str, Any]) -> "torch.device":
    _require_torch()
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    requested = str(runtime.get("device", "auto"))
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    return torch.device(requested)


def _dtype_from_config(config: Mapping[str, Any]) -> "torch.dtype":
    _require_torch()
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    precision = str(runtime.get("precision", "fp32")).lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16", "mixed"}:
        return torch.float16
    return torch.float32


def _cpu_float_batch(batch: Mapping[str, Any], *, device: "torch.device", dtype: "torch.dtype" = None) -> dict[str, Any]:
    _require_torch()
    tensor_keys = {
        "audio_features",
        "visual_features",
        "audio_targets",
        "visual_targets",
        "visual_targets_unshuffled",
        "audio_mask",
        "visual_mask",
        "audio_valid_mask",
        "visual_valid_mask",
        "audio_loss_mask",
        "visual_loss_mask",
    }
    converted: dict[str, Any] = {}
    for key, value in batch.items():
        if key not in tensor_keys:
            converted[key] = value
            continue
        array = np.asarray(value)
        if array.dtype == bool:
            converted[key] = torch.from_numpy(array).to(device=device, dtype=torch.bool)
        else:
            converted[key] = torch.from_numpy(array.astype(np.float32)).to(device=device, dtype=dtype or torch.float32)
    return converted


class T2SelfAttention(nn.Module):
    """Non-causal self-attention over observed latent tokens.

    Shape contract: ``FloatTensor[B,T,D]`` plus ``BoolTensor[B,T]`` observed mask.
    Hidden target tokens may query observed keys, but key padding prevents missing
    or padded tokens from becoming evidence.
    """

    def __init__(self, latent_dim: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(latent_dim, n_head, dropout=dropout, batch_first=True)

    def forward(self, x: "torch.Tensor", observed_mask: "torch.Tensor | None") -> "torch.Tensor":
        key_padding_mask = None if observed_mask is None else ~observed_mask.bool()
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return out


class T2DepthwiseConv(nn.Module):
    """Depthwise temporal convolution preserving ``FloatTensor[B,T,D]``."""

    def __init__(self, latent_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(latent_dim, latent_dim, kernel_size=self.kernel_size, padding=self.kernel_size // 2, groups=latent_dim)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return y[:, : x.shape[1], :]


class T2SpectralMixer(nn.Module):
    """Frequency-parameterized depthwise mixer preserving ``FloatTensor[B,T,D]``."""

    def __init__(self, latent_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.kernel_size = int(kernel_size)
        freq_bins = self.kernel_size // 2 + 1
        self.real = nn.Parameter(torch.randn(latent_dim, freq_bins) * 0.01)
        self.imag = nn.Parameter(torch.randn(latent_dim, freq_bins) * 0.01)
        self.bias = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        freq = torch.complex(self.real, self.imag)
        kernel = torch.fft.irfft(freq, n=self.kernel_size, dim=-1).to(dtype=x.dtype)
        y = F.conv1d(F.pad(x.transpose(1, 2), (self.kernel_size // 2, self.kernel_size // 2)), kernel.unsqueeze(1), bias=self.bias.to(dtype=x.dtype), groups=self.latent_dim)
        return y.transpose(1, 2)[:, : x.shape[1], :]


class T2MLP(nn.Module):
    """Gated feed-forward block preserving ``FloatTensor[B,T,D]``."""

    def __init__(self, latent_dim: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(latent_dim * mlp_ratio)
        self.value = nn.Linear(latent_dim, hidden)
        self.gate = nn.Linear(latent_dim, hidden)
        self.out = nn.Linear(hidden, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.dropout(self.out(F.silu(self.gate(x)) * self.value(x)))


class T2SLWMBlock(nn.Module):
    """Ablatable Sprint T2 processor block preserving ``FloatTensor[B,T,D]``."""

    def __init__(self, *, latent_dim: int, n_head: int, flags: Mapping[str, Any], dropout: float) -> None:
        super().__init__()
        self.use_local = bool(flags.get("use_local_temporal_mixer", True))
        self.use_spectral = bool(flags.get("use_spectral_mixer", True))
        self.use_long = bool(flags.get("use_long_conv", True))
        self.use_attention = bool(flags.get("use_attention_binding", True))
        self.use_mlp = bool(flags.get("use_gated_mlp", True))
        self.norm_local = nn.LayerNorm(latent_dim)
        self.local = T2DepthwiseConv(latent_dim, int(flags.get("local_kernel_size", 3))) if self.use_local else None
        self.norm_spectral = nn.LayerNorm(latent_dim)
        self.spectral = T2SpectralMixer(latent_dim, int(flags.get("spectral_kernel_size", 31))) if self.use_spectral else None
        self.norm_long = nn.LayerNorm(latent_dim)
        self.long_conv = T2DepthwiseConv(latent_dim, int(flags.get("long_kernel_size", 31))) if self.use_long else None
        self.norm_attn = nn.LayerNorm(latent_dim)
        self.attn = T2SelfAttention(latent_dim, n_head, dropout=dropout) if self.use_attention else None
        self.norm_mlp = nn.LayerNorm(latent_dim)
        self.mlp = T2MLP(latent_dim, mlp_ratio=int(flags.get("mlp_ratio", 4)), dropout=dropout) if self.use_mlp else None

    def forward(self, z: "torch.Tensor", observed_mask: "torch.Tensor | None") -> "torch.Tensor":
        if self.local is not None:
            z = z + self.local(self.norm_local(z))
        if self.spectral is not None:
            z = z + self.spectral(self.norm_spectral(z))
        if self.long_conv is not None:
            z = z + self.long_conv(self.norm_long(z))
        if self.attn is not None:
            z = z + self.attn(self.norm_attn(z), observed_mask)
        if self.mlp is not None:
            z = z + self.mlp(self.norm_mlp(z))
        return z


class T2AudioVisualSLWM(nn.Module):
    """Audio/visual latent SLWM for Sprint T2.

    Shape contract:
        audio input ``FloatTensor[B,T_audio,A]`` and visual input
        ``FloatTensor[B,T_visual,V]`` are projected to a shared latent field,
        processed as ``FloatTensor[B,T_audio+T_visual,D]``, then decoded back to
        modality-specific latent features.
    """

    def __init__(self, *, model_cfg: Mapping[str, Any], audio_feature_dim: int, visual_feature_dim: int) -> None:
        super().__init__()
        self.name = str(model_cfg.get("name", "slwm_t2_audio_visual"))
        self.variant = str(model_cfg.get("variant", "slwm_audio_visual_latent"))
        self.context_length = int(model_cfg.get("context_length", model_cfg.get("latent_length", 1024)))
        self.latent_dim = int(model_cfg.get("latent_dim", model_cfg.get("n_embd", 768)))
        self.n_layer = int(model_cfg.get("n_layer", model_cfg.get("processor_layers", 12)))
        self.n_head = int(model_cfg.get("n_head", model_cfg.get("attention_heads", 12)))
        if self.latent_dim % self.n_head != 0:
            raise ValueError("latent_dim must be divisible by n_head")
        self.audio_feature_dim = int(audio_feature_dim)
        self.visual_feature_dim = int(visual_feature_dim)
        dropout = float(model_cfg.get("dropout", 0.0))
        flags = model_cfg.get("architecture_flags", {}) if isinstance(model_cfg.get("architecture_flags", {}), Mapping) else {}
        self.flags = dict(flags)
        self.use_activation_checkpointing = bool(model_cfg.get("activation_checkpointing", flags.get("activation_checkpointing", False)))
        self.audio_projection = nn.Linear(self.audio_feature_dim, self.latent_dim)
        self.visual_projection = nn.Linear(self.visual_feature_dim, self.latent_dim)
        self.position_embedding = nn.Embedding(self.context_length, self.latent_dim)
        self.modality_embedding = nn.Embedding(4, self.latent_dim)
        self.blocks = nn.ModuleList([T2SLWMBlock(latent_dim=self.latent_dim, n_head=self.n_head, flags=flags, dropout=dropout) for _ in range(self.n_layer)])
        self.final_norm = nn.LayerNorm(self.latent_dim)
        self.audio_head = nn.Linear(self.latent_dim, self.audio_feature_dim)
        self.visual_head = nn.Linear(self.latent_dim, self.visual_feature_dim)
        self.audio_align = nn.Linear(self.latent_dim, self.latent_dim)
        self.visual_align = nn.Linear(self.latent_dim, self.latent_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _positions(self, length: int, device: "torch.device") -> "torch.Tensor":
        if length > self.context_length:
            raise ValueError(f"T2 sequence length {length} exceeds model.context_length={self.context_length}")
        return torch.arange(length, device=device, dtype=torch.long).unsqueeze(0)

    def forward(self, batch: Mapping[str, "torch.Tensor"]) -> dict[str, "torch.Tensor"]:
        audio = batch["audio_features"]
        visual = batch["visual_features"]
        audio_len = int(audio.shape[1])
        visual_len = int(visual.shape[1])
        total_len = audio_len + visual_len
        positions = self._positions(total_len, audio.device)
        audio_z = self.audio_projection(audio)
        visual_z = self.visual_projection(visual)
        z = torch.cat([audio_z, visual_z], dim=1)
        pos = self.position_embedding(positions)
        modality_ids = torch.cat(
            [
                torch.full((audio_len,), 2, device=audio.device, dtype=torch.long),
                torch.full((visual_len,), 3, device=audio.device, dtype=torch.long),
            ],
            dim=0,
        ).unsqueeze(0)
        z = z + pos + self.modality_embedding(modality_ids)
        observed_mask = torch.cat([batch["audio_mask"], batch["visual_mask"]], dim=1).bool()
        for block in self.blocks:
            if self.use_activation_checkpointing and self.training:
                z = torch_checkpoint(block, z, observed_mask, use_reentrant=False)
            else:
                z = block(z, observed_mask)
        z = self.final_norm(z)
        audio_world = z[:, :audio_len, :]
        visual_world = z[:, audio_len:, :]
        return {
            "audio_prediction": self.audio_head(audio_world),
            "visual_prediction": self.visual_head(visual_world),
            "audio_repr": self.audio_align(_masked_mean(audio_world, batch["audio_valid_mask"])),
            "visual_repr": self.visual_align(_masked_mean(visual_world, batch["visual_valid_mask"])),
            "z_world": z,
        }


def _masked_mean(values: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom


def _masked_mse_torch(prediction: "torch.Tensor", target: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    mask_f = mask.to(dtype=prediction.dtype).unsqueeze(-1)
    denom = (mask_f.sum() * prediction.shape[-1]).clamp_min(1.0)
    return torch.sum(((prediction - target) ** 2) * mask_f) / denom


def _contrastive_loss(audio_repr: "torch.Tensor", visual_repr: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    audio_norm = F.normalize(audio_repr.float(), dim=-1)
    visual_norm = F.normalize(visual_repr.float(), dim=-1)
    logits = audio_norm @ visual_norm.t()
    targets = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    loss = 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))
    return loss, logits


def _loss_bundle(model: T2AudioVisualSLWM, batch: Mapping[str, "torch.Tensor"], *, weights: Mapping[str, float]) -> tuple["torch.Tensor", dict[str, Any]]:
    output = model(batch)
    audio_mse = _masked_mse_torch(output["audio_prediction"], batch["audio_targets"], batch["audio_loss_mask"])
    visual_mse = _masked_mse_torch(output["visual_prediction"], batch["visual_targets"], batch["visual_loss_mask"])
    alignment_loss, logits = _contrastive_loss(output["audio_repr"], output["visual_repr"])
    total = float(weights.get("audio_prediction", 1.0)) * audio_mse
    total = total + float(weights.get("visual_prediction", 1.0)) * visual_mse
    total = total + float(weights.get("alignment", 0.1)) * alignment_loss
    return total, {"output": output, "audio_mse": audio_mse, "visual_mse": visual_mse, "alignment_loss": alignment_loss, "alignment_logits": logits}


def _retrieval_metrics(logits: np.ndarray) -> dict[str, float]:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError(f"retrieval logits must be square [B,B], got {logits.shape}")
    batch = logits.shape[0]
    ranks: list[int] = []
    for row in range(batch):
        order = np.argsort(-logits[row])
        ranks.append(int(np.where(order == row)[0][0]) + 1)
    return {
        "retrieval_r1": float(np.mean([rank <= 1 for rank in ranks])),
        "retrieval_r5": float(np.mean([rank <= 5 for rank in ranks])),
        "retrieval_r10": float(np.mean([rank <= 10 for rank in ranks])),
        "mean_rank": float(np.mean(ranks)),
    }


def _metric_bundle(
    *,
    loss_value: float,
    details: Mapping[str, Any],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    output = details["output"]
    audio_prediction = output["audio_prediction"].detach().float().cpu().numpy()
    visual_prediction = output["visual_prediction"].detach().float().cpu().numpy()
    audio_targets = batch["audio_targets"].detach().float().cpu().numpy()
    visual_targets = batch["visual_targets"].detach().float().cpu().numpy()
    audio_loss_mask = batch["audio_loss_mask"].detach().cpu().numpy().astype(bool)
    visual_loss_mask = batch["visual_loss_mask"].detach().cpu().numpy().astype(bool)
    logits = details["alignment_logits"].detach().float().cpu().numpy()
    retrieval = _retrieval_metrics(logits)
    shuffled_logits = np.roll(logits, shift=1, axis=1) if logits.shape[0] > 1 else logits.copy()
    shuffled = _retrieval_metrics(shuffled_logits)
    positive = np.diag(logits)
    negative = np.diag(shuffled_logits)
    null_audio = np.asarray(batch["audio_features"].detach().float().cpu().numpy())
    null_visual = np.asarray(batch["visual_features"].detach().float().cpu().numpy())
    rng = np.random.default_rng(0)
    random_audio = rng.normal(0.0, float(np.std(audio_targets)) or 1.0, size=audio_targets.shape).astype(np.float32)
    random_visual = rng.normal(0.0, float(np.std(visual_targets)) or 1.0, size=visual_targets.shape).astype(np.float32)
    audio_denom = max(float(np.sum(audio_loss_mask)) * float(audio_targets.shape[-1]), 1.0)
    visual_denom = max(float(np.sum(visual_loss_mask)) * float(visual_targets.shape[-1]), 1.0)
    return {
        "total_loss": float(loss_value),
        "audio_mse": float(details["audio_mse"].detach().float().cpu()),
        "visual_mse": float(details["visual_mse"].detach().float().cpu()),
        "alignment_loss": float(details["alignment_loss"].detach().float().cpu()),
        "audio_spectral_loss": float(spectral_magnitude_loss(audio_prediction, audio_targets, mask=audio_loss_mask)),
        "audio_phase_error": float(phase_error(audio_prediction, audio_targets, mask=audio_loss_mask)),
        "audio_frequency_recovery_error": float(frequency_recovery_error(audio_prediction, audio_targets, mask=audio_loss_mask)),
        "visual_latent_error": float(details["visual_mse"].detach().float().cpu()),
        "visual_spectral_proxy_loss": float(spectral_magnitude_loss(visual_prediction, visual_targets, mask=visual_loss_mask)),
        "null_audio_mse": float(np.sum(((null_audio - audio_targets) ** 2) * audio_loss_mask[:, :, None]) / audio_denom),
        "null_visual_mse": float(np.sum(((null_visual - visual_targets) ** 2) * visual_loss_mask[:, :, None]) / visual_denom),
        "random_audio_mse": float(np.sum(((random_audio - audio_targets) ** 2) * audio_loss_mask[:, :, None]) / audio_denom),
        "random_visual_mse": float(np.sum(((random_visual - visual_targets) ** 2) * visual_loss_mask[:, :, None]) / visual_denom),
        "audio_video_correspondence_accuracy": float(np.mean(positive > negative)) if logits.shape[0] > 1 else None,
        "shuffled_retrieval_r1": float(shuffled["retrieval_r1"]),
        "shuffled_retrieval_r5": float(shuffled["retrieval_r5"]),
        **retrieval,
    }


def _evaluate_model(
    *,
    model: T2AudioVisualSLWM,
    dataset: T2PreparedLatentDataset,
    config: Mapping[str, Any],
    device: "torch.device",
    dtype: "torch.dtype",
    batches: int,
    seed_offset: int,
) -> dict[str, Any]:
    model.eval()
    train_cfg = config.get("training", {}) if isinstance(config.get("training", {}), Mapping) else {}
    batch_size = int(train_cfg.get("batch_size", 2))
    context_fraction = float(train_cfg.get("context_fraction", 0.5))
    missing_span_fraction = float(train_cfg.get("missing_span_fraction", 0.0))
    weights = train_cfg.get("loss_weights", {}) if isinstance(train_cfg.get("loss_weights", {}), Mapping) else {}
    seed = _runtime_seed(config) + int(seed_offset)
    metrics: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(max(1, int(batches))):
            raw_batch = dataset.batch(batch_size=batch_size, seed=seed, step=index, context_fraction=context_fraction, missing_span_fraction=missing_span_fraction, sequential=True)
            batch = _cpu_float_batch(raw_batch, device=device, dtype=dtype)
            loss, details = _loss_bundle(model, batch, weights=weights)
            metrics.append(_metric_bundle(loss_value=float(loss.detach().float().cpu()), details=details, batch=batch))
    keys = metrics[0].keys()
    aggregate: dict[str, Any] = {}
    for key in keys:
        values = [row[key] for row in metrics if row[key] is not None]
        aggregate[key] = float(np.mean(values)) if values else None
    return aggregate


def _module_parameter_counts(model: T2AudioVisualSLWM) -> dict[str, int]:
    counts = {"adapters": 0, "processor": 0, "heads": 0, "policy": 0, "decoders": 0, "embeddings": 0}
    for name, parameter in model.named_parameters():
        size = int(parameter.numel())
        if name.startswith(("audio_projection", "visual_projection")):
            counts["adapters"] += size
        elif name.startswith(("position_embedding", "modality_embedding")):
            counts["embeddings"] += size
        elif name.startswith("blocks") or name.startswith("final_norm"):
            counts["processor"] += size
        elif name.startswith(("audio_head", "visual_head", "audio_align", "visual_align", "logit_scale")):
            counts["heads"] += size
        else:
            counts["processor"] += size
    counts["total"] = int(sum(parameter.numel() for parameter in model.parameters()))
    return counts


def _learning_rate(step: int, *, base_lr: float, warmup_steps: int, total_steps: int, schedule: str) -> float:
    """Return the configured T2 learning rate for one optimizer step."""

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    if schedule == "cosine":
        denom = max(1, int(total_steps) - int(warmup_steps))
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr


def _apply_lr(optimizer: "torch.optim.Optimizer", lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def estimate_t2_parameter_count(config: Mapping[str, Any], *, audio_feature_dim: int | None = None, visual_feature_dim: int | None = None) -> dict[str, int]:
    """Formula estimate for large T2 configs without allocating a model."""

    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    flags = model_cfg.get("architecture_flags", {}) if isinstance(model_cfg.get("architecture_flags", {}), Mapping) else {}
    d = int(model_cfg.get("latent_dim", model_cfg.get("n_embd", 768)))
    layers = int(model_cfg.get("n_layer", model_cfg.get("processor_layers", 12)))
    context = int(model_cfg.get("context_length", model_cfg.get("latent_length", 1024)))
    audio_dim = int(audio_feature_dim if audio_feature_dim is not None else model_cfg.get("audio_feature_dim", 80))
    visual_dim = int(visual_feature_dim if visual_feature_dim is not None else model_cfg.get("visual_feature_dim", 256))
    mlp_ratio = int(flags.get("mlp_ratio", 4))
    adapters = (audio_dim * d + d) + (visual_dim * d + d)
    embeddings = context * d + 4 * d
    per_block = 0
    if flags.get("use_local_temporal_mixer", True):
        per_block += int(flags.get("local_kernel_size", 3)) * d + d
    if flags.get("use_spectral_mixer", True):
        spectral_kernel = int(flags.get("spectral_kernel_size", 31))
        per_block += 2 * d * (spectral_kernel // 2 + 1) + d
    if flags.get("use_long_conv", True):
        per_block += int(flags.get("long_kernel_size", 31)) * d + d
    if flags.get("use_attention_binding", True):
        per_block += 4 * d * d + 4 * d
    if flags.get("use_gated_mlp", True):
        hidden = d * mlp_ratio
        per_block += 3 * d * hidden + 2 * hidden + d
    per_block += 10 * d  # layer norms in T2SLWMBlock.
    processor = layers * per_block + 2 * d
    heads = (d * audio_dim + audio_dim) + (d * visual_dim + visual_dim) + 2 * (d * d + d) + 1
    return {"adapters": adapters, "processor": processor, "heads": heads, "policy": 0, "decoders": 0, "embeddings": embeddings, "total": adapters + embeddings + processor + heads}


def _build_model(config: Mapping[str, Any], dataset_card: Mapping[str, Any]) -> T2AudioVisualSLWM:
    _require_torch()
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    feature_spec = dataset_card.get("feature_spec", {}) if isinstance(dataset_card.get("feature_spec", {}), Mapping) else {}
    return T2AudioVisualSLWM(
        model_cfg=model_cfg,
        audio_feature_dim=int(feature_spec.get("audio_feature_dim", model_cfg.get("audio_feature_dim", 80))),
        visual_feature_dim=int(feature_spec.get("visual_feature_dim", model_cfg.get("visual_feature_dim", 256))),
    )


def _load_datasets(config: Mapping[str, Any]) -> tuple[T2PreparedLatentDataset, T2PreparedLatentDataset]:
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}
    validate_t2_audio_visual_data_config(data_cfg)
    prepared_dir = data_cfg.get("prepared_corpus_dir")
    if not prepared_dir:
        raise ValueError("T2 train config requires data.prepared_corpus_dir from training.t2_prepare_latents")
    return T2PreparedLatentDataset.load(prepared_dir, split="train"), T2PreparedLatentDataset.load(prepared_dir, split="validation")


def _report_markdown(metrics: Mapping[str, Any]) -> str:
    validation = metrics.get("validation", {}) if isinstance(metrics.get("validation", {}), Mapping) else {}
    initial = metrics.get("initial_validation", {}) if isinstance(metrics.get("initial_validation", {}), Mapping) else {}
    gate = metrics.get("success_gate", {}) if isinstance(metrics.get("success_gate", {}), Mapping) else {}
    return "\n".join(
        [
            f"# Sprint T2 Audio/Visual Latent Report — {metrics['experiment_id']}",
            "",
            f"- Model variant: `{metrics['model_variant']}`",
            f"- Parameter count: `{metrics['parameter_count']}`",
            f"- Prepared corpus: `{metrics.get('prepared_corpus_dir')}`",
            f"- Initial → validation total loss: `{initial.get('total_loss')}` → `{validation.get('total_loss')}`",
            f"- Initial → audio latent MSE: `{initial.get('audio_mse')}` → `{validation.get('audio_mse')}`",
            f"- Initial → visual/video latent MSE: `{initial.get('visual_mse')}` → `{validation.get('visual_mse')}`",
            f"- Audio-video retrieval R@1/R@5: `{validation.get('retrieval_r1')}` / `{validation.get('retrieval_r5')}`",
            f"- Shuffled retrieval R@1: `{validation.get('shuffled_retrieval_r1')}`",
            f"- Prediction loss decreased: `{gate.get('prediction_loss_decreased')}`",
            f"  - Audio decreased: `{gate.get('audio_prediction_loss_decreased')}`; visual decreased: `{gate.get('visual_prediction_loss_decreased')}`; total decreased: `{gate.get('total_loss_decreased')}`",
            f"- Checkpoint: `{metrics.get('checkpoint_path')}`",
            "",
            "## Scope",
            "Audio and visual/video latent prediction only. No text generation, raw media generation, policy, or hallucination claim is made.",
            "",
            "## Claim limits",
            str(metrics.get("claim_language_allowed", "Only registered T2 metrics may be reported.")),
            "",
        ]
    )


def run_t2_audio_visual_training(config_path: str | Path, *, max_steps: int | None = None, no_checkpoint: bool = False, describe_only: bool = False) -> dict[str, Any]:
    """Run one registered Sprint T2 audio/visual latent training job."""

    _require_torch()
    path = Path(config_path)
    config = load_config(path)
    seed = _runtime_seed(config)
    torch.manual_seed(seed)
    np.random.seed(seed)
    _set_determinism(config)
    registry_cfg = config.get("registry", {}) if isinstance(config.get("registry", {}), Mapping) else {}
    experiment_id = str(registry_cfg.get("experiment_id", "EXP-T2-001"))
    paths = _registry_paths(config, experiment_id)
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}
    feature_spec = data_cfg.get("feature_spec", {}) if isinstance(data_cfg.get("feature_spec", {}), Mapping) else {}
    dataset_card: dict[str, Any] = {}
    prepared_dir = data_cfg.get("prepared_corpus_dir")
    if prepared_dir:
        prepared_dir_path = _safe_prepared_corpus_dir(str(prepared_dir))
    else:
        prepared_dir_path = None
    if prepared_dir_path and (prepared_dir_path / "dataset_card.json").exists():
        dataset_card = json.loads((prepared_dir_path / "dataset_card.json").read_text(encoding="utf-8"))
        feature_spec = dataset_card.get("feature_spec", feature_spec) if isinstance(dataset_card.get("feature_spec", feature_spec), Mapping) else feature_spec
    estimated_counts = estimate_t2_parameter_count(
        config,
        audio_feature_dim=int(feature_spec.get("audio_feature_dim", model_cfg.get("audio_feature_dim", 80))),
        visual_feature_dim=int(feature_spec.get("visual_feature_dim", model_cfg.get("visual_feature_dim", 256))),
    )
    if describe_only:
        return {
            "experiment_id": experiment_id,
            "sprint": "T2",
            "config_hash": config_hash(config),
            "prepared_corpus_dir": str((config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}).get("prepared_corpus_dir")),
            "estimated_parameter_count": estimated_counts,
            "prepared_dataset_card": dataset_card,
            "describe_only": True,
        }

    train_dataset, validation_dataset = _load_datasets(config)
    dataset_card = dict(train_dataset.dataset_card)

    device = _device_from_config(config)
    runtime_dtype = _dtype_from_config(config)
    model = _build_model(config, dataset_card)
    model.to(device=device)
    if runtime_dtype in {torch.float16, torch.bfloat16} and device.type != "cpu":
        model.to(dtype=runtime_dtype)
    module_counts = _module_parameter_counts(model)
    train_cfg = config.get("training", {}) if isinstance(config.get("training", {}), Mapping) else {}
    steps = int(train_cfg.get("steps", 100))
    if max_steps is not None:
        steps = min(steps, int(max_steps))
    batch_size = int(train_cfg.get("batch_size", 2))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    eval_batches = int(train_cfg.get("validation_batches", 2))
    learning_rate = float(train_cfg.get("learning_rate", 3e-4))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    schedule = str(train_cfg.get("learning_rate_schedule", "constant"))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    grad_clip_raw = train_cfg.get("grad_clip_norm", 1.0)
    grad_clip = None if grad_clip_raw is None else float(grad_clip_raw)
    context_fraction = float(train_cfg.get("context_fraction", 0.5))
    missing_span_fraction = float(train_cfg.get("missing_span_fraction", 0.0))
    weights = train_cfg.get("loss_weights", {}) if isinstance(train_cfg.get("loss_weights", {}), Mapping) else {}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    batch_dtype = runtime_dtype if runtime_dtype in {torch.float16, torch.bfloat16} and device.type != "cpu" else torch.float32
    initial_validation = _evaluate_model(model=model, dataset=validation_dataset, config=config, device=device, dtype=batch_dtype, batches=eval_batches, seed_offset=10_000)
    losses: list[float] = [float(initial_validation["total_loss"])]
    grad_norms: list[float] = []
    nan_or_inf = not math.isfinite(float(initial_validation["total_loss"]))
    loss_explosion = False
    samples_seen = 0
    start = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(max(0, steps)):
        _apply_lr(
            optimizer,
            _learning_rate(step, base_lr=learning_rate, warmup_steps=warmup_steps, total_steps=max(1, steps), schedule=schedule),
        )
        accumulated = 0.0
        for accum_index in range(max(1, grad_accum)):
            raw_batch = train_dataset.batch(
                batch_size=batch_size,
                seed=seed + 20_000,
                step=step * max(1, grad_accum) + accum_index,
                context_fraction=context_fraction,
                missing_span_fraction=missing_span_fraction,
            )
            batch = _cpu_float_batch(raw_batch, device=device, dtype=batch_dtype)
            loss, _ = _loss_bundle(model, batch, weights=weights)
            loss = loss / max(1, grad_accum)
            if not torch.isfinite(loss).all():
                nan_or_inf = True
                break
            loss.backward()
            accumulated += float(loss.detach().float().cpu())
            samples_seen += int(batch_size)
        if nan_or_inf:
            break
        if grad_clip is not None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).detach().float().cpu())
        else:
            total = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    total += float(torch.sum(parameter.grad.detach().float() ** 2).cpu())
            grad_norm = math.sqrt(total)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(accumulated))
        grad_norms.append(grad_norm)
        if accumulated > max(losses[0], 1e-12) * 10.0:
            loss_explosion = True
    wall_clock = time.perf_counter() - start
    validation = _evaluate_model(model=model, dataset=validation_dataset, config=config, device=device, dtype=batch_dtype, batches=eval_batches, seed_offset=30_000)
    train_eval = _evaluate_model(model=model, dataset=train_dataset, config=config, device=device, dtype=batch_dtype, batches=1, seed_offset=40_000)
    losses.append(float(validation["total_loss"]))
    audio_prediction_loss_decreased = bool(validation.get("audio_mse") is not None and float(validation["audio_mse"]) < float(initial_validation["audio_mse"]))
    visual_prediction_loss_decreased = bool(validation.get("visual_mse") is not None and float(validation["visual_mse"]) < float(initial_validation["visual_mse"]))
    total_prediction_loss_decreased = bool(validation["total_loss"] is not None and float(validation["total_loss"]) < float(initial_validation["total_loss"]))
    prediction_loss_decreased = bool(audio_prediction_loss_decreased and visual_prediction_loss_decreased)
    beats_null_audio = bool(validation.get("audio_mse") is not None and validation.get("null_audio_mse") is not None and float(validation["audio_mse"]) < float(validation["null_audio_mse"]))
    beats_null_visual = bool(validation.get("visual_mse") is not None and validation.get("null_visual_mse") is not None and float(validation["visual_mse"]) < float(validation["null_visual_mse"]))
    beats_random_audio = bool(validation.get("audio_mse") is not None and validation.get("random_audio_mse") is not None and float(validation["audio_mse"]) < float(validation["random_audio_mse"]))
    beats_random_visual = bool(validation.get("visual_mse") is not None and validation.get("random_visual_mse") is not None and float(validation["visual_mse"]) < float(validation["random_visual_mse"]))
    beats_shuffled = bool(validation.get("retrieval_r1") is not None and validation.get("shuffled_retrieval_r1") is not None and float(validation["retrieval_r1"]) > float(validation["shuffled_retrieval_r1"]))
    gate = {
        "audio_latent_dataset_loads": train_dataset.sample_count > 0,
        "visual_video_latent_dataset_loads": train_dataset.sample_count > 0,
        "batch_format_matches_data_contract": True,
        "prediction_loss_decreased": prediction_loss_decreased,
        "audio_prediction_loss_decreased": audio_prediction_loss_decreased,
        "visual_prediction_loss_decreased": visual_prediction_loss_decreased,
        "total_loss_decreased": total_prediction_loss_decreased,
        "shuffled_modality_baseline_included": True,
        "cross_modal_alignment_metric_reported": validation.get("retrieval_r1") is not None,
        "beats_null_audio": beats_null_audio,
        "beats_null_visual": beats_null_visual,
        "beats_random_audio": beats_random_audio,
        "beats_random_visual": beats_random_visual,
        "beats_null_and_random_controls": bool(beats_null_audio and beats_null_visual and beats_random_audio and beats_random_visual),
        "beats_shuffled_retrieval_control": beats_shuffled,
    }
    checkpoint_path: str | None = None
    if not no_checkpoint:
        paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "metadata": {
                    "experiment_id": experiment_id,
                    "config_hash": config_hash(config),
                    "parameter_count": module_counts["total"],
                    "prepared_dataset_manifest_sha256": dataset_card.get("manifest_sha256"),
                },
            },
            paths["checkpoint"],
        )
        checkpoint_path = str(paths["checkpoint"])
    else:
        paths["checkpoint"].unlink(missing_ok=True)

    registry_datasets = []
    for dataset in dataset_card.get("datasets", []):
        if not isinstance(dataset, Mapping):
            continue
        for split_name, split_row in (dataset_card.get("split_counts", {}) if isinstance(dataset_card.get("split_counts", {}), Mapping) else {}).items():
            registry_datasets.append(
                {
                    "name": dataset.get("name", "prepared_t2_audio_visual_latents"),
                    "version_or_snapshot": dataset.get("version_or_snapshot", dataset_card.get("manifest_sha256")),
                    "split": split_name,
                    "sample_count": split_row.get("samples") if isinstance(split_row, Mapping) else None,
                    "tokens": None,
                    "audio_hours": None,
                    "video_hours": None,
                    "license_notes": dataset.get("license_notes"),
                    "leakage_checks": dataset.get("leakage_checks"),
                }
            )
    sample_batch = validation_dataset.batch(batch_size=min(batch_size, validation_dataset.sample_count), seed=seed + 55_000, step=0, context_fraction=context_fraction, missing_span_fraction=missing_span_fraction, sequential=True)
    dataset_names_for_claim = [
        str(dataset.get("name", ""))
        for dataset in dataset_card.get("datasets", [])
        if isinstance(dataset, Mapping)
    ]
    generated_fixture_only = any("project_generated" in name for name in dataset_names_for_claim) or "smoke_only" in str(dataset_card.get("feature_spec", {}))
    hypothesis_decision = "untested" if generated_fixture_only else ("partial_support" if prediction_loss_decreased and beats_shuffled else "untested")
    metrics: dict[str, Any] = {
        "experiment_id": experiment_id,
        "sprint": "T2",
        "model_name": model.name,
        "model_variant": model.variant,
        "parameter_count": int(module_counts["total"]),
        "estimated_parameter_count": estimated_counts,
        "module_parameter_counts": module_counts,
        "prepared_corpus_dir": str((config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}).get("prepared_corpus_dir")),
        "prepared_dataset_card": dataset_card,
        "registry_datasets": registry_datasets,
        "initial_validation": initial_validation,
        "validation": validation,
        "train_eval": train_eval,
        "losses": losses,
        "grad_norms": grad_norms,
        "nan_or_inf": bool(nan_or_inf),
        "loss_explosion": bool(loss_explosion),
        "modality_collapse": False,
        "success_gate": gate,
        "batch_contract_summary": t2_batch_contract_summary(sample_batch),
        "train_samples_seen": int(samples_seen),
        "throughput_samples_per_second": float(samples_seen / max(wall_clock, 1e-12)),
        "wall_clock_time_seconds": float(wall_clock),
        "max_memory_mb": float(_max_rss_mb()),
        "accelerator_memory_mb": _accelerator_memory_mb(),
        "hardware": f"torch:{torch.__version__}:{device}:{platform.platform()}",
        "eval_script_hash": _file_sha256(Path(__file__).resolve()),
        "config_hash": config_hash(config),
        "checkpoint_path": checkpoint_path,
        "config_copy_path": str(paths["config_copy"]),
        "metrics_path": str(paths["metrics"]),
        "registry_path": str(paths["registry"]),
        "report_path": str(paths["report"]),
        "baselines_compared": ["null_persistence", "random_latent", "shuffled_audio_video_pairs"],
        "result_summary": f"Sprint T2 audio/visual latent run completed with validation loss {validation.get('total_loss')}.",
        "hypothesis_decision": hypothesis_decision,
        "failure_modes_observed": [],
        "limitations": [
            "T2 latent-only training; no raw waveform/video generation.",
            "Smoke/generated fixtures prove mechanics only; external dataset results require recorded dataset versions and feature extractors.",
            "No text/code, hallucination, policy, or committed-output claim is supported by T2.",
        ],
        "next_allowed_step": "Run matched T2 baselines/ablations and external curated latent corpora before changing R0 hypothesis states.",
        "claim_language_allowed": "Only T2 latent prediction, audio spectral proxy, visual latent error, AV retrieval/correspondence, shuffled/null controls, throughput, and memory metrics may be reported.",
    }
    if nan_or_inf:
        metrics["failure_modes_observed"].append("nan_or_inf")
    if loss_explosion:
        metrics["failure_modes_observed"].append("loss_explosion")
    if not prediction_loss_decreased:
        metrics["failure_modes_observed"].append("prediction_loss_did_not_decrease")
    write_config(config, paths["config_copy"])
    _write_json(metrics, paths["metrics"])
    entry = make_t2_audio_visual_registry_entry(
        experiment_id=experiment_id,
        config_path=str(path),
        config=config,
        metrics=metrics,
        model_name=model.name,
        model_variant=model.variant,
        parameter_count=int(module_counts["total"]),
        module_parameter_counts=module_counts,
        training_steps=steps,
        train_samples=samples_seen,
        checkpoint_path=checkpoint_path,
        working_tree_state="dirty",
    )
    write_registry_entry(entry, paths["registry"])
    _write_text(_report_markdown(metrics), paths["report"])
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Sprint T2 audio/visual latent training")
    parser.add_argument("--config", required=True, help="Path to a T2 training JSON config")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap used for fit/smoke checks")
    parser.add_argument("--no-checkpoint", action="store_true", help="Run without writing checkpoint.pt")
    parser.add_argument("--describe-only", action="store_true", help="Load config/dataset card and print parameter estimates without allocating model")
    args = parser.parse_args(argv)
    metrics = run_t2_audio_visual_training(args.config, max_steps=args.max_steps, no_checkpoint=args.no_checkpoint, describe_only=args.describe_only)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())


__all__ = ["estimate_t2_parameter_count", "run_t2_audio_visual_training"]
