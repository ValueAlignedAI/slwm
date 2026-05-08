"""Deterministic synthetic signal datasets for Sprint T0.

Sprint T0 is explicitly synthetic-only.  This module generates controlled
``FloatTensor[B,T,D]`` latent-signal batches without touching text/code/audio/
video datasets.  Each batch contains observed input latents, target latents,
input masks, loss masks, source tags, and reproducible sample IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np

from data.contract import SOURCE_TAGS


Array = np.ndarray

SUPPORTED_SYNTHETIC_TASKS: tuple[str, ...] = (
    "sine_mixture",
    "chirp_extrapolation",
    "phase_shift_detection",
    "noisy_periodic_denoising",
    "missing_span_reconstruction",
    "long_horizon_extrapolation",
)


@dataclass(frozen=True)
class SyntheticSignalBatch:
    """Canonical T0 synthetic signal batch.

    Shape contract:
        ``input_latents`` and ``target_latents`` use ``FloatTensor[B,T,D]``.
        ``input_mask`` and ``loss_mask`` use ``BoolTensor[B,T]``.  ``input_mask``
        marks observed context positions; ``loss_mask`` marks target positions
        that contribute to the training objective.
    """

    input_latents: Array
    target_latents: Array
    input_mask: Array
    loss_mask: Array
    sample_ids: list[str]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a dict representation used by runners and tests."""

        return {
            "input_latents": self.input_latents,
            "target_latents": self.target_latents,
            "input_mask": self.input_mask,
            "loss_mask": self.loss_mask,
            "sample_ids": list(self.sample_ids),
            "metadata": dict(self.metadata),
        }


def _stable_seed(seed: int, task: str, split: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}::{task}::{split}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _validate_task(task: str) -> str:
    name = str(task)
    if name not in SUPPORTED_SYNTHETIC_TASKS:
        raise ValueError(f"Unsupported synthetic T0 task {name!r}; expected one of {SUPPORTED_SYNTHETIC_TASKS}")
    return name


def _normalize(signal: Array) -> Array:
    mean = np.mean(signal, axis=1, keepdims=True)
    std = np.std(signal, axis=1, keepdims=True)
    return (signal - mean) / np.maximum(std, 1e-6)


def _time_grid(context_length: int, *, offset: int = 0) -> Array:
    return (np.arange(int(context_length), dtype=np.float64) + float(offset)) / float(context_length)


def _sine_mixture(
    rng: np.random.Generator,
    *,
    batch_size: int,
    context_length: int,
    latent_dim: int,
    horizon: int,
    min_frequency: int,
    max_frequency: int,
) -> tuple[Array, Array, dict[str, Any]]:
    components = 2
    freqs = rng.integers(min_frequency, max_frequency + 1, size=(batch_size, latent_dim, components))
    amps = rng.uniform(0.35, 1.0, size=(batch_size, latent_dim, components))
    phases = rng.uniform(-np.pi, np.pi, size=(batch_size, latent_dim, components))

    def signal_at(offset: int) -> Array:
        t = _time_grid(context_length, offset=offset)[None, :, None, None]
        values = amps[:, None, :, :] * np.sin(2.0 * np.pi * freqs[:, None, :, :] * t + phases[:, None, :, :])
        return np.sum(values, axis=-1)

    input_signal = _normalize(signal_at(0))
    target_signal = _normalize(signal_at(horizon))
    return input_signal, target_signal, {"frequencies": freqs.tolist(), "horizon": int(horizon)}


def _chirp(
    rng: np.random.Generator,
    *,
    batch_size: int,
    context_length: int,
    latent_dim: int,
    horizon: int,
    min_frequency: int,
    max_frequency: int,
) -> tuple[Array, Array, dict[str, Any]]:
    f0 = rng.uniform(float(min_frequency), max(float(min_frequency) + 0.1, float(max_frequency) / 2.0), size=(batch_size, latent_dim))
    f1 = rng.uniform(float(max_frequency) / 2.0, float(max_frequency) + 1.0, size=(batch_size, latent_dim))
    chirp_rate = f1 - f0
    phase0 = rng.uniform(-np.pi, np.pi, size=(batch_size, latent_dim))

    def signal_at(offset: int) -> Array:
        t = _time_grid(context_length, offset=offset)[None, :, None]
        phase = 2.0 * np.pi * (f0[:, None, :] * t + 0.5 * chirp_rate[:, None, :] * np.power(t, 2)) + phase0[:, None, :]
        return np.sin(phase)

    return _normalize(signal_at(0)), _normalize(signal_at(horizon)), {
        "start_frequency": f0.tolist(),
        "end_frequency": f1.tolist(),
        "horizon": int(horizon),
    }


def _phase_shift(
    rng: np.random.Generator,
    *,
    batch_size: int,
    context_length: int,
    latent_dim: int,
    phase_offset: float,
    min_frequency: int,
    max_frequency: int,
) -> tuple[Array, Array, dict[str, Any]]:
    freqs = rng.integers(min_frequency, max_frequency + 1, size=(batch_size, latent_dim))
    phases = rng.uniform(-np.pi, np.pi, size=(batch_size, latent_dim))
    t = _time_grid(context_length)[None, :, None]
    input_signal = np.sin(2.0 * np.pi * freqs[:, None, :] * t + phases[:, None, :])
    target_signal = np.sin(2.0 * np.pi * freqs[:, None, :] * t + phases[:, None, :] + float(phase_offset))
    return _normalize(input_signal), _normalize(target_signal), {
        "frequencies": freqs.tolist(),
        "phase_offset_radians": float(phase_offset),
    }


def _missing_span_mask(
    rng: np.random.Generator,
    *,
    batch_size: int,
    context_length: int,
    missing_fraction: float,
) -> Array:
    span = max(1, int(round(float(context_length) * float(missing_fraction))))
    span = min(span, int(context_length))
    mask = np.ones((batch_size, context_length), dtype=bool)
    max_start = max(1, context_length - span + 1)
    for row in range(batch_size):
        start = int(rng.integers(0, max_start))
        mask[row, start : start + span] = False
    return mask


def make_synthetic_signal_batch(
    task: str,
    *,
    batch_size: int,
    context_length: int,
    latent_dim: int,
    seed: int,
    split: str = "train",
    horizon: int | None = None,
    long_horizon: int | None = None,
    noise_std: float = 0.25,
    missing_fraction: float = 0.25,
    phase_offset: float = np.pi / 2.0,
    min_frequency: int = 1,
    max_frequency: int | None = None,
) -> SyntheticSignalBatch:
    """Generate one deterministic synthetic signal batch.

    Args:
        task: one of ``SUPPORTED_SYNTHETIC_TASKS``.
        batch_size: batch dimension ``B``.
        context_length: time dimension ``T``.
        latent_dim: latent channel dimension ``D``.
        seed: base seed; task and split are mixed into a stable derived seed.
        split: split label for sample IDs and deterministic seed derivation.

    Returns:
        ``SyntheticSignalBatch`` with canonical ``FloatTensor[B,T,D]`` latents.
    """

    name = _validate_task(task)
    if batch_size <= 0 or context_length <= 1 or latent_dim <= 0:
        raise ValueError("batch_size, context_length, and latent_dim must be positive; context_length must exceed 1")
    max_freq = int(max_frequency if max_frequency is not None else max(2, min(8, context_length // 3)))
    min_freq = max(1, int(min_frequency))
    if max_freq < min_freq:
        raise ValueError("max_frequency must be >= min_frequency")
    default_horizon = max(1, int(round(context_length * 0.125)))
    h = int(horizon if horizon is not None else default_horizon)
    h_long = int(long_horizon if long_horizon is not None else max(h + 1, int(round(context_length * 0.375))))

    rng = np.random.default_rng(_stable_seed(int(seed), name, str(split)))
    if name == "sine_mixture":
        input_signal, target_signal, task_metadata = _sine_mixture(
            rng,
            batch_size=batch_size,
            context_length=context_length,
            latent_dim=latent_dim,
            horizon=h,
            min_frequency=min_freq,
            max_frequency=max_freq,
        )
        input_mask = np.ones((batch_size, context_length), dtype=bool)
        loss_mask = np.ones_like(input_mask)
        target_tag = "predicted"
    elif name == "chirp_extrapolation":
        input_signal, target_signal, task_metadata = _chirp(
            rng,
            batch_size=batch_size,
            context_length=context_length,
            latent_dim=latent_dim,
            horizon=h,
            min_frequency=min_freq,
            max_frequency=max_freq,
        )
        input_mask = np.ones((batch_size, context_length), dtype=bool)
        loss_mask = np.ones_like(input_mask)
        target_tag = "predicted"
    elif name == "phase_shift_detection":
        input_signal, target_signal, task_metadata = _phase_shift(
            rng,
            batch_size=batch_size,
            context_length=context_length,
            latent_dim=latent_dim,
            phase_offset=phase_offset,
            min_frequency=min_freq,
            max_frequency=max_freq,
        )
        input_mask = np.ones((batch_size, context_length), dtype=bool)
        loss_mask = np.ones_like(input_mask)
        target_tag = "predicted"
    elif name == "noisy_periodic_denoising":
        clean, _, task_metadata = _sine_mixture(
            rng,
            batch_size=batch_size,
            context_length=context_length,
            latent_dim=latent_dim,
            horizon=0,
            min_frequency=min_freq,
            max_frequency=max_freq,
        )
        noise = rng.normal(0.0, float(noise_std), size=clean.shape)
        input_signal = clean + noise
        target_signal = clean
        task_metadata.update({"noise_std": float(noise_std)})
        input_mask = np.ones((batch_size, context_length), dtype=bool)
        loss_mask = np.ones_like(input_mask)
        target_tag = "reconstructed"
    elif name == "missing_span_reconstruction":
        clean, _, task_metadata = _sine_mixture(
            rng,
            batch_size=batch_size,
            context_length=context_length,
            latent_dim=latent_dim,
            horizon=0,
            min_frequency=min_freq,
            max_frequency=max_freq,
        )
        input_mask = _missing_span_mask(rng, batch_size=batch_size, context_length=context_length, missing_fraction=missing_fraction)
        input_signal = clean.copy()
        input_signal[~input_mask, :] = 0.0
        target_signal = clean
        loss_mask = ~input_mask
        task_metadata.update({"missing_fraction": float(missing_fraction)})
        target_tag = "reconstructed"
    elif name == "long_horizon_extrapolation":
        input_signal, target_signal, task_metadata = _sine_mixture(
            rng,
            batch_size=batch_size,
            context_length=context_length,
            latent_dim=latent_dim,
            horizon=h_long,
            min_frequency=min_freq,
            max_frequency=max_freq,
        )
        input_mask = np.ones((batch_size, context_length), dtype=bool)
        loss_mask = np.ones_like(input_mask)
        target_tag = "predicted"
    else:  # pragma: no cover - _validate_task makes this unreachable.
        raise AssertionError(name)

    if target_tag not in SOURCE_TAGS:
        raise ValueError(f"target tag {target_tag!r} must be in controlled SOURCE_TAGS")
    sample_ids = [f"synthetic::{name}::{split}::{seed}::{index}" for index in range(batch_size)]
    metadata = {
        "dataset": "synthetic_signal_t0",
        "dataset_version": "synthetic_signal_v0",
        "task": name,
        "split": str(split),
        "seed": int(seed),
        "source_tags": {"input": "observed", "target": target_tag},
        "modality_mix": {"synthetic_signal": 1.0},
        "shape": {"batch_size": int(batch_size), "context_length": int(context_length), "latent_dim": int(latent_dim)},
        "task_parameters": task_metadata,
    }
    return SyntheticSignalBatch(
        input_latents=np.asarray(input_signal, dtype=np.float64),
        target_latents=np.asarray(target_signal, dtype=np.float64),
        input_mask=np.asarray(input_mask, dtype=bool),
        loss_mask=np.asarray(loss_mask, dtype=bool),
        sample_ids=sample_ids,
        metadata=metadata,
    )


def make_batches_from_config(config: Mapping[str, Any], *, task: str, split: str) -> SyntheticSignalBatch:
    """Create a synthetic batch from the shared T0 config mapping."""

    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    runtime_cfg = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    training_cfg = config.get("training", {}) if isinstance(config.get("training", {}), Mapping) else {}
    signal_cfg = data_cfg.get("signal", {}) if isinstance(data_cfg.get("signal", {}), Mapping) else {}
    return make_synthetic_signal_batch(
        task,
        batch_size=int(training_cfg.get("batch_size", data_cfg.get("batch_size", 4))),
        context_length=int(model_cfg.get("context_length", model_cfg.get("latent_length", data_cfg.get("context_length", 32)))),
        latent_dim=int(model_cfg.get("latent_dim", model_cfg.get("n_embd", data_cfg.get("latent_dim", 8)))),
        seed=int(runtime_cfg.get("seed", 0)),
        split=split,
        horizon=signal_cfg.get("horizon"),
        long_horizon=signal_cfg.get("long_horizon"),
        noise_std=float(signal_cfg.get("noise_std", 0.25)),
        missing_fraction=float(signal_cfg.get("missing_fraction", 0.25)),
        phase_offset=float(signal_cfg.get("phase_offset", np.pi / 2.0)),
        min_frequency=int(signal_cfg.get("min_frequency", 1)),
        max_frequency=(None if signal_cfg.get("max_frequency") is None else int(signal_cfg.get("max_frequency"))),
    )


__all__ = [
    "SUPPORTED_SYNTHETIC_TASKS",
    "SyntheticSignalBatch",
    "make_batches_from_config",
    "make_synthetic_signal_batch",
]
