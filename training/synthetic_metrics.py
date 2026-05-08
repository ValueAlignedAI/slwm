"""Sprint T0 synthetic signal metrics and regression losses.

The functions in this module operate on the canonical latent signal contract
``FloatTensor[B,T,D]`` and intentionally avoid any text/code/audio/video dataset
assumptions.  They are used by the T0 runner to report the required controlled
signal metrics: MSE, spectral magnitude loss, phase error, frequency recovery
error, throughput, and stability.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


Array = np.ndarray


def _validate_prediction_pair(prediction: Array, target: Array) -> tuple[Array, Array]:
    pred = np.asarray(prediction, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if pred.shape != tgt.shape or pred.ndim != 3:
        raise ValueError(f"prediction and target must share [B,T,D] shape, got {pred.shape} and {tgt.shape}")
    return pred, tgt


def _normalized_mask(mask: Array | None, shape: tuple[int, int]) -> Array | None:
    if mask is None:
        return None
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != shape:
        raise ValueError(f"mask must have shape {shape}, got {mask_array.shape}")
    return mask_array


def masked_mse(prediction: Array, target: Array, mask: Array | None = None) -> float:
    """Return mean squared error over ``FloatTensor[B,T,D]`` predictions.

    Args:
        prediction: predicted latent signal with shape ``FloatTensor[B,T,D]``.
        target: target latent signal with identical shape.
        mask: optional ``BoolTensor[B,T]`` where true positions contribute.
    """

    pred, tgt = _validate_prediction_pair(prediction, target)
    diff = pred - tgt
    mask_array = _normalized_mask(mask, pred.shape[:2])
    if mask_array is not None:
        diff = diff * mask_array[:, :, None]
        denom = int(np.sum(mask_array)) * pred.shape[-1]
    else:
        denom = int(pred.size)
    if denom <= 0:
        raise ValueError("masked_mse requires at least one valid element")
    return float(np.sum(np.power(diff, 2)) / float(denom))


def masked_mse_loss(prediction: Array, target: Array, mask: Array | None = None) -> tuple[float, Array]:
    """Return MSE loss and gradient wrt prediction.

    Shape contract:
        prediction/target: ``FloatTensor[B,T,D]``; returned gradient has the
        same shape. Optional mask has shape ``BoolTensor[B,T]``.
    """

    pred, tgt = _validate_prediction_pair(prediction, target)
    diff = pred - tgt
    mask_array = _normalized_mask(mask, pred.shape[:2])
    if mask_array is not None:
        diff = diff * mask_array[:, :, None]
        denom = int(np.sum(mask_array)) * pred.shape[-1]
    else:
        denom = int(pred.size)
    if denom <= 0:
        raise ValueError("masked_mse_loss requires at least one valid element")
    return float(np.sum(np.power(diff, 2)) / float(denom)), 2.0 * diff / float(denom)


def spectral_magnitude_loss(prediction: Array, target: Array, mask: Array | None = None) -> float:
    """Return FFT magnitude MSE along the time axis.

    Input and output contract:
        ``prediction``/``target`` are ``FloatTensor[B,T,D]``. The optional mask
        zeroes invalid timesteps before the FFT but does not change the FFT
        length, keeping comparisons deterministic across models.
    """

    pred, tgt = _validate_prediction_pair(prediction, target)
    mask_array = _normalized_mask(mask, pred.shape[:2])
    if mask_array is not None:
        pred = pred * mask_array[:, :, None]
        tgt = tgt * mask_array[:, :, None]
    pred_mag = np.abs(np.fft.rfft(pred, axis=1))
    tgt_mag = np.abs(np.fft.rfft(tgt, axis=1))
    return float(np.mean(np.power(pred_mag - tgt_mag, 2)))


def phase_error(prediction: Array, target: Array, mask: Array | None = None, *, eps: float = 1e-12) -> float:
    """Return weighted mean wrapped phase error in radians.

    The target spectrum magnitude is used as the weight, and the DC bin is
    ignored because it has no useful phase for T0 periodic/frequency tasks.
    """

    pred, tgt = _validate_prediction_pair(prediction, target)
    mask_array = _normalized_mask(mask, pred.shape[:2])
    if mask_array is not None:
        pred = pred * mask_array[:, :, None]
        tgt = tgt * mask_array[:, :, None]
    pred_fft = np.fft.rfft(pred, axis=1)
    tgt_fft = np.fft.rfft(tgt, axis=1)
    weights = np.abs(tgt_fft)
    if weights.shape[1] > 0:
        weights[:, 0, :] = 0.0
    wrapped = np.angle(pred_fft * np.conj(tgt_fft))
    return float(np.sum(np.abs(wrapped) * weights) / float(np.sum(weights) + eps))


def frequency_recovery_error(prediction: Array, target: Array, mask: Array | None = None) -> float:
    """Return mean absolute dominant-frequency-bin error.

    Shape contract:
        ``prediction``/``target`` are ``FloatTensor[B,T,D]``. The returned value
        is measured in FFT bins (cycles per context window), averaged over batch
        and latent channels. DC is ignored when at least one non-DC bin exists.
    """

    pred, tgt = _validate_prediction_pair(prediction, target)
    mask_array = _normalized_mask(mask, pred.shape[:2])
    if mask_array is not None:
        pred = pred * mask_array[:, :, None]
        tgt = tgt * mask_array[:, :, None]
    pred_mag = np.abs(np.fft.rfft(pred, axis=1))
    tgt_mag = np.abs(np.fft.rfft(tgt, axis=1))
    if pred_mag.shape[1] > 1:
        pred_mag = pred_mag[:, 1:, :]
        tgt_mag = tgt_mag[:, 1:, :]
        offset = 1
    else:
        offset = 0
    pred_bins = np.argmax(pred_mag, axis=1) + offset
    tgt_bins = np.argmax(tgt_mag, axis=1) + offset
    return float(np.mean(np.abs(pred_bins - tgt_bins)))


def has_nan_or_inf(*arrays: Array) -> bool:
    """Return true if any supplied array contains NaN or Inf."""

    return any(not np.all(np.isfinite(np.asarray(array))) for array in arrays)


def prediction_metric_bundle(prediction: Array, target: Array, loss_mask: Array | None = None) -> dict[str, float]:
    """Return the T0 required signal-quality metrics for one prediction."""

    return {
        "mse": masked_mse(prediction, target, mask=loss_mask),
        "spectral_loss": spectral_magnitude_loss(prediction, target),
        "phase_error": phase_error(prediction, target),
        "frequency_recovery_error": frequency_recovery_error(prediction, target),
    }


def summarize_stability(losses: list[float], grad_norms: list[float]) -> dict[str, Any]:
    """Summarize finite-loss and gradient-norm stability for a train run."""

    finite_losses = all(np.isfinite(loss) for loss in losses)
    finite_grads = all(np.isfinite(norm) for norm in grad_norms)
    initial = float(losses[0]) if losses else float("nan")
    max_loss = float(max(losses)) if losses else float("nan")
    return {
        "nan_or_inf": not (finite_losses and finite_grads),
        "loss_explosion": bool(np.isfinite(initial) and max_loss > initial * 10.0),
        "max_grad_norm": float(max(grad_norms)) if grad_norms else 0.0,
        "mean_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
        "loss_count": len(losses),
    }


def flatten_metric_summary(metrics_by_task: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Aggregate nested task/model metrics into compact registry fields."""

    slwm_mses: list[float] = []
    slwm_spectral: list[float] = []
    slwm_phase: list[float] = []
    slwm_frequency: list[float] = []
    for task_metrics in metrics_by_task.values():
        final_metrics = task_metrics.get("slwm", {}).get("final_metrics", {})
        if final_metrics:
            slwm_mses.append(float(final_metrics.get("mse", np.nan)))
            slwm_spectral.append(float(final_metrics.get("spectral_loss", np.nan)))
            slwm_phase.append(float(final_metrics.get("phase_error", np.nan)))
            slwm_frequency.append(float(final_metrics.get("frequency_recovery_error", np.nan)))
    return {
        "synthetic_mse": float(np.nanmean(slwm_mses)) if slwm_mses else None,
        "spectral_magnitude_error": float(np.nanmean(slwm_spectral)) if slwm_spectral else None,
        "phase_or_coherence_error": float(np.nanmean(slwm_phase)) if slwm_phase else None,
        "frequency_recovery_error": float(np.nanmean(slwm_frequency)) if slwm_frequency else None,
    }


__all__ = [
    "Array",
    "flatten_metric_summary",
    "frequency_recovery_error",
    "has_nan_or_inf",
    "masked_mse",
    "masked_mse_loss",
    "phase_error",
    "prediction_metric_bundle",
    "spectral_magnitude_loss",
    "summarize_stability",
]
