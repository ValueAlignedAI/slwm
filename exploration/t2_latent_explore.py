"""Diagnostic Sprint T2 latent exploration.

This module intentionally explores *latent* audio/visual predictions from a
trained T2 checkpoint. By default it writes latent heatmaps/probes only. When
``--render-media`` is enabled it also writes approximate WAV/PNG/GIF/MP4 files
using fixed heuristic latent-to-media renderers; those files are diagnostic
decodes, not learned raw-media generation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from data.audio_visual_latents import T2PreparedLatentDataset, t2_batch_contract_summary
from training.run_support import SafePathPolicy, atomic_write_json, atomic_write_text
from training.t2_train_latents import T2AudioVisualSLWM, _contrastive_loss, _cpu_float_batch, _retrieval_metrics

try:  # pragma: no cover - covered by command-level smoke when available
    import torch
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

try:  # pragma: no cover - optional plotting dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _PLOT_IMPORT_ERROR = exc
else:
    _PLOT_IMPORT_ERROR = None

try:  # pragma: no cover - optional fallback plotting dependency
    from PIL import Image, ImageDraw
except Exception as exc:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None

try:  # pragma: no cover - optional diagnostic media renderer
    import librosa
except Exception as exc:  # pragma: no cover
    librosa = None  # type: ignore[assignment]
    _LIBROSA_IMPORT_ERROR = exc
else:
    _LIBROSA_IMPORT_ERROR = None

try:  # pragma: no cover - optional diagnostic media renderer
    import soundfile as sf
except Exception as exc:  # pragma: no cover
    sf = None  # type: ignore[assignment]
    _SOUNDFILE_IMPORT_ERROR = exc
else:
    _SOUNDFILE_IMPORT_ERROR = None

try:  # pragma: no cover - optional diagnostic media renderer
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


CLAIM_LIMIT = (
    "Diagnostic T2 latent exploration only: predicted/reconstructed audio and "
    "visual latent fields, retrieval probes, PCA views, and optional heuristic "
    "latent-to-media renders. No learned raw waveform/video decoder, text-generation, "
    "policy/commit, grounding, understanding, or faithful media-generation claim."
)
MEDIA_CLAIM_LIMIT = (
    "Optional WAV/PNG/MP4 outputs are approximate diagnostic latent-to-media renders "
    "using fixed inverse-log-mel/row-latent heuristics. The T2 model did not train "
    "a raw waveform/video decoder, phase/color/motion were discarded by extraction, "
    "and these files are not evidence of faithful media generation."
)

T2_EXPERIMENT_POLICY = SafePathPolicy(Path("experiments/multimodal/t2"), "T2 experiment")
T2_PREPARED_POLICY = SafePathPolicy(Path("artifacts/t2_audio_visual"), "T2 prepared corpus")
EXPLORATION_OUTPUT_POLICY = SafePathPolicy(Path("experiments/exploration"), "exploration output")


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise ImportError("exploration.t2_latent_explore requires PyTorch") from _TORCH_IMPORT_ERROR


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_device(requested: str) -> "torch.device":
    _require_torch()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    return torch.device(requested)


def _resolve_dtype(requested: str, device: "torch.device") -> "torch.dtype":
    _require_torch()
    requested = requested.lower()
    if device.type == "cpu":
        return torch.float32
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if requested in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _load_model(exp_dir: Path, config: Mapping[str, Any], device: "torch.device", dtype: "torch.dtype") -> T2AudioVisualSLWM:
    _require_torch()
    feature_spec = config.get("data", {}).get("feature_spec", {})
    model = T2AudioVisualSLWM(
        model_cfg=config["model"],
        audio_feature_dim=int(feature_spec.get("audio_feature_dim", config["model"].get("audio_feature_dim", 80))),
        visual_feature_dim=int(feature_spec.get("visual_feature_dim", config["model"].get("visual_feature_dim", 256))),
    )
    checkpoint_path = T2_EXPERIMENT_POLICY.validate(exp_dir / "checkpoint.pt", must_exist=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _mode_batch(batch: dict[str, Any], mode: str) -> dict[str, Any]:
    """Apply diagnostic probe mode without changing targets."""

    edited = dict(batch)
    if mode == "masked_av":
        return edited
    if mode == "audio_to_visual":
        edited["visual_features"] = np.zeros_like(np.asarray(batch["visual_features"], dtype=np.float32))
        edited["visual_mask"] = np.zeros_like(np.asarray(batch["visual_mask"], dtype=bool))
        edited["visual_loss_mask"] = np.asarray(batch["visual_valid_mask"], dtype=bool).copy()
        return edited
    if mode == "visual_to_audio":
        edited["audio_features"] = np.zeros_like(np.asarray(batch["audio_features"], dtype=np.float32))
        edited["audio_mask"] = np.zeros_like(np.asarray(batch["audio_mask"], dtype=bool))
        edited["audio_loss_mask"] = np.asarray(batch["audio_valid_mask"], dtype=bool).copy()
        return edited
    raise ValueError(f"Unsupported probe mode: {mode}")


def _masked_mse_by_time(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    error = np.mean((prediction - target) ** 2, axis=-1)
    return np.where(mask, error, np.nan)


def _masked_mse_scalar(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float("nan")
    error = (prediction - target) ** 2
    return float(np.mean(error[valid]))


def _nanmean_time(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    summed = np.where(finite, values, 0.0).sum(axis=0)
    counts = finite.sum(axis=0)
    return np.divide(summed, counts, out=np.full(summed.shape, np.nan, dtype=np.float64), where=counts > 0)


def _finite_range(*arrays: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array, dtype=np.float32).ravel() for array in arrays])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(values, low))
    vmax = float(np.percentile(values, high))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def _colormap(values: np.ndarray, *, vmin: float, vmax: float, palette: str = "viridis") -> np.ndarray:
    scaled = (np.asarray(values, dtype=np.float32) - float(vmin)) / max(float(vmax) - float(vmin), 1e-12)
    scaled = np.nan_to_num(np.clip(scaled, 0.0, 1.0), nan=0.0)
    if palette == "magma":
        anchors = np.asarray([[0, 0, 4], [87, 15, 109], [187, 55, 84], [249, 142, 8], [252, 253, 191]], dtype=np.float32)
    elif palette == "coolwarm":
        anchors = np.asarray([[59, 76, 192], [221, 221, 221], [180, 4, 38]], dtype=np.float32)
    else:
        anchors = np.asarray([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=np.float32)
    pos = scaled * (len(anchors) - 1)
    low = np.floor(pos).astype(np.int32)
    high = np.clip(low + 1, 0, len(anchors) - 1)
    frac = (pos - low)[..., None]
    rgb = anchors[low] * (1.0 - frac) + anchors[high] * frac
    return np.asarray(rgb, dtype=np.uint8)


def _downsample_2d(array: np.ndarray, *, max_rows: int = 96, max_cols: int = 192) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.shape[0] > max_rows:
        row_idx = np.linspace(0, values.shape[0] - 1, max_rows).astype(np.int64)
        values = values[row_idx]
    if values.shape[1] > max_cols:
        col_idx = np.linspace(0, values.shape[1] - 1, max_cols).astype(np.int64)
        values = values[:, col_idx]
    return values


def _save_heatmap_grid_pil(
    *,
    path: Path,
    title: str,
    input_values: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    max_feature_dims: int,
) -> None:
    if Image is None or ImageDraw is None:  # pragma: no cover
        return
    width = min(int(max_feature_dims), int(target.shape[-1]))
    panels = [
        ("input latent", np.asarray(input_values[:, :width], dtype=np.float32).T, "viridis"),
        ("predicted latent", np.asarray(prediction[:, :width], dtype=np.float32).T, "viridis"),
        ("target latent", np.asarray(target[:, :width], dtype=np.float32).T, "viridis"),
        (
            "abs error on target positions",
            np.where(np.asarray(mask, dtype=bool)[None, :], np.abs(np.asarray(prediction[:, :width]) - np.asarray(target[:, :width])).T, np.nan),
            "magma",
        ),
    ]
    vmin, vmax = _finite_range(panels[0][1], panels[1][1], panels[2][1])
    cell_w = 4
    cell_h = 4
    label_h = 32
    title_h = 42
    margin = 12
    rendered: list[tuple[str, Image.Image]] = []
    for _, values, palette in panels:
        sampled = _downsample_2d(values, max_rows=min(width, 96), max_cols=192)
        local_vmin, local_vmax = _finite_range(sampled) if palette == "magma" else (vmin, vmax)
        image = Image.fromarray(_colormap(sampled, vmin=local_vmin, vmax=local_vmax, palette=palette), mode="RGB")
        image = image.resize((sampled.shape[1] * cell_w, sampled.shape[0] * cell_h), resample=Image.Resampling.NEAREST)
        rendered.append((_, image))
    canvas_w = max(image.width for _, image in rendered) + 2 * margin
    canvas_h = title_h + sum(image.height + label_h for _, image in rendered) + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), title, fill="black")
    y = title_h
    for subtitle, image in rendered:
        draw.text((margin, y), subtitle, fill="black")
        y += label_h
        canvas.paste(image, (margin, y))
        y += image.height
    canvas.save(path)


def _save_curve_pil(path: Path, title: str, audio_mse_t: np.ndarray, visual_mse_t: np.ndarray) -> None:
    if Image is None or ImageDraw is None:  # pragma: no cover
        return
    width, height = 1000, 360
    margin_l, margin_r, margin_t, margin_b = 70, 30, 45, 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin_l, 14), title, fill="black")
    series = [_nanmean_time(audio_mse_t), _nanmean_time(visual_mse_t)]
    all_values = np.concatenate([item[np.isfinite(item)] for item in series if np.any(np.isfinite(item))])
    ymax = float(np.nanpercentile(all_values, 98)) if all_values.size else 1.0
    ymin = 0.0
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    draw.rectangle((margin_l, margin_t, margin_l + plot_w, margin_t + plot_h), outline="black")
    for values, color, label in [(series[0], "blue", "audio"), (series[1], "orange", "visual")]:
        points = []
        for idx, value in enumerate(values):
            if not np.isfinite(value):
                continue
            x = margin_l + int(idx / max(len(values) - 1, 1) * plot_w)
            y = margin_t + plot_h - int((float(value) - ymin) / max(ymax - ymin, 1e-12) * plot_h)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        draw.text((margin_l + 10, margin_t + (20 if color == "blue" else 40)), label, fill=color)
    draw.text((margin_l, height - 35), "latent time step", fill="black")
    draw.text((8, margin_t + 10), f"MSE 0..{ymax:.3g}", fill="black")
    canvas.save(path)


def _save_retrieval_matrix_pil(path: Path, logits: np.ndarray, title: str) -> None:
    if Image is None or ImageDraw is None:  # pragma: no cover
        return
    matrices = [("real pair logits", logits), ("shuffled visual control", np.roll(logits, shift=1, axis=1))]
    cell = 42
    label_h = 32
    margin = 16
    n = int(logits.shape[0])
    width = 2 * (n * cell + 2 * margin) + margin
    height = 70 + label_h + n * cell + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), title, fill="black")
    vmin, vmax = _finite_range(*[matrix for _, matrix in matrices], low=0, high=100)
    x0 = margin
    for subtitle, matrix in matrices:
        draw.text((x0, 52), subtitle, fill="black")
        image = Image.fromarray(_colormap(matrix, vmin=vmin, vmax=vmax, palette="coolwarm"), mode="RGB")
        image = image.resize((n * cell, n * cell), resample=Image.Resampling.NEAREST)
        canvas.paste(image, (x0, 52 + label_h))
        for idx in range(n):
            draw.rectangle((x0 + idx * cell, 52 + label_h + idx * cell, x0 + (idx + 1) * cell, 52 + label_h + (idx + 1) * cell), outline="black")
        x0 += n * cell + 2 * margin
    canvas.save(path)


def _save_pca_pil(path: Path, z_world: np.ndarray, audio_len: int, max_points: int, title: str) -> dict[str, Any]:
    if Image is None or ImageDraw is None:  # pragma: no cover
        return {"plot_created": False, "reason": repr(_PIL_IMPORT_ERROR)}
    flattened = z_world.reshape(-1, z_world.shape[-1]).astype(np.float32)
    modality = np.tile(np.concatenate([np.zeros(audio_len, dtype=np.int32), np.ones(z_world.shape[1] - audio_len, dtype=np.int32)]), z_world.shape[0])
    if flattened.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(flattened.shape[0], size=max_points, replace=False))
        flattened = flattened[keep]
        modality = modality[keep]
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    explained = (singular_values[:2] ** 2) / np.maximum(np.sum(singular_values**2), 1e-12)
    width, height, margin = 700, 620, 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), title, fill="black")
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    for point, value in zip(coords, modality, strict=False):
        x = margin + int((point[0] - xmin) / max(xmax - xmin, 1e-12) * (width - 2 * margin))
        y = height - margin - int((point[1] - ymin) / max(ymax - ymin, 1e-12) * (height - 2 * margin))
        color = "blue" if value == 0 else "orange"
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black")
    draw.text((margin, height - 38), f"PC1 ({explained[0] * 100:.1f}% var), PC2 ({explained[1] * 100:.1f}% var)", fill="black")
    draw.text((width - 210, 20), "blue=audio orange=visual", fill="black")
    canvas.save(path)
    return {"plot_created": True, "explained_variance_ratio": [float(explained[0]), float(explained[1])], "renderer": "PIL"}


def _completed_latent(input_values: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    completed = np.asarray(input_values, dtype=np.float32).copy()
    completed[np.asarray(mask, dtype=bool)] = np.asarray(prediction, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    return completed


def _pad_or_crop_2d(values: np.ndarray, rows: int, cols: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.zeros((int(rows), int(cols)), dtype=np.float32)
    use_rows = min(int(rows), arr.shape[0])
    use_cols = min(int(cols), arr.shape[1])
    out[:use_rows, :use_cols] = arr[:use_rows, :use_cols]
    return out


def _visual_latent_to_image(values: np.ndarray, *, rows: int = 256, cols: int = 256) -> "Image.Image | None":
    if Image is None:  # pragma: no cover
        return None
    frame = _pad_or_crop_2d(values, rows, cols)
    # The VGGSound visual extractor stored grayscale pixels normalized to [-1, 1].
    # Predictions can exceed that range, so clip for diagnostic rendering.
    pixels = np.clip((frame + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="L")


def _label_image(image: "Image.Image", label: str) -> "Image.Image":
    if ImageDraw is None:  # pragma: no cover
        return image.convert("RGB")
    rgb = image.convert("RGB")
    canvas = Image.new("RGB", (rgb.width, rgb.height + 28), "white")
    canvas.paste(rgb, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 7), label, fill="black")
    return canvas


def _save_visual_media(
    *,
    mode_dir: Path,
    mode: str,
    sample_index: int,
    visual_input: np.ndarray,
    visual_prediction: np.ndarray,
    visual_target: np.ndarray,
    visual_loss_mask: np.ndarray,
    fps: int,
    seconds: float,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if Image is None:
        artifacts["visual_media_unavailable"] = repr(_PIL_IMPORT_ERROR)
        return artifacts
    input_frame = _visual_latent_to_image(visual_input[sample_index])
    prediction_frame = _visual_latent_to_image(visual_prediction[sample_index])
    target_frame = _visual_latent_to_image(visual_target[sample_index])
    completed_frame = _visual_latent_to_image(
        _completed_latent(visual_input[sample_index], visual_prediction[sample_index], visual_loss_mask[sample_index])
    )
    if input_frame is None or prediction_frame is None or target_frame is None or completed_frame is None:  # pragma: no cover
        artifacts["visual_media_unavailable"] = repr(_PIL_IMPORT_ERROR)
        return artifacts
    frames = {
        "input_masked_visual_frame.png": input_frame,
        "prediction_only_visual_frame.png": prediction_frame,
        "completed_visual_frame.png": completed_frame,
        "target_visual_frame.png": target_frame,
    }
    for name, image in frames.items():
        path = mode_dir / name
        image.save(path)
        artifacts[name.removesuffix(".png")] = str(path)
    panel_frames = [
        _label_image(input_frame, "input / observed rows"),
        _label_image(prediction_frame, "prediction only"),
        _label_image(completed_frame, "generated completed frame"),
        _label_image(target_frame, "target reference"),
    ]
    panel = Image.new("RGB", (panel_frames[0].width * 4, panel_frames[0].height), "white")
    for idx, frame in enumerate(panel_frames):
        panel.paste(frame, (idx * frame.width, 0))
    panel_path = mode_dir / "diagnostic_visual_panels.png"
    panel.save(panel_path)
    artifacts["diagnostic_visual_panels"] = str(panel_path)
    gif_path = mode_dir / "diagnostic_visual_sequence.gif"
    repeated = []
    for frame in panel_frames:
        repeated.extend([frame] * max(1, int(fps * seconds / len(panel_frames))))
    repeated[0].save(gif_path, save_all=True, append_images=repeated[1:], duration=int(1000 / max(fps, 1)), loop=0)
    artifacts["diagnostic_visual_gif"] = str(gif_path)
    if cv2 is not None:
        mp4_path = mode_dir / "diagnostic_visual_sequence.mp4"
        rgb_frames = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in repeated]
        height, width = rgb_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if writer.isOpened():
            for frame in rgb_frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            artifacts["diagnostic_visual_mp4"] = str(mp4_path)
        else:
            artifacts["visual_mp4_unavailable"] = "cv2.VideoWriter could not open mp4v writer"
    else:
        artifacts["visual_mp4_unavailable"] = repr(_CV2_IMPORT_ERROR)
    return artifacts


def _latent_to_db(values: np.ndarray, reference: np.ndarray | None = None, *, db_floor: float = -80.0) -> np.ndarray:
    latent = np.asarray(values, dtype=np.float32)
    ref = np.asarray(reference if reference is not None else values, dtype=np.float32)
    finite = ref[np.isfinite(ref)]
    if finite.size == 0:
        low, high = -2.5, 2.5
    else:
        low, high = np.percentile(finite, [2, 98]).astype(np.float32)
        if math.isclose(float(low), float(high)):
            low, high = float(low) - 1.0, float(high) + 1.0
    scaled = np.clip((latent - float(low)) / max(float(high - low), 1e-6), 0.0, 1.0)
    return (float(db_floor) + (-float(db_floor)) * scaled).astype(np.float32)


def _latent_to_waveform(
    values: np.ndarray,
    *,
    reference: np.ndarray | None,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_iter: int,
) -> np.ndarray | None:
    if librosa is None:
        return None
    db = _latent_to_db(values, reference=reference).T
    power = librosa.db_to_power(db, ref=1.0)
    waveform = librosa.feature.inverse.mel_to_audio(
        power,
        sr=int(sample_rate),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        n_iter=int(n_iter),
        power=2.0,
    ).astype(np.float32)
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform = 0.95 * waveform / peak
    return waveform


def _save_audio_media(
    *,
    mode_dir: Path,
    sample_index: int,
    audio_input: np.ndarray,
    audio_prediction: np.ndarray,
    audio_target: np.ndarray,
    audio_loss_mask: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_iter: int,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if librosa is None or sf is None:
        artifacts["audio_media_unavailable"] = f"librosa={_LIBROSA_IMPORT_ERROR!r}; soundfile={_SOUNDFILE_IMPORT_ERROR!r}"
        return artifacts
    completed = _completed_latent(audio_input[sample_index], audio_prediction[sample_index], audio_loss_mask[sample_index])
    audio_items = {
        "input_masked_audio.wav": audio_input[sample_index],
        "prediction_only_audio.wav": audio_prediction[sample_index],
        "generated_completed_audio.wav": completed,
        "target_reference_audio.wav": audio_target[sample_index],
    }
    reference = audio_target[sample_index]
    for name, latent in audio_items.items():
        waveform = _latent_to_waveform(
            latent,
            reference=reference,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=n_iter,
        )
        if waveform is None:  # pragma: no cover
            artifacts["audio_media_unavailable"] = repr(_LIBROSA_IMPORT_ERROR)
            continue
        path = mode_dir / name
        sf.write(path, waveform, int(sample_rate), subtype="PCM_16")
        artifacts[name.removesuffix(".wav")] = str(path)
    return artifacts


def _write_media_artifacts(
    *,
    mode_dir: Path,
    mode: str,
    metrics: Mapping[str, Any],
    sample_index: int,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    audio_n_iter: int,
    video_fps: int,
    video_seconds: float,
) -> dict[str, str]:
    artifacts: dict[str, str] = {"media_claim_limit": MEDIA_CLAIM_LIMIT}
    artifacts.update(
        _save_visual_media(
            mode_dir=mode_dir,
            mode=mode,
            sample_index=sample_index,
            visual_input=metrics["visual_input"],
            visual_prediction=metrics["visual_prediction"],
            visual_target=metrics["visual_target"],
            visual_loss_mask=metrics["visual_loss_mask"],
            fps=video_fps,
            seconds=video_seconds,
        )
    )
    artifacts.update(
        _save_audio_media(
            mode_dir=mode_dir,
            sample_index=sample_index,
            audio_input=metrics["audio_input"],
            audio_prediction=metrics["audio_prediction"],
            audio_target=metrics["audio_target"],
            audio_loss_mask=metrics["audio_loss_mask"],
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=audio_n_iter,
        )
    )
    return artifacts


def _save_heatmap_grid(
    *,
    path: Path,
    title: str,
    input_values: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    max_feature_dims: int,
) -> None:
    if plt is None:
        _save_heatmap_grid_pil(
            path=path,
            title=title,
            input_values=input_values,
            prediction=prediction,
            target=target,
            mask=mask,
            max_feature_dims=max_feature_dims,
        )
        return
    width = min(int(max_feature_dims), int(target.shape[-1]))
    inp = np.asarray(input_values[:, :width], dtype=np.float32).T
    pred = np.asarray(prediction[:, :width], dtype=np.float32).T
    tgt = np.asarray(target[:, :width], dtype=np.float32).T
    err = np.abs(pred - tgt)
    masked_err = np.where(np.asarray(mask, dtype=bool)[None, :], err, np.nan)
    vmin, vmax = _finite_range(inp, pred, tgt)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), constrained_layout=True)
    for axis, data, subtitle, kwargs in [
        (axes[0], inp, "input latent (observed context; masked regions zeroed)", {"vmin": vmin, "vmax": vmax, "cmap": "viridis"}),
        (axes[1], pred, "predicted latent", {"vmin": vmin, "vmax": vmax, "cmap": "viridis"}),
        (axes[2], tgt, "target latent", {"vmin": vmin, "vmax": vmax, "cmap": "viridis"}),
        (axes[3], masked_err, "absolute error on prediction target positions", {"cmap": "magma"}),
    ]:
        image = axis.imshow(data, aspect="auto", interpolation="nearest", **kwargs)
        axis.set_ylabel("feature dim")
        axis.set_title(subtitle)
        fig.colorbar(image, ax=axis, fraction=0.015, pad=0.01)
    axes[-1].set_xlabel("latent time step")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_curve(path: Path, title: str, audio_mse_t: np.ndarray, visual_mse_t: np.ndarray) -> None:
    if plt is None:
        _save_curve_pil(path, title, audio_mse_t, visual_mse_t)
        return
    fig, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    axis.plot(_nanmean_time(audio_mse_t), label="audio masked MSE/time")
    axis.plot(_nanmean_time(visual_mse_t), label="visual masked MSE/time")
    axis.set_title(title)
    axis.set_xlabel("latent time step")
    axis.set_ylabel("MSE over feature dims")
    axis.legend()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_retrieval_matrix(path: Path, logits: np.ndarray, title: str) -> None:
    if plt is None:
        _save_retrieval_matrix_pil(path, logits, title)
        return
    shuffled = np.roll(logits, shift=1, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, matrix, subtitle in [(axes[0], logits, "real pair logits"), (axes[1], shuffled, "shuffled visual control logits")]:
        image = axis.imshow(matrix, aspect="equal", interpolation="nearest", cmap="coolwarm")
        axis.set_title(subtitle)
        axis.set_xlabel("visual sample")
        axis.set_ylabel("audio sample")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_pca(path: Path, z_world: np.ndarray, audio_len: int, max_points: int, title: str) -> dict[str, Any]:
    if plt is None:
        return _save_pca_pil(path, z_world, audio_len, max_points, title)
    flattened = z_world.reshape(-1, z_world.shape[-1]).astype(np.float32)
    modality = np.tile(np.concatenate([np.zeros(audio_len, dtype=np.int32), np.ones(z_world.shape[1] - audio_len, dtype=np.int32)]), z_world.shape[0])
    if flattened.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(flattened.shape[0], size=max_points, replace=False))
        flattened = flattened[keep]
        modality = modality[keep]
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    explained = (singular_values[:2] ** 2) / np.maximum(np.sum(singular_values**2), 1e-12)
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for value, label, color in [(0, "audio positions", "tab:blue"), (1, "visual positions", "tab:orange")]:
        mask = modality == value
        axis.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.35, label=label, c=color)
    axis.set_title(title)
    axis.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var)")
    axis.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var)")
    axis.legend(markerscale=3)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"plot_created": True, "explained_variance_ratio": [float(explained[0]), float(explained[1])]}


def _mode_metrics(
    *,
    output: Mapping[str, "torch.Tensor"],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    audio_prediction = output["audio_prediction"].detach().float().cpu().numpy()
    visual_prediction = output["visual_prediction"].detach().float().cpu().numpy()
    audio_target = batch["audio_targets"].detach().float().cpu().numpy()
    visual_target = batch["visual_targets"].detach().float().cpu().numpy()
    audio_input = batch["audio_features"].detach().float().cpu().numpy()
    visual_input = batch["visual_features"].detach().float().cpu().numpy()
    audio_mask = batch["audio_loss_mask"].detach().cpu().numpy().astype(bool)
    visual_mask = batch["visual_loss_mask"].detach().cpu().numpy().astype(bool)
    alignment_loss, logits_t = _contrastive_loss(output["audio_repr"], output["visual_repr"])
    logits = logits_t.detach().float().cpu().numpy()
    retrieval = _retrieval_metrics(logits) if logits.shape[0] > 1 else {}
    shuffled = _retrieval_metrics(np.roll(logits, shift=1, axis=1)) if logits.shape[0] > 1 else {}
    return {
        "audio_prediction": audio_prediction,
        "visual_prediction": visual_prediction,
        "audio_target": audio_target,
        "visual_target": visual_target,
        "audio_input": audio_input,
        "visual_input": visual_input,
        "audio_loss_mask": audio_mask,
        "visual_loss_mask": visual_mask,
        "audio_mse": _masked_mse_scalar(audio_prediction, audio_target, audio_mask),
        "visual_mse": _masked_mse_scalar(visual_prediction, visual_target, visual_mask),
        "audio_null_mse": _masked_mse_scalar(audio_input, audio_target, audio_mask),
        "visual_null_mse": _masked_mse_scalar(visual_input, visual_target, visual_mask),
        "alignment_loss": float(alignment_loss.detach().float().cpu().item()),
        "retrieval": retrieval,
        "shuffled_retrieval": {f"shuffled_{key}": value for key, value in shuffled.items()},
        "alignment_logits": logits,
        "z_world": output["z_world"].detach().float().cpu().numpy(),
    }


def _write_mode_artifacts(
    *,
    output_dir: Path,
    mode: str,
    metrics: Mapping[str, Any],
    sample_index: int,
    max_feature_dims: int,
    render_media: bool,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    audio_n_iter: int,
    video_fps: int,
    video_seconds: float,
) -> dict[str, Any]:
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    audio_mse_t = _masked_mse_by_time(metrics["audio_prediction"], metrics["audio_target"], metrics["audio_loss_mask"])
    visual_mse_t = _masked_mse_by_time(metrics["visual_prediction"], metrics["visual_target"], metrics["visual_loss_mask"])
    np.savez_compressed(
        mode_dir / "latent_predictions.npz",
        audio_prediction=metrics["audio_prediction"],
        visual_prediction=metrics["visual_prediction"],
        audio_target=metrics["audio_target"],
        visual_target=metrics["visual_target"],
        audio_loss_mask=metrics["audio_loss_mask"],
        visual_loss_mask=metrics["visual_loss_mask"],
        alignment_logits=metrics["alignment_logits"],
    )
    artifacts["latent_npz"] = str(mode_dir / "latent_predictions.npz")
    if plt is not None or Image is not None:
        audio_plot = mode_dir / "audio_latent_prediction.png"
        visual_plot = mode_dir / "visual_latent_prediction.png"
        curve_plot = mode_dir / "latent_mse_over_time.png"
        matrix_plot = mode_dir / "av_retrieval_matrix.png"
        pca_plot = mode_dir / "latent_world_pca.png"
        _save_heatmap_grid(
            path=audio_plot,
            title=f"{mode}: audio latent diagnostic sample {sample_index}",
            input_values=metrics["audio_input"][sample_index],
            prediction=metrics["audio_prediction"][sample_index],
            target=metrics["audio_target"][sample_index],
            mask=metrics["audio_loss_mask"][sample_index],
            max_feature_dims=max_feature_dims,
        )
        _save_heatmap_grid(
            path=visual_plot,
            title=f"{mode}: visual latent diagnostic sample {sample_index}",
            input_values=metrics["visual_input"][sample_index],
            prediction=metrics["visual_prediction"][sample_index],
            target=metrics["visual_target"][sample_index],
            mask=metrics["visual_loss_mask"][sample_index],
            max_feature_dims=max_feature_dims,
        )
        _save_curve(curve_plot, f"{mode}: masked latent prediction MSE over time", audio_mse_t, visual_mse_t)
        _save_retrieval_matrix(matrix_plot, metrics["alignment_logits"], f"{mode}: AV retrieval diagnostic")
        pca = _save_pca(pca_plot, metrics["z_world"], metrics["audio_target"].shape[1], 2048, f"{mode}: latent world PCA")
        artifacts.update(
            {
                "audio_heatmap": str(audio_plot),
                "visual_heatmap": str(visual_plot),
                "mse_curve": str(curve_plot),
                "retrieval_matrix": str(matrix_plot),
                "latent_world_pca": str(pca_plot),
                "pca_summary": pca,
            }
        )
    else:
        artifacts["plotting_unavailable"] = f"matplotlib={_PLOT_IMPORT_ERROR!r}; pillow={_PIL_IMPORT_ERROR!r}"
    if render_media:
        artifacts["diagnostic_media"] = _write_media_artifacts(
            mode_dir=mode_dir,
            mode=mode,
            metrics=metrics,
            sample_index=sample_index,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            audio_n_iter=audio_n_iter,
            video_fps=video_fps,
            video_seconds=video_seconds,
        )
    return artifacts


def _report_text(summary: Mapping[str, Any]) -> str:
    lines = [
        "# T2 Latent Exploration Report",
        "",
        f"**Claim limit:** {CLAIM_LIMIT}",
        f"**Media renderer limit:** {summary.get('media_claim_limit') or 'disabled'}",
        "",
        "## Run",
        "",
        f"- experiment: `{summary['experiment_id']}`",
        f"- checkpoint: `{summary['checkpoint_path']}`",
        f"- config hash: `{summary.get('config_hash')}`",
        f"- split: `{summary['split']}`",
        f"- batch size: `{summary['batch_size']}`",
        f"- device: `{summary['device']}`",
        f"- dtype: `{summary['dtype']}`",
        f"- dataset manifest hash: `{summary.get('dataset_manifest_sha256')}`",
        f"- feature spec: `{summary.get('feature_spec')}`",
        f"- dataset mix: `{summary.get('dataset_mix')}`",
        f"- security scope: `{summary.get('security_scope')}`",
        "",
        "## Model / compute summary",
        "",
        f"- model: `{summary.get('model_summary', {}).get('name')}`",
        f"- variant: `{summary.get('model_summary', {}).get('variant')}`",
        f"- context length: `{summary.get('model_summary', {}).get('context_length')}`",
        f"- latent dim: `{summary.get('model_summary', {}).get('latent_dim')}`",
        f"- layers / heads: `{summary.get('model_summary', {}).get('n_layer')}` / `{summary.get('model_summary', {}).get('n_head')}`",
        f"- estimated parameter count: `{summary.get('model_summary', {}).get('estimated_parameter_count')}`",
        f"- training compute summary: `{summary.get('training_compute_summary')}`",
        f"- dataset split counts: `{summary.get('dataset_split_counts')}`",
        f"- diagnostic media renderer: `{summary.get('diagnostic_media_renderer')}`",
        "",
        "## Samples",
        "",
    ]
    for sample_id, label in zip(summary["sample_ids"], summary["labels"], strict=False):
        lines.append(f"- `{sample_id}` — `{label}`")
    lines.extend(["", "## Mode metrics", ""])
    for mode, payload in summary["modes"].items():
        lines.extend(
            [
                f"### `{mode}`",
                "",
                f"- audio latent MSE: `{payload['metrics']['audio_mse']:.6f}`",
                f"- visual latent MSE: `{payload['metrics']['visual_mse']:.6f}`",
                f"- audio null/persistence MSE: `{payload['metrics']['audio_null_mse']:.6f}`",
                f"- visual null/persistence MSE: `{payload['metrics']['visual_null_mse']:.6f}`",
                f"- alignment loss: `{payload['metrics']['alignment_loss']:.6f}`",
                f"- retrieval: `{payload['metrics']['retrieval']}`",
                f"- shuffled retrieval: `{payload['metrics']['shuffled_retrieval']}`",
                "",
            ]
        )
        for name, path in payload["artifacts"].items():
            if isinstance(path, str):
                lines.append(f"- {name}: `{path}`")
            if isinstance(path, Mapping):
                lines.append(f"- {name}:")
                for sub_name, sub_path in path.items():
                    lines.append(f"  - {sub_name}: `{sub_path}`")
        lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "- These are latent-field diagnostics, not learned/raw-decoder audio/video generation.",
            "- Heatmaps visualize model-predicted latent tensors and errors.",
            "- Cross-modal modes are diagnostic/off-distribution probes unless evaluated separately.",
            "- Tiny VGGSound validation/test splits prevent strong natural-AV conclusions.",
            "- Optional WAV/PNG/MP4 files are heuristic renders of predicted latents, not learned raw-media decodes.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    _require_torch()
    exp_dir = T2_EXPERIMENT_POLICY.validate(args.experiment_dir, must_exist=True)
    config_path = exp_dir / "config.json"
    config = _load_json(config_path)
    metrics_path = exp_dir / "metrics.json"
    run_metrics = _load_json(metrics_path) if metrics_path.exists() else {}
    experiment_id = str(config.get("registry", {}).get("experiment_id", exp_dir.name))
    prepared_dir = T2_PREPARED_POLICY.validate(config["data"]["prepared_corpus_dir"], must_exist=True)
    dataset_card_path = prepared_dir / "dataset_card.json"
    dataset_card = _load_json(dataset_card_path) if dataset_card_path.exists() else {}
    dataset = T2PreparedLatentDataset.load(prepared_dir, split=args.split)
    batch_size = min(int(args.batch_size), dataset.sample_count) if args.no_duplicate_small_split else int(args.batch_size)
    raw_batch = dataset.batch(
        batch_size=batch_size,
        seed=int(args.seed),
        step=int(args.step),
        context_fraction=float(args.context_fraction),
        missing_span_fraction=float(args.missing_span_fraction),
        sequential=bool(args.sequential),
    )
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.precision, device)
    model = _load_model(exp_dir, config, device, dtype)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_base = EXPLORATION_OUTPUT_POLICY.validate(args.output_dir)
    output_dir = output_base / f"{experiment_id}_{args.split}_step{int(args.step):04d}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "claim_limit": CLAIM_LIMIT,
        "media_claim_limit": MEDIA_CLAIM_LIMIT if bool(args.render_media) else None,
        "security_scope": "Loads trusted local PyTorch checkpoints only from experiments/multimodal/t2 and writes exploration artifacts only under experiments/exploration.",
        "experiment_id": experiment_id,
        "checkpoint_path": str(exp_dir / "checkpoint.pt"),
        "config_path": str(config_path),
        "config_hash": run_metrics.get("config_hash"),
        "prepared_corpus_dir": str(prepared_dir),
        "dataset_card_path": str(dataset_card_path),
        "dataset_manifest_sha256": dataset_card.get("manifest_sha256"),
        "dataset_split_counts": dataset_card.get("split_counts"),
        "split": args.split,
        "batch_size": batch_size,
        "dataset_sample_count": dataset.sample_count,
        "model_summary": {
            "name": config.get("model", {}).get("name"),
            "variant": config.get("model", {}).get("variant"),
            "context_length": config.get("model", {}).get("context_length"),
            "latent_dim": config.get("model", {}).get("latent_dim"),
            "n_layer": config.get("model", {}).get("n_layer"),
            "n_head": config.get("model", {}).get("n_head"),
            "estimated_parameter_count": run_metrics.get("estimated_parameter_count"),
        },
        "feature_spec": config.get("data", {}).get("feature_spec"),
        "dataset_mix": config.get("data", {}).get("dataset_mix"),
        "training_compute_summary": {
            "configured_steps": config.get("training", {}).get("steps"),
            "configured_batch_size": config.get("training", {}).get("batch_size"),
            "configured_precision": config.get("runtime", {}).get("precision"),
            "accelerator_memory_mb": run_metrics.get("accelerator_memory_mb"),
            "max_memory_mb": run_metrics.get("max_memory_mb"),
        },
        "diagnostic_media_renderer": {
            "enabled": bool(args.render_media),
            "audio_renderer": "approximate normalized log-mel -> dB percentile scaling -> Griffin-Lim" if bool(args.render_media) else None,
            "visual_renderer": "grayscale row-latent [-1,1] -> uint8 frame; repeated/panel video" if bool(args.render_media) else None,
            "sample_rate": int(args.sample_rate),
            "n_fft": int(args.n_fft),
            "hop_length": int(args.hop_length),
            "audio_n_iter": int(args.audio_n_iter),
            "video_fps": int(args.video_fps),
            "video_seconds": float(args.video_seconds),
        },
        "seed": int(args.seed),
        "step": int(args.step),
        "context_fraction": float(args.context_fraction),
        "missing_span_fraction": float(args.missing_span_fraction),
        "device": str(device),
        "dtype": str(dtype),
        "batch_contract_summary": t2_batch_contract_summary(raw_batch),
        "sample_ids": raw_batch["sample_ids"],
        "labels": raw_batch["labels"],
        "modes": {},
    }
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        probe_np = _mode_batch(raw_batch, mode)
        probe = _cpu_float_batch(probe_np, device=device, dtype=dtype)
        with torch.no_grad():
            output = model(probe)
        metrics = _mode_metrics(output=output, batch=probe)
        artifacts = _write_mode_artifacts(
            output_dir=output_dir,
            mode=mode,
            metrics=metrics,
            sample_index=min(int(args.sample_index), batch_size - 1),
            max_feature_dims=int(args.max_feature_dims),
            render_media=bool(args.render_media),
            sample_rate=int(args.sample_rate),
            n_fft=int(args.n_fft),
            hop_length=int(args.hop_length),
            audio_n_iter=int(args.audio_n_iter),
            video_fps=int(args.video_fps),
            video_seconds=float(args.video_seconds),
        )
        public_metrics = {
            key: value
            for key, value in metrics.items()
            if key
            not in {
                "audio_prediction",
                "visual_prediction",
                "audio_target",
                "visual_target",
                "audio_input",
                "visual_input",
                "audio_loss_mask",
                "visual_loss_mask",
                "alignment_logits",
                "z_world",
            }
        }
        summary["modes"][mode] = {"metrics": public_metrics, "artifacts": artifacts}
    atomic_write_json(summary, output_dir / "manifest.json")
    atomic_write_text(_report_text(summary), output_dir / "report.md")
    print(json.dumps({"event": "t2_latent_exploration", "output_dir": str(output_dir), "claim_limit": CLAIM_LIMIT}, sort_keys=True))
    return output_dir


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", default="experiments/multimodal/t2/EXP-T2-111")
    parser.add_argument("--output-dir", default="experiments/exploration/t2_latent_generation")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12402)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--context-fraction", type=float, default=0.5)
    parser.add_argument("--missing-span-fraction", type=float, default=0.15)
    parser.add_argument("--modes", default="masked_av,audio_to_visual,visual_to_audio")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--max-feature-dims", type=int, default=96)
    parser.add_argument("--render-media", action="store_true", help="Write approximate diagnostic WAV/PNG/GIF/MP4 renders from predicted latents.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--audio-n-iter", type=int, default=32)
    parser.add_argument("--video-fps", type=int, default=2)
    parser.add_argument("--video-seconds", type=float, default=4.0)
    parser.add_argument("--sequential", action="store_true", default=True)
    parser.add_argument("--random-sample", dest="sequential", action="store_false")
    parser.add_argument("--no-duplicate-small-split", action="store_true", default=True)
    parser.add_argument("--allow-duplicate-small-split", dest="no_duplicate_small_split", action="store_false")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    main()
