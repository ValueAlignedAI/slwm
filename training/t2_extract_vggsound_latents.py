"""Extract T2 latents from downloaded VGGSound 10-second media clips.

This consumes the availability report produced by
``/data/slwm/scripts/download_vggsound_clips.py``. It writes precomputed audio
log-mel and visual grayscale patch latents plus a manifest compatible with
``training.t2_prepare_latents``. Captions/text are not used.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import librosa
import numpy as np
from PIL import Image

from utils.config import config_hash, load_config


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
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def _latest_available_rows(report_path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for line in report_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row.get("sample_id", ""))
        if sample_id:
            latest[sample_id] = row
    rows = [row for row in latest.values() if row.get("status") in {"downloaded", "already_exists"} and row.get("path")]
    rows.sort(key=lambda row: int(row.get("index", 0)))
    return rows


def _ffmpeg_audio(path: Path, *, sample_rate: int, duration_seconds: float) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-t",
        str(float(duration_seconds)),
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _audio_logmel(y: np.ndarray, *, sample_rate: int, duration_seconds: float, n_mels: int, hop_length: int, n_fft: int) -> np.ndarray:
    max_samples = int(sample_rate * duration_seconds)
    if y.shape[0] > max_samples:
        y = y[:max_samples]
    if y.shape[0] < max_samples:
        y = np.pad(y, (0, max_samples - int(y.shape[0])))
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)
    return ((logmel - float(np.mean(logmel))) / (float(np.std(logmel)) or 1.0)).astype(np.float32)


def _ffmpeg_frame(path: Path, *, visual_size: int, timestamp: float) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        str(float(timestamp)),
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    image = Image.open(io.BytesIO(proc.stdout)).convert("L").resize((visual_size, visual_size))
    return (np.asarray(image, dtype=np.float32) / 127.5 - 1.0).astype(np.float32)


def extract_vggsound_latents(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(config_path)
    config = load_config(path)
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}
    report_input = Path(str(data_cfg.get("media_availability_path", "/data/slwm/t2_signal_latents/manifests/vggsound_5k_v0/media_availability.jsonl")))
    output_root = Path(str(data_cfg.get("output_feature_root", "/data/slwm/t2_signal_latents/extracted_features/vggsound_5k_v0")))
    manifest_path = Path(str(data_cfg.get("manifest_path", "/data/slwm/t2_signal_latents/manifests/vggsound_5k_v0/latent_manifest.jsonl")))
    report_path = Path(str(data_cfg.get("report_path", "/data/slwm/t2_signal_latents/manifests/vggsound_5k_v0/latent_extraction_report.json")))
    max_samples = int(data_cfg.get("max_samples", 6000))
    feature_spec = data_cfg.get("feature_spec", {}) if isinstance(data_cfg.get("feature_spec", {}), Mapping) else {}
    sample_rate = int(feature_spec.get("target_sample_rate", 16000))
    duration_seconds = float(feature_spec.get("duration_seconds", 10.0))
    n_mels = int(feature_spec.get("audio_feature_dim", 80))
    hop_length = int(feature_spec.get("hop_length", 160))
    n_fft = int(feature_spec.get("n_fft", 400))
    visual_size = int(feature_spec.get("visual_feature_dim", 256))
    frame_timestamp = float(feature_spec.get("frame_timestamp_seconds", min(5.0, duration_seconds / 2.0)))
    rows = _latest_available_rows(report_input)[:max_samples]
    summary: dict[str, Any] = {
        "status": "dry_run_complete" if dry_run else "running",
        "config_path": str(path),
        "config_hash": config_hash(config),
        "media_availability_path": str(report_input),
        "available_media_rows": len(rows),
        "manifest_path": str(manifest_path),
        "output_feature_root": str(output_root),
        "claim_scope": "VGGSound downloaded-media latent extraction; YouTube availability filtered; no text objective.",
        "caption_use": "none",
    }
    if dry_run:
        _write_json(summary, report_path)
        return summary
    audio_dir = output_root / "audio"
    visual_dir = output_root / "visual"
    audio_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        media_path = Path(str(row["path"]))
        audio_path = audio_dir / f"{sample_id}.audio.npy"
        visual_path = visual_dir / f"{sample_id}.visual.npy"
        try:
            if not audio_path.exists():
                audio = _ffmpeg_audio(media_path, sample_rate=sample_rate, duration_seconds=duration_seconds)
                np.save(audio_path, _audio_logmel(audio, sample_rate=sample_rate, duration_seconds=duration_seconds, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft))
            if not visual_path.exists():
                np.save(visual_path, _ffmpeg_frame(media_path, visual_size=visual_size, timestamp=frame_timestamp))
        except Exception as exc:  # pragma: no cover - media dependent
            failures.append({"sample_id": sample_id, "path": str(media_path), "error": repr(exc)})
            continue
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "split": str(row.get("split", "train")),
                "audio_path": str(audio_path),
                "visual_path": str(visual_path),
                "dataset": "VGGSound_5k_available_clips_v0",
                "license": "VGGSound metadata CC BY 4.0; original YouTube video rights remain with owners",
                "label": str(row.get("label", "")),
                "metadata": {
                    "source_tag": "observed",
                    "natural_av_correspondence": True,
                    "youtube_id": row.get("youtube_id"),
                    "start_seconds": row.get("start_seconds"),
                    "end_seconds": row.get("end_seconds"),
                    "media_path": str(media_path),
                    "caption_use": "none",
                },
            }
        )
    _write_jsonl(manifest_rows, manifest_path)
    summary.update(
        {
            "status": "extracted",
            "written_records": len(manifest_rows),
            "failure_count": len(failures),
            "failures": failures[:50],
            "wall_clock_time_seconds": time.perf_counter() - started,
        }
    )
    _write_json(summary, report_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract VGGSound media latents")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(extract_vggsound_latents(args.config, dry_run=args.dry_run), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
