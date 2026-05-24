"""Extract first-pass external T2 audio/visual latent controls from local raw data.

This module intentionally creates **latent features**, not raw-media training
objectives.  The default control dataset pairs LibriSpeech audio with COCO images
deterministically so both T2 modalities are present for loader/training work.
Those pairs are marked as cross-source controls and must not be claimed as
natural audio-video correspondence.  Natural paired AV still requires a separate
VGGSound/MSR-VTT-style manifest and ToS/legal review.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from utils.config import config_hash, load_config


@dataclass(frozen=True)
class AudioMember:
    archive: Path
    member: str


@dataclass(frozen=True)
class VisualMember:
    archive: Path
    member: str


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


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _list_audio_members(archives: list[Path]) -> list[AudioMember]:
    members: list[AudioMember] = []
    for archive in archives:
        if not archive.exists():
            raise FileNotFoundError(f"Missing audio archive: {archive}")
        with tarfile.open(archive, "r:*") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith((".flac", ".wav")):
                    members.append(AudioMember(archive=archive, member=member.name))
    members.sort(key=lambda item: (item.archive.name, item.member))
    return members


def _list_visual_members(archives: list[Path]) -> list[VisualMember]:
    members: list[VisualMember] = []
    for archive in archives:
        if not archive.exists():
            raise FileNotFoundError(f"Missing visual archive: {archive}")
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if not name.endswith("/") and name.lower().endswith((".jpg", ".jpeg", ".png")):
                    members.append(VisualMember(archive=archive, member=name))
    members.sort(key=lambda item: (item.archive.name, item.member))
    return members


def _stable_split(index: int, split_counts: Mapping[str, int]) -> str:
    train = int(split_counts.get("train", 0))
    validation = int(split_counts.get("validation", 0))
    if index < train:
        return "train"
    if index < train + validation:
        return "validation"
    return "test"


def _read_audio_member_from_tar(tar: tarfile.TarFile, member_name: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    extracted = tar.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Could not extract {member_name}")
    payload = extracted.read()
    audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return array, int(sample_rate)


def _audio_logmel(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    target_sample_rate: int,
    duration_seconds: float,
    n_mels: int,
    hop_length: int,
    n_fft: int,
) -> np.ndarray:
    import librosa

    y = np.asarray(waveform, dtype=np.float32)
    if sample_rate != target_sample_rate:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=target_sample_rate).astype(np.float32)
    max_samples = int(float(duration_seconds) * int(target_sample_rate))
    if y.shape[0] > max_samples:
        y = y[:max_samples]
    if y.shape[0] < max_samples:
        y = np.pad(y, (0, max_samples - int(y.shape[0])))
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=int(target_sample_rate),
        n_fft=int(n_fft),
        hop_length=int(hop_length),
        n_mels=int(n_mels),
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).T.astype(np.float32)
    # Normalize per sample for stable first-pass latent scale.
    mean = float(np.mean(logmel))
    std = float(np.std(logmel)) or 1.0
    return ((logmel - mean) / std).astype(np.float32)


def _read_visual_member_from_zip(zf: zipfile.ZipFile, member_name: str, *, size: int) -> np.ndarray:
    payload = zf.read(member_name)
    image = Image.open(io.BytesIO(payload)).convert("L").resize((int(size), int(size)))
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    # Shape [T_visual=rows, D_visual=columns], e.g. [256, 256].
    return array.astype(np.float32)


def extract_external_t2_latents(config_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    start = time.perf_counter()
    path = Path(config_path)
    config = load_config(path)
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}
    raw_root = Path(str(data_cfg.get("raw_root", "/data/slwm/raw")))
    output_root = Path(str(data_cfg.get("output_feature_root", "/data/slwm/t2_signal_latents/extracted_features/external_t2_control_v0")))
    manifest_path = Path(str(data_cfg.get("manifest_path", "/data/slwm/t2_signal_latents/manifests/external_t2_control_v0/manifest.jsonl")))
    report_path = Path(str(data_cfg.get("report_path", manifest_path.with_name("extraction_report.json"))))

    audio_archives = [raw_root / str(item) for item in data_cfg.get("audio_archives", [])]
    visual_archives = [raw_root / str(item) for item in data_cfg.get("visual_archives", [])]
    max_samples = int(data_cfg.get("max_samples", 6000))
    split_counts = data_cfg.get("split_counts", {"train": 5000, "validation": 500, "test": 500})
    if not isinstance(split_counts, Mapping):
        raise ValueError("data.split_counts must be an object")
    if sum(int(split_counts.get(name, 0)) for name in ("train", "validation", "test")) != max_samples:
        raise ValueError("data.split_counts must sum to data.max_samples")

    audio_members = _list_audio_members(audio_archives)
    visual_members = _list_visual_members(visual_archives)
    selected_count = min(max_samples, len(audio_members), len(visual_members))
    if selected_count <= 0:
        raise ValueError("No audio/visual members available for T2 extraction")
    if selected_count != max_samples:
        raise ValueError(f"Requested {max_samples} samples but only {selected_count} paired control records are available")

    feature_spec = data_cfg.get("feature_spec", {}) if isinstance(data_cfg.get("feature_spec", {}), Mapping) else {}
    target_sample_rate = int(feature_spec.get("target_sample_rate", 16000))
    duration_seconds = float(feature_spec.get("duration_seconds", 6.0))
    n_mels = int(feature_spec.get("audio_feature_dim", 80))
    hop_length = int(feature_spec.get("hop_length", 160))
    n_fft = int(feature_spec.get("n_fft", 400))
    visual_size = int(feature_spec.get("visual_feature_dim", 256))

    report: dict[str, Any] = {
        "status": "planned" if dry_run else "running",
        "config_path": str(path),
        "config_hash": config_hash(config),
        "raw_root": str(raw_root),
        "output_feature_root": str(output_root),
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "audio_archives": [str(item) for item in audio_archives],
        "visual_archives": [str(item) for item in visual_archives],
        "available_audio_members": len(audio_members),
        "available_visual_members": len(visual_members),
        "selected_samples": selected_count,
        "split_counts": {name: int(split_counts.get(name, 0)) for name in ("train", "validation", "test")},
        "feature_spec": {
            "audio_codec_or_features": "frozen_log_mel_80bin_16khz_control",
            "visual_codec_or_features": "deterministic_coco_grayscale_row_patch_latents_control",
            "target_sample_rate": target_sample_rate,
            "duration_seconds": duration_seconds,
            "audio_feature_dim": n_mels,
            "visual_feature_dim": visual_size,
            "hop_length": hop_length,
            "n_fft": n_fft,
        },
        "claim_scope": "cross-source audio/image control latents only; natural_av_correspondence=false; not a T2 grounding claim",
        "caption_use": "metadata_only; COCO captions/annotations are not read by this extractor",
        "r_r0_9_status": "split/source-id records written; duplicate/leakage/contamination checks not complete",
    }
    if dry_run:
        report["status"] = "dry_run_complete"
        _write_json(report, report_path)
        return report

    audio_dir = output_root / "audio"
    visual_dir = output_root / "visual"
    audio_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    tar_handles: dict[Path, tarfile.TarFile] = {}
    zip_handles: dict[Path, zipfile.ZipFile] = {}
    try:
        for archive in sorted({item.archive for item in audio_members[:selected_count]}):
            tar_handles[archive] = tarfile.open(archive, "r:*")
        for archive in sorted({item.archive for item in visual_members[:selected_count]}):
            zip_handles[archive] = zipfile.ZipFile(archive)

        for index in range(selected_count):
            audio_member = audio_members[index]
            visual_member = visual_members[index]
            sample_id = f"external-control-{index:06d}"
            split = _stable_split(index, split_counts)
            audio_path = audio_dir / f"{sample_id}.audio.npy"
            visual_path = visual_dir / f"{sample_id}.visual.npy"
            try:
                if not audio_path.exists():
                    waveform, sample_rate = _read_audio_member_from_tar(tar_handles[audio_member.archive], audio_member.member)
                    np.save(
                        audio_path,
                        _audio_logmel(
                            waveform,
                            sample_rate,
                            target_sample_rate=target_sample_rate,
                            duration_seconds=duration_seconds,
                            n_mels=n_mels,
                            hop_length=hop_length,
                            n_fft=n_fft,
                        ),
                    )
                if not visual_path.exists():
                    np.save(visual_path, _read_visual_member_from_zip(zip_handles[visual_member.archive], visual_member.member, size=visual_size))
            except Exception as exc:  # pragma: no cover - data-dependent failures
                failures.append({"sample_id": sample_id, "error": repr(exc)})
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "audio_path": str(audio_path),
                    "visual_path": str(visual_path),
                    "dataset": "LibriSpeech_COCO2017_cross_source_control",
                    "license": "LibriSpeech CC BY 4.0; COCO dataset/image terms; captions metadata-only and not used",
                    "label": "cross_source_control_pair",
                    "metadata": {
                        "source_tag": "observed",
                        "natural_av_correspondence": False,
                        "pairing_policy": "deterministic_sorted_cross_source_control",
                        "audio_archive": str(audio_member.archive),
                        "audio_member": audio_member.member,
                        "visual_archive": str(visual_member.archive),
                        "visual_member": visual_member.member,
                        "source_id_hash": _sha256_text(audio_member.member + "|" + visual_member.member),
                        "caption_use": "none",
                    },
                }
            )
    finally:
        for handle in tar_handles.values():
            handle.close()
        for handle in zip_handles.values():
            handle.close()

    _write_jsonl(rows, manifest_path)
    report.update(
        {
            "status": "extracted",
            "written_records": len(rows),
            "failures": failures[:50],
            "failure_count": len(failures),
            "wall_clock_time_seconds": time.perf_counter() - start,
        }
    )
    _write_json(report, report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract local raw T2 audio/visual control latents")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(extract_external_t2_latents(args.config, dry_run=args.dry_run), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
