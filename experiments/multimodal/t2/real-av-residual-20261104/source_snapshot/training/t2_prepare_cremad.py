"""Reproducible real paired video/audio features for the narrow T2 pilot.

Each clip becomes audio [32,64] and visual [32,64]. PCA/normalization are fit
only on training people; a frame encoder never consumes neighboring frames.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

REPO = "CheyneyComputerScience/CREMA-D"
EMOTIONS = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")
SENTENCES = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.", "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.", "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_clips(names: list[str], cfg: dict) -> tuple[list[dict], dict]:
    """Return selected clip metadata and disjoint person splits; no arrays change."""
    rng = np.random.default_rng(cfg["seed"])
    if "actor_splits" in cfg:
        splits = cfg["actor_splits"]
    else:
        actors = sorted({name.split("_")[0] for name in names})
        rng.shuffle(actors)
        actors = actors[:cfg["actor_count"]]
        nt, nv = cfg["train_actor_count"], cfg["validation_actor_count"]
        splits = {"train": actors[:nt], "validation": actors[nt:nt+nv], "test": actors[nt+nv:]}
    if not all(splits.values()):
        raise ValueError("All person-disjoint splits must be nonempty")
    groups = [set(v) for v in splits.values()]
    if any(groups[i]&groups[j] for i in range(len(groups)) for j in range(i)):
        raise ValueError("People overlap across splits")
    selected = []
    for split, members in splits.items():
        for actor in members:
            for emotion in EMOTIONS:
                pool = sorted(n for n in names if n.startswith(actor+"_") and n.split("_")[2] == emotion)
                rng.shuffle(pool)
                for name in pool[:cfg["clips_per_emotion_per_actor"]]:
                    sentence = name.split("_")[1]
                    selected.append({"sample_id": name, "actor": actor, "emotion": emotion,
                                     "sentence_code": sentence, "transcript": SENTENCES[sentence], "split": split})
    return selected, splits


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 512:
        return path
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                payload = response.read()
            if len(payload) <= 512 or payload.startswith(b"version https://git-lfs"):
                raise ValueError("Missing media or LFS pointer")
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(path)
            return path
        except Exception as exc:
            last = exc
            time.sleep(attempt + 1)
    raise RuntimeError(f"Download failed: {url}: {last}")


def decode_clip(path: Path, steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    """FLV -> RGB uint8 [T,128,128,3], mono float32 [N], duration seconds.

    ffmpeg decodes silent video and audio in separate calls. Sampling selects
    real frames at normalized time; it never synthesizes either modality.
    """
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                           capture_output=True, check=True, timeout=30)
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    if not 0.7 <= duration <= 8.0:
        raise ValueError(f"Unexpected duration {duration}")
    video = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-an", "-vf", "scale=128:128",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "1", "-"], capture_output=True, check=True, timeout=45)
    frames = np.frombuffer(video.stdout, np.uint8).reshape(-1, 128, 128, 3)
    if len(frames) < 8:
        raise ValueError("Too few real video frames")
    frames = frames[np.linspace(0, len(frames)-1, steps).round().astype(int)].copy()
    sound = subprocess.run(["ffmpeg", "-v", "error", "-threads", "1", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                            "-f", "f32le", "-"], capture_output=True, check=True, timeout=30)
    waveform = np.frombuffer(sound.stdout, np.float32).copy()
    if len(waveform) < 8000 or float(np.std(waveform)) < 1e-6:
        raise ValueError("Missing or silent audio")
    return frames, waveform, duration


def fit_normalizer(audio: np.ndarray, visual: np.ndarray, train: np.ndarray, dim: int) -> dict[str, np.ndarray]:
    """Fit training-only [N,T,A/V] moments and visual [V,dim] PCA projection."""
    a = audio[train].reshape(-1, audio.shape[-1]).astype(np.float64)
    v = visual[train].reshape(-1, visual.shape[-1]).astype(np.float64)
    vm = v.mean(0)
    _, _, basis = np.linalg.svd((v-vm).T @ (v-vm), full_matrices=False)
    pca = basis[:dim].T
    p = (v-vm) @ pca
    return {"audio_mean": a.mean(0).astype(np.float32), "audio_std": a.std(0).clip(.1).astype(np.float32),
            "visual_mean": vm.astype(np.float32), "visual_pca": pca.astype(np.float32),
            "visual_std": p.std(0).clip(.1).astype(np.float32)}


def log_mel_sequence(waveform: torch.Tensor, mel: torch.nn.Module, steps: int) -> torch.Tensor:
    """Waveform [N] -> [T,64], with no FFT window crossing the midpoint.

    Context and target halves are featurized separately, so a 25ms FFT window
    near the observation boundary cannot include any unobserved future audio.
    """
    if steps % 2:
        raise ValueError("T2 midpoint-safe features require an even step count")
    halves = (waveform[:len(waveform)//2], waveform[len(waveform)//2:])
    return torch.cat([F.interpolate(mel(part).clamp_min(1e-8).log()[None],
                                    size=steps//2, mode="area")[0].T for part in halves], dim=0)


def prepare(config_path: str) -> dict:
    from torchvision.models import resnet18, ResNet18_Weights
    from torchaudio.transforms import MelSpectrogram

    cfg = json.loads(Path(config_path).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "prepare_config.json").write_text(json.dumps(cfg, indent=2)+"\n")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    revision = cfg["source_revision"]
    rawbase = f"https://raw.githubusercontent.com/{REPO}/{revision}"
    for name in ("SentenceFilenames.csv", "LICENSE.txt", "README.md"):
        (out / name).write_bytes(urllib.request.urlopen(f"{rawbase}/{name}", timeout=30).read())
    names = [r["Filename"] for r in csv.DictReader(io.StringIO((out/"SentenceFilenames.csv").read_text()))]
    records, splits = select_clips(names, cfg)
    reuse = Path(cfg["reuse_corpus_dir"]) if cfg.get("reuse_corpus_dir") else None
    reused_rows = []
    if reuse:
        old_card = json.loads((reuse/"dataset_card.json").read_text())
        if set(splits["test"]) & set(sum(old_card["actor_splits"].values(), [])):
            raise ValueError("Follow-up test people must be completely unused in the discovery corpus")
        for split in ("train","validation"):
            if splits[split] != old_card["actor_splits"][split]:
                raise ValueError("Follow-up must retain training and validation people exactly")
            reused_rows.extend(json.loads((reuse/f"{split}.json").read_text()))
        records = [r for r in records if r["split"] == "test"]
    (out / "selection.json").write_text(json.dumps({"records": records, "actor_splits": splits}, indent=2)+"\n")
    failures = []
    good = []
    def fetch(row):
        path = out/"raw"/(row["sample_id"]+".flv")
        url = f"https://media.githubusercontent.com/media/{REPO}/{revision}/VideoFlash/{path.name}"
        download(url, path)
        return {**row, "raw_path": str(path), "raw_sha256": sha256(path), "source_url": url}
    with ThreadPoolExecutor(max_workers=cfg["download_workers"]) as pool:
        futures = {pool.submit(fetch, r): r for r in records}
        for i, future in enumerate(as_completed(futures)):
            try:
                good.append(future.result())
            except Exception as exc:
                failures.append({"sample_id": futures[future]["sample_id"], "stage": "download", "error": str(exc)})
            if (i+1) % 100 == 0:
                print(json.dumps({"downloaded": len(good), "failed": len(failures), "total": len(records)}), flush=True)
    good.sort(key=lambda r: r["sample_id"])
    device = torch.device(cfg.get("device", "cuda"))
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    frozen_params = sum(p.numel() for p in backbone.parameters())
    backbone.fc = torch.nn.Identity()
    mel = MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=64,
                         center=False, power=2).to(device)
    image_mean = torch.tensor([.485,.456,.406], device=device)[None,:,None,None]
    image_std = torch.tensor([.229,.224,.225], device=device)[None,:,None,None]
    def decode(row):
        return row, decode_clip(Path(row["raw_path"]), cfg["steps_per_clip"])
    kept, audios, visuals = [], [], []
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=cfg["decode_workers"]) as pool:
        # executor.map consumes bounded-size chunks to avoid retaining all raw videos.
        for begin in range(0, len(good), 32):
            futures = [pool.submit(decode, r) for r in good[begin:begin+32]]
            for future, row in zip(futures, good[begin:begin+32]):
                try:
                    _, (frames, waveform, duration) = future.result()
                    x = torch.from_numpy(frames).to(device).permute(0,3,1,2).float()/255.
                    x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
                    features = backbone((x-image_mean)/image_std).cpu().numpy()
                    m = log_mel_sequence(torch.from_numpy(waveform).to(device), mel, cfg["steps_per_clip"]).cpu().numpy()
                    if not np.isfinite(features).all() or not np.isfinite(m).all():
                        raise ValueError("Nonfinite features")
                    kept.append({**row, "duration_seconds": duration})
                    audios.append(m)
                    visuals.append(features)
                except Exception as exc:
                    failures.append({"sample_id": row["sample_id"], "stage": "decode", "error": str(exc)})
            print(json.dumps({"features": len(kept), "failed": len(failures), "total": len(good)}), flush=True)
    if len(kept) < .9 * len(records):
        (out/"failures.json").write_text(json.dumps(failures, indent=2))
        raise RuntimeError("More than 10% of selected real media failed; inspect failures")
    audio, visual = np.stack(audios), np.stack(visuals)
    train = np.array([r["split"] == "train" for r in kept])
    if reuse:
        if sha256(reuse/"normalizer.npz") != old_card["files"]["normalizer.npz"]:
            raise ValueError("Frozen discovery normalizer changed")
        with np.load(reuse/"normalizer.npz",allow_pickle=False) as archive:
            normalizer = {k:archive[k] for k in archive.files}
        shutil.copy2(reuse/"normalizer.npz",out/"normalizer.npz")
    else:
        normalizer = fit_normalizer(audio, visual, train, cfg["visual_pca_dim"])
        np.savez(out/"normalizer.npz", **normalizer)
    audio = (audio-normalizer["audio_mean"])/normalizer["audio_std"]
    visual = ((visual-normalizer["visual_mean"]) @ normalizer["visual_pca"])/normalizer["visual_std"]
    for split in splits:
        if reuse and split in {"train","validation"}:
            for suffix in (".npz", ".json"):
                source = reuse/(split+suffix)
                if sha256(source) != old_card["files"][source.name]:
                    raise ValueError("Discovery train/validation features changed")
                shutil.copy2(source,out/source.name)
            continue
        indices = [i for i,r in enumerate(kept) if r["split"] == split]
        np.savez_compressed(out/f"{split}.npz", audio=audio[indices], visual=visual[indices])
        (out/f"{split}.json").write_text(json.dumps([kept[i] for i in indices], indent=2)+"\n")
    kept = reused_rows+kept
    weights_path = Path(torch.hub.get_dir())/"checkpoints/resnet18-f37072fd.pth"
    card = {"sprint": "T2", "dataset": "CREMA-D", "source_revision": revision,
            "source_url": f"https://github.com/{REPO}", "license": "ODbL-1.0 database; DbCL-1.0 contents",
            "config": cfg, "actor_splits": splits, "selected": len(reused_rows)+len(records), "valid": len(kept), "failures": failures,
            "normalization_reference": str(reuse) if reuse else "fit on this corpus train split",
            "split_counts": {s: sum(r["split"]==s for r in kept) for s in splits},
            "duration_hours": {s: sum(r["duration_seconds"] for r in kept if r["split"]==s)/3600 for s in splits},
            "feature_spec": {"audio": "64-bin log power mel, 16kHz, 25ms FFT, 10ms hop, center=False; midpoint halves encoded separately",
                             "visual": "frame-independent ImageNet ResNet18 pooled512 -> train-only PCA64",
                             "temporal_steps": cfg["steps_per_clip"], "normalization_fit_split": "train",
                             "frozen_encoder_parameters": frozen_params,
                             "frozen_weights_sha256": sha256(weights_path)},
            "files": {p.name: sha256(p) for p in [out/f"{s}{ext}" for s in splits for ext in (".npz", ".json")] + [out/"normalizer.npz"]},
            "limitations": ["Acted speech with 12 repeated sentences; narrow domain", "Normalized-time sequences, not fixed physical time",
                            "ImageNet pretraining is external; frozen encoder excluded from trainable count",
                            "Transcripts/emotions/actor IDs are metadata only; no text generation trained"]}
    (out/"dataset_card.json").write_text(json.dumps(card, indent=2)+"\n")
    print(json.dumps({"prepared": str(out), "split_counts": card["split_counts"], "failures": len(failures)}), flush=True)
    return card


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    prepare(parser.parse_args().config)
