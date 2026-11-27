"""SLWM-v4 Gate 2: Production Streaming Parallel GPU Pre-Encoder for Tier 1 (`~137 GB` Certified-int8 Single-Pass Pool).

Streams all 9 verified open-access SLWM-v4 corpora directly on datacenter bandwidth into strictly causal
160 ms interleaved `SLWM-v4` shards (`data/v4_heartbeat_contract.py` + `data/v4_embodiment_and_splits.py`),
encoding video with frozen `Wan 2.1 Causal VAE` (`bf16`, `192x320` @ `25 fps`, certified per-patch affine `int8`
at `61.44 dB` latent PSNR / `0.99957` SSIM), audio with frozen `Kyutai Mimi 16-RVQ` (`12.5 Hz`, `4.92%` WER),
motor/proprio/force into non-aliased `52`/`68` tensors, and aligned lossless UTF-8 bytes (`264` vocab).

Uploads each finished `~500-900 MB` shard + per-episode SHA-256 metadata manifest directly to its dedicated,
license-pure public HuggingFace Dataset repository under `valueai-org/slwm-v4-*` and deletes raw temporary media
immediately so a single Colab GPU VM never runs out of local SSD and every run is 100% crash-resumable.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from typing import Any, Iterator

import numpy as np
try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None  # type: ignore[assignment]
try:
    from scipy.io import wavfile
    from scipy.signal import resample_poly
except ImportError:
    wavfile = None  # type: ignore[assignment]
    resample_poly = None  # type: ignore[assignment]
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.v4_embodiment_and_splits import (
    assign_v4_split,
    audit_pcm_nonsilence,
    hash_payload_sha256,
    map_embodiment_to_unified_52_68,
    map_unitree_g1_named_columns,
    resample_video_and_kinematics_to_25hz,
)
from data.v4_heartbeat_contract import (
    SLWMv4HeartbeatSpec,
    UNIFIED_ACTION_DIM,
    UNIFIED_PROPRIO_DIM,
    V4_TIER1_PERSISTENT_CORPUS_SPEC,
    build_interleaved_heartbeat_stream,
)


# ---------------------------------------------------------------------------
# 1. Certified Per-Patch Affine int8 Quantization, HF Cache Cleanup & GPU Codec Engine
# ---------------------------------------------------------------------------


def safe_hf_download(repo_id: str, filename: str, *, repo_type: str = "dataset", retries: int = 8) -> str:
    """Retry transient HF Xet/HTTP/429 errors cleanly with rate-limit-aware backoff."""
    from huggingface_hub import hf_hub_download

    last_exc: Exception | None = None
    for att in range(1, retries + 1):
        try:
            return hf_hub_download(repo_id, filename, repo_type=repo_type, force_download=(att > 1))
        except Exception as exc:
            last_exc = exc
            msg = str(exc)
            is_429 = ("429" in msg) or ("Too Many Requests" in msg) or ("rate limit" in msg.lower())
            wait_s = min(180.0, 30.0 * att) if is_429 else min(30.0, 4.0 * att)
            print(f"[HF_DL WARN {att}/{retries}] {repo_id}/{filename} (429={is_429}): {msg[:180]} -> wait {wait_s:.0f}s", flush=True)
            time.sleep(wait_s)
    assert last_exc is not None
    raise last_exc


def cleanup_hf_file(lp: str | Path | None) -> None:
    """Remove BOTH the HF snapshot symlink AND its underlying cache blob so Colab SSD never fills up."""
    if not lp:
        return
    p = Path(lp)
    try:
        real = p.resolve() if p.exists() or p.is_symlink() else None
        p.unlink(missing_ok=True)
        if real is not None and real != p and "blobs" in real.parts and real.is_file():
            real.unlink(missing_ok=True)
    except Exception:
        pass


def decode_audio_bytes_24khz_mono(raw_bytes: bytes, fallback_ext: str = ".flac") -> np.ndarray:
    """Fast in-memory FLAC/WAV/OGG -> 24 kHz mono float32 PCM via soundfile (falls back to ffmpeg for MP3/M4A)."""
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=True)
        x = data.mean(axis=-1)
        if sr == 24000:
            return x.astype(np.float32, copy=False)
        g_gcd = math.gcd(int(sr), 24000)
        return resample_poly(x.astype(np.float64), 24000 // g_gcd, int(sr) // g_gcd).astype(np.float32)
    except Exception:
        tmp_p = f"/tmp/v4_aud_dec_{os.getpid()}{fallback_ext}"
        try:
            Path(tmp_p).write_bytes(raw_bytes)
            return decode_audio_24khz_mono(tmp_p)
        finally:
            Path(tmp_p).unlink(missing_ok=True)


def quantize_wan21_patches_int8(patches_fp16: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gate-2-certified per-patch affine int8 quantization (`61.44 dB` latent PSNR / `0.99957` SSIM)."""
    if patches_fp16.shape[0] == 0:
        return (
            np.zeros((0, 1024), dtype=np.uint8),
            np.zeros((0, 1), dtype=np.float16),
            np.zeros((0, 1), dtype=np.float16),
        )
    x = patches_fp16.astype(np.float32)
    xmin = np.min(x, axis=-1, keepdims=True)
    xmax = np.max(x, axis=-1, keepdims=True)
    scale = np.maximum((xmax - xmin) / 255.0, 1e-8)
    q = np.clip(np.round((x - xmin) / scale), 0.0, 255.0).astype(np.uint8)
    return q, xmin.astype(np.float16), scale.astype(np.float16)


def patchify_wan21_latent_batch(z: torch.Tensor) -> np.ndarray:
    """Convert Wan 2.1 latent `[B, 16, T_grp, 24, 40]` -> isotropic `float16[B, T_grp, 15, 1024]`."""
    arr = z.float().cpu().numpy()  # [B, 16, T, 24, 40]
    bsz, _, t_grp, _, _ = arr.shape
    p = arr.reshape(bsz, 16, t_grp, 3, 8, 5, 8).transpose(0, 2, 3, 5, 1, 4, 6).reshape(bsz, t_grp, 15, 1024)
    return p.astype(np.float16)


def patchify_wan21_latent(z: torch.Tensor) -> np.ndarray:
    """Convert Wan 2.1 latent `[1, 16, T_grp, 24, 40]` -> isotropic `float16[T_grp, 15, 1024]`."""
    return patchify_wan21_latent_batch(z)[0]


class FrozenV4CodecEngine:
    """GPU-accelerated frozen `Wan 2.1 Causal VAE` (`bf16`, `192x320`) + `Kyutai Mimi 16-RVQ` (`12.5 Hz`)."""

    def __init__(self, device: str = "cuda", load_video: bool = True, load_audio: bool = True) -> None:
        if device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.audio_device = torch.device("cpu") if self.device.type == "mps" else self.device
        self.vae = None
        self.mimi = None
        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if self.device.type == "cuda"
            else 0.0
        )
        # Auto-tune Wan 2.1 video batch size by VRAM (B=16 on A100-80GB ~15 GB, B=8 on L4-24GB ~8 GB, B=4 on M4 Max MPS)
        self.video_batch_size = 16 if vram_gb >= 60.0 else (8 if vram_gb >= 20.0 else 4)
        self.decode_workers = min(8, max(4, (os.cpu_count() or 4)))
        if load_video:
            from diffusers import AutoencoderKLWan

            dtype = torch.bfloat16 if self.device.type == "cuda" else (torch.float16 if self.device.type == "mps" else torch.float32)
            print(
                f"[CODEC] Loading Wan 2.1 Causal VAE on {self.device} ({dtype}, vram={vram_gb:.1f}GB, batch={self.video_batch_size}, workers={self.decode_workers})...",
                flush=True,
            )
            self.vae = AutoencoderKLWan.from_pretrained(
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=dtype
            ).eval().to(self.device)
        if load_audio:
            from transformers import MimiModel

            print(f"[CODEC] Loading Kyutai Mimi 16-RVQ on {self.audio_device}...", flush=True)
            self.mimi = MimiModel.from_pretrained("kyutai/mimi", torch_dtype=torch.float32).eval().to(self.audio_device)

    def encode_video_batch_1_plus_4h(self, frames_list: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """Strictly causal batched Wan 2.1 VAE encoder (`O(1)` VRAM via `feat_cache`, GPU normalization, 5-8x faster!).

        Because Wan 2.1 uses strictly causal 3D convolutions (zero future lookahead), right-padding shorter clips
        with zeros strictly after `1 + 4*H_i` has 0.0 effect on prefix latents `[0 .. H_i]`.
        """
        assert self.vae is not None
        with torch.no_grad():
            return self._encode_video_batch_inner(frames_list)

    def _encode_video_batch_inner(self, frames_list: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        if not frames_list:
            return []
        dtype = next(self.vae.parameters()).dtype
        # Sort by length to minimize padding waste, chunk by batch size, then restore original order!
        order = sorted(range(len(frames_list)), key=lambda i: frames_list[i].shape[0])
        results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(frames_list)

        pos = 0
        while pos < len(order):
            # Cap batch size on very long >32s chunks to keep host uint8 staging RAM safe (B=8 on A100-80GB, B=4 on L4)
            peek_hb = (int(frames_list[order[pos]].shape[0]) - 1) // 4
            b_lim = (8 if self.video_batch_size >= 16 else 4) if peek_hb > 200 else self.video_batch_size
            idxs = order[pos : pos + b_lim]
            pos += len(idxs)

            sub = [frames_list[i] for i in idxs]
            hbs = [(int(fr.shape[0]) - 1) // 4 for fr in sub]
            for fr in sub:
                assert fr.shape[0] >= 5 and (fr.shape[0] - 1) % 4 == 0, f"Expected 1+4H frames, got {fr.shape[0]}"
            max_hb = max(hbs)
            bsz = len(sub)

            # Stage uint8 on CPU, convert & normalize on GPU per 4-frame causal step (eliminates CPU float32 bottleneck!)
            host_u8 = np.zeros((bsz, 1 + 4 * max_hb, 192, 320, 3), dtype=np.uint8)
            for bi, fr in enumerate(sub):
                host_u8[bi, : fr.shape[0]] = fr

            def _slice_gpu(f0: int, f1: int) -> torch.Tensor:
                t_u8 = torch.from_numpy(host_u8[:, f0:f1]).to(device=self.device, non_blocking=True)
                # [B, T, H, W, 3] -> [B, 3, T, H, W] in [-1, 1]
                return t_u8.permute(0, 4, 1, 2, 3).contiguous().to(dtype=dtype).mul_(1.0 / 127.5).sub_(1.0)

            self.vae.clear_cache()
            self.vae._enc_feat_map = [None] * self.vae._cached_conv_counts["encoder"]
            self.vae._enc_conv_idx = [0]
            z0 = self.vae.quant_conv(
                self.vae.encoder(_slice_gpu(0, 1), feat_cache=self.vae._enc_feat_map, feat_idx=self.vae._enc_conv_idx)
            )[:, :16]
            z_steps = [z0]
            for h_i in range(max_hb):
                i0 = 1 + h_i * 4
                self.vae._enc_conv_idx = [0]
                zh = self.vae.quant_conv(
                    self.vae.encoder(
                        _slice_gpu(i0, i0 + 4),
                        feat_cache=self.vae._enc_feat_map,
                        feat_idx=self.vae._enc_conv_idx,
                    )
                )[:, :16]
                z_steps.append(zh)
            z_all = torch.cat(z_steps, dim=2)  # [B, 16, 1 + max_hb, 24, 40]
            patches_all = patchify_wan21_latent_batch(z_all)  # [B, 1 + max_hb, 15, 1024]
            self.vae.clear_cache()

            for bi, orig_i in enumerate(idxs):
                hb_i = hbs[bi]
                results[orig_i] = (patches_all[bi, 0].copy(), patches_all[bi, 1 : 1 + hb_i].copy())
        return [r for r in results if r is not None]

    def encode_video_1_plus_4h(self, frames_u8: np.ndarray, max_chunk_hb: int = 25) -> tuple[np.ndarray, np.ndarray]:
        """Strictly causally encode single clip `uint8[1 + 4*H, 192, 320, 3]` -> `(frame0[15, 1024], groups[H, 15, 1024])`."""
        return self.encode_video_batch_1_plus_4h([frames_u8])[0]

    def encode_audio_mimi_16rvq(self, pcm_24k: np.ndarray, chunk_sec: float = 16.0) -> np.ndarray:
        """Strictly causally encode 24 kHz mono float32 PCM -> `int16[T_ticks, 16]` (bit-exact with streaming)."""
        assert self.mimi is not None
        with torch.no_grad():
            return self._encode_audio_mimi_inner(pcm_24k, chunk_sec=chunk_sec)

    def _encode_audio_mimi_inner(self, pcm_24k: np.ndarray, chunk_sec: float = 16.0) -> np.ndarray:
        x = np.asarray(pcm_24k, dtype=np.float32).reshape(-1)
        tick_samples = 1920  # Exact 24000 / 12.5 Hz = 1920 samples per 80 ms tick
        n_ticks = x.size // tick_samples
        if n_ticks <= 0:
            return np.zeros((0, 16), dtype=np.int16)
        x = x[: n_ticks * tick_samples]

        # Single fused strictly-causal pass for <= 192 s clips (Gate 2 proved bit-exact equality with streaming!)
        if n_ticks <= 2400:
            seg = torch.from_numpy(x)[None, None, :].to(self.audio_device)
            out = self.mimi.encode(seg, num_quantizers=16)
            return out.audio_codes[0].T.cpu().numpy().astype(np.int16)

        # Streaming fallback for > 3 min continuous streams (compatible across transformers 4.x & 5.x)
        chunk_ticks = max(1, int(round(chunk_sec * 12.5)))
        enc_cache, pad_cache = None, None
        codes_parts: list[np.ndarray] = []
        pos = 0
        while pos < n_ticks:
            end = min(n_ticks, pos + chunk_ticks)
            seg = torch.from_numpy(x[pos * tick_samples : end * tick_samples])[None, None, :].to(self.audio_device)
            if enc_cache is None and pad_cache is None:
                out = self.mimi.encode(seg, num_quantizers=16, use_streaming=True)
            else:
                out = self.mimi.encode(
                    seg,
                    num_quantizers=16,
                    encoder_cache=enc_cache,
                    padding_cache=pad_cache,
                    use_streaming=True,
                )
            codes_parts.append(out.audio_codes[0].T.cpu().numpy().astype(np.int16))  # [T_seg, 16]
            enc_cache = getattr(out, "encoder_cache", getattr(out, "encoder_past_key_values", None))
            pad_cache = getattr(out, "padding_cache", None)
            pos = end
        return np.concatenate(codes_parts, axis=0)

    def encode_audio_batch_mimi_16rvq(self, pcm_list: list[np.ndarray], max_batch_ticks: int = 1600) -> list[np.ndarray]:
        """OOM-proof length-sorted strictly causal batched Mimi 16-RVQ encoder (`bsz * max_ticks <= 1600` -> < 2.2 GiB VRAM!)."""
        assert self.mimi is not None
        with torch.no_grad():
            return self._encode_audio_batch_inner(pcm_list, max_batch_ticks=max_batch_ticks)

    def _encode_audio_batch_inner(self, pcm_list: list[np.ndarray], max_batch_ticks: int = 1600) -> list[np.ndarray]:
        if not pcm_list:
            return []
        tick_samples = 1920
        lens = [int(x.size // tick_samples) for x in pcm_list]
        results: list[np.ndarray] = [np.zeros((0, 16), dtype=np.int16) for _ in pcm_list]
        valid_order = sorted([i for i, nt in enumerate(lens) if nt > 0], key=lambda i: lens[i])

        pos = 0
        while pos < len(valid_order):
            # Build length-matched sub-batch strictly bounded by `len(sub) * max_ticks <= max_batch_ticks`
            sub_idxs: list[int] = []
            while pos < len(valid_order):
                cand = valid_order[pos]
                cand_ticks = lens[cand]
                if cand_ticks > 300:
                    if not sub_idxs:
                        results[cand] = self.encode_audio_mimi_16rvq(pcm_list[cand])
                        pos += 1
                    break
                new_n = len(sub_idxs) + 1
                if sub_idxs and (new_n * cand_ticks > max_batch_ticks or new_n > 24):
                    break
                sub_idxs.append(cand)
                pos += 1
            if not sub_idxs:
                continue
            sub_max = max(lens[i] for i in sub_idxs)
            batch = np.zeros((len(sub_idxs), 1, sub_max * tick_samples), dtype=np.float32)
            for bi, orig_i in enumerate(sub_idxs):
                nt = lens[orig_i]
                batch[bi, 0, : nt * tick_samples] = pcm_list[orig_i][: nt * tick_samples]
            t_in = torch.from_numpy(batch).to(self.audio_device)
            out = self.mimi.encode(t_in, num_quantizers=16)
            codes_all = out.audio_codes.cpu().numpy()  # [B, 16, T_ticks]
            for bi, orig_i in enumerate(sub_idxs):
                results[orig_i] = codes_all[bi, :, : lens[orig_i]].T.astype(np.int16).copy()
            del t_in, out
        return results


# ---------------------------------------------------------------------------
# 2. FFmpeg Streaming Video/Audio Extractors (Zero Black Bars, 192x320 @ 25 fps)
# ---------------------------------------------------------------------------


def _get_ffmpeg_bin() -> str:
    b = shutil.which("ffmpeg")
    if b:
        return b
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _probe_duration_sec(mp4_path: str) -> float:
    import re

    ffp = shutil.which("ffprobe")
    if ffp:
        res = subprocess.run(
            [ffp, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", mp4_path],
            capture_output=True, text=True, timeout=30,
        )
        if res.returncode == 0 and res.stdout.strip():
            return float(res.stdout.strip())
    res = subprocess.run([_get_ffmpeg_bin(), "-nostdin", "-i", mp4_path], capture_output=True, text=True, timeout=30)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", res.stderr)
    if m:
        return int(m.group(1)) * 3600.0 + int(m.group(2)) * 60.0 + float(m.group(3))
    return 0.0


def decode_video_25fps_192x320(
    mp4_path: str,
    *,
    max_sec: float | None = None,
    start_sec: float = 0.0,
    threads: int = 1,
) -> np.ndarray:
    """Decode any MP4 into `uint8[N_frames, 192, 320, 3]` at exact 25.0 fps with 0% black bars (center-crop)."""
    cmd = [_get_ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-threads", str(max(1, int(threads)))]
    if start_sec > 0.0:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", mp4_path]
    if max_sec is not None:
        cmd += ["-t", f"{max_sec:.3f}"]
    # Scale preserving aspect ratio so BOTH dims >= target, then exact center crop to 320x192 (0% black bars!).
    # Strict `round=down` enforces causal floor timestamp selection (`src_ts <= tgt_ts`, zero future-frame lookahead!).
    vf = "fps=25:round=down,scale=320:192:force_original_aspect_ratio=increase,crop=320:192"
    cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.check_output(cmd, timeout=180)
    frame_bytes = 192 * 320 * 3
    n_f = len(raw) // frame_bytes
    if n_f <= 0:
        return np.zeros((0, 192, 320, 3), dtype=np.uint8)
    return np.frombuffer(raw[: n_f * frame_bytes], dtype=np.uint8).reshape(n_f, 192, 320, 3)


def decode_video_native_frames_192x320(
    mp4_path: str,
    src_frame_indices: np.ndarray,
    *,
    start_frame: int = 0,
    n_clip_frames: int | None = None,
    native_fps: float = 30.0,
) -> np.ndarray:
    """Memory-safe (<30 MiB RAM) frame-exact episode-slice decoder for LeRobot v2/v3 MP4s indexed causally via `src_frame_indices`."""
    cmd = [_get_ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-threads", "1"]
    if start_frame > 0:
        # Single fast + frame-accurate input seek (`-ss` before `-i`): verified 100% frame-exact without CFR dup/drop!
        cmd += ["-ss", f"{start_frame / float(native_fps):.6f}"]
    cmd += ["-i", mp4_path]
    if n_clip_frames is not None and n_clip_frames > 0:
        cmd += ["-frames:v", str(int(n_clip_frames))]
    cmd += [
        "-vf", "scale=320:192:force_original_aspect_ratio=increase,crop=320:192",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    raw = subprocess.check_output(cmd, timeout=240)
    frame_bytes = 192 * 320 * 3
    n_f = len(raw) // frame_bytes
    if n_f <= 0:
        return np.zeros((0, 192, 320, 3), dtype=np.uint8)
    clip_fr = np.frombuffer(raw[: n_f * frame_bytes], dtype=np.uint8).reshape(n_f, 192, 320, 3)
    idx = np.clip(src_frame_indices, 0, n_f - 1)
    return clip_fr[idx].copy()


def decode_audio_24khz_mono(media_path: str, *, max_sec: float | None = None, start_sec: float = 0.0) -> np.ndarray:
    """Extract 24 kHz mono float32 PCM from any MP4/WAV/FLAC/MP3."""
    cmd = [_get_ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-threads", "1"]
    if start_sec > 0.0:
        cmd += ["-ss", f"{start_sec:.3f}"]
    cmd += ["-i", media_path]
    if max_sec is not None:
        cmd += ["-t", f"{max_sec:.3f}"]
    cmd += ["-ac", "1", "-ar", "24000", "-f", "f32le", "pipe:1"]
    res = subprocess.run(cmd, capture_output=True, timeout=180)
    if res.returncode != 0 or not res.stdout:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# 3. Streaming Resumable Shard Buffer + Automatic Private HF Uploader
# ---------------------------------------------------------------------------


def _hf_commit_with_retry(
    api: Any,
    repo_id: str,
    operations: list[Any],
    commit_message: str,
    *,
    max_attempts: int = 20,
) -> None:
    """Atomic HF Hub commit with 429 rate-limit + transient network backoff (never silently drops files)."""
    for att in range(1, max_attempts + 1):
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=commit_message,
            )
            return
        except Exception as exc:
            msg = str(exc)
            is_429 = ("429" in msg) or ("Too Many Requests" in msg) or ("rate limit" in msg.lower())
            wait_s = min(300.0, 60.0 * att) if is_429 else min(60.0, 5.0 * att)
            print(
                f"[HF_COMMIT WARN {att}/{max_attempts}] {repo_id} ({len(operations)} ops, 429={is_429}): "
                f"{msg[:220]} -> retrying in {wait_s:.0f}s",
                flush=True,
            )
            if att == max_attempts:
                raise
            time.sleep(wait_s)


class StreamingV4ShardWriter:
    """Crash-proof, 100% resumable streaming shard buffer with atomic per-shard `.npz` + `_meta.json` HF commits."""

    def __init__(
        self,
        *,
        corpus_name: str,
        local_out_dir: Path,
        hf_repo_id: str | None,
        migrate_from_repo: str | None = None,
        flush_mb: float = 650.0,
        delete_local_after_upload: bool = True,
    ) -> None:
        self.corpus_name = corpus_name
        self.local_out_dir = local_out_dir / corpus_name
        self.local_out_dir.mkdir(parents=True, exist_ok=True)
        self.hf_repo_id = hf_repo_id
        self.flush_bytes = int(flush_mb * 1024 * 1024)
        self.delete_local_after_upload = delete_local_after_upload
        self.api = None
        self.existing_remote_files: set[str] = set()
        if hf_repo_id:
            from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

            self.api = HfApi()
            try:
                info = self.api.repo_info(hf_repo_id, repo_type="dataset")
                self.existing_remote_files = {s.rfilename for s in (info.siblings or [])}
            except Exception as exc:
                print(f"[HF WARN] Could not list existing files in {hf_repo_id}: {exc}", flush=True)

            # 1. Optional one-time batched migration from legacy repo -> public ValueAI repo (25 files/commit)!
            if migrate_from_repo and migrate_from_repo != hf_repo_id:
                try:
                    src_info = self.api.repo_info(migrate_from_repo, repo_type="dataset")
                    src_files = sorted(
                        s.rfilename for s in (src_info.siblings or [])
                        if s.rfilename.startswith(f"{corpus_name}/")
                    )
                    missing_src = [f for f in src_files if f not in self.existing_remote_files]
                    if missing_src:
                        print(f"[PREENCODE MIGRATE] Copying {len(missing_src)}/{len(src_files)} files from {migrate_from_repo} -> {hf_repo_id}...", flush=True)
                        for b0 in range(0, len(missing_src), 25):
                            batch_rel = missing_src[b0 : b0 + 25]
                            lps: list[tuple[str, str]] = []
                            try:
                                ops = []
                                for rel_f in batch_rel:
                                    lp_m = safe_hf_download(migrate_from_repo, rel_f, repo_type="dataset")
                                    lps.append((rel_f, lp_m))
                                    ops.append(CommitOperationAdd(path_in_repo=rel_f, path_or_fileobj=lp_m))
                                _hf_commit_with_retry(self.api, hf_repo_id, ops, f"Migrate {len(ops)} {corpus_name} files")
                                for rel_f, _ in lps:
                                    self.existing_remote_files.add(rel_f)
                            finally:
                                for _, lp_m in lps:
                                    cleanup_hf_file(lp_m)
                        print(f"[PREENCODE MIGRATE OK] {len(missing_src)} files synced to {hf_repo_id}", flush=True)
                except Exception as exc:
                    print(f"[MIGRATE WARN] {migrate_from_repo} -> {hf_repo_id}: {exc}", flush=True)

            # 2. Rescue stranded local `.npz` shards (both paired local `_meta.json` AND verified remote `_meta.json`!) in 1 atomic commit!
            rescue_ops: list[Any] = []
            rescue_local_paths: list[Path] = []
            rescue_remote_paths: list[str] = []
            for lp_loc in sorted(self.local_out_dir.glob("*_shard_*.npz")):
                rel_npz = f"{corpus_name}/{lp_loc.name}"
                lp_meta = lp_loc.with_name(lp_loc.name.replace(".npz", "_meta.json"))
                rel_meta = f"{corpus_name}/{lp_meta.name}"
                if rel_npz in self.existing_remote_files:
                    if self.delete_local_after_upload:
                        lp_loc.unlink(missing_ok=True)
                        lp_meta.unlink(missing_ok=True)
                    continue
                if lp_meta.is_file():
                    print(f"[PREENCODE RESCUE] Queuing stranded local pair {rel_npz} + _meta.json -> {hf_repo_id}", flush=True)
                    rescue_ops.append(CommitOperationAdd(path_in_repo=rel_npz, path_or_fileobj=str(lp_loc)))
                    rescue_ops.append(CommitOperationAdd(path_in_repo=rel_meta, path_or_fileobj=str(lp_meta)))
                    rescue_local_paths.extend([lp_loc, lp_meta])
                    rescue_remote_paths.extend([rel_npz, rel_meta])
                elif rel_meta in self.existing_remote_files:
                    try:
                        lp_rm = hf_hub_download(hf_repo_id, rel_meta, repo_type="dataset")
                        mdata = json.loads(Path(lp_rm).read_text())
                        cleanup_hf_file(lp_rm)
                        with np.load(lp_loc) as z:
                            n_ep_npz = int(z["episode_success"].size)
                        n_ep_meta = int(mdata.get("n_episodes", len(mdata.get("episodes", []))))
                        if n_ep_npz == n_ep_meta and n_ep_npz > 0:
                            print(
                                f"[PREENCODE RESCUE] Verified stranded local {rel_npz} ({n_ep_npz} eps == remote _meta.json) -> {hf_repo_id}",
                                flush=True,
                            )
                            rescue_ops.append(CommitOperationAdd(path_in_repo=rel_npz, path_or_fileobj=str(lp_loc)))
                            rescue_local_paths.append(lp_loc)
                            rescue_remote_paths.append(rel_npz)
                        else:
                            print(
                                f"[RESCUE MISMATCH] {rel_npz}: local_eps={n_ep_npz} != remote_meta_eps={n_ep_meta}; removing stale local .npz",
                                flush=True,
                            )
                            lp_loc.unlink(missing_ok=True)
                    except Exception as exc:
                        print(f"[RESCUE VERIFY WARN] {rel_npz}: {exc}", flush=True)
                else:
                    print(f"[RESCUE SKIP ORPHAN LOCAL] {rel_npz} has neither local nor remote _meta.json; removing to re-encode cleanly", flush=True)
                    lp_loc.unlink(missing_ok=True)

            if rescue_ops:
                _hf_commit_with_retry(
                    self.api,
                    hf_repo_id,
                    rescue_ops,
                    f"Rescue {len(rescue_remote_paths)} stranded {corpus_name} shard files",
                )
                for rp in rescue_remote_paths:
                    self.existing_remote_files.add(rp)
                if self.delete_local_after_upload:
                    for lp in rescue_local_paths:
                        lp.unlink(missing_ok=True)
                print(f"[PREENCODE RESCUE OK] Committed {len(rescue_remote_paths)} rescued files to {hf_repo_id}", flush=True)

        self.shard_idx: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
        for sp in ("train", "dev", "test"):
            while f"{corpus_name}/{sp}_shard_{self.shard_idx[sp]:03d}.npz" in self.existing_remote_files:
                self.shard_idx[sp] += 1

        self.bufs: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        self.buf_recs: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        self.buf_bytes: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
        self.all_episode_records: list[dict[str, Any]] = []
        self.seen_episode_ids: set[str] = set()
        self.total_hours_by_split: dict[str, float] = {"train": 0.0, "dev": 0.0, "test": 0.0}
        self.total_packets_by_split: dict[str, int] = {"train": 0, "dev": 0, "test": 0}
        self.uploaded_shards: list[str] = []
        self._t0 = time.monotonic()
        self._last_prog_t = self._t0
        self._up_pool = ThreadPoolExecutor(max_workers=1)
        self._up_futs: list[Any] = []

        # Exact crash-resumable state restoration strictly from PAIRED `*_meta.json` + `.npz` checkpoints on HF!
        if self.api and self.hf_repo_id:
            from huggingface_hub import hf_hub_download

            meta_files = sorted(
                f for f in self.existing_remote_files
                if f.startswith(f"{corpus_name}/") and f.endswith("_meta.json")
            )
            for mf in meta_files:
                paired_npz = mf.replace("_meta.json", ".npz")
                if paired_npz not in self.existing_remote_files:
                    print(f"[RESUME SKIP ORPHAN META] {mf} has no paired {paired_npz} on HF; will cleanly re-encode", flush=True)
                    continue
                try:
                    lp_m = hf_hub_download(self.hf_repo_id, mf, repo_type="dataset")
                    mdata = json.loads(Path(lp_m).read_text())
                    cleanup_hf_file(lp_m)
                    sp = str(mdata.get("split", "train"))
                    shard_rel = str(mdata.get("shard_path", paired_npz))
                    self.uploaded_shards.append(shard_rel)
                    for rec in mdata.get("episodes", []):
                        eid = str(rec["episode_id"])
                        if eid in self.seen_episode_ids:
                            continue
                        self.seen_episode_ids.add(eid)
                        self.all_episode_records.append(rec)
                        self.total_hours_by_split[sp] += float(rec.get("duration_s", 0.0)) / 3600.0
                        self.total_packets_by_split[sp] += int(rec.get("n_packets", 0))
                except Exception as exc:
                    print(f"[RESUME WARN] Could not restore {mf}: {exc}", flush=True)
            if self.seen_episode_ids:
                print(
                    f"[PREENCODE RESUME] {corpus_name} restored {len(self.seen_episode_ids)} eps, "
                    f"{sum(self.total_hours_by_split.values()):.2f}h, next_shards={self.shard_idx}",
                    flush=True,
                )

    def add_episode(
        self,
        *,
        split: str,
        episode_id: str,
        cluster_key: str,
        stream: dict[str, np.ndarray],
        duration_s: float,
        audio_valid: bool | None = None,
        episode_success: int = -1,
        extra_meta: dict[str, Any] | None = None,
    ) -> None:
        if episode_id in self.seen_episode_ids:
            return
        self.seen_episode_ids.add(episode_id)
        # Quantize Wan 2.1 video patches to certified int8 (61.44 dB latent PSNR / 0.99957 SSIM)
        vq, vxmin, vsc = quantize_wan21_patches_int8(stream["video_feat"])

        # Compute non-vacuous per-modality payload SHA-256 digests!
        media_hashes: dict[str, str] = {}
        if vq.shape[0] > 0:
            media_hashes["video"] = hash_payload_sha256(vq)
        if stream["audio_codes"].shape[0] > 0:
            media_hashes["audio"] = hash_payload_sha256(stream["audio_codes"])
        if stream["action_feat"].shape[0] > 0:
            media_hashes["motor"] = hash_payload_sha256(stream["action_feat"])
        txt_mask = stream["modality_id"] == 2
        if np.any(txt_mask):
            media_hashes["text"] = hash_payload_sha256(stream["text_byte"][txt_mask])

        item = {
            "stream": {
                k: v for k, v in stream.items() if k not in ("video_feat", "action_feat", "proprio_feat")
            },
            "video_q_int8": vq,
            "video_xmin_fp16": vxmin,
            "video_scale_fp16": vsc,
            "action_feat_fp16": stream["action_feat"].astype(np.float16),
            "proprio_feat_fp16": stream["proprio_feat"].astype(np.float16),
            "action_mask_u8": stream["action_mask"].astype(np.uint8),
            "proprio_mask_u8": stream["proprio_mask"].astype(np.uint8),
            "episode_success": int(episode_success),
        }
        est_b = (
            vq.nbytes
            + stream["audio_codes"].nbytes
            + item["action_feat_fp16"].nbytes
            + item["proprio_feat_fp16"].nbytes
            + int(stream["modality_id"].size) * 40
        )
        self.bufs[split].append(item)
        self.buf_bytes[split] += est_b
        self.total_hours_by_split[split] += float(duration_s) / 3600.0
        self.total_packets_by_split[split] += int(stream["modality_id"].size)

        rec = {
            "corpus": self.corpus_name,
            "split": split,
            "episode_id": episode_id,
            "cluster_key": cluster_key,
            "duration_s": round(float(duration_s), 3),
            "n_packets": int(stream["modality_id"].size),
            "n_video_patches": int(vq.shape[0]),
            "n_audio_ticks": int(stream["audio_codes"].shape[0]),
            "n_motor_ticks": int(item["action_feat_fp16"].shape[0]),
            "n_text_bytes": int(np.sum(txt_mask)),
            "audio_valid": audio_valid,
            "episode_success": int(episode_success),
            "media_hashes": media_hashes,
        }
        if extra_meta:
            rec.update(extra_meta)
        self.buf_recs[split].append(rec)
        self.all_episode_records.append(rec)

        now_t = time.monotonic()
        if (now_t - self._last_prog_t) >= 60.0:
            self._last_prog_t = now_t
            dt_h = max((now_t - self._t0) / 3600.0, 1e-6)
            h_tot = sum(self.total_hours_by_split.values())
            p_tot = sum(self.total_packets_by_split.values())
            print(
                f"[PREENCODE PROG] {self.corpus_name} eps={len(self.all_episode_records)} "
                f"hrs={h_tot:.2f}h ({h_tot / dt_h:.1f} h/hr) pkts={p_tot:,} buf_mb={self.buf_bytes[split]/1e6:.0f}M",
                flush=True,
            )

        # Flush dev/test splits aggressively (<= 80 MB) so smaller eval splits are never stranded in RAM across crashes!
        split_flush_lim = self.flush_bytes if split == "train" else min(self.flush_bytes, 80 * 1024 * 1024)
        if self.buf_bytes[split] >= split_flush_lim:
            self.flush_split(split)

    def _upload_shard_and_meta(self, local_npz: Path, rem_npz: str, local_meta: Path, rem_meta: str, sz_mb: float) -> str:
        from huggingface_hub import CommitOperationAdd

        assert self.api is not None and self.hf_repo_id is not None
        ops = [
            CommitOperationAdd(path_in_repo=rem_npz, path_or_fileobj=str(local_npz)),
            CommitOperationAdd(path_in_repo=rem_meta, path_or_fileobj=str(local_meta)),
        ]
        _hf_commit_with_retry(self.api, self.hf_repo_id, ops, f"Add {rem_npz} + _meta.json ({sz_mb:.1f} MiB)")
        if self.delete_local_after_upload:
            local_npz.unlink(missing_ok=True)
            local_meta.unlink(missing_ok=True)
        print(f"[PREENCODE HF_UP OK] {rem_npz} + _meta.json ({sz_mb:.1f} MiB, 1 atomic commit)", flush=True)
        return rem_npz

    def flush_split(self, split: str) -> None:
        items = self.bufs[split]
        recs = list(self.buf_recs[split])
        if not items:
            return
        idx = self.shard_idx[split]
        self.shard_idx[split] += 1

        # Concatenate dense stream arrays with exact offset re-indexing!
        vid_off = aud_off = mot_off = 0
        out_stream: dict[str, list[np.ndarray]] = {
            k: []
            for k in (
                "modality_id", "heartbeat_id", "capture_time_s", "last_sample_time_s",
                "wall_time_s", "dt_s", "reset_mask", "valid_mask", "target_valid_mask",
                "reliability", "spatial_pos", "text_byte", "video_idx", "audio_idx", "motor_idx",
                "episode_local_id",
            )
        }
        vq_l, vxmin_l, vsc_l, aud_l, act_l, prop_l = [], [], [], [], [], []
        act_m_l, prop_m_l, succ_l = [], [], []

        for ep_i, it in enumerate(items):
            st = it["stream"]
            n_tok = int(st["modality_id"].size)
            for k in (
                "modality_id", "heartbeat_id", "capture_time_s", "last_sample_time_s",
                "wall_time_s", "dt_s", "reset_mask", "valid_mask", "target_valid_mask",
                "reliability", "spatial_pos", "text_byte",
            ):
                out_stream[k].append(st[k])
            out_stream["episode_local_id"].append(np.full((n_tok,), ep_i, dtype=np.int32))

            vi = st["video_idx"].copy()
            vi[vi >= 0] += vid_off
            ai = st["audio_idx"].copy()
            ai[ai >= 0] += aud_off
            mi = st["motor_idx"].copy()
            mi[mi >= 0] += mot_off
            out_stream["video_idx"].append(vi)
            out_stream["audio_idx"].append(ai)
            out_stream["motor_idx"].append(mi)

            vq_l.append(it["video_q_int8"])
            vxmin_l.append(it["video_xmin_fp16"])
            vsc_l.append(it["video_scale_fp16"])
            aud_l.append(st["audio_codes"])
            act_l.append(it["action_feat_fp16"])
            prop_l.append(it["proprio_feat_fp16"])
            act_m_l.append(it["action_mask_u8"])
            prop_m_l.append(it["proprio_mask_u8"])
            succ_l.append(it["episode_success"])

            vid_off += int(it["video_q_int8"].shape[0])
            aud_off += int(st["audio_codes"].shape[0])
            mot_off += int(it["action_feat_fp16"].shape[0])

        packed = {k: np.concatenate(v, axis=0) for k, v in out_stream.items()}
        packed["video_q_int8"] = np.concatenate(vq_l, axis=0) if vid_off else np.zeros((0, 1024), dtype=np.uint8)
        packed["video_xmin_fp16"] = np.concatenate(vxmin_l, axis=0) if vid_off else np.zeros((0, 1), dtype=np.float16)
        packed["video_scale_fp16"] = np.concatenate(vsc_l, axis=0) if vid_off else np.zeros((0, 1), dtype=np.float16)
        packed["audio_codes"] = np.concatenate(aud_l, axis=0) if aud_off else np.zeros((0, 16), dtype=np.int16)
        packed["action_feat"] = np.concatenate(act_l, axis=0) if mot_off else np.zeros((0, UNIFIED_ACTION_DIM), dtype=np.float16)
        packed["proprio_feat"] = np.concatenate(prop_l, axis=0) if mot_off else np.zeros((0, UNIFIED_PROPRIO_DIM), dtype=np.float16)
        packed["episode_action_masks"] = np.stack(act_m_l, axis=0)
        packed["episode_proprio_masks"] = np.stack(prop_m_l, axis=0)
        packed["episode_success"] = np.asarray(succ_l, dtype=np.int8)

        npz_name = f"{split}_shard_{idx:03d}.npz"
        meta_name = f"{split}_shard_{idx:03d}_meta.json"
        local_npz = self.local_out_dir / npz_name
        local_meta = self.local_out_dir / meta_name
        np.savez(local_npz, **packed)
        local_meta.write_text(json.dumps({
            "corpus": self.corpus_name,
            "split": split,
            "shard_path": f"{self.corpus_name}/{npz_name}",
            "n_episodes": len(recs),
            "episodes": recs,
        }))
        sz_mb = local_npz.stat().st_size / (1024**2)
        print(f"[PREENCODE FLUSH] {self.corpus_name}/{npz_name} eps={len(items)} pkts={packed['modality_id'].size:,} hrs={sum(self.total_hours_by_split.values()):.2f}h size={sz_mb:.1f} MiB", flush=True)

        if self.api and self.hf_repo_id:
            # Backpressure: keep at most 2 background uploads in flight so GPU never stalls on upload!
            while len(self._up_futs) >= 2:
                done_rem = self._up_futs.pop(0).result()
                self.uploaded_shards.append(done_rem)
            fut = self._up_pool.submit(
                self._upload_shard_and_meta,
                local_npz,
                f"{self.corpus_name}/{npz_name}",
                local_meta,
                f"{self.corpus_name}/{meta_name}",
                sz_mb,
            )
            self._up_futs.append(fut)

        self.bufs[split].clear()
        self.buf_recs[split].clear()
        self.buf_bytes[split] = 0
        gc.collect()

    def finalize(self) -> dict[str, Any]:
        for sp in ("train", "dev", "test"):
            self.flush_split(sp)
        # Wait for all async background shard + _meta.json uploads to complete!
        while self._up_futs:
            self.uploaded_shards.append(self._up_futs.pop(0).result())
        self._up_pool.shutdown(wait=True)
        receipt = {
            "corpus": self.corpus_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_episodes": len(self.all_episode_records),
            "hours_by_split": {k: round(v, 3) for k, v in self.total_hours_by_split.items()},
            "total_hours": round(sum(self.total_hours_by_split.values()), 3),
            "packets_by_split": self.total_packets_by_split,
            "total_packets": sum(self.total_packets_by_split.values()),
            "uploaded_shards": sorted(set(self.uploaded_shards)),
            "episodes": self.all_episode_records,
        }
        rec_path = self.local_out_dir / f"{self.corpus_name}_preencode_manifest.json"
        rec_path.write_text(json.dumps(receipt, indent=2))
        if self.api and self.hf_repo_id:
            from huggingface_hub import CommitOperationAdd

            _hf_commit_with_retry(
                self.api,
                self.hf_repo_id,
                [CommitOperationAdd(path_in_repo=f"{self.corpus_name}/{rec_path.name}", path_or_fileobj=str(rec_path))],
                f"Finalize {self.corpus_name} manifest",
            )
            print(f"[HF_UP OK] {self.corpus_name}/{rec_path.name}", flush=True)
        return receipt


# ---------------------------------------------------------------------------
# 4. Corpus-Specific Streaming Encoders (All 9 Verified SLWM-v4 Suites)
# ---------------------------------------------------------------------------


def encode_vggsound(
    codec: FrozenV4CodecEngine,
    writer: StreamingV4ShardWriter,
    *,
    target_hours: float,
    max_episodes: int | None = None,
) -> None:
    """Stream `Loie/VGGSound` tar.gz shards (00..19) directly, 10 s clips @ 25 fps + 16-RVQ Mimi + class caption."""
    from huggingface_hub import hf_hub_download, hf_hub_url

    csv_p = hf_hub_download("Loie/VGGSound", "vggsound.csv", repo_type="dataset")
    yt_label: dict[str, str] = {}
    with open(csv_p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3 and parts[1].strip().isdigit():
                yt_label[f"{parts[0].strip()}_{int(parts[1].strip()):06d}"] = parts[2].strip().strip('"')
    cleanup_hf_file(csv_p)

    def _decode_one_vgg(item: tuple[int, str, str, str, bytes]) -> dict[str, Any] | None:
        idx_k, stem, sp, ckey, raw_mp4 = item
        tmp_mp4 = f"/tmp/v4_vgg_{os.getpid()}_{idx_k}.mp4"
        try:
            with open(tmp_mp4, "wb") as fout:
                fout.write(raw_mp4)
            fr = decode_video_25fps_192x320(tmp_mp4, max_sec=10.0)
            pcm = decode_audio_24khz_mono(tmp_mp4, max_sec=10.0)
            if fr.shape[0] < 9 or pcm.size < 24000:
                return None
            if not audit_pcm_nonsilence(pcm, 24000)["audio_valid"]:
                return None
            n_hb = min((fr.shape[0] - 1) // 4, int(pcm.size // 3840))
            if n_hb < 4:
                return None
            return {
                "stem": stem, "sp": sp, "ckey": ckey, "n_hb": n_hb,
                "fr": fr[: 1 + 4 * n_hb], "pcm": pcm[: n_hb * 3840],
                "lbl": yt_label.get(stem, "ambient sound"),
            }
        except Exception:
            return None
        finally:
            if os.path.exists(tmp_mp4):
                os.remove(tmp_mp4)

    batch_collect = codec.video_batch_size * 2
    done_eps = 0
    with ThreadPoolExecutor(max_workers=codec.decode_workers) as pool:
        def _flush_vgg_raw(raw_items: list[tuple[int, str, str, str, bytes]]) -> None:
            nonlocal done_eps
            if not raw_items:
                return
            decoded = [d for d in pool.map(_decode_one_vgg, raw_items) if d is not None]
            raw_items.clear()
            if not decoded:
                return
            if max_episodes is not None:
                rem = max(0, max_episodes - done_eps)
                decoded = decoded[:rem]
                if not decoded:
                    return
            vid_res = codec.encode_video_batch_1_plus_4h([d["fr"] for d in decoded])
            aud_res = codec.encode_audio_batch_mimi_16rvq([d["pcm"] for d in decoded])
            for d, (f0, v_grps), a_codes in zip(decoded, vid_res, aud_res):
                n_hb = min(int(v_grps.shape[0]), int(a_codes.shape[0] // 2))
                if n_hb < 4:
                    continue
                stream = build_interleaved_heartbeat_stream(
                    video_init_frame0=f0,
                    video_groups=v_grps[:n_hb],
                    audio_codes=a_codes[: n_hb * 2],
                    text_by_heartbeat={0: d["lbl"].encode("utf-8")},
                )
                writer.add_episode(
                    split=d["sp"],
                    episode_id=d["stem"],
                    cluster_key=d["ckey"],
                    stream=stream,
                    duration_s=n_hb * 0.16,
                    audio_valid=True,
                )
                done_eps += 1

        for shard_i in range(20):
            if sum(writer.total_hours_by_split.values()) >= target_hours:
                break
            if max_episodes is not None and done_eps >= max_episodes:
                break
            fname = f"vggsound_{shard_i:02d}.tar.gz"
            print(f"[VGGSound] Downloading {fname} via HF Xet (hours so far={sum(writer.total_hours_by_split.values()):.2f}/{target_hours:.1f}h)...", flush=True)
            lp_tar = hf_hub_download("Loie/VGGSound", fname, repo_type="dataset")
            try:
                raw_buf: list[tuple[int, str, str, str, bytes]] = []
                seq_k = 0
                with tarfile.open(lp_tar, "r:gz") as tf:
                    for m in tf:
                        if sum(writer.total_hours_by_split.values()) >= target_hours:
                            break
                        if max_episodes is not None and done_eps >= max_episodes:
                            break
                        if not m.isfile() or not m.name.endswith(".mp4"):
                            continue
                        fobj = tf.extractfile(m)
                        if fobj is None:
                            continue
                        stem = Path(m.name).stem
                        if stem in writer.seen_episode_ids:
                            continue
                        yt_id = stem.rsplit("_", 1)[0]
                        sp, ckey = assign_v4_split("vggsound_full_sync", {"youtube_id": yt_id})
                        raw_buf.append((seq_k, stem, sp, ckey, fobj.read()))
                        seq_k += 1
                        if len(raw_buf) >= batch_collect:
                            _flush_vgg_raw(raw_buf)
                _flush_vgg_raw(raw_buf)
            finally:
                cleanup_hf_file(lp_tar)


def parse_lerobot_v3_episode_metadata_rows(
    cols: dict[str, list[Any]],
    cam_key: str,
) -> dict[int, dict[str, Any]]:
    """Parse LeRobot v3.0 `meta/episodes/` columns into exact `episode_index -> {data_pq, video_mp4, from_ts, to_ts, length, task}`."""
    vc_col = f"videos/{cam_key}/chunk_index"
    vf_col = f"videos/{cam_key}/file_index"
    ft_col = f"videos/{cam_key}/from_timestamp"
    tt_col = f"videos/{cam_key}/to_timestamp"
    ep_indices = cols["episode_index"]
    dc_list = cols["data/chunk_index"]
    df_list = cols["data/file_index"]
    vc_list = cols[vc_col]
    vf_list = cols[vf_col]
    ft_list = cols[ft_col]
    tt_list = cols[tt_col]
    len_list = cols.get("length")
    tasks_list = cols.get("tasks")

    out: dict[int, dict[str, Any]] = {}
    for i, ep_raw in enumerate(ep_indices):
        if ep_raw is None or vc_list[i] is None or vf_list[i] is None or ft_list[i] is None or tt_list[i] is None:
            continue
        ep_i = int(ep_raw)
        task_str = ""
        if tasks_list is not None and tasks_list[i] is not None:
            t_raw = tasks_list[i]
            if isinstance(t_raw, (list, tuple)):
                task_str = next((str(x).strip() for x in t_raw if x and str(x).strip()), "")
            else:
                task_str = str(t_raw).strip()
        out[ep_i] = {
            "episode_index": ep_i,
            "data_pq": f"data/chunk-{int(dc_list[i]):03d}/file-{int(df_list[i]):03d}.parquet",
            "video_mp4": f"videos/{cam_key}/chunk-{int(vc_list[i]):03d}/file-{int(vf_list[i]):03d}.mp4",
            "from_ts": float(ft_list[i]),
            "to_ts": float(tt_list[i]),
            "length": int(len_list[i]) if (len_list is not None and len_list[i] is not None) else -1,
            "task": task_str,
        }
    return out


def load_lerobot_v3_episode_video_index(
    repo_id: str,
    cam_key: str,
    siblings: list[str],
) -> dict[int, dict[str, Any]]:
    """Read ALL LeRobot v3.0 `meta/episodes/chunk-*/file-*.parquet` files for exact video-to-parquet episode alignment."""
    ep_pqs = sorted(f for f in siblings if f.startswith("meta/episodes/") and f.endswith(".parquet"))
    if not ep_pqs:
        return {}
    req_cols = [
        "episode_index",
        "data/chunk_index",
        "data/file_index",
        f"videos/{cam_key}/chunk_index",
        f"videos/{cam_key}/file_index",
        f"videos/{cam_key}/from_timestamp",
        f"videos/{cam_key}/to_timestamp",
    ]
    index: dict[int, dict[str, Any]] = {}
    for ep_pq in ep_pqs:
        lp = safe_hf_download(repo_id, ep_pq, repo_type="dataset")
        try:
            pf = pq.ParquetFile(lp)
            avail = set(pf.schema_arrow.names)
            if not all(c in avail for c in req_cols):
                continue
            read_cols = list(req_cols)
            for opt_c in ("length", "tasks"):
                if opt_c in avail:
                    read_cols.append(opt_c)
            tbl = pq.read_table(lp, columns=read_cols)
            cols_dict = {c: tbl[c].to_pylist() for c in read_cols}
            index.update(parse_lerobot_v3_episode_metadata_rows(cols_dict, cam_key))
        finally:
            cleanup_hf_file(lp)
    return index


def encode_aloha_and_unitree_g1(
    codec: FrozenV4CodecEngine,
    writer: StreamingV4ShardWriter,
    *,
    corpus_name: str,
    target_hours: float,
    max_episodes: int | None = None,
) -> None:
    """Encode verified Unitree G1/H1 (30/50 fps) OR ALOHA-Sim (50 fps) LeRobot v3.0 repos using exact `meta/episodes/` indexing."""
    import hashlib
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    if corpus_name == "unitree_g1_h1_humanoid":
        entry = next(c for c in V4_TIER1_PERSISTENT_CORPUS_SPEC if c["name"] == "unitree_g1_h1_humanoid")
        raw_repos = list(entry["hf_repos"])
        # Prioritize rare/whole-body embodiments (H1, WBT, Dex3, Brainco) first so every humanoid family is guaranteed coverage!
        def _repo_prio(r: str) -> tuple[int, str]:
            rl = r.lower()
            if "unitreeh1" in rl:
                return (0, r)
            if "_wbt_" in rl:
                return (1, r)
            if "_dex3_" in rl:
                return (2, r)
            if "_brainco_" in rl:
                return (3, r)
            return (4, r)
        repos = sorted(raw_repos, key=_repo_prio)
    else:
        repos = ["lerobot/aloha_sim_transfer_cube_human", "lerobot/aloha_sim_insertion_human"]

    done_eps = 0
    # Cap per-repo hours for multi-repo humanoid suite so ALL 13 G1/H1 tasks & whole-body WBT embodiments are covered!
    per_repo_hour_cap = (
        max(3.6, (target_hours / max(1, len(repos))) * 1.25)
        if (corpus_name == "unitree_g1_h1_humanoid" and len(repos) > 1)
        else target_hours
    )
    for repo_id in repos:
        if sum(writer.total_hours_by_split.values()) >= target_hours:
            break
        if max_episodes is not None and done_eps >= max_episodes:
            break
        repo_prefix = f"{repo_id.split('/')[-1]}_ep"
        def _repo_hours_done() -> float:
            return sum(
                float(r.get("duration_s", 0.0))
                for r in writer.all_episode_records
                if str(r.get("episode_id", "")).startswith(repo_prefix)
            ) / 3600.0
        if _repo_hours_done() >= per_repo_hour_cap:
            print(f"[ROBOTICS RESUME SKIP] {repo_id} already at {_repo_hours_done():.2f}h >= cap {per_repo_hour_cap:.2f}h", flush=True)
            continue
        try:
            info_p = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset")
            info = json.load(open(info_p))
            cleanup_hf_file(info_p)
            native_fps = float(info.get("fps", 30.0))
            ver = str(info.get("codebase_version", "v3.0"))
            vid_keys = [k for k, v in info.get("features", {}).items() if v.get("dtype") == "video"]
            # Prefer head/top/left RGB camera (never depth/infrared)
            cam_key = next(
                (k for k in vid_keys if any(w in k.lower() for w in ("head_stereo_left", "head", "top", "cam_left_high", "cam_high", "cam_left"))),
                vid_keys[0],
            )
            siblings = [s.rfilename for s in (api.repo_info(repo_id, repo_type="dataset").siblings or [])]
            sib_set = set(siblings)
            ep_index = load_lerobot_v3_episode_video_index(repo_id, cam_key, siblings)
            if not ep_index:
                print(f"[ROBOTICS WARN] Skip {repo_id}: missing LeRobot v3.0 meta/episodes index for {cam_key}", flush=True)
                continue
            # Group episodes strictly by exact `video_mp4` file from `meta/episodes/` (handles both 1-pq->N-mp4 and N-pq->1-mp4!)
            by_mp4: dict[str, list[dict[str, Any]]] = {}
            for em in ep_index.values():
                if em["video_mp4"] in sib_set and em["data_pq"] in sib_set:
                    by_mp4.setdefault(em["video_mp4"], []).append(em)
            mp4_list = sorted(by_mp4.keys())
            # Deterministic seeded shuffle per repo so train/dev/test session blocks are sampled uniformly under hour caps!
            seed_u32 = int(hashlib.sha256(f"slwm-v4-rob::{repo_id}".encode("utf-8")).hexdigest()[:8], 16)
            rng_r = np.random.default_rng(seed_u32)
            mp4_list = [mp4_list[int(i)] for i in rng_r.permutation(len(mp4_list))]
            print(
                f"[ROBOTICS] {repo_id} ver={ver} fps={native_fps} cam={cam_key} "
                f"indexed_eps={len(ep_index)} mp4_groups={len(mp4_list)} cap={per_repo_hour_cap:.2f}h",
                flush=True,
            )
        except Exception as exc:
            print(f"[ROBOTICS WARN] Skip {repo_id}: {exc}", flush=True)
            continue

        for mp4_rel in mp4_list:
            if sum(writer.total_hours_by_split.values()) >= target_hours:
                break
            if _repo_hours_done() >= per_repo_hour_cap:
                break
            if max_episodes is not None and done_eps >= max_episodes:
                break
            ep_metas = [
                em for em in by_mp4[mp4_rel]
                if f"{repo_id.split('/')[-1]}_ep{em['episode_index']:05d}" not in writer.seen_episode_ids
            ]
            if not ep_metas:
                continue
            needed_pqs = sorted({em["data_pq"] for em in ep_metas})
            lp_mp4 = ""
            lp_pqs: dict[str, str] = {}
            pq_tables: dict[str, tuple[Any, np.ndarray, list[str]]] = {}
            try:
                lp_mp4 = safe_hf_download(repo_id, mp4_rel, repo_type="dataset")
                for pq_rel in needed_pqs:
                    lp_p = safe_hf_download(repo_id, pq_rel, repo_type="dataset")
                    lp_pqs[pq_rel] = lp_p
                    t_cur = pq.read_table(lp_p)
                    pq_tables[pq_rel] = (t_cur, t_cur["episode_index"].to_numpy(), t_cur.column_names)
            except Exception as exc:
                print(f"[ROBOTICS WARN] Skip failed {repo_id}/{mp4_rel}: {exc}", flush=True)
                cleanup_hf_file(lp_mp4)
                for lp_p in lp_pqs.values():
                    cleanup_hf_file(lp_p)
                continue
            try:
                # Prepare all aligned episodes in this MP4, sorted by length for optimal GPU batching!
                ep_jobs_all: list[dict[str, Any]] = []
                for em in ep_metas:
                    ep_val = int(em["episode_index"])
                    tbl, ep_ids, df_cols = pq_tables[em["data_pq"]]
                    rows = np.where(ep_ids == ep_val)[0]
                    if rows.size < 15:
                        continue
                    r0, t_len = int(rows[0]), int(rows.size)
                    exp_vid_frames = (float(em["to_ts"]) - float(em["from_ts"])) * native_fps
                    if abs(exp_vid_frames - t_len) > max(4.0, 0.05 * t_len):
                        continue
                    start_frame = max(0, int(round(float(em["from_ts"]) * native_fps)))
                    ep_tbl = tbl.slice(r0, t_len)
                    try:
                        if "action.left_arm" in df_cols:
                            col_dict = {
                                c: np.stack(ep_tbl[c].to_pylist(), axis=0)
                                for c in df_cols
                                if c in (
                                    "action.left_arm", "action.right_arm", "action.hand_cmd",
                                    "action.left_gripper", "action.right_gripper", "action.left_leg",
                                    "action.right_leg", "action.waist_action_joint", "action.base_command",
                                    "observation.state.left_arm", "observation.state.right_arm",
                                    "observation.state.hand_state", "observation.state.left_gripper",
                                    "observation.state.right_gripper", "observation.state.left_leg",
                                    "observation.state.right_leg", "observation.state.waist_state_joint",
                                    "observation.state.state_base_rot_quat",
                                )
                            }
                            a52, am, p68, pm = map_unitree_g1_named_columns(col_dict, t_len)
                        else:
                            raw_a = np.stack(ep_tbl["action"].to_pylist(), axis=0).astype(np.float32)
                            raw_p = (
                                np.stack(ep_tbl["observation.state"].to_pylist(), axis=0).astype(np.float32)
                                if "observation.state" in df_cols
                                else None
                            )
                            emb = (
                                "aloha_bimanual_14dof"
                                if raw_a.shape[1] == 14
                                else ("unitree_h1_40dof" if raw_a.shape[1] == 40 else ("unitree_g1_dex1_16dof" if raw_a.shape[1] == 16 else ("unitree_g1_brainco_26dof" if raw_a.shape[1] == 26 else "unitree_g1_dex3_28dof")))
                            )
                            a52, am, p68, pm = map_embodiment_to_unified_52_68(emb, raw_action=raw_a, raw_proprio=raw_p)

                        resamp = resample_video_and_kinematics_to_25hz(
                            native_fps=native_fps,
                            n_src_frames=t_len,
                            action_52=a52,
                            proprio_68=p68,
                            proprio_mask=pm,
                        )
                        task_s = em["task"] or repo_id.split("/")[-1].replace("_", " ")
                        ep_jobs_all.append({
                            "ep_val": ep_val,
                            "start_frame": start_frame,
                            "t_len": t_len,
                            "am": am,
                            "pm": pm,
                            "resamp": resamp,
                            "task": task_s,
                        })
                    except Exception:
                        continue
                ep_jobs_all.sort(key=lambda x: x["t_len"])

                with ThreadPoolExecutor(max_workers=codec.decode_workers) as pool:
                    def _decode_g1_job(j: dict[str, Any]) -> dict[str, Any] | None:
                        try:
                            ep_fr = decode_video_native_frames_192x320(
                                lp_mp4,
                                j["resamp"]["video_src_frame_indices"],
                                start_frame=j["start_frame"],
                                n_clip_frames=j["t_len"],
                                native_fps=native_fps,
                            )
                            if ep_fr.shape[0] != j["resamp"]["n_target_frames_1_plus_4h"]:
                                return None
                            j["fr"] = ep_fr
                            return j
                        except Exception:
                            return None

                    for b0 in range(0, len(ep_jobs_all), codec.video_batch_size):
                        if sum(writer.total_hours_by_split.values()) >= target_hours:
                            break
                        if _repo_hours_done() >= per_repo_hour_cap:
                            break
                        if max_episodes is not None and done_eps >= max_episodes:
                            break
                        sub_jobs = ep_jobs_all[b0 : b0 + codec.video_batch_size]
                        if max_episodes is not None:
                            sub_jobs = sub_jobs[: max(0, max_episodes - done_eps)]
                        decoded = [x for x in pool.map(_decode_g1_job, sub_jobs) if x is not None]
                        if not decoded:
                            continue
                        vid_res = codec.encode_video_batch_1_plus_4h([d["fr"] for d in decoded])
                        for d, (f0, v_grps) in zip(decoded, vid_res):
                            if sum(writer.total_hours_by_split.values()) >= target_hours:
                                break
                            if _repo_hours_done() >= per_repo_hour_cap:
                                break
                            n_hb = d["resamp"]["n_heartbeats"]
                            sp, ckey = assign_v4_split(
                                corpus_name,
                                {"repo_id": repo_id, "episode_idx": d["ep_val"], "source": repo_id},
                            )
                            stream = build_interleaved_heartbeat_stream(
                                video_init_frame0=f0,
                                video_groups=v_grps,
                                action_seq=d["resamp"]["action_25hz"],
                                proprio_seq=d["resamp"]["proprio_25hz"],
                                action_mask=d["am"],
                                proprio_mask=d["pm"],
                                text_by_heartbeat={0: d["task"].encode("utf-8")},
                            )
                            writer.add_episode(
                                split=sp,
                                episode_id=f"{repo_id.split('/')[-1]}_ep{d['ep_val']:05d}",
                                cluster_key=ckey,
                                stream=stream,
                                duration_s=n_hb * 0.16,
                                extra_meta={"subcorpus": repo_id.split("/")[-1]},
                            )
                            done_eps += 1
            finally:
                cleanup_hf_file(lp_mp4)
                for lp_p in lp_pqs.values():
                    cleanup_hf_file(lp_p)


def encode_droid_1_0_1(
    codec: FrozenV4CodecEngine,
    writer: StreamingV4ShardWriter,
    *,
    target_hours: float,
    max_episodes: int | None = None,
) -> None:
    """Encode `lerobot/droid_1.0.1` (15 fps -> 25 fps with exact LeRobot v3.0 `meta/episodes/` video indexing, causal ZOH, building+collector disjoint)."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = "lerobot/droid_1.0.1"
    cam_key = "observation.images.exterior_1_left"
    siblings = [s.rfilename for s in (api.repo_info(repo_id, repo_type="dataset").siblings or [])]
    sib_set = set(siblings)
    ep_index = load_lerobot_v3_episode_video_index(repo_id, cam_key, siblings)
    if not ep_index:
        raise RuntimeError(f"Failed to load LeRobot v3.0 meta/episodes index for {repo_id}/{cam_key}")

    by_mp4: dict[str, list[dict[str, Any]]] = {}
    for em in ep_index.values():
        if em["video_mp4"] in sib_set and em["data_pq"] in sib_set:
            by_mp4.setdefault(em["video_mp4"], []).append(em)
    mp4_list = sorted(by_mp4.keys())
    # Deterministic seeded shuffle across ALL 312 DROID MP4s so every building & collector is represented uniformly!
    rng_shuf = np.random.default_rng(20260401)
    mp4_list = [mp4_list[int(i)] for i in rng_shuf.permutation(len(mp4_list))]
    print(f"[DROID] Indexed {len(ep_index)} episodes across {len(mp4_list)} exterior_1_left MP4s", flush=True)
    done_eps = 0

    for mp4_rel in mp4_list:
        if sum(writer.total_hours_by_split.values()) >= target_hours:
            break
        if max_episodes is not None and done_eps >= max_episodes:
            break
        ep_metas = [em for em in by_mp4[mp4_rel] if f"droid_ep{em['episode_index']:06d}" not in writer.seen_episode_ids]
        if not ep_metas:
            continue
        needed_pqs = sorted({em["data_pq"] for em in ep_metas})
        lp_mp4 = ""
        lp_pqs: dict[str, str] = {}
        pq_tables: dict[str, tuple[Any, np.ndarray]] = {}
        try:
            lp_mp4 = safe_hf_download(repo_id, mp4_rel, repo_type="dataset")
            for pq_rel in needed_pqs:
                lp_p = safe_hf_download(repo_id, pq_rel, repo_type="dataset")
                lp_pqs[pq_rel] = lp_p
                t_cur = pq.read_table(lp_p)
                pq_tables[pq_rel] = (t_cur, t_cur["episode_index"].to_numpy())
        except Exception as exc:
            print(f"[DROID WARN] Skip failed {mp4_rel}: {exc}", flush=True)
            cleanup_hf_file(lp_mp4)
            for lp_p in lp_pqs.values():
                cleanup_hf_file(lp_p)
            continue
        try:
            ep_jobs_all: list[dict[str, Any]] = []
            for em in ep_metas:
                ep_val = int(em["episode_index"])
                tbl, ep_ids = pq_tables[em["data_pq"]]
                rows = np.where(ep_ids == ep_val)[0]
                if rows.size < 15:
                    continue
                r0, t_len = int(rows[0]), int(rows.size)
                exp_vid_frames = (float(em["to_ts"]) - float(em["from_ts"])) * 15.0
                if abs(exp_vid_frames - t_len) > max(4.0, 0.05 * t_len):
                    continue
                start_frame = max(0, int(round(float(em["from_ts"]) * 15.0)))
                ep_tbl = tbl.slice(r0, t_len)
                bldg = str(ep_tbl["building"][0].as_py() or "unknown_bldg")
                coll = str(ep_tbl["collector_id"][0].as_py() or "unknown_coll")
                if bldg == "unknown_bldg" or coll == "unknown_coll":
                    continue
                succ = int(bool(ep_tbl["is_episode_successful"][0].as_py())) if "is_episode_successful" in tbl.column_names else -1
                instr = str(ep_tbl["language_instruction"][0].as_py() or em["task"] or "manipulate object").strip()
                try:
                    raw_a = np.stack(ep_tbl["action"].to_pylist(), axis=0).astype(np.float32)
                    raw_p = np.stack(ep_tbl["observation.state"].to_pylist(), axis=0).astype(np.float32)
                    a52, am, p68, pm = map_embodiment_to_unified_52_68("droid_franka_8dof", raw_action=raw_a, raw_proprio=raw_p)
                    resamp = resample_video_and_kinematics_to_25hz(
                        native_fps=15.0, n_src_frames=t_len, action_52=a52, proprio_68=p68, proprio_mask=pm
                    )
                    ep_jobs_all.append({
                        "ep_val": ep_val, "start_frame": start_frame, "t_len": t_len, "bldg": bldg, "coll": coll,
                        "succ": succ, "instr": instr, "am": am, "pm": pm, "resamp": resamp,
                    })
                except Exception:
                    continue
            # Sort all episodes in this MP4 by frame count so every GPU batch of 16 has <5% padding waste!
            ep_jobs_all.sort(key=lambda x: x["t_len"])

            with ThreadPoolExecutor(max_workers=codec.decode_workers) as pool:
                def _dec_droid(j: dict[str, Any]) -> dict[str, Any] | None:
                    try:
                        ep_fr = decode_video_native_frames_192x320(
                            lp_mp4,
                            j["resamp"]["video_src_frame_indices"],
                            start_frame=j["start_frame"],
                            n_clip_frames=j["t_len"],
                            native_fps=15.0,
                        )
                        if ep_fr.shape[0] != j["resamp"]["n_target_frames_1_plus_4h"]:
                            return None
                        j["fr"] = ep_fr
                        return j
                    except Exception:
                        return None

                for b0 in range(0, len(ep_jobs_all), codec.video_batch_size):
                    if sum(writer.total_hours_by_split.values()) >= target_hours:
                        break
                    if max_episodes is not None and done_eps >= max_episodes:
                        break
                    sub_jobs = ep_jobs_all[b0 : b0 + codec.video_batch_size]
                    if max_episodes is not None:
                        sub_jobs = sub_jobs[: max(0, max_episodes - done_eps)]
                    decoded = [x for x in pool.map(_dec_droid, sub_jobs) if x is not None]
                    if not decoded:
                        continue
                    vid_res = codec.encode_video_batch_1_plus_4h([d["fr"] for d in decoded])
                    for d, (f0, v_grps) in zip(decoded, vid_res):
                        if sum(writer.total_hours_by_split.values()) >= target_hours:
                            break
                        n_hb = d["resamp"]["n_heartbeats"]
                        sp, ckey = assign_v4_split("droid_1_0_1", {"building": d["bldg"], "collector_id": d["coll"]})
                        stream = build_interleaved_heartbeat_stream(
                            video_init_frame0=f0,
                            video_groups=v_grps,
                            action_seq=d["resamp"]["action_25hz"],
                            proprio_seq=d["resamp"]["proprio_25hz"],
                            action_mask=d["am"],
                            proprio_mask=d["pm"],
                            text_by_heartbeat={0: d["instr"].encode("utf-8")},
                        )
                        novel = d["resamp"]["video_group_novel_mask"]
                        for h_i in range(n_hb):
                            if novel[h_i] == 0:
                                m_dup = (stream["heartbeat_id"] == h_i) & (stream["modality_id"] == 0)
                                stream["target_valid_mask"][m_dup] = 0
                        writer.add_episode(
                            split=sp,
                            episode_id=f"droid_ep{d['ep_val']:06d}",
                            cluster_key=ckey,
                            stream=stream,
                            duration_s=n_hb * 0.16,
                            episode_success=d["succ"],
                        )
                        done_eps += 1
        finally:
            cleanup_hf_file(lp_mp4)
            for lp_p in lp_pqs.values():
                cleanup_hf_file(lp_p)


def _load_epic100_narrations() -> dict[str, list[tuple[float, str]]]:
    """Download official EPIC-KITCHENS-100 train+val narration CSVs -> {video_id: [(stop_sec, narration), ...]}."""
    import csv

    out: dict[str, list[tuple[float, str]]] = {}
    base = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master"
    for fn in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
        try:
            raw = urllib.request.urlopen(f"{base}/{fn}", timeout=30).read().decode("utf-8", errors="ignore")
            for row in csv.DictReader(io.StringIO(raw)):
                vid = (row.get("video_id") or "").strip()
                ts = (row.get("stop_timestamp") or "").strip()
                narr = (row.get("narration") or "").strip()
                if not vid or not ts or not narr:
                    continue
                parts = ts.split(":")
                if len(parts) != 3:
                    continue
                stop_s = int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
                out.setdefault(vid, []).append((stop_s, narr))
        except Exception as exc:
            print(f"[EPIC100 WARN] Could not load {fn}: {exc}", flush=True)
    for v in out.values():
        v.sort(key=lambda x: x[0])
    print(f"[EPIC100] Loaded causal narrations for {len(out)} videos", flush=True)
    return out


def encode_long_egocentric_av(
    codec: FrozenV4CodecEngine,
    writer: StreamingV4ShardWriter,
    *,
    corpus_name: str,
    target_hours: float,
    chunk_sec: float = 136.0,  # >= 2.25 min continuous chunks (850 heartbeats) for 2+ min live memory training!
    max_episodes: int | None = None,
) -> None:
    """Encode continuous multi-minute egocentric AV streams (`EPIC-KITCHENS-100` w/ causal narrations or `HoloAssist`)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    repo_id = "a1raman/epic_kitchens_100" if corpus_name == "epic_kitchens_100_long_ego_av" else "SitongGong/HoloAssist"
    siblings = [s.rfilename for s in (api.repo_info(repo_id, repo_type="dataset").siblings or [])]
    epic_narrs: dict[str, list[tuple[float, str]]] = {}
    if corpus_name == "epic_kitchens_100_long_ego_av":
        raw_mp4s = sorted(f for f in siblings if f.upper().endswith(".MP4") and "/videos/" in f)
        # Round-robin across ALL participants P01..P37 so every participant & split (train/dev/test) is covered!
        by_pid: dict[str, list[str]] = {}
        for f in raw_mp4s:
            by_pid.setdefault(f.split("/")[0].upper(), []).append(f)
        mp4s = []
        max_len = max((len(v) for v in by_pid.values()), default=0)
        for idx_i in range(max_len):
            for pid_k in sorted(by_pid.keys()):
                if idx_i < len(by_pid[pid_k]):
                    mp4s.append(by_pid[pid_k][idx_i])
        epic_narrs = _load_epic100_narrations()
    else:
        # Official HoloAssist synchronized AV streams are `pitchshift_videos/<SessionDir>_Video_pitchshift.mp4`
        # (896x504 H.264 + 48 kHz privacy-anonymized AAC coaching+manipulation audio; `Video_compress.mp4` is video-only!).
        raw_mp4s = sorted(
            f for f in siblings if f.startswith("pitchshift_videos/") and f.endswith("_Video_pitchshift.mp4")
        )
        rng_h = np.random.default_rng(20260402)
        mp4s = [raw_mp4s[int(i)] for i in rng_h.permutation(len(raw_mp4s))]

    done_eps = 0
    ego_batch = 8 if codec.video_batch_size >= 16 else 4
    n_cpu = os.cpu_count() or 4
    ego_workers = min(8, max(4, n_cpu)) if codec.video_batch_size >= 16 else min(6, max(4, n_cpu // 2))
    ff_threads = max(2, n_cpu // max(1, ego_workers))

    def _stem_of(rel: str) -> str:
        return Path(rel).stem if corpus_name == "epic_kitchens_100_long_ego_av" else Path(rel).stem.replace("_Video_pitchshift", "")

    # Fast exact resume: skip all MP4s strictly before the last partially/fully seen MP4 (0 redundant GB downloads!)
    if writer.seen_episode_ids and mp4s:
        seen_stems = {eid.rsplit("_seg", 1)[0] for eid in writer.seen_episode_ids if "_seg" in eid}
        last_seen_idx = -1
        for _i, _rel in enumerate(mp4s):
            if _stem_of(_rel) in seen_stems:
                last_seen_idx = _i
        if last_seen_idx > 0:
            print(
                f"[EGO RESUME FAST-SKIP] Skipping {last_seen_idx} already-completed MP4s (resuming at #{last_seen_idx}: {mp4s[last_seen_idx]})",
                flush=True,
            )
            mp4s = mp4s[last_seen_idx:]

    # Async next-MP4 download prefetch (hides multi-GB EPIC-100 download latency completely!)
    def _dl_one(rel: str) -> str | None:
        try:
            return safe_hf_download(repo_id, rel, repo_type="dataset")
        except Exception as exc:
            print(f"[EGO WARN] Skip failed {rel}: {exc}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=1) as dl_pool, ThreadPoolExecutor(max_workers=ego_workers) as dec_pool:
        next_dl_fut = dl_pool.submit(_dl_one, mp4s[0]) if mp4s else None
        for idx_m, mp4_rel in enumerate(mp4s):
            if sum(writer.total_hours_by_split.values()) >= target_hours:
                if next_dl_fut is not None:
                    lp_drop = next_dl_fut.result()
                    if lp_drop:
                        cleanup_hf_file(lp_drop)
                break
            if max_episodes is not None and done_eps >= max_episodes:
                if next_dl_fut is not None:
                    lp_drop = next_dl_fut.result()
                    if lp_drop:
                        cleanup_hf_file(lp_drop)
                break
            if corpus_name == "epic_kitchens_100_long_ego_av":
                pid = mp4_rel.split("/")[0].upper()
                sp, ckey = assign_v4_split(corpus_name, {"participant_id": pid})
                vid_stem = Path(mp4_rel).stem
            else:
                # HoloAssist AV paths are `pitchshift_videos/<SessionDir>_Video_pitchshift.mp4`!
                sess_dir = Path(mp4_rel).stem.replace("_Video_pitchshift", "")
                sess = sess_dir.split("-")[0].upper()
                sp, ckey = assign_v4_split(corpus_name, {"source": "holoassist", "session_prefix": sess})
                vid_stem = sess_dir

            lp_mp4 = next_dl_fut.result() if next_dl_fut is not None else None
            # Immediately prefetch the next MP4 while we decode & encode the current one!
            next_dl_fut = (
                dl_pool.submit(_dl_one, mp4s[idx_m + 1])
                if (idx_m + 1 < len(mp4s) and sum(writer.total_hours_by_split.values()) < target_hours)
                else None
            )
            if not lp_mp4:
                continue
            try:
                dur = _probe_duration_sec(lp_mp4)
                vid_narrs = epic_narrs.get(vid_stem, [])
                segs: list[tuple[int, float, float]] = []
                t_cur = 0.0
                seg_idx = 0
                while t_cur + 10.0 < dur:
                    if f"{vid_stem}_seg{seg_idx:02d}" not in writer.seen_episode_ids:
                        segs.append((seg_idx, t_cur, min(chunk_sec, dur - t_cur)))
                    t_cur += chunk_sec
                    seg_idx += 1
                if not segs:
                    continue

                def _dec_ego_seg(s_item: tuple[int, float, float]) -> dict[str, Any] | None:
                    s_i, s_t0, s_dur = s_item
                    try:
                        fr = decode_video_25fps_192x320(lp_mp4, start_sec=s_t0, max_sec=s_dur, threads=ff_threads)
                        pcm = decode_audio_24khz_mono(lp_mp4, start_sec=s_t0, max_sec=s_dur)
                        if fr.shape[0] < 9:
                            return None
                        has_aud = pcm.size >= 24000 and audit_pcm_nonsilence(pcm, 24000)["audio_valid"]
                        n_hb = min((fr.shape[0] - 1) // 4, int(pcm.size // 3840)) if has_aud else ((fr.shape[0] - 1) // 4)
                        if n_hb < 10:
                            return None
                        return {
                            "seg_idx": s_i, "t_cur": s_t0, "n_hb": n_hb, "has_aud": has_aud,
                            "fr": fr[: 1 + 4 * n_hb], "pcm": pcm[: n_hb * 3840] if has_aud else None,
                        }
                    except Exception:
                        return None

                # Double-buffered batch decode prefetch: decode batch k+1 across all vCPUs while GPU encodes batch k!
                batches = [segs[b0 : b0 + ego_batch] for b0 in range(0, len(segs), ego_batch)]
                cur_futs = [dec_pool.submit(_dec_ego_seg, s_it) for s_it in batches[0]]
                for bi, _chunk_segs in enumerate(batches):
                    if sum(writer.total_hours_by_split.values()) >= target_hours:
                        break
                    if max_episodes is not None and done_eps >= max_episodes:
                        break
                    decoded = [x for x in (f.result() for f in cur_futs) if x is not None]
                    cur_futs = (
                        [dec_pool.submit(_dec_ego_seg, s_it) for s_it in batches[bi + 1]]
                        if (bi + 1 < len(batches) and sum(writer.total_hours_by_split.values()) < target_hours)
                        else []
                    )
                    if max_episodes is not None:
                        decoded = decoded[: max(0, max_episodes - done_eps)]
                    if not decoded:
                        continue
                    vid_res = codec.encode_video_batch_1_plus_4h([d["fr"] for d in decoded])
                    for d, (f0, v_grps) in zip(decoded, vid_res):
                        if sum(writer.total_hours_by_split.values()) >= target_hours:
                            break
                        if max_episodes is not None and done_eps >= max_episodes:
                            break
                        n_hb = d["n_hb"]
                        a_codes = codec.encode_audio_mimi_16rvq(d["pcm"]) if d["has_aud"] and d["pcm"] is not None else None
                        t_map: dict[int, bytes] = {}
                        for stop_s, narr_txt in vid_narrs:
                            rel_s = stop_s - d["t_cur"]
                            if 0.0 < rel_s <= n_hb * 0.16:
                                hb_i = min(n_hb - 1, max(0, int(math.ceil(rel_s / 0.16)) - 1))
                                t_map[hb_i] = t_map.get(hb_i, b"") + (narr_txt + "\n").encode("utf-8")
                        stream = build_interleaved_heartbeat_stream(
                            video_init_frame0=f0,
                            video_groups=v_grps,
                            audio_codes=a_codes,
                            text_by_heartbeat=t_map or None,
                        )
                        writer.add_episode(
                            split=sp,
                            episode_id=f"{vid_stem}_seg{d['seg_idx']:02d}",
                            cluster_key=ckey,
                            stream=stream,
                            duration_s=n_hb * 0.16,
                            audio_valid=d["has_aud"],
                        )
                        done_eps += 1
                # Drain any prefetched segment decode futures before deleting `lp_mp4`
                for _f in cur_futs:
                    try:
                        _f.result()
                    except Exception:
                        pass
            finally:
                cleanup_hf_file(lp_mp4)


def encode_fineweb_text_code(
    writer: StreamingV4ShardWriter,
    *,
    target_bytes_m: float,
    max_docs: int | None = None,
) -> None:
    """Stream 70% `HuggingFaceTB/smollm-corpus` (`fineweb-edu-dedup`) + 30% `codeparrot/github-code-clean` (repo-disjoint)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    total_target_bytes = int(target_bytes_m * 1e6)
    text_target_bytes = int(0.70 * total_target_bytes)
    text_max_docs = None if max_docs is None else max(1, (max_docs + 1) // 2)
    code_max_docs = None if max_docs is None else max(1, max_docs - (text_max_docs or 0))

    # 1. Educational English prose (`fineweb-edu-dedup`, ~70%)
    fw_nodes = list(api.list_repo_tree("HuggingFaceTB/smollm-corpus", path_in_repo="fineweb-edu-dedup", repo_type="dataset"))
    fw_pqs = sorted(n.path for n in fw_nodes if n.path.endswith(".parquet"))
    done_text = 0
    for pq_rel in fw_pqs:
        if sum(writer.total_packets_by_split.values()) >= text_target_bytes:
            break
        if text_max_docs is not None and done_text >= text_max_docs:
            break
        print(f"[TEXT] Downloading {pq_rel}...", flush=True)
        lp = hf_hub_download("HuggingFaceTB/smollm-corpus", pq_rel, repo_type="dataset")
        try:
            pf = pq.ParquetFile(lp)
            for rg_i in range(pf.num_row_groups):
                if sum(writer.total_packets_by_split.values()) >= text_target_bytes:
                    break
                if text_max_docs is not None and done_text >= text_max_docs:
                    break
                rg = pf.read_row_group(rg_i, columns=["text"])
                for txt in rg["text"].to_pylist():
                    if sum(writer.total_packets_by_split.values()) >= text_target_bytes:
                        break
                    if text_max_docs is not None and done_text >= text_max_docs:
                        break
                    if not isinstance(txt, str) or len(txt) < 128:
                        continue
                    raw_b = txt.encode("utf-8")[:16384]
                    doc_sha = hash_payload_sha256(np.frombuffer(raw_b, dtype=np.uint8))
                    sp, ckey = assign_v4_split("fineweb_edu_and_stack_code", {"doc_sha256": doc_sha})
                    t_map = {i: raw_b[i * 64 : (i + 1) * 64] for i in range((len(raw_b) + 63) // 64)}
                    stream = build_interleaved_heartbeat_stream(text_by_heartbeat=t_map)
                    writer.add_episode(
                        split=sp,
                        episode_id=f"fw_{doc_sha[:12]}",
                        cluster_key=ckey,
                        stream=stream,
                        duration_s=0.0,
                        extra_meta={"subcorpus": "fineweb_edu"},
                    )
                    done_text += 1
        finally:
            cleanup_hf_file(lp)

    # 2. Clean open-source code (`codeparrot/github-code-clean`, ~30%, strictly repository-disjoint!)
    code_nodes = list(api.list_repo_tree("codeparrot/github-code-clean", path_in_repo="data", repo_type="dataset"))
    code_pqs = sorted(n.path for n in code_nodes if n.path.endswith(".parquet"))
    done_code = 0
    for pq_rel in code_pqs:
        if sum(writer.total_packets_by_split.values()) >= total_target_bytes:
            break
        if code_max_docs is not None and done_code >= code_max_docs:
            break
        print(f"[CODE] Downloading {pq_rel}...", flush=True)
        lp = hf_hub_download("codeparrot/github-code-clean", pq_rel, repo_type="dataset")
        try:
            pf = pq.ParquetFile(lp)
            for rg_i in range(pf.num_row_groups):
                if sum(writer.total_packets_by_split.values()) >= total_target_bytes:
                    break
                if code_max_docs is not None and done_code >= code_max_docs:
                    break
                rg = pf.read_row_group(rg_i, columns=["code", "repo_name", "language"])
                for code_txt, repo_name, lang in zip(
                    rg["code"].to_pylist(), rg["repo_name"].to_pylist(), rg["language"].to_pylist()
                ):
                    if sum(writer.total_packets_by_split.values()) >= total_target_bytes:
                        break
                    if code_max_docs is not None and done_code >= code_max_docs:
                        break
                    if not isinstance(code_txt, str) or len(code_txt) < 128 or not repo_name:
                        continue
                    raw_b = code_txt.encode("utf-8")[:16384]
                    doc_sha = hash_payload_sha256(np.frombuffer(raw_b, dtype=np.uint8))
                    sp, ckey = assign_v4_split("fineweb_edu_and_stack_code", {"repo_name": str(repo_name), "doc_sha256": doc_sha})
                    t_map = {i: raw_b[i * 64 : (i + 1) * 64] for i in range((len(raw_b) + 63) // 64)}
                    stream = build_interleaved_heartbeat_stream(text_by_heartbeat=t_map)
                    writer.add_episode(
                        split=sp,
                        episode_id=f"code_{doc_sha[:12]}",
                        cluster_key=ckey,
                        stream=stream,
                        duration_s=0.0,
                        extra_meta={"subcorpus": f"github_code_{str(lang or 'other').lower()}"},
                    )
                    done_code += 1
        finally:
            cleanup_hf_file(lp)


def encode_agibot_world_2026(
    codec: FrozenV4CodecEngine,
    writer: StreamingV4ShardWriter,
    *,
    target_hours: float,
    max_episodes: int | None = None,
) -> None:
    """Encode `agibot-world/AgiBotWorld2026` PhysicalReasoning LeRobot v2.1 tasks (`action[44]`, `state[216]` + 6D wrench + `succeed`)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    repo_id = "agibot-world/AgiBotWorld2026"
    base_prefix = "PhysicalReasoning/opensourcev4-20261001/lerobot"
    task_nodes = list(api.list_repo_tree(repo_id, path_in_repo=base_prefix, repo_type="dataset"))
    task_dirs = sorted(n.path for n in task_nodes if n.path.split("/")[-1].startswith("task_"))
    done_eps = 0
    per_task_hour_cap = max(2.5, (target_hours / max(1, len(task_dirs))) * 1.35)
    for tdir in task_dirs:
        if sum(writer.total_hours_by_split.values()) >= target_hours:
            break
        if max_episodes is not None and done_eps >= max_episodes:
            break
        task_id = tdir.split("/")[-1]
        task_prefix = f"agibot_{task_id}_"
        def _task_hours_done() -> float:
            return sum(float(r.get("duration_s", 0.0)) for r in writer.all_episode_records if str(r.get("episode_id", "")).startswith(task_prefix)) / 3600.0
        if _task_hours_done() >= per_task_hour_cap:
            print(f"[AGIBOT RESUME SKIP] {task_id} already at {_task_hours_done():.2f}h >= cap {per_task_hour_cap:.2f}h", flush=True)
            continue
        sub_nodes = list(api.list_repo_tree(repo_id, path_in_repo=tdir, repo_type="dataset"))
        for sn in sorted(sub_nodes, key=lambda x: x.path):
            if sum(writer.total_hours_by_split.values()) >= target_hours:
                break
            if _task_hours_done() >= per_task_hour_cap:
                break
            if max_episodes is not None and done_eps >= max_episodes:
                break
            scene_id = sn.path.split("/")[-1]
            data_dir = f"{sn.path}/data/chunk-000"
            try:
                pq_nodes = list(api.list_repo_tree(repo_id, path_in_repo=data_dir, repo_type="dataset"))
            except Exception:
                continue
            pq_files = [
                p for p in sorted(n.path for n in pq_nodes if n.path.endswith(".parquet"))
                if f"agibot_{task_id}_{scene_id}_{Path(p).stem}" not in writer.seen_episode_ids
            ]
            print(f"[AGIBOT] {task_id}/{scene_id} episodes={len(pq_files)}", flush=True)

            def _dl_and_dec_agibot(pq_rel: str) -> dict[str, Any] | None:
                ep_stem = Path(pq_rel).stem
                mp4_rel = f"{sn.path}/videos/chunk-000/observation.images.top_head/{ep_stem}.mp4"
                lp_pq = lp_mp4 = ""
                try:
                    lp_pq = safe_hf_download(repo_id, pq_rel, repo_type="dataset")
                    lp_mp4 = safe_hf_download(repo_id, mp4_rel, repo_type="dataset")
                    tbl = pq.read_table(lp_pq)
                    t_len = tbl.num_rows
                    if t_len < 20:
                        return None
                    raw_a = np.stack(tbl["action"].to_pylist(), axis=0).astype(np.float32)
                    raw_s = np.stack(tbl["observation.state"].to_pylist(), axis=0).astype(np.float32)
                    # AgiBotWorld2026 PhysicalReasoning upstream `succeed` is empirically 100% False (0/3,058 positive);
                    # mark strictly unlabeled (-1) rather than emitting degenerate constant-zero success labels!
                    succ = -1
                    a52, am, p68, pm = map_embodiment_to_unified_52_68("agibot_g2a_bimanual", raw_action=raw_a, raw_proprio=raw_s)
                    resamp = resample_video_and_kinematics_to_25hz(
                        native_fps=30.0, n_src_frames=t_len, action_52=a52, proprio_68=p68, proprio_mask=pm
                    )
                    ep_fr = decode_video_native_frames_192x320(lp_mp4, resamp["video_src_frame_indices"])
                    if ep_fr.shape[0] != resamp["n_target_frames_1_plus_4h"]:
                        return None
                    return {"ep_stem": ep_stem, "succ": succ, "am": am, "pm": pm, "resamp": resamp, "fr": ep_fr}
                except Exception:
                    return None
                finally:
                    cleanup_hf_file(lp_mp4)
                    cleanup_hf_file(lp_pq)

            with ThreadPoolExecutor(max_workers=codec.decode_workers) as pool:
                for b0 in range(0, len(pq_files), codec.video_batch_size):
                    if sum(writer.total_hours_by_split.values()) >= target_hours:
                        break
                    if _task_hours_done() >= per_task_hour_cap:
                        break
                    if max_episodes is not None and done_eps >= max_episodes:
                        break
                    sub_pqs = pq_files[b0 : b0 + codec.video_batch_size]
                    if max_episodes is not None:
                        sub_pqs = sub_pqs[: max(0, max_episodes - done_eps)]
                    decoded = [x for x in pool.map(_dl_and_dec_agibot, sub_pqs) if x is not None]
                    if not decoded:
                        continue
                    vid_res = codec.encode_video_batch_1_plus_4h([d["fr"] for d in decoded])
                    for d, (f0, v_grps) in zip(decoded, vid_res):
                        n_hb = d["resamp"]["n_heartbeats"]
                        sp, ckey = assign_v4_split(
                            "agibot_and_aloha_sim", {"source": "agibot", "task_id": task_id, "scene_id": scene_id}
                        )
                        stream = build_interleaved_heartbeat_stream(
                            video_init_frame0=f0,
                            video_groups=v_grps,
                            action_seq=d["resamp"]["action_25hz"],
                            proprio_seq=d["resamp"]["proprio_25hz"],
                            action_mask=d["am"],
                            proprio_mask=d["pm"],
                            text_by_heartbeat={0: f"agibot {task_id}".encode("utf-8")},
                        )
                        writer.add_episode(
                            split=sp,
                            episode_id=f"agibot_{task_id}_{scene_id}_{d['ep_stem']}",
                            cluster_key=ckey,
                            stream=stream,
                            duration_s=n_hb * 0.16,
                            episode_success=d["succ"],
                        )
                        done_eps += 1


def build_rh20t_kinematic_action_and_proprio(raw_kin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct causal 1-step kinematic target `(action = q[t+1], proprio = q[t])` for RH20T (zero identity shortcut!)."""
    q = np.asarray(raw_kin, dtype=np.float32)
    if q.ndim != 2 or q.shape[0] < 1:
        raise ValueError(f"Expected 2D raw_kin [T, D] with T >= 1, got {q.shape}")
    if q.shape[0] == 1:
        return q.copy(), q.copy()
    raw_prop = q.copy()
    raw_act = np.concatenate([q[1:], q[-1:]], axis=0)
    return raw_act, raw_prop


def encode_rh20t_contact_av_force(
    codec: FrozenV4CodecEngine,
    writer: StreamingV4ShardWriter,
    *,
    target_hours: float,
    cfgs: tuple[str, ...] = ("cfg3", "cfg1", "cfg2", "cfg4", "cfg7", "cfg6"),
    max_episodes: int | None = None,
) -> None:
    """Encode `tonyfanggalaxies/RH20T` quad-modal Video + Contact Audio (SNR-filtered) + 6D Force/Torque + Joints.

    Extracts 1 camera per episode directly to disk (zero multi-GB RAM buffering) and immediately removes HF cache
    tarball blobs after each `cfg` so even an L4 SSD (`~112 GB`) streams safely!
    """
    from huggingface_hub import hf_hub_download

    done_eps = 0
    # Fast exact resume: skip cfgs strictly before the latest rh20t_cfg already present in paired HF checkpoints!
    seen_cfgs = {str(r.get("rh20t_cfg", "")) for r in writer.all_episode_records if r.get("rh20t_cfg")}
    last_cfg_i = max((i for i, c in enumerate(cfgs) if c in seen_cfgs), default=0)
    if last_cfg_i > 0:
        print(f"[RH20T RESUME FAST-SKIP] Skipping completed cfgs {cfgs[:last_cfg_i]}; resuming at {cfgs[last_cfg_i]}", flush=True)
        cfgs = cfgs[last_cfg_i:]
    for cfg in cfgs:
        if sum(writer.total_hours_by_split.values()) >= target_hours:
            break
        if max_episodes is not None and done_eps >= max_episodes:
            break
        print(f"[RH20T] Downloading matched color + lowdim archives for {cfg}...", flush=True)
        lp_col = hf_hub_download("tonyfanggalaxies/RH20T", f"RH20T_color/RH20T_{cfg}.tar.gz", repo_type="dataset")
        lp_low = hf_hub_download("tonyfanggalaxies/RH20T", f"RH20T_lowdim/RH20T_{cfg}.tar.gz", repo_type="dataset")
        work_dir = Path(f"/tmp/rh20t_{cfg}")
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Extract 1 camera per episode straight to disk (keeps RAM < 100 MiB!)
            ep_cam: dict[str, str] = {}
            with tarfile.open(lp_col, "r:gz") as tf_c:
                for m in tf_c:
                    if not m.isfile():
                        continue
                    parts = m.name.split("/")
                    if len(parts) < 5 or not (m.name.endswith("color.mp4") or m.name.endswith("timestamps.npy")):
                        continue
                    ep_dir, cam_dir, fn = parts[1], parts[2], parts[-1]
                    if ep_dir not in ep_cam:
                        if max_episodes is not None and len(ep_cam) >= max_episodes * 2:
                            continue
                        ep_cam[ep_dir] = cam_dir
                    if ep_cam[ep_dir] == cam_dir:
                        fobj = tf_c.extractfile(m)
                        if fobj is not None:
                            d_ep = work_dir / ep_dir
                            d_ep.mkdir(parents=True, exist_ok=True)
                            (d_ep / fn).write_bytes(fobj.read())
            cleanup_hf_file(lp_col)
            lp_col = ""

            with tarfile.open(lp_low, "r:gz") as tf_l:
                for m in tf_l:
                    if not m.isfile():
                        continue
                    parts = m.name.split("/")
                    if len(parts) < 4:
                        continue
                    ep_dir = parts[1]
                    if ep_dir not in ep_cam:
                        continue
                    bn = parts[-1]
                    if bn.endswith(".wav") or bn in ("force_torque.npy", "joint.npy", "gripper.npy"):
                        fobj = tf_l.extractfile(m)
                        if fobj is not None:
                            d_ep = work_dir / ep_dir
                            d_ep.mkdir(parents=True, exist_ok=True)
                            if bn.endswith(".wav"):
                                (d_ep / "contact.wav").write_bytes(fobj.read())
                                (d_ep / "wav_name.txt").write_text(bn)
                            else:
                                (d_ep / bn).write_bytes(fobj.read())
            cleanup_hf_file(lp_low)
            lp_low = ""

            n_cfg_ok = 0
            n_cfg_aud = 0
            n_cfg_err = 0
            ep_items = [(k, v) for k, v in sorted(ep_cam.items()) if f"rh20t_{k}" not in writer.seen_episode_ids]

            def _prep_rh20t_ep(pair: tuple[str, str]) -> dict[str, Any] | None:
                ep_dir, cam_dir = pair
                d_ep = work_dir / ep_dir
                p_mp4 = d_ep / "color.mp4"
                p_ts = d_ep / "timestamps.npy"
                p_j = d_ep / "joint.npy"
                p_ft = d_ep / "force_torque.npy"
                if not (p_mp4.is_file() and p_ts.is_file() and p_j.is_file() and p_ft.is_file()):
                    shutil.rmtree(d_ep, ignore_errors=True)
                    return None
                try:
                    sp, ckey = assign_v4_split("rh20t_contact_av_force", {"episode_dir": ep_dir})
                    cam_id = cam_dir.replace("cam_", "")
                    ts_vid = np.load(str(p_ts), allow_pickle=True).astype(np.float64)
                    if ts_vid.size < 15:
                        return None
                    t_start_ms, t_end_ms = float(ts_vid[0]), float(ts_vid[-1])
                    dur_s = (t_end_ms - t_start_ms) / 1000.0
                    if dur_s < 1.5 or dur_s > 90.0:
                        return None
                    n_hb = int(math.floor(dur_s / 0.16))
                    if n_hb < 4:
                        return None

                    fr = decode_video_25fps_192x320(str(p_mp4), max_sec=n_hb * 0.16 + 0.08)
                    if fr.shape[0] < 1 + 4 * n_hb:
                        n_hb = (fr.shape[0] - 1) // 4
                    if n_hb < 4:
                        return None

                    j_dict = np.load(str(p_j), allow_pickle=True).item()
                    ft_dict = np.load(str(p_ft), allow_pickle=True).item()
                    p_g = d_ep / "gripper.npy"
                    g_dict = np.load(str(p_g), allow_pickle=True).item() if p_g.is_file() else {}
                    j_cam = j_dict.get(cam_id) or next(iter(j_dict.values()))
                    ft_cam = ft_dict.get(cam_id) or next(iter(ft_dict.values()))
                    g_cam = g_dict.get(cam_id) or (next(iter(g_dict.values())) if g_dict else {})

                    j_ts = np.array(sorted(j_cam.keys()), dtype=np.float64)
                    j_vals_raw = [np.asarray(j_cam[int(t)], dtype=np.float32).reshape(-1) for t in j_ts]
                    n_arm = min(7, int(j_vals_raw[0].size))
                    if n_arm not in (6, 7):
                        return None
                    j_vals = np.stack([v[:n_arm] for v in j_vals_raw], axis=0)
                    ft_ts = np.array([float(x["timestamp"]) for x in ft_cam], dtype=np.float64)
                    ft_vals = np.stack([np.asarray(x["zeroed"], dtype=np.float32)[:6] for x in ft_cam], axis=0)

                    q_ts_ms = t_start_ms + (np.arange(1, 4 * n_hb + 1, dtype=np.float64) * 40.0)
                    j_idx = np.clip(np.searchsorted(j_ts, q_ts_ms, side="right") - 1, 0, j_vals.shape[0] - 1)
                    ft_idx = np.clip(np.searchsorted(ft_ts, q_ts_ms, side="right") - 1, 0, ft_vals.shape[0] - 1)
                    grip_col = np.zeros((4 * n_hb, 1), dtype=np.float32)
                    if g_cam:
                        g_ts = np.array(sorted(g_cam.keys()), dtype=np.float64)
                        g_vals = np.array([float(g_cam[int(t)]["gripper_info"][0]) for t in g_ts], dtype=np.float32)
                        g_idx = np.clip(np.searchsorted(g_ts, q_ts_ms, side="right") - 1, 0, g_vals.shape[0] - 1)
                        grip_col[:, 0] = g_vals[g_idx] / 100.0

                    raw_kin = np.concatenate([j_vals[j_idx], grip_col], axis=-1)
                    raw_act, raw_prop = build_rh20t_kinematic_action_and_proprio(raw_kin)
                    emb_name = "rh20t_single_arm_8dof" if n_arm == 7 else "rh20t_single_arm_7dof"
                    a52, am, p68, pm = map_embodiment_to_unified_52_68(
                        emb_name, raw_action=raw_act, raw_proprio=raw_prop, raw_force_torque_right=ft_vals[ft_idx]
                    )

                    pcm_valid = None
                    p_wav = d_ep / "contact.wav"
                    if p_wav.is_file():
                        sr, arr = wavfile.read(str(p_wav))
                        x = arr.astype(np.float32) / (32768.0 if arr.dtype == np.int16 else float(np.max(np.abs(arr)) + 1e-9))
                        if x.ndim > 1:
                            x = x.mean(axis=-1)
                        wn = (d_ep / "wav_name.txt").read_text().strip() if (d_ep / "wav_name.txt").is_file() else ""
                        try:
                            wav_t0 = float(Path(wn).stem.split("_")[-1])
                            off_s = max(0.0, (t_start_ms - wav_t0) / 1000.0)
                            s_off = int(round(off_s * sr))
                            if s_off < x.size:
                                x = x[s_off:]
                        except Exception:
                            pass
                        g_gcd = math.gcd(int(sr), 24000)
                        x24 = resample_poly(x.astype(np.float64), 24000 // g_gcd, int(sr) // g_gcd).astype(np.float32)
                        if x24.size >= n_hb * 3840 and audit_pcm_nonsilence(x24[: n_hb * 3840], 24000)["audio_valid"]:
                            pcm_valid = x24[: n_hb * 3840]

                    return {
                        "ep_dir": ep_dir, "sp": sp, "ckey": ckey, "n_hb": n_hb, "n_arm": n_arm,
                        "t_start_ms": t_start_ms, "ts_vid": ts_vid,
                        "fr": fr[: 1 + 4 * n_hb], "pcm": pcm_valid,
                        "a52": a52, "am": am, "p68": p68, "pm": pm,
                    }
                except Exception as exc:
                    return {"_err": repr(exc), "ep_dir": ep_dir}
                finally:
                    shutil.rmtree(d_ep, ignore_errors=True)

            with ThreadPoolExecutor(max_workers=codec.decode_workers) as pool:
                for b0 in range(0, len(ep_items), codec.video_batch_size):
                    if sum(writer.total_hours_by_split.values()) >= target_hours:
                        break
                    if max_episodes is not None and done_eps >= max_episodes:
                        break
                    sub_pairs = ep_items[b0 : b0 + codec.video_batch_size]
                    if max_episodes is not None:
                        sub_pairs = sub_pairs[: max(0, max_episodes - done_eps)]
                    res_list = list(pool.map(_prep_rh20t_ep, sub_pairs))
                    decoded = []
                    for r in res_list:
                        if r is None:
                            continue
                        if "_err" in r:
                            n_cfg_err += 1
                            if n_cfg_err <= 3:
                                print(f"[RH20T {cfg} WARN] {r['ep_dir']}: {r['_err']}", flush=True)
                            continue
                        decoded.append(r)
                    if not decoded:
                        continue
                    vid_res = codec.encode_video_batch_1_plus_4h([d["fr"] for d in decoded])
                    for d, (f0, v_grps) in zip(decoded, vid_res):
                        n_hb = d["n_hb"]
                        aud_valid = d["pcm"] is not None
                        a_codes = codec.encode_audio_mimi_16rvq(d["pcm"]) if aud_valid else None
                        # Upstream RH20T has no natural-language task descriptions; omit raw directory names from MOD_TEXT!
                        stream = build_interleaved_heartbeat_stream(
                            video_init_frame0=f0,
                            video_groups=v_grps,
                            audio_codes=a_codes,
                            action_seq=d["a52"],
                            proprio_seq=d["p68"],
                            action_mask=d["am"],
                            proprio_mask=d["pm"],
                            text_by_heartbeat=None,
                        )
                        # Mask final motor tick where 1-step kinematic target q[t+1] is held at boundary!
                        m_last_mot = stream["motor_idx"] == (4 * n_hb - 1)
                        stream["target_valid_mask"][m_last_mot] = 0
                        vid_q_ms = d["t_start_ms"] + (np.arange(1 + 4 * n_hb, dtype=np.float64) * 40.0)
                        v_src_idx = np.clip(np.searchsorted(d["ts_vid"], vid_q_ms, side="right") - 1, 0, d["ts_vid"].size - 1)
                        for h_i in range(n_hb):
                            if len(set(int(z) for z in v_src_idx[1 + 4 * h_i : 1 + 4 * (h_i + 1)])) < 2:
                                m_dup = (stream["heartbeat_id"] == h_i) & (stream["modality_id"] == 0)
                                stream["target_valid_mask"][m_dup] = 0
                        writer.add_episode(
                            split=d["sp"],
                            episode_id=f"rh20t_{d['ep_dir']}",
                            cluster_key=d["ckey"],
                            stream=stream,
                            duration_s=n_hb * 0.16,
                            audio_valid=aud_valid,
                            extra_meta={"rh20t_cfg": cfg, "arm_dof": d["n_arm"]},
                        )
                        done_eps += 1
                        n_cfg_ok += 1
                        n_cfg_aud += int(aud_valid)
            print(f"[RH20T {cfg}] encoded={n_cfg_ok} errs={n_cfg_err} audio_valid={n_cfg_aud} ({100.0 * n_cfg_aud / max(1, n_cfg_ok):.1f}%)", flush=True)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            cleanup_hf_file(lp_col)
            cleanup_hf_file(lp_low)


def encode_libriheavy_speech(
    codec: FrozenV4CodecEngine,
    writer: StreamingV4ShardWriter,
    *,
    target_hours: float,
    max_episodes: int | None = None,
    batch_size: int = 32,
    split_filter: str = "",
) -> None:
    """Encode general captioned audio (`OpenSound/AudioCaps`, ~100h, shared `yt::` split) + speech (`mythicinfinity/libriheavy`) with batched Mimi GPU encoding."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    sp_filt = split_filter.strip().lower()
    ac_target_h = min(80.0, target_hours * 0.15) if not sp_filt else 80.0
    ac_max_eps = None if max_episodes is None else max(1, max_episodes // 2)
    lh_max_eps = None if max_episodes is None else max(1, max_episodes - (ac_max_eps or 0))

    def _ac_hours_done() -> float:
        return sum(
            float(r.get("duration_s", 0.0))
            for r in writer.all_episode_records
            if (r.get("subcorpus") == "audiocaps" or str(r.get("episode_id", "")).startswith("ac_"))
            and (not sp_filt or str(r.get("split", "")) == sp_filt)
        ) / 3600.0

    def _active_hours_done() -> float:
        return float(writer.total_hours_by_split.get(sp_filt, 0.0)) if sp_filt else float(sum(writer.total_hours_by_split.values()))

    def _dec_one_aud(item: tuple[bytes, str, str, str, str, bytes, str]) -> tuple[np.ndarray, str, str, str, bytes, str] | None:
        raw_b_aud, ext, sp, ckey, eid, raw_b_txt, subc = item
        try:
            pcm = decode_audio_bytes_24khz_mono(raw_b_aud, fallback_ext=ext)
            n_hb = int(pcm.size // 3840)
            if n_hb < 4 or not audit_pcm_nonsilence(pcm[: n_hb * 3840], 24000)["audio_valid"]:
                return None
            return (pcm[: n_hb * 3840], sp, ckey, eid, raw_b_txt, subc)
        except Exception:
            return None

    def _flush_audio_batch(pending: list[tuple[np.ndarray, str, str, str, bytes, str]]) -> None:
        if not pending:
            return
        codes_list = codec.encode_audio_batch_mimi_16rvq([p[0] for p in pending])
        for (pcm_seg, sp, ckey, eid, raw_b, subc), a_codes in zip(pending, codes_list):
            n_hb = a_codes.shape[0] // 2
            if n_hb < 2:
                continue
            a_codes = a_codes[: n_hb * 2]
            stream = build_interleaved_heartbeat_stream(
                audio_codes=a_codes,
                text_by_heartbeat={n_hb - 1: raw_b},
            )
            writer.add_episode(
                split=sp,
                episode_id=eid,
                cluster_key=ckey,
                stream=stream,
                duration_s=n_hb * 0.16,
                audio_valid=True,
                extra_meta={"subcorpus": subc},
            )
        pending.clear()

    # 1. General captioned environmental audio (`OpenSound/AudioCaps`, cluster-keyed by canonical `yt::{youtube_id}`!)
    ac_nodes = list(api.list_repo_tree("OpenSound/AudioCaps", path_in_repo="data", repo_type="dataset"))
    ac_pqs = sorted(n.path for n in ac_nodes if n.path.endswith(".parquet"))
    done_ac = 0
    with ThreadPoolExecutor(max_workers=codec.decode_workers) as pool:
        for pq_rel in ac_pqs:
            if _ac_hours_done() >= ac_target_h:
                break
            if ac_max_eps is not None and done_ac >= ac_max_eps:
                break
            lp = hf_hub_download("OpenSound/AudioCaps", pq_rel, repo_type="dataset")
            try:
                tbl = pq.read_table(lp, columns=["audio", "youtube_id", "caption", "audiocap_id"])
                raw_jobs: list[tuple[bytes, str, str, str, str, bytes, str]] = []
                for a_obj, ytid, cap, acid in zip(
                    tbl["audio"].to_pylist(),
                    tbl["youtube_id"].to_pylist(),
                    tbl["caption"].to_pylist(),
                    tbl["audiocap_id"].to_pylist(),
                ):
                    if ac_max_eps is not None and (done_ac + len(raw_jobs)) >= ac_max_eps:
                        break
                    if not a_obj or not isinstance(a_obj, dict) or not a_obj.get("bytes") or not ytid or not cap:
                        continue
                    if f"ac_{acid}" in writer.seen_episode_ids:
                        continue
                    sp, ckey = assign_v4_split(
                        "wavcaps_and_libriheavy_speech", {"source": "audiocaps", "youtube_id": str(ytid)}
                    )
                    if sp_filt and sp != sp_filt:
                        continue
                    raw_jobs.append((a_obj["bytes"], ".wav", sp, ckey, f"ac_{acid}", str(cap).strip().encode("utf-8"), "audiocaps"))
                for b0 in range(0, len(raw_jobs), 64):
                    if _ac_hours_done() >= ac_target_h:
                        break
                    decoded = [x for x in pool.map(_dec_one_aud, raw_jobs[b0 : b0 + 64]) if x is not None]
                    if decoded:
                        done_ac += len(decoded)
                        _flush_audio_batch(decoded)
            finally:
                cleanup_hf_file(lp)
        print(f"[AUDIOCAPS] split_filter={sp_filt or 'all'} done_eps={done_ac} ac_hours={_ac_hours_done():.2f}h", flush=True)

        # 2. Multi-speaker continuous speech (`mythicinfinity/libriheavy`, speaker-disjoint!)
        siblings = [s.rfilename for s in (api.repo_info("mythicinfinity/libriheavy", repo_type="dataset").siblings or [])]
        pq_files = sorted(f for f in siblings if f.startswith("default/large/") and f.endswith(".parquet"))
        done_lh = 0
        for pq_rel in pq_files:
            if _active_hours_done() >= target_hours:
                break
            if lh_max_eps is not None and done_lh >= lh_max_eps:
                break
            print(
                f"[LIBRIHEAVY] Reading {pq_rel} (split_filter={sp_filt or 'all'} hours={_active_hours_done():.1f}/{target_hours:.1f}h)...",
                flush=True,
            )
            lp = hf_hub_download("mythicinfinity/libriheavy", pq_rel, repo_type="dataset")
            try:
                tbl = pq.read_table(lp, columns=["audio", "speaker_id", "text_transcription", "id"])
                raw_jobs = []
                for a_obj, spk, txt, uid in zip(
                    tbl["audio"].to_pylist(),
                    tbl["speaker_id"].to_pylist(),
                    tbl["text_transcription"].to_pylist(),
                    tbl["id"].to_pylist(),
                ):
                    if lh_max_eps is not None and (done_lh + len(raw_jobs)) >= lh_max_eps:
                        break
                    if not a_obj or not isinstance(a_obj, dict) or not a_obj.get("bytes") or not txt:
                        continue
                    if f"lh_{uid}" in writer.seen_episode_ids:
                        continue
                    sp, ckey = assign_v4_split("wavcaps_and_libriheavy_speech", {"source": "libriheavy", "speaker_id": str(spk)})
                    if sp_filt and sp != sp_filt:
                        continue
                    raw_jobs.append((a_obj["bytes"], ".flac", sp, ckey, f"lh_{uid}", str(txt).strip().encode("utf-8"), "libriheavy"))
                for b0 in range(0, len(raw_jobs), 64):
                    if _active_hours_done() >= target_hours:
                        break
                    decoded = [x for x in pool.map(_dec_one_aud, raw_jobs[b0 : b0 + 64]) if x is not None]
                    if decoded:
                        done_lh += len(decoded)
                        _flush_audio_batch(decoded)
            finally:
                cleanup_hf_file(lp)


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v4 Gate 2 Streaming Parallel GPU Tier-1 Pre-Encoder.")
    ap.add_argument(
        "--corpus",
        type=str,
        required=True,
        choices=(
            "unitree_g1_h1_humanoid",
            "droid_1_0_1",
            "rh20t_contact_av_force",
            "agibot_and_aloha_sim",
            "vggsound_full_sync",
            "epic_kitchens_100_long_ego_av",
            "holoassist_and_greatesthits_av",
            "wavcaps_and_libriheavy_speech",
            "fineweb_edu_and_stack_code",
        ),
    )
    ap.add_argument("--target-hours", type=float, default=1.0)
    ap.add_argument("--target-bytes-m", type=float, default=50.0)
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = unlimited up to target-hours.")
    ap.add_argument("--split-filter", type=str, default="", choices=("", "train", "dev", "test"), help="Optional single-split recovery filter (e.g. dev top-up).")
    ap.add_argument("--out-dir", type=Path, default=Path("/content/v4_shards"))
    ap.add_argument("--hf-repo", type=str, default="auto", help="'auto' routes each corpus to its license-pure public valueai-org/slwm-v4-* repo.")
    ap.add_argument("--migrate-from-repo", type=str, default="", help="Optional legacy HF repo to copy existing shards from before resuming.")
    ap.add_argument("--flush-mb", type=float, default=650.0)
    ap.add_argument("--device", type=str, default="cuda", choices=("cuda", "mps", "cpu"))
    ap.add_argument("--keep-local", action="store_true")
    args = ap.parse_args()

    resolved_hf_repo: str | None
    if args.hf_repo.strip().lower() == "auto":
        from scripts.data.v4_valueai_registry import get_valueai_repo_for_corpus

        resolved_hf_repo = get_valueai_repo_for_corpus(args.corpus)
        print(f"[VALUEAI TARGET] corpus={args.corpus} -> public repo={resolved_hf_repo}", flush=True)
    else:
        resolved_hf_repo = args.hf_repo or None

    max_eps = None if args.max_episodes <= 0 else args.max_episodes
    need_vid = args.corpus not in ("fineweb_edu_and_stack_code", "wavcaps_and_libriheavy_speech")
    need_aud = args.corpus in (
        "rh20t_contact_av_force",
        "vggsound_full_sync",
        "epic_kitchens_100_long_ego_av",
        "holoassist_and_greatesthits_av",
        "wavcaps_and_libriheavy_speech",
    )
    codec = FrozenV4CodecEngine(device=args.device, load_video=need_vid, load_audio=need_aud) if (need_vid or need_aud) else None
    writer = StreamingV4ShardWriter(
        corpus_name=args.corpus,
        local_out_dir=args.out_dir,
        hf_repo_id=resolved_hf_repo,
        migrate_from_repo=args.migrate_from_repo.strip() or None,
        flush_mb=args.flush_mb,
        delete_local_after_upload=not args.keep_local,
    )

    t0 = time.monotonic()
    if args.corpus == "vggsound_full_sync":
        encode_vggsound(codec, writer, target_hours=args.target_hours, max_episodes=max_eps)
    elif args.corpus == "unitree_g1_h1_humanoid":
        encode_aloha_and_unitree_g1(codec, writer, corpus_name=args.corpus, target_hours=args.target_hours, max_episodes=max_eps)
    elif args.corpus == "agibot_and_aloha_sim":
        # Encode ALOHA-Sim (0.25h) first, then AgiBotWorld2026 PhysicalReasoning up to target_hours!
        encode_aloha_and_unitree_g1(codec, writer, corpus_name=args.corpus, target_hours=min(0.5, args.target_hours), max_episodes=max_eps)
        encode_agibot_world_2026(codec, writer, target_hours=args.target_hours, max_episodes=max_eps)
    elif args.corpus == "rh20t_contact_av_force":
        encode_rh20t_contact_av_force(codec, writer, target_hours=args.target_hours, max_episodes=max_eps)
    elif args.corpus == "droid_1_0_1":
        encode_droid_1_0_1(codec, writer, target_hours=args.target_hours, max_episodes=max_eps)
    elif args.corpus in ("epic_kitchens_100_long_ego_av", "holoassist_and_greatesthits_av"):
        encode_long_egocentric_av(codec, writer, corpus_name=args.corpus, target_hours=args.target_hours, max_episodes=max_eps)
    elif args.corpus == "wavcaps_and_libriheavy_speech":
        encode_libriheavy_speech(codec, writer, target_hours=args.target_hours, max_episodes=max_eps, split_filter=args.split_filter)
    elif args.corpus == "fineweb_edu_and_stack_code":
        encode_fineweb_text_code(writer, target_bytes_m=args.target_bytes_m, max_docs=max_eps)

    receipt = writer.finalize()
    dt_m = (time.monotonic() - t0) / 60.0
    print(
        f"[PREENCODE COMPLETE] corpus={args.corpus} eps={receipt['n_episodes']} "
        f"hours={receipt['total_hours']:.2f}h pkts={receipt['total_packets']:,} wall={dt_m:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
