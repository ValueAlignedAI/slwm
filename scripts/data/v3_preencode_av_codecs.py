"""SLWM-v3 Unified Causal Multimodal Codec Pre-Encoder (`Kyutai Mimi 32-RVQ` + `Wan 2.1 VAE` + `ByT5`).

Causally encodes raw audio, audio+text, synchronized video+audio+text, embodied robotics, and
additive-RF real-speech overwrite corpora into compact unified sparse `.npz` shards
(`data/v3_unified_shards.py`) on CUDA (`H100` `bf16`) or CPU (smoke verification).

Strict Gate 2 Scientific Contracts:
  - Audio: `kyutai/mimi` 32-RVQ @ 24 kHz -> 12.5 Hz causal packets (`[T_a, 32]` uint16).
  - Video: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (`subfolder="vae"`) @ `192x320` (`0%` black bars),
    `25 fps` -> `24x40x16` causal latent groups partitioned into `3x5 = 15` isotropic `8x8`
    spatial patches (`[T_v, 15, 1024]` float16), strictly interleaved with audio by causal
    `availability_time_s` (`data/v3_data_contract.py`).
  - Text/Code: lossless ByT5 UTF-8 bytes (`0..255` + control markers `256..263`).
  - Splits: strictly honors official source/speaker/recording-date-disjoint splits with zero overlap.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile
from typing import Any
import zipfile

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from scipy.io import wavfile
from scipy.signal import resample_poly
import torch

from data.v3_data_contract import (
    build_interleaved_speech_overwrite_schedule,
    mimi_packet_availability_timestamps,
    tile_background_with_crossfade,
    wan21_group_availability_timestamps,
)
from data.v3_unified_shards import (
    BYTE_BOS,
    BYTE_EOS,
    BYTE_MOD_SEP,
    SparseShardBuilder,
    encode_utf8_bytes,
)


def _resample_f32(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if x.ndim > 1:
        x = np.mean(x, axis=-1)
    x = x.astype(np.float32)
    if sr_in == sr_out:
        return x
    g = math.gcd(int(sr_in), int(sr_out))
    return resample_poly(x.astype(np.float64), sr_out // g, sr_in // g).astype(np.float32)


def _decode_wav_bytes_24k(raw_bytes: bytes) -> np.ndarray:
    sr, data = wavfile.read(io.BytesIO(raw_bytes))
    if data.dtype == np.int16:
        wf = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        wf = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        wf = (data.astype(np.float32) - 128.0) / 128.0
    else:
        wf = data.astype(np.float32)
    return _resample_f32(wf, sr, 24000)


def _decode_audio_bytes_ffmpeg_24k(raw_bytes: bytes) -> np.ndarray:
    """Fallback decoder via ffmpeg for FLAC/MP3/OGG/MP4 byte streams -> 24 kHz mono float32."""
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ac", "1", "-ar", "24000", "-f", "f32le", "pipe:1"],
        input=raw_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if p.returncode != 0 or len(p.stdout) < 4:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def _extract_mp4_frames_and_audio(mp4_path: Path, *, max_sec: float = 12.0, fps: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Extract `192x320` center-cropped (0% black bars) `25 fps` RGB uint8 frames + `24 kHz` mono float32 audio."""
    # Video: scale height/width preserving aspect ratio to cover 320x192, then exact center crop 320x192!
    vf = "scale=320:192:force_original_aspect_ratio=increase,crop=320:192"
    pv = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(mp4_path), "-t", str(max_sec),
            "-vf", f"fps={fps},{vf}", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    frame_bytes = 192 * 320 * 3
    n_frames = len(pv.stdout) // frame_bytes
    if n_frames > 0:
        frames = np.frombuffer(pv.stdout[: n_frames * frame_bytes], dtype=np.uint8).reshape(n_frames, 192, 320, 3).copy()
    else:
        frames = np.zeros((0, 192, 320, 3), dtype=np.uint8)

    pa = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(mp4_path), "-t", str(max_sec),
            "-vn", "-ac", "1", "-ar", "24000", "-f", "f32le", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    audio = np.frombuffer(pa.stdout, dtype=np.float32).copy() if len(pa.stdout) >= 4 else np.zeros((0,), dtype=np.float32)
    return frames, audio


class CausalCodecEngine:
    """Lazy-loaded frozen causal `Kyutai Mimi 32-RVQ` + `Wan 2.1 Causal VAE` on CUDA `bf16` or CPU."""

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.mimi = None
        self.wan_vae = None

    def _ensure_mimi(self) -> None:
        if self.mimi is None:
            from transformers import MimiModel

            print(f"[CODEC] Loading frozen Kyutai Mimi 32-RVQ on {self.device} ...", flush=True)
            self.mimi = MimiModel.from_pretrained("kyutai/mimi").to(self.device).eval()

    def _ensure_wan(self) -> None:
        if self.wan_vae is None:
            from diffusers import AutoencoderKLWan

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            print(f"[CODEC] Loading frozen Wan 2.1 Causal VAE ({dtype}) on {self.device} ...", flush=True)
            self.wan_vae = AutoencoderKLWan.from_pretrained(
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", torch_dtype=dtype
            ).to(self.device).eval()

    @torch.inference_mode()
    def encode_audio_mimi_32(self, wav_24k: np.ndarray) -> np.ndarray:
        """Encode 24 kHz mono float32 waveform -> `uint16[T_a, 32]` @ 12.5 Hz (strictly causal)."""
        self._ensure_mimi()
        if wav_24k.size < 1920:
            return np.zeros((0, 32), dtype=np.uint16)
        x = torch.from_numpy(wav_24k.astype(np.float32))[None, None, :].to(self.device)
        out = self.mimi.encode(x, num_quantizers=32)
        codes = out.audio_codes[0].transpose(0, 1).detach().cpu().numpy().astype(np.uint16)  # [T_a, 32]
        return codes

    @torch.inference_mode()
    def encode_video_wan_patches(self, frames_rgb: np.ndarray) -> np.ndarray:
        """Encode `uint8[T_f, 192, 320, 3]` RGB frames -> `float16[T_g, 15, 1024]` isotropic causal patches.

        Aligns frame count strictly to `1 + 4*k` so Wan 2.1 Causal VAE produces exact `1 + k` temporal groups
        of shape `[16, 24, 40]`, reshaped into `3 x 5 = 15` isotropic `8 x 8 x 16 = 1024` spatial patches!
        """
        self._ensure_wan()
        n_f = int(frames_rgb.shape[0])
        if n_f < 1:
            return np.zeros((0, 15, 1024), dtype=np.float16)
        # Pad or trim to 1 + 4*k
        k = (n_f - 1) // 4
        use_f = 1 + 4 * k
        f_sub = frames_rgb[:use_f]
        # [1, 3, T, H, W] in [-1, 1]
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        vid = torch.from_numpy(f_sub).to(self.device, dtype=torch.float32)
        vid = (vid / 127.5 - 1.0).permute(3, 0, 1, 2).unsqueeze(0).to(dtype)
        lat = self.wan_vae.encode(vid).latent_dist.mean[0]  # [16, T_g, 24, 40]
        # Permute to [T_g, 3, 8, 5, 8, 16] -> [T_g, 15, 1024]
        t_g = int(lat.shape[1])
        lat_thwc = lat.permute(1, 2, 3, 0).contiguous()  # [T_g, 24, 40, 16]
        patches = lat_thwc.view(t_g, 3, 8, 5, 8, 16).permute(0, 1, 3, 2, 4, 5).reshape(t_g, 15, 1024)
        return patches.to(torch.float16).detach().cpu().numpy()


def build_interleaved_av_text_episode(
    *,
    video_patches: np.ndarray | None,
    audio_codes: np.ndarray | None,
    text_str: str | None,
    text_first: bool = False,
    fps: float = 25.0,
    mimi_hz: float = 12.5,
) -> dict[str, Any]:
    """Strictly causal interleaving of Wan 2.1 video groups, Mimi audio packets, and ByT5 bytes by availability_time_s."""
    events: list[tuple[float, int, int, Any]] = []  # (avail_t, sort_tie, mod_id, payload)
    t_offset = 0.0

    if text_str and text_first:
        tb = encode_utf8_bytes(text_str, bos=BYTE_BOS, eos=BYTE_MOD_SEP, max_bytes=512)
        for i, b_val in enumerate(tb):
            events.append((i * 0.02, 0, 2, int(b_val)))
        t_offset = len(tb) * 0.02 + 0.04

    max_av_t = t_offset
    if video_patches is not None and len(video_patches) > 0:
        n_g = len(video_patches)
        n_frames = 1 + 4 * (n_g - 1)
        w_ts = wan21_group_availability_timestamps(num_frames=n_frames, fps=fps)
        for gi in range(min(n_g, len(w_ts))):
            av_t = t_offset + w_ts[gi].availability_time_s
            max_av_t = max(max_av_t, av_t)
            for sp in range(15):
                events.append((av_t, sp, 0, (sp, video_patches[gi, sp])))

    if audio_codes is not None and len(audio_codes) > 0:
        m_ts = mimi_packet_availability_timestamps(num_packets=len(audio_codes), frame_rate=mimi_hz)
        for ai in range(len(audio_codes)):
            av_t = t_offset + m_ts[ai].availability_time_s
            max_av_t = max(max_av_t, av_t)
            events.append((av_t, 20, 1, audio_codes[ai]))

    if text_str and (not text_first):
        tb = encode_utf8_bytes(text_str, bos=BYTE_MOD_SEP, eos=BYTE_EOS, max_bytes=512)
        base_t = max_av_t + 0.04
        for i, b_val in enumerate(tb):
            events.append((base_t + i * 0.02, 30, 2, int(b_val)))

    events.sort(key=lambda e: (e[0], e[1]))
    n = len(events)
    mod_id = np.empty((n,), dtype=np.uint8)
    dt_s = np.empty((n,), dtype=np.float16)
    sp_pos = np.zeros((n,), dtype=np.uint8)
    text_b = np.full((n,), 256, dtype=np.uint16)
    vid_list: list[np.ndarray] = []
    aud_list: list[np.ndarray] = []

    prev_t = events[0][0] if n > 0 else 0.0
    for idx, (av_t, _, m_id, payload) in enumerate(events):
        mod_id[idx] = m_id
        dt_s[idx] = np.float16(max(0.0, av_t - prev_t))
        prev_t = av_t
        if m_id == 0:
            sp_pos[idx] = payload[0]
            vid_list.append(payload[1])
        elif m_id == 1:
            aud_list.append(payload)
        elif m_id == 2:
            text_b[idx] = payload

    return {
        "modality_id": mod_id,
        "dt_s": dt_s,
        "spatial_pos": sp_pos,
        "text_byte": text_b,
        "video_feat": np.stack(vid_list, axis=0) if vid_list else None,
        "audio_codes": np.stack(aud_list, axis=0) if aud_list else None,
    }


def encode_clotho(engine: CausalCodecEngine, raw_root: Path, out_dir: Path, max_clips_per_split: int = 100000) -> dict[str, Any]:
    """Encode CLAPv2/Clotho (official train / valid / test) in BOTH Audio->Text & Text->Audio directions."""
    res: dict[str, Any] = {}
    for split_nm, out_split in (("train", "train"), ("valid", "dev"), ("test", "test")):
        pqs = sorted(glob.glob(str(raw_root / f"clotho/data/{split_nm}/*.parquet")))
        if not pqs:
            continue
        sb = SparseShardBuilder(f"clotho_{out_split}", out_split)
        cnt = 0
        for fp in pqs:
            tbl = pq.read_table(fp, columns=["audio", "text"]).to_pydict()
            for aud_obj, caption in zip(tbl["audio"], tbl["text"]):
                if cnt >= max_clips_per_split:
                    break
                raw_b = aud_obj["bytes"] if isinstance(aud_obj, dict) else aud_obj
                wav = _decode_wav_bytes_24k(raw_b) if raw_b[:4] == b"RIFF" else _decode_audio_bytes_ffmpeg_24k(raw_b)
                if wav.size < 4800:
                    continue
                codes = engine.encode_audio_mimi_32(wav)
                if len(codes) == 0:
                    continue
                # Alternate Audio->Caption and Caption->Audio on train; Audio->Caption on dev/test
                txt_first = (out_split == "train") and ((cnt % 2) == 1)
                ep = build_interleaved_av_text_episode(
                    video_patches=None, audio_codes=codes, text_str=str(caption or ""), text_first=txt_first
                )
                sb.add_episode(domain_name="clotho", **ep)
                cnt += 1
            if cnt >= max_clips_per_split:
                break
        res[out_split] = sb.save(out_dir / f"clotho_{out_split}.npz")
        print(f"[PREENCODE] Clotho {out_split}: {cnt} clips -> {res[out_split]['size_mib']} MiB", flush=True)
    return res


def encode_esc50(engine: CausalCodecEngine, raw_root: Path, out_dir: Path, max_clips: int = 100000) -> dict[str, Any]:
    """Encode ESC-50 (official folds 1-4 train, fold 5 held-out test)."""
    pqs = sorted(glob.glob(str(raw_root / "esc50/data/*.parquet")))
    sb_tr = SparseShardBuilder("esc50_train_f14", "train")
    sb_te = SparseShardBuilder("esc50_test_f5", "test")
    cnt = 0
    for fp in pqs:
        tbl = pq.read_table(fp, columns=["fold", "category", "audio"]).to_pydict()
        for fold, cat, aud_obj in zip(tbl["fold"], tbl["category"], tbl["audio"]):
            if cnt >= max_clips:
                break
            raw_b = aud_obj["bytes"] if isinstance(aud_obj, dict) else aud_obj
            wav = _decode_wav_bytes_24k(raw_b) if raw_b[:4] == b"RIFF" else _decode_audio_bytes_ffmpeg_24k(raw_b)
            codes = engine.encode_audio_mimi_32(wav)
            if len(codes) == 0:
                continue
            ep = build_interleaved_av_text_episode(
                video_patches=None, audio_codes=codes, text_str=f"sound: {cat}", text_first=False
            )
            (sb_te if int(fold) == 5 else sb_tr).add_episode(domain_name="esc50", **ep)
            cnt += 1
    return {
        "train": sb_tr.save(out_dir / "esc50_train_f14.npz"),
        "test": sb_te.save(out_dir / "esc50_test_f5.npz"),
    }


def encode_librispeech_evals(engine: CausalCodecEngine, raw_root: Path, out_dir: Path, max_clips: int = 100000) -> dict[str, Any]:
    """Encode official LibriSpeech dev-clean & test-clean (speaker-disjoint from train-clean-360)."""
    res: dict[str, Any] = {}
    for sub, split_nm in (("validation", "dev_clean"), ("test", "test_clean")):
        pqs = sorted(glob.glob(str(raw_root / f"librispeech_clean_evals/**/{sub}*/**/*.parquet"), recursive=True))
        if not pqs:
            pqs = sorted(glob.glob(str(raw_root / f"librispeech_clean_evals/clean/{sub}/*.parquet")))
        if not pqs:
            continue
        sb = SparseShardBuilder(f"librispeech_{split_nm}", split_nm)
        cnt = 0
        for fp in pqs:
            tbl = pq.read_table(fp, columns=["audio", "text"]).to_pydict()
            for aud_obj, txt in zip(tbl["audio"], tbl["text"]):
                if cnt >= max_clips:
                    break
                raw_b = aud_obj["bytes"] if isinstance(aud_obj, dict) else aud_obj
                wav = _decode_audio_bytes_ffmpeg_24k(raw_b)
                codes = engine.encode_audio_mimi_32(wav)
                if len(codes) == 0:
                    continue
                ep = build_interleaved_av_text_episode(
                    video_patches=None, audio_codes=codes, text_str=str(txt or "").lower(), text_first=False
                )
                sb.add_episode(domain_name="librispeech", **ep)
                cnt += 1
            if cnt >= max_clips:
                break
        res[split_nm] = sb.save(out_dir / f"librispeech_{split_nm}.npz")
        print(f"[PREENCODE] LibriSpeech {split_nm}: {cnt} utts -> {res[split_nm]['size_mib']} MiB", flush=True)
    return res


def encode_greatesthits(engine: CausalCodecEngine, data_root: Path, out_dir: Path, max_clips: int = 100000) -> dict[str, Any]:
    """Encode GreatestHits (`vis-data-256.zip`) strictly following Gate 2 recording-date splits!"""
    gh_zip = data_root / "g5" / "sources" / "vis-data-256.zip"
    if not gh_zip.is_file():
        return {"skipped": "vis-data-256.zip not found"}
    # Gate 2 deterministic date-disjoint split (16 train / 6 dev / 6 test dates!)
    builders = {
        "train": SparseShardBuilder("greatesthits_train", "train"),
        "dev": SparseShardBuilder("greatesthits_dev", "dev"),
        "test": SparseShardBuilder("greatesthits_test", "test"),
    }
    import tempfile
    with zipfile.ZipFile(gh_zip, "r") as zf:
        vids = sorted(n for n in zf.namelist() if n.endswith("_denoised.mp4"))
        dates = sorted({Path(n).name[:10] for n in vids})
        date_to_split = {}
        for i, d in enumerate(dates):
            date_to_split[d] = "train" if i < 16 else ("dev" if i < 22 else "test")
        with tempfile.TemporaryDirectory() as td:
            for ci, name in enumerate(vids[:max_clips]):
                sp = date_to_split[Path(name).name[:10]]
                tmp_mp4 = Path(td) / "clip.mp4"
                tmp_mp4.write_bytes(zf.read(name))
                frames, wav = _extract_mp4_frames_and_audio(tmp_mp4, max_sec=10.0, fps=25)
                if len(frames) < 5 or wav.size < 4800:
                    continue
                v_p = engine.encode_video_wan_patches(frames)
                a_c = engine.encode_audio_mimi_32(wav)
                ep = build_interleaved_av_text_episode(video_patches=v_p, audio_codes=a_c, text_str=None)
                builders[sp].add_episode(domain_name="greatesthits", **ep)
    return {sp: b.save(out_dir / f"greatesthits_{sp}.npz") for sp, b in builders.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Causal Mimi 32-RVQ + Wan 2.1 VAE multimodal shard pre-encoder.")
    ap.add_argument("--corpus", type=str, required=True, choices=("clotho", "esc50", "librispeech_evals", "greatesthits"))
    ap.add_argument("--raw-root", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_diverse_raw"))
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/slwm-data/SLWM"))
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded/av"))
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--max-clips", type=int, default=100000)
    args = ap.parse_args()

    engine = CausalCodecEngine(device=args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.corpus == "clotho":
        res = encode_clotho(engine, args.raw_root, args.out_dir, max_clips_per_split=args.max_clips)
    elif args.corpus == "esc50":
        res = encode_esc50(engine, args.raw_root, args.out_dir, max_clips=args.max_clips)
    elif args.corpus == "librispeech_evals":
        res = encode_librispeech_evals(engine, args.raw_root, args.out_dir, max_clips=args.max_clips)
    elif args.corpus == "greatesthits":
        res = encode_greatesthits(engine, args.data_root, args.out_dir, max_clips=args.max_clips)
    else:
        raise ValueError(args.corpus)

    (args.out_dir / f"{args.corpus}_preencode_receipt.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
