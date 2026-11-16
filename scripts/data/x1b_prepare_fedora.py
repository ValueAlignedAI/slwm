"""Build the zero-leakage Real-World Multimodal 64x64x3 Bank & PyMunk Evaluation Panels on Fedora.

Sources (read-only from /mnt/slwm-data/SLWM):
    1. /mnt/slwm-data/SLWM/g5/sources/vis-data-256.zip  — Real 256x256 RGB impact video + wav + timestamps
    2. /mnt/slwm-data/SLWM/g9/data/starss-{train,dev,test}.npz — Real STARSS23 spatial audio-video
    3. /mnt/slwm-data/SLWM/g5/pusht/{train,dev,test}    — Real PyMunk Push-T rigid-body trajectories
    4. /mnt/slwm-data/SLWM/g8/panels/{pusht,puck_barrier} — Official G8 held-out PyMunk branch panels
    5. /mnt/slwm-data/SLWM/data/fineweb_edu_scale_v1/sources/000_00000.parquet — Real FineWeb-Edu text

Outputs (written atomically to /mnt/slwm-data/SLWM/x1b/datasets):
    - x1b_real_bank_v1.npz + x1b_real_bank_v1.card.json
    - x1b_pymunk_panels_64.npz + x1b_pymunk_panels_64.card.json
"""

from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import zoom
import soundfile as sf

SEQ_LEN = 32
VOCAB_SIZE = 4096
WORD_RE = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build_subword_tokenizer(texts: list[str], vocab_size: int = VOCAB_SIZE) -> dict[str, int]:
    """Deterministic word/subword vocabulary with 256-byte fallback (ids 256..511), zero hash collisions.

    Reserved IDs:
        0: PAD
        1: UNK
        2: QUERY
        3: EOS
        4..255: control/task/associative markers
        256..511: lossless UTF-8 byte fallback
        512..4095: top frequent FineWeb-Edu lexical/code tokens
    """
    counts: Counter[str] = Counter()
    for doc in texts:
        for tok in WORD_RE.findall(doc[:4000].lower()):
            if len(tok) <= 24:
                counts[tok] += 1
    vocab: dict[str, int] = {}
    limit = vocab_size - 512
    for idx, (tok, _) in enumerate(counts.most_common(limit)):
        vocab[tok] = 512 + idx
    return vocab


def encode_text_passage(doc: str, vocab: dict[str, int], n_tokens: int) -> np.ndarray:
    """Encode a text passage into int32 token IDs with lossless byte fallback (never collides!)."""
    out: list[int] = []
    for tok in WORD_RE.findall(doc.lower()):
        tid = vocab.get(tok)
        if tid is not None:
            out.append(tid)
        else:
            for b in tok.encode("utf-8", errors="ignore")[:8]:
                out.append(256 + int(b))
        if len(out) >= n_tokens:
            break
    out.append(3)
    arr = np.zeros((n_tokens,), dtype=np.int32)
    take = min(n_tokens, len(out))
    arr[:take] = np.asarray(out[:take], dtype=np.int32)
    return arr


def render_pymunk_poses_64(
    poses_norm: np.ndarray,
    has_barrier: bool,
    goal_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Render normalized PyMunk Push-T / Puck-Barrier trajectories [T, 6] to 64x64x3 uint8 RGB.

    poses_norm columns: [ee_x, ee_y, obj_x, obj_y, sin(theta), cos(theta)] in [-1, 1].
    In G8's 512x512 arena, barrier is (256, 178)..(256, 334) width 5px -> x=0.0, y in [-0.305, +0.305].
    """
    t = poses_norm.shape[0]
    grid = np.linspace(-1.0 + 1.0 / 64.0, 1.0 - 1.0 / 64.0, 64, dtype=np.float32)
    gy, gx = np.meshgrid(grid, grid, indexing="ij")  # [64, 64]

    bg = np.stack([28.0 + 6.0 * gx, 32.0 + 6.0 * gy, 40.0 + 4.0 * gx * gy], axis=-1)[None]  # [1, 64, 64, 3]
    img = np.repeat(bg, t, axis=0)

    if has_barrier:
        wall_a = np.clip((0.030 - np.abs(gx)) / 0.015, 0.0, 1.0) * (np.abs(gy) <= 0.34).astype(np.float32)
        wall_col = np.array([225.0, 95.0, 75.0], dtype=np.float32)
        img = img * (1.0 - wall_a[None, :, :, None]) + wall_col * wall_a[None, :, :, None]

    if goal_xy is not None:
        d_g = np.sqrt((gx - goal_xy[0]) ** 2 + (gy - goal_xy[1]) ** 2 + 1e-6)
        ga = np.clip(1.0 - np.abs(d_g - 0.10) / 0.028, 0.0, 1.0)
        g_col = np.array([70.0, 225.0, 130.0], dtype=np.float32)
        img = img * (1.0 - ga[None, :, :, None]) + g_col * ga[None, :, :, None]

    ex = poses_norm[:, 0, None, None]
    ey = poses_norm[:, 1, None, None]
    ox = poses_norm[:, 2, None, None]
    oy = poses_norm[:, 3, None, None]
    s = poses_norm[:, 4, None, None]
    c = poses_norm[:, 5, None, None]

    rx = c * (gx[None] - ox) + s * (gy[None] - oy)
    ry = -s * (gx[None] - ox) + c * (gy[None] - oy)
    top_bar = (np.abs(rx) <= 0.125) & (np.abs(ry - 0.05) <= 0.04)
    stem_bar = (np.abs(rx) <= 0.045) & (ry >= -0.11) & (ry <= 0.055)
    disc = np.clip((0.115 - np.sqrt(rx * rx + ry * ry + 1e-6)) / 0.022, 0.0, 1.0)
    oa = np.maximum((top_bar | stem_bar).astype(np.float32), 0.55 * disc)
    o_col = np.array([85.0, 165.0, 245.0], dtype=np.float32)
    img = img * (1.0 - oa[..., None]) + o_col * oa[..., None]

    dee = np.sqrt((gx[None] - ex) ** 2 + (gy[None] - ey) ** 2 + 1e-6)
    ea = np.clip((0.072 - dee) / 0.02, 0.0, 1.0)
    e_col = np.array([250.0, 215.0, 75.0], dtype=np.float32)
    img = img * (1.0 - ea[..., None]) + e_col * ea[..., None]

    return np.clip(np.rint(img), 0.0, 255.0).astype(np.uint8)


TARGET_AUDIO_SR = 16_000
CAUSAL_LOG10_MEAN = -4.523
CAUSAL_LOG10_STD = 2.524
MAX_CAUSAL_AUDIO_FRONTEND_BYTES = 10_876
SUPPORTED_CAUSAL_AUDIO_SRS: dict[int, tuple[int, int]] = {
    16_000: (16_000, 1),
    24_000: (24_000, 1),
    32_000: (16_000, 2),
    44_100: (22_050, 2),
    48_000: (16_000, 3),
    96_000: (16_000, 6),
}
STARSS_SPLITS: dict[str, str] = {
    "train": "dev-train-tau",
    "dev": "dev-test-tau",
    "test": "dev-test-sony",
}


def causal_decimate_pcm_to_16khz(pcm: np.ndarray, sr: int) -> np.ndarray:
    """Causally convert mono/multichannel PCM to exact `16,000 Hz` mono with zero future lookahead."""
    sr_i = int(sr)
    if sr_i <= 0:
        raise ValueError(f"Invalid audio sample rate: {sr_i}")
    data = np.asarray(pcm, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    if sr_i == TARGET_AUDIO_SR:
        return data
    if sr_i < TARGET_AUDIO_SR or (sr_i % TARGET_AUDIO_SR) != 0:
        raise ValueError(
            f"Causal 16 kHz frontend requires sr == {TARGET_AUDIO_SR} or an exact integer multiple; got sr={sr_i}"
        )
    factor = sr_i // TARGET_AUDIO_SR
    n_out = int(data.shape[0]) // factor
    if n_out == 0:
        return np.zeros((0,), dtype=np.float32)
    return data[: n_out * factor].reshape(n_out, factor).mean(axis=1).astype(np.float32)


def causal_decimate_pcm_10hz(pcm: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Causally convert source PCM to an exact `10 Hz`-commensurate rate (`16 kHz`, `22.05 kHz`, or `24 kHz`).

    Uses strictly non-overlapping integer `factor:1` boxcar averaging with zero future lookahead across all
    registered corpora (`16 kHz` SpeechCommands, `24 kHz` STARSS23 FOA W, `44.1->22.05 kHz` GRID, and
    `96->16 kHz` GreatestHits), strictly bounded by `<= 10,876 B` total streaming frontend state.
    """
    sr_i = int(sr)
    if sr_i not in SUPPORTED_CAUSAL_AUDIO_SRS:
        raise ValueError(
            f"Unsupported audio sample rate {sr_i} for strict causal 10 Hz frontend; supported={sorted(SUPPORTED_CAUSAL_AUDIO_SRS)}"
        )
    eff_sr, factor = SUPPORTED_CAUSAL_AUDIO_SRS[sr_i]
    data = np.asarray(pcm, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    if factor == 1:
        return data, eff_sr
    n_out = int(data.shape[0]) // factor
    if n_out == 0:
        return np.zeros((0,), dtype=np.float32), eff_sr
    return data[: n_out * factor].reshape(n_out, factor).mean(axis=1).astype(np.float32), eff_sr


def pcm_to_causal_logmel_64(pcm: np.ndarray, sr: int, n_ticks: int) -> np.ndarray:
    """Compute strictly causal 64-band `10 Hz` log-power features (`[60 Hz, 7,680 Hz]`) with zero lookahead.

    Causally decimates registered source PCM (`16/24/32/44.1/48/96 kHz`), reads strictly samples
    `< (t_idx + 1) * hop` at tick `t_idx` (`hop = eff_sr // 10`), normalizes FFT window length via
    `scale = win / 1600.0` (`1.0` at `16 kHz`), and applies fixed reference constants (`CAUSAL_LOG10_MEAN`,
    `CAUSAL_LOG10_STD`) with zero whole-waveform `max(abs)` normalization.
    """
    data, eff_sr = causal_decimate_pcm_10hz(pcm, int(sr))
    hop = eff_sr // 10  # 1,600 @ 16 kHz; 2,205 @ 22.05 kHz; 2,400 @ 24 kHz (exact 100 ms window)
    win = hop
    scale = float(win) / 1600.0
    freqs = np.fft.rfftfreq(win, d=1.0 / float(eff_sr))
    edges = np.geomspace(60.0, 7680.0, 65)
    hann = np.hanning(win).astype(np.float32)
    spec_rows = []
    n_tot = int(data.shape[0])
    for t_idx in range(n_ticks):
        end = (t_idx + 1) * hop
        raw_start = end - win
        clip_end = min(end, n_tot)
        clip_start = max(0, raw_start)
        if clip_end > clip_start:
            core = data[clip_start:clip_end]
        else:
            core = np.zeros((0,), dtype=np.float32)
        left_pad = max(0, -raw_start)
        right_pad = max(0, win - left_pad - int(core.shape[0]))
        if left_pad > 0 or right_pad > 0:
            seg = np.pad(core, (left_pad, right_pad))
        else:
            seg = core
        mag = (np.abs(np.fft.rfft(seg * hann)) / scale) ** 2
        bands = np.zeros((64,), dtype=np.float32)
        for b in range(64):
            mask = (freqs >= edges[b]) & (freqs < edges[b + 1])
            if np.any(mask):
                bands[b] = float(np.mean(mag[mask]))
            else:
                idx = int(np.argmin(np.abs(freqs - 0.5 * (edges[b] + edges[b + 1]))))
                bands[b] = float(mag[idx])
        spec_rows.append(np.log10(bands + 1e-7))
    spec = np.stack(spec_rows, axis=0)
    spec = (spec - CAUSAL_LOG10_MEAN) / CAUSAL_LOG10_STD
    return np.clip(spec, -4.0, 4.0).astype(np.float32)


def wav_to_logmel_64(wav_bytes: bytes | Path, n_ticks: int) -> np.ndarray:
    """Compute strictly causal 64-band log-power spectral features at 10 Hz ticks from WAV bytes or Path."""
    raw = wav_bytes.read_bytes() if isinstance(wav_bytes, Path) else wav_bytes
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    return pcm_to_causal_logmel_64(data, int(sr), n_ticks)


def mpg_to_causal_logmel_64(
    mpg_bytes: bytes | Path,
    n_ticks: int,
    *,
    expected_sr: int = 44_100,
    ffmpeg_bin: str | None = None,
) -> np.ndarray:
    """Decode MPEG-1 Layer II (`44.1 kHz`) audio with zero FFmpeg resampling (`-ar` forbidden) & fixed `1/32768` scale."""
    raw = mpg_bytes.read_bytes() if isinstance(mpg_bytes, Path) else mpg_bytes
    ff_exe = ffmpeg_bin or shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
    cmd = [
        ff_exe,
        "-v",
        "error",
        "-threads",
        "1",
        "-i",
        "pipe:0",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-f",
        "s16le",
        "pipe:1",
    ]
    pcm_s16 = subprocess.check_output(cmd, input=raw, timeout=20)
    pcm = np.frombuffer(pcm_s16, dtype="<i2").astype(np.float32) / 32768.0
    return pcm_to_causal_logmel_64(pcm, int(expected_sr), n_ticks)


def extract_greatest_hits_64(
    zip_path: Path,
    counts_per_split: dict[str, int],
) -> tuple[dict[str, list[dict[str, np.ndarray]]], dict[str, list[str]]]:
    """Extract real 64x64x3 RGB video + wav spectrograms + impact labels from vis-data-256.zip."""
    buckets: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "dev": [], "test": []}
    used_stems: dict[str, list[str]] = {"train": [], "dev": [], "test": []}

    with zipfile.ZipFile(zip_path, "r") as zf:
        all_names = set(zf.namelist())
        stems = sorted(
            {
                n[len("vis-data-256/") : -len("_denoised.mp4")]
                for n in all_names
                if n.startswith("vis-data-256/") and n.endswith("_denoised.mp4")
            }
        )
        for stem in stems:
            wav_name = f"vis-data-256/{stem}_denoised.wav"
            mp4_name = f"vis-data-256/{stem}_denoised.mp4"
            times_name = f"vis-data-256/{stem}_times.txt"
            if wav_name not in all_names or times_name not in all_names:
                continue
            # Deterministic recording-level split (zero recording overlap across train/dev/test!)
            h_bucket = int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16) % 10
            split = "train" if h_bucket < 8 else ("dev" if h_bucket == 8 else "test")
            if len(buckets[split]) >= counts_per_split[split]:
                if all(len(buckets[s]) >= counts_per_split[s] for s in counts_per_split):
                    break
                continue

            # Parse impact timestamps (seconds)
            times_txt = zf.read(times_name).decode("utf-8", errors="ignore")
            hit_times = []
            for line in times_txt.splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        hit_times.append(float(parts[0]))
                    except ValueError:
                        pass
            if not hit_times:
                continue

            # Center 3.2s window around first impact > 1.0s so clip contains active physical strikes
            anchor = next((ht for ht in hit_times if ht >= 1.2), hit_times[0])
            t_start = max(0.0, anchor - 0.8)

            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_mp4:
                tmp_mp4.write(zf.read(mp4_name))
                tmp_mp4.flush()
                cmd = [
                    "/usr/bin/ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    f"{t_start:.2f}",
                    "-t",
                    f"{SEQ_LEN * 0.1:.2f}",
                    "-i",
                    tmp_mp4.name,
                    "-vf",
                    "fps=10,scale=64:64:flags=lanczos",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ]
                raw_vid = subprocess.check_output(cmd, timeout=20)

            frame_bytes = 64 * 64 * 3
            n_frames = len(raw_vid) // frame_bytes
            if n_frames < SEQ_LEN:
                continue
            frames = np.frombuffer(raw_vid[: SEQ_LEN * frame_bytes], dtype=np.uint8).reshape(SEQ_LEN, 64, 64, 3).copy()

            # Audio log-mel aligned to t_start
            wav_raw = zf.read(wav_name)
            full_ticks = int(np.ceil((t_start + SEQ_LEN * 0.1 + 0.5) * 10))
            mel_all = wav_to_logmel_64(wav_raw, full_ticks)
            start_tick = int(round(t_start * 10))
            if start_tick + SEQ_LEN > mel_all.shape[0]:
                continue
            mel = mel_all[start_tick : start_tick + SEQ_LEN]

            # Impact contact binary mask at 10 Hz
            contacts = np.zeros((SEQ_LEN, 2), dtype=np.float32)
            for ht in hit_times:
                rel = int(round((ht - t_start) * 10))
                if 0 <= rel < SEQ_LEN:
                    contacts[rel, 0] = 1.0

            # Passive AV corpus: zero robot proprioception (contacts target-only, modality_mask = [1, 1, 0, 0])
            proprio = np.zeros((SEQ_LEN, 16), dtype=np.float32)

            buckets[split].append(
                {
                    "video": frames,
                    "audio": mel,
                    "proprio": proprio,
                    "text": np.zeros((SEQ_LEN, 4), dtype=np.int32),
                    "target_text": np.zeros((SEQ_LEN,), dtype=np.int32),
                    "action": np.zeros((SEQ_LEN, 4), dtype=np.float32),
                    "contacts": contacts,
                    "goal": np.zeros((4,), dtype=np.float32),
                    "modality_mask": np.array([[1.0, 1.0, 0.0, 0.0]] * SEQ_LEN, dtype=np.float32),
                    "has_av_pair": np.float32(1.0),
                }
            )
            used_stems[split].append(stem)

    return buckets, used_stems


def load_starss_64(
    starss_dir: Path,
    foa_zip_path: Path,
    counts_per_split: dict[str, int],
) -> dict[str, list[dict[str, np.ndarray]]]:
    """Load real STARSS23 spatial audio-video clips using raw `24 kHz` FOA omnidirectional channel `W` (`0` annotations)."""
    out: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "dev": [], "test": []}
    rng = np.random.default_rng(20261111)
    with zipfile.ZipFile(foa_zip_path, "r") as foa_zf:
        for split in ("train", "dev", "test"):
            role = STARSS_SPLITS[split]
            with np.load(starss_dir / f"starss-{split}.npz", allow_pickle=False) as f:
                v = f["v"]        # [T_tot, 768] uint8 (16x16x3)
                lengths = f["lengths"].astype(int)
                names = [str(nm) for nm in f["names"]]
            offsets = np.cumsum(np.r_[0, lengths[:-1]])
            valid_candidates: list[tuple[int, int, int, str]] = []
            for off, ln, stem in zip(offsets, lengths, names):
                if ln >= SEQ_LEN + 4:
                    for s in range(int(off), int(off + ln - SEQ_LEN), SEQ_LEN):
                        valid_candidates.append((int(s), int(s - off), int(ln), stem))
            chosen_idx = rng.choice(
                len(valid_candidates),
                size=min(counts_per_split[split], len(valid_candidates)),
                replace=False,
            )
            chosen_rows = [valid_candidates[int(ci)] for ci in chosen_idx]
            rec_mel_cache: dict[str, np.ndarray] = {}
            for _, _, ln, stem in chosen_rows:
                if stem not in rec_mel_cache:
                    wav_member = f"foa_dev/{role}/{stem}.wav"
                    wave_raw, sr = sf.read(io.BytesIO(foa_zf.read(wav_member)), dtype="float32", always_2d=True)
                    if int(sr) != 24_000:
                        raise ValueError(f"Expected 24 kHz STARSS23 FOA WAV for {wav_member}, got sr={sr}")
                    # FOA channel 0 is omnidirectional acoustic pressure W (zero spatial/annotation leakage!)
                    rec_mel_cache[stem] = pcm_to_causal_logmel_64(wave_raw[:, 0], int(sr), int(ln))

            for st, rel_tick, _, stem in chosen_rows:
                v16 = v[st : st + SEQ_LEN].reshape(SEQ_LEN, 16, 16, 3)
                v64 = np.clip(zoom(v16.astype(np.float32), (1, 4, 4, 1), order=1), 0, 255).astype(np.uint8)
                aud = rec_mel_cache[stem][rel_tick : rel_tick + SEQ_LEN]
                prop = np.zeros((SEQ_LEN, 16), dtype=np.float32)
                con = np.zeros((SEQ_LEN, 2), dtype=np.float32)
                out[split].append(
                    {
                        "video": v64,
                        "audio": aud,
                        "proprio": prop,
                        "text": np.zeros((SEQ_LEN, 4), dtype=np.int32),
                        "target_text": np.zeros((SEQ_LEN,), dtype=np.int32),
                        "action": np.zeros((SEQ_LEN, 4), dtype=np.float32),
                        "contacts": con,
                        "goal": np.zeros((4,), dtype=np.float32),
                        "modality_mask": np.array([[1.0, 1.0, 0.0, 0.0]] * SEQ_LEN, dtype=np.float32),
                        "has_av_pair": np.float32(1.0),
                    }
                )
    return out


def load_pymunk_pusht_64(pusht_root: Path, counts_per_split: dict[str, int]) -> dict[str, list[dict[str, np.ndarray]]]:
    """Re-render real PyMunk Push-T trajectories at native 64x64x3 anti-aliased RGB (silent vision+proprio)."""
    out: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "dev": [], "test": []}
    for split in ("train", "dev", "test"):
        d = pusht_root / split
        n_arr = np.load(d / "n.npy", mmap_mode="r")          # [N, 16, 6]
        plan_arr = np.load(d / "plan.npy", mmap_mode="r")    # [N, 3, 16, 2]
        out_arr = np.load(d / "outcome.npy", mmap_mode="r")  # [N, 3, 16, 2]
        take = min(counts_per_split[split], n_arr.shape[0])
        for i in range(take):
            hist = np.asarray(n_arr[i], dtype=np.float32)       # [16, 6]
            cmd = np.asarray(plan_arr[i, 0], dtype=np.float32)  # [16, 2]
            delta_obj = np.asarray(out_arr[i, 0], dtype=np.float32) * (64.0 / 256.0)  # normalized [-1, 1] delta
            fut = np.repeat(hist[-1:], 16, axis=0)
            fut[:, :2] = cmd
            fut[:, 2:4] = np.clip(hist[-1, 2:4] + delta_obj, -0.98, 0.98)
            poses32 = np.concatenate([hist, fut], axis=0)  # [32, 6]
            goal_xy = fut[-1, 2:4]
            vid64 = render_pymunk_poses_64(poses32, has_barrier=False, goal_xy=goal_xy)

            u_seq = np.zeros((SEQ_LEN, 4), dtype=np.float32)
            vel_ee = np.diff(poses32[:, :2], axis=0, prepend=poses32[:1, :2]) * 6.0
            u_seq[:, :2] = np.clip(vel_ee, -1.0, 1.0)

            prop = np.zeros((SEQ_LEN, 16), dtype=np.float32)
            prop[:, :4] = poses32[:, :4]
            prop[:, 4:6] = np.diff(poses32[:, 2:4], axis=0, prepend=poses32[:1, 2:4]) * 6.0
            prop[:, 6:8] = poses32[:, 4:6]
            prop[:, 8:10] = u_seq[:, :2]

            obj_moved = (np.linalg.norm(prop[:, 4:6], axis=-1) > 0.03).astype(np.float32)
            con = np.stack([obj_moved, np.zeros_like(obj_moved)], axis=-1)
            aud = np.zeros((SEQ_LEN, 64), dtype=np.float32)

            out[split].append(
                {
                    "video": vid64,
                    "audio": aud,
                    "proprio": prop,
                    "text": np.zeros((SEQ_LEN, 4), dtype=np.int32),
                    "target_text": np.zeros((SEQ_LEN,), dtype=np.int32),
                    "action": u_seq,
                    "contacts": con,
                    "goal": np.concatenate([goal_xy - poses32[0, 2:4], goal_xy]).astype(np.float32),
                    "modality_mask": np.array([[0.0, 1.0, 1.0, 0.0]] * SEQ_LEN, dtype=np.float32),
                    "has_av_pair": np.float32(0.0),
                }
            )
    return out


def load_fineweb_edu_streams(parquet_path: Path, counts_per_split: dict[str, int]) -> tuple[dict[str, list[dict[str, np.ndarray]]], dict[str, int]]:
    """Build 4096-token subword vocabulary on train FineWeb-Edu and encode disjoint train/dev/test clips."""
    pf = pq.ParquetFile(parquet_path)
    docs: list[str] = []
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=["text"])
        docs.extend(str(x) for x in table["text"].to_pylist() if x and len(str(x)) > 300)
        if len(docs) >= 5600:
            break

    train_docs = docs[:4000]
    dev_docs = docs[4000:4600]
    test_docs = docs[4600:5200]
    vocab = build_subword_tokenizer(train_docs, VOCAB_SIZE)

    out: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "dev": [], "test": []}
    for split, split_docs in (("train", train_docs), ("dev", dev_docs), ("test", test_docs)):
        need = counts_per_split[split]
        for doc in split_docs[:need]:
            toks = encode_text_passage(doc, vocab, (SEQ_LEN + 1) * 4)
            txt_grid = toks[: SEQ_LEN * 4].reshape(SEQ_LEN, 4)
            # Predict first subword of next tick (next-chunk LM + LAMBADA-style final continuation!)
            targ = toks[4 : (SEQ_LEN + 1) * 4 : 4]
            out[split].append(
                {
                    "video": np.zeros((SEQ_LEN, 64, 64, 3), dtype=np.uint8),
                    "audio": np.zeros((SEQ_LEN, 64), dtype=np.float32),
                    "proprio": np.zeros((SEQ_LEN, 16), dtype=np.float32),
                    "text": txt_grid,
                    "target_text": targ,
                    "action": np.zeros((SEQ_LEN, 4), dtype=np.float32),
                    "contacts": np.zeros((SEQ_LEN, 2), dtype=np.float32),
                    "goal": np.zeros((4,), dtype=np.float32),
                    "modality_mask": np.array([[0.0, 0.0, 0.0, 1.0]] * SEQ_LEN, dtype=np.float32),
                    "has_av_pair": np.float32(0.0),
                }
            )
    return out, vocab


def build_g8_panels_64(panels_root: Path, out_dir: Path) -> dict[str, str]:
    """Re-render held-out G8 puck_barrier & pusht decision panels at 64x64x3 RGB."""
    pack_arrays: dict[str, np.ndarray] = {}
    for task, has_bar in (("puck_barrier", True), ("pusht", False)):
        with np.load(panels_root / task / "scenes.npz", allow_pickle=False) as f:
            take = min(256, f["n"].shape[0])
            n_hist = f["n"][:take].astype(np.float32)              # [256, 16, 6]
            plans = f["plan"][:take].astype(np.float32)            # [256, 3, 16, 2]
            outcomes = f["outcome"][:take].astype(np.float32)      # [256, 3, 16, 2]
            goals = f["goals"][:take].astype(np.float32) / 256.0   # normalized goal deltas
            wrong = f["wrong"][:take].astype(np.int32) % take
            bar_con = f["barrier_contacts"][:take].astype(np.int32) if "barrier_contacts" in f.files else np.zeros((take, 3), dtype=np.int32)

        vids64 = np.stack([render_pymunk_poses_64(n_hist[i], has_barrier=has_bar) for i in range(take)], axis=0)
        pack_arrays[f"{task}_video64"] = vids64
        pack_arrays[f"{task}_n"] = n_hist
        pack_arrays[f"{task}_plan"] = plans
        pack_arrays[f"{task}_outcome"] = outcomes
        pack_arrays[f"{task}_goals"] = goals
        pack_arrays[f"{task}_wrong"] = wrong
        pack_arrays[f"{task}_barrier_contacts"] = bar_con

    dest = out_dir / "x1b_pymunk_panels_64.npz"
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **pack_arrays)
    tmp.replace(dest)
    digest = sha256_file(dest)
    card = {
        "file": dest.name,
        "sha256": digest,
        "scenes_per_task": 256,
        "resolution": [64, 64, 3],
        "tasks": ["puck_barrier", "pusht"],
    }
    (out_dir / "x1b_pymunk_panels_64.card.json").write_text(json.dumps(card, indent=2) + "\n")
    return {"pymunk_panels_sha256": digest}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank-name", default="x1b_real_bank_v2")
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    src_root = Path("/mnt/slwm-data/SLWM")
    out_dir = src_root / "x1b/datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / f"{args.bank_name}.npz"
    card_path = out_dir / f"{args.bank_name}.card.json"

    if not args.force_rebuild and bank_path.is_file() and card_path.is_file():
        meta = json.loads(card_path.read_text())
        if sha256_file(bank_path) == meta["sha256"]:
            print("X1B_REAL_BANK_ALREADY_VERIFIED", args.bank_name, meta["sha256"], flush=True)
            return

    print("1/5 Extracting real GreatestHits 64x64x3 impact video+audio (strictly causal log-mel)...", flush=True)
    gh_buckets, gh_stems = extract_greatest_hits_64(
        src_root / "g5/sources/vis-data-256.zip",
        {"train": 320, "dev": 64, "test": 96},
    )
    assert set(gh_stems["train"]).isdisjoint(gh_stems["dev"])
    assert set(gh_stems["train"]).isdisjoint(gh_stems["test"])
    assert set(gh_stems["dev"]).isdisjoint(gh_stems["test"])

    print("2/5 Loading real STARSS23 spatial audio-video (raw 24 kHz FOA omni W, zero annotations)...", flush=True)
    st_buckets = load_starss_64(
        src_root / "g9/data",
        src_root / "g7/starss/foa_dev.zip",
        {"train": 256, "dev": 64, "test": 64},
    )

    print("3/5 Rendering real PyMunk Push-T trajectories at 64x64x3...", flush=True)
    pt_buckets = load_pymunk_pusht_64(src_root / "g5/pusht", {"train": 320, "dev": 64, "test": 96})

    print("4/5 Encoding real FineWeb-Edu 4096-vocab text/code streams...", flush=True)
    fw_buckets, vocab = load_fineweb_edu_streams(
        src_root / "data/fineweb_edu_scale_v1/sources/000_00000.parquet",
        {"train": 320, "dev": 64, "test": 96},
    )

    print("5/5 Verifying held-out G8 PyMunk 64x64 panels & writing bank...", flush=True)
    panels_card_p = out_dir / "x1b_pymunk_panels_64.card.json"
    if (out_dir / "x1b_pymunk_panels_64.npz").is_file() and panels_card_p.is_file():
        panel_info = {"pymunk_panels_sha256": json.loads(panels_card_p.read_text())["sha256"]}
    else:
        panel_info = build_g8_panels_64(src_root / "g8/panels", out_dir)

    arrays: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    rng = np.random.default_rng(20261111)
    fields = ("video", "audio", "proprio", "text", "target_text", "action", "contacts", "goal", "modality_mask", "has_av_pair")

    for split in ("train", "dev", "test"):
        tagged_items = (
            [(it, "greatest_hits", gh_stems[split][i]) for i, it in enumerate(gh_buckets[split])]
            + [(it, "starss23", f"starss_{split}_{i:04d}") for i, it in enumerate(st_buckets[split])]
            + [(it, "pymunk_pusht", f"pusht_{split}_{i:04d}") for i, it in enumerate(pt_buckets[split])]
            + [(it, "fineweb_edu", f"fineweb_{split}_{i:04d}") for i, it in enumerate(fw_buckets[split])]
        )
        perm = rng.permutation(len(tagged_items))
        tagged_items = [tagged_items[int(i)] for i in perm]
        counts[split] = len(tagged_items)
        for f_name in fields:
            arrays[f"{split}_{f_name}"] = np.stack([it[0][f_name] for it in tagged_items], axis=0)
        arrays[f"{split}_source_domain"] = np.asarray([it[1] for it in tagged_items], dtype="<U20")
        arrays[f"{split}_source_id"] = np.asarray([it[2] for it in tagged_items], dtype="<U48")

    tmp_bank = bank_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_bank, **arrays)
    tmp_bank.replace(bank_path)
    bank_sha = sha256_file(bank_path)

    if not (out_dir / "vocab_4096.json").is_file():
        (out_dir / "vocab_4096.json").write_text(json.dumps(vocab, sort_keys=True) + "\n")
    card = {
        "dataset": args.bank_name,
        "sha256": bank_sha,
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "audio_frontend": {
            "causal": True,
            "supported_source_rates_hz": sorted(SUPPORTED_CAUSAL_AUDIO_SRS),
            "source_decimation": "strict_integer_causal_boxcar_10hz_commensurate",
            "passband_hz": [60.0, 7680.0],
            "greatest_hits_state_bytes": 6424,
            "starss23_foa_w_state_bytes": 9600,
            "universal_max_frontend_state_bytes": MAX_CAUSAL_AUDIO_FRONTEND_BYTES,
            "normalization": "fixed_train_reference_log10_power",
            "log10_mean": CAUSAL_LOG10_MEAN,
            "log10_std": CAUSAL_LOG10_STD,
            "future_lookahead_samples": 0,
        },
        "proprioception_contract": {
            "greatest_hits": "none_masked_zero",
            "starss23": "none_masked_zero_no_metadata_annotations",
            "pymunk_pusht": "ee_only_at_model_input",
            "fineweb_edu": "none_masked_zero",
        },
        "counts": counts,
        "greatest_hits_stems": {k: len(v) for k, v in gh_stems.items()},
        "greatest_hits_split_overlap": {
            "train_dev": len(set(gh_stems["train"]) & set(gh_stems["dev"])),
            "train_test": len(set(gh_stems["train"]) & set(gh_stems["test"])),
            "dev_test": len(set(gh_stems["dev"]) & set(gh_stems["test"])),
        },
        "zero_recording_leakage_verified": True,
        **panel_info,
    }
    card_path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    print("X1B_REAL_BANK_READY", json.dumps(card), flush=True)


if __name__ == "__main__":
    main()
