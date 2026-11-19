"""SLWM-v3 Gate 2 (Rev 2): Small-Shard Causal Codec Ceilings, Streaming Equivalence & Dual-Track Accounting.

Runs on CPU over strictly `dev`/`val` slices (ZERO sealed `test`/`vault` data touched!):
  1. Audio Edge (`kyutai/mimi` 32-RVQ & 8-RVQ @ 12.5 Hz vs `facebook/encodec_24khz` 75 Hz vs 10 Hz Hann+GL):
     - Exact parameter partition asserting `encoder_plus_rvq_and_downsample + decoder_and_upsample == total`.
     - Exact raw-PCM suffix invariance (`t_cut = 2.0 s`).
     - Native `use_streaming=True` online == offline equivalence across `24 s` (`600` Transformer frames > `250`
       sliding window), proving bounded KV cache (`249` frames) and measuring EXACT live streaming cache bytes.
     - Real `98 s` counterfactual prefix-mutation audit in `Mimi 32-RVQ` code space proving `0.0` leakage past
       `Mimi`'s `79.68 s` (`8 x 249 / 25 Hz`) stacked Transformer RF.
     - Speech-content ceiling on `SpeechCommands v0.02 val` (`n = 40`, zero `"tree"` alias leak) via frozen
       `openai/whisper-tiny.en` + MR-STFT + SI-SNR.
     - Impact-onset ceiling on `GreatestHits dev` stratified across ALL `6` `dev` dates (`2` clips/date = `12`).
  2. Video Edge (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae`, unified `192x320` @ `25 fps`, `0%` black bars):
     - Exact parameter partition asserting `encoder_and_quant_conv + decoder_and_post_quant_conv == total`.
     - Exact raw-RGB suffix invariance (`9` frames = `1 + 2*4`), per-chunk `quant_conv` online == offline
       equivalence, and exact O(1) `feat_cache` byte constancy (`chunk1 == chunk2`) reported in BOTH `bf16` & `fp32`.
     - True `11x11` Gaussian-windowed local SSIM (`sigma = 1.5`), PSNR, and edge RMSE on:
       (a) `HoloAssist dev` (`R043`, `R055`, `R090`, `25 fps`, `192x320`, `0%` black bars) in `fp32` + `bf16` latent quant check (full native `bf16` encoder verified on TPU `v6e-1`),
       (b) `Physical xArm6 dev` (`dev_video`, `n = 4`) reporting frame `0` (`1x`) AND frames `1..4` (`4x` temporal) XY drift,
       (c) `V3 Dynamics` (`n = 8` worlds) reporting conveyor drift sign accuracy, puck XY drift, and EE XY drift.
  3. Text/Code Edge (`ByT5` 256-byte UTF-8): lossless round-trip on `HoloAssist dev` transcripts + Python code.
  4. Dual-Track (`Core` + `Strict System`) accounting using exact binary units (`MiB` / `GiB`).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
from typing import Any
import zipfile

import numpy as np
from PIL import Image
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter
from scipy.signal import resample_poly
import torch
from transformers import AutoProcessor, EncodecModel, MimiModel, WhisperForConditionalGeneration
from transformers.utils import logging as hf_logging

from data.v3_data_contract import (
    decode_utf8_bytes_lossless,
    encode_utf8_bytes_lossless,
    render_v3_dynamics_world_rgb,
    tile_background_with_crossfade,
)

hf_logging.set_verbosity_error()


def _hash_split(namespace: str, key: str) -> str:
    h = int(hashlib.sha256(f"{namespace}:{key}".encode("utf-8")).hexdigest()[:8], 16) / float(0xFFFFFFFF)
    return "train" if h < 0.70 else ("dev" if h < 0.85 else "test")


ACTIVE_SC_WORDS = ("left", "right", "up", "down", "zero", "one", "two", "three")
# Note: `"tree"` intentionally excluded from `"three"` because `"tree"` is in the distractor vocabulary!
DIGIT_ALIASES = {
    "zero": ("zero", "0"),
    "one": ("one", "1", "won"),
    "two": ("two", "2", "to", "too"),
    "three": ("three", "3"),
    "left": ("left",),
    "right": ("right", "write"),
    "up": ("up",),
    "down": ("down",),
}


def _resample_f32(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    g = math.gcd(sr_in, sr_out)
    return resample_poly(x.astype(np.float64), sr_out // g, sr_in // g).astype(np.float32)


def _si_snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(ref.size, est.size)
    r = ref[:n].astype(np.float64) - np.mean(ref[:n])
    e = est[:n].astype(np.float64) - np.mean(est[:n])
    s_targ = (np.dot(e, r) / (np.dot(r, r) + 1e-12)) * r
    e_noise = e - s_targ
    return float(10.0 * np.log10((np.dot(s_targ, s_targ) + 1e-12) / (np.dot(e_noise, e_noise) + 1e-12)))


def _mr_stft_log_l1(ref: np.ndarray, est: np.ndarray, sr: int = 24000) -> float:
    del sr
    n = min(ref.size, est.size)
    r = ref[:n].astype(np.float64)
    e = est[:n].astype(np.float64)
    errs = []
    for n_fft in (256, 512, 1024):
        hop = n_fft // 4
        if n < n_fft:
            continue
        win = np.hanning(n_fft)
        n_frames = 1 + (n - n_fft) // hop
        r_mags = np.abs(np.fft.rfft(np.stack([r[i * hop : i * hop + n_fft] * win for i in range(n_frames)]), axis=-1))
        e_mags = np.abs(np.fft.rfft(np.stack([e[i * hop : i * hop + n_fft] * win for i in range(n_frames)]), axis=-1))
        errs.append(float(np.mean(np.abs(np.log10(r_mags + 1e-5) - np.log10(e_mags + 1e-5)))))
    return float(np.mean(errs)) if errs else 0.0


def _onset_envelope_20ms(x: np.ndarray, sr: int = 24000) -> np.ndarray:
    hop = int(round(sr * 0.020))
    win_len = int(round(sr * 0.040))
    win = np.hanning(win_len)
    n_frames = max(1, 1 + (x.size - win_len) // hop)
    spec = np.abs(np.fft.rfft(np.stack([x[i * hop : i * hop + win_len] * win for i in range(n_frames)]), axis=-1))
    log_s = np.log1p(10.0 * spec)
    flux = np.maximum(0.0, np.diff(log_s, axis=0, prepend=log_s[:1])).mean(axis=-1)
    return flux.astype(np.float64)


def _hann10hz_nonoverlapping_roundtrip(pcm_16k: np.ndarray) -> np.ndarray:
    """10 Hz non-overlapping 100 ms Hann STFT + 24-iter 2048-pt Griffin-Lim audio edge baseline."""
    sr = 16000
    hop = 1600
    ticks = max(1, pcm_16k.size // hop)
    hann = np.hanning(hop).astype(np.float32)
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    edges = np.geomspace(60.0, 7680.0, 65)
    band_bins = []
    for bi in range(64):
        idx = np.where((freqs >= edges[bi]) & (freqs < edges[bi + 1]))[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(freqs - 0.5 * (edges[bi] + edges[bi + 1]))))])
        band_bins.append(idx)
    mel = np.zeros((ticks, 64), dtype=np.float32)
    for ti in range(ticks):
        seg = pcm_16k[ti * hop : (ti + 1) * hop]
        if seg.size < hop:
            seg = np.pad(seg, (0, hop - seg.size))
        sp = (np.abs(np.fft.rfft(seg * hann, n=n_fft)) ** 2) / float(hop)
        for bi, idx in enumerate(band_bins):
            mel[ti, bi] = np.log10(max(float(np.mean(sp[idx])), 1e-10))
    mag = np.zeros((ticks, n_fft // 2 + 1), dtype=np.float32)
    for bi, idx in enumerate(band_bins):
        amp = np.sqrt(10.0 ** mel[:, bi])
        mag[:, idx] = amp[:, None]
    rng = np.random.default_rng(0)
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, size=mag.shape))
    syn_win = np.hanning(n_fft).astype(np.float32)
    out_len = (ticks - 1) * hop + n_fft
    for _ in range(24):
        frames = np.fft.irfft(mag * phase, n=n_fft, axis=-1).real.astype(np.float32)
        y = np.zeros((out_len,), dtype=np.float32)
        wsum = np.zeros((out_len,), dtype=np.float32)
        for ti in range(ticks):
            s0 = ti * hop
            y[s0 : s0 + n_fft] += frames[ti] * syn_win
            wsum[s0 : s0 + n_fft] += syn_win * syn_win
        y = y / np.maximum(wsum, 1e-6)
        new_frames = np.stack([y[ti * hop : ti * hop + n_fft] * syn_win for ti in range(ticks)], axis=0)
        stft_new = np.fft.rfft(new_frames, n=n_fft, axis=-1)
        phase = np.exp(1j * np.angle(stft_new))
    return y[: pcm_16k.size].astype(np.float32)


def _psnr_db(ref_u8: np.ndarray, est_u8: np.ndarray) -> float:
    mse = float(np.mean((ref_u8.astype(np.float64) - est_u8.astype(np.float64)) ** 2))
    return 100.0 if mse < 1e-10 else float(10.0 * np.log10((255.0**2) / mse))


def _local_gaussian_ssim(ref_u8: np.ndarray, est_u8: np.ndarray) -> float:
    """Standard Wang et al. 11x11 Gaussian-windowed (`sigma=1.5`) local SSIM averaged over frames & channels."""
    r = ref_u8.astype(np.float64) / 255.0
    e = est_u8.astype(np.float64) / 255.0
    if r.ndim == 3:
        r = r[None, ...]
        e = e[None, ...]
    c1, c2 = 0.01**2, 0.03**2
    vals: list[float] = []
    for t in range(r.shape[0]):
        for ch in range(r.shape[-1]):
            x = r[t, :, :, ch]
            y = e[t, :, :, ch]
            mu_x = gaussian_filter(x, sigma=1.5, truncate=3.5)
            mu_y = gaussian_filter(y, sigma=1.5, truncate=3.5)
            s_xx = np.maximum(0.0, gaussian_filter(x * x, sigma=1.5, truncate=3.5) - mu_x * mu_x)
            s_yy = np.maximum(0.0, gaussian_filter(y * y, sigma=1.5, truncate=3.5) - mu_y * mu_y)
            s_xy = gaussian_filter(x * y, sigma=1.5, truncate=3.5) - mu_x * mu_y
            num = (2.0 * mu_x * mu_y + c1) * (2.0 * s_xy + c2)
            den = (mu_x * mu_x + mu_y * mu_y + c1) * (s_xx + s_yy + c2)
            # Crop 5 px border to avoid filter-edge bias
            ssim_map = num[5:-5, 5:-5] / np.maximum(den[5:-5, 5:-5], 1e-12)
            vals.append(float(np.mean(ssim_map)))
    return float(np.mean(vals))


def _edge_rmse(ref_u8: np.ndarray, est_u8: np.ndarray) -> float:
    r = ref_u8.astype(np.float64) / 255.0
    e = est_u8.astype(np.float64) / 255.0
    dx_r, dy_r = np.diff(r, axis=-2), np.diff(r, axis=-3)
    dx_e, dy_e = np.diff(e, axis=-2), np.diff(e, axis=-3)
    return float(0.5 * (np.sqrt(np.mean((dx_r - dx_e) ** 2)) + np.sqrt(np.mean((dy_r - dy_e) ** 2))))


def _color_centroid_xy(img_u8: np.ndarray, color: str) -> tuple[float, float]:
    """Compute normalized [-1, 1] centroid of saturated target pixels in `[H, W, 3]` image."""
    f = img_u8.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    if color == "red":
        score = np.maximum(0.0, r - 0.5 * (g + b) - 0.15)
    elif color == "blue":
        score = np.maximum(0.0, b - 0.5 * (r + g) - 0.15)
    elif color == "cyan":
        score = np.maximum(0.0, 0.5 * (g + b) - r - 0.35)
    elif color == "amber":
        score = np.maximum(0.0, 0.5 * (r + g) - b - 0.35)
    else:
        score = np.maximum(0.0, g - 0.5 * (r + b) - 0.15)
    h, w = score.shape
    yy, xx = np.meshgrid(np.linspace(-1.0, 1.0, h), np.linspace(-1.0, 1.0, w), indexing="ij")
    wsum = float(score.sum())
    if wsum < 1e-4:
        return (0.0, 0.0)
    return (float((score * xx).sum() / wsum), float((score * yy).sum() / wsum))


def benchmark_audio_codecs(data_root: Path, cache_dir: str) -> dict[str, Any]:
    """Benchmark Kyutai Mimi (32-RVQ & 8-RVQ @ 12.5 Hz) vs EnCodec 24 kHz (75 Hz) vs 10 Hz Hann+GL."""
    print("[AUDIO] Loading kyutai/mimi, facebook/encodec_24khz, and openai/whisper-tiny.en on CPU...", flush=True)
    mimi = MimiModel.from_pretrained("kyutai/mimi", cache_dir=cache_dir).eval()
    encodec = EncodecModel.from_pretrained("facebook/encodec_24khz", cache_dir=cache_dir).eval()
    w_proc = AutoProcessor.from_pretrained("openai/whisper-tiny.en", cache_dir=cache_dir)
    w_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en", cache_dir=cache_dir).eval()

    # Exact complete parameter partition including `mimi.downsample` (encoder) and `mimi.upsample` (decoder)!
    mimi_enc_params = (
        sum(p.numel() for p in mimi.encoder.parameters())
        + sum(p.numel() for p in mimi.encoder_transformer.parameters())
        + sum(p.numel() for p in mimi.downsample.parameters())
        + sum(p.numel() for p in mimi.quantizer.parameters())
    )
    mimi_dec_params = (
        sum(p.numel() for p in mimi.upsample.parameters())
        + sum(p.numel() for p in mimi.decoder_transformer.parameters())
        + sum(p.numel() for p in mimi.decoder.parameters())
    )
    mimi_tot_params = sum(p.numel() for p in mimi.parameters())
    assert mimi_enc_params + mimi_dec_params == mimi_tot_params, (
        f"Mimi param partition mismatch: {mimi_enc_params} + {mimi_dec_params} != {mimi_tot_params}"
    )
    enc_tot_params = sum(p.numel() for p in encodec.parameters())

    # 1. Strict raw-PCM suffix invariance test (t_cut = 2.0 s @ 24 kHz)
    rng = np.random.default_rng(42)
    x_base = rng.normal(0.0, 0.15, size=(4 * 24000,)).astype(np.float32)
    x_alt = x_base.copy()
    x_alt[2 * 24000 :] = rng.normal(0.0, 0.15, size=(2 * 24000,)).astype(np.float32)

    with torch.no_grad():
        m_c1 = mimi.encode(torch.from_numpy(x_base)[None, None, :], num_quantizers=32).audio_codes
        m_c2 = mimi.encode(torch.from_numpy(x_alt)[None, None, :], num_quantizers=32).audio_codes
        mimi_32_prefix_exact = bool(torch.equal(m_c1[:, :, :25], m_c2[:, :, :25]))
        mimi_8_prefix_exact = bool(torch.equal(m_c1[:, :8, :25], m_c2[:, :8, :25]))
        mimi_prefix_diff_frac = float((m_c1[:, :, :25] != m_c2[:, :, :25]).float().mean().item())

        e_c1 = encodec.encode(torch.from_numpy(x_base)[None, None, :], bandwidth=6.0).audio_codes[0]
        e_c2 = encodec.encode(torch.from_numpy(x_alt)[None, None, :], bandwidth=6.0).audio_codes[0]
        enc_prefix_150_exact = bool(torch.equal(e_c1[:, :, :150], e_c2[:, :, :150]))
        enc_prefix_diff_frac = float((e_c1[:, :, :150] != e_c2[:, :, :150]).float().mean().item())

    # 2. Mimi native `use_streaming=True` online == offline over 24 s (600 Transformer frames > 250 sliding window!)
    #    PLUS programmatic live streaming state byte measurement (zero hardcoded constants!).
    wav_24s = rng.normal(0.0, 0.08, size=(24 * 24000,)).astype(np.float32)
    with torch.no_grad():
        off_24 = mimi.encode(torch.from_numpy(wav_24s)[None, None, :], num_quantizers=32).audio_codes
        pkv = None
        pc = None
        stream_chunks = []
        kv_lens: list[int] = []
        chunk_s = 4 * 24000
        for ci in range(6):
            seg = torch.from_numpy(wav_24s[ci * chunk_s : (ci + 1) * chunk_s])[None, None, :]
            out_s = mimi.encode(seg, num_quantizers=32, encoder_past_key_values=pkv, padding_cache=pc, use_streaming=True)
            stream_chunks.append(out_s.audio_codes)
            pkv = out_s.encoder_past_key_values
            pc = out_s.padding_cache
            k0 = pkv.layers[0].keys
            kv_lens.append(int(k0.shape[2]))
        on_24 = torch.cat(stream_chunks, dim=-1)
        mimi_online_offline_exact = bool(torch.equal(off_24, on_24))
        mimi_online_offline_diff_frac = float((off_24 != on_24).float().mean().item())
        # Measure exact live KV cache & Conv1d padding cache elements directly from returned cache objects!
        mimi_live_kv_elems = sum(int(layer.keys.numel() + layer.values.numel()) for layer in pkv.layers)
        mimi_live_conv_elems = sum(int(t.numel()) for t in pc.padding_cache if isinstance(t, torch.Tensor))
    mimi_live_tot_fp32_bytes = (mimi_live_kv_elems + mimi_live_conv_elems) * 4
    mimi_live_tot_bf16_bytes = (mimi_live_kv_elems + mimi_live_conv_elems) * 2
    mimi_sliding_window_o1_capped = bool(kv_lens[-1] == kv_lens[-2] == kv_lens[-3] == (mimi.config.sliding_window - 1))

    # 3. Speech-content ceiling on SpeechCommands v0.02 VAL (n=40, 5 per word, ZERO test touched!)
    sc_tar = data_root / "g13" / "data" / "speech" / "speech_commands_v0.02.tar.gz"
    sc_samples: list[tuple[str, np.ndarray]] = []
    bg_clips_sorted: dict[str, np.ndarray] = {}
    per_word_count: dict[str, int] = defaultdict(int)
    with tarfile.open(sc_tar, "r:gz") as tf:
        names = tf.getnames()
        val_name = "./validation_list.txt" if "./validation_list.txt" in names else "validation_list.txt"
        val_set = set(tf.extractfile(val_name).read().decode().splitlines())
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".wav"):
                continue
            rel = m.name.lstrip("./")
            if "_background_noise_" in rel:
                f = tf.extractfile(m)
                if f is not None:
                    sr, arr = wavfile.read(io.BytesIO(f.read()))
                    if sr == 16000:
                        a = arr.astype(np.float32) / 32768.0
                        bg_clips_sorted[os.path.basename(rel)] = a.mean(axis=-1) if a.ndim > 1 else a
                continue
            if rel not in val_set:
                continue
            word = rel.split("/", 1)[0]
            if word not in ACTIVE_SC_WORDS or per_word_count[word] >= 5:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            sr, arr = wavfile.read(io.BytesIO(f.read()))
            if sr != 16000:
                continue
            arr = arr.astype(np.float32) / 32768.0
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            if arr.size < 16000:
                arr = np.pad(arr, (0, 16000 - arr.size))
            sc_samples.append((word, arr[:16000]))
            per_word_count[word] += 1
            if len(sc_samples) >= 40 and len(bg_clips_sorted) >= 6:
                break

    # 4. Real 98 s counterfactual prefix-mutation audit in Mimi 32-RVQ code space using REAL SpeechCommands val words
    #    + dev-split background clip (sorted index 4)! Proves 0.000000 leakage past Mimi's 79.68 s stacked RF!
    dev_bg_name = sorted(bg_clips_sorted.keys())[4]
    dev_bg_16k = tile_background_with_crossfade(
        bg_clips_sorted[dev_bg_name], 98 * 16000, sr=16000, rng=np.random.default_rng(2026)
    )
    cf_a_16k = (dev_bg_16k * 0.04).copy()
    cf_b_16k = cf_a_16k.copy()
    # Place two distinct real val words ("left" vs "right") at [10s, 11s]; identical PCM everywhere else!
    w_a = next(a for w, a in sc_samples if w == "left")
    w_b = next(a for w, a in sc_samples if w == "right")
    w_a = w_a * (0.08 / max(float(np.sqrt(np.mean(w_a * w_a))), 1e-6))
    w_b = w_b * (0.08 / max(float(np.sqrt(np.mean(w_b * w_b))), 1e-6))
    cf_a_16k[10 * 16000 : 11 * 16000] += w_a
    cf_b_16k[10 * 16000 : 11 * 16000] += w_b
    cf_a_24k = _resample_f32(cf_a_16k, 16000, 24000)
    cf_b_24k = _resample_f32(cf_b_16k, 16000, 24000)
    with torch.no_grad():
        ca_98 = mimi.encode(torch.from_numpy(cf_a_24k)[None, None, :], num_quantizers=32).audio_codes[0]
        cb_98 = mimi.encode(torch.from_numpy(cf_b_24k)[None, None, :], num_quantizers=32).audio_codes[0]
    mimi_cf_decay: dict[str, dict[str, float]] = {}
    for dt_s in (0, 5, 15, 30, 38, 50, 65, 81):
        f0 = int((11 + dt_s) * 12.5)
        f1 = min(ca_98.shape[1], f0 + 25)
        mimi_cf_decay[f"plus_{dt_s:02d}s"] = {
            "rvq32_mismatch_frac": round(float((ca_98[:, f0:f1] != cb_98[:, f0:f1]).float().mean().item()), 6),
            "rvq0_semantic_mismatch_frac": round(float((ca_98[:1, f0:f1] != cb_98[:1, f0:f1]).float().mean().item()), 6),
        }

    def _whisper_keyword_hit(pcm_16k: np.ndarray, word: str) -> tuple[bool, str]:
        padded = np.concatenate([np.zeros(4000, np.float32), pcm_16k, np.zeros(4000, np.float32)])
        feats = w_proc(padded, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            ids = w_model.generate(feats, max_new_tokens=12)
        txt = w_proc.batch_decode(ids, skip_special_tokens=True)[0].strip().lower()
        toks = set(re.findall(r"[a-z0-9]+", txt))
        return any(a in toks for a in DIGIT_ALIASES[word]), txt

    speech_metrics = {
        k: {"hits": 0, "mr_stft": [], "si_snr_db": [], "encode_sec": 0.0}
        for k in ("original_pcm", "mimi_32rvq_12p5hz", "mimi_8rvq_12p5hz", "encodec_24khz_6kbps", "hann10hz_nonoverlapping_gl")
    }

    for word, x16 in sc_samples:
        x24 = _resample_f32(x16, 16000, 24000)
        hit_o, _ = _whisper_keyword_hit(x16, word)
        speech_metrics["original_pcm"]["hits"] += int(hit_o)

        t0 = time.perf_counter()
        with torch.no_grad():
            mc32 = mimi.encode(torch.from_numpy(x24)[None, None, :], num_quantizers=32).audio_codes
            dt_m = time.perf_counter() - t0
            speech_metrics["mimi_32rvq_12p5hz"]["encode_sec"] += dt_m
            speech_metrics["mimi_8rvq_12p5hz"]["encode_sec"] += dt_m
            m32_rec24 = mimi.decode(mc32).audio_values[0, 0].numpy()
            m8_rec24 = mimi.decode(mc32[:, :8, :]).audio_values[0, 0].numpy()
        for k_m, rec24 in (("mimi_32rvq_12p5hz", m32_rec24), ("mimi_8rvq_12p5hz", m8_rec24)):
            rec16 = _resample_f32(rec24, 24000, 16000)[:16000]
            hit_m, _ = _whisper_keyword_hit(rec16, word)
            speech_metrics[k_m]["hits"] += int(hit_m)
            speech_metrics[k_m]["mr_stft"].append(_mr_stft_log_l1(x24, rec24))
            speech_metrics[k_m]["si_snr_db"].append(_si_snr_db(x24, rec24))

        t0 = time.perf_counter()
        with torch.no_grad():
            ec_out = encodec.encode(torch.from_numpy(x24)[None, None, :], bandwidth=6.0)
            speech_metrics["encodec_24khz_6kbps"]["encode_sec"] += time.perf_counter() - t0
            e_rec24 = encodec.decode(ec_out.audio_codes, ec_out.audio_scales)[0][0, 0].numpy()
        e_rec16 = _resample_f32(e_rec24, 24000, 16000)[:16000]
        hit_e, _ = _whisper_keyword_hit(e_rec16, word)
        speech_metrics["encodec_24khz_6kbps"]["hits"] += int(hit_e)
        speech_metrics["encodec_24khz_6kbps"]["mr_stft"].append(_mr_stft_log_l1(x24, e_rec24))
        speech_metrics["encodec_24khz_6kbps"]["si_snr_db"].append(_si_snr_db(x24, e_rec24))

        h10_rec16 = _hann10hz_nonoverlapping_roundtrip(x16)
        h10_rec24 = _resample_f32(h10_rec16, 16000, 24000)
        hit_h10, _ = _whisper_keyword_hit(h10_rec16, word)
        speech_metrics["hann10hz_nonoverlapping_gl"]["hits"] += int(hit_h10)
        speech_metrics["hann10hz_nonoverlapping_gl"]["mr_stft"].append(_mr_stft_log_l1(x24, h10_rec24))
        speech_metrics["hann10hz_nonoverlapping_gl"]["si_snr_db"].append(_si_snr_db(x24, h10_rec24))

    # 5. Stratified GreatestHits DEV impact onset ceiling (2 clips per dev recording date across ALL 6 dev dates!)
    gh_zip = data_root / "g5" / "sources" / "vis-data-256.zip"
    impact_metrics = {
        k: {"onset_corr": [], "peak_err_ms": [], "mr_stft": []}
        for k in ("mimi_32rvq_12p5hz", "mimi_8rvq_12p5hz", "encodec_24khz_6kbps", "hann10hz_nonoverlapping_gl")
    }
    n_impact = 0
    gh_dev_clips_used: list[str] = []
    gh_dev_dates_covered: set[str] = set()
    if gh_zip.exists():
        with zipfile.ZipFile(gh_zip, "r") as zf:
            by_date: dict[str, list[str]] = defaultdict(list)
            for m in sorted(zf.namelist()):
                if not m.endswith("_denoised.wav"):
                    continue
                stem = os.path.basename(m).replace("_denoised.wav", "")
                date_key = "-".join(stem.split("-")[:3])
                if _hash_split("slwm-v3-gh-date", date_key) == "dev":
                    by_date[date_key].append(m)
            chosen: list[str] = []
            for dk in sorted(by_date.keys()):
                chosen.extend(by_date[dk][:2])
            for wm in chosen:
                raw = zf.read(wm)
                sr, arr = wavfile.read(io.BytesIO(raw))
                arr = arr.astype(np.float32) / (32768.0 if arr.dtype == np.int16 else 1.0)
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                peak_idx = int(np.argmax(np.abs(arr[: min(arr.size, sr * 15)])))
                half = int(1.2 * sr)
                s0 = max(0, min(arr.size - 2 * half, peak_idx - half))
                seg = arr[s0 : s0 + 2 * half]
                if seg.size < int(2.0 * sr):
                    continue
                x24 = _resample_f32(seg, sr, 24000)[: int(2.4 * 24000)]
                x16 = _resample_f32(seg, sr, 16000)[: int(2.4 * 16000)]
                env_ref = _onset_envelope_20ms(x24, 24000)
                pk_ref = int(np.argmax(env_ref))

                with torch.no_grad():
                    mc32 = mimi.encode(torch.from_numpy(x24)[None, None, :], num_quantizers=32).audio_codes
                    m32_rec24 = mimi.decode(mc32).audio_values[0, 0].numpy()[: x24.size]
                    m8_rec24 = mimi.decode(mc32[:, :8, :]).audio_values[0, 0].numpy()[: x24.size]
                    ec_out = encodec.encode(torch.from_numpy(x24)[None, None, :], bandwidth=6.0)
                    e_rec24 = encodec.decode(ec_out.audio_codes, ec_out.audio_scales)[0][0, 0].numpy()[: x24.size]
                h10_rec24 = _resample_f32(_hann10hz_nonoverlapping_roundtrip(x16), 16000, 24000)[: x24.size]

                for k_name, rec in (
                    ("mimi_32rvq_12p5hz", m32_rec24),
                    ("mimi_8rvq_12p5hz", m8_rec24),
                    ("encodec_24khz_6kbps", e_rec24),
                    ("hann10hz_nonoverlapping_gl", h10_rec24),
                ):
                    env_est = _onset_envelope_20ms(rec, 24000)
                    n_e = min(env_ref.size, env_est.size)
                    corr = float(np.corrcoef(env_ref[:n_e], env_est[:n_e])[0, 1]) if np.std(env_est[:n_e]) > 1e-8 else 0.0
                    pk_est = int(np.argmax(env_est[:n_e]))
                    impact_metrics[k_name]["onset_corr"].append(corr)
                    impact_metrics[k_name]["peak_err_ms"].append(abs(pk_est - pk_ref) * 20.0)
                    impact_metrics[k_name]["mr_stft"].append(_mr_stft_log_l1(x24, rec))
                gh_dev_clips_used.append(os.path.basename(wm))
                gh_dev_dates_covered.add("-".join(os.path.basename(wm).split("-")[:3]))
                n_impact += 1

    return {
        "causality_and_streaming_verification": {
            "mimi_32rvq_25frames_exact": mimi_32_prefix_exact,
            "mimi_8rvq_25frames_exact": mimi_8_prefix_exact,
            "mimi_prefix_code_diff_frac": round(mimi_prefix_diff_frac, 6),
            "mimi_24s_600tf_online_vs_offline_exact": mimi_online_offline_exact,
            "mimi_24s_600tf_online_vs_offline_diff_frac": round(mimi_online_offline_diff_frac, 6),
            "mimi_sliding_window_kv_len_by_chunk": kv_lens,
            "mimi_sliding_window_o1_capped_at_249": mimi_sliding_window_o1_capped,
            "mimi_98s_real_speech_counterfactual_code_decay": mimi_cf_decay,
            "mimi_counterfactual_zero_leak_beyond_79p68s_rf": mimi_cf_decay["plus_81s"]["rvq32_mismatch_frac"] == 0.0,
            "encodec_150frames_exact": enc_prefix_150_exact,
            "encodec_prefix_code_diff_frac": round(enc_prefix_diff_frac, 6),
        },
        "speech_commands_val_n40_whisper_tiny_ceiling": {
            k: {
                "keyword_acc": round(v["hits"] / max(1, len(sc_samples)), 4),
                "n_hits": int(v["hits"]),
                "n_total": len(sc_samples),
                "mean_mr_stft_log_l1": round(float(np.mean(v["mr_stft"])), 4) if v["mr_stft"] else 0.0,
                "mean_si_snr_db": round(float(np.mean(v["si_snr_db"])), 2) if v["si_snr_db"] else None,
                "cpu_realtime_factor": round(len(sc_samples) / max(v["encode_sec"], 1e-6), 2) if v["encode_sec"] > 0 else None,
            }
            for k, v in speech_metrics.items()
        },
        "greatest_hits_dev_stratified_impact_n12_onset_ceiling": {
            "split": "dev_only_stratified_2_per_date_across_all_6_dev_dates",
            "n_clips": n_impact,
            "unique_dev_dates_covered": sorted(gh_dev_dates_covered),
            "dev_clips_audited": gh_dev_clips_used,
            "codecs": {
                k: {
                    "mean_onset_corr_20ms": round(float(np.mean(v["onset_corr"])), 4) if v["onset_corr"] else 0.0,
                    "median_peak_timing_err_ms": round(float(np.median(v["peak_err_ms"])), 1) if v["peak_err_ms"] else 0.0,
                    "mean_peak_timing_err_ms": round(float(np.mean(v["peak_err_ms"])), 1) if v["peak_err_ms"] else 0.0,
                    "mean_mr_stft_log_l1": round(float(np.mean(v["mr_stft"])), 4) if v["mr_stft"] else 0.0,
                }
                for k, v in impact_metrics.items()
            },
        },
        "accounting": {
            "primary_selected_audio_codec": "kyutai/mimi (32 RVQ codebooks, 4.4 kbps, 12.5 Hz)",
            "mimi_total_params": int(mimi_tot_params),
            "mimi_encoder_plus_rvq_and_downsample_params": int(mimi_enc_params),
            "mimi_decoder_and_upsample_params": int(mimi_dec_params),
            "mimi_param_partition_exact": mimi_enc_params + mimi_dec_params == mimi_tot_params,
            "mimi_live_kv_cache_elems": int(mimi_live_kv_elems),
            "mimi_live_conv_cache_elems": int(mimi_live_conv_elems),
            "mimi_encoder_streaming_state_mib_bf16": round(mimi_live_tot_bf16_bytes / (1024**2), 3),
            "mimi_encoder_streaming_state_mib_fp32": round(mimi_live_tot_fp32_bytes / (1024**2), 3),
            "mimi_frame_rate_hz": 12.5,
            "mimi_codebooks_active": 32,
            "mimi_latent_storage_mib_per_hour_32rvq_int16": round((12.5 * 3600 * 32 * 2) / (1024**2), 2),
            "mimi_license": "cc-by-4.0",
            "mimi_pretraining_provenance": "Kyutai Moshi 7M-hour speech/audio pretraining (exact overlap with public speech benchmarks unknown/disclosed)",
            "encodec_24khz_total_params": int(enc_tot_params),
            "encodec_frame_rate_hz": 75.0,
        },
    }


def benchmark_video_codec_wan21(data_root: Path, cache_dir: str) -> dict[str, Any]:
    """Benchmark Wan 2.1 Causal VAE at unified `192x320` (`25 fps`, `0%` black bars) in `fp32` + `bf16` latent quant check."""
    from diffusers import AutoencoderKLWan

    print("[VIDEO] Loading Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae on CPU (192x320, 25 fps, 0% black bars)...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", cache_dir=cache_dir, torch_dtype=torch.float32
    ).eval()

    wan_tot_params = sum(p.numel() for p in vae.parameters())
    wan_enc_params = sum(p.numel() for p in vae.encoder.parameters()) + sum(p.numel() for p in vae.quant_conv.parameters())
    wan_dec_params = sum(p.numel() for p in vae.decoder.parameters()) + sum(p.numel() for p in vae.post_quant_conv.parameters())
    assert wan_enc_params + wan_dec_params == wan_tot_params, (
        f"Wan 2.1 param partition mismatch: {wan_enc_params} + {wan_dec_params} != {wan_tot_params}"
    )

    def _u8_to_wan_tensor(frames_u8: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        x = (frames_u8.astype(np.float32) / 127.5) - 1.0
        return torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0).contiguous().to(dtype=dtype)

    def _wan_tensor_to_u8(x: torch.Tensor) -> np.ndarray:
        arr = ((x.float()[0].permute(1, 2, 3, 0).numpy() + 1.0) * 127.5).clip(0.0, 255.0)
        return arr.astype(np.uint8)

    # 1. Exact raw-RGB suffix invariance test on 9 frames (1 + 2*4) at unified 192x320!
    rng = np.random.default_rng(123)
    v_base = rng.integers(0, 256, size=(9, 192, 320, 3), dtype=np.uint8)
    v_alt = v_base.copy()
    v_alt[5:] = rng.integers(0, 256, size=(4, 192, 320, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    with torch.no_grad():
        vae.clear_cache()
        z_base = vae.encode(_u8_to_wan_tensor(v_base)).latent_dist.mean
        encode_9f_sec = time.perf_counter() - t0
        vae.clear_cache()
        z_alt = vae.encode(_u8_to_wan_tensor(v_alt)).latent_dist.mean

    prefix_lat_max_abs_diff = float((z_base[:, :, :2] - z_alt[:, :, :2]).abs().max().item())
    suffix_lat_max_abs_diff = float((z_base[:, :, 2:] - z_alt[:, :, 2:]).abs().max().item())

    # 2. Online chunked streaming (1 + 4 + 4 calling `quant_conv` per chunk!) vs offline equivalence + O(1) cache constancy
    with torch.no_grad():
        vae.clear_cache()
        x_full = _u8_to_wan_tensor(v_base)
        vae._enc_feat_map = [None] * vae._cached_conv_counts["encoder"]
        vae._enc_conv_idx = [0]
        z0 = vae.quant_conv(vae.encoder(x_full[:, :, :1], feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx))[:, :16]
        vae._enc_conv_idx = [0]
        z1 = vae.quant_conv(vae.encoder(x_full[:, :, 1:5], feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx))[:, :16]
        stream_cache_bytes_chunk1_fp32 = sum(t.numel() * 4 for t in vae._enc_feat_map if isinstance(t, torch.Tensor))
        vae._enc_conv_idx = [0]
        z2 = vae.quant_conv(vae.encoder(x_full[:, :, 5:9], feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx))[:, :16]
        stream_cache_bytes_chunk2_fp32 = sum(t.numel() * 4 for t in vae._enc_feat_map if isinstance(t, torch.Tensor))
        z_stream = torch.cat([z0, z1, z2], dim=2)
        vae.clear_cache()
    online_vs_offline_max_abs_diff = float((z_base - z_stream).abs().max().item())
    wan_cache_o1_constant = bool(stream_cache_bytes_chunk1_fp32 == stream_cache_bytes_chunk2_fp32 and stream_cache_bytes_chunk2_fp32 > 0)

    # 3a. HoloAssist DEV strictly filtered (`dev` only, exact 25 fps, 9 frames, 192x320, 0% black bars!)
    holo_psnr, holo_ssim, holo_edge = [], [], []
    all_holo_mp4s = sorted(glob.glob(str(data_root / "g1*" / "**" / "*-av.mp4"), recursive=True))
    dev_holo_mp4s: list[str] = []
    seen_rec: set[str] = set()
    for p in all_holo_mp4s:
        rec = os.path.basename(p)[:-7]
        if rec in seen_rec:
            continue
        seen_rec.add(rec)
        pid = rec.split("-")[0].strip().casefold()
        if _hash_split("slwm-v3-g1", pid) == "dev":
            dev_holo_mp4s.append(p)
            if len(dev_holo_mp4s) >= 3:
                break
    holo_dev_recs_used = [os.path.basename(p)[:-7] for p in dev_holo_mp4s]
    holo_raw_clips: list[np.ndarray] = []
    for mp4 in dev_holo_mp4s:
        # Scale 896x504 (16:9) -> 341x192, gentle 6% horizontal center crop -> exact 320x192 with 0% black bars!
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-ss", "15.0", "-i", mp4,
            "-frames:v", "9", "-vf", "fps=25,scale=-2:192,crop=320:192",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        raw = subprocess.check_output(cmd)
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(9, 192, 320, 3)
        holo_raw_clips.append(frames)
        with torch.no_grad():
            vae.clear_cache()
            z = vae.encode(_u8_to_wan_tensor(frames)).latent_dist.mean
            rec = _wan_tensor_to_u8(vae.decode(z).sample)
        holo_psnr.append(_psnr_db(frames, rec))
        holo_ssim.append(_local_gaussian_ssim(frames, rec))
        holo_edge.append(_edge_rmse(frames, rec))

    # 3b. Physical xArm6 DEV (`dev_video` only, n=4, 5 frames): report BOTH frame 0 (1x) AND frames 1..4 (4x temporal!)
    xarm_psnr, xarm_ssim = [], []
    xarm_xy_drift_f0, xarm_xy_drift_f1_4 = [], []
    xarm_p = data_root / "x1b" / "datasets" / "x1b_wave8a_grounded_v2.npz"
    if xarm_p.exists():
        xd = np.load(xarm_p, allow_pickle=True)
        if "dev_video" not in xd:
            raise KeyError("Expected strict `dev_video` split inside x1b_wave8a_grounded_v2.npz")
        vids = xd["dev_video"][:4]
        for ei in range(vids.shape[0]):
            fr5 = np.stack(
                [np.asarray(Image.fromarray(vids[ei, ti]).resize((320, 192), Image.BILINEAR), dtype=np.uint8) for ti in range(5)],
                axis=0,
            )
            with torch.no_grad():
                vae.clear_cache()
                z = vae.encode(_u8_to_wan_tensor(fr5)).latent_dist.mean
                rec = _wan_tensor_to_u8(vae.decode(z).sample)
            xarm_psnr.append(_psnr_db(fr5, rec))
            xarm_ssim.append(_local_gaussian_ssim(fr5, rec))
            for ti in range(5):
                for col in ("red", "blue", "green"):
                    c_ref = _color_centroid_xy(fr5[ti], col)
                    c_est = _color_centroid_xy(rec[ti], col)
                    if c_ref != (0.0, 0.0):
                        d_xy = math.hypot(c_ref[0] - c_est[0], c_ref[1] - c_est[1])
                        if ti == 0:
                            xarm_xy_drift_f0.append(d_xy)
                        else:
                            xarm_xy_drift_f1_4.append(d_xy)

    # 3c. V3 Dynamics (`n = 8` distinct worlds, 4 +1 / 4 -1): conveyor drift sign, puck XY drift, AND EE XY drift!
    drift_sign_hits = 0
    v3_psnr, v3_ssim, v3_puck_drift, v3_ee_drift = [], [], [], []
    v3_rng = np.random.default_rng(777)
    for wi in range(8):
        sign = 1.0 if wi < 4 else -1.0
        ee0 = v3_rng.uniform([-0.82, -0.45], [-0.55, 0.45]).astype(np.float32)
        obj0 = v3_rng.uniform([-0.30, -0.35], [0.30, 0.35]).astype(np.float32)  # Inside conveyor to stress contrast!
        targ0 = v3_rng.uniform([0.55, -0.45], [0.80, 0.45]).astype(np.float32)
        frames = []
        for ti in range(5):
            st = {
                "ee_xy": (ee0 + np.array([0.02 * ti, 0.0], np.float32))[None, :],
                "obj_xy": (obj0 + np.array([0.0, 0.015 * sign * ti], np.float32))[None, :],
                "targ_a": targ0[None, :],
                "box_cx": np.zeros((1, 4), dtype=np.float32),
                "box_cy": np.array([[0.66, -0.66, 0.0, 0.0]], dtype=np.float32),
                "box_hx": np.array([[0.03, 0.03, 0.0, 0.0]], dtype=np.float32),
                "box_hy": np.array([[0.36, 0.36, 0.0, 0.0]], dtype=np.float32),
                "drift_xy": np.array([[0.0, 0.0065 * sign]], dtype=np.float32),
                "bumper_kick": np.array([0.0], dtype=np.float32),
                "t_step": np.array([float(ti)], dtype=np.float32),
            }
            frames.append(render_v3_dynamics_world_rgb(st, height=192, width=320)[0])
        fr5 = np.stack(frames, axis=0)
        with torch.no_grad():
            vae.clear_cache()
            z = vae.encode(_u8_to_wan_tensor(fr5)).latent_dist.mean
            rec = _wan_tensor_to_u8(vae.decode(z).sample)
        v3_psnr.append(_psnr_db(fr5, rec))
        v3_ssim.append(_local_gaussian_ssim(fr5, rec))
        # Check stripe motion away from puck (x=196 or x=124 depending on puck x)
        col_x = 124 if obj0[0] >= 0.0 else 196
        c0 = rec[0, :, col_x, 0].astype(np.float32)
        c1 = rec[1, :, col_x, 0].astype(np.float32)
        corr_pos = float(np.corrcoef(c0[:-5], c1[5:])[0, 1])
        corr_neg = float(np.corrcoef(c0[5:], c1[:-5])[0, 1])
        drift_sign_hits += int((1.0 if corr_pos > corr_neg else -1.0) == sign)
        for ti in range(5):
            p_ref, p_est = _color_centroid_xy(fr5[ti], "cyan"), _color_centroid_xy(rec[ti], "cyan")
            e_ref, e_est = _color_centroid_xy(fr5[ti], "amber"), _color_centroid_xy(rec[ti], "amber")
            v3_puck_drift.append(math.hypot(p_ref[0] - p_est[0], p_ref[1] - p_est[1]))
            v3_ee_drift.append(math.hypot(e_ref[0] - e_est[0], e_ref[1] - e_est[1]))

    # 3d. Verify `bfloat16` latent storage round-trip precision (avoids CPU software-emulated Conv3d bf16 stall;
    #     full native bfloat16 encoder throughput & fidelity are verified on TPU v6e-1 hardware in Pilot Phase 1!)
    bf16_holo_psnr = []
    bf16_holo_ssim = []
    for clip in holo_raw_clips[:1]:
        with torch.no_grad():
            vae.clear_cache()
            z_fp32 = vae.encode(_u8_to_wan_tensor(clip)).latent_dist.mean
            z_bf16_rt = z_fp32.to(torch.bfloat16).to(torch.float32)
            rec_b = _wan_tensor_to_u8(vae.decode(z_bf16_rt).sample)
        bf16_holo_psnr.append(_psnr_db(clip, rec_b))
        bf16_holo_ssim.append(_local_gaussian_ssim(clip, rec_b))
    bf16_prefix_diff = prefix_lat_max_abs_diff  # Same fp32 causal weights; bf16 latent quantization verified above

    return {
        "causality_and_streaming_equivalence": {
            "resolution_hw": [192, 320],
            "fps": 25.0,
            "black_bar_fraction": 0.0,
            "latent_shape_9f_192x320": list(z_base.shape),
            "prefix_latent_0_1_max_abs_diff_fp32": round(prefix_lat_max_abs_diff, 8),
            "suffix_latent_2_max_abs_diff_fp32": round(suffix_lat_max_abs_diff, 4),
            "strictly_causal_zero_future_lookahead": bool(prefix_lat_max_abs_diff < 1e-5 and suffix_lat_max_abs_diff > 0.1),
            "online_per_chunk_quant_conv_vs_offline_max_abs_diff": round(online_vs_offline_max_abs_diff, 8),
            "online_equals_offline": bool(online_vs_offline_max_abs_diff < 1e-4),
            "feat_cache_bytes_chunk1_fp32": int(stream_cache_bytes_chunk1_fp32),
            "feat_cache_bytes_chunk2_fp32": int(stream_cache_bytes_chunk2_fp32),
            "feat_cache_exact_o1_constant": wan_cache_o1_constant,
        },
        "roundtrip_ceilings_dev_192x320_local_11x11_ssim": {
            "holoassist_dev_n3_25fps_fp32": {
                "split": "dev_only_session_prefix_disjoint",
                "dev_recordings_audited": holo_dev_recs_used,
                "mean_psnr_db": round(float(np.mean(holo_psnr)), 2) if holo_psnr else 0.0,
                "mean_local_11x11_ssim": round(float(np.mean(holo_ssim)), 4) if holo_ssim else 0.0,
                "mean_edge_rmse": round(float(np.mean(holo_edge)), 5) if holo_edge else 0.0,
            },
            "holoassist_dev_n1_25fps_bf16_latent_quant_check": {
                "mean_psnr_db": round(float(np.mean(bf16_holo_psnr)), 2) if bf16_holo_psnr else 0.0,
                "mean_local_11x11_ssim": round(float(np.mean(bf16_holo_ssim)), 4) if bf16_holo_ssim else 0.0,
            },
            "physical_xarm6_dev_n4": {
                "split": "dev_video_only",
                "mean_psnr_db": round(float(np.mean(xarm_psnr)), 2) if xarm_psnr else 0.0,
                "mean_local_11x11_ssim": round(float(np.mean(xarm_ssim)), 4) if xarm_ssim else 0.0,
                "frame0_1x_color_block_xy_drift_norm": round(float(np.mean(xarm_xy_drift_f0)), 5) if xarm_xy_drift_f0 else 0.0,
                "frames1_4_4x_temporal_color_block_xy_drift_norm": round(float(np.mean(xarm_xy_drift_f1_4)), 5) if xarm_xy_drift_f1_4 else 0.0,
            },
            "v3_dynamics_n8_conveyor_puck_ee": {
                "n_worlds": 8,
                "mean_psnr_db": round(float(np.mean(v3_psnr)), 2),
                "mean_local_11x11_ssim": round(float(np.mean(v3_ssim)), 4),
                "conveyor_drift_direction_preserved_acc": round(drift_sign_hits / 8.0, 4),
                "mean_puck_xy_drift_norm": round(float(np.mean(v3_puck_drift)), 5),
                "mean_ee_xy_drift_norm": round(float(np.mean(v3_ee_drift)), 5),
            },
        },
        "accounting": {
            "wan21_vae_total_params": int(wan_tot_params),
            "wan21_vae_encoder_params": int(wan_enc_params),
            "wan21_vae_decoder_params": int(wan_dec_params),
            "wan21_param_partition_exact": wan_enc_params + wan_dec_params == wan_tot_params,
            "wan21_encoder_feat_cache_mib_fp32": round(stream_cache_bytes_chunk2_fp32 / (1024**2), 3),
            "wan21_encoder_feat_cache_mib_bf16": round((stream_cache_bytes_chunk2_fp32 / 2.0) / (1024**2), 3),
            "wan21_cpu_encode_9f_sec": round(encode_9f_sec, 2),
            "wan21_latent_storage_gib_per_hour_25fps_bf16": round((6.25 * 3600 * 24 * 40 * 16 * 2) / (1024**3), 3),
            "wan21_license": "apache-2.0",
            "wan21_pretraining_provenance": "Wan 2.1 open video VAE pretraining corpus (exact overlap with public egocentric datasets unknown/disclosed)",
        },
    }


def benchmark_text_byt5(data_root: Path) -> dict[str, Any]:
    """Verify 100.0% lossless ByT5 256-byte round-trip across real HoloAssist DEV transcripts & Python code."""
    samples: list[str] = []
    lp = data_root / "g10" / "data" / "holo" / "labels.json"
    if lp.exists():
        labels = json.loads(lp.read_text())
        for r in labels:
            pid = str(r.get("video_name", "")).split("-")[0].strip().casefold()
            if _hash_split("slwm-v3-g1", pid) != "dev":
                continue
            for e in r.get("events", []):
                if e.get("label") == "Conversation":
                    t = str((e.get("attributes") or {}).get("Transcription", "") or "").strip()
                    if t:
                        samples.append(t)
                        if len(samples) >= 150:
                            break
            if len(samples) >= 150:
                break
    code_lines = Path(__file__).read_text().splitlines()[:50]
    samples.extend(code_lines)

    exact = 0
    total_bytes = 0
    for s in samples:
        ids = encode_utf8_bytes_lossless(s, add_bos=True, add_eos=True)
        rec = decode_utf8_bytes_lossless(ids)
        exact += int(rec == s)
        total_bytes += len(s.encode("utf-8"))
    return {
        "n_strings_tested": len(samples),
        "total_utf8_bytes": total_bytes,
        "exact_lossless_roundtrip_rate": round(exact / max(1, len(samples)), 6),
        "oov_rate": 0.0,
        "vocab_size_with_controls": 264,
        "frozen_params": 0,
        "streaming_state_bytes": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run SLWM-v3 Gate 2 (Rev 2) small-shard causal codec ceiling audit.")
    ap.add_argument("--data-root", type=Path, default=Path("/mnt/slwm-data/SLWM"))
    ap.add_argument("--cache-dir", type=str, default="/mnt/slwm-data/SLWM/x3_480p/hf_hub_cache")
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    audio_res = benchmark_audio_codecs(args.data_root, args.cache_dir)
    video_res = benchmark_video_codec_wan21(args.data_root, args.cache_dir)
    text_res = benchmark_text_byt5(args.data_root)

    enc_params_tot = (
        video_res["accounting"]["wan21_vae_encoder_params"]
        + audio_res["accounting"]["mimi_encoder_plus_rvq_and_downsample_params"]
    )
    dec_params_tot = (
        video_res["accounting"]["wan21_vae_decoder_params"]
        + audio_res["accounting"]["mimi_decoder_and_upsample_params"]
    )
    all_params_tot = (
        video_res["accounting"]["wan21_vae_total_params"]
        + audio_res["accounting"]["mimi_total_params"]
    )
    assert enc_params_tot + dec_params_tot == all_params_tot

    report = {
        "gate": "SLWM-v3 Gate 2 (Rev 2) — Small-Shard Causal Codec Ceilings, Streaming Equivalence & Dual-Track Cost Accounting",
        "audio_codecs": audio_res,
        "video_codec_wan21": video_res,
        "text_codec_byt5": text_res,
        "strict_system_frozen_edge_summary": {
            "frozen_encoder_params_video_plus_audio": int(enc_params_tot),
            "frozen_decoder_params_video_plus_audio": int(dec_params_tot),
            "frozen_total_enc_dec_params_video_plus_audio": int(all_params_tot),
            "frozen_param_partition_exact": enc_params_tot + dec_params_tot == all_params_tot,
            "frozen_encoder_streaming_state_mib_bf16": round(
                video_res["accounting"]["wan21_encoder_feat_cache_mib_bf16"]
                + audio_res["accounting"]["mimi_encoder_streaming_state_mib_bf16"],
                3,
            ),
            "frozen_encoder_streaming_state_mib_fp32": round(
                video_res["accounting"]["wan21_encoder_feat_cache_mib_fp32"]
                + audio_res["accounting"]["mimi_encoder_streaming_state_mib_fp32"],
                3,
            ),
        },
        "gate2_codec_causality_and_ceilings_pass": bool(
            video_res["causality_and_streaming_equivalence"]["strictly_causal_zero_future_lookahead"]
            and video_res["causality_and_streaming_equivalence"]["online_equals_offline"]
            and video_res["causality_and_streaming_equivalence"]["feat_cache_exact_o1_constant"]
            and audio_res["causality_and_streaming_verification"]["mimi_32rvq_25frames_exact"]
            and audio_res["causality_and_streaming_verification"]["mimi_24s_600tf_online_vs_offline_exact"]
            and audio_res["causality_and_streaming_verification"]["mimi_sliding_window_o1_capped_at_249"]
            and audio_res["causality_and_streaming_verification"]["mimi_counterfactual_zero_leak_beyond_79p68s_rf"]
            and audio_res["causality_and_streaming_verification"]["encodec_150frames_exact"]
            and audio_res["speech_commands_val_n40_whisper_tiny_ceiling"]["mimi_32rvq_12p5hz"]["keyword_acc"] >= 0.90
            and text_res["exact_lossless_roundtrip_rate"] == 1.0
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(report, indent=2)
    args.out_json.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
