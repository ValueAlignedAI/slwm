"""SLWM-v3 Zero-Pretrained Strictly-Causal 64-Bin Log-Mel Pre-Encoder (`AGENTS.md §4.1`).

Encodes ESC-50 (official Folds 1-4 train / Fold 5 test) and Clotho (official train / dev / test)
using ZERO pretrained weights and ZERO future lookahead:
  - Sample rate: `24,000 Hz` mono float32
  - Strictly causal framing: packet `k` uses ONLY `wav[k*1920 : (k+1)*1920]` (`80 ms` = exact `12.5 Hz`
    matching Kyutai Mimi's `12.5 Hz` packet rate 1:1 with zero future samples!)
  - Representation: 64-bin mel filterbank (`40 Hz .. 11,500 Hz`) log-energy standardized with fixed
    causal constants `(log(E + 1e-6) + 6.0) / 4.0` -> `float16[T_a, 64]` stored in `audio_cont`.

Used for the mandatory `AGENTS.md §4.1` `Backbone x Representation` 2x2 causal ablation decoupling
sequence-backbone gains from pretrained-codec gains at matched parameters, splits, and token budgets.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.v3_data_contract import mimi_packet_availability_timestamps
from data.v3_unified_shards import BYTE_BOS, BYTE_EOS, BYTE_MOD_SEP, SparseShardBuilder, encode_utf8_bytes
from scripts.data.v3_preencode_av_codecs import _decode_audio_bytes_ffmpeg_24k, _decode_wav_bytes_24k


def _build_mel_filterbank(sr: int = 24000, n_fft: int = 2048, n_mels: int = 64, fmin: float = 40.0, fmax: float = 11500.0) -> np.ndarray:
    """Deterministic Slaney-style triangular mel filterbank `[n_fft // 2 + 1, n_mels]`."""
    def _hz_to_mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def _mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(_hz_to_mel(np.array(fmin)), _hz_to_mel(np.array(fmax)), n_mels + 2)
    hz = _mel_to_hz(mels)
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float32)
    for m in range(n_mels):
        f_l, f_c, f_r = hz[m], hz[m + 1], hz[m + 2]
        up = (freqs - f_l) / max(f_c - f_l, 1e-6)
        dn = (f_r - freqs) / max(f_r - f_c, 1e-6)
        fb[:, m] = np.maximum(0.0, np.minimum(up, dn))
    # Area-normalize each filter
    norm = np.maximum(fb.sum(axis=0, keepdims=True), 1e-6)
    return (fb / norm).astype(np.float32)


_MEL_FB = _build_mel_filterbank()
_HANN_1920 = np.hanning(1920).astype(np.float32)


def encode_audio_causal_logmel64(wav_24k: np.ndarray) -> np.ndarray:
    """Strictly causal 12.5 Hz (`1,920`-sample / `80 ms` non-overlapping past window) 64-bin log-mel -> `float16[T_a, 64]`."""
    n_frames = wav_24k.size // 1920
    if n_frames <= 0:
        return np.zeros((0, 64), dtype=np.float16)
    frames = wav_24k[: n_frames * 1920].reshape(n_frames, 1920) * _HANN_1920[None, :]
    spec = np.fft.rfft(frames, n=2048, axis=-1)
    pwr = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32) / 2048.0
    mel_e = pwr @ _MEL_FB  # [T_a, 64]
    log_mel = (np.log(mel_e + 1e-6) + 6.0) / 4.0
    return np.clip(log_mel, -4.0, 4.0).astype(np.float16)


def _build_logmel_text_episode(audio_cont: np.ndarray, text_str: str, *, text_first: bool = False) -> dict[str, Any]:
    events: list[tuple[float, int, int, Any]] = []
    t_off = 0.0
    if text_str and text_first:
        tb = encode_utf8_bytes(text_str, bos=BYTE_BOS, eos=BYTE_MOD_SEP, max_bytes=512)
        for i, b_val in enumerate(tb):
            events.append((i * 0.02, 0, 2, int(b_val)))
        t_off = len(tb) * 0.02 + 0.04
    max_t = t_off
    m_ts = mimi_packet_availability_timestamps(num_packets=len(audio_cont), frame_rate=12.5)
    for ai in range(len(audio_cont)):
        av_t = t_off + m_ts[ai].availability_time_s
        max_t = max(max_t, av_t)
        events.append((av_t, 20, 1, audio_cont[ai]))
    if text_str and (not text_first):
        tb = encode_utf8_bytes(text_str, bos=BYTE_MOD_SEP, eos=BYTE_EOS, max_bytes=512)
        base_t = max_t + 0.04
        for i, b_val in enumerate(tb):
            events.append((base_t + i * 0.02, 30, 2, int(b_val)))
    events.sort(key=lambda e: (e[0], e[1]))
    n = len(events)
    mod_id = np.empty((n,), dtype=np.uint8)
    dt_s = np.empty((n,), dtype=np.float16)
    text_b = np.full((n,), 256, dtype=np.uint16)
    aud_c_list: list[np.ndarray] = []
    prev_t = events[0][0] if n > 0 else 0.0
    for idx, (av_t, _, m_id, payload) in enumerate(events):
        mod_id[idx] = m_id
        dt_s[idx] = np.float16(max(0.0, av_t - prev_t))
        prev_t = av_t
        if m_id == 1:
            aud_c_list.append(payload)
        elif m_id == 2:
            text_b[idx] = payload
    return {
        "modality_id": mod_id,
        "dt_s": dt_s,
        "text_byte": text_b,
        "audio_cont": np.stack(aud_c_list, axis=0) if aud_c_list else np.zeros((0, 64), dtype=np.float16),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-encode zero-pretrained causal 64-bin log-mel ESC-50 & Clotho shards.")
    ap.add_argument("--raw-root", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_diverse_raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded_scratch_logmel/audio_text"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {"representation": "causal_logmel64_12p5hz_zero_pretrained", "shards": {}}

    # 1. ESC-50 (official Folds 1-4 train / Fold 5 test)
    pqs = sorted(glob.glob(str(args.raw_root / "esc50/data/*.parquet")))
    sb_tr = SparseShardBuilder("esc50_train_f14_logmel64", "train")
    sb_te = SparseShardBuilder("esc50_test_f5_logmel64", "test")
    for fp in pqs:
        tbl = pq.read_table(fp, columns=["fold", "category", "audio"]).to_pydict()
        for fold, cat, aud_obj in zip(tbl["fold"], tbl["category"], tbl["audio"]):
            raw_b = aud_obj["bytes"] if isinstance(aud_obj, dict) else aud_obj
            wav = _decode_wav_bytes_24k(raw_b) if raw_b[:4] == b"RIFF" else _decode_audio_bytes_ffmpeg_24k(raw_b)
            lm = encode_audio_causal_logmel64(wav)
            if len(lm) == 0:
                continue
            ep = _build_logmel_text_episode(lm, f"sound: {cat}", text_first=False)
            (sb_te if int(fold) == 5 else sb_tr).add_episode(domain_name="esc50", **ep)
    receipt["shards"]["esc50_train_f14"] = sb_tr.save(args.out_dir / "esc50_train_f14.npz")
    receipt["shards"]["esc50_test_f5"] = sb_te.save(args.out_dir / "esc50_test_f5.npz")

    # 2. Clotho (official train / valid->dev / test)
    for split_nm, out_split in (("train", "train"), ("valid", "dev"), ("test", "test")):
        cpqs = sorted(glob.glob(str(args.raw_root / f"clotho/data/{split_nm}/*.parquet")))
        if not cpqs:
            continue
        sb = SparseShardBuilder(f"clotho_{out_split}_logmel64", out_split)
        cnt = 0
        for fp in cpqs:
            tbl = pq.read_table(fp, columns=["audio", "text"]).to_pydict()
            for aud_obj, caption in zip(tbl["audio"], tbl["text"]):
                raw_b = aud_obj["bytes"] if isinstance(aud_obj, dict) else aud_obj
                wav = _decode_wav_bytes_24k(raw_b) if raw_b[:4] == b"RIFF" else _decode_audio_bytes_ffmpeg_24k(raw_b)
                if wav.size < 4800:
                    continue
                lm = encode_audio_causal_logmel64(wav)
                if len(lm) == 0:
                    continue
                txt_first = (out_split == "train") and ((cnt % 2) == 1)
                ep = _build_logmel_text_episode(lm, str(caption or ""), text_first=txt_first)
                sb.add_episode(domain_name="clotho", **ep)
                cnt += 1
        receipt["shards"][f"clotho_{out_split}"] = sb.save(args.out_dir / f"clotho_{out_split}.npz")

    (args.out_dir / "scratch_logmel64_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
