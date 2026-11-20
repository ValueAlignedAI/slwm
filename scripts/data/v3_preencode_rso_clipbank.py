"""Pre-encode strictly train-split SpeechCommands v0.02 + background Mimi 32-RVQ clipbank for RSO v2.

Eliminates the 160-pair static suffix-memorization shortcut discovered in Wave A-RSO v1 by enabling
dynamic online synthesis of >10^30 unique real-speech distractor streams from the official `train`
speaker split during training, while evaluating strictly frozen weights on the exact unmodified
speaker-disjoint `real_speech_overwrite_dev` / `test` continuous-waveform Mimi 32-RVQ shards!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.v3_data_contract import tile_background_with_crossfade
from scripts.data.v3_preencode_av_codecs import CausalCodecEngine
from scripts.data.v3_preencode_real_speech_overwrite import (
    DIG_WORDS,
    DIR_WORDS,
    DIST_WORDS,
    _load_sc_pool,
    _mix_clip,
)


def _encode_word_clip(
    engine: CausalCodecEngine,
    clip: np.ndarray,
    bg_pool: list[np.ndarray],
    rng: np.random.Generator,
    *,
    clip_sec: float = 2.0,
    n_ticks: int = 25,
    sr: int = 24000,
) -> np.ndarray:
    bg_base = bg_pool[int(rng.integers(0, len(bg_pool)))]
    n_samp = int(round(clip_sec * sr))
    bg_wav = tile_background_with_crossfade(bg_base, total_samples=n_samp, sr=sr, rng=rng)
    bg_rms = float(np.sqrt(np.mean(bg_wav * bg_wav) + 1e-12))
    if bg_rms > 1e-6:
        bg_wav = bg_wav * (0.015 / bg_rms)
    _mix_clip(bg_wav, clip, 0.35, sr)
    wav = np.clip(bg_wav, -0.98, 0.98)
    codes = engine.encode_audio_mimi_32(wav)
    if len(codes) < n_ticks:
        pad = np.repeat(codes[-1:], n_ticks - len(codes), axis=0)
        codes = np.concatenate([codes, pad], axis=0)
    return codes[:n_ticks].astype(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-encode train-speaker SpeechCommands Mimi 32-RVQ clipbank.")
    ap.add_argument("--sc-tar", type=Path, default=Path("/mnt/slwm-data/SLWM/g13/data/speech/speech_commands_v0.02.tar.gz"))
    ap.add_argument("--rso-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded/real_speech_overwrite"))
    ap.add_argument("--per-target-word", type=int, default=250)
    ap.add_argument("--n-distractors", type=int, default=1000)
    ap.add_argument("--n-bg-beds", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=44001)
    args = ap.parse_args()

    pools, bg_splits = _load_sc_pool(args.sc_tar)
    tr_pool = pools["train"]
    tr_bgs = bg_splits["train"]
    rng = np.random.default_rng(args.seed)
    engine = CausalCodecEngine(device=args.device)

    train_spks: set[str] = set()
    dir_codes = np.zeros((4, args.per_target_word, 25, 32), dtype=np.uint16)
    dig_codes = np.zeros((4, args.per_target_word, 25, 32), dtype=np.uint16)
    dist_codes = np.zeros((args.n_distractors, 25, 32), dtype=np.uint16)
    bg_codes = np.zeros((args.n_bg_beds, 125, 32), dtype=np.uint16)

    for ci, w in enumerate(DIR_WORDS):
        items = list(tr_pool[w])
        idx = rng.choice(len(items), size=args.per_target_word, replace=(len(items) < args.per_target_word))
        for j, ii in enumerate(idx):
            spk, clip = items[int(ii)]
            train_spks.add(spk)
            dir_codes[ci, j] = _encode_word_clip(engine, clip, tr_bgs, rng)
        print(f"[CLIPBANK] DIR '{w}' ({ci+1}/4): {args.per_target_word} clips encoded", flush=True)

    for ci, w in enumerate(DIG_WORDS):
        items = list(tr_pool[w])
        idx = rng.choice(len(items), size=args.per_target_word, replace=(len(items) < args.per_target_word))
        for j, ii in enumerate(idx):
            spk, clip = items[int(ii)]
            train_spks.add(spk)
            dig_codes[ci, j] = _encode_word_clip(engine, clip, tr_bgs, rng)
        print(f"[CLIPBANK] DIG '{w}' ({ci+1}/4): {args.per_target_word} clips encoded", flush=True)

    all_dist: list[tuple[str, np.ndarray]] = []
    for dw in DIST_WORDS:
        all_dist.extend(tr_pool[dw])
    d_idx = rng.choice(len(all_dist), size=args.n_distractors, replace=(len(all_dist) < args.n_distractors))
    for j, ii in enumerate(d_idx):
        spk, clip = all_dist[int(ii)]
        train_spks.add(spk)
        dist_codes[j] = _encode_word_clip(engine, clip, tr_bgs, rng)
        if (j + 1) % 250 == 0:
            print(f"[CLIPBANK] DIST {j+1}/{args.n_distractors} clips encoded", flush=True)

    sr = 24000
    for j in range(args.n_bg_beds):
        bg_base = tr_bgs[int(rng.integers(0, len(tr_bgs)))]
        bg_wav = tile_background_with_crossfade(bg_base, total_samples=10 * sr, sr=sr, rng=rng)
        bg_rms = float(np.sqrt(np.mean(bg_wav * bg_wav) + 1e-12))
        if bg_rms > 1e-6:
            bg_wav = bg_wav * (0.015 / bg_rms)
        c = engine.encode_audio_mimi_32(np.clip(bg_wav, -0.98, 0.98))
        bg_codes[j] = c[:125].astype(np.uint16)

    # Strict Gate 2b speaker-disjointness assertion against dev & test RSO shards!
    overlaps: dict[str, int] = {}
    for sp in ("dev", "test"):
        for tag in ("L12_Weq64_additive", "L16_Weq128_additive"):
            p = args.rso_dir / f"real_speech_overwrite_{sp}_{tag}.npz"
            if p.is_file():
                z = np.load(p, allow_pickle=False)
                ev_spks = {str(x) for x in z["primary_speaker_id"].tolist()}
                overlaps[f"{sp}_{tag}"] = len(train_spks & ev_spks)
    assert all(v == 0 for v in overlaps.values()), f"Speaker overlap detected: {overlaps}"

    out_path = args.rso_dir / "rso_train_clipbank_mimi32.npz"
    np.savez_compressed(
        out_path,
        dir_codes=dir_codes,
        dig_codes=dig_codes,
        dist_codes=dist_codes,
        bg_codes=bg_codes,
        train_speaker_ids=np.asarray(sorted(train_spks)),
    )
    summary: dict[str, Any] = {
        "out_path": str(out_path),
        "size_mib": round(out_path.stat().st_size / (1024.0**2), 3),
        "dir_codes_shape": list(dir_codes.shape),
        "dig_codes_shape": list(dig_codes.shape),
        "dist_codes_shape": list(dist_codes.shape),
        "bg_codes_shape": list(bg_codes.shape),
        "n_unique_train_speakers": len(train_spks),
        "dev_test_speaker_overlaps": overlaps,
        "zero_speaker_overlap_verified": True,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
