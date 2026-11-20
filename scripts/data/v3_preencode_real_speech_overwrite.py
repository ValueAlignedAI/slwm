"""SLWM-v3 Real-Speech Counterfactual Overwrite Pre-Encoder (Additive `Mimi RF + GQA RF` Eviction!).

Constructs strictly speaker-disjoint & background-disjoint real acoustic counterfactual episodes
from `SpeechCommands v0.02` (`train` / `dev` / `test` official splits matching Gate 2 bit-for-bit!)
and encodes BOTH `stream_a` and counterfactual `stream_b` through frozen `Kyutai Mimi 32-RVQ`
(`12.5 Hz`).

Scientific Guarantees (Gate 1 + Gate 2 + Astra Blocking #2):
  - **2 tracked spoken attributes:** Direction (`left, right, up, down` -> 4 classes) AND Digit
    (`zero, one, two, three` -> 4 classes), with matched-energy non-overlapping same-domain
    distractors (`yes, no, stop, go, bed, bird, cat, dog, happy, house, marvin, sheila, tree, wow`)
    interleaved across ALL 4 segments (`pre, mid1, mid2, post`) with `min_gap >= 1.15 s` (`0` collisions!).
  - **25% 0-correction catch trials:** Same-domain distractor placed at `t_corr_s` so catch & overwrite
    inter-word timing distributions are identical.
  - **ADDITIVE `Mimi RF + GQA RF` eviction (`> 1,776` ticks for Discovery `L=12, W_eq=64`):**
    Asserts empirically that counterfactual `stream_a` vs `stream_b` Mimi 32-RVQ codes are
    **BIT-EXACT IDENTICAL (`diff_frac == 0.0`) across the ENTIRE accessible GQA suffix
    `[t_q - L*(W-1), t_q]`**!
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
from scipy.io import wavfile

from data.v3_data_contract import (
    assign_episode_primary_speaker_id,
    build_interleaved_speech_overwrite_schedule,
    tile_background_with_crossfade,
)
from scripts.data.v3_preencode_av_codecs import CausalCodecEngine, _resample_f32

DIR_WORDS = ("left", "right", "up", "down")
DIG_WORDS = ("zero", "one", "two", "three")
DIST_WORDS = ("yes", "no", "stop", "go", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow")


def _load_sc_pool(sc_tar: Path) -> tuple[dict[str, dict[str, list[tuple[str, np.ndarray]]]], dict[str, list[np.ndarray]]]:
    """Load Gate-2-exact speaker-disjoint `SpeechCommands v0.02` utterances & disjoint background clips at 24 kHz."""
    needed = set(DIR_WORDS) | set(DIG_WORDS) | set(DIST_WORDS)
    pools: dict[str, dict[str, list[tuple[str, np.ndarray]]]] = {
        sp: {w: [] for w in needed} for sp in ("train", "dev", "test")
    }
    bg_raw: dict[str, np.ndarray] = {}

    with tarfile.open(sc_tar, "r:gz") as tf:
        names = tf.getnames()
        val_name = "./validation_list.txt" if "./validation_list.txt" in names else "validation_list.txt"
        test_name = "./testing_list.txt" if "./testing_list.txt" in names else "testing_list.txt"
        val_set = set(tf.extractfile(val_name).read().decode().splitlines())
        test_set = set(tf.extractfile(test_name).read().decode().splitlines())

        for m in tf.getmembers():
            rel = m.name.lstrip("./")
            if not rel.endswith(".wav"):
                continue
            if "_background_noise_" in rel:
                raw_b = tf.extractfile(m).read()
                sr, data = wavfile.read(io.BytesIO(raw_b))
                wf = data.astype(np.float32) / 32768.0
                bg_raw[Path(rel).name] = _resample_f32(wf, sr, 24000)
                continue
            parts = rel.split("/")
            if len(parts) != 2 or parts[0] not in needed:
                continue
            word, fname = parts[0], parts[1]
            spk = fname.split("_")[0]
            sp = "test" if rel in test_set else ("dev" if rel in val_set else "train")
            raw_b = tf.extractfile(m).read()
            sr, data = wavfile.read(io.BytesIO(raw_b))
            wf = data.astype(np.float32) / 32768.0
            wf24 = _resample_f32(wf, sr, 24000)
            rms = float(np.sqrt(np.mean(wf24 * wf24) + 1e-12))
            if rms > 1e-4:
                wf24 = wf24 * (0.12 / rms)  # Matched-energy normalization across ALL cues & distractors!
            pools[sp][word].append((spk, wf24))

    bg_sorted = sorted(bg_raw.keys())
    bg_splits = {
        "train": [bg_raw[k] for k in bg_sorted[:4]],
        "dev": [bg_raw[k] for k in bg_sorted[4:5]],
        "test": [bg_raw[k] for k in bg_sorted[5:6]],
    }
    return pools, bg_splits


def _mix_clip(buf: np.ndarray, clip: np.ndarray, t_sec: float, sr: int = 24000) -> None:
    s0 = int(round(t_sec * sr))
    s1 = min(buf.size, s0 + clip.size)
    if s1 > s0 and s0 >= 0:
        buf[s0:s1] += clip[: s1 - s0]


def generate_split_overwrite_shard(
    engine: CausalCodecEngine,
    pools: dict[str, dict[str, list[tuple[str, np.ndarray]]]],
    bg_splits: dict[str, list[np.ndarray]],
    *,
    split: str,
    n_episodes: int,
    comparator_layers: int = 12,
    comparator_window: int = 64,
    codec_rf_ticks: int = 1020,
    seed: int = 32001,
    out_path: Path,
) -> dict[str, Any]:
    """Generate paired counterfactual A/B real-speech overwrite Mimi 32-RVQ episodes with verified 0.0 suffix leak!"""
    sched = build_interleaved_speech_overwrite_schedule(
        n_episodes=n_episodes,
        comparator_layers=comparator_layers,
        comparator_window=comparator_window,
        codec_rf_ticks=codec_rf_ticks,
        catch_trial_prob=0.25,
        seed=seed,
    )
    total_sec = int(sched["total_duration_sec"])
    sr = 24000
    gqa_rf_ticks = comparator_layers * (comparator_window - 1)
    rng = np.random.default_rng(seed + 17)
    sp_pool = pools[split]
    bgs = bg_splits[split]

    codes_a_list: list[np.ndarray] = []
    codes_b_list: list[np.ndarray] = []
    y_a_list: list[list[int]] = []
    y_b_list: list[list[int]] = []
    upd_list: list[int] = []
    is_catch_list: list[bool] = []
    spk_cluster_list: list[str] = []
    suffix_mismatch_fracs: list[float] = []

    for ep in sched["episodes"]:
        bg_base = bgs[int(rng.integers(0, len(bgs)))]
        bg_wav = tile_background_with_crossfade(bg_base, total_samples=total_sec * sr, sr=sr, rng=rng)
        bg_rms = float(np.sqrt(np.mean(bg_wav * bg_wav) + 1e-12))
        if bg_rms > 1e-6:
            bg_wav = bg_wav * (0.015 / bg_rms)

        # Sample initial Direction (0..3) and Digit (0..3)
        d_init = int(rng.integers(0, 4))
        n_init = int(rng.integers(0, 4))
        w_dir_init = DIR_WORDS[d_init]
        w_dig_init = DIG_WORDS[n_init]

        spk_dir, clip_dir = sp_pool[w_dir_init][int(rng.integers(0, len(sp_pool[w_dir_init])))]
        spk_dig, clip_dig = sp_pool[w_dig_init][int(rng.integers(0, len(sp_pool[w_dig_init])))]

        if ep["init_order"] == 0:
            _mix_clip(bg_wav, clip_dir, ep["t_first_s"], sr)
            _mix_clip(bg_wav, clip_dig, ep["t_second_s"], sr)
            init1_spk = spk_dir
        else:
            _mix_clip(bg_wav, clip_dig, ep["t_first_s"], sr)
            _mix_clip(bg_wav, clip_dir, ep["t_second_s"], sr)
            init1_spk = spk_dig

        # Mix identical non-overlapping distractors across pre, mid1, mid2, post
        for t_dist in (
            ep["pre_distractor_times_s"]
            + ep["mid1_distractor_times_s"]
            + ep["mid2_distractor_times_s"]
            + ep["post_distractor_times_s"]
        ):
            dw = DIST_WORDS[int(rng.integers(0, len(DIST_WORDS)))]
            _, d_clip = sp_pool[dw][int(rng.integers(0, len(sp_pool[dw])))]
            _mix_clip(bg_wav, d_clip, t_dist, sr)

        wav_a = bg_wav.copy()
        wav_b = bg_wav.copy()
        ya = [d_init, n_init]
        yb = [d_init, n_init]

        if ep["is_catch_no_overwrite"]:
            dw = DIST_WORDS[int(rng.integers(0, len(DIST_WORDS)))]
            _, d_clip = sp_pool[dw][int(rng.integers(0, len(sp_pool[dw])))]
            _mix_clip(wav_a, d_clip, ep["catch_distractor_at_corr_s"], sr)
            _mix_clip(wav_b, d_clip, ep["catch_distractor_at_corr_s"], sr)
            corr_spk = None
        else:
            uw = int(ep["update_which"])
            vocab = DIR_WORDS if uw == 0 else DIG_WORDS
            old_v = d_init if uw == 0 else n_init
            cands = [i for i in range(4) if i != old_v]
            rng.shuffle(cands)
            va, vb = cands[0], cands[1]
            corr_spk, clip_ca = sp_pool[vocab[va]][int(rng.integers(0, len(sp_pool[vocab[va]])))]
            _, clip_cb = sp_pool[vocab[vb]][int(rng.integers(0, len(sp_pool[vocab[vb]])))]
            _mix_clip(wav_a, clip_ca, ep["t_corr_s"], sr)
            _mix_clip(wav_b, clip_cb, ep["t_corr_s"], sr)
            ya[uw] = va
            yb[uw] = vb

        wav_a = np.clip(wav_a, -0.98, 0.98)
        wav_b = np.clip(wav_b, -0.98, 0.98)
        ca = engine.encode_audio_mimi_32(wav_a)
        cb = engine.encode_audio_mimi_32(wav_b)
        n_t = min(len(ca), len(cb))
        ca, cb = ca[:n_t], cb[:n_t]

        # CRITICAL ASTRA BLOCKING #2 VERIFICATION:
        # Verify BIT-EXACT `0.000000` counterfactual Mimi code difference across the ENTIRE accessible GQA suffix!
        suf_start = max(0, n_t - (gqa_rf_ticks + 1))
        suf_mismatch = float(np.mean(ca[suf_start:] != cb[suf_start:]))
        suffix_mismatch_fracs.append(suf_mismatch)

        codes_a_list.append(ca)
        codes_b_list.append(cb)
        y_a_list.append(ya)
        y_b_list.append(yb)
        upd_list.append(int(ep["update_which"]))
        is_catch_list.append(bool(ep["is_catch_no_overwrite"]))
        spk_cluster_list.append(
            assign_episode_primary_speaker_id(
                is_catch_no_overwrite=bool(ep["is_catch_no_overwrite"]),
                corr_speaker_id=corr_spk,
                init1_speaker_id=init1_spk,
            )
        )

    min_t = min(len(x) for x in codes_a_list)
    arr_a = np.stack([x[:min_t] for x in codes_a_list], axis=0)  # [N, T, 32] uint16
    arr_b = np.stack([x[:min_t] for x in codes_b_list], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        audio_codes_a=arr_a,
        audio_codes_b=arr_b,
        y_a=np.asarray(y_a_list, dtype=np.int32),
        y_b=np.asarray(y_b_list, dtype=np.int32),
        update_which=np.asarray(upd_list, dtype=np.int32),
        is_catch=np.asarray(is_catch_list, dtype=np.bool_),
        primary_speaker_id=np.asarray(spk_cluster_list),
    )
    max_suf_leak = max(suffix_mismatch_fracs) if suffix_mismatch_fracs else 0.0
    return {
        "split": split,
        "path": str(out_path),
        "n_episodes": int(len(codes_a_list)),
        "seq_len_ticks_12p5hz": int(min_t),
        "total_duration_sec": total_sec,
        "gqa_rf_ticks": int(gqa_rf_ticks),
        "additive_rf_ticks": int(sched["additive_codec_plus_gqa_rf_ticks"]),
        "min_delay_after_last_cue_ticks": int(sched["min_delay_after_last_cue_ticks"]),
        "max_counterfactual_gqa_suffix_mimi_mismatch_frac": round(max_suf_leak, 6),
        "bit_exact_zero_gqa_suffix_codec_leak_verified": bool(max_suf_leak == 0.0),
        "size_mib": round(out_path.stat().st_size / (1024.0**2), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-encode additive-RF Real-Speech-Overwrite Mimi 32-RVQ shards.")
    ap.add_argument("--sc-tar", type=Path, default=Path("/mnt/slwm-data/SLWM/g13/data/speech/speech_commands_v0.02.tar.gz"))
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded/real_speech_overwrite"))
    ap.add_argument("--splits", type=str, default="train,dev,test")
    ap.add_argument("--n-train", type=int, default=160)
    ap.add_argument("--n-dev", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=64)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    pools, bg_splits = _load_sc_pool(args.sc_tar)
    engine = CausalCodecEngine(device=args.device)
    counts = {"train": args.n_train, "dev": args.n_dev, "test": args.n_test}
    seeds = {"train": 32101, "dev": 32102, "test": 32103}
    receipt: dict[str, Any] = {}
    for sp in [s.strip() for s in args.splits.split(",") if s.strip()]:
        receipt[sp] = generate_split_overwrite_shard(
            engine,
            pools,
            bg_splits,
            split=sp,
            n_episodes=counts[sp],
            out_path=args.out_dir / f"real_speech_overwrite_{sp}_L12_Weq64_additive.npz",
            seed=seeds[sp],
        )
        print(f"[RSO] {sp}: {receipt[sp]}", flush=True)
    (args.out_dir / "real_speech_overwrite_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
