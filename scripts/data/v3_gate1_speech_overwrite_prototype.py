"""SLWM-v3 Gate 1: Real-Audio Same-Domain Selective-Overwrite Prototype & Anti-Splice Audit (`SpeechCommands v0.02`).

Closes Gate 1 Blocking Flaws #6 and #7 by constructing and auditing real 16 kHz continuous audio
streams directly from held-out `SpeechCommands v0.02` test speakers (`250` unseen test speakers,
zero train/val overlap):
  - Fact Family 1 (`Direction`, 4 classes): `{left, right, up, down}`
  - Fact Family 2 (`Digit`, 4 classes): `{zero, one, two, three}`
  - Same-Domain Non-Correction Distractors (exact same 16 kHz domain & speaker pool):
    `{yes, no, happy, house, marvin, sheila, cat, dog, bird, tree}`
  - Zero 6-word grammar contradiction: each single-word correction updates EXCLUSIVELY Direction
    OR EXCLUSIVELY Digit (50/50 randomized), while the untouched concurrent fact must survive.
  - Zero studio-vs-audiobook or loudness splice shortcut:
    * All initial, correction, and distractor words are drawn from the exact same `SpeechCommands v0.02`
      recording domain (and matched-speaker on 50% of correction/distractor pairs).
    * Every utterance is RMS-matched (`target_rms = 0.08`) with 15 ms cosine fade-in/out over continuous
      real `_background_noise_` ambience (zero dead-silence click boundaries).
    * Anti-splice acoustic audit verifies that low-level acoustic features (RMS energy, peak amplitude,
      zero-crossing rate, spectral centroid) cannot distinguish correction windows from same-domain
      non-correction distractor windows (`AUROC ~ 0.50`).
  - Bit-exact counterfactual PCM audit: verifies `max_abs_diff == 0.0` across ALL audio samples after
    `t_corr_end`, randomized event times/order, `t_corr - t_first > W`, and `delay_after_corr > L*(W-1)`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
from scipy.io import wavfile

DIRECTION_WORDS = ("left", "right", "up", "down")
DIGIT_WORDS = ("zero", "one", "two", "three")
DISTRACTOR_WORDS = ("yes", "no", "happy", "house", "marvin", "sheila", "cat", "dog", "bird", "tree")
ALL_USED_WORDS = set(DIRECTION_WORDS) | set(DIGIT_WORDS) | set(DISTRACTOR_WORDS)
SR = 16000


def _load_wav_mono_f32(raw_bytes: bytes, target_len: int = SR) -> np.ndarray:
    sr, x = wavfile.read(io.BytesIO(raw_bytes))
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {sr}")
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    out = np.zeros((target_len,), dtype=np.float32)
    n = min(target_len, x.size)
    out[:n] = x[:n]
    return out


def _rms_match_and_fade(x: np.ndarray, target_rms: float = 0.08, fade_ms: float = 15.0) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x * x) + 1e-10))
    y = x * (target_rms / max(rms, 1e-4))
    y = np.clip(y, -0.95, 0.95)
    n_fade = min(int(round(SR * fade_ms / 1000.0)), y.size // 4)
    if n_fade > 1:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_fade, dtype=np.float32)))
        y[:n_fade] *= ramp
        y[-n_fade:] *= ramp[::-1]
    return y.astype(np.float32)


def _acoustic_features(seg: np.ndarray) -> np.ndarray:
    """Low-level non-semantic acoustic features [rms, peak, zcr, spectral_centroid_hz]."""
    rms = float(np.sqrt(np.mean(seg * seg) + 1e-10))
    peak = float(np.max(np.abs(seg)))
    zcr = float(np.mean((seg[:-1] * seg[1:]) < 0.0))
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size))) + 1e-10
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / SR)
    centroid = float(np.sum(freqs * spec) / np.sum(spec))
    return np.array([rms, peak, zcr, centroid], dtype=np.float64)


def _binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # Exact Mann-Whitney U
    gt = np.sum(pos[:, None] > neg[None, :])
    eq = np.sum(pos[:, None] == neg[None, :])
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def run_speech_overwrite_prototype_audit(
    tar_path: Path,
    *,
    n_episodes: int = 64,
    mimi_hz: float = 12.5,
    comparator_layers: int = 12,
    comparator_window: int = 32,
    seed: int = 31009,
) -> dict[str, Any]:
    """Build N continuous real-audio SpeechCommands test episodes and verify causality & anti-splice properties."""
    receptive_field_ticks = comparator_layers * (comparator_window - 1)
    # Schedule window in seconds (at 12.5 Hz Mimi cadence, W=32 ticks = 2.56s; L*(W-1)=372 ticks = 29.76s)
    w_sec = comparator_window / mimi_hz
    rf_sec = receptive_field_ticks / mimi_hz
    total_sec = int(np.ceil(12.0 + rf_sec + 4.0))  # ~46s for L=12, W=32 (575 Mimi ticks)
    total_samples = total_sec * SR

    word_bank: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
    bg_noises: list[np.ndarray] = []

    with tarfile.open(tar_path, "r:gz") as tf:
        names = tf.getnames()
        test_name = "./testing_list.txt" if "./testing_list.txt" in names else "testing_list.txt"
        test_set = set(tf.extractfile(test_name).read().decode().splitlines())
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".wav"):
                continue
            rel = m.name.lstrip("./")
            if "_background_noise_" in rel:
                f = tf.extractfile(m)
                if f is not None:
                    sr, arr = wavfile.read(io.BytesIO(f.read()))
                    if sr == SR:
                        arr = arr.astype(np.float32) / (32768.0 if arr.dtype == np.int16 else 1.0)
                        if arr.ndim > 1:
                            arr = arr.mean(axis=-1)
                        bg_noises.append(arr)
                continue
            if rel not in test_set:
                continue
            word, fn = rel.split("/", 1)
            if word not in ALL_USED_WORDS:
                continue
            spk = fn.split("_")[0]
            f = tf.extractfile(m)
            if f is not None:
                word_bank[word].append((spk, f.read()))

    rng = np.random.default_rng(seed)
    corr_feats: list[np.ndarray] = []
    dist_feats: list[np.ndarray] = []
    max_post_corr_pcm_diff = 0.0
    min_corr_pcm_diff = float("inf")
    min_delay_ticks = 10**9
    min_init_to_corr_ticks = 10**9
    used_speakers: set[str] = set()
    t_corrs: list[float] = []
    which_updates: list[int] = []
    init_orders: list[int] = []

    for ei in range(n_episodes):
        # Continuous real SpeechCommands ambient background bed (RMS ~ 0.008)
        bg_src = bg_noises[int(rng.integers(0, len(bg_noises)))]
        start_bg = int(rng.integers(0, max(1, bg_src.size - total_samples)))
        if bg_src.size >= total_samples:
            bed = bg_src[start_bg : start_bg + total_samples].copy()
        else:
            reps = (total_samples // bg_src.size) + 2
            bed = np.tile(bg_src, reps)[start_bg : start_bg + total_samples].copy()
        bed = bed * (0.008 / max(float(np.sqrt(np.mean(bed * bed) + 1e-10)), 1e-5))

        init_ord = int(rng.integers(0, 2))
        upd_which = int(rng.integers(0, 2))
        init_orders.append(init_ord)
        which_updates.append(upd_which)

        dir_init, dir_a, dir_b = rng.choice(DIRECTION_WORDS, size=3, replace=False)
        dig_init, dig_a, dig_b = rng.choice(DIGIT_WORDS, size=3, replace=False)

        t_first_s = float(rng.uniform(0.5, 2.5))
        t_second_s = t_first_s + float(rng.uniform(1.5, 3.0))
        t_corr_s = max(t_second_s + 1.5, t_first_s + w_sec + 1.0) + float(rng.uniform(0.0, 2.0))
        t_corrs.append(t_corr_s)

        # Place 8 same-domain distractor words across the post-correction retention window
        dist_times = sorted(float(rng.uniform(t_corr_s + 2.0, total_sec - 2.0)) for _ in range(8))

        def _draw_word(w: str, prefer_spk: str | None = None) -> tuple[str, np.ndarray]:
            pool = word_bank[w]
            if prefer_spk is not None:
                same = [item for item in pool if item[0] == prefer_spk]
                if same:
                    spk, raw = same[int(rng.integers(0, len(same)))]
                    used_speakers.add(spk)
                    return spk, _rms_match_and_fade(_load_wav_mono_f32(raw))
            spk, raw = pool[int(rng.integers(0, len(pool)))]
            used_speakers.add(spk)
            return spk, _rms_match_and_fade(_load_wav_mono_f32(raw))

        f_words = (dir_init, dig_init) if init_ord == 0 else (dig_init, dir_init)
        _, w1 = _draw_word(f_words[0])
        _, w2 = _draw_word(f_words[1])

        corr_w_a = dir_a if upd_which == 0 else dig_a
        corr_w_b = dir_b if upd_which == 0 else dig_b
        corr_spk, wc_a = _draw_word(corr_w_a)
        _, wc_b = _draw_word(corr_w_b, prefer_spk=corr_spk)

        wav_a = bed.copy()
        wav_b = bed.copy()

        def _mix_at(buf: np.ndarray, t_s: float, seg: np.ndarray) -> None:
            s0 = int(round(t_s * SR))
            s1 = min(buf.size, s0 + seg.size)
            buf[s0:s1] += seg[: s1 - s0]

        _mix_at(wav_a, t_first_s, w1)
        _mix_at(wav_b, t_first_s, w1)
        _mix_at(wav_a, t_second_s, w2)
        _mix_at(wav_b, t_second_s, w2)
        _mix_at(wav_a, t_corr_s, wc_a)
        _mix_at(wav_b, t_corr_s, wc_b)

        # Mix identical same-domain distractors (50% same speaker as corr_spk!) into BOTH wav_a and wav_b
        for di, dt_s in enumerate(dist_times):
            dw = str(rng.choice(DISTRACTOR_WORDS))
            pref = corr_spk if (di % 2 == 0) else None
            _, wd = _draw_word(dw, prefer_spk=pref)
            _mix_at(wav_a, dt_s, wd)
            _mix_at(wav_b, dt_s, wd)
            if di == 0:
                s0 = int(round(dt_s * SR))
                dist_feats.append(_acoustic_features(wav_a[s0 : s0 + SR]))

        sc0 = int(round(t_corr_s * SR))
        sc1 = min(total_samples, sc0 + SR)
        corr_feats.append(_acoustic_features(wav_a[sc0:sc1]))

        post_diff = float(np.max(np.abs(wav_a[sc1:] - wav_b[sc1:])))
        corr_diff = float(np.max(np.abs(wav_a[sc0:sc1] - wav_b[sc0:sc1])))
        max_post_corr_pcm_diff = max(max_post_corr_pcm_diff, post_diff)
        min_corr_pcm_diff = min(min_corr_pcm_diff, corr_diff)

        delay_ticks = int(np.floor((total_sec - (t_corr_s + 1.0)) * mimi_hz))
        init_gap_ticks = int(np.floor((t_corr_s - t_first_s) * mimi_hz))
        min_delay_ticks = min(min_delay_ticks, delay_ticks)
        min_init_to_corr_ticks = min(min_init_to_corr_ticks, init_gap_ticks)

    # Anti-splice audit: 1D AUROC of each low-level acoustic feature (rms, peak, zcr, spectral centroid)
    X_pos = np.stack(corr_feats, axis=0)
    X_neg = np.stack(dist_feats, axis=0)
    X_all = np.concatenate([X_pos, X_neg], axis=0)
    y_all = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    feat_names = ("rms", "peak", "zcr", "spectral_centroid_hz")
    anti_splice_auroc = {
        fn: round(_binary_auroc(X_all[:, i], y_all), 4) for i, fn in enumerate(feat_names)
    }
    max_dev_from_chance = max(abs(v - 0.5) for v in anti_splice_auroc.values())

    return {
        "n_prototype_episodes": int(n_episodes),
        "total_duration_sec": int(total_sec),
        "mimi_cadence_hz": float(mimi_hz),
        "comparator_layers": int(comparator_layers),
        "comparator_window": int(comparator_window),
        "stacked_receptive_field_ticks": int(receptive_field_ticks),
        "min_delay_after_corr_ticks": int(min_delay_ticks),
        "strictly_beyond_receptive_field": bool(min_delay_ticks > receptive_field_ticks),
        "min_init_to_corr_ticks": int(min_init_to_corr_ticks),
        "overwrite_after_local_window_eviction": bool(min_init_to_corr_ticks > comparator_window),
        "unique_unseen_test_speakers_drawn": len(used_speakers),
        "init_order_frac_digit_first": round(float(np.mean(init_orders)), 4),
        "update_target_frac_digit": round(float(np.mean(which_updates)), 4),
        "post_corr_pcm_max_abs_diff": float(max_post_corr_pcm_diff),
        "corr_window_pcm_min_abs_diff": round(float(min_corr_pcm_diff), 4),
        "anti_splice_lowlevel_acoustic_auroc": anti_splice_auroc,
        "anti_splice_max_deviation_from_0p5": round(float(max_dev_from_chance), 4),
        "gate1_speech_prototype_pass": bool(
            max_post_corr_pcm_diff == 0.0
            and min_corr_pcm_diff > 0.05
            and min_delay_ticks > receptive_field_ticks
            and min_init_to_corr_ticks > comparator_window
            and max_dev_from_chance < 0.15
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Gate 1 SpeechCommands v0.02 Real-Audio Overwrite Prototype Audit.")
    ap.add_argument("--tar-path", type=Path, default=Path("/mnt/slwm-data/SLWM/g13/data/speech/speech_commands_v0.02.tar.gz"))
    ap.add_argument("--n-episodes", type=int, default=64)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    res = run_speech_overwrite_prototype_audit(args.tar_path, n_episodes=args.n_episodes)
    txt = json.dumps(res, indent=2)
    print(txt)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(txt + "\n")


if __name__ == "__main__":
    main()
