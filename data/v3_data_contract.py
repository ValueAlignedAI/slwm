"""SLWM-v3 Gate 2: Connected Data Contract, Strict Causal Timestamps, Pixel-Observable Renderer & Interleaved Streams.

Enforces all Gate 1 & Gate 2 invariants and AGENTS.md §4.1 / §12 Gate 2 contracts:
  1. Strictly causal timestamp alignment (`causal_align_past_index`): rejects NaN/Inf, picks the latest
     sample with `src_ts <= query_ts` (zero future lookahead), returning `-1` when no causal sample exists.
  2. Explicit packet timestamp contract (`CausalPacketTimestamps`: finite `capture <= last_sample <= availability`)
     for 4-frame Wan 2.1 VAE video groups, 80 ms Mimi audio packets, and arriving ByT5 UTF-8 bytes.
  3. Lossless ByT5-style 256-byte text/code interface (`0..255` UTF-8 bytes + `256..263` control symbols).
  4. Pixel-observable `render_v3_dynamics_world_rgb`: renders high-contrast neutral silver-gray directional
     chevrons inside `|x| < 0.42` scrolling at `v_belt = 0.055 * sign(drift_y)` from `t = 0` with apex
     pointing in `sign(drift_y)` (distinct from the cyan puck) so `drift_sign` is 100% pixel-observable.
  5. Collision-free interleaved-distractor + 0-correction catch-trial schedule (`build_interleaved_speech_overwrite_schedule`):
     - Strict `>= 1.15 s` non-overlapping separation across ALL spoken utterances (zero `+3 dB` double-talk spikes).
     - Places a same-domain distractor at `t_corr_s` on catch trials (identical timing distribution, zero ghost hole).
     - Explicit `assign_episode_primary_speaker_id` cluster-key contract.
     - Constant-power `sin/cos` crossfaded background noise tiling (`w_in^2 + w_out^2 == 1.0`) with randomized
       circular phase offset across all durations (including `104 s–190 s` confirmation/eqbyte streams).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

# Reserved control symbol IDs strictly outside UTF-8 byte values [0..255]
CTRL_PAD = 256
CTRL_BOS = 257
CTRL_EOS = 258
CTRL_QUERY = 259
CTRL_MISSING = 260
BYT5_VOCAB_SIZE = 264


@dataclass(frozen=True)
class CausalPacketTimestamps:
    """Strict causal provenance timestamps for any packet entering the SLWM-v3 streaming core."""

    capture_time_s: float
    last_sample_time_s: float
    availability_time_s: float

    def __post_init__(self) -> None:
        for name, val in (
            ("capture_time_s", self.capture_time_s),
            ("last_sample_time_s", self.last_sample_time_s),
            ("availability_time_s", self.availability_time_s),
        ):
            if not math.isfinite(float(val)):
                raise ValueError(f"Non-finite timestamp: {name}={val}")
        if self.last_sample_time_s < self.capture_time_s - 1e-9:
            raise ValueError(
                f"Causal violation: last_sample_time_s ({self.last_sample_time_s}) < capture_time_s ({self.capture_time_s})"
            )
        if self.availability_time_s < self.last_sample_time_s - 1e-9:
            raise ValueError(
                f"Causal violation: availability_time_s ({self.availability_time_s}) < last_sample_time_s ({self.last_sample_time_s})"
            )


def causal_align_past_index(src_timestamps: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    """Return index of latest source timestamp strictly `<= query_ts`, or `-1` if none exists.

    Rejects NaN/Inf timestamps and unsorted source arrays. Replaces legacy `np.searchsorted(src_ts, q)`
    which returned the first index `>= q` (future lookahead).
    """
    src = np.asarray(src_timestamps, dtype=np.float64).reshape(-1)
    q = np.asarray(query_timestamps, dtype=np.float64)
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(q)):
        raise ValueError("Timestamps must be finite (no NaN or Inf) for causal alignment")
    if src.size == 0:
        return np.full(q.shape, -1, dtype=np.int64)
    if np.any(np.diff(src) < 0.0):
        raise ValueError("src_timestamps must be sorted non-decreasing for causal alignment")
    idx = np.searchsorted(src, q, side="right") - 1
    return idx.astype(np.int64)


def wan21_group_availability_timestamps(num_frames: int, fps: float = 25.0, t0_s: float = 0.0) -> list[CausalPacketTimestamps]:
    """Compute exact Wan 2.1 VAE causal group timestamps: frame 0 alone, then 4-frame groups [1..4], [5..8], ...

    Each latent group becomes available ONLY when its final contributing RGB frame has arrived.
    """
    if num_frames < 1:
        return []
    dt = 1.0 / float(fps)
    out = [CausalPacketTimestamps(t0_s, t0_s, t0_s)]
    k = 1
    while k + 3 < num_frames:
        t_cap = t0_s + k * dt
        t_last = t0_s + (k + 3) * dt
        out.append(CausalPacketTimestamps(t_cap, t_last, t_last))
        k += 4
    return out


def mimi_packet_availability_timestamps(num_packets: int, frame_rate: float = 12.5, t0_s: float = 0.0) -> list[CausalPacketTimestamps]:
    """Compute exact Kyutai Mimi 12.5 Hz (80 ms) causal packet timestamps (available strictly at frame end)."""
    dt = 1.0 / float(frame_rate)
    return [
        CausalPacketTimestamps(t0_s + i * dt, t0_s + (i + 1) * dt, t0_s + (i + 1) * dt)
        for i in range(num_packets)
    ]


# ---------------------------------------------------------------------------
# Lossless ByT5-Style 256-Byte Text/Code Interface
# ---------------------------------------------------------------------------


def encode_utf8_bytes_lossless(text: str, *, add_bos: bool = False, add_eos: bool = False) -> np.ndarray:
    """Losslessly encode arbitrary UTF-8 text or code into int32 byte IDs in `[0..255]` (+ optional BOS/EOS)."""
    raw = text.encode("utf-8")
    ids = [CTRL_BOS] if add_bos else []
    ids.extend(raw)
    if add_eos:
        ids.append(CTRL_EOS)
    return np.asarray(ids, dtype=np.int32)


def decode_utf8_bytes_lossless(token_ids: np.ndarray | list[int]) -> str:
    """Decode ByT5 byte IDs back to exact UTF-8 string, stripping reserved control IDs `>= 256`."""
    arr = np.asarray(token_ids, dtype=np.int32).reshape(-1)
    bs = bytes(int(x) for x in arr if 0 <= int(x) <= 255)
    return bs.decode("utf-8", errors="strict")


# ---------------------------------------------------------------------------
# Pixel-Observable RGB Renderer for V3 Dynamics Control Worlds
# ---------------------------------------------------------------------------


def render_v3_dynamics_world_rgb(
    state: dict[str, np.ndarray],
    *,
    height: int = 192,
    width: int = 320,
    show_goal: bool = True,
) -> np.ndarray:
    """Render `uint8[B, H, W, 3]` RGB frames for `v3_dynamics_control_worlds` with pixel-observable conveyor drift.

    Inside `|x| < 0.42` on `conveyor_drift_channel` worlds (`|drift_y| > 1e-5`), renders high-contrast
    neutral silver-gray directional chevrons scrolling smoothly at `v_belt = 0.055 * sign(drift_y)` from
    `t = 0` with apex at `x = 0` pointing strictly in `sign(drift_y)` (visually separated from the cyan puck).
    """
    ee_xy = np.asarray(state["ee_xy"], dtype=np.float32)
    obj_xy = np.asarray(state["obj_xy"], dtype=np.float32)
    targ_a = np.asarray(state["targ_a"], dtype=np.float32)
    box_cx = np.asarray(state["box_cx"], dtype=np.float32)
    box_cy = np.asarray(state["box_cy"], dtype=np.float32)
    box_hx = np.asarray(state["box_hx"], dtype=np.float32)
    box_hy = np.asarray(state["box_hy"], dtype=np.float32)
    drift_xy = np.asarray(state["drift_xy"], dtype=np.float32)
    bumper_kick = np.asarray(state["bumper_kick"], dtype=np.float32)
    t_step = np.asarray(state["t_step"], dtype=np.float32)

    b = ee_xy.shape[0]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    px = xx[None, :, :]  # [1, H, W]
    py = yy[None, :, :]

    img = np.full((b, height, width, 3), 24.0, dtype=np.float32)

    # 1. Pixel-observable directional conveyor belt inside |x| < 0.42 whenever |drift_y| > 1e-5
    drift_y = drift_xy[:, 1]
    has_drift = np.abs(drift_y) > 1e-5
    if np.any(has_drift):
        in_ch = (np.abs(px) < 0.42) & has_drift[:, None, None]
        # Neutral charcoal belt surface + high-contrast silver-gray chevrons (never camouflages cyan puck!)
        img = np.where(in_ch[..., None], np.array([42.0, 44.0, 48.0], dtype=np.float32), img)
        v_belt = np.sign(drift_y) * 0.055  # world units per tick
        # Apex at x=0 leads in sign(drift_y): py = v_belt * t + C - 0.35 * sign(drift_y) * |x|
        y_shift = py - (v_belt * t_step)[:, None, None] + 0.35 * np.sign(drift_y)[:, None, None] * np.abs(px)
        stripe_phase = np.mod(y_shift * 3.5, 1.0)
        on_stripe = in_ch & (stripe_phase < 0.28)
        img = np.where(on_stripe[..., None], np.array([178.0, 182.0, 188.0], dtype=np.float32), img)

    # 2. Obstacle boxes (crimson for static walls, bright orange-red for active moving bumper gates)
    n_boxes = box_cx.shape[1]
    is_bumper = (bumper_kick > 1e-4)[:, None, None]
    for bi in range(n_boxes):
        cx = box_cx[:, bi, None, None]
        cy = box_cy[:, bi, None, None]
        hx = box_hx[:, bi, None, None]
        hy = box_hy[:, bi, None, None]
        in_box = (hx > 1e-4) & (np.abs(px - cx) <= hx) & (np.abs(py - cy) <= hy)
        box_col = np.where(
            is_bumper[..., None],
            np.array([250.0, 110.0, 36.0], dtype=np.float32),
            np.array([218.0, 48.0, 52.0], dtype=np.float32),
        )
        img = np.where(in_box[..., None], box_col, img)

    # 3. Goal disc (emerald)
    if show_goal:
        dg2 = (px - targ_a[:, 0, None, None]) ** 2 + (py - targ_a[:, 1, None, None]) ** 2
        img = np.where((dg2 <= 0.095**2)[..., None], np.array([46.0, 214.0, 96.0], dtype=np.float32), img)

    # 4. Manipulated object (vivid cyan puck — high contrast against neutral charcoal/silver belt)
    do2 = (px - obj_xy[:, 0, None, None]) ** 2 + (py - obj_xy[:, 1, None, None]) ** 2
    img = np.where((do2 <= 0.085**2)[..., None], np.array([24.0, 188.0, 252.0], dtype=np.float32), img)

    # 5. End-effector (amber tip)
    de2 = (px - ee_xy[:, 0, None, None]) ** 2 + (py - ee_xy[:, 1, None, None]) ** 2
    img = np.where((de2 <= 0.055**2)[..., None], np.array([248.0, 196.0, 44.0], dtype=np.float32), img)

    return np.clip(img, 0.0, 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Interleaved-Distractor + 0-Correction Catch-Trial Real-Speech-Overwrite Contract
# ---------------------------------------------------------------------------


def assign_episode_primary_speaker_id(
    *,
    is_catch_no_overwrite: bool,
    init1_speaker_id: str,
    corr_speaker_id: str | None = None,
) -> str:
    """Assign deterministic crossed-bootstrap cluster key (`primary_speaker_id`) per episode.

    Policy: anchors `primary_speaker_id = corr_speaker_id` on overwrite episodes and
    `primary_speaker_id = init1_speaker_id` on 0-correction catch trials.
    """
    if is_catch_no_overwrite:
        if not init1_speaker_id:
            raise ValueError("init1_speaker_id must be non-empty for catch trials")
        return str(init1_speaker_id)
    if not corr_speaker_id:
        raise ValueError("corr_speaker_id must be non-empty for overwrite episodes")
    return str(corr_speaker_id)


def tile_background_with_crossfade(
    bg_src: np.ndarray,
    total_samples: int,
    *,
    sr: int = 16000,
    crossfade_ms: float = 50.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Tile a background noise clip to arbitrary `total_samples` with randomized circular phase and constant-power crossfade.

    Uses equal-power `sin(pi*u/2)` / `cos(pi*u/2)` crossfades (`w_in^2 + w_out^2 == 1.0`) so uncorrelated
    ambient noise exhibits zero `-3 dB` seam energy dip, and applies a random circular start offset via `rng`
    regardless of whether `bg_src` is shorter or longer than `total_samples`.
    """
    src = np.asarray(bg_src, dtype=np.float32).reshape(-1)
    if src.size == 0:
        return np.zeros((total_samples,), dtype=np.float32)

    n_xf = min(int(round(sr * crossfade_ms / 1000.0)), src.size // 4)
    if n_xf < 2:
        step = src.size
        need = total_samples + (step if rng is not None else 0)
        reps = (need // src.size) + 2
        tiled = np.tile(src, reps)
    else:
        u = np.linspace(0.0, 1.0, n_xf, dtype=np.float32)
        fade_in = np.sin(0.5 * np.pi * u)
        fade_out = np.cos(0.5 * np.pi * u)
        step = src.size - n_xf
        # Smooth EVERY true end-to-start boundary with constant-power sin/cos FIRST, then slice at random offset!
        need = total_samples + (step if rng is not None else 0)
        chunks: list[np.ndarray] = [src[:step].copy()]
        cur_tail = src[step:].copy()
        tot = step
        while tot < need:
            overlap = cur_tail * fade_out + src[:n_xf] * fade_in
            chunks.append(overlap)
            chunks.append(src[n_xf:step].copy())
            cur_tail = src[step:].copy()
            tot += step
        tiled = np.concatenate(chunks, axis=0)

    shift = int(rng.integers(0, max(1, step))) if (rng is not None and step > 1) else 0
    return tiled[shift : shift + total_samples].copy()


def _sample_non_overlapping_times(
    rng: np.random.Generator,
    n_events: int,
    t_start: float,
    t_end: float,
    min_gap_s: float = 1.15,
) -> list[float]:
    """Sample `n_events` sorted event start times inside `[t_start, t_end]` with strict `>= min_gap_s` spacing."""
    if n_events <= 0:
        return []
    span = t_end - t_start
    slack = span - (n_events - 1) * min_gap_s
    if slack < 0.0:
        raise ValueError(
            f"Insufficient interval [{t_start:.3f}, {t_end:.3f}] (span={span:.3f}s) for {n_events} events at gap={min_gap_s}s"
        )
    # Order-statistics uniform slack placement guarantees exact non-overlap with zero collisions!
    u = np.sort(rng.uniform(0.0, slack, size=(n_events,)))
    return [float(t_start + u[i] + i * min_gap_s) for i in range(n_events)]


def build_interleaved_speech_overwrite_schedule(
    *,
    n_episodes: int = 128,
    mimi_hz: float = 12.5,
    comparator_layers: int = 12,
    comparator_window: int = 32,
    min_delay_ticks_override: int | None = None,
    catch_trial_prob: float = 0.25,
    min_event_gap_s: float = 1.15,
    seed: int = 32001,
) -> dict[str, Any]:
    """Generate collision-free interleaved-distractor + 0-correction catch-trial schedules.

    Guarantees:
      - Zero speech-on-speech collisions (`min(diff(all_sorted_events)) >= min_event_gap_s > 1.0 s`)
        eliminating any `+3 dB` double-talk loudness shortcut in distractor segments.
      - Interleaves distractors across ALL 4 temporal segments (`pre`, `mid1`, `mid2`, `post`).
      - On `catch_trial_prob` (`25%`) `0`-correction catch episodes, places a same-domain distractor at
        `t_corr_s` (`catch_distractor_at_corr_s`) so catch & overwrite word-timing distributions are identical.
      - Enforces `t_corr - t_first > comparator_window` ticks and
        `delay_after_last_cue > max(L*(W-1), min_delay_ticks_override)` ticks.
    """
    rng = np.random.default_rng(seed)
    rf_ticks = comparator_layers * (comparator_window - 1)
    req_delay_ticks = max(rf_ticks, int(min_delay_ticks_override or 0))
    w_sec = comparator_window / mimi_hz
    rf_sec = req_delay_ticks / mimi_hz

    # Conservative upper bound on latest possible cue time + full retention window > req_delay_ticks
    max_cue_end_s = 3.2 + 1.5 + 3.2 + 1.5 + max(w_sec + 3.5, 5.5) + 2.5
    total_sec = int(math.ceil(max_cue_end_s + rf_sec + 4.0))

    episodes: list[dict[str, Any]] = []
    min_observed_gap_s = float("inf")

    for ei in range(n_episodes):
        is_catch = bool(rng.random() < catch_trial_prob)
        init_order = int(rng.integers(0, 2))  # 0: Direction first; 1: Digit first
        upd_which = -1 if is_catch else int(rng.integers(0, 2))

        cursor = float(rng.uniform(0.25, 0.75))

        # Segment 0: 0..2 non-overlapping pre-init distractors
        n_pre = int(rng.integers(0, 3))
        pre_dist: list[float] = []
        for _ in range(n_pre):
            pre_dist.append(cursor)
            cursor += min_event_gap_s + float(rng.uniform(0.05, 0.45))

        # Fact 1 arrival
        t_first_s = cursor
        cursor += min_event_gap_s + float(rng.uniform(0.05, 0.45))

        # Segment 1: 0..2 non-overlapping distractors between t_first and t_second
        n_mid1 = int(rng.integers(0, 3))
        mid1_dist: list[float] = []
        for _ in range(n_mid1):
            mid1_dist.append(cursor)
            cursor += min_event_gap_s + float(rng.uniform(0.05, 0.45))

        # Fact 2 arrival
        t_second_s = cursor
        cursor += min_event_gap_s + float(rng.uniform(0.05, 0.45))

        # Segment 2: 1..2 non-overlapping distractors between t_second and t_corr
        n_mid2 = int(rng.integers(1, 3))
        mid2_dist: list[float] = []
        for _ in range(n_mid2):
            mid2_dist.append(cursor)
            cursor += min_event_gap_s + float(rng.uniform(0.05, 0.45))

        # Correction slot: strictly > W ticks after t_first AND >= min_event_gap_s after last mid2 distractor!
        min_corr_s = max(cursor, t_first_s + w_sec + 0.65)
        t_corr_slot_s = min_corr_s + float(rng.uniform(0.0, 0.6))

        # Segment 3: 6 strictly non-overlapping post-correction distractors across the beyond-stack retention window
        anchor_end = t_corr_slot_s + 1.0
        post_dist = _sample_non_overlapping_times(
            rng, n_events=6, t_start=t_corr_slot_s + min_event_gap_s + 0.25, t_end=total_sec - 1.5, min_gap_s=min_event_gap_s
        )

        # Verify zero collisions across ALL spoken utterances in this episode!
        all_events = sorted(pre_dist + [t_first_s] + mid1_dist + [t_second_s] + mid2_dist + [t_corr_slot_s] + post_dist)
        ep_min_gap = float(np.min(np.diff(all_events)))
        min_observed_gap_s = min(min_observed_gap_s, ep_min_gap)

        episodes.append(
            {
                "episode_idx": ei,
                "is_catch_no_overwrite": is_catch,
                "init_order": init_order,
                "update_which": upd_which,
                "t_first_s": round(t_first_s, 3),
                "t_second_s": round(t_second_s, 3),
                "t_corr_s": None if is_catch else round(t_corr_slot_s, 3),
                "catch_distractor_at_corr_s": round(t_corr_slot_s, 3) if is_catch else None,
                "pre_distractor_times_s": [round(x, 3) for x in pre_dist],
                "mid1_distractor_times_s": [round(x, 3) for x in mid1_dist],
                "mid2_distractor_times_s": [round(x, 3) for x in mid2_dist],
                "post_distractor_times_s": [round(x, 3) for x in post_dist],
                "delay_after_last_cue_ticks": int(math.floor((total_sec - anchor_end) * mimi_hz)),
                "init_to_corr_ticks": None if is_catch else int(math.floor((t_corr_slot_s - t_first_s) * mimi_hz)),
                "min_inter_event_gap_s": round(ep_min_gap, 4),
            }
        )

    non_catch = [e for e in episodes if not e["is_catch_no_overwrite"]]
    return {
        "n_episodes": n_episodes,
        "total_duration_sec": total_sec,
        "mimi_hz": mimi_hz,
        "comparator_layers": comparator_layers,
        "comparator_window": comparator_window,
        "stacked_receptive_field_ticks": rf_ticks,
        "catch_trial_fraction": round(sum(1 for e in episodes if e["is_catch_no_overwrite"]) / n_episodes, 4),
        "frac_with_pre_init_distractors": round(sum(1 for e in episodes if len(e["pre_distractor_times_s"]) > 0) / n_episodes, 4),
        "frac_with_mid1_distractors": round(sum(1 for e in episodes if len(e["mid1_distractor_times_s"]) > 0) / n_episodes, 4),
        "frac_with_mid2_distractors": round(sum(1 for e in episodes if len(e["mid2_distractor_times_s"]) > 0) / n_episodes, 4),
        "min_inter_event_gap_s": round(min_observed_gap_s, 4),
        "zero_speech_collisions_verified": bool(min_observed_gap_s >= 1.05),
        "min_delay_after_last_cue_ticks": min(e["delay_after_last_cue_ticks"] for e in episodes),
        "min_init_to_corr_ticks": min(e["init_to_corr_ticks"] for e in non_catch) if non_catch else 0,
        "episodes": episodes,
    }
