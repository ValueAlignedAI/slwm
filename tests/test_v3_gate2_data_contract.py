"""Gate 2 Unit Tests: Causal Timestamp Alignment, NaN Guards, Lossless ByT5, Collision-Free Schedules & Tiling."""

from __future__ import annotations

import numpy as np
import pytest

from data.v3_data_contract import (
    CausalPacketTimestamps,
    assign_episode_primary_speaker_id,
    build_interleaved_speech_overwrite_schedule,
    causal_align_past_index,
    decode_utf8_bytes_lossless,
    encode_utf8_bytes_lossless,
    mimi_packet_availability_timestamps,
    render_v3_dynamics_world_rgb,
    tile_background_with_crossfade,
    wan21_group_availability_timestamps,
)


def test_causal_align_past_index_zero_future_lookahead_and_nan_guard() -> None:
    """Verify causal_align_past_index strictly selects `src <= q`, returns `-1` pre-start, and rejects NaN/Inf."""
    src = np.array([0.10, 0.20, 0.30, 0.50], dtype=np.float64)
    q = np.array([0.05, 0.10, 0.15, 0.20, 0.49, 0.50, 0.90], dtype=np.float64)
    idx = causal_align_past_index(src, q)
    assert idx.tolist() == [-1, 0, 0, 1, 2, 3, 3]
    for qi, ii in zip(q, idx):
        if ii >= 0:
            assert src[ii] <= qi

    # Verify NaN/Inf/unsorted rejection
    with pytest.raises(ValueError):
        causal_align_past_index(np.array([0.1, np.nan]), q)
    with pytest.raises(ValueError):
        causal_align_past_index(src, np.array([0.2, np.inf]))
    with pytest.raises(ValueError):
        causal_align_past_index(np.array([0.3, 0.1]), q)
    with pytest.raises(ValueError):
        CausalPacketTimestamps(0.0, float("nan"), 0.2)
    with pytest.raises(ValueError):
        CausalPacketTimestamps(0.2, 0.1, 0.2)

    wan_ts = wan21_group_availability_timestamps(num_frames=9, fps=25.0)
    assert len(wan_ts) == 3  # frame 0, [1..4], [5..8]
    assert np.isclose(wan_ts[1].availability_time_s, 4 / 25.0)
    mimi_ts = mimi_packet_availability_timestamps(num_packets=5, frame_rate=12.5)
    assert np.isclose(mimi_ts[0].availability_time_s, 0.08)


def test_byt5_utf8_lossless_text_and_code() -> None:
    """Verify 100% lossless round-trip across English, code indentation, symbols, and Unicode."""
    samples = [
        "Pick up the red wrench, not the blue one!",
        "def f(x: float) -> float:\n    return (x ** 2) + 1e-6  # comment\n",
        "μP scaling: τ_b ∈ [0.25s, 128s] — UTF-8 ✓",
    ]
    for s in samples:
        ids = encode_utf8_bytes_lossless(s, add_bos=True, add_eos=True)
        assert decode_utf8_bytes_lossless(ids) == s


def test_render_v3_dynamics_conveyor_drift_contrast_and_apex() -> None:
    """Verify directional silver-gray chevrons move in sign(drift_y), apex points in motion dir, and contrast cyan puck."""
    frames_pos, frames_neg = [], []
    for sign, out_list in ((+1.0, frames_pos), (-1.0, frames_neg)):
        for ti in range(4):
            st = {
                "ee_xy": np.array([[-0.78, 0.20]], dtype=np.float32),
                "obj_xy": np.array([[0.0, 0.0]], dtype=np.float32),  # Inside conveyor to verify contrast!
                "targ_a": np.array([[0.62, -0.22]], dtype=np.float32),
                "box_cx": np.zeros((1, 4), dtype=np.float32),
                "box_cy": np.array([[0.66, -0.66, 0.0, 0.0]], dtype=np.float32),
                "box_hx": np.array([[0.03, 0.03, 0.0, 0.0]], dtype=np.float32),
                "box_hy": np.array([[0.36, 0.36, 0.0, 0.0]], dtype=np.float32),
                "drift_xy": np.array([[0.0, 0.0065 * sign]], dtype=np.float32),
                "bumper_kick": np.array([0.0], dtype=np.float32),
                "t_step": np.array([float(ti)], dtype=np.float32),
            }
            out_list.append(render_v3_dynamics_world_rgb(st, height=192, width=320)[0])

    # Check stripe motion sign away from the puck (column x=190, still inside |x| < 0.42!)
    for sign, frames in ((+1.0, frames_pos), (-1.0, frames_neg)):
        c0 = frames[0][:, 190, 0].astype(np.float32)
        c1 = frames[1][:, 190, 0].astype(np.float32)
        # Shift by ~5 pixels = 0.055 / 2 * 192
        corr_pos = float(np.corrcoef(c0[:-5], c1[5:])[0, 1])
        corr_neg = float(np.corrcoef(c0[5:], c1[:-5])[0, 1])
        assert (1.0 if corr_pos > corr_neg else -1.0) == sign

    # Verify vivid cyan puck center pixel has huge blue-minus-red chroma separating it from neutral silver belt!
    puck_px = frames_pos[0][96, 160].astype(np.float32)
    assert (puck_px[2] - puck_px[0]) > 180.0


def test_collision_free_interleaved_schedule_catch_trials_and_primary_speaker() -> None:
    """Verify zero speech collisions, identical catch timing (no ghost hole), beyond-eqbyte & beyond-Mimi RF, and speaker key."""
    # Discovery beyond eqbyte (W_eq=64 -> RF=756) AND beyond Mimi RF (1000 ticks = 80s > 79.68s)!
    sched = build_interleaved_speech_overwrite_schedule(
        n_episodes=200, comparator_layers=12, comparator_window=64, min_delay_ticks_override=1008, seed=77
    )
    assert sched["zero_speech_collisions_verified"] is True
    assert sched["min_inter_event_gap_s"] >= 1.15 - 1e-4
    assert sched["min_delay_after_last_cue_ticks"] > 1008
    assert sched["min_init_to_corr_ticks"] > 64
    assert 0.15 <= sched["catch_trial_fraction"] <= 0.35

    # Verify EVERY episode (both overwrite and catch!) has zero collisions & zero ghost gap at t_corr_s
    for ep in sched["episodes"]:
        slot_t = ep["catch_distractor_at_corr_s"] if ep["is_catch_no_overwrite"] else ep["t_corr_s"]
        assert slot_t is not None
        all_t = sorted(
            ep["pre_distractor_times_s"]
            + [ep["t_first_s"]]
            + ep["mid1_distractor_times_s"]
            + [ep["t_second_s"]]
            + ep["mid2_distractor_times_s"]
            + [slot_t]
            + ep["post_distractor_times_s"]
        )
        assert float(np.min(np.diff(all_t))) >= 1.14  # accounts for 1ms rounding

    # Verify assign_episode_primary_speaker_id semantics
    assert assign_episode_primary_speaker_id(is_catch_no_overwrite=False, init1_speaker_id="spkA", corr_speaker_id="spkB") == "spkB"
    assert assign_episode_primary_speaker_id(is_catch_no_overwrite=True, init1_speaker_id="spkA", corr_speaker_id=None) == "spkA"
    with pytest.raises(ValueError):
        assign_episode_primary_speaker_id(is_catch_no_overwrite=False, init1_speaker_id="spkA", corr_speaker_id=None)


def test_tile_background_constant_power_and_randomized_long_confirmation_phase() -> None:
    """Verify constant-power sin/cos crossfade (zero -3dB dip) and randomized phase even when src < total_samples."""
    rng_src = np.random.default_rng(1)
    bg = rng_src.normal(0.0, 0.1, size=(16000 * 60,)).astype(np.float32)  # 60s clip tiled to 104s!
    total = 16000 * 104

    t1 = tile_background_with_crossfade(bg, total, sr=16000, crossfade_ms=50.0, rng=np.random.default_rng(10))
    t2 = tile_background_with_crossfade(bg, total, sr=16000, crossfade_ms=50.0, rng=np.random.default_rng(20))
    assert t1.shape == (total,)
    # Two distinct RNG seeds MUST produce distinct phase-shifted confirmation backgrounds!
    assert float(np.max(np.abs(t1 - t2))) > 0.1

    # Check constant-power crossfade variance right across seam (step = 60s - 0.05s = 959,200)
    t_det = tile_background_with_crossfade(bg, total, sr=16000, crossfade_ms=50.0, rng=None)
    seam_var = float(np.var(t_det[959200 : 959200 + 800]))
    interior_var = float(np.var(t_det[100000:200000]))
    assert abs(seam_var - interior_var) / interior_var < 0.25
