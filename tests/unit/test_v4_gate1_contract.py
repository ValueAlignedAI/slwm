"""Unit tests certifying SLWM-v4 Gate 1 160 ms causal heartbeat, 52/68 embodiment schema, seam-free desync, and programmatic single-pass corpus arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from data.v4_heartbeat_contract import (
    ACTION_SLICES,
    MOD_AUDIO,
    MOD_MOTOR,
    MOD_TEXT,
    MOD_VIDEO,
    PROPRIO_SLICES,
    SLWMv4HeartbeatSpec,
    UNIFIED_ACTION_DIM,
    UNIFIED_PROPRIO_DIM,
    build_corpus_spec_for_tier,
    build_interleaved_heartbeat_stream,
    make_same_clip_temporal_desync_pair,
    validate_disjoint_embodiment_slices,
    verify_single_pass_budget,
)


def test_physical_clock_lock_and_disjoint_52_68_humanoid_slices() -> None:
    spec = SLWMv4HeartbeatSpec()
    clk = spec.validate_physical_clock_lock()
    assert clk["exact_integer_clock_lock"] is True
    assert validate_disjoint_embodiment_slices() is True
    assert clk["video_frames_per_heartbeat"] == 4.0
    assert clk["audio_ticks_per_heartbeat"] == 2.0
    assert clk["motor_ticks_per_heartbeat"] == 4.0
    assert clk["unified_action_dim"] == 52
    assert clk["unified_proprio_dim"] == 68
    assert clk["heartbeats_per_120s_2min"] == 750
    assert clk["packets_per_120s_av_stream"] == 750 * 17
    assert clk["packets_per_120s_av_motor_stream"] == 750 * 21

    # Verify full-body Unitree G1 WBT+Inspire (7L+7R arms + 6L+6R hands + 6L+6R legs + 3 waist + 4 base = 45 DoFs)
    # fits strictly on disjoint physical channels with ZERO truncation and ZERO aliasing!
    g1_mask = np.zeros((UNIFIED_ACTION_DIM,), dtype=np.float32)
    g1_mask[ACTION_SLICES["left_arm_7dof"]] = 1.0
    g1_mask[ACTION_SLICES["right_arm_7dof"]] = 1.0
    g1_mask[ACTION_SLICES["left_hand_or_gripper_7dof"].start : ACTION_SLICES["left_hand_or_gripper_7dof"].start + 6] = 1.0
    g1_mask[ACTION_SLICES["right_hand_or_gripper_7dof"].start : ACTION_SLICES["right_hand_or_gripper_7dof"].start + 6] = 1.0
    g1_mask[ACTION_SLICES["left_leg_6dof"]] = 1.0
    g1_mask[ACTION_SLICES["right_leg_6dof"]] = 1.0
    g1_mask[ACTION_SLICES["waist_3dof"]] = 1.0
    g1_mask[ACTION_SLICES["mobile_base_twist_4dof"]] = 1.0
    assert int(np.sum(g1_mask)) == 45


def test_interleaved_stream_exact_causal_availability_and_zero_lookahead() -> None:
    rng = np.random.default_rng(42)
    n_hb = 6  # 0.96 s synchronized AV + Motor + Text (+ Wan 2.1 bootstrap Frame 0 at t=0.00 s)
    f0 = rng.normal(size=(15, 1024)).astype(np.float16)
    vid = rng.normal(size=(n_hb, 15, 1024)).astype(np.float16)
    aud = rng.integers(0, 2048, size=(n_hb * 2, 16), dtype=np.int16)
    act = rng.normal(size=(n_hb * 4, UNIFIED_ACTION_DIM)).astype(np.float32)
    prop = rng.normal(size=(n_hb * 4, UNIFIED_PROPRIO_DIM)).astype(np.float32)

    # Simulate a 7-DoF + 1-gripper single-arm robot with right-wrist 6D force/torque (e.g. RH20T single-arm)
    act_mask = np.zeros((UNIFIED_ACTION_DIM,), dtype=np.float32)
    act_mask[ACTION_SLICES["right_arm_7dof"]] = 1.0
    act_mask[ACTION_SLICES["right_hand_or_gripper_7dof"].start] = 1.0
    prop_mask = np.zeros((UNIFIED_PROPRIO_DIM,), dtype=np.float32)
    prop_mask[ACTION_SLICES["right_arm_7dof"]] = 1.0
    prop_mask[PROPRIO_SLICES["right_wrist_force_torque_6d"]] = 1.0

    stream = build_interleaved_heartbeat_stream(
        video_groups=vid,
        audio_codes=aud,
        action_seq=act,
        proprio_seq=prop,
        action_mask=act_mask,
        proprio_mask=prop_mask,
        text_by_heartbeat={1: b"grasp", 4: b"lift"},
        video_init_frame0=f0,
    )

    # 1. Exact causal timestamp chain capture <= last_sample <= wall_time_s (availability)
    assert np.all(stream["capture_time_s"] <= stream["last_sample_time_s"] + 1e-6)
    assert np.all(stream["last_sample_time_s"] <= stream["wall_time_s"] + 1e-6)
    assert np.all(np.diff(stream["wall_time_s"]) >= -1e-7)
    assert np.all(stream["dt_s"] >= 0.0)
    assert np.all(np.diff(stream["heartbeat_id"]) >= 0)
    assert int(stream["reset_mask"][0]) == 1 and int(np.sum(stream["reset_mask"])) == 1

    # 2. Bootstrap Frame 0 sits strictly at heartbeat_id == -1, wall_time_s == 0.0
    hb_m1 = stream["heartbeat_id"] == -1
    assert int(np.sum(hb_m1)) == 15
    assert np.allclose(stream["wall_time_s"][hb_m1], 0.0, atol=1e-7)

    # 3. Verify EXACT sub-heartbeat chronological order & exact 40 ms / 80 ms / 160 ms clocks!
    # Expected intra-heartbeat order: MOT0(0.04) -> MOT1(0.08) -> AUD0(0.08) -> MOT2(0.12) -> MOT3(0.16) -> VID*15(0.16) -> AUD1(0.16) -> TXT*(0.16)
    for h in range(n_hb):
        idx = np.where(stream["heartbeat_id"] == h)[0]
        mods = stream["modality_id"][idx]
        wts = stream["wall_time_s"][idx]
        dts = stream["dt_s"][idx]
        t0 = h * 0.16

        # Exact motor tick ordering & zero visual/acoustic future lookahead into ANY motor tick!
        mot_pos = np.where(mods == MOD_MOTOR)[0]
        aud_pos = np.where(mods == MOD_AUDIO)[0]
        vid_pos = np.where(mods == MOD_VIDEO)[0]
        assert len(mot_pos) == 4 and len(aud_pos) == 2 and len(vid_pos) == 15

        # Every motor tick MUST precede ALL video patches of the same heartbeat (0 ms visual lookahead!)
        assert int(mot_pos[-1]) < int(vid_pos[0])
        # MOT0 & MOT1 precede AUD0; MOT2 & MOT3 precede AUD1 (0 ms acoustic lookahead!)
        assert int(mot_pos[1]) < int(aud_pos[0]) < int(mot_pos[2]) < int(mot_pos[3]) < int(vid_pos[0]) < int(aud_pos[1])

        # Exact 25 Hz motor wall times [t0+0.04, t0+0.08, t0+0.12, t0+0.16] and exact dt_s == 0.04 s!
        assert np.allclose(wts[mot_pos], [t0 + 0.04, t0 + 0.08, t0 + 0.12, t0 + 0.16], atol=1e-5)
        assert np.allclose(dts[mot_pos], [0.04, 0.04, 0.04, 0.04], atol=1e-5)
        # Exact 12.5 Hz Mimi audio availability times [t0+0.08, t0+0.16]
        assert np.allclose(wts[aud_pos], [t0 + 0.08, t0 + 0.16], atol=1e-5)
        # Exact Wan 2.1 4-frame group availability time [t0+0.16] (capture = t0+0.04)
        assert np.allclose(wts[vid_pos], t0 + 0.16, atol=1e-5)
        assert np.allclose(stream["capture_time_s"][idx[vid_pos]], t0 + 0.04, atol=1e-5)

    # 4. Zero fabricated unobserved DoFs in BOTH action_feat AND proprio_feat!
    assert float(np.max(np.abs(stream["action_feat"][:, act_mask == 0.0]))) == 0.0
    assert float(np.max(np.abs(stream["proprio_feat"][:, prop_mask == 0.0]))) == 0.0
    assert float(np.max(np.abs(stream["action_feat"][:, act_mask == 1.0]))) > 0.0
    assert float(np.max(np.abs(stream["proprio_feat"][:, prop_mask == 1.0]))) > 0.0


def test_input_validation_and_missing_modality_edge_cases() -> None:
    rng = np.random.default_rng(11)
    # 1. Proprio-only stream works cleanly (no action_seq, video, audio)
    prop = rng.normal(size=(12, UNIFIED_PROPRIO_DIM)).astype(np.float32)  # 3 heartbeats
    s_prop = build_interleaved_heartbeat_stream(proprio_seq=prop)
    assert int(np.sum(s_prop["modality_id"] == MOD_MOTOR)) == 12
    assert np.allclose(s_prop["dt_s"][1:], 0.04, atol=1e-5)

    # 2. Shorter proprio_seq safely bounds n_hb instead of raising IndexError
    vid = rng.normal(size=(10, 15, 1024)).astype(np.float16)
    s_trunc = build_interleaved_heartbeat_stream(video_groups=vid, proprio_seq=prop)
    assert int(np.max(s_trunc["heartbeat_id"])) == 2  # 3 complete heartbeats (0, 1, 2)

    # 3. Text-only stream (e.g. FineWeb-Edu) works cleanly
    s_txt = build_interleaved_heartbeat_stream(text_by_heartbeat={0: b"def f(x):", 2: b" return x+1"})
    assert int(np.sum(s_txt["modality_id"] == MOD_TEXT)) == len(b"def f(x):") + len(b" return x+1")
    assert np.all(np.diff(s_txt["wall_time_s"]) >= -1e-7)

    # 4. Non-binary masks and wrong shapes fail loudly!
    bad_mask = np.full((UNIFIED_ACTION_DIM,), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="strictly binary"):
        build_interleaved_heartbeat_stream(proprio_seq=prop, action_mask=bad_mask)
    with pytest.raises(ValueError, match="Expected proprio_mask shape"):
        build_interleaved_heartbeat_stream(proprio_seq=prop, proprio_mask=np.ones((10,), dtype=np.float32))
    with pytest.raises(ValueError, match="At least one active modality"):
        build_interleaved_heartbeat_stream()


def test_same_clip_temporal_desync_pair_seam_free_and_text_stripped() -> None:
    rng = np.random.default_rng(7)
    n_hb_src = 35  # 5.60 s source clip
    vid = rng.normal(size=(n_hb_src, 15, 1024)).astype(np.float16)
    aud = rng.integers(0, 2048, size=(n_hb_src * 2, 16), dtype=np.int16)

    clean, desync, meta = make_same_clip_temporal_desync_pair(
        video_groups=vid,
        audio_codes=aud,
        window_heartbeats=20,
        shift_heartbeats=5,
        start_heartbeat=3,
    )

    # 1. Strictly contiguous interior audio indices -> ZERO circular wraparound splice seam!
    assert meta["zero_wraparound_seam"] is True
    assert np.all(np.diff(meta["clean_audio_src_ticks"]) == 1)
    assert np.all(np.diff(meta["desync_audio_src_ticks"]) == 1)
    assert int(meta["desync_audio_src_ticks"][0] - meta["clean_audio_src_ticks"][0]) == 10  # 5 hb * 2 ticks = 0.80 s

    # 2. Zero audio-text shortcut: MOD_TEXT strictly absent on both windows!
    assert int(np.sum(clean["modality_id"] == MOD_TEXT)) == 0
    assert int(np.sum(desync["modality_id"] == MOD_TEXT)) == 0

    # 3. Zero metadata / timestamp / video shortcut!
    assert np.array_equal(clean["reliability"], desync["reliability"])
    assert np.array_equal(clean["valid_mask"], desync["valid_mask"])
    assert np.array_equal(clean["wall_time_s"], desync["wall_time_s"])
    assert np.array_equal(clean["dt_s"], desync["dt_s"])
    assert np.array_equal(clean["video_feat"], desync["video_feat"])
    assert not np.array_equal(clean["audio_codes"], desync["audio_codes"])


def test_programmatic_corpus_arithmetic_and_two_tier_single_pass_budgets() -> None:
    spec = SLWMv4HeartbeatSpec()
    # 1. Independently certify exact formula consistency for EVERY domain in both tiers (prevents 10x errors!)
    for tier in ("tier1_v4s3", "tier2_v4s4"):
        rows = build_corpus_spec_for_tier(tier)
        for row in rows:
            assert float(row["target_hours"]) <= float(row["verified_source_hours"]) + 1e-9, (
                f"{tier}/{row['name']} target_hours ({row['target_hours']}) exceeds verified_source_hours ({row['verified_source_hours']})"
            )
            hrs = float(row["target_hours"])
            mods = set(row["modalities"])
            hb = hrs * (3600.0 / spec.heartbeat_sec)
            exp_pkts_m = (
                (hb * 15 if "video" in mods else 0.0)
                + (hb * 2 if "audio" in mods else 0.0)
                + (hb * 4 if ("motor_action" in mods or "proprio" in mods or "proprio_force_torque" in mods) else 0.0)
                + (
                    float(row["raw_text_bytes_m"]) * 1e6
                    if "raw_text_bytes_m" in row
                    else hrs * 3600.0 * float(row.get("text_bytes_per_sec", 0.0))
                )
            ) / 1e6
            assert abs(float(row["derived_packets_m"]) - exp_pkts_m) < 1e-3, (
                f"Formula mismatch in {tier}/{row['name']}: got {row['derived_packets_m']}, expected {exp_pkts_m}"
            )

    # 2. Tier 1 (`V4-S3` 45M HPO Sweep, 600M single-pass pos): strictly <= 1.0 epoch, >= 70% signal, inside 200-500 GB fp16!
    t1 = verify_single_pass_budget(600.0, tier="tier1_v4s3")
    assert t1["strict_single_pass_le_1_epoch"] is True
    assert t1["max_planned_epochs_any_domain"] <= 1.0001
    assert t1["multimodal_signal_share_pct"] >= 70.0
    assert 200.0 <= t1["total_preencoded_storage_gb_fp16"] <= 500.0

    # 3. Tier 2 (`V4-S4` 150M Confirmation, 3.0B single-pass pos): strictly <= 1.0 epoch, >= 70% signal, honest multi-hundred-GB/TB storage!
    t2 = verify_single_pass_budget(3000.0, tier="tier2_v4s4")
    assert t2["strict_single_pass_le_1_epoch"] is True
    assert t2["max_planned_epochs_any_domain"] <= 1.0001
    assert t2["multimodal_signal_share_pct"] >= 70.0
    assert t2["total_preencoded_storage_gb_fp16"] > 1000.0  # Honest multi-TB fp16 accounting!
    assert t2["total_preencoded_storage_gb_selected"] < t2["total_preencoded_storage_gb_fp16"]
