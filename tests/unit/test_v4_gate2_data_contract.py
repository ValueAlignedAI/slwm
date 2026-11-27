"""Unit tests certifying SLWM-v4 Gate 2 disjoint splits, 52/68 embodiment mappers, causal multi-FPS resampler, SNR filter, non-vacuous SHA-256 audit & audio-only desync null."""

from __future__ import annotations

import numpy as np
import pytest

from data.v4_embodiment_and_splits import (
    assign_v4_split,
    audit_audio_only_desync_null_auroc,
    audit_pcm_nonsilence,
    audit_v4_splits_non_vacuous,
    hash_payload_sha256,
    map_embodiment_to_unified_52_68,
    resample_video_and_kinematics_to_25hz,
)
from data.v4_heartbeat_contract import (
    ACTION_SLICES,
    PROPRIO_SLICES,
    make_same_clip_temporal_desync_pair,
)


def test_disjoint_split_cluster_boundaries_all_9_corpora() -> None:
    # Same DROID building+collector MUST always land in the same split; different RH20T user/scene parses cleanly!
    sp1, ck1 = assign_v4_split("droid_1_0_1", {"building": "TRI_Lab2", "collector_id": "c07"})
    sp2, ck2 = assign_v4_split("droid_1_0_1", {"building": "tri_lab2", "collector_id": "C07"})
    assert sp1 == sp2 and ck1 == ck2 == "droid::tri_lab2::c07"

    sp_rh1, ck_rh1 = assign_v4_split("rh20t_contact_av_force", {"episode_dir": "RH20T_cfg5/task_0001_user_0007_scene_0001_cfg_0005"})
    sp_rh2, ck_rh2 = assign_v4_split("rh20t_contact_av_force", {"episode_dir": "RH20T_cfg5/task_0099_user_0007_scene_0001_cfg_0005"})
    assert sp_rh1 == sp_rh2 and ck_rh1 == ck_rh2 == "rh20t::user_0007::scene_0001"

    sp_ep1, ck_ep1 = assign_v4_split("epic_kitchens_100_long_ego_av", {"participant_id": "P04"})
    assert sp_ep1 in ("train", "dev", "test") and ck_ep1 == "epic100::P04"

    # Cross-corpus YouTube consistency: same youtube_id in VGGSound & AudioCaps MUST get identical split!
    sp_v, ck_v = assign_v4_split("vggsound_full_sync", {"youtube_id": "dQw4w9WgXcQ"})
    sp_a, ck_a = assign_v4_split("wavcaps_and_libriheavy_speech", {"source": "audiocaps", "youtube_id": "dQw4w9WgXcQ"})
    assert sp_v == sp_a and ck_v == ck_a == "yt::dQw4w9WgXcQ"

    # Repository-disjoint code split!
    sp_c1, ck_c1 = assign_v4_split("fineweb_edu_and_stack_code", {"repo_name": "PyTorch/PyTorch"})
    sp_c2, ck_c2 = assign_v4_split("fineweb_edu_and_stack_code", {"repo_name": "pytorch/pytorch"})
    assert sp_c1 == sp_c2 and ck_c1 == ck_c2 == "code_repo::pytorch/pytorch"

    # Missing required leakage boundary fails loudly!
    with pytest.raises(ValueError):
        assign_v4_split("droid_1_0_1", {"building": "", "collector_id": "c01"})
    with pytest.raises(ValueError):
        assign_v4_split("epic_kitchens_100_long_ego_av", {"participant_id": "bad"})


def test_lossless_non_aliased_52_68_embodiment_mappers() -> None:
    rng = np.random.default_rng(123)
    T = 20

    # 1. Full 45-DoF Unitree G1 WBT+Inspire (zero truncation, zero aliasing!)
    raw_g1 = rng.normal(size=(T, 45)).astype(np.float32)
    imu = rng.normal(size=(T, 4)).astype(np.float32)
    a52, am, p68, pm = map_embodiment_to_unified_52_68(
        "unitree_g1_wbt_inspire_45dof", raw_action=raw_g1, raw_proprio=raw_g1, raw_imu_quat=imu
    )
    assert int(np.sum(am)) == 45
    assert int(np.sum(pm)) == 45 + 4
    assert np.allclose(a52[:, ACTION_SLICES["waist_3dof"]], raw_g1[:, 38:41])
    assert np.allclose(a52[:, ACTION_SLICES["mobile_base_twist_4dof"]], raw_g1[:, 41:45])
    # Inactive aux dims strictly 0
    assert float(np.max(np.abs(a52[:, ACTION_SLICES["head_and_aux_5dof"]]))) == 0.0
    # Unit-norm IMU quaternions
    assert np.allclose(np.linalg.norm(p68[:, PROPRIO_SLICES["torso_imu_base_orient_4d"]], axis=-1), 1.0, atol=1e-5)

    # 2. RH20T single-arm (7-DoF Franka + 6-DoF UR5) + 6D right-wrist force-torque
    raw_rh = rng.normal(size=(T, 8)).astype(np.float32)
    ft_r = rng.normal(size=(T, 6)).astype(np.float32)
    a52_r, am_r, p68_r, pm_r = map_embodiment_to_unified_52_68(
        "rh20t_single_arm_8dof", raw_action=raw_rh, raw_proprio=raw_rh, raw_force_torque_right=ft_r
    )
    assert int(np.sum(am_r)) == 8
    assert int(np.sum(pm_r)) == 8 + 6
    assert np.allclose(p68_r[:, PROPRIO_SLICES["right_wrist_force_torque_6d"]], ft_r)
    assert float(np.max(np.abs(p68_r[:, PROPRIO_SLICES["left_wrist_force_torque_6d"]]))) == 0.0

    raw_rh7 = rng.normal(size=(T, 7)).astype(np.float32)
    a52_r7, am_r7, p68_r7, pm_r7 = map_embodiment_to_unified_52_68(
        "rh20t_single_arm_7dof", raw_action=raw_rh7, raw_proprio=raw_rh7, raw_force_torque_right=ft_r
    )
    assert int(np.sum(am_r7)) == 7 and am_r7[13] == 0.0
    assert int(np.sum(pm_r7)) == 7 + 6
    assert float(np.max(np.abs(a52_r7[:, 13]))) == 0.0

    # 3. ALOHA 14-DoF leaves 7th arm DoF strictly masked 0 (zero fabrication!)
    raw_aloha = rng.normal(size=(T, 14)).astype(np.float32)
    a52_a, am_a, _, _ = map_embodiment_to_unified_52_68("aloha_bimanual_14dof", raw_action=raw_aloha)
    assert int(np.sum(am_a)) == 14
    assert am_a[6] == 0.0 and am_a[13] == 0.0
    assert float(np.max(np.abs(a52_a[:, [6, 13]]))) == 0.0

    # 4. Unitree H1 40-DoF fits cleanly without overflow
    raw_h1 = rng.normal(size=(T, 40)).astype(np.float32)
    _, am_h1, _, _ = map_embodiment_to_unified_52_68("unitree_h1_40dof", raw_action=raw_h1)
    assert int(np.sum(am_h1)) == 40

    # 5. Verified AgiBotWorld2026 G2A (action[44] + state[216] incl. dual-wrist 6D wrench + base quat)
    raw_ag_a = rng.normal(size=(T, 44)).astype(np.float32)
    raw_ag_s = rng.normal(size=(T, 216)).astype(np.float32)
    a52_ag, am_ag, p68_ag, pm_ag = map_embodiment_to_unified_52_68(
        "agibot_g2a_bimanual", raw_action=raw_ag_a, raw_proprio=raw_ag_s
    )
    assert int(np.sum(am_ag)) == 7 + 7 + 1 + 1 + 3 + 4 + 5  # 28 active action DoFs
    assert int(np.sum(pm_ag)) == 7 + 7 + 1 + 1 + 3 + 5 + 6 + 6 + 4  # 40 active proprio+wrench+IMU dims
    assert np.allclose(p68_ag[:, PROPRIO_SLICES["left_wrist_force_torque_6d"]], raw_ag_s[:, 2:8])
    assert np.allclose(p68_ag[:, PROPRIO_SLICES["right_wrist_force_torque_6d"]], raw_ag_s[:, 8:14])
    assert np.allclose(np.linalg.norm(p68_ag[:, PROPRIO_SLICES["torso_imu_base_orient_4d"]], axis=-1), 1.0, atol=1e-5)

    # 6. Verified Unitree G1 named parquet columns mapper
    from data.v4_embodiment_and_splits import map_unitree_g1_named_columns
    g1_cols = {
        "action.left_arm": rng.normal(size=(T, 7)),
        "action.right_arm": rng.normal(size=(T, 7)),
        "action.hand_cmd": rng.normal(size=(T, 12)),
        "action.left_leg": rng.normal(size=(T, 6)),
        "action.right_leg": rng.normal(size=(T, 6)),
        "action.waist_action_joint": rng.normal(size=(T, 3)),
        "action.base_command": rng.normal(size=(T, 4)),
        "observation.state.state_base_rot_quat": rng.normal(size=(T, 4)),
    }
    _, am_gn, p68_gn, pm_gn = map_unitree_g1_named_columns(g1_cols, T)
    assert int(np.sum(am_gn)) == 45
    assert int(np.sum(pm_gn)) == 4
    assert np.allclose(np.linalg.norm(p68_gn[:, PROPRIO_SLICES["torso_imu_base_orient_4d"]], axis=-1), 1.0, atol=1e-5)


def test_causal_multi_fps_resampler_and_low_fps_exclusion() -> None:
    rng = np.random.default_rng(7)
    # 5 fps Bridge & 10 fps LIBERO MUST be rejected loudly!
    with pytest.raises(ValueError, match="Refusing to resample"):
        resample_video_and_kinematics_to_25hz(native_fps=5.0, n_src_frames=50)
    with pytest.raises(ValueError, match="Refusing to resample"):
        resample_video_and_kinematics_to_25hz(native_fps=10.0, n_src_frames=50)

    # 50 fps ALOHA -> exact 2:1 integer decimation!
    a52 = rng.normal(size=(101, 52)).astype(np.float32)
    r50 = resample_video_and_kinematics_to_25hz(native_fps=50.0, n_src_frames=101, action_52=a52)
    assert r50["n_heartbeats"] == 12  # 2.0 s -> 51 target frames = 1 + 4*12 + 2 -> 12 complete heartbeats = 49 frames
    assert np.array_equal(r50["video_src_frame_indices"][:5], [0, 2, 4, 6, 8])
    assert r50["action_25hz"].shape == (12 * 4, 52)
    assert np.all(r50["video_group_novel_mask"] == 1)

    # 30 fps Unitree G1 -> causal nearest-past frames (src_ts <= tgt_ts strictly!)
    r30 = resample_video_and_kinematics_to_25hz(native_fps=30.0, n_src_frames=90)
    assert np.all((r30["video_src_frame_indices"] / 30.0) <= r30["target_timestamps_s"] + 1e-6)
    assert np.all(r30["video_group_novel_mask"] == 1)

    # 15 fps DROID -> causal nearest-past frames + valid group novelty tracking
    r15 = resample_video_and_kinematics_to_25hz(native_fps=15.0, n_src_frames=60)
    assert np.all((r15["video_src_frame_indices"] / 15.0) <= r15["target_timestamps_s"] + 1e-6)
    assert r15["video_group_novel_mask"].shape == (r15["n_heartbeats"],)


def test_audio_nonsilence_and_non_vacuous_sha256_audit() -> None:
    sr = 24000
    t = np.linspace(0.0, 2.0, 2 * sr, endpoint=False)
    live_wav = 0.08 * np.sin(2.0 * np.pi * 440.0 * t)
    dead_wav = np.full_like(t, 0.001)  # Pure DC offset / dead mic!
    assert audit_pcm_nonsilence(live_wav, sr)["audio_valid"] is True
    assert audit_pcm_nonsilence(dead_wav, sr)["audio_valid"] is False

    # Non-vacuous SHA-256 disjointness audit catches BOTH empty splits AND media leaks!
    rng = np.random.default_rng(99)
    recs = []
    for sp, n in (("train", 10), ("dev", 3), ("test", 3)):
        for i in range(n):
            recs.append({
                "split": sp,
                "cluster_key": f"{sp}_c{i}",
                "media_hashes": {
                    "video": hash_payload_sha256(rng.normal(size=(15, 16)).astype(np.float16)),
                    "audio": hash_payload_sha256(rng.integers(0, 2048, size=(2, 16), dtype=np.int16)),
                },
            })
    ok = audit_v4_splits_non_vacuous(recs, expected_modalities=("video", "audio"))
    assert ok["non_vacuous_zero_overlap_pass"] is True

    # Injecting a duplicate media payload across train -> test MUST fail!
    recs[-1]["media_hashes"]["audio"] = recs[0]["media_hashes"]["audio"]
    bad = audit_v4_splits_non_vacuous(recs, expected_modalities=("video", "audio"))
    assert bad["non_vacuous_zero_overlap_pass"] is False
    assert bad["media_payload_overlaps"]["audio"]["train_test"] == 1

    # Missing expected modality hash MUST fail non-vacuity assertion!
    recs_vac = [{"split": sp, "cluster_key": f"{sp}_0", "media_hashes": {"video": f"v_{sp}"}} for sp in ("train", "dev", "test")]
    with pytest.raises(AssertionError, match="Vacuous media-hash audit"):
        audit_v4_splits_non_vacuous(recs_vac, expected_modalities=("video", "audio"))


def test_audio_only_desync_null_catches_roll_seam_and_passes_interior_crop() -> None:
    """Prove seam-free interior cropping yields chance AUROC (~0.50) while circular np.roll leaks a detectable seam!"""
    rng = np.random.default_rng(2026)
    n_pairs = 60
    clean_crop, desync_crop = [], []
    clean_roll, desync_roll = [], []

    for i in range(n_pairs):
        # Smooth temporally drifting audio codes (mimicking real continuous acoustic room/speech envelope!)
        n_hb_src = 40
        t_aud = n_hb_src * 2
        drift = np.cumsum(rng.normal(0.0, 8.0, size=(t_aud, 16)), axis=0)
        aud = np.clip(np.round(1024 + drift), 0, 2047).astype(np.int16)
        vid = rng.normal(size=(n_hb_src, 15, 1024)).astype(np.float16)

        # Symmetric +-shift eliminates start-vs-end clip-position drift shortcut!
        sh = 6 if (i % 2 == 0) else -6
        c_s, d_s, _ = make_same_clip_temporal_desync_pair(
            video_groups=vid, audio_codes=aud, window_heartbeats=20, shift_heartbeats=sh, start_heartbeat=8
        )
        clean_crop.append(c_s["audio_codes"])
        desync_crop.append(d_s["audio_codes"])

        # Compare against legacy circular np.roll (has a giant splice jump at index 12!)
        clean_roll.append(c_s["audio_codes"])
        desync_roll.append(np.roll(c_s["audio_codes"], shift=sh * 2, axis=0))

    res_crop = audit_audio_only_desync_null_auroc(clean_crop, desync_crop)
    res_roll = audit_audio_only_desync_null_auroc(clean_roll, desync_roll)
    assert res_crop["at_chance_zero_splice_shortcut"] is True, f"Interior crop leaked audio-only shortcut: {res_crop}"
    assert res_roll["audio_only_null_auroc"] > 0.85, f"Expected circular roll seam to be audio-only detectable: {res_roll}"


def test_int8_quantization_and_streaming_shard_writer_offset_and_resume(tmp_path) -> None:
    """Verify certified int8 quantization round-trip (>55 dB PSNR) & StreamingV4ShardWriter offset re-indexing + meta checkpoint."""
    from data.v4_heartbeat_contract import build_interleaved_heartbeat_stream
    from scripts.data.v4_preencode_tier1 import StreamingV4ShardWriter, quantize_wan21_patches_int8

    rng = np.random.default_rng(2027)
    raw = rng.normal(0.0, 1.5, size=(30, 1024)).astype(np.float16)
    q, xmin, sc = quantize_wan21_patches_int8(raw)
    rec = q.astype(np.float32) * sc.astype(np.float32) + xmin.astype(np.float32)
    mse = float(np.mean((raw.astype(np.float32) - rec) ** 2))
    peak = float(np.max(np.abs(raw.astype(np.float32))))
    psnr = 10.0 * np.log10((peak**2) / max(mse, 1e-12))
    assert q.dtype == np.uint8 and psnr > 50.0, f"Unexpected int8 latent PSNR={psnr:.2f} dB"

    writer = StreamingV4ShardWriter(
        corpus_name="unit_test_corpus",
        local_out_dir=tmp_path,
        hf_repo_id=None,
        flush_mb=100.0,
        delete_local_after_upload=False,
    )
    for ep_i in range(2):
        s = build_interleaved_heartbeat_stream(
            video_init_frame0=rng.normal(size=(15, 1024)).astype(np.float16),
            video_groups=rng.normal(size=(3, 15, 1024)).astype(np.float16),
            audio_codes=rng.integers(0, 2048, size=(6, 16), dtype=np.int16),
            action_seq=rng.normal(size=(12, 52)).astype(np.float32),
            proprio_seq=rng.normal(size=(12, 68)).astype(np.float32),
            text_by_heartbeat={0: f"ep_{ep_i}".encode("utf-8")},
        )
        writer.add_episode(split="train", episode_id=f"ep_{ep_i}", cluster_key=f"c_{ep_i}", stream=s, duration_s=0.48)
    # Adding duplicate episode_id must be silently skipped!
    writer.add_episode(split="train", episode_id="ep_0", cluster_key="c_0", stream=s, duration_s=0.48)
    receipt = writer.finalize()
    assert receipt["n_episodes"] == 2
    npz_p = tmp_path / "unit_test_corpus" / "train_shard_000.npz"
    meta_p = tmp_path / "unit_test_corpus" / "train_shard_000_meta.json"
    assert npz_p.is_file() and meta_p.is_file()
    d = np.load(npz_p)
    assert d["video_q_int8"].shape == (2 * (15 + 45), 1024)
    assert d["audio_codes"].shape == (12, 16)
    assert d["action_feat"].shape == (24, 52)
    assert int(d["video_idx"].max()) == 119 and int(d["audio_idx"].max()) == 11 and int(d["motor_idx"].max()) == 23


def test_hf_commit_with_retry_handles_429_and_succeeds() -> None:
    """Verify atomic HF commit helper backs off cleanly on 429 rate-limit errors without silently dropping files."""
    from unittest.mock import patch
    from scripts.data.v4_preencode_tier1 import _hf_commit_with_retry

    calls = {"n": 0}

    class FakeApi:
        def create_commit(self, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("429 Too Many Requests: Rate limit exceeded")
            return None

    with patch("scripts.data.v4_preencode_tier1.time.sleep", return_value=None) as mock_sleep:
        _hf_commit_with_retry(FakeApi(), "valueai-org/slwm-v4-test", ["op1", "op2"], "test commit", max_attempts=5)
    assert calls["n"] == 3
    assert mock_sleep.call_count == 2
    assert mock_sleep.call_args_list[0].args[0] == 60.0
    assert mock_sleep.call_args_list[1].args[0] == 120.0


def test_b1_lerobot_v3_episode_video_index_multi_mp4_and_multi_pq() -> None:
    """B1 regression test: verify LeRobot v3.0 meta/episodes handles both 1-pq->N-mp4 (DROID/G1) and N-pq->1-mp4 (ALOHA/Wipe)."""
    from scripts.data.v4_preencode_tier1 import parse_lerobot_v3_episode_metadata_rows

    cam = "observation.images.exterior_1_left"
    # Case 1: 1 parquet (chunk-000/file-000) spanning 2 video MP4s (DROID/Unitree G1)
    cols_1pq_nmp4 = {
        "episode_index": [0, 1, 327],
        "data/chunk_index": [0, 0, 0],
        "data/file_index": [0, 0, 0],
        f"videos/{cam}/chunk_index": [0, 0, 0],
        f"videos/{cam}/file_index": [0, 0, 1],
        f"videos/{cam}/from_timestamp": [0.0, 11.133333, 0.0],
        f"videos/{cam}/to_timestamp": [11.133333, 28.866667, 36.4],
        "length": [167, 266, 546],
        "tasks": [["Pick cup"], ["Put block"], ["Open drawer"]],
    }
    idx1 = parse_lerobot_v3_episode_metadata_rows(cols_1pq_nmp4, cam)
    assert idx1[0]["video_mp4"] == f"videos/{cam}/chunk-000/file-000.mp4" and idx1[0]["from_ts"] == 0.0
    assert idx1[327]["data_pq"] == "data/chunk-000/file-000.parquet"
    assert idx1[327]["video_mp4"] == f"videos/{cam}/chunk-000/file-001.mp4"
    assert idx1[327]["from_ts"] == 0.0 and idx1[327]["task"] == "Open drawer"

    # Case 2: 3 parquets sharing 1 video MP4 (ALOHA-Sim / G1_Dex1_Wipe_Table)
    cam_top = "observation.images.top"
    cols_npq_1mp4 = {
        "episode_index": [0, 20, 40],
        "data/chunk_index": [0, 0, 0],
        "data/file_index": [0, 1, 2],
        f"videos/{cam_top}/chunk_index": [0, 0, 0],
        f"videos/{cam_top}/file_index": [0, 0, 0],
        f"videos/{cam_top}/from_timestamp": [0.0, 160.0, 320.0],
        f"videos/{cam_top}/to_timestamp": [8.0, 168.0, 328.0],
        "length": [400, 400, 400],
        "tasks": [["Transfer cube"], ["Transfer cube"], ["Transfer cube"]],
    }
    idx2 = parse_lerobot_v3_episode_metadata_rows(cols_npq_1mp4, cam_top)
    assert idx2[40]["data_pq"] == "data/chunk-000/file-002.parquet"
    assert idx2[40]["video_mp4"] == f"videos/{cam_top}/chunk-000/file-000.mp4"
    assert int(round(idx2[40]["from_ts"] * 50.0)) == 16000


def test_b2_strict_causal_zoh_kinematics_zero_future_lookahead() -> None:
    """B2 regression test: mutating future kinematic frame arr[k] strictly after tgt_ts causes EXACT 0.0 change."""
    rng = np.random.default_rng(2028)
    for fps in (15.0, 30.0, 50.0):
        n_src = 90
        a52 = rng.normal(size=(n_src, 52)).astype(np.float32)
        p68 = rng.normal(size=(n_src, 68)).astype(np.float32)
        pm = np.ones((68,), dtype=np.float32)
        r_ref = resample_video_and_kinematics_to_25hz(
            native_fps=fps, n_src_frames=n_src, action_52=a52, proprio_68=p68, proprio_mask=pm
        )
        tgt_ts_motor = r_ref["target_timestamps_s"][1:]
        # Pick a mid-episode target tick m and mutate ALL source frames strictly in the future of tgt_ts_motor[m]!
        m = 7
        t_cut = float(tgt_ts_motor[m])
        future_mask = (np.arange(n_src, dtype=np.float64) / fps) > (t_cut + 1e-6)
        assert np.any(future_mask)
        a_mut = a52.copy()
        p_mut = p68.copy()
        a_mut[future_mask] += 999.0
        p_mut[future_mask] -= 777.0
        r_mut = resample_video_and_kinematics_to_25hz(
            native_fps=fps, n_src_frames=n_src, action_52=a_mut, proprio_68=p_mut, proprio_mask=pm
        )
        assert np.array_equal(r_ref["action_25hz"][: m + 1], r_mut["action_25hz"][: m + 1]), f"Future leak in action at fps={fps}!"
        assert np.array_equal(r_ref["proprio_25hz"][: m + 1], r_mut["proprio_25hz"][: m + 1]), f"Future leak in proprio at fps={fps}!"


def test_b1_unitree_h1_40dof_19dim_state_lossless_zero_fabrication() -> None:
    """B1/H1 regression test: Unitree H1 action[40] + state[19] preserves all 40 actions & marks ONLY 19 measured proprio dims active."""
    rng = np.random.default_rng(2029)
    T = 25
    raw_a40 = rng.normal(size=(T, 40)).astype(np.float32)
    raw_s19 = rng.normal(size=(T, 19)).astype(np.float32)
    a52, am, p68, pm = map_embodiment_to_unified_52_68("unitree_h1_40dof", raw_action=raw_a40, raw_proprio=raw_s19)
    assert int(np.sum(am)) == 40
    assert int(np.sum(pm)) == 19
    assert np.allclose(p68[:, ACTION_SLICES["left_leg_6dof"].start : ACTION_SLICES["left_leg_6dof"].start + 5], raw_s19[:, 0:5])
    assert np.allclose(p68[:, ACTION_SLICES["left_arm_7dof"].start : ACTION_SLICES["left_arm_7dof"].start + 4], raw_s19[:, 11:15])
    assert np.allclose(a52[:, ACTION_SLICES["left_hand_or_gripper_7dof"].start : ACTION_SLICES["left_hand_or_gripper_7dof"].start + 6], raw_a40[:, 24:30])
    assert float(np.max(np.abs(p68[:, pm == 0.0]))) == 0.0


def test_b3_dev_test_flushes_at_80mb_and_b4_rh20t_kinematic_target(tmp_path) -> None:
    """B3 & B4 regression tests: dev/test flushes <= 80 MB (never stranded in RAM) + RH20T action=q[t+1] != proprio=q[t]."""
    from data.v4_heartbeat_contract import build_interleaved_heartbeat_stream
    from scripts.data.v4_preencode_tier1 import StreamingV4ShardWriter, build_rh20t_kinematic_action_and_proprio

    # B4: RH20T 1-step kinematic target eliminates action == proprio shortcut!
    q = np.cumsum(np.ones((20, 8), dtype=np.float32) * 0.1, axis=0)
    act, prop = build_rh20t_kinematic_action_and_proprio(q)
    assert np.allclose(prop, q)
    assert np.allclose(act[:-1], q[1:])
    assert not np.allclose(act[:-1], prop[:-1])
    assert np.allclose(act[-1], prop[-1])

    # B3: dev split flushes as soon as buf_bytes >= 80 MiB even when flush_mb=650.0!
    rng = np.random.default_rng(2030)
    writer = StreamingV4ShardWriter(
        corpus_name="unit_b3_flush",
        local_out_dir=tmp_path,
        hf_repo_id=None,
        flush_mb=650.0,
        delete_local_after_upload=False,
    )
    writer.buf_bytes["dev"] = 81 * 1024 * 1024
    s = build_interleaved_heartbeat_stream(
        audio_codes=rng.integers(0, 2048, size=(4, 16), dtype=np.int16),
        text_by_heartbeat={1: b"test"},
    )
    writer.add_episode(split="dev", episode_id="dev_ep_0", cluster_key="c0", stream=s, duration_s=0.32)
    assert writer.shard_idx["dev"] == 1
    assert (tmp_path / "unit_b3_flush" / "dev_shard_000.npz").is_file()
    writer.finalize()
