"""CPU Unit Tests for ALL 7 Peer-Reviewer + Astra 6.2 Wave A/B/C Audit Fixes (`0.00 CU`).

Verifies:
  1. `evaluate_real_speech_overwrite_frozen_core` handles exact `audio_codes_a/b`, `update_which`,
     and `is_catch=True` (`update_which=-1`) with zero KeyError / IndexError / TracerConversionError.
  2. `LoadedSparseShard.sample_batch(drop_modality_prob=1.0)` preserves `valid_mask == 1.0` (dense MXU
     fast path), masks dropped payload + sets `reliability == 0.0` + `target_valid_mask == 0.0`, and
     yields non-zero `z_seq` and finite gradients.
  3. Zero-false-positive negative controls:
     - An action-blind model (`edge_action_w == 0`) gets exact `0.0` gain on `evaluate_robotics_action_intervention`.
     - A video-blind model (`edge_video_w == 0`) gets exact `0.0` gain on `evaluate_cross_modal_condition_shuffle`.
  4. Wave B `PolicyCommitGate` blind-reliability regime defeats the `reliability < 0.75` shortcut
     (`reliability_only_threshold_baseline.unsupported_claim_rate == 1.0`).
  5. `build_wave_a_comparator_specs` satisfies `±1.0%` param parity & `<= 5%` byte parity at `scale_700m` & `scale_1b`.
  6. End-to-end Wave A (`--grad-accum 2`, sealed-test locked) -> checkpoint reload -> Wave B -> Wave C
     (`max_core_grad == 0.0`) + free-running 32-byte `inference_showcase/` smoke test.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from data.v3_unified_shards import BYTE_MOD_SEP, LoadedSparseShard, SparseShardBuilder, encode_utf8_bytes
from evals.v3_sota_benchmarks import (
    evaluate_cross_modal_condition_shuffle,
    evaluate_real_speech_overwrite_frozen_core,
    evaluate_robotics_action_intervention,
    generate_cross_modal_text_showcase,
)
from models.v3.any2any_heads import (
    compute_any2any_multimodal_loss,
    init_any2any_heads_params,
    init_policy_commit_gate_params,
)
from models.v3.core import init_core_state, init_slwm_v3_params
from models.v3.fast_core import scan_core_fast
from models.v3.spec import (
    build_wave_a_comparator_specs,
    compute_core_streaming_state_bytes,
    count_tree_params,
    make_slwm_v3_spec,
)
from scripts.train.v3_wave_b_policy_gate import (
    build_epistemic_commitment_dataset,
    evaluate_policy_commit_gate,
    make_blind_unsupported_batch,
)


def _make_synthetic_shards(root: Path) -> None:
    rng = np.random.default_rng(1234)

    # 1. Text/code train, dev, val, and sealed test shards
    for sub, name in (
        ("text_code", "text_code_train_24m_balanced.npz"),
        ("text_code", "text_code_dev_v1.npz"),
        ("text_code", "wikitext103_validation.npz"),
        ("text_code", "wikitext103_test.npz"),
    ):
        b = SparseShardBuilder(name, "dev")
        for i in range(12):
            txt = encode_utf8_bytes(f"Episode {i}: The quick brown fox jumps over the lazy dog and computes sum({i}). ")
            n = len(txt)
            b.add_episode(
                modality_id=np.full((n,), 2, dtype=np.uint8),
                dt_s=np.full((n,), 0.02, dtype=np.float16),
                domain_name="wikitext_103",
                text_byte=txt,
            )
        b.save(root / sub / name)

    # 2. Clotho audio -> text train & dev
    for name in ("clotho_train.npz", "clotho_dev.npz"):
        b = SparseShardBuilder(name, "dev")
        for i in range(10):
            n_aud = 20
            txt = encode_utf8_bytes(f"Caption {i}: birds chirping loudly in morning rain.")
            n = n_aud + len(txt)
            mod = np.concatenate([np.ones((n_aud,), dtype=np.uint8), np.full((len(txt),), 2, dtype=np.uint8)])
            dt = np.concatenate([np.full((n_aud,), 0.08, dtype=np.float16), np.full((len(txt),), 0.02, dtype=np.float16)])
            tb = np.concatenate([np.full((n_aud,), 256, dtype=np.uint16), txt])
            ac = rng.integers(0, 2048, size=(n_aud, 32), dtype=np.uint16)
            b.add_episode(modality_id=mod, dt_s=dt, domain_name="clotho", text_byte=tb, audio_codes=ac)
        b.save(root / "audio_text" / name)

    # 3. MSR-VTT & VGGSound video+audio+text train & dev
    for cname in ("msrvtt", "vggsound"):
        for sp in ("train", "dev"):
            b = SparseShardBuilder(f"{cname}_{sp}", sp)
            for i in range(10):
                n_vid, n_aud = 16, 16
                txt = encode_utf8_bytes(f"{cname} clip {i}: drumming wood and moving camera.")
                n = n_vid + n_aud + len(txt)
                mod = np.concatenate([
                    np.zeros((n_vid,), dtype=np.uint8),
                    np.ones((n_aud,), dtype=np.uint8),
                    np.full((len(txt),), 2, dtype=np.uint8),
                ])
                dt = np.full((n,), 0.04, dtype=np.float16)
                tb = np.concatenate([np.full((n_vid + n_aud,), 256, dtype=np.uint16), txt])
                vf = rng.standard_normal((n_vid, 1024)).astype(np.float16)
                ac = rng.integers(0, 2048, size=(n_aud, 32), dtype=np.uint16)
                sp_pos = np.concatenate([np.arange(n_vid, dtype=np.uint8) % 15, np.zeros((n_aud + len(txt),), dtype=np.uint8)])
                b.add_episode(
                    modality_id=mod, dt_s=dt, domain_name=cname,
                    text_byte=tb, spatial_pos=sp_pos, video_feat=vf, audio_codes=ac,
                )
            b.save(root / "av" / f"{cname}_{sp}.npz")

    # 4. Robotics train & dev (interleaved video + proprio/action)
    for sp in ("train", "dev"):
        b = SparseShardBuilder(f"robotics_{sp}", sp)
        for i in range(10):
            n_vid, n_act = 10, 24
            n = n_vid + n_act
            mod = np.concatenate([np.zeros((n_vid,), dtype=np.uint8), np.full((n_act,), 3, dtype=np.uint8)])
            dt = np.full((n,), 0.04, dtype=np.float16)
            ha = np.concatenate([np.zeros((n_vid,), dtype=np.uint8), np.ones((n_act,), dtype=np.uint8)])
            vf = rng.standard_normal((n_vid, 1024)).astype(np.float16)
            pf = rng.standard_normal((n_act, 16)).astype(np.float16)
            af = rng.standard_normal((n_act, 14)).astype(np.float16)
            b.add_episode(
                modality_id=mod, dt_s=dt, domain_name="lerobot_pusht",
                video_feat=vf, proprio_feat=pf, action_feat=af, has_true_action=ha,
            )
        b.save(root / "robotics" / f"robotics_{sp}.npz")

    # 5. Real-Speech-Overwrite train & dev using EXACT committed schema (`audio_codes_a/b`, `update_which`, `is_catch` with `-1`!)
    rso_dir = root / "real_speech_overwrite"
    rso_dir.mkdir(parents=True, exist_ok=True)
    for sp, n_p in (("train", 8), ("dev", 8)):
        ca = rng.integers(0, 2048, size=(n_p, 40, 32), dtype=np.uint16)
        cb = ca.copy()
        cb[:, :10, :] = rng.integers(0, 2048, size=(n_p, 10, 32), dtype=np.uint16)
        ya = rng.integers(0, 4, size=(n_p, 2), dtype=np.int32)
        yb = ya.copy()
        upd = np.array([0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
        is_c = upd < 0
        for i in range(n_p):
            if not is_c[i]:
                yb[i, upd[i]] = (ya[i, upd[i]] + 1) % 4
        np.savez_compressed(
            rso_dir / f"real_speech_overwrite_{sp}_L12_Weq64_additive.npz",
            audio_codes_a=ca,
            audio_codes_b=cb,
            y_a=ya,
            y_b=yb,
            update_which=upd,
            is_catch=is_c,
            primary_speaker_id=np.array([f"{sp}_spk_{i}" for i in range(n_p)]),
        )


def test_all_seven_audit_fixes(tmp_path: Path) -> None:
    _make_synthetic_shards(tmp_path)
    spec = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_routed")
    k1, k2 = jax.random.split(jax.random.PRNGKey(42), 2)
    all_params = {
        "core": init_slwm_v3_params(k1, spec),
        "heads": init_any2any_heads_params(k2, spec),
    }

    # --- FIX #1: RSO exact schema + catch trials (`update_which == -1`) + zero TracerConversionError ---
    rso_res = evaluate_real_speech_overwrite_frozen_core(
        all_params["core"],
        spec,
        tmp_path / "real_speech_overwrite/real_speech_overwrite_train_L12_Weq64_additive.npz",
        tmp_path / "real_speech_overwrite/real_speech_overwrite_dev_L12_Weq64_additive.npz",
        probe_steps=5,
        eval_micro_bsz=4,
        matmul_dtype=jnp.float32,
    )
    assert rso_res["n_pairs"] == 8
    assert rso_res["n_overwrite_pairs"] == 6
    assert rso_res["n_catch_pairs"] == 2
    assert 0.0 <= rso_res["overall_joint_acc"] <= 1.0

    # --- FIX #2: Modality dropout keeps `valid_mask == 1.0`, non-zero `z_seq`, and finite grads ---
    sh_av = LoadedSparseShard.load(tmp_path / "av/msrvtt_train.npz")
    b_drop = sh_av.sample_batch(np.random.default_rng(7), batch_size=4, seq_len=48, drop_modality_prob=1.0)
    assert np.all(b_drop["valid_mask"] == 1.0), "Modality dropout MUST preserve valid_mask==1 for dense MXU fast path!"
    assert np.any(b_drop["reliability"] == 0.0), "Dropped modality MUST set reliability==0.0!"
    assert np.any(b_drop["target_valid_mask"] == 0.0), "Dropped source modality MUST have target_valid_mask==0.0!"
    b_jax = {k: jnp.asarray(v) for k, v in b_drop.items()}
    z, _, _ = scan_core_fast(all_params["core"], init_core_state(spec, 4), b_jax, spec, remat_layers=False, matmul_dtype=jnp.float32, assume_all_valid=True)
    assert float(jnp.min(jnp.linalg.norm(z, axis=-1))) > 0.0, "Dropped ticks must NOT zero out residual stream z_seq!"
    lv, _ = compute_any2any_multimodal_loss(all_params["heads"], z, b_jax, spec, matmul_dtype=jnp.float32)
    assert np.isfinite(float(lv))

    # --- FIX #4a: Zero-False-Positive Action-Intervention Negative Control ---
    # Make model strictly action-blind (`edge_action_w == 0`) AND non-zero readout weights!
    blind_act_params = jax.tree_util.tree_map(lambda x: x, all_params)
    blind_act_params["core"] = dict(blind_act_params["core"])
    blind_act_params["core"]["edge"] = dict(blind_act_params["core"]["edge"])
    blind_act_params["core"]["edge"]["edge_action_w"] = jnp.zeros_like(blind_act_params["core"]["edge"]["edge_action_w"])
    blind_act_params["heads"] = dict(blind_act_params["heads"])
    blind_act_params["heads"]["head_action"] = dict(blind_act_params["heads"]["head_action"])
    blind_act_params["heads"]["head_action"]["w"] = jnp.ones_like(blind_act_params["heads"]["head_action"]["w"]) * 0.01
    act_res = evaluate_robotics_action_intervention(
        blind_act_params, spec, tmp_path / "robotics/robotics_dev.npz",
        seq_len=34, batch_size=4, max_episodes=8, matmul_dtype=jnp.float32,
    )
    assert act_res["targets_strictly_held_fixed"] is True
    assert abs(act_res["action_intervention_proprio_gain"]) < 1e-6, f"Action-blind model got false positive: {act_res}"
    assert abs(act_res["action_intervention_video_gain"]) < 1e-6, f"Action-blind model got video false positive: {act_res}"
    assert act_res["causal_action_conditioning_verified"] is False

    # --- FIX #4b: Zero-False-Positive Condition-Shuffle Negative Control ---
    blind_vid_params = jax.tree_util.tree_map(lambda x: x, all_params)
    blind_vid_params["core"] = dict(blind_vid_params["core"])
    blind_vid_params["core"]["edge"] = dict(blind_vid_params["core"]["edge"])
    blind_vid_params["core"]["edge"]["edge_video_w"] = jnp.zeros_like(blind_vid_params["core"]["edge"]["edge_video_w"])
    blind_vid_params["core"]["edge"]["edge_spatial_pos_emb"] = jnp.zeros_like(blind_vid_params["core"]["edge"]["edge_spatial_pos_emb"])
    shuf_res = evaluate_cross_modal_condition_shuffle(
        blind_vid_params, spec, tmp_path / "av/msrvtt_dev.npz",
        source_mod=0, target_mod=2, seq_len=96, batch_size=4, max_episodes=8, matmul_dtype=jnp.float32,
    )
    assert shuf_res["n_evaluated_episodes"] == 8
    assert abs(shuf_res["causal_shuffle_gain"]) < 1e-6, f"Video-blind model got false positive: {shuf_res}"
    assert shuf_res["causal_conditioning_verified"] is False

    # --- FIX #5 + BLOCKER B1: Variable-length Wave B blind-reliability zero-pad / zero-code check ---
    # Build variable-length Clotho-like shard (lengths 20..90) + 700-tick MSR-VTT-like shard (text at tick 645!)
    var_clotho = SparseShardBuilder("var_clotho_dev", "dev")
    rng_v = np.random.default_rng(777)
    for i, n_aud in enumerate((12, 48, 18, 60, 24, 52)):
        txt = encode_utf8_bytes(f"VarLen episode {i}: rain tapping steadily on roof glass.")
        n = n_aud + len(txt)
        mod = np.concatenate([np.ones((n_aud,), dtype=np.uint8), np.full((len(txt),), 2, dtype=np.uint8)])
        dt = np.full((n,), 0.04, dtype=np.float16)
        tb = np.concatenate([np.full((n_aud,), 256, dtype=np.uint16), txt])
        # Strictly positive non-zero audio codes `[1..2047]` so any injected zero-code bug is caught immediately!
        ac = rng_v.integers(1, 2048, size=(n_aud, 32), dtype=np.uint16)
        var_clotho.add_episode(modality_id=mod, dt_s=dt, domain_name="clotho", text_byte=tb, audio_codes=ac)
    var_clotho_p = tmp_path / "audio_text/var_clotho_dev.npz"
    var_clotho.save(var_clotho_p)
    # Directly verify ZERO all-zero audio_codes and ZERO BYTE_PAD on active [:qp] ticks across variable lengths!
    from evals.v3_sota_benchmarks import _build_episode_aligned_batch
    sh_vc = LoadedSparseShard.load(var_clotho_p)
    rst_vc = np.nonzero(sh_vc.data["reset_mask"])[0]
    ends_vc = np.append(rst_vc[1:], sh_vc.n_ticks)
    b_vc_clean = _build_episode_aligned_batch(sh_vc, [(int(rst_vc[i]), int(ends_vc[i])) for i in range(6)], spec, 128)
    qp_vc = np.array([12, 48, 18, 60, 24, 52], dtype=np.int32)
    b_vc_unsup = make_blind_unsupported_batch(b_vc_clean, qp_vc)
    assert np.all(b_vc_unsup["reliability"] == 1.0)
    for bi_vc in range(6):
        qp_i = int(qp_vc[bi_vc])
        aud_slice = b_vc_unsup["audio_codes"][:qp_i, bi_vc]
        assert np.all(aud_slice > 0), f"Zero-code shortcut leaked into ep {bi_vc}!"
        assert not np.array_equal(aud_slice, b_vc_clean["audio_codes"][:qp_i, bi_vc]), "Unsupported must differ from clean!"

    ev_blind = build_epistemic_commitment_dataset(
        all_params, spec, [var_clotho_p],
        regime="blind_reliability_unsupported", seq_len=128, batch_size=4, n_batches=2, matmul_dtype=jnp.float32,
    )
    assert np.allclose(ev_blind["ev"][:, 3], 1.0), "Blind-reliability regime MUST hold reliability==1.0 identically!"
    gate_p = init_policy_commit_gate_params(jax.random.PRNGKey(9), spec.d_model, hidden_dim=32)
    wb_eval = evaluate_policy_commit_gate(gate_p, ev_blind)
    assert wb_eval["reliability_only_threshold_baseline"]["unsupported_claim_rate"] == 1.0
    assert wb_eval["reliability_only_threshold_baseline"]["abstention_noop_rate"] == 0.0

    # --- BLOCKER B2: 700-tick MSR-VTT-like video-then-text episode at `seq_len=512` (text starts at tick 645!) ---
    long_av = SparseShardBuilder("long_msrvtt_dev", "dev")
    for i in range(4):
        n_vid, n_aud = 570, 75
        txt = encode_utf8_bytes(f"Long MSR-VTT clip {i}: someone cooks pasta in a bright kitchen.")
        n = n_vid + n_aud + len(txt)
        mod = np.concatenate([
            np.zeros((n_vid,), dtype=np.uint8),
            np.ones((n_aud,), dtype=np.uint8),
            np.full((len(txt),), 2, dtype=np.uint8),
        ])
        dt = np.full((n,), 0.04, dtype=np.float16)
        tb = np.concatenate([np.full((n_vid + n_aud,), 256, dtype=np.uint16), txt])
        vf = rng_v.standard_normal((n_vid, 1024)).astype(np.float16)
        ac = rng_v.integers(1, 2048, size=(n_aud, 32), dtype=np.uint16)
        sp_pos = np.concatenate([np.arange(n_vid, dtype=np.uint8) % 15, np.zeros((n_aud + len(txt),), dtype=np.uint8)])
        long_av.add_episode(modality_id=mod, dt_s=dt, domain_name="msrvtt", text_byte=tb, spatial_pos=sp_pos, video_feat=vf, audio_codes=ac)
    long_av_p = tmp_path / "av/long_msrvtt_dev.npz"
    long_av.save(long_av_p)
    sc_long = generate_cross_modal_text_showcase(
        all_params, spec, long_av_p, out_dir=tmp_path / "sc_long", split="dev", model_name="tiny_long",
        n_episodes=4, seq_len=512, matmul_dtype=jnp.float32,
    )
    assert sc_long.get("n_ranked_episodes", 0) == 4, f"700-tick MSR-VTT showcase skipped: {sc_long}"
    sc_long_doc = json.loads(Path(sc_long["manifest_path"]).read_text())
    assert "free_running_32b_continuation" in sc_long_doc["selected_quantiles"]["P50_median"]
    # Also verify Wave B handles 700-tick MSR-VTT episodes at seq_len=256 and produces text targets!
    ev_blind_long = build_epistemic_commitment_dataset(
        all_params, spec, [long_av_p],
        regime="blind_reliability_unsupported", seq_len=256, batch_size=4, n_batches=1, matmul_dtype=jnp.float32,
    )
    assert np.any(ev_blind_long["target_action"] == 0), "Target-anchored Wave B on 700-tick MSR-VTT MUST include SPEAK_TEXT targets!"

    # --- FIX #7a: `scale_700m` and `scale_1b` comparator ±1.0% param & <=5% byte parity ---
    for prof in ("scale_700m", "scale_1b"):
        specs = build_wave_a_comparator_specs(prof, match_params_to_anchor=True)
        anc = specs["slwm_v3_routed"]
        anc_p = count_tree_params(jax.eval_shape(lambda k: init_slwm_v3_params(k, anc), jax.random.PRNGKey(0)))
        anc_b = compute_core_streaming_state_bytes(anc, batch_size=1)["total_core_state_bytes"]
        for name, sp in specs.items():
            np_c = count_tree_params(jax.eval_shape(lambda k, s=sp: init_slwm_v3_params(k, s), jax.random.PRNGKey(0)))
            nb_c = compute_core_streaming_state_bytes(sp, batch_size=1)["total_core_state_bytes"]
            assert abs(np_c - anc_p) / anc_p <= 0.01, f"{prof}/{name} param mismatch: {np_c} vs {anc_p}"
            if name != "sliding_window_gqa_eqparam":
                assert abs(nb_c - anc_b) / anc_b <= 0.05, f"{prof}/{name} byte mismatch: {nb_c} vs {anc_b}"

    # --- FIX #3 & End-to-End Wave A (grad_accum=2, sealed-test locked) -> Reload -> Wave B -> Wave C -> Showcase ---
    out_dir = tmp_path / "e2e_out"
    repo_root = Path(__file__).resolve().parents[1]
    res_a = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts/train/v3_any2any_trainer.py"),
            "--profile", "tiny_unit", "--backbone", "slwm_v3_routed",
            "--shard-dir", str(tmp_path), "--steps", "2", "--seq-len", "64",
            "--batch-size", "2", "--grad-accum", "2", "--seed", "4201", "--out-dir", str(out_dir),
        ],
        capture_output=True, text=True, check=False,
    )
    assert res_a.returncode == 0, f"Wave A failed:\n{res_a.stdout}\n{res_a.stderr}"
    rec_a = json.loads((out_dir / "tiny_unit_slwm_v3_routed_s4201_receipt.json").read_text())
    assert rec_a["eval_sealed_test_enabled"] is False
    assert "sealed_test" not in rec_a["frozen_evaluations"], "Sealed test MUST NOT run when --eval-sealed-test is False!"
    assert rec_a["grad_accum"] == 2 and rec_a["effective_batch_size"] == 4
    assert "error" not in rec_a["frozen_evaluations"]["real_speech_overwrite_dev_additive_rf"]
    # Verify showcase has free-running 32-byte generation!
    sc_path = Path(rec_a["frozen_evaluations"]["inference_showcase"]["clotho_dev"]["manifest_path"])
    sc_doc = json.loads(sc_path.read_text())
    assert "free_running_32b_continuation" in sc_doc["selected_quantiles"]["P50_median"]

    ckpt_p = out_dir / "tiny_unit_slwm_v3_routed_s4201_ckpt.pkl"
    assert ckpt_p.is_file()
    res_b = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts/train/v3_wave_b_policy_gate.py"),
            "--ckpt", str(ckpt_p), "--shard-dir", str(tmp_path), "--tag", "tiny_test",
            "--steps", "10", "--seed", "5201", "--out-dir", str(out_dir),
        ],
        capture_output=True, text=True, check=False,
    )
    assert res_b.returncode == 0, f"Wave B failed:\n{res_b.stdout}\n{res_b.stderr}"
    rec_b = json.loads((out_dir / "wave_b_policy_gate_tiny_test_receipt.json").read_text())
    assert rec_b["train_shards_strictly_disjoint_from_dev"] is True
    assert rec_b["primary_blind_reliability_regime"]["reliability_only_threshold_baseline"]["unsupported_claim_rate"] == 1.0

    res_c = subprocess.run(
        [
            sys.executable, str(repo_root / "scripts/train/v3_wave_c_decoders.py"),
            "--ckpt", str(ckpt_p), "--shard-dir", str(tmp_path), "--tag", "tiny_test",
            "--steps", "2", "--seq-len", "64", "--batch-size", "2", "--seed", "6201", "--out-dir", str(out_dir),
        ],
        capture_output=True, text=True, check=False,
    )
    assert res_c.returncode == 0, f"Wave C failed:\n{res_c.stdout}\n{res_c.stderr}"
    rec_c = json.loads((out_dir / "wave_c_decoders_tiny_test_receipt.json").read_text())
    assert rec_c["max_core_grad_verified"] == 0.0
    assert Path(rec_c["wave_c_showcases"]["clotho_dev"]["manifest_path"]).is_file()
