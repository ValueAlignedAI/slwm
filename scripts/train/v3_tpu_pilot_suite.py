"""SLWM-v3 TPU v6e-1 4-Phase Hard-Capped Validation Pilot Suite (Rev 4, <= 45 CU).

Executes the four pre-registered pilot validation phases cleanly on Google TPU v6e-1 (or `--smoke-cpu`
for fast CPU verification at `0.00 CU`), saving an individual JSON receipt after EVERY phase:
  1. `profile`:      Verifies JAX/Optax versions & device backend; profiles ACTUAL worst training shapes
                     (`pilot_d256` Phase 3 train `B=8, T=896` & microbatched eval `B=16, T=896`;
                     `discovery_45m` `B=4, T=512`; `confirmation_135m` `B=2, T=256`) with chunked remat
                     (`chunk=64`) + native JAX `bfloat16` 3D-conv probe.
  2. `mup_check`:    Fixed-depth (`L=12`) 3-width (`d in {256, 512, 768}`) muP vs SP check AND separate
                     fixed-width (`d=256`) 3-depth (`L in {8, 12, 16}`) CompleteP residual-scaling check
                     across 5 base LRs `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` using exact masked AdamW,
                     `val_batch_size=16`, compiled-step reuse, and unrounded transfer regret on BOTH grids!
  3. `dev_ablation`: Frozen-codebook, 100%-observable, oracle-verified multimodal counterfactual
                     delayed-overwrite benchmark (`T = 896 > RF_eqbyte = 756 > RF_eqparam = 372` on TPU)
                     with real single-tick observable future sensory echoes on `pkts["probe_feat"]`
                     (0% delayed-value copy shortcut), strict pre-echo `stop_gradient` readout across BOTH
                     `stream_a` and `stream_b`, background-calibrated `3.5-sigma` burst stress, and ALL 9
                     parameter- (`±1.0%`) & byte-matched `pilot_d256` comparators!
  4. `long_stream`:  2-seed repeatability on `discovery_45m` (`s3401`, `s3402`) AND `T = 256 -> 512 -> 1024 -> 2048`
                     infinite-horizon stability audit executed on BOTH trained seed checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.v3.core import init_core_state, init_slwm_v3_params, scan_core
from models.v3.objectives import (
    apply_decoupled_probe,
    init_decoupled_probe_params,
    masked_multi_horizon_predictive_loss,
)
from models.v3.spec import (
    SLWMv3Spec,
    build_wave_a_comparator_specs,
    compute_core_streaming_state_bytes,
    count_tree_params,
    make_slwm_v3_spec,
)
from scripts.train.v3_pilot_train import (
    build_optimizer,
    compile_pilot_steps,
    effective_rank_participation_ratio,
    lr_schedule_mult,
    make_synthetic_multimodal_batch,
    run_pilot_train,
)


# ---------------------------------------------------------------------------
# Phase 3 Frozen-Codebook Multimodal Counterfactual Generator & 100% Oracle
# ---------------------------------------------------------------------------


def build_frozen_phase3_codebook(
    spec: SLWMv3Spec, vocab_size: int = 8, codebook_seed: int = 202601
) -> dict[str, np.ndarray]:
    """Build ONE frozen sensory codebook shared identically across ALL Phase 3 train and dev batches.

    Completely independent of episode seeds so visual/audio/text cue signatures map consistently to
    underlying semantic values across unseen train/dev episodes and counterfactual pairs.
    """
    rng = np.random.default_rng(codebook_seed)

    def _unit_rms(arr: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(arr * arr, axis=-1, keepdims=True) + 1e-8)
        return (arr / rms).astype(np.float32)

    # 2 target attributes + 1 distractor attribute (16-d key signature each, pairwise well-separated)
    attr_keys = _unit_rms(rng.normal(0.0, 1.0, size=(3, 16)))
    query_marker = _unit_rms(rng.normal(0.0, 1.0, size=(16,)))

    # Value signatures across ALL 3 real input modalities (video patch, Mimi RVQ audio codes, ByT5 text byte)
    # PLUS fixed 16-d observable sensory echo templates in `target_feat_dim`!
    val_echo = _unit_rms(rng.normal(0.0, 1.0, size=(vocab_size, 16)))

    # Modality 0 (Video): `[3, V, video_patch_dim]` -> deterministic key+value visual patch patterns
    vid_attr_base = _unit_rms(rng.normal(0.0, 1.0, size=(3, spec.video_patch_dim)))
    vid_val_base = _unit_rms(rng.normal(0.0, 1.0, size=(vocab_size, spec.video_patch_dim)))

    # Modality 1 (Audio RVQ): distinct non-overlapping codebook token patterns per `(attr, val)`!
    # First 16 codebooks carry attribute signature; last 16 codebooks carry value signature.
    aud_attr_codes = np.zeros((3, spec.audio_codebooks // 2), dtype=np.int32)
    for a in range(3):
        aud_attr_codes[a] = (np.arange(spec.audio_codebooks // 2, dtype=np.int32) * 17 + 100 + a * 31) % spec.audio_vocab_size
    aud_val_codes = np.zeros((vocab_size, spec.audio_codebooks // 2), dtype=np.int32)
    for v in range(vocab_size):
        aud_val_codes[v] = (np.arange(spec.audio_codebooks // 2, dtype=np.int32) * 23 + 500 + v * 37) % spec.audio_vocab_size

    # Modality 2 (ByT5 UTF-8 byte): unique printable ASCII byte per `(attr, val)` in `33..126`!
    # Encoding `(attr in 0..2, val in 0..V-1)` bijectively into `33 + attr * vocab_size + val`
    txt_bytes = np.zeros((3, vocab_size), dtype=np.int32)
    for a in range(3):
        for v in range(vocab_size):
            txt_bytes[a, v] = 33 + a * vocab_size + v

    return {
        "attr_keys": attr_keys,  # [3, 16]
        "query_marker": query_marker,  # [16]
        "val_echo": val_echo,  # [V, 16]
        "vid_attr_base": vid_attr_base,  # [3, video_patch_dim]
        "vid_val_base": vid_val_base,  # [V, video_patch_dim]
        "aud_attr_codes": aud_attr_codes,  # [3, C/2]
        "aud_val_codes": aud_val_codes,  # [V, C/2]
        "txt_bytes": txt_bytes,  # [3, V]
        "vocab_size": np.asarray(vocab_size, dtype=np.int32),
    }


def generate_phase3_counterfactual_batch(
    spec: SLWMv3Spec,
    codebook: dict[str, np.ndarray],
    *,
    batch_size: int,
    t_len: int = 832,
    min_delay_after_corr: int = 768,
    stress_mode: str = "clean",
    seed: int = 4000,
) -> dict[str, Any]:
    """Generate paired counterfactual multimodal delayed-overwrite streams (`stream_a` & `stream_b`).

    Key scientific guarantees:
      1. **Strictly beyond-stack delay:** `t_q1 - t_corr >= min_delay_after_corr` (`768 > RF_eqbyte = 756 > RF_eqparam = 372`
         at `pilot_d256`!), AND `t_corr - t_init_1 > W = 32` (overwrite arrives after initial cue exits local ring!).
      2. **Genuine counterfactual A/B pairs:** `stream_a` and `stream_b` are BIT-EXACT identical on every tick
         EXCEPT the single correction tick `t_corr` (where `val_new_b != val_new_a != val_old`) and the post-query
         future sensory echo of the updated attribute!
      3. **High-MI self-supervised predictive target (ZERO downstream labels):**
         At `t_q1` and `t_q2` (strictly after `t_pre_echo = t_q1 - 1`), sensory query prompts arrive on `modality_id = 4`
         and the environment emits observable future sensory echoes `val_echo[latest_val(attr)]` at `t_q + 1..t_q + 4`!
      4. **Zero probe leakage:** Downstream evaluation probes extract `Z[t_pre_echo]` STRICTLY BEFORE `t_q1`,
         before EITHER query echo has ever entered the stream!
      5. **True multimodal cues:** Initial and correction facts arrive across `0:video`, `1:audio` (Mimi RVQ), and
         `2:text` (ByT5 UTF-8 byte) via their real native edge adapters!
    """
    rng = np.random.default_rng(seed)
    v_sz = int(codebook["vocab_size"])
    # Two distinct single-tick observable sensory echo rounds (`qa1` then `qa2 != qa1`), 9 ticks each:
    # 8 query-prompt ticks (`t_q .. t_q+7`, NO answer in inputs) followed by 1 REAL observable sensory
    # echo tick (`t_echo = t_q + 8`) emitted directly into `pkts["probe_feat"]`!
    # Every horizon h in {1, 4, 8} predicting `t_echo` runs strictly BEFORE `t_echo` arrives (0% delayed-value copy shortcut!).
    round_len = 9
    query_span = 2 * round_len + 1  # 19 ticks total: t_pre_echo + 9 + 9
    if t_len < min_delay_after_corr + query_span + 20:
        raise ValueError(
            f"t_len={t_len} too short for min_delay_after_corr={min_delay_after_corr} + query_span={query_span}"
        )

    t_q1 = t_len - 2 * round_len
    t_echo1 = t_q1 + round_len - 1
    t_q2 = t_echo1 + 1
    t_echo2 = t_q2 + round_len - 1
    t_pre_echo = t_q1 - 1  # Strict probe readout index (BEFORE t_q1 and BEFORE any query or echo!)
    t_corr_max = t_q1 - min_delay_after_corr

    # Episode latent ground truth
    v0_init = rng.integers(0, v_sz, size=(batch_size,), dtype=np.int32)
    v1_init = rng.integers(0, v_sz, size=(batch_size,), dtype=np.int32)
    upd_attr = rng.integers(0, 2, size=(batch_size,), dtype=np.int32)  # 0 or 1 updated
    first_attr = rng.integers(0, 2, size=(batch_size,), dtype=np.int32)  # randomized arrival order!
    q1_attr = rng.integers(0, 2, size=(batch_size,), dtype=np.int32)  # randomized query order!

    # Guarantee val_new_a != val_old AND val_new_b != val_old AND val_new_b != val_new_a for 100% of episodes!
    val_old = np.where(upd_attr == 0, v0_init, v1_init)
    val_pres = np.where(upd_attr == 0, v1_init, v0_init)
    off_a = rng.integers(1, v_sz, size=(batch_size,), dtype=np.int32)
    val_new_a = (val_old + off_a) % v_sz
    off_b = rng.integers(1, v_sz - 1, size=(batch_size,), dtype=np.int32)
    # Pick distinct val_new_b from remaining V-2 values
    val_new_b = np.empty_like(val_new_a)
    for b in range(batch_size):
        cands = [x for x in range(v_sz) if x != int(val_old[b]) and x != int(val_new_a[b])]
        val_new_b[b] = cands[int(off_b[b]) % len(cands)]

    # Cue modalities in {0:video, 1:audio, 2:text}
    mod_init_0 = rng.integers(0, 3, size=(batch_size,), dtype=np.int32)
    mod_init_1 = rng.integers(0, 3, size=(batch_size,), dtype=np.int32)
    mod_corr = rng.integers(0, 3, size=(batch_size,), dtype=np.int32)

    t_init_0 = rng.integers(1, 5, size=(batch_size,), dtype=np.int32)
    t_init_1 = t_init_0 + rng.integers(4, 8, size=(batch_size,), dtype=np.int32)
    # Place overwrite past local window W when possible, strictly <= t_corr_max
    t_corr_lo = np.minimum(t_init_1 + max(4, min(spec.window_size + 2, t_corr_max - 15)), t_corr_max)
    t_corr = np.array(
        [int(rng.integers(int(t_corr_lo[b]), max(int(t_corr_lo[b]) + 1, int(t_corr_max) + 1))) for b in range(batch_size)],
        dtype=np.int32,
    )

    # Shared neutral background packet templates (identical across stream_a & stream_b!)
    base_mod_id = np.full((t_len, batch_size), 4, dtype=np.int32)
    base_dt = np.full((t_len, batch_size), 0.10, dtype=np.float32)
    base_valid = np.ones((t_len, batch_size), dtype=np.float32)
    base_rel = np.ones((t_len, batch_size), dtype=np.float32)
    base_vid = np.zeros((t_len, batch_size, spec.video_patch_dim), dtype=np.float32)
    base_sp = np.zeros((t_len, batch_size), dtype=np.int32)
    base_aud_c = np.zeros((t_len, batch_size, spec.audio_codebooks), dtype=np.int32)
    base_aud_cont = np.zeros((t_len, batch_size, spec.audio_cont_dim), dtype=np.float32)
    base_txt = np.full((t_len, batch_size), 32, dtype=np.int32)  # space byte
    base_prop = np.zeros((t_len, batch_size, spec.proprio_dim), dtype=np.float32)
    base_act = np.zeros((t_len, batch_size, spec.action_dim), dtype=np.float32)
    base_has_act = np.zeros((t_len, batch_size), dtype=np.float32)
    # Gentle observable background oscillation on probe_feat (0.08 amp so the two 2.0-amp real sensory
    # echoes naturally carry ~68% of uniform-weight Huber variance with ZERO special target masking!)
    t_grid = np.arange(t_len, dtype=np.float32)[:, None]
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(1, batch_size)).astype(np.float32)
    base_probe = np.zeros((t_len, batch_size, spec.probe_dim), dtype=np.float32)
    for d_i in range(spec.probe_dim):
        freq = 0.05 + 0.01 * (d_i % 4)
        base_probe[:, :, d_i] = 0.08 * np.sin(freq * t_grid + phase + 0.3 * d_i)

    def _inject_cue(
        mod_arr: np.ndarray,
        vid_arr: np.ndarray,
        aud_c_arr: np.ndarray,
        aud_cont_arr: np.ndarray,
        txt_arr: np.ndarray,
        probe_arr: np.ndarray,
        t_idx: int,
        b_idx: int,
        mod: int,
        attr: int,
        val: int,
    ) -> None:
        mod_arr[t_idx, b_idx] = mod
        if mod == 0:
            vid_arr[t_idx, b_idx] = 0.7071 * (codebook["vid_attr_base"][attr] + codebook["vid_val_base"][val])
        elif mod == 1:
            half_c = spec.audio_codebooks // 2
            aud_c_arr[t_idx, b_idx, :half_c] = codebook["aud_attr_codes"][attr]
            aud_c_arr[t_idx, b_idx, half_c:] = codebook["aud_val_codes"][val]
            # Also populate continuous audio path for `scratch_edge` parity!
            aud_cont_arr[t_idx, b_idx, :16] = codebook["attr_keys"][attr]
            aud_cont_arr[t_idx, b_idx, 16:32] = codebook["val_echo"][val]
        else:
            txt_arr[t_idx, b_idx] = int(codebook["txt_bytes"][attr, val])
        # Clear probe_feat at non-probe cue ticks so zero cross-channel shortcut exists!
        probe_arr[t_idx, b_idx] = 0.0

    # Inject shared initial cues & randomized delay distractors/bursts (IDENTICAL across A and B!)
    for b in range(batch_size):
        a0 = int(first_attr[b])
        a1 = 1 - a0
        v_a0 = int(v0_init[b] if a0 == 0 else v1_init[b])
        v_a1 = int(v0_init[b] if a1 == 0 else v1_init[b])
        _inject_cue(
            base_mod_id, base_vid, base_aud_c, base_aud_cont, base_txt, base_probe,
            int(t_init_0[b]), b, int(mod_init_0[b]), a0, v_a0,
        )
        _inject_cue(
            base_mod_id, base_vid, base_aud_c, base_aud_cont, base_txt, base_probe,
            int(t_init_1[b]), b, int(mod_init_1[b]), a1, v_a1,
        )

        # Stress injections strictly inside the delay region `[t_corr + 2, t_pre_echo - 2]`!
        d_lo = int(t_corr[b]) + 2
        d_hi = t_pre_echo - 2
        if d_hi > d_lo + 4:
            if stress_mode == "distractor":
                n_dist = min(12, max(2, (d_hi - d_lo) // 40))
                d_ticks = rng.choice(np.arange(d_lo, d_hi), size=n_dist, replace=False)
                for dt_i in d_ticks:
                    d_mod = int(rng.integers(0, 3))
                    d_val = int(rng.integers(0, v_sz))
                    # Matched-norm unrelated distractor attribute (`attr = 2`)!
                    _inject_cue(
                        base_mod_id, base_vid, base_aud_c, base_aud_cont, base_txt, base_probe,
                        int(dt_i), b, d_mod, 2, d_val,
                    )
            elif stress_mode == "corrupt":
                n_burst = min(12, max(2, (d_hi - d_lo) // 40))
                b_ticks = rng.choice(np.arange(d_lo, d_hi), size=n_burst, replace=False)
                # Calibrate `3.5-sigma` burst strictly relative to clean background SD (`0.08 / sqrt(2)`)!
                bg_sd = 0.08 / float(np.sqrt(2.0))
                burst_sd = 3.5 * bg_sd
                for bt_i in b_ticks:
                    base_mod_id[int(bt_i), b] = 4
                    base_probe[int(bt_i), b] = rng.normal(0.0, burst_sd, size=(spec.probe_dim,)).astype(np.float32)

        # Query prompts on `[t_q1 .. t_echo1 - 1]` and `[t_q2 .. t_echo2 - 1]` (8 ticks each; NO answer yet!)
        qa1 = int(q1_attr[b])
        qa2 = 1 - qa1
        for t_p in range(t_q1, t_echo1):
            base_mod_id[t_p, b] = 4
            base_probe[t_p, b, :16] = codebook["query_marker"]
            base_probe[t_p, b, 16:32] = codebook["attr_keys"][qa1]
        for t_p in range(t_q2, t_echo2):
            base_mod_id[t_p, b] = 4
            base_probe[t_p, b, :16] = codebook["query_marker"]
            base_probe[t_p, b, 16:32] = codebook["attr_keys"][qa2]

    def _build_stream(val_new_arr: np.ndarray) -> tuple[dict[str, jax.Array], jax.Array, jax.Array, np.ndarray]:
        mod_id = base_mod_id.copy()
        vid = base_vid.copy()
        aud_c = base_aud_c.copy()
        aud_cont = base_aud_cont.copy()
        txt = base_txt.copy()
        probe = base_probe.copy()

        tvalid = np.ones((t_len, batch_size), dtype=np.float32)
        y_attr_pair = np.zeros((batch_size, 2), dtype=np.int32)

        for b in range(batch_size):
            ua = int(upd_attr[b])
            vn = int(val_new_arr[b])
            _inject_cue(
                mod_id, vid, aud_c, aud_cont, txt, probe,
                int(t_corr[b]), b, int(mod_corr[b]), ua, vn,
            )

            # Final physical world state `[val_final_attr0, val_final_attr1]`
            final_v = [int(v0_init[b]), int(v1_init[b])]
            final_v[ua] = vn
            y_attr_pair[b, 0] = final_v[0]
            y_attr_pair[b, 1] = final_v[1]

            qa1 = int(q1_attr[b])
            qa2 = 1 - qa1
            echo1 = codebook["val_echo"][final_v[qa1]]  # [16]
            echo2 = codebook["val_echo"][final_v[qa2]]  # [16]

            # Emit REAL SINGLE-TICK OBSERVABLE SENSORY ECHOES directly into `pkts["probe_feat"]`
            # at `t_echo1 = t_q1 + 8` and `t_echo2 = t_q2 + 8` (100% real future sensory packets,
            # 0% delayed-value copy shortcut because every horizon h in {1, 4, 8} predicts strictly BEFORE `t_echo`!)
            mod_id[t_echo1, b] = 4
            probe[t_echo1, b, 0:16] = 2.0 * codebook["attr_keys"][qa1]
            probe[t_echo1, b, 16:32] = 2.0 * echo1

            mod_id[t_echo2, b] = 4
            probe[t_echo2, b, 0:16] = 2.0 * codebook["attr_keys"][qa2]
            probe[t_echo2, b, 16:32] = 2.0 * echo2

        # Target features are STRICTLY a deterministic readout of the observable sensory packet `probe`
        # (ZERO hidden label-only injection into `targs`!)
        targs = np.zeros((t_len, batch_size, spec.target_feat_dim), dtype=np.float32)
        targs[:, :, 0:32] = probe[:, :, 0:32]
        if spec.target_feat_dim >= 48:
            targs[:, :, 32:48] = probe[:, :, 16:32]
        if spec.target_feat_dim >= 64:
            targs[:, :, 48:64] = -probe[:, :, 16:32]

        pkts = {
            "modality_id": jnp.asarray(mod_id, dtype=jnp.int32),
            "dt_s": jnp.asarray(base_dt, dtype=jnp.float32),
            "valid_mask": jnp.asarray(base_valid, dtype=jnp.float32),
            "reliability": jnp.asarray(base_rel, dtype=jnp.float32),
            "video_feat": jnp.asarray(vid, dtype=jnp.float32),
            "spatial_pos": jnp.asarray(base_sp, dtype=jnp.int32),
            "audio_codes": jnp.asarray(aud_c, dtype=jnp.int32),
            "audio_cont": jnp.asarray(aud_cont, dtype=jnp.float32),
            "text_byte": jnp.asarray(txt, dtype=jnp.int32),
            "proprio_feat": jnp.asarray(base_prop, dtype=jnp.float32),
            "action_feat": jnp.asarray(base_act, dtype=jnp.float32),
            "has_true_action": jnp.asarray(base_has_act, dtype=jnp.float32),
            "probe_feat": jnp.asarray(probe, dtype=jnp.float32),
        }
        return pkts, jnp.asarray(targs, dtype=jnp.float32), jnp.asarray(tvalid, dtype=jnp.float32), y_attr_pair

    pkts_a, targs_a, tvalid_a, y_attr_a = _build_stream(val_new_a)
    pkts_b, targs_b, tvalid_b, y_attr_b = _build_stream(val_new_b)
    return {
        "pkts_a": pkts_a,
        "targs_a": targs_a,
        "tvalid_a": tvalid_a,
        "pkts_b": pkts_b,
        "targs_b": targs_b,
        "tvalid_b": tvalid_b,
        # Physical world state labels `[val_final_attr0, val_final_attr1]`!
        "y_a": jnp.asarray(y_attr_a, dtype=jnp.int32),
        "y_b": jnp.asarray(y_attr_b, dtype=jnp.int32),
        "val_new_a": val_new_a,
        "val_new_b": val_new_b,
        "val_pres": val_pres,
        "upd_attr": upd_attr,
        "mod_corr": mod_corr,
        "t_corr": t_corr,
        "t_pre_echo": int(t_pre_echo),
        "t_q1": int(t_q1),
        "t_echo1": int(t_echo1),
        "t_q2": int(t_q2),
        "t_echo2": int(t_echo2),
    }


def verify_phase3_counterfactual_oracle(
    batch: dict[str, Any], codebook: dict[str, np.ndarray], spec: SLWMv3Spec
) -> dict[str, Any]:
    """Analytic 100% causal oracle decoding strictly from `pkts[:t_pre_echo]` (ZERO future echo access!)."""
    t_pre = int(batch["t_pre_echo"])
    y_a = np.asarray(batch["y_a"])
    y_b = np.asarray(batch["y_b"])
    upd_attr = np.asarray(batch["upd_attr"])
    t_corr = np.asarray(batch["t_corr"])
    b_sz = y_a.shape[0]
    v_sz = int(codebook["vocab_size"])
    half_c = spec.audio_codebooks // 2

    def _decode_stream(pkts: dict[str, jax.Array]) -> np.ndarray:
        mod_id = np.asarray(pkts["modality_id"])[:t_pre]
        vid = np.asarray(pkts["video_feat"])[:t_pre]
        aud_c = np.asarray(pkts["audio_codes"])[:t_pre]
        txt = np.asarray(pkts["text_byte"])[:t_pre]

        preds = np.full((b_sz, 2), -1, dtype=np.int32)
        for b in range(b_sz):
            state = [-1, -1]
            for t in range(t_pre):
                m = int(mod_id[t, b])
                if m == 0:
                    a_id = int(np.argmax(codebook["vid_attr_base"] @ vid[t, b]))
                    v_id = int(np.argmax(codebook["vid_val_base"] @ vid[t, b]))
                elif m == 1:
                    a_scores = np.sum(codebook["aud_attr_codes"] == aud_c[t, b, :half_c][None, :], axis=-1)
                    v_scores = np.sum(codebook["aud_val_codes"] == aud_c[t, b, half_c:][None, :], axis=-1)
                    a_id = int(np.argmax(a_scores))
                    v_id = int(np.argmax(v_scores))
                elif m == 2:
                    tb = int(txt[t, b])
                    idx = tb - 33
                    if 0 <= idx < 3 * v_sz:
                        a_id, v_id = idx // v_sz, idx % v_sz
                    else:
                        continue
                else:
                    continue
                if a_id in (0, 1):
                    state[a_id] = v_id
            # Output physical world state `[attr0_latest, attr1_latest]` matching `y_a` / `y_b`!
            preds[b, 0] = state[0]
            preds[b, 1] = state[1]
        return preds

    pa = _decode_stream(batch["pkts_a"])
    pb = _decode_stream(batch["pkts_b"])

    # Verify bit-exact A == B after t_corr and strictly before t_q1!
    max_post_corr_diff = 0.0
    for k in ("modality_id", "video_feat", "audio_codes", "text_byte", "probe_feat"):
        aa = np.asarray(batch["pkts_a"][k])
        bb = np.asarray(batch["pkts_b"][k])
        for b in range(b_sz):
            tc1 = int(t_corr[b]) + 1
            diff = float(np.max(np.abs(aa[tc1:t_pre, b].astype(np.float64) - bb[tc1:t_pre, b].astype(np.float64))))
            max_post_corr_diff = max(max_post_corr_diff, diff)

    b_idx = np.arange(b_sz)
    acc_a = float(np.mean((pa[:, 0] == y_a[:, 0]) & (pa[:, 1] == y_a[:, 1])))
    acc_b = float(np.mean((pb[:, 0] == y_b[:, 0]) & (pb[:, 1] == y_b[:, 1])))
    ab_flip_ok = (pa[:, 0] == y_a[:, 0]) & (pb[:, 0] == y_b[:, 0]) & (pa[:, 1] == y_a[:, 1]) & (pb[:, 1] == y_b[:, 1])
    ab_distinct_frac = float(np.mean(y_a[b_idx, upd_attr] != y_b[b_idx, upd_attr]))
    ab_pres_same_frac = float(np.mean(y_a[b_idx, 1 - upd_attr] == y_b[b_idx, 1 - upd_attr]))
    return {
        "oracle_joint_acc_a": acc_a,
        "oracle_joint_acc_b": acc_b,
        "oracle_ab_flip_and_preserve_acc": float(np.mean(ab_flip_ok)),
        "ab_target_flip_frac": ab_distinct_frac,
        "ab_preserved_equal_frac": ab_pres_same_frac,
        "post_corr_pre_echo_max_abs_diff": max_post_corr_diff,
        "oracle_pass": bool(
            acc_a == 1.0
            and acc_b == 1.0
            and ab_distinct_frac == 1.0
            and ab_pres_same_frac == 1.0
            and max_post_corr_diff == 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Phase 1: Actual Worst-Shape Profiling + Native BF16 Wan VAE Check
# ---------------------------------------------------------------------------


def _check_native_bf16_3d_conv_on_accelerator() -> dict[str, Any]:
    """Verify native JAX `bfloat16` 3D convolution matching Wan 2.1 VAE's 5-frame 192x320 tensor on TPU/accelerator.

    Note: Pretrained PyTorch `AutoencoderKLWan` checkpoint weights are NOT shipped inside the lightweight JAX
    TPU pilot tarball; full PyTorch/XLA or GPU Wan 2.1 BF16 weight parity is deferred to the pre-encoding stage.
    """
    try:
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 5, 192, 320, 3), dtype=jnp.bfloat16)
        k = jax.random.normal(jax.random.PRNGKey(1), (3, 3, 3, 3, 16), dtype=jnp.bfloat16) * jnp.asarray(0.05, dtype=jnp.bfloat16)
        t0 = time.perf_counter()
        y = jax.lax.conv_general_dilated(
            x, k, window_strides=(1, 2, 2), padding="SAME", dimension_numbers=("NDHWC", "DHWIO", "NDHWC")
        )
        y.block_until_ready()
        ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status": "verified_jax_bf16_conv3d",
            "device_kind": str(jax.devices()[0].device_kind),
            "output_shape": list(y.shape),
            "all_finite": bool(jnp.all(jnp.isfinite(y))),
            "wall_ms": round(ms, 2),
            "wan21_pretrained_weights_note": "deferred_to_preencode_stage_not_bundled_in_jax_pilot",
        }
    except Exception as exc:
        return {"status": "failed", "detail": str(exc)[:200]}


def run_phase_profile(*, smoke_cpu: bool = False) -> dict[str, Any]:
    """Phase 1: Profile actual worst training shapes AND Phase 3 microbatched eval readout with chunked remat."""
    if smoke_cpu:
        configs = (
            ("pilot_d256_phase3_shape", "pilot_d256", 2, 64, 16, 3),
            ("discovery_45m_shape", "tiny_unit", 2, 32, 8, 3),
        )
        eval_prof_shape = ("pilot_d256", 4, 64)
    else:
        configs = (
            ("pilot_d256_phase3_worst_train_B8_T896", "pilot_d256", 8, 896, 64, 10),
            ("discovery_45m_T512", "discovery_45m", 4, 512, 64, 12),
            ("confirmation_135m_T256", "confirmation_135m", 2, 256, 64, 10),
        )
        eval_prof_shape = ("pilot_d256", 16, 896)

    bf16_3d = _check_native_bf16_3d_conv_on_accelerator()
    out: dict[str, Any] = {
        "jax_version": jax.__version__,
        "optax_version": optax.__version__,
        "device_kind": str(jax.devices()[0].device_kind),
        "device_count": len(jax.devices()),
        "jax_bf16_conv3d_probe": bf16_3d,
        "profiles": {},
    }
    all_ok = bool(bf16_3d.get("status") == "verified_jax_bf16_conv3d" and bf16_3d.get("all_finite", False))
    for label, prof, bsz, slen, chunk_sz, steps in configs:
        spec = make_slwm_v3_spec(prof, backbone_type="slwm_v3_routed")
        r = run_pilot_train(
            spec,
            steps=steps,
            seq_len=slen,
            batch_size=bsz,
            val_batch_size=max(4, bsz),
            base_lr=1e-3,
            remat_step=True,
            remat_chunk_size=chunk_sz,
            seed=3101,
        )
        out["profiles"][label] = r
        all_ok = all_ok and bool(r["all_finite"])

    # Also profile Phase 3's microbatched dev eval readout shape (`B=16, T=896`) explicitly!
    ep_name, ep_b, ep_t = eval_prof_shape
    ep_spec = make_slwm_v3_spec(ep_name, backbone_type="slwm_v3_routed")
    ep_p = init_slwm_v3_params(jax.random.PRNGKey(3109), ep_spec)
    ep_pkts, _, _ = make_synthetic_multimodal_batch(ep_spec, seq_len=ep_t, batch_size=ep_b, seed=3109)
    t0 = time.perf_counter()
    z_ev, _, _ = jax.jit(lambda p, pk: scan_core(p, init_core_state(ep_spec, ep_b), pk, ep_spec, remat_step=False))(ep_p, ep_pkts)
    z_ev.block_until_ready()
    ev_finite = bool(jnp.all(jnp.isfinite(z_ev)))
    out["phase3_microbatch_eval_profile"] = {
        "profile": ep_name,
        "batch_size": ep_b,
        "seq_len": ep_t,
        "all_finite": ev_finite,
        "wall_sec": round(time.perf_counter() - t0, 3),
    }
    all_ok = all_ok and ev_finite
    out["status"] = "pass" if all_ok else "fail"
    return out


# ---------------------------------------------------------------------------
# Phase 2: Fixed-Depth 3-Width muP + Separate 3-Depth CompleteP LR Transfer
# ---------------------------------------------------------------------------


def evaluate_lr_transfer_verdict(
    grid: dict[str, dict[str, Any]],
    scale_keys: list[str],
    lrs: tuple[float, ...],
    *,
    max_regret_threshold: float = 0.015,
) -> dict[str, Any]:
    """Pure decision function for Phase 2 LR transfer (strictly unrounded, rejecting flat/boundary/non-finite grids!)."""
    best_lr_by_scale: dict[str, float | None] = {}
    flat_flags: dict[str, bool] = {}
    boundary_flags: dict[str, bool] = {}
    nonfinite_counts: dict[str, int] = {}

    for sc_k in scale_keys:
        finite_losses: list[tuple[float, float]] = []
        n_nf = 0
        for lr in lrs:
            r = grid[sc_k][f"lr_{lr:.0e}"]
            val_ex = float(r.get("val_loss_end_exact", float("nan")))
            if bool(r.get("all_finite", False)) and np.isfinite(val_ex):
                finite_losses.append((lr, val_ex))
            else:
                n_nf += 1
        nonfinite_counts[sc_k] = n_nf
        if not finite_losses:
            best_lr_by_scale[sc_k] = None
            flat_flags[sc_k] = True
            boundary_flags[sc_k] = True
        else:
            losses_only = [v for _, v in finite_losses]
            min_loss = min(losses_only)
            best_lr, _ = min(finite_losses, key=lambda kv: kv[1])
            best_lr_by_scale[sc_k] = float(best_lr)
            flat_flags[sc_k] = (max(losses_only) - min_loss) < 1e-3
            # Inspect ALL tied minimizing cells so a tie hitting a grid endpoint is flagged as boundary!
            min_lrs = [lr for lr, v in finite_losses if v <= min_loss + 1e-12]
            boundary_flags[sc_k] = any((lr == lrs[0]) or (lr == lrs[-1]) for lr in min_lrs)

    base_k = scale_keys[0]
    base_lr_star = best_lr_by_scale[base_k]
    transfer_regrets_exact: dict[str, float] = {}
    transferred_cell_finite = True
    for sc_k in scale_keys[1:]:
        if base_lr_star is None or best_lr_by_scale[sc_k] is None:
            transfer_regrets_exact[sc_k] = float("inf")
            transferred_cell_finite = False
        else:
            r_trans = grid[sc_k][f"lr_{base_lr_star:.0e}"]
            r_opt = grid[sc_k][f"lr_{best_lr_by_scale[sc_k]:.0e}"]
            v_trans = float(r_trans.get("val_loss_end_exact", float("nan")))
            v_opt = float(r_opt.get("val_loss_end_exact", float("nan")))
            if bool(r_trans.get("all_finite", False)) and np.isfinite(v_trans) and np.isfinite(v_opt):
                transfer_regrets_exact[sc_k] = v_trans - v_opt
            else:
                transfer_regrets_exact[sc_k] = float("inf")
                transferred_cell_finite = False

    any_flat = any(flat_flags.values())
    any_missing = any(v is None for v in best_lr_by_scale.values())
    any_boundary = any(boundary_flags.values())
    any_nonfinite = any(c > 0 for c in nonfinite_counts.values())
    max_regret_exact = max(transfer_regrets_exact.values()) if transfer_regrets_exact else 0.0
    exact_match = (not any_missing) and (len(set(best_lr_by_scale.values())) == 1)

    # Strict decision: flat, ANY scale hitting a grid boundary, missing optima, or ANY non-finite cell -> inconclusive!
    # Unrounded `float64` comparison strictly against `max_regret_threshold`!
    if any_flat or any_missing or any_boundary or (not transferred_cell_finite) or any_nonfinite:
        status = "inconclusive"
    elif exact_match or (np.isfinite(max_regret_exact) and max_regret_exact <= max_regret_threshold):
        status = "pass"
    else:
        status = "fail"

    return {
        "best_lr_by_scale": best_lr_by_scale,
        "flat_curve_flags": flat_flags,
        "boundary_optimum_flags": boundary_flags,
        "nonfinite_cell_counts": nonfinite_counts,
        "transfer_regret_exact_from_base": transfer_regrets_exact,
        "exact_lr_match": exact_match,
        "status": status,
        "grid": grid,
    }


def _evaluate_lr_transfer_grid(
    specs_by_key: dict[str, SLWMv3Spec],
    lrs: tuple[float, ...],
    *,
    steps: int,
    seq_len: int,
    batch_size: int,
    val_batch_size: int,
    use_mup_lr_scaling: bool,
    seed: int,
) -> dict[str, Any]:
    grid: dict[str, dict[str, Any]] = {}
    for sc_k, spec in specs_by_key.items():
        params0 = init_slwm_v3_params(jax.random.PRNGKey(seed), spec)
        compiled = compile_pilot_steps(
            params0,
            spec,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            use_mup_lr_scaling=use_mup_lr_scaling,
            remat_step=True,
            remat_chunk_size=min(64, seq_len),
        )
        val_b = make_synthetic_multimodal_batch(spec, seq_len=seq_len, batch_size=val_batch_size, seed=seed + 999_999)
        grid[sc_k] = {}
        for lr in lrs:
            lr_k = f"lr_{lr:.0e}"
            grid[sc_k][lr_k] = run_pilot_train(
                spec,
                steps=steps,
                seq_len=seq_len,
                batch_size=batch_size,
                val_batch_size=val_batch_size,
                base_lr=lr,
                use_mup_lr_scaling=use_mup_lr_scaling,
                remat_step=True,
                remat_chunk_size=min(64, seq_len),
                seed=seed,
                compiled_bundle=compiled,
                val_batch=val_b,
            )

    return evaluate_lr_transfer_verdict(grid, list(specs_by_key.keys()), lrs)


def combine_phase2_verdict(
    width_mup: dict[str, Any],
    depth_completep: dict[str, Any],
    widths: tuple[int, ...],
    *,
    max_regret_threshold: float = 0.015,
) -> tuple[float, dict[str, float], str]:
    """Select `recommended_base_lr` and verify unrounded regret `<= max_regret_threshold` on BOTH width AND depth grids!"""
    base_w_key = f"d{widths[0]}"
    rec_lr_opt = width_mup["best_lr_by_scale"].get(base_w_key)
    rec_lr = float(rec_lr_opt) if rec_lr_opt is not None else 1e-3
    rec_k = f"lr_{rec_lr:.0e}"

    depth_regret_at_rec: dict[str, float] = {}
    depth_ok = True
    for dk, opt_lr in depth_completep["best_lr_by_scale"].items():
        if opt_lr is None or rec_k not in depth_completep["grid"][dk]:
            depth_regret_at_rec[dk] = float("inf")
            depth_ok = False
        else:
            r_rec = depth_completep["grid"][dk][rec_k]
            r_opt = depth_completep["grid"][dk][f"lr_{opt_lr:.0e}"]
            v_rec = float(r_rec.get("val_loss_end_exact", float("nan")))
            v_opt = float(r_opt.get("val_loss_end_exact", float("nan")))
            if bool(r_rec.get("all_finite", False)) and np.isfinite(v_rec) and np.isfinite(v_opt):
                reg = v_rec - v_opt
                depth_regret_at_rec[dk] = reg
                if reg > max_regret_threshold:
                    depth_ok = False
            else:
                depth_regret_at_rec[dk] = float("inf")
                depth_ok = False

    if "inconclusive" in (width_mup["status"], depth_completep["status"]):
        overall = "inconclusive"
    elif width_mup["status"] == "pass" and depth_completep["status"] == "pass" and depth_ok:
        overall = "pass"
    else:
        overall = "fail"
    return rec_lr, depth_regret_at_rec, overall


def run_phase_mup_check(*, smoke_cpu: bool = False, pilot_bundle_sha256: str | None = None) -> dict[str, Any]:
    """Phase 2: Separate width (L=12, d in {256,512,768}) AND depth (d=256, L in {8,12,16}) 5-LR transfer check."""
    if smoke_cpu:
        widths = (128, 256)
        depths = (2, 4)
        lrs = (3e-4, 1e-3, 3e-3)
        steps, seq_len, bsz, vbsz = 6, 24, 2, 4
        base_prof = "tiny_unit"
        fixed_d = 128
        fixed_l = 4
    else:
        widths = (256, 512, 768)
        depths = (8, 12, 16)
        lrs = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
        steps, seq_len, bsz, vbsz = 60, 64, 4, 16
        base_prof = "discovery_45m"
        fixed_d = 256
        fixed_l = 12

    width_specs = {
        f"d{d}": make_slwm_v3_spec(base_prof, backbone_type="slwm_v3_routed", d_model_override=d, n_layers_override=fixed_l)
        for d in widths
    }
    depth_specs = {
        f"L{l}": make_slwm_v3_spec(base_prof, backbone_type="slwm_v3_routed", d_model_override=fixed_d, n_layers_override=l)
        for l in depths
    }

    width_mup = _evaluate_lr_transfer_grid(
        width_specs, lrs, steps=steps, seq_len=seq_len, batch_size=bsz, val_batch_size=vbsz, use_mup_lr_scaling=True, seed=3201
    )
    width_sp = _evaluate_lr_transfer_grid(
        width_specs, lrs, steps=steps, seq_len=seq_len, batch_size=bsz, val_batch_size=vbsz, use_mup_lr_scaling=False, seed=3201
    )
    depth_completep = _evaluate_lr_transfer_grid(
        depth_specs, lrs, steps=steps, seq_len=seq_len, batch_size=bsz, val_batch_size=vbsz, use_mup_lr_scaling=True, seed=3202
    )

    chosen_lr, depth_regret_at_rec, overall_status = combine_phase2_verdict(width_mup, depth_completep, widths)
    return {
        "device_kind": str(jax.devices()[0].device_kind),
        "smoke_cpu": bool(smoke_cpu),
        "pilot_bundle_sha256": pilot_bundle_sha256,
        "widths": list(widths),
        "depths": list(depths),
        "lrs": list(lrs),
        "width_mup": width_mup,
        "width_sp_control": width_sp,
        "depth_completep": depth_completep,
        "depth_regret_at_recommended_lr": depth_regret_at_rec,
        "recommended_base_lr": float(chosen_lr),
        "status": overall_status,
    }


# ---------------------------------------------------------------------------
# Phase 3: Oracle-Verified Counterfactual Dev Ablation Across Matched Pilot Arms
# ---------------------------------------------------------------------------


def run_phase_dev_ablation(base_lr: float = 1e-3, *, smoke_cpu: bool = False) -> dict[str, Any]:
    """Phase 3: Paired counterfactual A/B flip + preservation + stress ablation at `pilot_d256`."""
    if smoke_cpu:
        profile = "tiny_unit"
        arms = ("slwm_v3_routed", "slwm_v3_no_routing", "explicit_bounded_kv", "sliding_window_gqa_eqbyte")
        vocab_sz = 4
        t_len, min_delay = 80, 34
        n_train_batches, train_bsz, epochs = 2, 4, 1
        dev_bsz, probe_steps, remat_chunk = 8, 12, 16
        eval_micro_bsz = 4
    else:
        profile = "pilot_d256"
        arms = (
            "slwm_v3_routed",
            "slwm_v3_no_routing",
            "slwm_v3_no_nowrite",
            "slwm_v3_no_zerolag",
            "single_bank_gated_deltanet",
            "diag_ssm",
            "explicit_bounded_kv",
            "sliding_window_gqa_eqparam",
            "sliding_window_gqa_eqbyte",
        )
        vocab_sz = 8
        # T=896 (14 * 64 chunks), delay >= 768 > RF_eqbyte = 756 > RF_eqparam = 372!
        t_len, min_delay = 896, 768
        n_train_batches, train_bsz, epochs = 32, 8, 3  # 256 train episodes x 3 epochs
        dev_bsz, probe_steps, remat_chunk = 128, 120, 64
        eval_micro_bsz = 16

    matched_specs = build_wave_a_comparator_specs(profile, match_params_to_anchor=True)
    anchor_spec = matched_specs["slwm_v3_routed"]
    anchor_params = count_tree_params(jax.eval_shape(lambda k: init_slwm_v3_params(k, anchor_spec), jax.random.PRNGKey(0)))
    codebook = build_frozen_phase3_codebook(anchor_spec, vocab_size=vocab_sz, codebook_seed=202601)

    # Pre-build frozen-codebook train & dev batches and verify 100% analytic oracle FIRST!
    train_batches = [
        generate_phase3_counterfactual_batch(
            anchor_spec,
            codebook,
            batch_size=train_bsz,
            t_len=t_len,
            min_delay_after_corr=min_delay,
            stress_mode=("clean", "distractor", "corrupt")[i % 3],
            seed=4000 + i,
        )
        for i in range(n_train_batches)
    ]
    dev_batches = {
        sm: generate_phase3_counterfactual_batch(
            anchor_spec,
            codebook,
            batch_size=dev_bsz,
            t_len=t_len,
            min_delay_after_corr=min_delay,
            stress_mode=sm,
            seed=9901,  # Identical episode draw across clean/distractor/corrupt for paired stress drops!
        )
        for sm in ("clean", "distractor", "corrupt")
    }
    oracle_clean = verify_phase3_counterfactual_oracle(dev_batches["clean"], codebook, anchor_spec)
    oracle_dist = verify_phase3_counterfactual_oracle(dev_batches["distractor"], codebook, anchor_spec)
    oracle_corr = verify_phase3_counterfactual_oracle(dev_batches["corrupt"], codebook, anchor_spec)
    if not (oracle_clean["oracle_pass"] and oracle_dist["oracle_pass"] and oracle_corr["oracle_pass"]):
        raise RuntimeError(f"Phase 3 oracle check failed: {oracle_clean} / {oracle_dist} / {oracle_corr}")

    t_pre_echo = int(dev_batches["clean"]["t_pre_echo"])
    arm_results: dict[str, Any] = {}
    for arm in arms:
        spec = matched_specs[arm]
        params = init_slwm_v3_params(jax.random.PRNGKey(3301), spec)
        n_p = count_tree_params(params)
        p_delta_pct = (n_p - anchor_params) / float(anchor_params) * 100.0
        state_b = compute_core_streaming_state_bytes(spec, batch_size=1)
        assert abs(p_delta_pct) <= 1.0, f"Pilot arm {arm} param delta {p_delta_pct:+.2f}% exceeds ±1.0%"

        probe_p = init_decoupled_probe_params(jax.random.PRNGKey(3302), spec.d_model, out_dim=2 * vocab_sz, hidden_dim=128)
        tx_core_unit = build_optimizer(params, spec, base_lr=1.0, use_mup_lr_scaling=True)
        tx_probe = optax.adamw(learning_rate=3e-3, weight_decay=1e-4)
        opt_core = tx_core_unit.init(params)
        opt_probe = tx_probe.init(probe_p)

        @jax.jit(donate_argnums=(0, 1))
        def core_step(p, o_st, pkts, targs, tvalid, step_lr_arr):
            b = pkts["valid_mask"].shape[1]
            st0 = init_core_state(spec, b)
            (loss, m), g = jax.value_and_grad(
                lambda pp: masked_multi_horizon_predictive_loss(
                    pp, st0, pkts, targs, tvalid, spec, remat_step=True, remat_chunk_size=remat_chunk
                ),
                has_aux=True,
            )(p)
            unit_upd, new_o = tx_core_unit.update(g, o_st, p)
            upd = jax.tree_util.tree_map(lambda u: u * step_lr_arr.astype(u.dtype), unit_upd)
            return optax.apply_updates(p, upd), new_o, loss, m

        @jax.jit
        def _extract_pre_echo_z_micro(p, pkts_mb):
            """Extract strictly pre-echo latent `Z[t_pre_echo]` on a bounded microbatch (ZERO future echo leak!)."""
            b = pkts_mb["valid_mask"].shape[1]
            pkts_pre = {k: v[: t_pre_echo + 1] for k, v in pkts_mb.items()}
            z_seq, _, diags = scan_core(p, init_core_state(spec, b), pkts_pre, spec, remat_step=False)
            return jax.lax.stop_gradient(z_seq[t_pre_echo]), {k: jnp.mean(v) for k, v in diags.items() if k != "layer_rms"}

        def extract_pre_echo_z(p, pkts):
            """Chunk large dev batches (`B=128`) into profiled `eval_micro_bsz` (`B=16`) slices so peak HBM stays bounded."""
            b_tot = int(pkts["valid_mask"].shape[1])
            if b_tot <= eval_micro_bsz:
                return _extract_pre_echo_z_micro(p, pkts)
            z_parts, diag_parts = [], []
            for start in range(0, b_tot, eval_micro_bsz):
                end = min(start + eval_micro_bsz, b_tot)
                mb = {k: v[:, start:end] for k, v in pkts.items()}
                zm, dm = _extract_pre_echo_z_micro(p, mb)
                z_parts.append(zm)
                diag_parts.append(dm)
            mean_diags = {k: jnp.mean(jnp.stack([d[k] for d in diag_parts])) for k in diag_parts[0]}
            return jnp.concatenate(z_parts, axis=0), mean_diags

        @jax.jit
        def probe_step(pp, o_st, z_f, y_pair):
            def _loss(pr):
                logits = apply_decoupled_probe(pr, z_f).reshape(-1, 2, vocab_sz)
                l1 = optax.softmax_cross_entropy_with_integer_labels(logits[:, 0], y_pair[:, 0]).mean()
                l2 = optax.softmax_cross_entropy_with_integer_labels(logits[:, 1], y_pair[:, 1]).mean()
                return 0.5 * (l1 + l2)

            loss, g = jax.value_and_grad(_loss)(pp)
            upd, new_o = tx_probe.update(g, o_st, pp)
            return optax.apply_updates(pp, upd), new_o, loss

        # 1. Self-supervised predictive core training on observable sensory stream + future echoes (ZERO downstream labels!)
        core_losses: list[float] = []
        last_m: dict[str, Any] = {}
        total_core_steps = epochs * len(train_batches)
        step_counter = 0
        for ep in range(epochs):
            for bi, tb in enumerate(train_batches):
                use_b = ((ep + bi) % 2) == 1
                pkts = tb["pkts_b"] if use_b else tb["pkts_a"]
                targs = tb["targs_b"] if use_b else tb["targs_a"]
                tvalid = tb["tvalid_b"] if use_b else tb["tvalid_a"]
                step_lr = jnp.asarray(base_lr * lr_schedule_mult(step_counter, total_core_steps), dtype=jnp.float32)
                params, opt_core, l_val, last_m = core_step(params, opt_core, pkts, targs, tvalid, step_lr)
                step_counter += 1
                core_losses.append(float(l_val))

        # 2. Strictly frozen-core decoupled probe training on pre-echo `Z[t_pre_echo]` (both A and B train views)
        z_tr_list, y_tr_list = [], []
        for tb in train_batches:
            za, _ = extract_pre_echo_z(params, tb["pkts_a"])
            zb, _ = extract_pre_echo_z(params, tb["pkts_b"])
            z_tr_list.extend([za, zb])
            y_tr_list.extend([tb["y_a"], tb["y_b"]])
        z_tr = jnp.concatenate(z_tr_list, axis=0)
        y_tr = jnp.concatenate(y_tr_list, axis=0)
        probe_losses: list[float] = []
        for _ in range(probe_steps):
            probe_p, opt_probe, pl_val = probe_step(probe_p, opt_probe, z_tr, y_tr)
            probe_losses.append(float(pl_val))

        # 3. Evaluate BOTH `stream_a` AND counterfactual `stream_b` flip + preservation + modality breakdown + strict finiteness!
        stress_scores: dict[str, dict[str, Any]] = {}
        eval_tensors_finite = bool(jnp.all(jnp.isfinite(z_tr)))
        for sm in ("clean", "distractor", "corrupt"):
            db = dev_batches[sm]
            za, diag_a = extract_pre_echo_z(params, db["pkts_a"])
            zb, diag_b = extract_pre_echo_z(params, db["pkts_b"])
            logits_a = apply_decoupled_probe(probe_p, za).reshape(-1, 2, vocab_sz)
            logits_b = apply_decoupled_probe(probe_p, zb).reshape(-1, 2, vocab_sz)
            eval_tensors_finite = (
                eval_tensors_finite
                and bool(jnp.all(jnp.isfinite(za)))
                and bool(jnp.all(jnp.isfinite(zb)))
                and bool(jnp.all(jnp.isfinite(logits_a)))
                and bool(jnp.all(jnp.isfinite(logits_b)))
                and all(bool(jnp.isfinite(v)) for v in diag_a.values())
                and all(bool(jnp.isfinite(v)) for v in diag_b.values())
            )
            pa = np.asarray(jnp.argmax(logits_a, axis=-1))
            pb = np.asarray(jnp.argmax(logits_b, axis=-1))
            ya = np.asarray(db["y_a"])
            yb = np.asarray(db["y_b"])
            ua = np.asarray(db["upd_attr"])
            mc = np.asarray(db["mod_corr"])
            b_idx = np.arange(ya.shape[0])

            upd_ok_a = pa[b_idx, ua] == ya[b_idx, ua]
            upd_ok_b = pb[b_idx, ua] == yb[b_idx, ua]
            pres_ok_a = pa[b_idx, 1 - ua] == ya[b_idx, 1 - ua]
            pres_ok_b = pb[b_idx, 1 - ua] == yb[b_idx, 1 - ua]
            ok_a = upd_ok_a & pres_ok_a
            ok_b = upd_ok_b & pres_ok_b
            ab_joint = ok_a & ok_b
            by_mod: dict[str, float] = {}
            for m_idx, m_name in ((0, "video"), (1, "audio"), (2, "text")):
                sel = mc == m_idx
                by_mod[m_name] = round(float(np.mean(ab_joint[sel])), 4) if np.any(sel) else 0.0

            stress_scores[sm] = {
                "joint_acc_a": round(float(np.mean(ok_a)), 4),
                "joint_acc_b": round(float(np.mean(ok_b)), 4),
                "updated_flip_ab_acc": round(float(np.mean(upd_ok_a & upd_ok_b)), 4),
                "preserved_ab_acc": round(float(np.mean(pres_ok_a & pres_ok_b)), 4),
                "counterfactual_ab_flip_and_preserve": round(float(np.mean(ab_joint)), 4),
                "ab_flip_by_corr_modality": by_mod,
                "mean_nowrite": round(float(diag_a["mean_nowrite"]), 4),
                "mean_eff_write_eta": round(float(diag_a["mean_eff_write_eta"]), 5),
                "mean_eff_decay": round(float(diag_a["mean_eff_decay"]), 5),
                "mean_state_delta_norm": round(float(diag_a["mean_state_delta_norm"]), 5),
            }

        def _all_numeric_finite(obj: Any) -> bool:
            if isinstance(obj, dict):
                return all(_all_numeric_finite(v) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return all(_all_numeric_finite(v) for v in obj)
            if isinstance(obj, (int, float)) and not isinstance(obj, bool):
                return bool(np.isfinite(obj))
            return True

        c_ab = stress_scores["clean"]["counterfactual_ab_flip_and_preserve"]
        d_ab = stress_scores["distractor"]["counterfactual_ab_flip_and_preserve"]
        k_ab = stress_scores["corrupt"]["counterfactual_ab_flip_and_preserve"]
        arm_finite = bool(
            all(np.isfinite(x) for x in core_losses)
            and all(np.isfinite(x) for x in probe_losses)
            and eval_tensors_finite
            and _all_numeric_finite(stress_scores)
        )
        arm_results[arm] = {
            "trainable_core_params": int(n_p),
            "param_delta_vs_anchor_pct": round(p_delta_pct, 3),
            "core_total_state_bytes": int(state_b["total_core_state_bytes"]),
            "stacked_window_rf_ticks": int(spec.stacked_window_receptive_field_ticks),
            "core_pred_loss_init": round(core_losses[0], 6),
            "core_pred_loss_end": round(core_losses[-1], 6),
            "probe_loss_init": round(probe_losses[0], 6),
            "probe_loss_end": round(probe_losses[-1], 6),
            "arm_all_finite": arm_finite,
            "stress_metrics": stress_scores,
            "distractor_drop_pp": round((c_ab - d_ab) * 100.0, 2),
            "corrupt_drop_pp": round((c_ab - k_ab) * 100.0, 2),
        }

    all_arms_finite = all(bool(r["arm_all_finite"]) for r in arm_results.values())
    return {
        "profile": profile,
        "t_len": int(t_len),
        "min_delay_after_corr": int(min_delay),
        "t_pre_echo": int(t_pre_echo),
        "vocab_size": int(vocab_sz),
        "chance_joint_ab_flip_baseline": round(1.0 / float(vocab_sz * (vocab_sz - 1) * vocab_sz), 6),
        "oracle_verification": {"clean": oracle_clean, "distractor": oracle_dist, "corrupt": oracle_corr},
        "arms": arm_results,
        "all_arms_finite": all_arms_finite,
        "status": "pass" if all_arms_finite else "fail",
    }


# ---------------------------------------------------------------------------
# Phase 4: 2-Seed Repeatability & Long-Stream Stability on TRAINED Weights
# ---------------------------------------------------------------------------


def run_phase_long_stream(base_lr: float = 1e-3, *, smoke_cpu: bool = False) -> dict[str, Any]:
    """Phase 4: 2-seed repeatability & `T=256->2048` stability audit on BOTH trained seed checkpoints!"""
    if smoke_cpu:
        spec = make_slwm_v3_spec("tiny_unit", backbone_type="slwm_v3_routed")
        steps, seq_len, bsz, vbsz = 6, 24, 2, 4
        horizons = (32, 64)
    else:
        spec = make_slwm_v3_spec("discovery_45m", backbone_type="slwm_v3_routed")
        steps, seq_len, bsz, vbsz = 40, 64, 4, 16
        horizons = (256, 512, 1024, 2048)

    seeds = (3401, 3402)
    seed_runs: dict[str, Any] = {}
    trained_horizon_audits: dict[str, Any] = {}
    all_ok = True

    for s in seeds:
        r = run_pilot_train(
            spec,
            steps=steps,
            seq_len=seq_len,
            batch_size=bsz,
            val_batch_size=vbsz,
            base_lr=base_lr,
            remat_step=True,
            remat_chunk_size=min(64, seq_len),
            seed=s,
            return_params=True,
        )
        trained_params = r.pop("_trained_params")
        seed_runs[f"s{s}"] = r
        all_ok = all_ok and bool(r["all_finite"])

        # Audit long-horizon stability on the ACTUAL TRAINED CHECKPOINT weights!
        h_audit: dict[str, Any] = {}
        ref_state_bytes = compute_core_streaming_state_bytes(spec, batch_size=1)["total_core_state_bytes"]
        for t_len in horizons:
            rng = np.random.default_rng(s * 100 + t_len)
            pkts = {
                "modality_id": jnp.full((t_len, 1), 4, dtype=jnp.int32),
                "dt_s": jnp.full((t_len, 1), 0.08, dtype=jnp.float32),
                "valid_mask": jnp.ones((t_len, 1), dtype=jnp.float32),
                "reliability": jnp.ones((t_len, 1), dtype=jnp.float32),
                "probe_feat": jnp.asarray(rng.normal(0.0, 1.0, size=(t_len, 1, spec.probe_dim)), dtype=jnp.float32),
            }
            st0 = init_core_state(spec, 1)
            t0 = time.perf_counter()
            z_seq, st_end, diags = jax.jit(lambda p, st, pk: scan_core(p, st, pk, spec))(trained_params, st0, pkts)
            z_seq.block_until_ready()
            dt_s = time.perf_counter() - t0
            state_bytes = sum(int(x.size) * int(x.dtype.itemsize) for x in jax.tree_util.tree_leaves(st_end))
            finite_z = bool(jnp.all(jnp.isfinite(z_seq)))
            finite_st = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(st_end))
            zero_growth = state_bytes == ref_state_bytes
            all_ok = all_ok and finite_z and finite_st and zero_growth
            h_audit[f"T{t_len}"] = {
                "all_finite_outputs": finite_z,
                "all_finite_state": finite_st,
                "exact_zero_state_growth": zero_growth,
                "z_rms": round(float(jnp.sqrt(jnp.mean(z_seq * z_seq))), 5),
                "effective_rank_pr": round(float(effective_rank_participation_ratio(z_seq)), 5),
                "mean_key_norm": round(float(jnp.mean(diags["mean_key_norm"])), 6),
                "mean_nowrite": round(float(jnp.mean(diags["mean_nowrite"])), 4),
                "final_state_bytes": int(state_bytes),
                "wall_sec": round(dt_s, 3),
            }
        trained_horizon_audits[f"s{s}"] = h_audit

    return {
        "seed_repeatability": seed_runs,
        "trained_checkpoint_long_stream_stability": trained_horizon_audits,
        "status": "pass" if all_ok else "fail",
    }


def is_tpu_v6e_device_kind(device_kind: str) -> bool:
    """Recognize strictly exact official JAX 0.12 TPU v6e identifiers (`TPU v6e` and `TPU v6 lite`)."""
    return device_kind.strip().lower() in {"tpu v6e", "tpu v6 lite"}


def validate_reusable_mup_receipt(
    mup_receipt_path: Path,
    *,
    expected_bundle_sha256: str | None = None,
    require_tpu_v6e: bool = True,
) -> float:
    """Strictly validate a saved Phase 2 receipt (status, production grid, exact TPU v6e, bundle digest, on-grid finite LR)."""
    if not mup_receipt_path.is_file():
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: missing {mup_receipt_path}")
    data = json.loads(mup_receipt_path.read_text())
    if not isinstance(data, dict) or data.get("status") != "pass":
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: status={data.get('status')!r} (expected 'pass')")
    if data.get("smoke_cpu") is not False:
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: smoke_cpu={data.get('smoke_cpu')!r} (expected False)")
    if require_tpu_v6e and not is_tpu_v6e_device_kind(str(data.get("device_kind", ""))):
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: invalid device_kind={data.get('device_kind')!r}")
    if data.get("widths") != [256, 512, 768] or data.get("depths") != [8, 12, 16]:
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: non-production grid widths={data.get('widths')} depths={data.get('depths')}")
    rec_sha = data.get("pilot_bundle_sha256")
    if not isinstance(rec_sha, str) or len(rec_sha) != 64:
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: missing/invalid pilot_bundle_sha256={rec_sha!r}")
    if expected_bundle_sha256 is not None and rec_sha != expected_bundle_sha256:
        raise RuntimeError(
            f"PHASE 2 PREREQUISITE LOCK: bundle digest mismatch ({rec_sha!r} != {expected_bundle_sha256!r})"
        )
    lr_raw = data.get("recommended_base_lr")
    lrs_grid = data.get("lrs")
    if (
        isinstance(lr_raw, bool)
        or (not isinstance(lr_raw, (int, float)))
        or (not np.isfinite(float(lr_raw)))
        or float(lr_raw) <= 0.0
    ):
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: invalid recommended_base_lr={lr_raw!r}")
    if not isinstance(lrs_grid, list) or not any(
        (not isinstance(x, bool)) and isinstance(x, (int, float)) and abs(float(x) - float(lr_raw)) <= 1e-12
        for x in lrs_grid
    ):
        raise RuntimeError(f"PHASE 2 PREREQUISITE LOCK: recommended_base_lr={lr_raw!r} not in lrs grid {lrs_grid!r}")
    return float(lr_raw)


def main() -> None:
    ap = argparse.ArgumentParser(description="Execute SLWM-v3 TPU v6e-1 4-phase validation pilot suite (Rev 4).")
    ap.add_argument("--phases", type=str, default="profile,mup_check,dev_ablation,long_stream")
    ap.add_argument("--base-lr", type=float, default=None, help="Explicit base LR for smoke_cpu mode only.")
    ap.add_argument("--bundle-sha256", type=str, default=None, help="Approved pilot_bundle_sha256 bound to Phase 2 receipt.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for per-phase + combined JSON receipts.")
    ap.add_argument("--smoke-cpu", action="store_true", help="Run compact CPU smoke mode (0.00 CU verification).")
    ap.add_argument("--require-tpu-v6e", action="store_true", help="Abort if active JAX backend is not TPU v6e / v6 lite.")
    args = ap.parse_args()

    dev_kind = str(jax.devices()[0].device_kind)
    if args.require_tpu_v6e and (not is_tpu_v6e_device_kind(dev_kind)):
        raise RuntimeError(f"STRICT TPU v6e POLICY VIOLATION: active JAX device is {dev_kind!r}, expected TPU v6e / TPU v6 lite!")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [p.strip() for p in args.phases.split(",") if p.strip()]

    # Non-smoke dispatch guard: ALWAYS enforce Phase 2 prerequisite & validated LR (never allow off-grid override!)
    chosen_lr = float(args.base_lr) if (args.smoke_cpu and args.base_lr is not None) else 1e-3
    needs_mup = [p for p in wanted if p in ("dev_ablation", "long_stream")]
    if (not args.smoke_cpu) and needs_mup:
        runs_mup_first = ("mup_check" in wanted) and (wanted.index("mup_check") < min(wanted.index(p) for p in needs_mup))
        if not runs_mup_first:
            if not args.bundle_sha256:
                raise RuntimeError("PHASE 2 PREREQUISITE LOCK: --bundle-sha256 is required for standalone non-smoke Phases 3/4!")
            chosen_lr = validate_reusable_mup_receipt(
                args.out_dir / "phase_mup_check_receipt.json",
                expected_bundle_sha256=args.bundle_sha256,
                require_tpu_v6e=True,
            )

    report: dict[str, Any] = {
        "jax_version": jax.__version__,
        "optax_version": optax.__version__,
        "device_kind": dev_kind,
        "device_count": len(jax.devices()),
        "smoke_cpu": bool(args.smoke_cpu),
        "pilot_bundle_sha256": args.bundle_sha256,
        "phases_requested": wanted,
        "phases_completed": [],
        "halted_reason": None,
        "results": {},
    }

    for ph in wanted:
        if ph == "profile":
            res = run_phase_profile(smoke_cpu=args.smoke_cpu)
        elif ph == "mup_check":
            res = run_phase_mup_check(smoke_cpu=args.smoke_cpu, pilot_bundle_sha256=args.bundle_sha256)
            if (not args.smoke_cpu) or (args.base_lr is None):
                chosen_lr = float(res["recommended_base_lr"])
        elif ph == "dev_ablation":
            res = run_phase_dev_ablation(base_lr=chosen_lr, smoke_cpu=args.smoke_cpu)
        elif ph == "long_stream":
            res = run_phase_long_stream(base_lr=chosen_lr, smoke_cpu=args.smoke_cpu)
        else:
            raise ValueError(f"Unknown pilot phase: {ph!r}")

        report["results"][ph] = res
        report["phases_completed"].append(ph)
        # Write individual phase receipt immediately so partial results survive any budget halt!
        (args.out_dir / f"phase_{ph}_receipt.json").write_text(json.dumps(res, indent=2) + "\n")

        if res.get("status") != "pass":
            report["halted_reason"] = f"Phase {ph!r} returned status={res.get('status')!r}; halting progression."
            break

    combined_path = args.out_dir / "pilot_suite_receipt.json"
    combined_path.write_text(json.dumps(report, indent=2) + "\n")
    # Print ONLY a compact (< 1 KiB) summary to stdout so SSH buffers never overflow!
    compact_summary = {
        "device_kind": dev_kind,
        "phases_completed": report["phases_completed"],
        "phase_statuses": {k: v.get("status") for k, v in report["results"].items()},
        "halted_reason": report["halted_reason"],
        "combined_receipt_path": str(combined_path),
    }
    print(json.dumps(compact_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
