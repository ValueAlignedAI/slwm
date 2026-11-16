"""Compiled JAX/XLA Training & Evaluation Runner for Study X2 Wave 8A (`SLWM-X2`).

Trains the 5-cohort Wave 8A `453M` matrix under strict parameter & data-exposure parity:
  1. `w8a_slwm_x2_isolated_450m`: `SLWM-X2` with `Z_mem ⊕ Z_task` head/gradient isolation + `assoc_ce`.
  2. `w8a_ablation_no_zmem_iso_450m`: Single-factor ablation (`zmem_isolation=False`) testing shared trunk interference.
  3. `w8a_ablation_spatial_control_only_450m`: Spatial-control-only ceiling anchor (`enable_assoc_loss=False`).
  4. `w8a_sliding_window_w64_gqa_450m`: Contemporary `9:1` GQA `SlidingWindow-W64-GQA` (`13.50 MiB` KV cache).
  5. `w8a_diagonal_ssm_450m`: Matched `DiagonalSSM` (`4.91 MiB` state, identical `H_sym ⊕ H_goal ⊕ H_phys` split).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from data.x1b_multimodal import (
    ASSOC_KEY_OFFSET,
    ASSOC_NUM_KEYS,
    ASSOC_NUM_VALS,
    ASSOC_VAL_OFFSET,
    RealMultimodalBank,
    generate_synthetic_world_batch,
)
from data.x1b_unseen_worlds import (
    mask_privileged_proprio,
    render_unseen_world_rgb,
    step_unseen_layout_world,
)
from evals.x1b_eval_wave8a import run_wave8a_evaluation
from evals.x1b_showcase_wave8a import collect_wave8a_showcase_bundle, save_showcase_bundle_npz
from models.x1b_wave7_core import (
    encode_goal_query_tokens,
    encode_goal_tokens_2d,
    quantize_goal_2d_val_id,
)
from models.x1b_wave8a_core import (
    Wave8AInterventions,
    convert_legacy_proprio_to_ee_only,
    convert_legacy_target_proprio,
    world_model_loss_wave8a,
)
from models.x1b_world import (
    compute_streaming_state_bytes,
    count_parameters,
    init_x1b_params,
    make_x1b_spec,
)
from scripts.train.x1b_train import (
    atomic_write_json,
    load_bf16_weights,
    memory_efficient_adamw_step,
    save_bf16_weights,
    save_portable_int8,
    sha256_file,
)
from scripts.train.x1b_train_wave7 import Wave7GroundedRobotBank, mix_wave7_batches


class Wave8AGroundedRobotBank:
    """Speaker- & session-disjoint physical `xArm6` bank with dual opposite-direction & same-scene wrong-color counterfactuals."""

    def __init__(self, npz_path: str | Path, split: str = "train") -> None:
        self.path = Path(npz_path)
        self.split = split
        with np.load(self.path, allow_pickle=False) as d:
            pfx = f"{split}_"
            required_keys = (
                "video",
                "spoken_audio",
                "wrong_spoken_audio",
                "wrong_color_spoken_audio",
                "ee_xy",
                "action_xy",
                "length",
                "true_text_ids",
                "wrong_text_ids",
                "wrong_color_text_ids",
                "cf_index",
                "scene_cluster_id",
                "dir_id",
                "color_id",
                "distractor_color_id",
                "target_block_xy",
                "distractor_block_xy",
            )
            missing = [pfx + k for k in required_keys if (pfx + k) not in d]
            if missing:
                raise KeyError(
                    f"Wave 8A grounded robot bank {self.path} is missing required verified v2 fields: {missing}"
                )
            self.video = np.asarray(d[pfx + "video"], dtype=np.uint8)
            self.spoken_audio = np.asarray(d[pfx + "spoken_audio"], dtype=np.float32)
            self.wrong_spoken_audio = np.asarray(d[pfx + "wrong_spoken_audio"], dtype=np.float32)
            self.wrong_color_spoken_audio = np.asarray(d[pfx + "wrong_color_spoken_audio"], dtype=np.float32)
            self.ee_xy = np.asarray(d[pfx + "ee_xy"], dtype=np.float32)
            self.action_xy = np.asarray(d[pfx + "action_xy"], dtype=np.float32)
            self.length = np.asarray(d[pfx + "length"], dtype=np.int32)
            self.true_txt16 = np.asarray(d[pfx + "true_text_ids"], dtype=np.int32)
            self.wrong_txt16 = np.asarray(d[pfx + "wrong_text_ids"], dtype=np.int32)
            self.wrong_color_txt16 = np.asarray(d[pfx + "wrong_color_text_ids"], dtype=np.int32)
            self.cf_index = np.asarray(d[pfx + "cf_index"], dtype=np.int64)
            self.scene_cluster_id = np.asarray(d[pfx + "scene_cluster_id"], dtype=np.int64)
            self.dir_id = np.asarray(d[pfx + "dir_id"], dtype=np.int32)
            self.color_id = np.asarray(d[pfx + "color_id"], dtype=np.int32)
            self.distractor_color_id = np.asarray(d[pfx + "distractor_color_id"], dtype=np.int32)
            self.target_block_xy = np.asarray(d[pfx + "target_block_xy"], dtype=np.float32)
            self.distractor_block_xy = np.asarray(d[pfx + "distractor_block_xy"], dtype=np.float32)

        self.size = int(self.video.shape[0])
        if np.any((self.cf_index < 0) | (self.cf_index >= self.size)):
            raise ValueError(f"Invalid cf_index values out of [0, {self.size}) in {self.path}")
        opp_map = np.array([1, 0, 3, 2], dtype=np.int32)
        if not np.array_equal(self.dir_id[self.cf_index], opp_map[self.dir_id]):
            raise ValueError(f"Counterfactual dir_id in {self.path} violates strict opposite-direction contract")
        if not np.array_equal(self.wrong_txt16, self.true_txt16[self.cf_index]):
            raise ValueError(f"wrong_text_ids in {self.path} does not match true_text_ids[cf_index]")
        if np.any(self.distractor_color_id == self.color_id):
            raise ValueError(f"Same-scene wrong-color counterfactual in {self.path} matches instructed color_id")
        if not np.allclose(self.wrong_color_spoken_audio[:, 16:32], self.spoken_audio[:, 16:32], atol=1e-6):
            raise ValueError(
                f"wrong_color_spoken_audio[:, 16:32] in {self.path} must preserve identical direction audio"
            )
        if np.any(np.linalg.norm(self.target_block_xy - self.distractor_block_xy, axis=-1) < 0.175):
            raise ValueError(f"target_block_xy and distractor_block_xy in {self.path} violate minimum separation")

    def sample_batch(self, rng: np.random.Generator, batch_size: int, seq_len: int = 32) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=(batch_size,))
        t = min(seq_len, 32)
        vid = self.video[idx, :t]
        aud = self.spoken_audio[idx, :t]
        wrong_aud = self.wrong_spoken_audio[idx, :t]
        wrong_col_aud = self.wrong_color_spoken_audio[idx, :t]
        ee = self.ee_xy[idx, :t]
        u2 = self.action_xy[idx, :t]
        ln = np.clip(self.length[idx], 2, t)
        tok16 = self.true_txt16[idx]
        wtok16 = self.wrong_txt16[idx]
        wcol_tok16 = self.wrong_color_txt16[idx]
        targ_bxy = self.target_block_xy[idx]
        dist_bxy = self.distractor_block_xy[idx]

        txt = np.zeros((batch_size, t, 4), dtype=np.int32)
        wtxt = np.zeros((batch_size, t, 4), dtype=np.int32)
        wcol_txt = np.zeros((batch_size, t, 4), dtype=np.int32)
        h44 = tok16.reshape(batch_size, 4, 4)
        wh44 = wtok16.reshape(batch_size, 4, 4)
        wch44 = wcol_tok16.reshape(batch_size, 4, 4)
        transient = rng.random((batch_size,)) < 0.65
        for bi in range(batch_size):
            if transient[bi]:
                txt[bi, : min(4, t)] = h44[bi, : min(4, t)]
                wtxt[bi, : min(4, t)] = wh44[bi, : min(4, t)]
                wcol_txt[bi, : min(4, t)] = wch44[bi, : min(4, t)]
            else:
                reps = (t + 3) // 4
                txt[bi] = np.tile(h44[bi], (reps, 1))[:t]
                wtxt[bi] = np.tile(wh44[bi], (reps, 1))[:t]
                wcol_txt[bi] = np.tile(wch44[bi], (reps, 1))[:t]

        ee_vel = np.zeros_like(ee)
        ee_vel[:, 1:] = ee[:, 1:] - ee[:, :-1]
        moving = (np.linalg.norm(ee_vel, axis=-1) > 0.015).astype(np.float32)
        prop = np.concatenate(
            [
                ee,
                ee,
                ee_vel,
                np.zeros((batch_size, t, 1), dtype=np.float32),
                moving[..., None],
                np.zeros((batch_size, t, 8), dtype=np.float32),
            ],
            axis=-1,
        )
        next_prop = np.concatenate([prop[:, 1:], prop[:, -1:]], axis=1)
        u4 = np.concatenate([u2, np.zeros_like(u2)], axis=-1)
        prev_u4 = np.concatenate([np.zeros((batch_size, 1, 4), dtype=np.float32), u4[:, :-1]], axis=1)

        end_xy = np.array([ee[bi, ln[bi] - 1] for bi in range(batch_size)], dtype=np.float32)
        disp_xy = np.clip((end_xy - ee[:, 0]) * 1.25, -0.95, 0.95).astype(np.float32)
        dist = np.sqrt(np.sum((ee - end_xy[:, None, :]) ** 2, axis=-1))
        val = (-1.35 * dist).astype(np.float32)
        tgoal = np.broadcast_to(disp_xy[:, None, :], (batch_size, t, 2)).copy().astype(np.float32)
        zero_goal = np.zeros((batch_size, 4), dtype=np.float32)
        true_goal = np.concatenate([disp_xy, end_xy], axis=-1).astype(np.float32)
        con = np.stack([moving, np.zeros_like(moving)], axis=-1)
        valid = np.arange(t)[None, :] < ln[:, None]
        active = (valid & (np.linalg.norm(u2, axis=-1) > 1e-3)).astype(np.float32)
        mm = np.ones((batch_size, t, 4), dtype=np.float32)

        return {
            "video": vid,
            "clean_video": vid,
            "audio": aud,
            "wrong_audio": wrong_aud,
            "wrong_color_audio": wrong_col_aud,
            "proprio": prop,
            "target_proprio": prop,
            "target_next_proprio": next_prop,
            "text": txt,
            "wrong_text": wtxt,
            "wrong_color_text": wcol_txt,
            "target_text": np.zeros((batch_size, t), dtype=np.int32),
            "target_block_xy": targ_bxy,
            "distractor_block_xy": dist_bxy,
            "action": prev_u4,
            "next_action": u4,
            "target_action": u4,
            "target_contacts": con,
            "target_value": val,
            "target_commit": np.zeros((batch_size, t), dtype=np.int32),
            "target_goal_xy": tgoal,
            "goal": zero_goal,
            "true_goal": true_goal,
            "modality_mask": mm,
            "full_modality_mask": mm,
            "has_av_pair": np.ones((batch_size,), dtype=np.float32),
            "action_supervision_mask": active,
            "valid": valid,
            "scenario": np.full((batch_size,), -2, dtype=np.int32),
        }


def build_multi_item_assoc_tokens_t64(
    rng_key: jax.Array,
    batch_size: int,
    seq_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Synthesize concurrent `K in {1, 2, 4, 8, 16}` associative writes + selective overwrites + balanced queries.

    For `seq_len >= 64` (Wave 8A training horizon):
      - Ticks `0 .. 15`: write `K` distinct key-value bindings `[0, k_j, v_init_j, 0]` (`j < K`).
      - Ticks `16 .. 23`: selectively overwrite `M = max(1, K // 2)` (`<= 8`) keys `[0, k_j, v_over_j, 0]`
        (`v_over != v_init`).
      - Ticks `seq_len - 8 .. seq_len - 5` (`56 .. 59` at `T = 64`, lag `>= 33 > 28`-tick 4-layer `W = 8` RoPE span):
        query `min(M, 4)` overwritten keys `[0, k_q, 0, 2]`.
      - Ticks `seq_len - 4 .. seq_len - 1` (`60 .. 63` at `T = 64`, lag `>= 45 > 28`):
        query `min(K - M, 4)` retained keys `[0, k_q, 0, 2]`.
    For `32 <= seq_len < 64`, queries occupy `seq_len - 8 .. seq_len - 1`; for `seq_len < 32` (unit tests),
    uses a compact `K = 2` schedule.
    """
    if seq_len < 8:
        return jnp.zeros((batch_size, seq_len, 4), dtype=jnp.int32), jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    k_ksel, k_perm, k_vi, k_vo, k_qord, k_qn = jax.random.split(rng_key, 6)
    all_keys = jnp.arange(ASSOC_KEY_OFFSET, ASSOC_KEY_OFFSET + ASSOC_NUM_KEYS, dtype=jnp.int32)
    perm_keys = jax.vmap(lambda rk: jax.random.permutation(rk, all_keys))(jax.random.split(k_perm, batch_size))
    v_init = jax.random.randint(k_vi, (batch_size, ASSOC_NUM_KEYS), ASSOC_VAL_OFFSET, ASSOC_VAL_OFFSET + ASSOC_NUM_VALS)
    v_off = jax.random.randint(k_vo, (batch_size, ASSOC_NUM_KEYS), 1, ASSOC_NUM_VALS)
    v_over = ASSOC_VAL_OFFSET + ((v_init - ASSOC_VAL_OFFSET + v_off) % ASSOC_NUM_VALS)

    if seq_len < 32:
        t_ov = seq_len // 2
        txt = jnp.zeros((batch_size, seq_len, 4), dtype=jnp.int32)
        ttxt = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
        txt = txt.at[:, 1, 1].set(perm_keys[:, 0]).at[:, 1, 2].set(v_init[:, 0])
        txt = txt.at[:, 2, 1].set(perm_keys[:, 1]).at[:, 2, 2].set(v_init[:, 1])
        txt = txt.at[:, t_ov, 1].set(perm_keys[:, 0]).at[:, t_ov, 2].set(v_over[:, 0])
        txt = txt.at[:, seq_len - 2, 1].set(perm_keys[:, 0]).at[:, seq_len - 2, 3].set(2)
        ttxt = ttxt.at[:, seq_len - 2].set(v_over[:, 0])
        txt = txt.at[:, seq_len - 1, 1].set(perm_keys[:, 1]).at[:, seq_len - 1, 3].set(2)
        ttxt = ttxt.at[:, seq_len - 1].set(v_init[:, 1])
        return txt, ttxt

    k_choices = jnp.array([1, 2, 4, 8, 16], dtype=jnp.int32)
    k_per_ep = k_choices[jax.random.randint(k_ksel, (batch_size,), 0, 5)]  # [B]
    m_per_ep = jnp.where(k_per_ep == 1, 1, k_per_ep // 2)  # [B] in {1, 1, 2, 4, 8}
    r_per_ep = k_per_ep - m_per_ep  # [B] in {0, 1, 2, 4, 8}
    safe_r = jnp.maximum(r_per_ep, 1)

    txt = jnp.zeros((batch_size, seq_len, 4), dtype=jnp.int32)
    ttxt = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    k_qn_coin, k_qn_w, k_qn_ov = jax.random.split(k_qn, 3)
    use_quad_prefix = jax.random.bernoulli(k_qn_coin, 0.5, (batch_size,))
    quad_w = jnp.where(use_quad_prefix[:, None], jax.random.randint(k_qn_w, (batch_size, 16), 4, 8), 0)
    quad_ov = jnp.where(use_quad_prefix[:, None], jax.random.randint(k_qn_ov, (batch_size, 8), 4, 8), 0)

    # 1. Ticks 0..15: write K initial bindings (with 50% quadrant-token slot-0 invariance augmentation)
    for j in range(16):
        active_w = j < k_per_ep
        qj = jnp.where(active_w, quad_w[:, j], 0)
        kj = jnp.where(active_w, perm_keys[:, j], 0)
        vj = jnp.where(active_w, v_init[:, j], 0)
        txt = txt.at[:, j, 0].set(qj).at[:, j, 1].set(kj).at[:, j, 2].set(vj)

    # 2. Ticks 16..23: selectively overwrite M bindings
    for j in range(8):
        active_ov = j < m_per_ep
        qoj = jnp.where(active_ov, quad_ov[:, j], 0)
        kj = jnp.where(active_ov, perm_keys[:, j], 0)
        voj = jnp.where(active_ov, v_over[:, j], 0)
        txt = txt.at[:, 16 + j, 0].set(qoj).at[:, 16 + j, 1].set(kj).at[:, 16 + j, 2].set(voj)

    # 3. Ticks (seq_len - 8)..(seq_len - 1): all 8 beyond-window query ticks active on 100% of episodes
    q_base = seq_len - 8
    q_shifts = jax.random.randint(k_qord, (batch_size, 8), 0, 16)
    b_idx = jnp.arange(batch_size)
    for qi in range(4):
        ov_slot = (qi + q_shifts[:, qi]) % m_per_ep
        kq = perm_keys[b_idx, ov_slot]
        vq = v_over[b_idx, ov_slot]
        txt = txt.at[:, q_base + qi, 1].set(kq).at[:, q_base + qi, 3].set(2)
        ttxt = ttxt.at[:, q_base + qi].set(vq)

        ret_slot = m_per_ep + ((qi + q_shifts[:, 4 + qi]) % safe_r)
        ov_fb_slot = (4 + qi + q_shifts[:, 4 + qi]) % m_per_ep
        has_ret = r_per_ep > 0
        kqr = jnp.where(has_ret, perm_keys[b_idx, ret_slot], perm_keys[b_idx, ov_fb_slot])
        vqr = jnp.where(has_ret, v_init[b_idx, ret_slot], v_over[b_idx, ov_fb_slot])
        txt = txt.at[:, q_base + 4 + qi, 1].set(kqr).at[:, q_base + 4 + qi, 3].set(2)
        ttxt = ttxt.at[:, q_base + 4 + qi].set(vqr)

    return txt, ttxt


build_multi_item_assoc_tokens_t32 = build_multi_item_assoc_tokens_t64


def generate_in_domain_deprivileged_sim_batch_v8a(
    rng_key: jax.Array,
    batch_size: int,
    seq_len: int = 64,
    vocab_size: int = 4096,
) -> dict[str, jax.Array]:
    """Generate strictly in-domain (`open` or single `x=0` bar) de-privileged spatial + multi-item episodes.

    Enforces `ee_only` proprioception across 100% of episodes (`rep` and `base[n_u:]`), synthesizes
    causal contact/motion audio for `rep`, and places `T = 64` associative overwrite queries strictly
    beyond the 28-tick 4-layer `W = 8` local RoPE attention span.
    """
    k_base, k_w, k_mode, k_bar, k_exp, k_mi_u, k_mi_b, k_aud, k_flt = jax.random.split(rng_key, 9)
    base = generate_synthetic_world_batch(k_base, batch_size, seq_len=seq_len, vocab_size=vocab_size)

    n_u = (batch_size + 1) // 2
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(k_w, 7)
    side = jnp.where(jax.random.bernoulli(k1, 0.5, (n_u,)), 1.0, -1.0)
    obj0 = jnp.stack([-0.54 * side, jax.random.uniform(k2, (n_u,), minval=-0.38, maxval=0.38)], axis=-1)

    exp_u = jax.random.uniform(k_exp, (n_u,))
    is_free_sweep = exp_u < 0.25
    is_wall_collide = (~is_free_sweep) & (exp_u < 0.48)
    is_exploratory = is_free_sweep | is_wall_collide

    ee_near = jnp.clip(
        obj0 + jnp.stack([-0.16 * side, jax.random.uniform(k3, (n_u,), minval=-0.12, maxval=0.12)], axis=-1),
        -0.88,
        0.88,
    )
    ee_free = jax.random.uniform(k7, (n_u, 2), minval=-0.78, maxval=0.78)
    ee0 = jnp.where(is_free_sweep[:, None], ee_free, ee_near)

    targ_a = jnp.stack([0.54 * side, jax.random.uniform(k4, (n_u,), minval=-0.42, maxval=0.42)], axis=-1)
    targ_b = jnp.stack([0.54 * side, jax.random.uniform(k5, (n_u,), minval=-0.42, maxval=0.42)], axis=-1)
    dist_xy = jax.random.uniform(k6, (n_u, 2), minval=-0.65, maxval=0.65)

    mode_u = jax.random.uniform(k_mode, (n_u,))
    is_switch_ep = (mode_u < 0.44) & (seq_len >= 20)
    is_trans_ep = (~is_switch_ep) & (mode_u < 0.56)
    is_ext_goal_ep = (~is_switch_ep) & (~is_trans_ep)
    has_bar = jax.random.bernoulli(k_bar, 0.72, (n_u,))

    k_hx, k_hy = jax.random.split(k_bar)
    bar_hx = jax.random.uniform(k_hx, (n_u,), minval=0.022, maxval=0.085)
    bar_hy = jax.random.uniform(k_hy, (n_u,), minval=0.20, maxval=0.38)
    cx = jnp.zeros((n_u, 3), dtype=jnp.float32)
    cy = jnp.zeros((n_u, 3), dtype=jnp.float32)
    hx = jnp.stack([jnp.where(has_bar, bar_hx, 0.0), jnp.zeros((n_u,)), jnp.zeros((n_u,))], axis=-1)
    hy = jnp.stack([jnp.where(has_bar, bar_hy, 0.0), jnp.zeros((n_u,)), jnp.zeros((n_u,))], axis=-1)

    st0 = {
        "ee_xy": ee0,
        "ee_vel": jnp.zeros((n_u, 2), dtype=jnp.float32),
        "obj_xy": obj0,
        "obj_vel": jnp.zeros((n_u, 2), dtype=jnp.float32),
        "box_cx": cx,
        "box_cy": cy,
        "box_hx": hx,
        "box_hy": hy,
        "mass_scale": jnp.ones((n_u,), dtype=jnp.float32),
        "damping": jnp.full((n_u,), 0.72, dtype=jnp.float32),
    }

    tok_ga = encode_goal_tokens_2d(targ_a, is_distractor=False)
    tok_gb = encode_goal_tokens_2d(targ_b, is_distractor=False)
    tok_dist = encode_goal_tokens_2d(dist_xy, is_distractor=True)
    tok_query = encode_goal_query_tokens((n_u,))
    val_ga = quantize_goal_2d_val_id(targ_a)
    val_gb = quantize_goal_2d_val_id(targ_b)

    t_sw = 16 if seq_len >= 64 else 12
    t_dist = 32 if seq_len >= 64 else 20
    sweep_ph = jnp.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False, dtype=jnp.float32)
    freqs = jnp.linspace(0.0, 1.0, 64, dtype=jnp.float32)[None, :]
    init_aud = jnp.full((n_u, 64), -0.35, dtype=jnp.float32)

    def sim_step(carry, t_idx):
        st, prev_u, prev_c, prev_aud, rk = carry
        rk, nk1, nk2 = jax.random.split(rk, 3)

        active_targ = jnp.where(is_switch_ep[:, None] & (t_idx >= t_sw), targ_b, targ_a)
        active_val = jnp.where(is_switch_ep & (t_idx >= t_sw), val_gb, val_ga)

        write_a = t_idx < 4
        write_b = is_switch_ep & (t_idx >= t_sw) & (t_idx < t_sw + 4)
        write_d = is_switch_ep & (t_idx >= t_dist) & (t_idx < t_dist + 3)

        show_g = write_a | write_b | is_ext_goal_ep
        txt_t = jnp.where(
            write_a,
            tok_ga,
            jnp.where(write_b[:, None], tok_gb, jnp.where(write_d[:, None], tok_dist, tok_query)),
        )
        is_query_tick = (~write_a) & (~write_b) & (~write_d) & (is_switch_ep | is_trans_ep)
        ttxt_t = jnp.where(is_query_tick, active_val, 0)

        vid_t = render_unseen_world_rgb(st, active_targ, jnp.broadcast_to(show_g, (n_u,)))
        ee_p = mask_privileged_proprio(st["ee_xy"], st["ee_vel"], prev_c)
        full_tprop = ee_p.at[:, 2:4].set(st["obj_xy"]).at[:, 6:8].set(st["obj_vel"])

        det_y = jnp.where(st["obj_xy"][:, 1] >= 0.0, 0.54, -0.54)
        blocked = has_bar & ((st["obj_xy"][:, 0] * active_targ[:, 0]) < 0.0) & (jnp.abs(st["obj_xy"][:, 1]) < 0.39)
        wp = jnp.where(blocked[:, None], jnp.stack([jnp.zeros((n_u,)), det_y], axis=-1), active_targ)
        to_wp = wp - st["obj_xy"]
        u_dir = to_wp * jax.lax.rsqrt(jnp.sum(to_wp * to_wp, axis=-1, keepdims=True) + 1e-6)
        stage = st["obj_xy"] - 0.17 * u_dir
        rel = st["ee_xy"] - st["obj_xy"]
        aligned = jnp.sum(rel * u_dir, axis=-1, keepdims=True) <= -0.08
        aim = jnp.where(aligned, st["obj_xy"] + 0.08 * u_dir, stage)
        u_expert_xy = jnp.clip(3.2 * (aim - st["ee_xy"]) + 0.10 * jax.random.normal(nk1, (n_u, 2)), -1.0, 1.0)

        to_direct = active_targ - st["obj_xy"]
        dir_d = to_direct * jax.lax.rsqrt(jnp.sum(to_direct * to_direct, axis=-1, keepdims=True) + 1e-6)
        rel_d = st["ee_xy"] - st["obj_xy"]
        al_d = jnp.sum(rel_d * dir_d, axis=-1, keepdims=True) <= -0.08
        aim_d = jnp.where(al_d, st["obj_xy"] + 0.08 * dir_d, st["obj_xy"] - 0.17 * dir_d)
        u_collide_xy = jnp.clip(3.2 * (aim_d - st["ee_xy"]) + 0.15 * jax.random.normal(nk2, (n_u, 2)), -1.0, 1.0)

        sw_goal = jnp.stack(
            [0.70 * jnp.cos(sweep_ph + 0.27 * t_idx), 0.70 * jnp.sin(sweep_ph + 0.23 * t_idx)],
            axis=-1,
        )
        u_sweep_xy = jnp.clip(3.2 * (sw_goal - st["ee_xy"]), -1.0, 1.0)

        u_exec_xy = jnp.where(
            is_free_sweep[:, None],
            u_sweep_xy,
            jnp.where(is_wall_collide[:, None], u_collide_xy, u_expert_xy),
        )
        u_exec_4 = jnp.concatenate([u_exec_xy, jnp.zeros_like(u_exec_xy)], axis=-1)
        u_exp_4 = jnp.concatenate([u_expert_xy, jnp.zeros_like(u_expert_xy)], axis=-1)

        st_next, c_next = step_unseen_layout_world(st, u_exec_4)
        next_tprop = (
            mask_privileged_proprio(st_next["ee_xy"], st_next["ee_vel"], c_next)
            .at[:, 2:4]
            .set(st_next["obj_xy"])
            .at[:, 6:8]
            .set(st_next["obj_vel"])
        )
        near_box = has_bar & (
            ((jnp.abs(st_next["obj_xy"][:, 0]) < bar_hx + 0.11) & (jnp.abs(st_next["obj_xy"][:, 1]) < bar_hy + 0.06))
            | ((jnp.abs(st_next["ee_xy"][:, 0]) < bar_hx + 0.09) & (jnp.abs(st_next["ee_xy"][:, 1]) < bar_hy + 0.05))
        )
        bar_ev = near_box.astype(jnp.float32)
        con_vec = jnp.stack([c_next, bar_ev], axis=-1)
        speed = jnp.sqrt(jnp.sum(st_next["obj_vel"] * st_next["obj_vel"], axis=-1, keepdims=True) + 1e-6)
        obj_tone = c_next[:, None] * jnp.exp(-((freqs - 0.28) ** 2) / 0.018) * (1.2 + 2.0 * speed)
        wall_tone = bar_ev[:, None] * jnp.exp(-((freqs - 0.72) ** 2) / 0.012) * 2.2
        roll_hum = 0.25 * speed * jnp.exp(-freqs * 3.0)
        aud_next = obj_tone + wall_tone + roll_hum - 0.35
        val_t = -1.35 * jnp.sqrt(jnp.sum((st["obj_xy"] - active_targ) ** 2, axis=-1) + 1e-6) - 0.65 * bar_ev
        return (st_next, u_exec_4, c_next, aud_next, rk), (
            vid_t,
            prev_aud,
            ee_p,
            full_tprop,
            next_tprop,
            txt_t,
            ttxt_t,
            prev_u,
            u_exec_4,
            u_exp_4,
            con_vec,
            val_t,
            active_targ,
        )

    _, (v_s, aud_s, p_s, tp_s, ntp_s, txt_s, ttxt_s, pu_s, uexec_s, uexp_s, c_s, val_s, tg_s) = jax.lax.scan(
        sim_step,
        (st0, jnp.zeros((n_u, 4)), jnp.zeros((n_u,)), init_aud, k_bar),
        jnp.arange(seq_len, dtype=jnp.int32),
    )
    mv = lambda a: jnp.swapaxes(a, 0, 1)
    v_bt = mv(v_s)
    p_bt = mv(p_s)
    aud_bt = mv(aud_s) + 0.025 * jax.random.normal(k_aud, (n_u, seq_len, 64), dtype=jnp.float32)
    txt_u = mv(txt_s)
    ttxt_u = mv(ttxt_s)

    # Strictly in-domain sensor fault calibration (mode 1 = 50% 180-deg flip / 50% constant camera mount extrinsic
    # translation offset of +-6..12 px; mode 3 = RGB static + ee_only spike) applied ONLY to exploratory non-associative
    # episodes after tick 24 (`clean_video` remains uncorrupted `v_bt`).
    if seq_len >= 32:
        k_fc, k_ft, k_fv, k_fp, k_fsh = jax.random.split(k_flt, 5)
        can_fault = is_exploratory & (~is_ext_goal_ep)
        flt_u = jax.random.uniform(k_fc, (n_u,))
        is_m1 = can_fault & (flt_u < 0.48)
        is_m3 = can_fault & (~is_m1) & (flt_u < 0.82)
        t_flt = jax.random.randint(k_ft, (n_u, 1), 24, max(25, seq_len - 8))
        flt_active = jnp.arange(seq_len, dtype=jnp.int32)[None, :] >= t_flt
        m1_mask = is_m1[:, None] & flt_active
        m3_mask = is_m3[:, None] & flt_active
        k_sh_kind, k_sh_mag, k_sh_sgn = jax.random.split(k_fsh, 3)
        use_flip_m1 = jax.random.uniform(k_sh_kind, (n_u,)) < 0.50
        sh_mag = jax.random.randint(k_sh_mag, (n_u, 2), 6, 13)
        sh_sgn = jnp.where(jax.random.uniform(k_sh_sgn, (n_u, 2)) < 0.5, -1, 1)
        sh_yx = sh_mag * sh_sgn
        v_pad = jnp.pad(v_bt, ((0, 0), (0, 0), (12, 12), (12, 12), (0, 0)), mode="edge")
        h_v, w_v = v_bt.shape[2], v_bt.shape[3]
        v_shift = jax.vmap(
            lambda vp, dyx: jax.lax.dynamic_slice(
                vp, (0, 12 + dyx[0], 12 + dyx[1], 0), (seq_len, h_v, w_v, 3)
            )
        )(v_pad, sh_yx)
        v_m1 = jnp.where(use_flip_m1[:, None, None, None, None], jnp.flip(v_bt, axis=(2, 3)), v_shift)
        rgb_static = jax.random.randint(k_fv, v_bt.shape, 0, 256).astype(jnp.uint8)
        v_obs_bt = jnp.where(
            m1_mask[:, :, None, None, None],
            v_m1,
            jnp.where(m3_mask[:, :, None, None, None], rgb_static, v_bt),
        )
        sp_xy = jnp.clip(p_bt[:, :, 0:2] + 0.85 * jnp.sign(jax.random.normal(k_fp, (n_u, seq_len, 2))), -1.5, 1.5)
        sp_vel = jnp.clip(p_bt[:, :, 4:6] + 0.65 * jax.random.normal(k_fp, (n_u, seq_len, 2)), -1.5, 1.5)
        p_m3 = p_bt.at[:, :, 0:2].set(sp_xy).at[:, :, 2:4].set(sp_xy).at[:, :, 4:6].set(sp_vel)
        p_obs_bt = jnp.where(m3_mask[:, :, None], p_m3, p_bt)
        t_commit_u = jnp.where(m1_mask, 1, jnp.where(m3_mask, 3, 0)).astype(jnp.int32)
    else:
        v_obs_bt = v_bt
        p_obs_bt = p_bt
        t_commit_u = jnp.zeros((n_u, seq_len), dtype=jnp.int32)

    # Inject K in {1, 2, 4, 8, 16} multi-item associative binding + selective overwrite onto scen == 0 (`is_ext_goal_ep`)
    mi_txt_u, mi_ttxt_u = build_multi_item_assoc_tokens_t64(k_mi_u, n_u, seq_len)
    txt_u = jnp.where(is_ext_goal_ep[:, None, None], mi_txt_u % vocab_size, txt_u)
    ttxt_u = jnp.where(is_ext_goal_ep[:, None], mi_ttxt_u % vocab_size, ttxt_u)

    g_ext = jnp.where((is_switch_ep | is_trans_ep)[:, None], 0.0, targ_a)
    g4 = jnp.concatenate([jnp.zeros_like(g_ext), g_ext], axis=-1)
    scen_u = jnp.where(is_switch_ep, 5, jnp.where(is_trans_ep, 1, 0)).astype(jnp.int32)
    act_sup = jnp.broadcast_to((~is_exploratory)[:, None].astype(jnp.float32), (n_u, seq_len))
    mm_u = jnp.ones((n_u, seq_len, 4), dtype=jnp.float32)

    rep = {
        "video": v_obs_bt,
        "clean_video": v_bt,
        "audio": aud_bt,
        "proprio": p_obs_bt,
        "target_proprio": mv(tp_s),
        "target_next_proprio": mv(ntp_s),
        "text": txt_u,
        "target_text": ttxt_u,
        "action": mv(pu_s),
        "next_action": mv(uexec_s),
        "target_action": mv(uexp_s),
        "target_contacts": mv(c_s),
        "target_value": mv(val_s),
        "target_commit": t_commit_u,
        "target_goal_xy": mv(tg_s),
        "goal": g4,
        "true_goal": jnp.concatenate([jnp.zeros_like(targ_a), targ_a], axis=-1),
        "modality_mask": mm_u,
        "full_modality_mask": mm_u,
        "has_av_pair": jnp.ones((n_u,), dtype=jnp.float32),
        "action_supervision_mask": act_sup,
        "valid": jnp.ones((n_u, seq_len), dtype=bool),
        "scenario": scen_u,
    }

    n_rem = batch_size - n_u
    if n_rem == 0:
        return {
            **rep,
            "in_domain_bar_hx": jnp.where(has_bar, bar_hx, 0.0).astype(jnp.float32),
            "in_domain_bar_hy": jnp.where(has_bar, bar_hy, 0.0).astype(jnp.float32),
            "ee_only_proprio": jnp.ones((n_u,), dtype=bool),
        }

    # Strictly de-privilege legacy `base[n_u:]` proprioception (`ee_only`) and align target_proprio schema
    base_raw_p = base["proprio"][n_u:]
    base_prop_rem = convert_legacy_proprio_to_ee_only(base_raw_p)
    base_tprop_rem = convert_legacy_target_proprio(base["target_proprio"][n_u:])
    base_ntprop_rem = convert_legacy_target_proprio(
        base["target_next_proprio"][n_u:], prev_ee_xy=base_raw_p[:, :, 0:2]
    )

    # On 75% of non-transient `base[n_u:]` episodes, upgrade K=2 associative stream to K in {1, 2, 4, 8, 16};
    # retain 25% legacy K=2 `[quad_tok, k, v, 0]` episodes (`t_ov = 32, q_final = 63` at T=64 -> lag 31 > 28).
    k_mi_coin, k_mi_seq = jax.random.split(k_mi_b)
    mi_txt_b, mi_ttxt_b = build_multi_item_assoc_tokens_t64(k_mi_seq, n_rem, seq_len)
    use_mi_b = (base["scenario"][n_u:] != 1) & jax.random.bernoulli(k_mi_coin, 0.75, (n_rem,))
    base_txt_rem = jnp.where(use_mi_b[:, None, None], mi_txt_b % vocab_size, base["text"][n_u:])
    base_ttxt_rem = jnp.where(use_mi_b[:, None], mi_ttxt_b % vocab_size, base["target_text"][n_u:])

    out = {}
    for k, bv in base.items():
        if k == "text":
            out[k] = jnp.concatenate([rep["text"], base_txt_rem], axis=0)
        elif k == "target_text":
            out[k] = jnp.concatenate([rep["target_text"], base_ttxt_rem], axis=0)
        elif k == "proprio":
            out[k] = jnp.concatenate([rep["proprio"], base_prop_rem], axis=0)
        elif k == "target_proprio":
            out[k] = jnp.concatenate([rep["target_proprio"], base_tprop_rem], axis=0)
        elif k == "target_next_proprio":
            out[k] = jnp.concatenate([rep["target_next_proprio"], base_ntprop_rem], axis=0)
        elif k in rep:
            out[k] = jnp.concatenate([rep[k], bv[n_u:]], axis=0)
        else:
            out[k] = bv

    base_scen = base["scenario"][n_u:]
    base_bar = (base_scen >= 0) & (base_scen != 4) & (base["target_contacts"][n_u:, 0, 1] > 0.5)
    out["in_domain_bar_hx"] = jnp.concatenate(
        [jnp.where(has_bar, bar_hx, 0.0), jnp.where(base_bar, 0.025, 0.0)], axis=0
    ).astype(jnp.float32)
    out["in_domain_bar_hy"] = jnp.concatenate(
        [jnp.where(has_bar, bar_hy, 0.0), jnp.where(base_bar, 0.38, 0.0)], axis=0
    ).astype(jnp.float32)
    out["ee_only_proprio"] = jnp.ones((batch_size,), dtype=bool)
    return out


_TEMPORAL_BATCH_KEYS = (
    "video",
    "clean_video",
    "audio",
    "wrong_audio",
    "wrong_color_audio",
    "proprio",
    "target_proprio",
    "target_next_proprio",
    "text",
    "wrong_text",
    "wrong_color_text",
    "target_text",
    "action",
    "next_action",
    "target_action",
    "target_contacts",
    "target_value",
    "target_commit",
    "target_goal_xy",
    "modality_mask",
    "full_modality_mask",
    "action_supervision_mask",
    "valid",
)


def _pad_temporal_batch_to_seq_len(part: dict[str, Any], seq_len: int) -> dict[str, jax.Array]:
    """Right-pad temporal `[B, T_cur, ...]` tensors to `seq_len` with `valid[:, T_cur:] = False`."""
    t_cur = int(part["video"].shape[1])
    out = {k: jnp.asarray(v) for k, v in part.items()}
    if t_cur >= seq_len:
        return out
    pad_t = seq_len - t_cur
    for k in _TEMPORAL_BATCH_KEYS:
        if k in out and out[k].ndim >= 2 and out[k].shape[1] == t_cur:
            arr = out[k]
            pad_spec = [(0, 0)] * arr.ndim
            pad_spec[1] = (0, pad_t)
            out[k] = jnp.pad(arr, pad_spec, mode="constant", constant_values=False if arr.dtype == jnp.bool_ else 0)
    return out


def mix_wave8a_batches(
    r_batch: dict[str, Any] | None,
    g_batch: dict[str, Any] | None,
    s_batch: dict[str, Any],
    *,
    seq_len: int = 64,
) -> dict[str, jax.Array]:
    """Normalize `ee_only` proprioception, mask passive-AV proprio to 0, right-pad 32-tick clips to `seq_len`, & concatenate."""
    norm_parts: list[dict[str, jax.Array]] = []
    if r_batch is not None:
        rp_raw = jnp.asarray(r_batch["proprio"], dtype=jnp.float32)
        rmm_p = (
            jnp.asarray(r_batch["modality_mask"], dtype=jnp.float32)[..., 2:3]
            if "modality_mask" in r_batch
            else jnp.ones((rp_raw.shape[0], rp_raw.shape[1], 1), dtype=jnp.float32)
        )
        rp_ee = convert_legacy_proprio_to_ee_only(rp_raw) * rmm_p
        rtp = convert_legacy_target_proprio(jnp.asarray(r_batch["target_proprio"], dtype=jnp.float32)) * rmm_p
        rntp = (
            convert_legacy_target_proprio(
                jnp.asarray(r_batch["target_next_proprio"], dtype=jnp.float32), prev_ee_xy=rp_raw[:, :, 0:2]
            )
            * rmm_p
        )
        r_norm = {
            **r_batch,
            "proprio": rp_ee,
            "target_proprio": rtp,
            "target_next_proprio": rntp,
            "ee_only_proprio": jnp.ones((rp_raw.shape[0],), dtype=bool),
        }
        norm_parts.append(_pad_temporal_batch_to_seq_len(r_norm, seq_len))

    if g_batch is not None:
        gp_raw = jnp.asarray(g_batch["proprio"], dtype=jnp.float32)
        g_norm = {
            **g_batch,
            "ee_only_proprio": jnp.ones((gp_raw.shape[0],), dtype=bool),
        }
        norm_parts.append(_pad_temporal_batch_to_seq_len(g_norm, seq_len))

    s_norm = _pad_temporal_batch_to_seq_len(s_batch, seq_len)
    if "ee_only_proprio" not in s_norm:
        s_norm["ee_only_proprio"] = jnp.ones((int(s_norm["video"].shape[0]),), dtype=bool)
    norm_parts.append(s_norm)

    mixed = mix_wave7_batches(*norm_parts)
    mixed["ee_only_proprio"] = jnp.concatenate([p["ee_only_proprio"] for p in norm_parts], axis=0)
    if g_batch is not None:
        gp_f32 = jnp.asarray(g_batch["proprio"], dtype=jnp.float32)
        mixed["rob_vid0"] = jnp.asarray(g_batch["video"][:, :1])
        mixed["rob_prop0"] = gp_f32[:, :1]
        mixed["rob_audio_32"] = jnp.asarray(g_batch["audio"], dtype=jnp.float32)
        mixed["rob_wrong_audio_32"] = jnp.asarray(g_batch["wrong_audio"], dtype=jnp.float32)
        mixed["rob_text_32"] = jnp.asarray(g_batch["text"], dtype=jnp.int32)
        mixed["rob_wrong_text_32"] = jnp.asarray(g_batch["wrong_text"], dtype=jnp.int32)
        mixed["rob_end_xy"] = jnp.asarray(g_batch["target_goal_xy"][:, 0, :], dtype=jnp.float32)
        if "modality_mask" in g_batch:
            mixed["rob_modality_mask"] = jnp.asarray(g_batch["modality_mask"], dtype=jnp.float32)
        if "wrong_color_audio" in g_batch and "target_block_xy" in g_batch:
            mixed["rob_wrong_col_audio_32"] = jnp.asarray(g_batch["wrong_color_audio"], dtype=jnp.float32)
            mixed["rob_wrong_col_text_32"] = jnp.asarray(g_batch["wrong_color_text"], dtype=jnp.int32)
            mixed["rob_target_block_xy"] = jnp.asarray(g_batch["target_block_xy"], dtype=jnp.float32)
            mixed["rob_distractor_block_xy"] = jnp.asarray(g_batch["distractor_block_xy"], dtype=jnp.float32)
    return mixed


def verify_wave8a_dataset_roles_fail_closed(
    real_bank_path: str | Path | None,
    grounded_robot_path: str | Path | None,
    vault_robot_path: str | Path | None,
    *,
    smoke: bool,
    cards_cfg_path: Path | None = None,
) -> dict[str, Any]:
    """Role-bound, filename-independent dataset SHA verification (fail-closed whenever `smoke=False`)."""
    cfg_p = cards_cfg_path if cards_cfg_path is not None else (REPO_ROOT / "configs/x1b/wave8a_dataset_cards.json")
    role_specs = (
        ("real_bank_sha256", real_bank_path, "x1b_real_bank_v2"),
        ("grounded_robot_sha256", grounded_robot_path, "x1b_wave8a_grounded_v2"),
        ("vault_robot_sha256", vault_robot_path, "x1b_wave8a_vault_v2"),
    )
    out: dict[str, Any] = {}
    for prov_k, p_val, _ in role_specs:
        p_obj = Path(p_val) if p_val else None
        out[prov_k] = sha256_file(p_obj) if (p_obj is not None and p_obj.is_file()) else None

    if smoke:
        out["registered_cards_verified"] = False
        return out

    if not cfg_p.is_file():
        raise RuntimeError(f"Non-smoke Wave 8A training requires frozen dataset card registry at {cfg_p}")
    reg_cards = json.loads(cfg_p.read_text())
    for prov_k, p_val, card_k in role_specs:
        if card_k not in reg_cards or not isinstance(reg_cards[card_k], dict) or "sha256" not in reg_cards[card_k]:
            raise RuntimeError(f"Frozen registry {cfg_p} is missing required dataset role {card_k!r}")
        if not p_val or not Path(p_val).is_file():
            raise RuntimeError(
                f"Non-smoke Wave 8A run requires verified local dataset file for role {card_k!r}; got {p_val!r}"
            )
        exp_sha = str(reg_cards[card_k]["sha256"])
        got_sha = out[prov_k]
        if got_sha != exp_sha:
            raise RuntimeError(
                f"Registered dataset SHA mismatch for role {card_k!r} (file {p_val}): got {got_sha}, expected {exp_sha}"
            )
    out["registered_cards_verified"] = True
    return out


WAVE8A_BOUND_SOURCE_FILES = (
    "configs/x1b/study_x2_wave8a.json",
    "configs/x1b/study_x2_wave8a_r2.json",
    "configs/x1b/wave8a_dataset_cards.json",
    "configs/x1b/wave8a_showcase_preregistration.json",
    "models/x1b_world.py",
    "models/x1b_jepa.py",
    "models/x1b_commit.py",
    "models/x1b_sigreg_epps_pulley.py",
    "models/x1b_wave6_core.py",
    "models/x1b_wave7_core.py",
    "models/x1b_wave8a_core.py",
    "data/x1b_multimodal.py",
    "data/x1b_unseen_worlds.py",
    "evals/x1b_mpc.py",
    "evals/x1b_learned_mpc.py",
    "evals/x1b_stats.py",
    "evals/x1b_eval_wave6.py",
    "evals/x1b_eval_wave7.py",
    "evals/x1b_eval_wave8a.py",
    "evals/x1b_showcase_wave8a.py",
    "scripts/train/x1b_train.py",
    "scripts/train/x1b_train_wave6.py",
    "scripts/train/x1b_train_wave7.py",
    "scripts/train/x1b_train_wave8a.py",
)

WAVE8A_PEAK_LRS = {"x_tiny": 9e-4, "x150m": 3.0e-4, "x450m": 2.7e-4, "x1b": 2.4e-4}


def compute_wave8a_source_code_sha256() -> str:
    """Compute a deterministic SHA-256 digest across all executed Wave 8A model, data, eval, trainer, and frozen config files."""
    h = hashlib.sha256()
    for rel in WAVE8A_BOUND_SOURCE_FILES:
        fp = REPO_ROOT / rel
        if not fp.is_file():
            raise FileNotFoundError(f"Missing bound Wave 8A source file for provenance verification: {fp}")
        h.update(f"{rel}:{sha256_file(fp)}\n".encode())
    return h.hexdigest()


def build_wave8a_binding_doc(
    *,
    tag: str,
    cohort: str,
    profile: str,
    arch: str,
    interventions: Wave8AInterventions,
    seed: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    smoke: bool,
    save_bf16: bool,
    export_showcase: bool,
) -> dict[str, Any]:
    """Construct the canonical execution-binding manifest for a Wave 8A run (binds smoke, exports, spec, and schedule)."""
    spec = make_x1b_spec(profile=profile, arch=arch)
    return {
        "tag": str(tag),
        "cohort": str(cohort),
        "wave": "wave8a_trunk_isolation_and_gqa_parity",
        "spec": asdict(spec),
        "interventions": asdict(interventions),
        "seed": int(seed),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "peak_lr": float(WAVE8A_PEAK_LRS[profile]),
        "smoke": bool(smoke),
        "save_bf16": bool(save_bf16),
        "export_showcase": bool(export_showcase),
    }


def compute_wave8a_binding_sha256(**kwargs: Any) -> str:
    """Compute the SHA-256 digest of `build_wave8a_binding_doc`."""
    doc = build_wave8a_binding_doc(**kwargs)
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def train_and_evaluate_wave8a(
    *,
    tag: str,
    cohort: str,
    profile: str,
    arch: str,
    regularizer: str,
    key_shield: bool,
    zerolag_bypass: bool,
    decoupled_heads: bool,
    zmem_isolation: bool,
    enable_assoc_loss: bool,
    seed: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    out_dir: Path,
    real_bank_path: str | None = None,
    grounded_robot_path: str | None = None,
    vault_robot_path: str | None = None,
    save_bf16: bool = False,
    export_showcase: bool = False,
    allow_verified_resume: bool = False,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train one Wave 8A cohort seed, run all 6 evaluation panels, and optionally export `showcase_bundle.npz`."""
    job_dir = out_dir / tag
    done_path = job_dir / "done.json"
    spec = make_x1b_spec(profile=profile, arch=arch)
    interventions = Wave8AInterventions(
        key_shield=key_shield,
        zerolag_bypass=zerolag_bypass,
        decoupled_heads=decoupled_heads,
        zmem_isolation=zmem_isolation,
        enable_assoc_loss=enable_assoc_loss,
        regularizer=regularizer,
    )
    peak_lr = WAVE8A_PEAK_LRS[profile]
    state_mem = compute_streaming_state_bytes(spec, batch_size=1, cache_len=64)
    source_code_sha = compute_wave8a_source_code_sha256()

    data_provenance: dict[str, Any] = {
        **verify_wave8a_dataset_roles_fail_closed(
            real_bank_path,
            grounded_robot_path,
            vault_robot_path,
            smoke=smoke,
        ),
        "sequence_tensor_budget_ticks": int(steps * batch_size * seq_len),
    }
    expected_dataset_shas = {
        k: data_provenance[k]
        for k in ("real_bank_sha256", "grounded_robot_sha256", "vault_robot_sha256")
        if k in data_provenance
    }

    binding_doc = build_wave8a_binding_doc(
        tag=tag,
        cohort=cohort,
        profile=profile,
        arch=arch,
        interventions=interventions,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        seq_len=seq_len,
        smoke=smoke,
        save_bf16=save_bf16,
        export_showcase=export_showcase,
    )
    binding_sha = hashlib.sha256(json.dumps(binding_doc, sort_keys=True).encode()).hexdigest()

    int8_path = job_dir / "portable_int8.npz"
    bf16_path = job_dir / "weights_bf16.npz"
    resume_bf16_path = job_dir / ".resume_weights_bf16.npz"
    ckpt_prov_path = job_dir / "checkpoint_provenance.json"
    showcase_path = job_dir / "showcase_bundle.npz"
    prov_path = job_dir / "data_provenance.json"
    receipt_path = job_dir / "receipt.json"
    eval_metrics_path = job_dir / "eval_metrics.json"
    config_json_path = job_dir / "config.json"
    train_log_path = job_dir / "train.jsonl"

    if done_path.is_file():
        if not allow_verified_resume:
            raise RuntimeError(
                f"Pre-existing done.json found at {done_path} for {tag!r} while allow_verified_resume=False; "
                "refusing unbound completion reuse (fail-closed)."
            )
        existing_done = json.loads(done_path.read_text())
        if not receipt_path.is_file() or json.loads(receipt_path.read_text()) != existing_done:
            raise RuntimeError(f"Missing or mismatched receipt.json vs done.json at {job_dir} for {tag!r}.")
        done_prov = existing_done.get("data_provenance", {})
        done_ds_shas = {
            k: done_prov[k]
            for k in ("real_bank_sha256", "grounded_robot_sha256", "vault_robot_sha256")
            if k in done_prov
        }
        if (
            existing_done.get("binding_sha256") != binding_sha
            or existing_done.get("source_code_sha256") != source_code_sha
            or existing_done.get("smoke") is not bool(smoke)
            or existing_done.get("save_bf16") is not bool(save_bf16)
            or existing_done.get("export_showcase") is not bool(export_showcase)
            or int(existing_done.get("steps", -1)) != int(steps)
            or done_prov.get("registered_cards_verified") is not (not smoke)
            or done_ds_shas != expected_dataset_shas
        ):
            raise RuntimeError(
                f"Refusing to reuse stale/mismatched done.json at {done_path}: "
                f"binding/source/smoke/steps/dataset provenance mismatch for {tag!r}."
            )
        required_files_and_shas: list[tuple[Path, Any]] = [
            (int8_path, existing_done.get("portable_int8_sha256")),
            (eval_metrics_path, existing_done.get("eval_metrics_sha256")),
            (config_json_path, existing_done.get("config_file_sha256")),
            (train_log_path, existing_done.get("train_log_sha256")),
        ]
        if save_bf16:
            required_files_and_shas.append((bf16_path, existing_done.get("weights_bf16_sha256")))
        if export_showcase:
            required_files_and_shas.append((showcase_path, existing_done.get("showcase_bundle_sha256")))
        for art_p, exp_art_sha in required_files_and_shas:
            if not isinstance(exp_art_sha, str) or len(exp_art_sha) != 64 or not art_p.is_file():
                raise RuntimeError(
                    f"Completion artifact missing or unhashed for {tag!r}: path={art_p}, expected_sha={exp_art_sha!r}"
                )
            got_art_sha = sha256_file(art_p)
            if got_art_sha != exp_art_sha:
                raise RuntimeError(
                    f"Completion artifact SHA-256 mismatch on {art_p} for {tag!r}: got={got_art_sha}, expected={exp_art_sha}"
                )
        print(f"[SKIP] {tag} already completed and cryptographically verified at {done_path}", flush=True)
        return existing_done

    job_dir.mkdir(parents=True, exist_ok=True)

    resumed_from_verified_ckpt = False
    if bf16_path.is_file() or resume_bf16_path.is_file():
        src_ckpt = bf16_path if bf16_path.is_file() else resume_bf16_path
        if not allow_verified_resume:
            raise RuntimeError(
                f"Pre-existing checkpoint {src_ckpt} found for {tag!r} while allow_verified_resume=False; "
                "refusing unbound checkpoint reuse (fail-closed)."
            )
        if not ckpt_prov_path.is_file() or not train_log_path.is_file():
            raise RuntimeError(
                f"Pre-existing checkpoint {src_ckpt} is missing {ckpt_prov_path} or {train_log_path}; "
                "refusing unbound checkpoint reuse."
            )
        ckpt_prov = json.loads(ckpt_prov_path.read_text())
        actual_ckpt_sha = sha256_file(src_ckpt)
        actual_tlog_sha = sha256_file(train_log_path)
        if (
            ckpt_prov.get("weights_sha256") != actual_ckpt_sha
            or ckpt_prov.get("train_log_sha256") != actual_tlog_sha
            or int(ckpt_prov.get("completed_steps", -1)) != int(steps)
            or ckpt_prov.get("smoke") is not bool(smoke)
            or ckpt_prov.get("registered_cards_verified") is not (not smoke)
            or ckpt_prov.get("binding_sha256") != binding_sha
            or ckpt_prov.get("source_code_sha256") != source_code_sha
            or ckpt_prov.get("dataset_role_shas") != expected_dataset_shas
        ):
            raise RuntimeError(
                f"Checkpoint provenance mismatch on {src_ckpt} for {tag!r}: "
                f"got={ckpt_prov}, expected_sha={actual_ckpt_sha}, expected_steps={steps}, "
                f"expected_binding={binding_sha}, expected_source={source_code_sha}, expected_datasets={expected_dataset_shas}."
            )
        params = load_bf16_weights(src_ckpt)
        param_counts = count_parameters(params)
        train_wall_s = float(ckpt_prov.get("train_wall_s", 0.0))
        resumed_from_verified_ckpt = True
    else:
        params = init_x1b_params(spec, seed=seed)
        param_counts = count_parameters(params)
        ema_params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
        m_state = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), params)
        v_state = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), params)

    config_doc = {
        **binding_doc,
        "de_privileged_proprio": "ee_only",
        "visual_pose_readout": "pure_vision_spatial_soft_argmax_4x4",
        "planner": "spatial_geometry_free_subgoal_mppi",
        "streaming_state_memory": state_mem,
        "save_bf16": bool(save_bf16),
        "export_showcase": bool(export_showcase),
        "parameter_counts": param_counts,
        "binding_sha256": binding_sha,
        "source_code_sha256": source_code_sha,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
    }
    config_hash = hashlib.sha256(json.dumps(config_doc, sort_keys=True).encode()).hexdigest()
    config_doc["config_sha256"] = config_hash
    atomic_write_json(job_dir / "config.json", config_doc)

    print(
        f"[START W8A] {tag} | profile={profile} arch={arch} iv={asdict(interventions)} "
        f"| params={param_counts['strict_total']:,} | state_mib={state_mem['total_state_mib']:.3f} | backend={jax.default_backend()}",
        flush=True,
    )

    if not resumed_from_verified_ckpt:
        real_train = RealMultimodalBank(real_bank_path, split="train") if real_bank_path else None
        grounded_train = Wave8AGroundedRobotBank(grounded_robot_path, split="train") if grounded_robot_path else None
        np_rng = np.random.default_rng(seed + 80013)

        def _step_kernel(p, ema_p, m, v, batch_in, step_idx, rk_loss):
            (loss, metrics), grads = jax.value_and_grad(
                lambda w: world_model_loss_wave8a(w, ema_p, spec, batch_in, interventions, rk_loss),
                has_aux=True,
            )(p)
            p_new, m_new, v_new, gnorm, lr_now = memory_efficient_adamw_step(
                p, m, v, grads, step_idx, steps, peak_lr
            )
            ema_decay = 0.992 + 0.006 * (step_idx.astype(jnp.float32) / float(max(1, steps)))
            ema_new = jax.tree.map(
                lambda e, w: (ema_decay * e.astype(jnp.float32) + (1.0 - ema_decay) * w.astype(jnp.float32)).astype(
                    jnp.bfloat16
                ),
                ema_p,
                p_new,
            )
            return p_new, ema_new, m_new, v_new, {**metrics, "grad_norm": gnorm, "lr": lr_now}

        compiled_step = jax.jit(_step_kernel, donate_argnums=(0, 1, 2, 3))
        has_real = (real_train is not None) or (grounded_train is not None)
        synth_n = (batch_size * 2) // 3 if has_real else batch_size
        synth_n = min(max(2, synth_n), batch_size - (1 if has_real else 0))
        rem = batch_size - synth_n
        if (real_train is not None) and (grounded_train is not None):
            rob_n = max(1, (2 * rem + 1) // 3)
            real_n = rem - rob_n
        elif grounded_train is not None:
            rob_n, real_n = rem, 0
        else:
            rob_n, real_n = 0, rem

        jit_sim = jax.jit(
            lambda rk: generate_in_domain_deprivileged_sim_batch_v8a(
                rk, synth_n, seq_len=seq_len, vocab_size=spec.vocab_size
            )
        )

        valid_ticks_sim = 0
        valid_ticks_rob = 0
        valid_ticks_real = 0
        t_start = time.monotonic()
        log_file = train_log_path
        log_file.write_text("")
        for step in range(steps):
            rk_sim, rk_loss = jax.random.split(jax.random.fold_in(jax.random.key(seed), step))
            s_batch = jit_sim(rk_sim)
            r_batch = real_train.sample_batch(np_rng, real_n, seq_len=seq_len) if (real_train and real_n > 0) else None
            g_batch = (
                grounded_train.sample_batch(np_rng, rob_n, seq_len=seq_len) if (grounded_train and rob_n > 0) else None
            )
            valid_ticks_sim += int(synth_n * seq_len)
            if r_batch is not None:
                valid_ticks_real += int(np.sum(r_batch["valid"]))
            if g_batch is not None:
                valid_ticks_rob += int(np.sum(g_batch["valid"]))
            batch = mix_wave8a_batches(r_batch, g_batch, s_batch, seq_len=seq_len) if has_real else s_batch

            params, ema_params, m_state, v_state, metrics = compiled_step(
                params, ema_params, m_state, v_state, batch, jnp.asarray(step, dtype=jnp.int32), rk_loss
            )
            if step == 0 or (step + 1) % 50 == 0 or (step + 1) == steps:
                m_host = {k: float(v) for k, v in jax.device_get(metrics).items()}
                if not np.isfinite(m_host["loss"]):
                    raise RuntimeError(f"Non-finite Wave 8A loss on {tag} at step {step + 1}: {m_host}")
                rec_line = {"step": step + 1, "elapsed_s": round(time.monotonic() - t_start, 2), **m_host}
                with log_file.open("a") as lf:
                    lf.write(json.dumps(rec_line, sort_keys=True) + "\n")
                atomic_write_json(out_dir / f"{tag}-progress.json", rec_line)
                if step == 0 or (step + 1) % 200 == 0 or (step + 1) == steps:
                    print(
                        f"[{tag}] step {step + 1}/{steps} | loss={m_host['loss']:.4f} jepa={m_host['jepa']:.4f} "
                        f"vobj={m_host['vis_obj_mse']:.4f} goal={m_host['goal_mse']:.4f} "
                        f"assoc_ce={m_host['text_ce']:.4f} pr={m_host['pr']:.3f}",
                        flush=True,
                    )

        train_wall_s = time.monotonic() - t_start
        data_provenance.update(
            {
                "real_bank_train_episodes": int(real_train.size) if real_train is not None else 0,
                "grounded_robot_train_episodes": int(grounded_train.size) if grounded_train is not None else 0,
                "batch_partition": {"sim": int(synth_n), "grounded_robot": int(rob_n), "real_multimodal": int(real_n)},
                "valid_ticks_sim": valid_ticks_sim,
                "valid_ticks_grounded_robot": valid_ticks_rob,
                "valid_ticks_real_multimodal": valid_ticks_real,
                "valid_ticks_total": valid_ticks_sim + valid_ticks_rob + valid_ticks_real,
            }
        )
        atomic_write_json(prov_path, data_provenance)
        del ema_params, m_state, v_state
        target_ckpt = bf16_path if save_bf16 else resume_bf16_path
        saved_ckpt_sha = save_bf16_weights(target_ckpt, params)
        atomic_write_json(
            ckpt_prov_path,
            {
                "tag": tag,
                "weights_sha256": saved_ckpt_sha,
                "train_log_sha256": sha256_file(train_log_path),
                "completed_steps": int(steps),
                "smoke": bool(smoke),
                "registered_cards_verified": bool(not smoke),
                "binding_sha256": binding_sha,
                "source_code_sha256": source_code_sha,
                "dataset_role_shas": expected_dataset_shas,
                "train_wall_s": round(train_wall_s, 2),
            },
        )
    elif prov_path.is_file():
        data_provenance.update(json.loads(prov_path.read_text()))

    int8_sha = save_portable_int8(int8_path, params)
    bf16_sha = save_bf16_weights(bf16_path, params) if (save_bf16 and not bf16_path.is_file()) else None
    if save_bf16 and bf16_path.is_file() and bf16_sha is None:
        bf16_sha = sha256_file(bf16_path)

    t_ev0 = time.monotonic()
    eval_res = run_wave8a_evaluation(
        params,
        spec,
        interventions,
        seed=seed,
        grounded_npz_path=grounded_robot_path,
        vault_npz_path=vault_robot_path,
        smoke=smoke,
    )
    eval_wall_s = time.monotonic() - t_ev0
    atomic_write_json(job_dir / "eval_metrics.json", eval_res)

    showcase_sha = None
    showcase_bytes = 0
    if export_showcase:
        sc_bundle = collect_wave8a_showcase_bundle(
            params,
            spec,
            interventions,
            cohort=cohort,
            seed=seed,
            grounded_npz_path=grounded_robot_path,
            smoke=smoke,
        )
        showcase_sha = save_showcase_bundle_npz(showcase_path, sc_bundle)
        showcase_bytes = showcase_path.stat().st_size

    cap = eval_res["multi_item_overwrite_capacity"]
    eval_summary = {
        "oracle_mppi_ceiling": eval_res["unseen_layout_deprivileged_mpc"].get("oracle_perception_dynamics_mppi_ceiling"),
        "oracle_terminal16_hold_ceiling": eval_res["unseen_layout_deprivileged_mpc"].get(
            "oracle_terminal16_all_ticks_hold_success"
        ),
        "unseen_mppi_success": eval_res["unseen_layout_deprivileged_mpc"]["geometry_free_mppi_success"],
        "no_jepa_subgoal_success": eval_res["unseen_layout_deprivileged_mpc"].get("no_jepa_subgoal_success"),
        "explicit_latch_reactive_success": eval_res["unseen_layout_deprivileged_mpc"]["explicit_latch_reactive_success"],
        "vis_decoded_obj_rmse": eval_res["unseen_layout_deprivileged_mpc"]["vis_decoded_obj_rmse"],
        "T160_post_perturb_goal_b_success": eval_res["beyond_window_T160_distractor_switch"][
            "T160_post_perturbation_goal_b_success"
        ],
        "T160_paired_causal_switch_rate": eval_res["beyond_window_T160_distractor_switch"].get(
            "T160_paired_causal_switch_rate"
        ),
        "T160_uninstructed_goal_b_leak_rate": eval_res["beyond_window_T160_distractor_switch"].get(
            "T160_uninstructed_goal_b_leak_rate"
        ),
        "latch_mse_beyond_W64": eval_res["beyond_window_T160_distractor_switch"]["latch_mse_beyond_W64_t92_160"],
        "symbolic_key_visual_drift": eval_res["symbolic_key_drift_and_recall"]["symbolic_key_visual_drift"],
        "dyn_visual_assoc_T1024": eval_res["symbolic_key_drift_and_recall"]["dyn_visual_assoc_T1024"],
        "K8_T256_overwrite_acc": cap.get("K8_T256_overwritten_acc"),
        "K16_T1024_overwrite_acc": cap.get("K16_T1024_overwritten_acc"),
        "unseen_ood_corruption_auroc": eval_res["unseen_ood_sensor_corruptions"]["mean_unseen_ood_corruption_auroc"],
        "commit_selective_routing_acc": eval_res["unseen_ood_sensor_corruptions"].get("commit_selective_routing_acc"),
        "vault_step4_mm": eval_res["untouched_vault_spoken_robot"].get("step4_mm"),
        "vault_spoken_disc_acc": eval_res["untouched_vault_spoken_robot"].get("spoken_16tok_discrimination_acc"),
        "vault_spoken_audio_only_disc_acc": eval_res["untouched_vault_spoken_robot"].get(
            "spoken_audio_only_discrimination_acc"
        ),
        "vault_scene_color_disc_acc": eval_res["untouched_vault_spoken_robot"].get("spoken_scene_color_disc_acc"),
        "vault_scene_color_audio_only_disc_acc": eval_res["untouched_vault_spoken_robot"].get(
            "spoken_audio_only_scene_color_disc_acc"
        ),
        "vault_shuffled_scene_color_disc_acc": eval_res["untouched_vault_spoken_robot"].get(
            "shuffled_scene_color_disc_acc"
        ),
        "vault_color_prior_scene_color_disc_acc": eval_res["untouched_vault_spoken_robot"].get(
            "color_prior_scene_color_disc_acc"
        ),
        "vault_paired_flip_dual_color_disc_acc": eval_res["untouched_vault_spoken_robot"].get(
            "paired_flip_dual_scene_color_disc_acc"
        ),
        "vault_text_only_disc_acc": eval_res["untouched_vault_spoken_robot"].get("text_16tok_only_discrimination_acc"),
        "compiled_inference_ms_per_tick": eval_res.get("compiled_inference_ms_per_tick"),
        "streaming_state_mib": state_mem["total_state_mib"],
        "core_state_mib": state_mem["core_state_mib"],
        "strict_system_state_mib": state_mem["strict_system_state_mib"],
    }
    config_file_sha = sha256_file(config_json_path)
    eval_metrics_sha = sha256_file(eval_metrics_path)
    train_log_sha = sha256_file(train_log_path)
    receipt = {
        "tag": tag,
        "cohort": cohort,
        "wave": "wave8a_trunk_isolation_and_gqa_parity",
        "profile": profile,
        "arch": arch,
        "interventions": asdict(interventions),
        "streaming_state_memory": state_mem,
        "data_provenance": data_provenance,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "smoke": bool(smoke),
        "save_bf16": bool(save_bf16),
        "export_showcase": bool(export_showcase),
        "backend": jax.default_backend(),
        "parameter_counts": param_counts,
        "binding_sha256": binding_sha,
        "source_code_sha256": source_code_sha,
        "config_sha256": config_hash,
        "config_file_sha256": config_file_sha,
        "eval_metrics_sha256": eval_metrics_sha,
        "train_log_sha256": train_log_sha,
        "portable_int8_sha256": int8_sha,
        "portable_int8_bytes": int8_path.stat().st_size,
        "weights_bf16_sha256": bf16_sha,
        "weights_bf16_bytes": bf16_path.stat().st_size if bf16_path.is_file() else 0,
        "showcase_bundle_sha256": showcase_sha,
        "showcase_bundle_bytes": showcase_bytes,
        "train_wall_s": round(train_wall_s, 2),
        "eval_wall_s": round(eval_wall_s, 2),
        "eval_summary": eval_summary,
    }
    atomic_write_json(receipt_path, receipt)
    resume_bf16_path.unlink(missing_ok=True)
    if not save_bf16:
        ckpt_prov_path.unlink(missing_ok=True)
    atomic_write_json(done_path, receipt)
    print(f"[DONE W8A] {tag} | summary={json.dumps(eval_summary, sort_keys=True)}", flush=True)
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cohort", default="w8a_slwm_x2_isolated_450m")
    ap.add_argument("--profile", default="x450m")
    ap.add_argument("--arch", default="slwm_x1b")
    ap.add_argument("--regularizer", default="epps_pulley_sigreg")
    ap.add_argument("--no-key-shield", action="store_true")
    ap.add_argument("--no-zerolag-bypass", action="store_true")
    ap.add_argument("--no-decoupled-heads", action="store_true")
    ap.add_argument("--no-zmem-isolation", action="store_true")
    ap.add_argument("--disable-assoc-loss", action="store_true")
    ap.add_argument("--seed", type=int, default=2001)
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--real-bank-path", default=None)
    ap.add_argument("--grounded-robot-path", default=None)
    ap.add_argument("--vault-robot-path", default=None)
    ap.add_argument("--save-bf16", action="store_true")
    ap.add_argument("--export-showcase", action="store_true")
    ap.add_argument("--allow-verified-resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train_and_evaluate_wave8a(
        tag=args.tag,
        cohort=args.cohort,
        profile=args.profile,
        arch=args.arch,
        regularizer=args.regularizer,
        key_shield=(not args.no_key_shield),
        zerolag_bypass=(not args.no_zerolag_bypass),
        decoupled_heads=(not args.no_decoupled_heads),
        zmem_isolation=(not args.no_zmem_isolation),
        enable_assoc_loss=(not args.disable_assoc_loss),
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        out_dir=args.out_dir,
        real_bank_path=args.real_bank_path,
        grounded_robot_path=args.grounded_robot_path,
        vault_robot_path=args.vault_robot_path,
        save_bf16=args.save_bf16,
        export_showcase=args.export_showcase,
        allow_verified_resume=args.allow_verified_resume,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
