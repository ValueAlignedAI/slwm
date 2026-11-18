"""Study X3 (`SLWM-1B-480p`) Joint Multimodal World-Model & Any-to-Any Composition Trainer.

Trains `SLWM-X3` (`140.0M`, `473.5M`, `1.022B`), `SlidingWindow-W64-GQA-1B`, `DiagonalSSM-1B`, and the
`SLWM-X3-1B-240p` single-factor resolution ablation on native `640x480` (`480p`) multi-corpus streams
paired with:
  - Every-step `T_aux = 80, B_aux = 2` beyond-window (`> 42` multi-hop `W=8` span & `> 64` `GQA` window)
    `K in {1, 2, 4, 8, 16}` selective associative overwrite + delayed/distracted `SLOT_GOAL_LANG` RoPE calibration,
  - Strictly in-domain de-privileged `480p` arena physics (`mass_scale in [0.55, 1.85]`, `damping in [0.55, 0.86]`)
    with current-tick `target_vis_pose_t` supervision and online `SLOT_SYSID` adaptive gain learning,
  - Physical `xArm6` two-utterance speech + `H_obj` (`slot 56`) / `H_dir` (`slots 57..58`) disentangled grounding.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gc
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from models.x1b_sigreg_epps_pulley import epps_pulley_sigreg_loss
from models.x1b_wave8a_core import (
    encode_goal_query_tokens,
    encode_goal_tokens_2d,
)
from models.x3_480p_core import (
    X3_ASSOC_KEY_OFFSET,
    X3_ASSOC_NUM_KEYS,
    X3_ASSOC_NUM_VALS,
    X3_ASSOC_VAL_OFFSET,
    X3_SLOTS_REGISTER,
    X3_SLOTS_VISUAL,
    X3Spec,
    compute_x3_streaming_state_bytes,
    count_x3_parameters,
    decode_480p_video_from_latents,
    decode_x3_any_to_any_heads,
    decode_x3_target_block_xy,
    encode_480p_visual_stem,
    fork_x3_streams,
    forward_x3_world_model,
    init_x3_params,
    make_x3_spec,
    predict_x3_causal_action_from_ztask,
    predict_x3_latent_step,
)
from scripts.train.x1b_train import memory_efficient_adamw_step, save_bf16_weights, save_portable_int8, sha256_file
from scripts.train.x1b_train_wave8a import Wave8AGroundedRobotBank

REPO_ROOT = Path(__file__).resolve().parents[2]

X3_PEAK_LRS = {
    "x3_tiny": 9.0e-4,
    "x3_150m": 3.0e-4,
    "x3_450m": 2.5e-4,
    "x3_1b": 2.0e-4,
}

X3_BOUND_SOURCE_FILES = (
    "configs/x3/study_x3_1b_480p.json",
    "configs/x3/x3_dataset_cards.json",
    "data/x1b_unseen_worlds.py",
    "models/x1b_sigreg_epps_pulley.py",
    "models/x1b_world.py",
    "models/x1b_commit.py",
    "models/x1b_wave8a_core.py",
    "models/x3_480p_core.py",
    "evals/x1b_stats.py",
    "evals/x1b_eval_wave6.py",
    "evals/x1b_eval_wave7.py",
    "evals/x1b_eval_wave8a.py",
    "evals/x3_eval_480p.py",
    "scripts/train/x1b_train.py",
    "scripts/train/x1b_train_wave8a.py",
    "scripts/train/x3_train_480p.py",
    "scripts/eval/x3_analyze_480p.py",
)


def compute_x3_source_code_sha256() -> str:
    h = hashlib.sha256()
    for rel in X3_BOUND_SOURCE_FILES:
        fp = REPO_ROOT / rel
        if not fp.is_file():
            raise FileNotFoundError(f"Missing bound Study X3 source file: {fp}")
        h.update(f"{rel}:{sha256_file(fp)}\n".encode())
    return h.hexdigest()


def encode_wave8a_two_utterance_tokens_32x4(tok16: np.ndarray, seq_len: int = 32) -> np.ndarray:
    """Place `16`-token `xArm6` directive (`[0:8]` color + `[8:16]` direction) onto causal two-utterance ticks.

    Ticks `t = 0..1` carry the 8 color/object tokens (`[0:8]`, aligned with `t = 0..15` GRID speech),
    and ticks `t = 16..17` carry the 8 direction tokens (`[8:16]`, aligned with `t = 16..31` SpeechCommands speech).
    Shared identically between `x3_train_480p.py` and `eval_x3_spoken_xarm6_grounding`.
    """
    n = int(tok16.shape[0])
    out = np.zeros((n, seq_len, 4), dtype=np.int32)
    h44 = np.asarray(tok16, dtype=np.int32).reshape(n, 4, 4)
    if seq_len >= 2:
        out[:, 0:2, :] = h44[:, 0:2, :]
    if seq_len >= 18:
        out[:, 16:18, :] = h44[:, 2:4, :]
    elif seq_len >= 4:
        out[:, 2:4, :] = h44[:, 2:4, :]
    return out


class X3Multimodal480pBank:
    """Zero-pickle (`allow_pickle=False`) `640x480` (`480p`) multi-corpus dataloader with parallel JPEG decode."""

    def __init__(self, npz_path: str | Path) -> None:
        self.path = Path(npz_path)
        with np.load(self.path, allow_pickle=False) as d:
            self.jpeg_blob = np.asarray(d["jpeg_blob"], dtype=np.uint8)
            self.jpeg_offsets = np.asarray(d["jpeg_offsets"], dtype=np.int64)
            self.audio = np.asarray(d["audio"], dtype=np.float32)
            self.movement = np.asarray(d["movement"], dtype=np.float32)
            self.action = np.asarray(d["action"], dtype=np.float32)
            self.target_action = np.asarray(d["target_action"], dtype=np.float32)
            self.has_true_action = (
                np.asarray(d["has_true_action"], dtype=np.float32)
                if "has_true_action" in d.files
                else (np.max(np.abs(self.target_action), axis=(1, 2)) > 1e-4).astype(np.float32)
            )
            self.lang_tokens = np.asarray(d["lang_tokens"], dtype=np.int32)
            self.target_lang_tokens = np.asarray(d["target_lang_tokens"], dtype=np.int32)
            self.modality_mask = np.asarray(d["modality_mask"], dtype=np.float32)
            self.source_id = np.asarray(d["source_id"], dtype=np.int32)
            self.embodiment_id = np.asarray(d["embodiment_id"], dtype=np.int32)
            self.group_id = np.asarray(d["group_id"], dtype=np.int64)
        self.size = int(self.source_id.shape[0])
        self._pool = ThreadPoolExecutor(max_workers=8)

    def decode_clip_frames(self, clip_idx: int, ticks: int = 24) -> np.ndarray:
        offs = self.jpeg_offsets[clip_idx, :ticks]
        out = np.zeros((ticks, 480, 640, 3), dtype=np.uint8)
        if np.all(offs[:, 1] == 0):
            return out
        for ti in range(offs.shape[0]):
            st, ln = int(offs[ti, 0]), int(offs[ti, 1])
            if ln > 0:
                raw_b = memoryview(self.jpeg_blob)[st : st + ln]
                out[ti] = np.asarray(Image.open(io.BytesIO(raw_b)).convert("RGB"), dtype=np.uint8)
        return out

    def sample_batch(
        self,
        rng: np.random.Generator,
        batch_size: int,
        seq_len: int = 24,
        *,
        apply_composition_masking: bool = True,
        augment_photometric: bool = True,
    ) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=(batch_size,))
        t = min(seq_len, int(self.audio.shape[1]))
        frames_list = list(self._pool.map(lambda ci: self.decode_clip_frames(int(ci), t), idx))
        vid = np.stack(frames_list, axis=0)
        full_mm = self.modality_mask[idx, :t].copy()
        obs_mm = full_mm.copy()
        has_act = self.has_true_action[idx].copy()
        act_in = self.action[idx, :t].copy() * has_act[:, None, None]

        # Clean clip-consistent photometric augmentation on training real clips to prevent 1.022B pixel memorization
        if augment_photometric:
            gains = rng.uniform(0.88, 1.12, size=(batch_size, 1, 1, 1, 1)).astype(np.float32)
            biases = rng.uniform(-8.0, 8.0, size=(batch_size, 1, 1, 1, 1)).astype(np.float32)
            vis_ep = full_mm[:, 0, 1] > 0.5
            if np.any(vis_ep):
                vf = vid[vis_ep].astype(np.float32) * gains[vis_ep] + biases[vis_ep]
                vid[vis_ep] = np.clip(vf, 0.0, 255.0).astype(np.uint8)

        if apply_composition_masking:
            u_draw = rng.random(batch_size)
            drop_prev_u = rng.random(batch_size) < 0.55
            for bi in range(batch_size):
                avail = full_mm[bi, 0]
                u = u_draw[bi]
                can_drop_txt = (avail[1] > 0.5 or avail[0] > 0.5) and avail[3] > 0.5
                can_drop_aud = avail[1] > 0.5 and avail[0] > 0.5
                can_drop_mov = (avail[1] > 0.5 or avail[3] > 0.5) and avail[2] > 0.5
                can_drop_vis = avail[1] > 0.5 and (avail[3] > 0.5 or avail[0] > 0.5 or avail[2] > 0.5)
                if u < 0.28 and can_drop_txt:
                    obs_mm[bi, :, 3] = 0.0
                    act_in[bi] = 0.0
                elif u < 0.50 and can_drop_aud:
                    obs_mm[bi, :, 0] = 0.0
                elif u < 0.70 and can_drop_mov:
                    obs_mm[bi, :, 2] = 0.0
                    act_in[bi] = 0.0  # Strict zero-action sanitization when movement is masked!
                elif u < 0.84 and avail[1] > 0.5 and avail[0] > 0.5:
                    # Full blind-vision masking (`mm[vis] = 0` on all ticks) training Direction 6 (`Audio -> Occluded Visual Motion`)!
                    obs_mm[bi, :, 1] = 0.0
                    act_in[bi] = 0.0
                elif can_drop_vis:
                    obs_mm[bi, 1:, 1] = 0.0
                    obs_mm[bi, 1:, 2] = 0.0
                elif can_drop_txt:
                    obs_mm[bi, :, 3] = 0.0
                elif can_drop_mov:
                    obs_mm[bi, :, 2] = 0.0
                    act_in[bi] = 0.0
                elif can_drop_aud:
                    obs_mm[bi, :, 0] = 0.0

                # Train Direction 5 (`AV + Text -> Robot Actions`) without u_{t-1} autoregressive shortcut!
                if drop_prev_u[bi]:
                    act_in[bi] = 0.0
                    obs_mm[bi, 1:, 2] = 0.0

        mov = self.movement[idx, :t]
        next_mov = np.concatenate([mov[:, 1:], mov[:, -1:]], axis=1)
        sh4 = min(4, max(1, t - 1))
        step4_mov = np.concatenate([mov[:, sh4:], np.repeat(mov[:, -1:], sh4, axis=1)], axis=1)
        return {
            "video": vid,
            "clean_video": vid,
            "audio": self.audio[idx, :t],
            "movement": mov,
            "target_mov_next": next_mov,
            "target_mov_step4": step4_mov,
            "target_vis_pose_t": np.zeros((batch_size, t, 4), dtype=np.float32),
            "action": act_in,
            "next_action": act_in,  # Never feed future ground-truth `target_action` into latent predictor on real clips!
            "target_action": self.target_action[idx, :t],
            "has_true_action": has_act,
            "sym_tokens": np.zeros((batch_size, t, 4), dtype=np.int32),
            "target_sym_val": np.zeros((batch_size, t), dtype=np.int32),
            "lang_tokens": self.lang_tokens[idx, :t],
            "target_lang_tokens": self.target_lang_tokens[idx, :t],
            "target_goal_xy": np.zeros((batch_size, t, 2), dtype=np.float32),
            "target_value": np.zeros((batch_size, t), dtype=np.float32),
            "target_contacts": np.zeros((batch_size, t, 2), dtype=np.float32),
            "target_commit": np.zeros((batch_size, t), dtype=np.int32),
            "modality_mask": obs_mm,
            "full_modality_mask": full_mm,
            "embodiment_id": self.embodiment_id[idx],
            "valid": np.ones((batch_size, t), dtype=bool),
            "scenario": np.full((batch_size,), -1, dtype=np.int32),
        }


def build_x3_sym_assoc_tokens(
    rng_key: jax.Array,
    batch_size: int,
    seq_len: int,
) -> tuple[jax.Array, jax.Array]:
    """Synthesize pure discrete `K in {1, 2, 4, 8, 16}` bindings + selective overwrites + queries on `SLOT_SYM_MEM`.

    For `seq_len >= 64` (including `T_aux = 80` beyond-window auxiliary pass):
      - Ticks `0..15`: write up to `K = 16` distinct bindings `[0, k_j, v_init_j, 0]`.
      - Ticks `16..23`: selectively overwrite `M = max(1, K // 2)` (`<= 8`) bindings `[0, k_j, v_over_j, 0]`.
      - Ticks `24 .. seq_len - 9`: strict null ticks (`sym_tokens == 0`, forcing `H_sym` recurrence past the `42`-tick
        6-layer `W = 8` local RoPE span and past the `64`-tick `GQA` window at `T_aux = 80`).
      - Ticks `seq_len - 8 .. seq_len - 1` (`72..79` at `T_aux = 80`, lag `49..79 > 42`):
        query 4 overwritten + 4 retained keys `[0, k_q, 0, 2]`.
    """
    k_ksel, k_perm, k_vi, k_vo, k_q = jax.random.split(rng_key, 5)
    all_keys = jnp.arange(X3_ASSOC_KEY_OFFSET, X3_ASSOC_KEY_OFFSET + X3_ASSOC_NUM_KEYS, dtype=jnp.int32)
    perm_keys = jax.vmap(lambda rk: jax.random.permutation(rk, all_keys))(jax.random.split(k_perm, batch_size))
    v_init = jax.random.randint(k_vi, (batch_size, X3_ASSOC_NUM_KEYS), X3_ASSOC_VAL_OFFSET, X3_ASSOC_VAL_OFFSET + X3_ASSOC_NUM_VALS)
    v_off = jax.random.randint(k_vo, (batch_size, X3_ASSOC_NUM_KEYS), 1, X3_ASSOC_NUM_VALS)
    v_over = X3_ASSOC_VAL_OFFSET + ((v_init - X3_ASSOC_VAL_OFFSET + v_off) % X3_ASSOC_NUM_VALS)

    sym = jnp.zeros((batch_size, seq_len, 4), dtype=jnp.int32)
    targ = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
    if seq_len < 16:
        sym = sym.at[:, 0, 1].set(perm_keys[:, 0]).at[:, 0, 2].set(v_init[:, 0])
        sym = sym.at[:, 2, 1].set(perm_keys[:, 0]).at[:, 2, 2].set(v_over[:, 0])
        sym = sym.at[:, -1, 1].set(perm_keys[:, 0]).at[:, -1, 3].set(2)
        targ = targ.at[:, -1].set(v_over[:, 0])
        return sym, targ

    if seq_len >= 64:
        k_choices = jnp.array([1, 2, 4, 8, 16], dtype=jnp.int32)
        k_ep = k_choices[jax.random.randint(k_ksel, (batch_size,), 0, 5)]
        m_ep = jnp.where(k_ep == 1, 1, k_ep // 2)  # in {1, 1, 2, 4, 8}
        r_ep = jnp.maximum(0, k_ep - m_ep)
        safe_r = jnp.maximum(1, r_ep)
        for j in range(16):
            act_w = j < k_ep
            sym = sym.at[:, j, 1].set(jnp.where(act_w, perm_keys[:, j], 0)).at[:, j, 2].set(jnp.where(act_w, v_init[:, j], 0))
        for j in range(8):
            act_ov = j < m_ep
            sym = sym.at[:, 16 + j, 1].set(jnp.where(act_ov, perm_keys[:, j], 0)).at[:, 16 + j, 2].set(jnp.where(act_ov, v_over[:, j], 0))
        b_idx = jnp.arange(batch_size)
        q_sh = jax.random.randint(k_q, (batch_size, 8), 0, 16)
        qb = seq_len - 8
        for qi in range(4):
            os = (qi + q_sh[:, qi]) % m_ep
            sym = sym.at[:, qb + qi, 1].set(perm_keys[b_idx, os]).at[:, qb + qi, 3].set(2)
            targ = targ.at[:, qb + qi].set(v_over[b_idx, os])

            rs = m_ep + ((qi + q_sh[:, 4 + qi]) % safe_r)
            fb = (qi + 4 + q_sh[:, 4 + qi]) % m_ep
            has_r = r_ep > 0
            sym = sym.at[:, qb + 4 + qi, 1].set(jnp.where(has_r, perm_keys[b_idx, rs], perm_keys[b_idx, fb])).at[:, qb + 4 + qi, 3].set(2)
            targ = targ.at[:, qb + 4 + qi].set(jnp.where(has_r, v_init[b_idx, rs], v_over[b_idx, fb]))
        return sym, targ

    k_choices = jnp.array([1, 2, 4, 8, 12], dtype=jnp.int32)
    k_ep = k_choices[jax.random.randint(k_ksel, (batch_size,), 0, 5)]
    m_ep = jnp.maximum(1, jnp.minimum(4, k_ep // 2))
    r_ep = jnp.maximum(0, k_ep - m_ep)
    safe_r = jnp.maximum(1, r_ep)
    for j in range(12):
        act_w = j < k_ep
        sym = sym.at[:, j, 1].set(jnp.where(act_w, perm_keys[:, j], 0)).at[:, j, 2].set(jnp.where(act_w, v_init[:, j], 0))
    for j in range(4):
        act_ov = j < m_ep
        sym = sym.at[:, 12 + j, 1].set(jnp.where(act_ov, perm_keys[:, j], 0)).at[:, 12 + j, 2].set(jnp.where(act_ov, v_over[:, j], 0))

    b_idx = jnp.arange(batch_size)
    q_sh = jax.random.randint(k_q, (batch_size, 4), 0, 12)
    qb = seq_len - 4
    for qi in range(2):
        os = (qi + q_sh[:, qi]) % m_ep
        sym = sym.at[:, qb + qi, 1].set(perm_keys[b_idx, os]).at[:, qb + qi, 3].set(2)
        targ = targ.at[:, qb + qi].set(v_over[b_idx, os])
        rs = m_ep + ((qi + q_sh[:, 2 + qi]) % safe_r)
        fb = (qi + 2 + q_sh[:, 2 + qi]) % m_ep
        has_r = r_ep > 0
        sym = sym.at[:, qb + 2 + qi, 1].set(jnp.where(has_r, perm_keys[b_idx, rs], perm_keys[b_idx, fb])).at[:, qb + 2 + qi, 3].set(2)
        targ = targ.at[:, qb + 2 + qi].set(jnp.where(has_r, v_init[b_idx, rs], v_over[b_idx, fb]))
    return sym, targ


def build_x3_long_horizon_aux_batch_t80(
    rng_key: jax.Array,
    batch_size: int = 2,
    seq_len: int = 80,
) -> dict[str, jax.Array]:
    """Build low-HBM (`skip_visual_conv=True`) `T_aux = 80` beyond-window batch for ALL cohorts.

    Simultaneously trains:
      1. `K in {1, 2, 4, 8, 16}` (`M = max(1, K // 2)`) associative overwrites at lag `49..79 > 42`
         (strictly beyond `SLWM-X3`'s 6-layer `W = 8` local RoPE span, forcing `H_sym` recurrence), AND
      2. Delayed & distracted `SLOT_GOAL_LANG` goal-switch (`Goal A` at `t=0..2`, `Goal B` at `t=12..14`,
         distractor at `t=36..38`, queries across `t=20..79`), calibrating ALL `0..63` RoPE relative lags
         for `SlidingWindow-W64-GQA` while training beyond-`W=64` retention on `t=76..79`.
    """
    k_sym, k_ga, k_gb, k_fd, k_sw, k_aud = jax.random.split(rng_key, 6)
    sym_tok, targ_sym = build_x3_sym_assoc_tokens(k_sym, batch_size, seq_len)

    ga = jax.random.uniform(k_ga, (batch_size, 2), minval=-0.65, maxval=0.65)
    gb = jax.random.uniform(k_gb, (batch_size, 2), minval=-0.65, maxval=0.65)
    fdist = jax.random.uniform(k_fd, (batch_size, 2), minval=-0.65, maxval=0.65)
    is_sw = jax.random.bernoulli(k_sw, 0.70, (batch_size,))

    tok_ga = encode_goal_tokens_2d(ga, is_distractor=False)
    tok_gb = encode_goal_tokens_2d(gb, is_distractor=False)
    tok_fd = encode_goal_tokens_2d(fdist, is_distractor=True)
    tok_gq = encode_goal_query_tokens((batch_size,))

    lang_tok = jnp.zeros((batch_size, seq_len, 4), dtype=jnp.int32)
    for ti in range(min(3, seq_len)):
        lang_tok = lang_tok.at[:, ti, :].set(tok_ga)
    t_sw = min(12, max(4, seq_len // 5))
    for ti in range(t_sw, min(t_sw + 3, seq_len)):
        lang_tok = lang_tok.at[:, ti, :].set(jnp.where(is_sw[:, None], tok_gb, 0))
    if seq_len >= 48:
        for ti in range(36, 39):
            lang_tok = lang_tok.at[:, ti, :].set(tok_fd)
    # Dense query sweep across `t = 20..seq_len-1` (excluding distractor ticks `36..38`) so GQA W=64 calibrates lags 6..63!
    q_start = min(20, max(6, seq_len // 3))
    for ti in range(q_start, seq_len):
        if not (36 <= ti < 39):
            lang_tok = lang_tok.at[:, ti, :].set(tok_gq)

    t_grid = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    g_active = jnp.where((is_sw[:, None] & (t_grid >= t_sw))[..., None], gb[:, None, :], ga[:, None, :])
    has_lang = jnp.any(lang_tok > 0, axis=-1).astype(jnp.float32)
    mm = jnp.zeros((batch_size, seq_len, 4), dtype=jnp.float32).at[:, :, 3].set(has_lang)
    goal_sup_mask = ((t_grid >= q_start) & ((t_grid < 36) | (t_grid >= 39))).astype(jnp.float32)

    return {
        "audio": 0.02 * jax.random.normal(k_aud, (batch_size, seq_len, 64), dtype=jnp.float32),
        "movement": jnp.zeros((batch_size, seq_len, 32), dtype=jnp.float32),
        "action": jnp.zeros((batch_size, seq_len, 16), dtype=jnp.float32),
        "sym_tokens": sym_tok,
        "target_sym_val": targ_sym,
        "lang_tokens": lang_tok,
        "target_goal_xy": g_active,
        "goal_sup_mask": goal_sup_mask,
        "modality_mask": mm,
        "embodiment_id": jnp.zeros((batch_size,), dtype=jnp.int32),
        "valid": jnp.ones((batch_size, seq_len), dtype=bool),
    }


def render_arena_480p_batch(
    ee_bt: jax.Array,
    obj_bt: jax.Array,
    goal_bt: jax.Array,
    box_cx: jax.Array,
    box_cy: jax.Array,
    box_hx: jax.Array,
    box_hy: jax.Array,
) -> jax.Array:
    """Fast vectorized `480x640x3` (`480p`) top-down arena renderer in JAX (`uint8`)."""
    b, t = ee_bt.shape[:2]
    ys = jnp.linspace(-1.0, 1.0, 120, dtype=jnp.float32)[:, None]
    xs = jnp.linspace(-1.0, 1.0, 160, dtype=jnp.float32)[None, :]
    bg = jnp.full((b, t, 120, 160, 3), 24.0, dtype=jnp.float32)

    for bi in range(box_cx.shape[-1]):
        act_b = (box_hx[:, bi] > 1e-4)[:, None, None, None]
        in_x = jnp.abs(xs[None, None, :, :] - box_cx[:, bi][:, None, None, None]) <= box_hx[:, bi][:, None, None, None]
        in_y = jnp.abs(ys[None, None, :, :] - box_cy[:, bi][:, None, None, None]) <= box_hy[:, bi][:, None, None, None]
        m_box = (act_b & in_x & in_y)[..., None]
        bg = jnp.where(m_box, jnp.array([192.0, 96.0, 48.0], dtype=jnp.float32), bg)

    dg2 = (xs[None, None, :, :] - goal_bt[:, :, 0:1, None]) ** 2 + (ys[None, None, :, :] - goal_bt[:, :, 1:2, None]) ** 2
    m_goal = ((dg2 <= 0.095**2) & (dg2 >= 0.055**2))[..., None]
    bg = jnp.where(m_goal, jnp.array([46.0, 204.0, 113.0], dtype=jnp.float32), bg)

    do2 = (xs[None, None, :, :] - obj_bt[:, :, 0:1, None]) ** 2 + (ys[None, None, :, :] - obj_bt[:, :, 1:2, None]) ** 2
    m_obj = (do2 <= 0.082**2)[..., None]
    bg = jnp.where(m_obj, jnp.array([235.0, 68.0, 68.0], dtype=jnp.float32), bg)

    de2 = (xs[None, None, :, :] - ee_bt[:, :, 0:1, None]) ** 2 + (ys[None, None, :, :] - ee_bt[:, :, 1:2, None]) ** 2
    m_ee = (de2 <= 0.062**2)[..., None]
    bg = jnp.where(m_ee, jnp.array([52.0, 182.0, 248.0], dtype=jnp.float32), bg)

    img_u8 = jnp.clip(bg, 0.0, 255.0).astype(jnp.uint8)
    return jnp.repeat(jnp.repeat(img_u8, 4, axis=2), 4, axis=3)


def generate_x3_sim_batch_480p(
    rng_key: jax.Array,
    batch_size: int,
    seq_len: int = 24,
) -> dict[str, jax.Array]:
    """Generate strictly in-domain de-privileged `480p` (`480x640x3`) arena physics episodes:
      - Obstacle geometry strictly in-domain (`28%` open, `72%` single `x = 0` centered bar; zero unseen layouts).
      - Continuous randomized in-domain dynamics (`mass_scale in [0.55, 1.85]`, `damping in [0.55, 0.86]`, strictly
        disjoint from held-out `(2.20, 0.90)` `dynamics_shift`) + adaptive gain supervision in `target_action[:, :, 2]`.
      - Exact current-tick `target_vis_pose_t = [ee_t, obj_t]` supervision (zero off-by-one phase-lead bias).
    """
    from data.x1b_unseen_worlds import step_unseen_layout_world
    from evals.x1b_eval_wave7 import _proportional_push_action

    k_w, k_dyn, k_u, k_sym, k_flt, k_sw = jax.random.split(rng_key, 6)
    kw1, kw2, kw3, kw4, kw5, kw_bar, kw_bh = jax.random.split(k_w, 7)
    side = jnp.where(jax.random.bernoulli(kw1, 0.5, (batch_size,)), 1.0, -1.0)
    obj0 = jnp.stack([-0.54 * side, jax.random.uniform(kw2, (batch_size,), minval=-0.38, maxval=0.38)], axis=-1)
    ee0 = jnp.clip(
        obj0 + jnp.stack([-0.16 * side, jax.random.uniform(kw3, (batch_size,), minval=-0.14, maxval=0.14)], axis=-1),
        -0.88,
        0.88,
    )
    targ_a = jnp.stack([0.54 * side, jax.random.uniform(kw4, (batch_size,), minval=-0.42, maxval=0.42)], axis=-1)
    targ_b = jnp.stack([-0.54 * side, jax.random.uniform(kw5, (batch_size,), minval=-0.42, maxval=0.42)], axis=-1)

    has_bar = jax.random.bernoulli(kw_bar, 0.72, (batch_size,))
    khx, khy = jax.random.split(kw_bh, 2)
    bar_hx = jnp.where(has_bar, jax.random.uniform(khx, (batch_size,), minval=0.024, maxval=0.080), 0.0)
    bar_hy = jnp.where(has_bar, jax.random.uniform(khy, (batch_size,), minval=0.20, maxval=0.38), 0.0)
    z_col = jnp.zeros((batch_size,), dtype=jnp.float32)

    kmass, kdamp = jax.random.split(k_dyn, 2)
    mass_sc = jax.random.uniform(kmass, (batch_size,), minval=0.55, maxval=1.85)
    damp_sc = jax.random.uniform(kdamp, (batch_size,), minval=0.55, maxval=0.86)
    # Adaptive push-gain target in [-0.95, 0.95]: heavier mass -> positive boost, low-friction/high-damping-carry -> braking
    dyn_gain_targ = jnp.clip(0.85 * (mass_sc - 1.0) - 1.20 * (damp_sc - 0.72), -0.95, 0.95)

    st0 = {
        "layout_id": jnp.zeros((batch_size,), dtype=jnp.int32),
        "ee_xy": ee0,
        "ee_vel": jnp.zeros((batch_size, 2), dtype=jnp.float32),
        "obj_xy": obj0,
        "obj_vel": jnp.zeros((batch_size, 2), dtype=jnp.float32),
        "targ_a": targ_a,
        "targ_b": targ_b,
        "box_cx": jnp.zeros((batch_size, 3), dtype=jnp.float32),
        "box_cy": jnp.zeros((batch_size, 3), dtype=jnp.float32),
        "box_hx": jnp.stack([bar_hx, z_col, z_col], axis=-1),
        "box_hy": jnp.stack([bar_hy, z_col, z_col], axis=-1),
        "mass_scale": mass_sc,
        "damping": damp_sc,
    }

    is_switch = jax.random.bernoulli(k_sw, 0.45, (batch_size,))
    t_sw = max(4, seq_len // 3)

    def _sim_step(st, t_i):
        use_b = is_switch & (t_i >= t_sw)
        g_cur = jnp.where(use_b[:, None], targ_b, targ_a)
        kn = jax.random.fold_in(k_u, t_i)
        u_nom = _proportional_push_action(st["ee_xy"], st["obj_xy"], g_cur, final_goal_xy=g_cur)[:, :2]
        gain_mult = 1.0 + 0.45 * dyn_gain_targ[:, None]
        u_exec = jnp.clip(u_nom * gain_mult + 0.18 * jax.random.normal(kn, u_nom.shape), -1.0, 1.0)
        st_nxt, contact = step_unseen_layout_world(st, u_exec)
        dist = jnp.sqrt(jnp.sum((st_nxt["obj_xy"] - g_cur) ** 2, axis=-1) + 1e-6)
        val = -1.35 * dist
        return st_nxt, (st["ee_xy"], st["obj_xy"], st_nxt["ee_xy"], st_nxt["obj_xy"], u_exec, g_cur, val, contact)

    _, hist = jax.lax.scan(_sim_step, st0, jnp.arange(seq_len, dtype=jnp.int32))
    ee_t, obj_t, ee_nxt, obj_nxt, u2_t, g_bt, val_t, con_t = [jnp.swapaxes(x, 0, 1) for x in hist]

    t_idx_grid = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    show_g_mask = (t_idx_grid < 3) | (is_switch[:, None] & (t_idx_grid >= t_sw) & (t_idx_grid < t_sw + 3))
    shown_g_bt = jnp.where(show_g_mask[:, :, None], g_bt, jnp.full_like(g_bt, 9.0))
    clean_v = render_arena_480p_batch(
        ee_t, obj_t, shown_g_bt, st0["box_cx"], st0["box_cy"], st0["box_hx"], st0["box_hy"]
    )
    ee_vel = jnp.concatenate([jnp.zeros((batch_size, 1, 2), dtype=jnp.float32), ee_t[:, 1:] - ee_t[:, :-1]], axis=1)
    ee_vel_nxt = ee_nxt - ee_t
    mov_in = jnp.concatenate([ee_t, ee_t, ee_vel, jnp.zeros((batch_size, seq_len, 26), dtype=jnp.float32)], axis=-1)
    t_vis_pose = jnp.concatenate([ee_t, obj_t], axis=-1)
    t_mov_next = jnp.concatenate([ee_nxt, obj_nxt, ee_vel_nxt, jnp.zeros((batch_size, seq_len, 26), dtype=jnp.float32)], axis=-1)
    sh4 = min(4, max(1, seq_len - 1))
    t_mov_s4 = jnp.concatenate([t_mov_next[:, sh4 - 1 :], jnp.repeat(t_mov_next[:, -1:], sh4 - 1, axis=1)], axis=1)

    dyn_ch = jnp.broadcast_to(dyn_gain_targ[:, None, None], (batch_size, seq_len, 1))
    u16_exec = jnp.concatenate([u2_t, jnp.zeros((batch_size, seq_len, 14), dtype=jnp.float32)], axis=-1)
    u16_targ = jnp.concatenate([u2_t, dyn_ch, jnp.zeros((batch_size, seq_len, 13), dtype=jnp.float32)], axis=-1)
    prev_u16 = jnp.concatenate([jnp.zeros((batch_size, 1, 16), dtype=jnp.float32), u16_exec[:, :-1]], axis=1)

    sym_tok, targ_sym = build_x3_sym_assoc_tokens(k_sym, batch_size, seq_len)
    lang_tok = jnp.zeros((batch_size, seq_len, 4), dtype=jnp.int32)
    ga_tok = encode_goal_tokens_2d(targ_a, is_distractor=False)
    gb_tok = encode_goal_tokens_2d(targ_b, is_distractor=False)
    gq_tok = encode_goal_query_tokens((batch_size,))
    for ti in range(min(3, seq_len)):
        lang_tok = lang_tok.at[:, ti, :].set(ga_tok)
    for ti in range(t_sw, min(t_sw + 3, seq_len)):
        lang_tok = lang_tok.at[:, ti, :].set(jnp.where(is_switch[:, None], gb_tok, 0))
    for ti in range(max(0, seq_len - 6), seq_len):
        lang_tok = lang_tok.at[:, ti, :].set(gq_tok)

    k_fc, k_ft, k_fv, k_fp, k_fsh = jax.random.split(k_flt, 5)
    fu = jax.random.uniform(k_fc, (batch_size,))
    is_m1 = (~is_switch) & (fu < 0.34)
    is_m3 = (~is_switch) & (~is_m1) & (fu < 0.56)
    t_flt = jax.random.randint(k_ft, (batch_size, 1), max(6, seq_len // 2), max(7, seq_len - 3))
    flt_on = jnp.arange(seq_len)[None, :] >= t_flt
    m1_m = is_m1[:, None] & flt_on
    m3_m = is_m3[:, None] & flt_on

    k_sk, k_sm, k_ss = jax.random.split(k_fsh, 3)
    use_flip = jax.random.bernoulli(k_sk, 0.5, (batch_size,))
    sh_mag = jax.random.randint(k_sm, (batch_size, 2), 36, 73)
    sh_sgn = jnp.where(jax.random.bernoulli(k_ss, 0.5, (batch_size, 2)), 1, -1)
    sh_yx = sh_mag * sh_sgn
    v_pad = jnp.pad(clean_v, ((0, 0), (0, 0), (72, 72), (72, 72), (0, 0)), mode="edge")
    v_shift = jax.vmap(
        lambda vp, dyx: jax.lax.dynamic_slice(vp, (0, 72 + dyx[0], 72 + dyx[1], 0), (seq_len, 480, 640, 3))
    )(v_pad, sh_yx)
    v_m1 = jnp.where(use_flip[:, None, None, None, None], jnp.flip(clean_v, axis=(2, 3)), v_shift)
    v_stat = jax.random.randint(k_fv, clean_v.shape, 0, 256).astype(jnp.uint8)
    obs_v = jnp.where(m1_m[:, :, None, None, None], v_m1, jnp.where(m3_m[:, :, None, None, None], v_stat, clean_v))

    sp_xy = jnp.clip(ee_t + 0.85 * jnp.sign(jax.random.normal(k_fp, ee_t.shape)), -1.5, 1.5)
    mov_m3 = mov_in.at[:, :, 0:2].set(sp_xy).at[:, :, 2:4].set(sp_xy)
    obs_mov = jnp.where(m3_m[:, :, None], mov_m3, mov_in)
    t_commit = jnp.where(m1_m, 1, jnp.where(m3_m, 3, 0)).astype(jnp.int32)

    is_miss_v = (~is_switch) & (~is_m1) & (~is_m3) & (fu > 0.85)
    has_lang_t = jnp.any(lang_tok > 0, axis=-1).astype(jnp.float32)
    mm = jnp.ones((batch_size, seq_len, 4), dtype=jnp.float32).at[:, :, 0].set(0.0).at[:, :, 3].set(has_lang_t)
    mm = jnp.where(is_miss_v[:, None, None], mm.at[:, :, 1].set(0.0), mm)
    t_commit = jnp.where(is_miss_v[:, None], jnp.int32(2), t_commit)

    con2 = jnp.stack([con_t.astype(jnp.float32), jnp.zeros_like(con_t, dtype=jnp.float32)], axis=-1)
    return {
        "video": obs_v,
        "clean_video": clean_v,
        "audio": jnp.zeros((batch_size, seq_len, 64), dtype=jnp.float32),
        "movement": obs_mov,
        "target_mov_next": t_mov_next,
        "target_mov_step4": t_mov_s4,
        "target_vis_pose_t": t_vis_pose,
        "action": prev_u16,
        "next_action": u16_exec,
        "target_action": u16_targ,
        "has_true_action": jnp.ones((batch_size,), dtype=jnp.float32),
        "sym_tokens": sym_tok,
        "target_sym_val": targ_sym,
        "lang_tokens": lang_tok,
        "target_lang_tokens": jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        "target_goal_xy": g_bt,
        "target_value": val_t,
        "target_contacts": con2,
        "target_commit": t_commit,
        "target_occ_boxes": jnp.stack([st0["box_cx"], st0["box_cy"], st0["box_hx"], st0["box_hy"]], axis=-1),
        "modality_mask": mm,
        "full_modality_mask": jnp.ones((batch_size, seq_len, 4), dtype=jnp.float32).at[:, :, 0].set(0.0).at[:, :, 3].set(has_lang_t),
        "embodiment_id": jnp.zeros((batch_size,), dtype=jnp.int32),
        "valid": jnp.ones((batch_size, seq_len), dtype=bool),
        "scenario": jnp.ones((batch_size,), dtype=jnp.int32),
    }


def compute_true_occ_8x8_from_boxes(boxes_b34: jax.Array) -> jax.Array:
    cs = -1.0 + (2.0 * (jnp.arange(8, dtype=jnp.float32) + 0.5) / 8.0)
    gy, gx = jnp.meshgrid(cs, cs, indexing="ij")
    occ = jnp.zeros((boxes_b34.shape[0], 8, 8), dtype=jnp.float32)
    for bi in range(boxes_b34.shape[1]):
        cx, cy, hx, hy = boxes_b34[:, bi, 0], boxes_b34[:, bi, 1], boxes_b34[:, bi, 2], boxes_b34[:, bi, 3]
        act = (hx > 1e-4)[:, None, None]
        hit = act & (jnp.abs(gx[None, :, :] - cx[:, None, None]) <= (hx[:, None, None] + 0.12)) & (
            jnp.abs(gy[None, :, :] - cy[:, None, None]) <= (hy[:, None, None] + 0.12)
        )
        occ = jnp.maximum(occ, hit.astype(jnp.float32))
    return occ


def world_model_loss_x3(
    params: dict[str, jax.Array],
    ema_params: dict[str, jax.Array],
    spec: X3Spec,
    batch: dict[str, jax.Array],
    rng_key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Unified Study X3 loss training world dynamics, `T=80` beyond-window recall/RoPE, spoken grounding, & 6-way composition."""
    b, t = batch["video"].shape[:2]
    valid = batch.get("valid", jnp.ones((b, t), dtype=bool)).astype(jnp.float32)
    mm_obs = batch["modality_mask"].astype(jnp.float32) * valid[:, :, None]
    mm_full = batch.get("full_modality_mask", mm_obs).astype(jnp.float32) * valid[:, :, None]
    scen = batch["scenario"].astype(jnp.int32)
    is_arena = (scen > 0).astype(jnp.float32)[:, None]
    has_true_act = batch.get("has_true_action", is_arena[:, 0]).astype(jnp.float32)

    # Strict causal action sanitization: on real clips (`scen < 0`), zero `action` whenever `mm_obs[mov] == 0` or `has_true_action == 0`!
    allow_obs_act = jnp.maximum(is_arena, has_true_act[:, None] * mm_obs[:, :, 2])[:, :, None]
    san_batch = {
        **batch,
        "audio": jnp.where(mm_obs[:, :, 0:1] > 0.5, batch["audio"], 0.0),
        "video": jnp.where(mm_obs[:, :, 1:2, None, None] > 0.5, batch["video"], jnp.zeros_like(batch["video"])),
        "movement": jnp.where(mm_obs[:, :, 2:3] > 0.5, batch["movement"], 0.0),
        "action": batch["action"].astype(jnp.float32) * allow_obs_act,
        "lang_tokens": jnp.where(mm_obs[:, :, 3:4] > 0.5, batch["lang_tokens"], 0),
        "modality_mask": mm_obs,
    }

    z_top, fine_300, beta_aud, _ = forward_x3_world_model(params, spec, san_batch)
    z_sym, z_task = fork_x3_streams(z_top)

    # Zero future `target_action` leak into `predict_x3_latent_step` on real clips:
    # Arena (`scen > 0`) uses executed transition action `next_action`; real clips use `san_batch["action"]` if unmasked, else `stop_grad(pred_action)`!
    pred_act_early = jax.lax.stop_gradient(predict_x3_causal_action_from_ztask(params, z_task, san_batch))
    has_obs_act = jnp.sum(jnp.abs(san_batch["action"]), axis=-1, keepdims=True) > 1e-5
    u_step = jnp.where(
        is_arena[:, :, None] > 0.5,
        san_batch.get("next_action", san_batch["action"]),
        jnp.where(has_obs_act, san_batch["action"], pred_act_early),
    )
    z_next_1, lv_1 = predict_x3_latent_step(params, spec, z_task, u_step)
    z_next_2, _ = predict_x3_latent_step(params, spec, z_next_1[:, :-1], u_step[:, 1:]) if t > 2 else (None, None)

    commit_target = san_batch["target_commit"].astype(jnp.int32)
    clean_tick = ((commit_target == 0).astype(jnp.float32)) * valid
    clean_pair = clean_tick[:, :-1] * clean_tick[:, 1:]

    allow_full_act = jnp.maximum(is_arena, has_true_act[:, None] * mm_full[:, :, 2])[:, :, None]
    clean_in = {
        **san_batch,
        "audio": jnp.where(mm_full[:, :, 0:1] > 0.5, batch["audio"], 0.0),
        "video": batch.get("clean_video", batch["video"]),
        "movement": jnp.where(mm_full[:, :, 2:3] > 0.5, batch["movement"], 0.0),
        "action": batch["action"].astype(jnp.float32) * allow_full_act,
        "lang_tokens": jnp.where(mm_full[:, :, 3:4] > 0.5, batch["lang_tokens"], 0),
        "modality_mask": mm_full,
    }
    z_ema_raw, _, _, _ = forward_x3_world_model(jax.lax.stop_gradient(ema_params), spec, clean_in)
    _, z_ema_task = fork_x3_streams(jax.lax.stop_gradient(z_ema_raw))

    diff1 = (z_next_1[:, :-1].astype(jnp.float32) - z_ema_task[:, 1:].astype(jnp.float32)) ** 2
    jepa_1 = jnp.sum(jnp.mean(diff1, axis=(2, 3)) * clean_pair) / jnp.maximum(jnp.sum(clean_pair), 1.0)
    if z_next_2 is not None and t > 2:
        cp2 = clean_tick[:, :-2] * clean_tick[:, 2:]
        diff2 = (z_next_2[:, :-1].astype(jnp.float32) - z_ema_task[:, 2:].astype(jnp.float32)) ** 2
        jepa_2 = jnp.sum(jnp.mean(diff2, axis=(2, 3)) * cp2) / jnp.maximum(jnp.sum(cp2), 1.0)
    else:
        jepa_2 = jnp.zeros((), dtype=jnp.float32)
    jepa_loss = jepa_1 + 0.5 * jepa_2

    inv_v = jnp.exp(-lv_1[:, :-1].astype(jnp.float32))
    unc_nll = 0.5 * (diff1 * inv_v + lv_1[:, :-1].astype(jnp.float32))
    unc_loss = jnp.sum(jnp.mean(unc_nll, axis=(2, 3)) * clean_pair) / jnp.maximum(jnp.sum(clean_pair), 1.0)

    k_sv, k_sr, k_flip, k_cmd, k_ctx, k_aux = jax.random.split(rng_key, 6)
    vis_clean = (clean_tick * mm_obs[:, :, 1]).reshape(-1)
    sig_v, pr_v = epps_pulley_sigreg_loss(
        jnp.mean(z_task[:, :, X3_SLOTS_VISUAL, :].astype(jnp.float32), axis=2).reshape(-1, spec.d_model),
        k_sv,
        mask=vis_clean,
    )
    sig_r, _ = epps_pulley_sigreg_loss(
        jnp.mean(z_task[:, :, X3_SLOTS_REGISTER, :].astype(jnp.float32), axis=2).reshape(-1, spec.d_model),
        k_sr,
        mask=clean_tick.reshape(-1),
    )
    sig_loss = 0.5 * (sig_v + sig_r)

    vid_t_idx = jnp.asarray(
        np.unique(np.clip(np.array([0, t // 3, (2 * t) // 3, max(0, t - 2)], dtype=np.int32), 0, max(0, t - 2))),
        dtype=jnp.int32,
    )
    keep_ctx = jax.random.bernoulli(k_ctx, 0.70, (b, 1, 1, 1, 1)) & (mm_obs[:, :1, 1:2, None, None] > 0.5)
    ctx_f0 = jnp.where(keep_ctx, san_batch["video"][:, :1], jnp.zeros_like(san_batch["video"][:, :1]))
    heads = decode_x3_any_to_any_heads(
        params,
        spec,
        z_sym,
        z_task,
        z_next_1,
        lv_1,
        fine_300,
        san_batch,
        decode_480p_frames=False,
    )
    reg_sub = jnp.mean(z_task[:, vid_t_idx, X3_SLOTS_REGISTER, :].astype(jnp.float32), axis=2)
    pred_v480_sub = decode_480p_video_from_latents(
        params, spec, z_next_1[:, vid_t_idx, X3_SLOTS_VISUAL, :], reg_sub, context_frame_480p=ctx_f0
    )
    targ_v480_sub = clean_in["video"][:, vid_t_idx + 1].astype(jnp.float32) / 127.5 - 1.0
    ctx_f0_norm = clean_in["video"][:, :1].astype(jnp.float32) / 127.5 - 1.0
    motion_w = 1.0 + 2.5 * (jnp.abs(targ_v480_sub - ctx_f0_norm) > 0.06).astype(jnp.float32)
    v_sub_mask = mm_full[:, vid_t_idx + 1, 1] * clean_tick[:, vid_t_idx + 1]
    l1_v = jnp.mean(motion_w * jnp.abs(pred_v480_sub - targ_v480_sub), axis=(2, 3, 4))
    gx_p = pred_v480_sub[:, :, :, 1:, :] - pred_v480_sub[:, :, :, :-1, :]
    gx_t = targ_v480_sub[:, :, :, 1:, :] - targ_v480_sub[:, :, :, :-1, :]
    gy_p = pred_v480_sub[:, :, 1:, :, :] - pred_v480_sub[:, :, :-1, :, :]
    gy_t = targ_v480_sub[:, :, 1:, :, :] - targ_v480_sub[:, :, :-1, :, :]
    edge_v = 0.5 * (jnp.mean(jnp.abs(gx_p - gx_t), axis=(2, 3, 4)) + jnp.mean(jnp.abs(gy_p - gy_t), axis=(2, 3, 4)))
    vid_480p_loss = jnp.sum((l1_v + 0.7 * edge_v) * v_sub_mask) / jnp.maximum(jnp.sum(v_sub_mask), 1.0)

    # 1. Pure `SLOT_SYM_MEM` Associative Overwrite CE (`[64..95]`) at `seq_len = 24` + `T_aux = 80` (`lag = 49..79 > 42`!)
    tsym = san_batch["target_sym_val"].astype(jnp.int32)
    sym_m = (tsym >= X3_ASSOC_VAL_OFFSET).astype(jnp.float32) * valid
    sym_idx = jnp.clip(tsym - X3_ASSOC_VAL_OFFSET, 0, X3_ASSOC_NUM_VALS - 1)
    sym_lp = jax.nn.log_softmax(heads["sym_val_logits"], axis=-1)
    sym_nll = -jnp.take_along_axis(sym_lp, sym_idx[..., None], axis=-1)[..., 0]
    sym_assoc_ce = jnp.sum(sym_nll * sym_m) / jnp.maximum(jnp.sum(sym_m), 1.0)

    # Every-step `T_aux = 80` beyond-window auxiliary pass (`skip_visual_conv=True`, active when `target_sym_val` supervision is enabled)
    has_any_sym_sup = jnp.sum(sym_m) > 0.5
    aux_t = 80 if t >= 16 else 16
    aux_b = build_x3_long_horizon_aux_batch_t80(k_aux, batch_size=2, seq_len=aux_t)
    z_aux_top, _, _, _ = forward_x3_world_model(params, spec, aux_b, skip_visual_conv=True)
    z_aux_sym, z_aux_task = fork_x3_streams(z_aux_top)
    assoc_vw = params["text_emb"].astype(jnp.float32)[X3_ASSOC_VAL_OFFSET : X3_ASSOC_VAL_OFFSET + X3_ASSOC_NUM_VALS].T
    aux_sym_logits = ((z_aux_sym.astype(jnp.float32) @ params["head_sym_w"].astype(jnp.float32)) @ assoc_vw) * (spec.d_model**-0.5)
    aux_tsym = aux_b["target_sym_val"].astype(jnp.int32)
    aux_sym_m = (aux_tsym >= X3_ASSOC_VAL_OFFSET).astype(jnp.float32)
    aux_sym_lp = jax.nn.log_softmax(aux_sym_logits, axis=-1)
    aux_sym_nll = -jnp.take_along_axis(aux_sym_lp, jnp.clip(aux_tsym - X3_ASSOC_VAL_OFFSET, 0, X3_ASSOC_NUM_VALS - 1)[..., None], axis=-1)[..., 0]
    sym_aux80_ce = jnp.where(
        has_any_sym_sup,
        jnp.sum(aux_sym_nll * aux_sym_m) / jnp.maximum(jnp.sum(aux_sym_m), 1.0),
        0.0,
    )
    aux_reg_dir = jnp.mean(z_aux_task[:, :, 57:59, :].astype(jnp.float32), axis=2)
    aux_pg = jnp.tanh(aux_reg_dir @ params["head_goal_xy_w"].astype(jnp.float32) + params["head_goal_xy_b"].astype(jnp.float32))
    goal_aux80_mse = jnp.sum(jnp.mean((aux_pg - aux_b["target_goal_xy"]) ** 2, axis=-1) * aux_b["goal_sup_mask"]) / jnp.maximum(
        jnp.sum(aux_b["goal_sup_mask"]), 1.0
    )
    sym_assoc_ce = 0.45 * sym_assoc_ce + 0.55 * sym_aux80_ce

    # 2. Direction (1): `Video / Audio -> Text` subword generation CE (STRICTLY on `mm_obs[txt] == 0`)
    tlang = batch["target_lang_tokens"].astype(jnp.int32)
    lang_gen_m = (tlang > 0).astype(jnp.float32) * mm_full[:, :, 3] * (1.0 - mm_obs[:, :, 3]) * clean_tick
    lang_lp = jax.nn.log_softmax(heads["pred_lang_logits"], axis=-1)
    lang_nll = -jnp.take_along_axis(lang_lp, jnp.clip(tlang, 0, spec.vocab_size - 1)[..., None], axis=-1)[..., 0]
    vid_aud_to_txt_ce = jnp.sum(lang_nll * lang_gen_m) / jnp.maximum(jnp.sum(lang_gen_m), 1.0)

    # 3. Direction (3): `Video (+ Action) -> 64-band Audio` synthesis MSE
    aud_gen_m = mm_full[:, :, 0] * clean_tick * (0.25 + 3.75 * (1.0 - mm_obs[:, :, 0]))
    targ_aud = batch["audio"].astype(jnp.float32)
    aud_mse = jnp.mean((heads["pred_audio_mel"] - targ_aud) ** 2, axis=-1)
    vid_to_aud_loss = jnp.sum(aud_mse * aud_gen_m) / jnp.maximum(jnp.sum(aud_gen_m), 1.0)

    # 4. Direction (4): `Text / Video -> 32D Movement` flow MSE (supervises strictly physical kinematics `mm_full[mov] == 1`)
    mov_gen_m = mm_full[:, :, 2] * clean_tick * (0.35 + 3.65 * (1.0 - mm_obs[:, :, 2]))
    mov1 = jnp.mean((heads["pred_mov_next"] - batch["target_mov_next"].astype(jnp.float32)) ** 2, axis=-1)
    mov4 = jnp.mean((heads["pred_mov_step4"] - batch["target_mov_step4"].astype(jnp.float32)) ** 2, axis=-1)
    txt_vid_to_mov_loss = jnp.sum((mov1 + 0.6 * mov4) * mov_gen_m) / jnp.maximum(jnp.sum(mov_gen_m), 1.0)

    # 5. Direction (5) & (6): True Robot Actions (`has_true_action > 0.5`), Blind Contact, & Exact Current-Tick Arena Pose
    act_m = clean_tick * jnp.maximum(is_arena, has_true_act[:, None])
    act_mse = jnp.mean((heads["pred_action"] - batch["target_action"].astype(jnp.float32)) ** 2, axis=-1)
    act_loss = jnp.sum(act_mse * act_m) / jnp.maximum(jnp.sum(act_m), 1.0)

    # Direction 6 supervises genuinely occluded 480p visual frame-difference motion/impact (`||V_t - V_{t-1}||_1 > 0.025`),
    # NEVER an instantaneous threshold of observed `audio_t`!
    cv_f = clean_in["video"].astype(jnp.float32) / 255.0
    cv_prev = jnp.concatenate([cv_f[:, :1], cv_f[:, :-1]], axis=1)
    vis_motion_mag = jnp.mean(jnp.abs(cv_f - cv_prev), axis=(2, 3, 4))
    real_vis_impact = ((vis_motion_mag > 0.025) & (mm_full[:, :, 1] > 0.5)).astype(jnp.float32)
    real_moving = ((jnp.linalg.norm(batch["target_mov_next"][:, :, :6] - batch["movement"][:, :, :6], axis=-1) > 0.04) & (mm_full[:, :, 2] > 0.5)).astype(jnp.float32)
    targ_con = jnp.where(is_arena[..., None] > 0.5, batch["target_contacts"].astype(jnp.float32), jnp.stack([real_vis_impact, real_moving], axis=-1))
    con_bce = jnp.mean(
        jnp.maximum(heads["contact_logits"], 0.0)
        - heads["contact_logits"] * targ_con
        + jnp.log1p(jnp.exp(-jnp.abs(heads["contact_logits"]))),
        axis=-1,
    )
    contact_loss = jnp.sum(con_bce * clean_tick) / jnp.maximum(jnp.sum(clean_tick), 1.0)

    arena_vis_m = is_arena * clean_tick * mm_obs[:, :, 1]
    denom_av = jnp.maximum(jnp.sum(arena_vis_m), 1.0)
    t_vp = batch.get("target_vis_pose_t", batch["target_mov_next"][:, :, 0:4]).astype(jnp.float32)
    vis_pose_loss = (
        jnp.sum(jnp.mean((heads["vis_ee_xy"] - t_vp[:, :, 0:2]) ** 2, axis=-1) * arena_vis_m) / denom_av
        + jnp.sum(jnp.mean((heads["vis_obj_xy"] - t_vp[:, :, 2:4]) ** 2, axis=-1) * arena_vis_m) / denom_av
    )
    goal_loss = (
        jnp.sum(jnp.mean((heads["pred_goal_xy"] - batch["target_goal_xy"].astype(jnp.float32)) ** 2, axis=-1) * is_arena * valid)
        / jnp.maximum(jnp.sum(is_arena * valid), 1.0)
        + 0.85 * goal_aux80_mse
    )
    val_loss = jnp.sum(((heads["value_pred"] - batch["target_value"].astype(jnp.float32)) ** 2) * is_arena * clean_tick) / jnp.maximum(jnp.sum(is_arena * clean_tick), 1.0)

    if "target_occ_boxes" in batch:
        occ_true = compute_true_occ_8x8_from_boxes(batch["target_occ_boxes"])[:, None, :, :]
        occ_bce = jnp.mean(
            jnp.maximum(heads["occ_logits_8x8"], 0.0)
            - heads["occ_logits_8x8"] * occ_true
            + jnp.log1p(jnp.exp(-jnp.abs(heads["occ_logits_8x8"]))),
            axis=(2, 3),
        )
        occ_loss = jnp.sum(occ_bce * arena_vis_m) / denom_av
    else:
        occ_loss = jnp.zeros((), dtype=jnp.float32)

    commit_sup_m = is_arena * valid
    commit_w = jnp.array([1.0, 4.0, 4.5, 2.0], dtype=jnp.float32)[commit_target] * commit_sup_m
    commit_lp = jax.nn.log_softmax(heads["commit_logits"], axis=-1)
    commit_nll = -jnp.take_along_axis(commit_lp, commit_target[..., None], axis=-1)[..., 0]
    commit_ce = jnp.sum(commit_nll * commit_w) / jnp.maximum(jnp.sum(commit_w), 1.0)

    # 6. Physical `xArm6` Two-Utterance Speech + Disentangled `H_obj` (`slot 56`) / `H_dir` (`slots 57..58`) Grounding
    rank_margin = jnp.zeros((), dtype=jnp.float32)
    aud_gate_reg = jnp.zeros((), dtype=jnp.float32)
    if "rob_audio_32" in batch:
        b_r = int(batch["rob_audio_32"].shape[0])
        t_r = int(batch["rob_audio_32"].shape[1])
        b2 = 2 * b_r
        cmd_u = jax.random.uniform(k_cmd, (b_r,))
        txt_only = cmd_u < 0.30
        keep_a = jnp.concatenate([jnp.ones((b_r,), dtype=jnp.float32), (~txt_only).astype(jnp.float32)], axis=0)
        keep_t = jnp.concatenate([jnp.zeros((b_r,), dtype=jnp.int32), jnp.ones((b_r,), dtype=jnp.int32)], axis=0)
        has_lt_r = jnp.any(jnp.concatenate([batch["rob_text_32"], batch["rob_text_32"]], axis=0) > 0, axis=-1).astype(jnp.float32)
        mm_r = (
            jnp.ones((b2, t_r, 4), dtype=jnp.float32)
            .at[:, :, 0]
            .set(keep_a[:, None])
            .at[:, :, 3]
            .set(keep_t.astype(jnp.float32)[:, None] * has_lt_r)
        )

        v0_64 = jnp.concatenate([batch["rob_vid0"], batch["rob_vid0"]], axis=0)
        flip_m = jax.random.bernoulli(k_flip, 0.5, (b2,))
        v0_64 = jnp.where(flip_m[:, None, None, None, None], jnp.flip(v0_64, axis=(2, 3)), v0_64)
        flip_s = jnp.where(flip_m[:, None], -1.0, 1.0)
        v0_480 = jax.image.resize(v0_64[:, 0].astype(jnp.float32), (b2, 480, 640, 3), method="linear").astype(jnp.uint8)[:, None]
        # Encode the static t=0 xArm6 scene frame ONCE (`b2` frames instead of `3 * b2 * 32` frames!)
        # and fuse `[true, wrong_dir, wrong_col]` into ONE batched `forward_x3_world_model` call (`3x` fewer XLA scans)!
        vm_0, f300_0, vp4_0 = encode_480p_visual_stem(params, spec, v0_480, jnp.ones((b2, 1), dtype=jnp.float32))
        f300_r = jnp.broadcast_to(f300_0, (b2, t_r, 300, spec.d_model))
        vm_cat = jnp.broadcast_to(jnp.concatenate([vm_0, vm_0, vm_0], axis=0), (3 * b2, t_r, 48, spec.d_model))
        f3_cat = jnp.broadcast_to(jnp.concatenate([f300_0, f300_0, f300_0], axis=0), (3 * b2, t_r, 300, spec.d_model))
        vp_cat = jnp.broadcast_to(jnp.concatenate([vp4_0, vp4_0, vp4_0], axis=0), (3 * b2, t_r, 4))

        p0_16 = jnp.concatenate([batch["rob_prop0"], batch["rob_prop0"]], axis=0)
        mov_r = jnp.broadcast_to(jnp.pad(p0_16, ((0, 0), (0, 0), (0, 16))), (b2, t_r, 32))
        aud_tr = jnp.concatenate([batch["rob_audio_32"], batch["rob_audio_32"]], axis=0) * keep_a[:, None, None]
        aud_wr = jnp.concatenate([batch["rob_wrong_audio_32"], batch["rob_wrong_audio_32"]], axis=0) * keep_a[:, None, None]
        aud_wc = jnp.concatenate([batch["rob_wrong_col_audio_32"], batch["rob_wrong_col_audio_32"]], axis=0) * keep_a[:, None, None]
        txt_tr = jnp.concatenate([batch["rob_text_32"], batch["rob_text_32"]], axis=0) * keep_t[:, None, None]
        txt_wr = jnp.concatenate([batch["rob_wrong_text_32"], batch["rob_wrong_text_32"]], axis=0) * keep_t[:, None, None]
        txt_wc = jnp.concatenate([batch["rob_wrong_col_text_32"], batch["rob_wrong_col_text_32"]], axis=0) * keep_t[:, None, None]
        end_xy = jnp.concatenate([batch["rob_end_xy"], batch["rob_end_xy"]], axis=0)
        tb_xy = jnp.concatenate([batch["rob_target_block_xy"], batch["rob_target_block_xy"]], axis=0) * flip_s
        db_xy = jnp.concatenate([batch["rob_distractor_block_xy"], batch["rob_distractor_block_xy"]], axis=0) * flip_s

        mm_tr = mm_r
        mm_wr = mm_r.at[:, :, 3].set(keep_t.astype(jnp.float32)[:, None] * jnp.any(txt_wr > 0, axis=-1).astype(jnp.float32))
        mm_wc = mm_r.at[:, :, 3].set(keep_t.astype(jnp.float32)[:, None] * jnp.any(txt_wc > 0, axis=-1).astype(jnp.float32))
        cat_rb = {
            "audio": jnp.concatenate([aud_tr, aud_wr, aud_wc], axis=0),
            "movement": jnp.concatenate([mov_r, mov_r, mov_r], axis=0),
            "action": jnp.zeros((3 * b2, t_r, 16), dtype=jnp.float32),
            "sym_tokens": jnp.zeros((3 * b2, t_r, 4), dtype=jnp.int32),
            "lang_tokens": jnp.concatenate([txt_tr, txt_wr, txt_wc], axis=0),
            "modality_mask": jnp.concatenate([mm_tr, mm_wr, mm_wc], axis=0),
            "embodiment_id": jnp.full((3 * b2,), 7, dtype=jnp.int32),
            "valid": jnp.ones((3 * b2, t_r), dtype=bool),
        }
        z_cat, _, ba_cat, _ = forward_x3_world_model(params, spec, cat_rb, precomputed_visual=(vm_cat, f3_cat, vp_cat))
        zt_r, zw_r, zc_r = z_cat[:b2], z_cat[b2 : 2 * b2], z_cat[2 * b2 :]
        ba_tr = ba_cat[:b2]
        _, zt_tk = fork_x3_streams(zt_r)
        _, zw_tk = fork_x3_streams(zw_r)
        _, zc_tk = fork_x3_streams(zc_r)

        max_band = jnp.max(aud_tr[:b_r], axis=-1, keepdims=True)
        speech_on = (max_band > -0.95).astype(jnp.float32)
        silent_on = (max_band < -1.20).astype(jnp.float32)
        aud_gate_reg = jnp.mean(silent_on * (ba_tr[:b_r] ** 2)) + 0.5 * jnp.mean(speech_on * ((1.0 - ba_tr[:b_r]) ** 2))

        rg_tr = jnp.mean(zt_tk[:, -1, 57:59, :].astype(jnp.float32), axis=1)
        rg_wr = jnp.mean(zw_tk[:, -1, 57:59, :].astype(jnp.float32), axis=1)
        pg_tr = jnp.tanh(rg_tr @ params["head_goal_xy_w"].astype(jnp.float32) + params["head_goal_xy_b"].astype(jnp.float32))
        pg_wr = jnp.tanh(rg_wr @ params["head_goal_xy_w"].astype(jnp.float32) + params["head_goal_xy_b"].astype(jnp.float32))
        d_tr = jnp.mean((pg_tr - end_xy) ** 2, axis=-1)
        d_wr = jnp.mean((pg_wr - end_xy) ** 2, axis=-1)

        pb_tr = decode_x3_target_block_xy(params, zt_tk[:, -1:], f300_r[:, -1:])[:, 0]
        pb_wc = decode_x3_target_block_xy(params, zc_tk[:, -1:], f300_r[:, -1:])[:, 0]
        bt_t = jnp.mean((pb_tr - tb_xy) ** 2, axis=-1)
        bt_d = jnp.mean((pb_tr - db_xy) ** 2, axis=-1)
        bw_d = jnp.mean((pb_wc - db_xy) ** 2, axis=-1)
        bw_t = jnp.mean((pb_wc - tb_xy) ** 2, axis=-1)

        goal_loss = goal_loss + 2.5 * jnp.mean(d_tr) + 3.2 * 0.5 * jnp.mean(bt_t + bw_d)
        rank_margin = jnp.mean(jax.nn.relu(0.16 + d_tr - d_wr)) + 0.5 * jnp.mean(
            jax.nn.relu(0.12 + bt_t - bt_d) + jax.nn.relu(0.12 + bw_d - bw_t)
        )

    total_loss = (
        6.0 * jepa_loss
        + 0.20 * unc_loss
        + 1.0 * sig_loss
        + 4.5 * sym_assoc_ce
        + 1.2 * vid_aud_to_txt_ce
        + 1.5 * vid_480p_loss
        + 1.4 * vid_to_aud_loss
        + 1.6 * txt_vid_to_mov_loss
        + 1.8 * act_loss
        + 2.5 * vis_pose_loss
        + 5.5 * goal_loss
        + 0.4 * val_loss
        + 0.8 * occ_loss
        + 0.6 * contact_loss
        + 0.8 * commit_ce
        + 12.0 * rank_margin
        + 1.5 * aud_gate_reg
    )
    mets = {
        "loss": total_loss,
        "jepa": jepa_loss,
        "sigreg": sig_loss,
        "pr": pr_v,
        "sym_ce": sym_assoc_ce,
        "sym_aux80": sym_aux80_ce,
        "va2txt_ce": vid_aud_to_txt_ce,
        "vid480_l1": vid_480p_loss,
        "v2aud_mse": vid_to_aud_loss,
        "mov_mse": txt_vid_to_mov_loss,
        "act_mse": act_loss,
        "vis_pose": vis_pose_loss,
        "goal_mse": goal_loss,
        "commit_ce": commit_ce,
        "rank_margin": rank_margin,
    }
    return total_loss, mets


def mix_x3_real_and_sim_batches(
    real_b: dict[str, np.ndarray] | None,
    rob_b: dict[str, np.ndarray] | None,
    sim_b: dict[str, jax.Array],
    *,
    rob_tok16_tuple: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, jax.Array]:
    """Combine `real_b` (`480p` multi-corpus) + `sim_b` (`480p` arena) + `rob_b` (`32`-tick two-utterance `xArm6`)."""
    if real_b is None:
        out = dict(sim_b)
    else:
        out = {}
        common_keys = (
            "video", "clean_video", "audio", "movement", "target_mov_next", "target_mov_step4",
            "target_vis_pose_t", "action", "next_action", "target_action", "has_true_action",
            "sym_tokens", "target_sym_val", "lang_tokens", "target_lang_tokens", "target_goal_xy",
            "target_value", "target_contacts", "target_commit", "modality_mask", "full_modality_mask",
            "embodiment_id", "valid", "scenario",
        )
        for k in common_keys:
            out[k] = jnp.concatenate([jnp.asarray(real_b[k]), sim_b[k]], axis=0)
        if "target_occ_boxes" in sim_b:
            nr = int(real_b["video"].shape[0])
            z_box = jnp.zeros((nr, *sim_b["target_occ_boxes"].shape[1:]), dtype=jnp.float32)
            out["target_occ_boxes"] = jnp.concatenate([z_box, sim_b["target_occ_boxes"]], axis=0)

    if rob_b is not None:
        t_r = int(rob_b["audio"].shape[1])
        if rob_tok16_tuple is not None:
            t_tr, t_wr, t_wc = rob_tok16_tuple
            txt32_tr = encode_wave8a_two_utterance_tokens_32x4(t_tr, t_r)
            txt32_wr = encode_wave8a_two_utterance_tokens_32x4(t_wr, t_r)
            txt32_wc = encode_wave8a_two_utterance_tokens_32x4(t_wc, t_r)
        else:
            txt32_tr = rob_b["text"]
            txt32_wr = rob_b["wrong_text"]
            txt32_wc = rob_b["wrong_color_text"]
        out["rob_vid0"] = jnp.asarray(rob_b["video"][:, :1])
        out["rob_prop0"] = jnp.asarray(rob_b["proprio"][:, :1], dtype=jnp.float32)
        out["rob_audio_32"] = jnp.asarray(rob_b["audio"], dtype=jnp.float32)
        out["rob_wrong_audio_32"] = jnp.asarray(rob_b["wrong_audio"], dtype=jnp.float32)
        out["rob_wrong_col_audio_32"] = jnp.asarray(rob_b["wrong_color_audio"], dtype=jnp.float32)
        out["rob_text_32"] = jnp.asarray(txt32_tr, dtype=jnp.int32)
        out["rob_wrong_text_32"] = jnp.asarray(txt32_wr, dtype=jnp.int32)
        out["rob_wrong_col_text_32"] = jnp.asarray(txt32_wc, dtype=jnp.int32)
        out["rob_end_xy"] = jnp.asarray(rob_b["target_goal_xy"][:, 0, :], dtype=jnp.float32)
        out["rob_target_block_xy"] = jnp.asarray(rob_b["target_block_xy"], dtype=jnp.float32)
        out["rob_distractor_block_xy"] = jnp.asarray(rob_b["distractor_block_xy"], dtype=jnp.float32)
    return out


def verify_x3_dataset_roles_fail_closed(
    train_480p_path: str | Path | None,
    dev_480p_path: str | Path | None,
    vault_480p_path: str | Path | None,
    grounded_robot_path: str | Path | None,
    vault_robot_path: str | Path | None,
    *,
    smoke: bool,
    cards_cfg_path: Path | None = None,
) -> dict[str, Any]:
    cfg_p = cards_cfg_path if cards_cfg_path is not None else (REPO_ROOT / "configs/x3/x3_dataset_cards.json")
    roles = (
        ("x3_train_480p_sha256", train_480p_path, "x3_multimodal_480p_train"),
        ("x3_dev_480p_sha256", dev_480p_path, "x3_multimodal_480p_dev"),
        ("x3_vault_480p_sha256", vault_480p_path, "x3_multimodal_480p_vault"),
        ("grounded_robot_sha256", grounded_robot_path, "x1b_wave8a_grounded_v2"),
        ("vault_robot_sha256", vault_robot_path, "x1b_wave8a_vault_v2"),
    )
    out: dict[str, Any] = {}
    for pk, pv, _ in roles:
        po = Path(pv) if pv else None
        out[pk] = sha256_file(po) if (po is not None and po.is_file()) else None
    if smoke:
        out["registered_cards_verified"] = False
        return out
    if not cfg_p.is_file():
        raise RuntimeError(f"Non-smoke Study X3 training requires frozen dataset cards registry at {cfg_p}")
    reg = json.loads(cfg_p.read_text())
    for pk, pv, ck in roles:
        if ck not in reg or "sha256" not in reg[ck]:
            raise RuntimeError(f"Missing dataset role {ck!r} in {cfg_p}")
        if not pv or not Path(pv).is_file() or out[pk] != reg[ck]["sha256"]:
            raise RuntimeError(f"Dataset SHA mismatch for role {ck!r}: got={out[pk]}, expected={reg[ck]['sha256']}")
    out["registered_cards_verified"] = True
    return out


def build_x3_binding_doc(
    *,
    tag: str,
    cohort: str,
    spec: X3Spec,
    seed: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    smoke: bool,
    save_bf16: bool,
) -> dict[str, Any]:
    return {
        "tag": str(tag),
        "cohort": str(cohort),
        "wave": "x3_1b_480p_scaling_composition_and_livestream",
        "spec": asdict(spec),
        "seed": int(seed),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "aux_long_horizon_batch_size": 2,
        "aux_long_horizon_seq_len": 80 if seq_len >= 16 else 16,
        "peak_lr": float(X3_PEAK_LRS[spec.profile]),
        "smoke": bool(smoke),
        "save_bf16": bool(save_bf16),
    }


def compute_x3_binding_sha256(**kwargs: Any) -> str:
    return hashlib.sha256(json.dumps(build_x3_binding_doc(**kwargs), sort_keys=True).encode()).hexdigest()


def _sample_rob_batch_two_utterance(rob_bank: Wave8AGroundedRobotBank, rng: np.random.Generator, batch_size: int) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    idx = rng.integers(0, rob_bank.size, size=(batch_size,))
    ee = rob_bank.ee_xy[idx, :32]
    ee_vel = np.zeros_like(ee)
    ee_vel[:, 1:] = ee[:, 1:] - ee[:, :-1]
    moving = (np.linalg.norm(ee_vel, axis=-1) > 0.015).astype(np.float32)
    prop = np.concatenate(
        [ee, ee, ee_vel, np.zeros((batch_size, 32, 1), dtype=np.float32), moving[..., None], np.zeros((batch_size, 32, 8), dtype=np.float32)],
        axis=-1,
    )
    ln = np.clip(rob_bank.length[idx], 2, 32)
    end_xy = np.array([ee[bi, ln[bi] - 1] for bi in range(batch_size)], dtype=np.float32)
    disp_xy = np.clip((end_xy - ee[:, 0]) * 1.25, -0.95, 0.95).astype(np.float32)
    tgoal = np.broadcast_to(disp_xy[:, None, :], (batch_size, 32, 2)).copy()
    rb = {
        "video": rob_bank.video[idx, :32],
        "proprio": prop,
        "audio": rob_bank.spoken_audio[idx, :32],
        "wrong_audio": rob_bank.wrong_spoken_audio[idx, :32],
        "wrong_color_audio": rob_bank.wrong_color_spoken_audio[idx, :32],
        "target_goal_xy": tgoal,
        "target_block_xy": rob_bank.target_block_xy[idx],
        "distractor_block_xy": rob_bank.distractor_block_xy[idx],
    }
    toks = (rob_bank.true_txt16[idx], rob_bank.wrong_txt16[idx], rob_bank.wrong_color_txt16[idx])
    return rb, toks


POST_TRAIN_PARAMS_NAME = "post_train_params_f32.npz"
POST_TRAIN_META_NAME = "post_train.json"


def save_x3_post_train_checkpoint(
    npz_path: Path,
    meta_path: Path,
    params: dict[str, jax.Array],
    *,
    binding_sha: str,
    source_sha: str,
    steps: int,
    train_wall_s: float,
) -> str:
    """Atomically persist exact float32 post-train weights + binding metadata (local resume only; never archived)."""
    host_p = {k: np.asarray(v, dtype=np.float32) for k, v in jax.device_get(params).items()}
    tmp = npz_path.with_name(f".{npz_path.name}.tmp.npz")
    np.savez(tmp, **host_p)
    tmp.replace(npz_path)
    npz_sha = sha256_file(npz_path)
    meta = {
        "binding_sha256": binding_sha,
        "source_code_sha256": source_sha,
        "steps": int(steps),
        "train_wall_s": float(train_wall_s),
        "params_sha256": npz_sha,
    }
    tmp_m = meta_path.with_name(f".{meta_path.name}.tmp")
    tmp_m.write_text(json.dumps(meta, sort_keys=True) + "\n")
    tmp_m.replace(meta_path)
    return npz_sha


def load_x3_post_train_checkpoint_meta(
    meta_path: Path,
    npz_path: Path,
    log_path: Path,
    *,
    binding_sha: str,
    source_sha: str,
    steps: int,
) -> dict[str, Any] | None:
    """Return checkpoint metadata only if binding, source, steps, train log, and weight digest all verify."""
    if not (meta_path.is_file() and npz_path.is_file() and log_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if (
            meta.get("binding_sha256") != binding_sha
            or meta.get("source_code_sha256") != source_sha
            or int(meta.get("steps", -1)) != int(steps)
        ):
            return None
        last_step = -1
        for ln in log_path.read_text().splitlines():
            if ln.strip():
                last_step = int(json.loads(ln).get("step", -1))
    except (OSError, ValueError, TypeError):
        return None
    if last_step != int(steps):
        return None
    if sha256_file(npz_path) != meta.get("params_sha256"):
        return None
    return meta


def load_x3_post_train_params(npz_path: Path, ref_params: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Load float32 post-train weights, failing closed on any key/shape mismatch with the freshly built spec."""
    out: dict[str, jax.Array] = {}
    with np.load(npz_path, allow_pickle=False) as d:
        if set(d.files) != set(ref_params.keys()):
            raise RuntimeError(f"Post-train checkpoint key mismatch in {npz_path}")
        for k, ref in ref_params.items():
            arr = d[k]
            if tuple(arr.shape) != tuple(ref.shape):
                raise RuntimeError(f"Post-train checkpoint shape mismatch for {k}: {arr.shape} vs {ref.shape}")
            out[k] = jnp.asarray(arr, dtype=ref.dtype)
    return out


def train_and_evaluate_x3(
    *,
    tag: str,
    cohort: str,
    profile: str,
    arch: str,
    resolution_mode: str = "480p",
    seed: int = 3001,
    steps: int = 2200,
    batch_size: int = 8,
    seq_len: int = 24,
    out_dir: Path,
    train_480p_path: str | None = None,
    dev_480p_path: str | None = None,
    vault_480p_path: str | None = None,
    grounded_robot_path: str | None = None,
    vault_robot_path: str | None = None,
    vocab_path: str | None = None,
    save_bf16: bool = False,
    export_showcase: bool = True,
    smoke: bool = False,
) -> dict[str, Any]:
    """Train one Study X3 model, run full 6-way Any-to-Any + Livestream + Control evaluation, & write verified artifacts."""
    from evals.x3_eval_480p import run_x3_full_evaluation

    job_dir = out_dir / tag
    job_dir.mkdir(parents=True, exist_ok=True)
    spec = make_x3_spec(profile=profile, arch=arch, resolution_mode=resolution_mode)
    peak_lr = X3_PEAK_LRS[profile]
    state_mem = compute_x3_streaming_state_bytes(spec, batch_size=1)
    source_sha = compute_x3_source_code_sha256()
    binding_doc = build_x3_binding_doc(
        tag=tag, cohort=cohort, spec=spec, seed=seed, steps=steps, batch_size=batch_size, seq_len=seq_len, smoke=smoke, save_bf16=save_bf16
    )
    binding_sha = hashlib.sha256(json.dumps(binding_doc, sort_keys=True).encode()).hexdigest()
    role_shas = verify_x3_dataset_roles_fail_closed(
        train_480p_path, dev_480p_path, vault_480p_path, grounded_robot_path, vault_robot_path, smoke=smoke
    )

    cfg_path = job_dir / "config.json"
    cfg_path.write_text(json.dumps({**binding_doc, "source_code_sha256": source_sha}, indent=2, sort_keys=True) + "\n")

    params = init_x3_params(spec, seed=seed)
    pcounts = count_x3_parameters(params)
    ema_params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
    m_st = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), params)
    v_st = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=jnp.bfloat16), params)

    real_bank = X3Multimodal480pBank(train_480p_path) if (train_480p_path and Path(train_480p_path).is_file()) else None
    dev_val_bank = X3Multimodal480pBank(dev_480p_path) if (dev_480p_path and Path(dev_480p_path).is_file()) else None
    rob_bank = Wave8AGroundedRobotBank(grounded_robot_path, split="train") if (grounded_robot_path and Path(grounded_robot_path).is_file()) else None
    has_real = real_bank is not None
    real_n = (batch_size // 2) if has_real else 0
    sim_n = batch_size - real_n
    rob_n = min(1, batch_size) if rob_bank is not None else 0

    # Pre-materialize 2 deterministic held-out `dev` validation batches (`seed=309901`, `augment_photometric=False`)
    # so out-of-sample `val_loss` is tracked across `(N, D)` without training-set memorization confounds!
    val_rng = np.random.default_rng(309901)
    val_batches: list[dict[str, Any]] = []
    for vi in range(2):
        vk_sim = jax.random.fold_in(jax.random.key(309901), vi)
        v_sim = generate_x3_sim_batch_480p(vk_sim, sim_n if dev_val_bank is not None else batch_size, seq_len=seq_len)
        v_real = (
            dev_val_bank.sample_batch(val_rng, batch_size - sim_n, seq_len=seq_len, apply_composition_masking=True, augment_photometric=False)
            if dev_val_bank is not None
            else None
        )
        val_batches.append(mix_x3_real_and_sim_batches(v_real, None, v_sim, rob_tok16_tuple=None))

    print(
        f"[START X3] {tag} | profile={profile} arch={arch} res={resolution_mode} | "
        f"params={pcounts['strict_total']:,} | state_mib={state_mem['strict_system_state_mib']:.4f} | backend={jax.default_backend()}",
        flush=True,
    )

    @jax.jit
    def compiled_step(p, ep, m, v, b_in, step_i, rk):
        (loss, mets), grads = jax.value_and_grad(
            lambda w: world_model_loss_x3(w, ep, spec, b_in, rk), has_aux=True
        )(p)
        pn, mn, vn, gnorm, lr = memory_efficient_adamw_step(p, m, v, grads, step_i, steps, peak_lr)
        decay = jnp.float32(0.992)
        epn = jax.tree.map(lambda e, w: (decay * e.astype(jnp.float32) + (1.0 - decay) * w).astype(jnp.bfloat16), ep, pn)
        return pn, epn, mn, vn, {**mets, "lr": lr, "gnorm": gnorm}

    @jax.jit
    def compiled_val_loss(p, ep, b_in, rk):
        v_loss, v_mets = world_model_loss_x3(p, ep, spec, b_in, rk)
        return v_loss, v_mets

    np_rng = np.random.default_rng(seed + 17)
    base_key = jax.random.key(seed)
    t0 = time.monotonic()
    log_path = job_dir / "train.jsonl"
    prog_path = out_dir / f"{tag}-progress.json"
    post_npz = job_dir / POST_TRAIN_PARAMS_NAME
    post_meta = job_dir / POST_TRAIN_META_NAME
    resumed_meta = load_x3_post_train_checkpoint_meta(
        post_meta, post_npz, log_path, binding_sha=binding_sha, source_sha=source_sha, steps=steps
    )
    if resumed_meta is not None:
        params = load_x3_post_train_params(post_npz, params)
        train_wall_s = float(resumed_meta["train_wall_s"])
        print(f"[RESUME X3] {tag} | exact fp32 post-train checkpoint verified; skipping {steps} training steps -> eval.", flush=True)
    else:
        with log_path.open("w", encoding="utf-8") as lf:
            for step in range(steps):
                sk = jax.random.fold_in(base_key, step)
                k_sim, k_loss = jax.random.split(sk, 2)
                sim_b = generate_x3_sim_batch_480p(k_sim, sim_n, seq_len=seq_len)
                real_b = real_bank.sample_batch(np_rng, real_n, seq_len=seq_len, augment_photometric=True) if real_bank is not None else None
                if rob_bank is not None and rob_n > 0:
                    rob_b, rob_toks = _sample_rob_batch_two_utterance(rob_bank, np_rng, rob_n)
                else:
                    rob_b, rob_toks = None, None
                batch = mix_x3_real_and_sim_batches(real_b, rob_b, sim_b, rob_tok16_tuple=rob_toks)

                params, ema_params, m_st, v_st, mets = compiled_step(
                    params, ema_params, m_st, v_st, batch, jnp.int32(step), k_loss
                )
                if step == 0 or (step + 1) % 100 == 0 or (step + 1) == steps:
                    mh = {k: float(v) for k, v in jax.device_get(mets).items()}
                    if not np.isfinite(mh["loss"]):
                        raise FloatingPointError(f"Non-finite loss at step {step + 1} in {tag}: {mh}")
                    if step == 0 or (step + 1) % 400 == 0 or (step + 1) == steps:
                        vl_list, vv_list, vt_list, va_list = [], [], [], []
                        for vi_b, vb_eval in enumerate(val_batches):
                            _, vm_d = jax.device_get(
                                compiled_val_loss(params, ema_params, vb_eval, jax.random.fold_in(jax.random.key(309902), vi_b))
                            )
                            vl_list.append(float(vm_d["loss"]))
                            vv_list.append(float(vm_d["vid480_l1"]))
                            vt_list.append(float(vm_d["va2txt_ce"]))
                            va_list.append(float(vm_d["v2aud_mse"]))
                        mh["val_loss"] = float(np.mean(vl_list))
                        mh["val_vid480_l1"] = float(np.mean(vv_list))
                        mh["val_va2txt_ce"] = float(np.mean(vt_list))
                        mh["val_v2aud_mse"] = float(np.mean(va_list))
                    mh["step"] = step + 1
                    mh["elapsed_s"] = round(time.monotonic() - t0, 2)
                    lf.write(json.dumps(mh, sort_keys=True) + "\n")
                    lf.flush()
                    prog_path.write_text(json.dumps({"tag": tag, "step": step + 1, "steps": steps, "loss": mh["loss"]}, sort_keys=True) + "\n")
                    if step == 0 or (step + 1) % 200 == 0 or (step + 1) == steps:
                        v_str = f" val={mh['val_loss']:.4f}" if "val_loss" in mh else ""
                        print(
                            f"[{tag}] step {step + 1}/{steps} | loss={mh['loss']:.4f}{v_str} sym={mh['sym_ce']:.3f}(aux80={mh['sym_aux80']:.3f}) "
                            f"va2t={mh['va2txt_ce']:.3f} v480={mh['vid480_l1']:.3f} v2a={mh['v2aud_mse']:.3f} "
                            f"mov={mh['mov_mse']:.3f} act={mh['act_mse']:.3f} pr={mh['pr']:.3f}",
                            flush=True,
                        )

        train_wall_s = round(time.monotonic() - t0, 2)
        # Persist exact fp32 post-train weights BEFORE any export/eval so kernel death during eval
        # resumes straight into evaluation instead of silently re-running all training steps.
        save_x3_post_train_checkpoint(
            post_npz, post_meta, params, binding_sha=binding_sha, source_sha=source_sha, steps=steps, train_wall_s=train_wall_s
        )
    del m_st, v_st, ema_params, compiled_step, compiled_val_loss
    # Release training-only host banks before evaluation (eval re-opens dev/vault shards itself).
    del real_bank, dev_val_bank, rob_bank, val_batches
    gc.collect()

    int8_path = job_dir / "portable_int8.npz"
    int8_sha = save_portable_int8(int8_path, params)

    bf16_sha = None
    bf16_bytes = 0
    if save_bf16:
        bf16_p = job_dir / "weights_bf16.npz"
        bf16_sha = save_bf16_weights(bf16_p, params)
        bf16_bytes = int(bf16_p.stat().st_size)

    t_eval0 = time.monotonic()
    eval_res, showcase_sha, showcase_bytes = run_x3_full_evaluation(
        params,
        spec,
        dev_480p_path=dev_480p_path,
        vault_480p_path=vault_480p_path,
        grounded_robot_path=grounded_robot_path,
        vault_robot_path=vault_robot_path,
        vocab_path=vocab_path,
        showcase_out_npz=(job_dir / "showcase_bundle.npz") if export_showcase else None,
        smoke=smoke,
        progress_cb=lambda stage: prog_path.write_text(
            json.dumps({"tag": tag, "phase": "eval", "stage": stage, "steps": steps}, sort_keys=True) + "\n"
        ),
    )
    eval_wall_s = round(time.monotonic() - t_eval0, 2)
    eval_path = job_dir / "eval_metrics.json"
    eval_path.write_text(json.dumps(eval_res, indent=2, sort_keys=True) + "\n")

    aux_t_rec = 80 if seq_len >= 16 else 16
    tok_per_step = int((batch_size * seq_len + 2 * aux_t_rec) * spec.slots_per_tick)
    xarm6_tok_per_step = int((3 * (2 * rob_n) * 32) * spec.slots_per_tick) if rob_n > 0 else 0
    total_grad_tok_per_step = tok_per_step + xarm6_tok_per_step
    receipt = {
        "tag": tag,
        "cohort": cohort,
        "wave": "x3_1b_480p_scaling_composition_and_livestream",
        "profile": profile,
        "arch": arch,
        "resolution_mode": resolution_mode,
        "seed": int(seed),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "smoke": bool(smoke),
        "backend": jax.default_backend(),
        "parameter_counts": pcounts,
        "streaming_state_memory": state_mem,
        "source_code_sha256": source_sha,
        "binding_sha256": binding_sha,
        "config_file_sha256": sha256_file(cfg_path),
        "train_log_sha256": sha256_file(log_path),
        "eval_metrics_sha256": sha256_file(eval_path),
        "portable_int8_sha256": int8_sha,
        "portable_int8_bytes": int(int8_path.stat().st_size),
        "save_bf16": bool(save_bf16),
        "weights_bf16_sha256": bf16_sha,
        "weights_bf16_bytes": bf16_bytes,
        "showcase_bundle_sha256": showcase_sha,
        "showcase_bundle_bytes": showcase_bytes,
        "data_provenance": {
            **role_shas,
            "slot_tokens_per_step": tok_per_step,
            "xarm6_slot_tokens_per_step": xarm6_tok_per_step,
            "total_grad_slot_tokens_per_step": total_grad_tok_per_step,
            "total_slot_tokens": int(steps * tok_per_step),
            "total_grad_slot_tokens": int(steps * total_grad_tok_per_step),
        },
        "train_wall_s": train_wall_s,
        "resumed_from_post_train_checkpoint": bool(resumed_meta is not None),
        "eval_wall_s": eval_wall_s,
        "eval_summary": eval_res["summary"],
    }
    rec_path = job_dir / "receipt.json"
    rec_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (job_dir / "done.json").write_text(json.dumps({"tag": tag, "status": "ok"}, sort_keys=True) + "\n")
    print(f"[DONE X3] {tag} | summary={json.dumps(eval_res['summary'], sort_keys=True)}", flush=True)
    return receipt
