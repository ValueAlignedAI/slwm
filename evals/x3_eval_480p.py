"""Study X3 (`SLWM-1B-480p`) Comprehensive Leak-Free Evaluation Suite:
6-Way Any-to-Any Composition, `T=1,024` Causal Livestreaming, & Hardened `480p` Embodied Control.
"""

from __future__ import annotations

import functools
import gc
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from data.x1b_unseen_worlds import (
    LAYOUT_DUAL_CHICANE,
    LAYOUT_DYNAMICS_SHIFT,
    LAYOUT_MULTI_CLUTTER,
    LAYOUT_OFF_CENTER,
    step_unseen_layout_world,
)
from evals.x1b_eval_wave6 import _binary_auroc
from evals.x1b_eval_wave7 import (
    _proportional_push_action,
    compute_learned_value_grid_8x8,
    init_unseen_blocked_worlds,
    update_self_filtered_occ_8x8,
)
from evals.x1b_eval_wave8a import (
    _balanced_binary_ece_10bin,
    _propose_subgoal_candidates_shared,
    _score_subgoal_candidates_shared,
    _symmetric_two_block_grounding_score,
    forced_retention_sweep_action,
    init_shortcut_free_t160_switch_worlds,
)
from models.x1b_commit import route_proposals_with_commit_gate
from models.x1b_wave8a_core import encode_goal_query_tokens, encode_goal_tokens_2d
from models.x3_480p_core import (
    X3_ASSOC_KEY_OFFSET,
    X3_ASSOC_NUM_KEYS,
    X3_ASSOC_NUM_VALS,
    X3_ASSOC_VAL_OFFSET,
    X3_SLOTS_REGISTER,
    X3_SLOTS_VISUAL,
    X3Spec,
    compute_x3_streaming_state_bytes,
    decode_480p_video_from_latents,
    decode_x3_any_to_any_heads,
    decode_x3_target_block_xy,
    fork_x3_streams,
    forward_x3_world_model,
    init_x3_streaming_carry,
    mel64_to_waveform_griffin_lim_16khz,
    predict_x3_causal_action_from_ztask,
    predict_x3_latent_step,
)
from scripts.train.x1b_train import sha256_file
from scripts.train.x3_train_480p import (
    X3Multimodal480pBank,
    encode_wave8a_two_utterance_tokens_32x4,
    render_arena_480p_batch,
)


def _carry_nbytes(tree: Any) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    return int(sum(int(np.asarray(x).nbytes) for x in leaves))


def _load_id_to_token_map(vocab_path: str | Path | None) -> dict[int, str]:
    inv: dict[int, str] = {}
    if vocab_path and Path(vocab_path).is_file():
        raw = json.loads(Path(vocab_path).read_text())
        for tok, tid in raw.items():
            inv[int(tid)] = str(tok)
    return inv


def decode_pred_tokens_to_text(token_ids_1d: np.ndarray, inv_vocab: dict[int, str]) -> str:
    pieces: list[str] = []
    byte_buf = bytearray()

    def _flush_bytes() -> None:
        if byte_buf:
            pieces.append(bytes(byte_buf).decode("utf-8", errors="replace"))
            byte_buf.clear()

    for raw_t in np.asarray(token_ids_1d, dtype=np.int32).tolist():
        tid = int(raw_t)
        if tid <= 0 or tid == 3:
            _flush_bytes()
            if tid == 3 and pieces:
                break
            continue
        if tid in inv_vocab:
            _flush_bytes()
            pieces.append(" " + inv_vocab[tid])
        elif 256 <= tid < 512:
            byte_buf.append(tid - 256)
    _flush_bytes()
    txt = "".join(pieces).strip()
    return " ".join(txt.split())[:160]


def _disjoint_group_perm(group_ids: np.ndarray, *, same_source_ids: np.ndarray | None = None) -> np.ndarray:
    """Deterministic permutation guaranteeing `group_ids[perm[i]] != group_ids[i]` (and within-source if >=2 groups)."""
    gids = np.asarray(group_ids, dtype=np.int64)
    n = int(gids.shape[0])
    perm = np.arange(n, dtype=np.int64)

    def _derange_subset(sub_idx: np.ndarray) -> bool:
        sub_g = gids[sub_idx]
        uniq_g = sorted(set(int(x) for x in sub_g.tolist()))
        if len(uniq_g) < 2:
            return False
        buckets = [sub_idx[sub_g == ug] for ug in uniq_g]
        k_b = len(buckets)
        for bi in range(k_b):
            donor = buckets[(bi + 1) % k_b]
            rec = buckets[bi]
            for ri_pos, r_idx in enumerate(rec.tolist()):
                perm[r_idx] = int(donor[ri_pos % len(donor)])
        return True

    if same_source_ids is not None:
        sids = np.asarray(same_source_ids, dtype=np.int32)
        leftover: list[int] = []
        for sid in sorted(set(int(x) for x in sids.tolist())):
            sub = np.where(sids == sid)[0]
            if not _derange_subset(sub):
                leftover.extend(int(x) for x in sub.tolist())
        if leftover:
            # Fallback across full bank for any singleton-group source in tiny smoke fixtures
            all_u = sorted(set(int(x) for x in gids.tolist()))
            if len(all_u) >= 2:
                for idx_l in leftover:
                    cand = np.where(gids != gids[idx_l])[0]
                    if cand.size > 0:
                        perm[idx_l] = int(cand[idx_l % cand.size])
        return perm

    if not _derange_subset(np.arange(n, dtype=np.int64)):
        perm = np.roll(np.arange(n, dtype=np.int64), shift=max(1, n // 2))
    return perm


def _psnr_and_ssim_480p(pred_u8: np.ndarray, targ_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pf = pred_u8.astype(np.float32) / 255.0
    tf = targ_u8.astype(np.float32) / 255.0
    mse = np.mean((pf - tf) ** 2, axis=(1, 2, 3, 4))
    psnr = 10.0 * np.log10(1.0 / np.maximum(mse, 1e-8))

    b, t, h, w, c = pf.shape
    p_blk = pf.reshape(b, t, h // 8, 8, w // 8, 8, c)
    t_blk = tf.reshape(b, t, h // 8, 8, w // 8, 8, c)
    mu_p = np.mean(p_blk, axis=(3, 5))
    mu_t = np.mean(t_blk, axis=(3, 5))
    var_p = np.mean((p_blk - mu_p[:, :, :, None, :, None, :]) ** 2, axis=(3, 5))
    var_t = np.mean((t_blk - mu_t[:, :, :, None, :, None, :]) ** 2, axis=(3, 5))
    cov_pt = np.mean((p_blk - mu_p[:, :, :, None, :, None, :]) * (t_blk - mu_t[:, :, :, None, :, None, :]), axis=(3, 5))
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2.0 * mu_p * mu_t + c1) * (2.0 * cov_pt + c2)) / ((mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2))
    ssim = np.mean(ssim_map, axis=(1, 2, 3, 4))

    gx_p = pf[:, :, :, 1:, :] - pf[:, :, :, :-1, :]
    gx_t = tf[:, :, :, 1:, :] - tf[:, :, :, :-1, :]
    gy_p = pf[:, :, 1:, :, :] - pf[:, :, :-1, :, :]
    gy_t = tf[:, :, 1:, :, :] - tf[:, :, :-1, :, :]
    edge_rmse = np.sqrt(
        0.5 * (np.mean((gx_p - gx_t) ** 2, axis=(1, 2, 3, 4)) + np.mean((gy_p - gy_t) ** 2, axis=(1, 2, 3, 4))) + 1e-10
    )
    return psnr, ssim, edge_rmse


_METRIC_CHUNK = 16


def _pred480_to_u8(v: np.ndarray) -> np.ndarray:
    """Exact elementwise `[-1, 1] -> uint8` quantization used for all 480p video metrics."""
    return np.clip((v + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)


def _chunked_psnr_ssim_480p(pred_u8: np.ndarray, targ_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-episode `_psnr_and_ssim_480p` evaluated in episode chunks (bounded float32 temporaries)."""
    parts = [
        _psnr_and_ssim_480p(pred_u8[c0 : c0 + _METRIC_CHUNK], targ_u8[c0 : c0 + _METRIC_CHUNK])
        for c0 in range(0, pred_u8.shape[0], _METRIC_CHUNK)
    ]
    if not parts:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z
    return tuple(np.concatenate([pt[i] for pt in parts], axis=0) for i in range(3))  # type: ignore[return-value]


def _host_rss_gb() -> tuple[float, float]:
    """(current, peak) host RSS in GB; current read from /proc on Linux, else peak only."""
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9
    cur_gb = peak_gb
    try:
        with open("/proc/self/statm") as f:
            import os

            cur_gb = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except OSError:
        pass
    return cur_gb, peak_gb


def _build_synthetic_480p_eval_clips(seed: int = 3009, n_clips: int = 12, seq_len: int = 24) -> dict[str, np.ndarray]:
    """Self-contained `480p` multimodal probe bank used ONLY in `smoke=True` unit tests."""
    rng = np.random.default_rng(seed)
    st = init_unseen_blocked_worlds(jax.random.key(seed), batch_size=n_clips)
    ee_l, ob_l, act_l = [], [], []
    cur = st
    for _ in range(seq_len):
        u2 = _proportional_push_action(cur["ee_xy"], cur["obj_xy"], st["targ_a"], final_goal_xy=st["targ_a"])[:, :2]
        ee_l.append(cur["ee_xy"])
        ob_l.append(cur["obj_xy"])
        act_l.append(u2)
        cur, _ = step_unseen_layout_world(cur, u2)
    ee_bt = jnp.stack(ee_l, axis=1)
    ob_bt = jnp.stack(ob_l, axis=1)
    u2_bt = jnp.stack(act_l, axis=1)
    g_bt = jnp.full((n_clips, seq_len, 2), 9.0, dtype=jnp.float32)
    vid = np.asarray(
        render_arena_480p_batch(ee_bt, ob_bt, g_bt, st["box_cx"], st["box_cy"], st["box_hx"], st["box_hy"]),
        dtype=np.uint8,
    )
    t_grid = np.linspace(0.0, 2.4, seq_len, dtype=np.float32)[None, :, None]
    freq_mod = np.asarray(ee_bt[:, :, 0:1], dtype=np.float32)
    bands = np.linspace(0.5, 3.0, 64, dtype=np.float32)[None, None, :]
    aud = (0.65 * np.sin(2.0 * np.pi * (1.2 + freq_mod) * t_grid * bands) + 0.05 * rng.standard_normal((n_clips, seq_len, 64))).astype(np.float32)
    mov = np.zeros((n_clips, seq_len, 32), dtype=np.float32)
    mov[:, :, 0:2] = np.asarray(ee_bt)
    mov[:, :, 2:4] = np.asarray(ob_bt)
    mov[:, 1:, 4:6] = mov[:, 1:, 0:2] - mov[:, :-1, 0:2]
    t_act = np.zeros((n_clips, seq_len, 16), dtype=np.float32)
    t_act[:, :, 0:2] = np.asarray(u2_bt)
    p_act = np.concatenate([np.zeros((n_clips, 1, 16), dtype=np.float32), t_act[:, :-1]], axis=1)
    lang = np.zeros((n_clips, seq_len, 4), dtype=np.int32)
    tlang = np.zeros((n_clips, seq_len), dtype=np.int32)
    for bi in range(n_clips):
        base_tok = 300 + (bi % 8) * 5
        for ti in range(min(8, seq_len)):
            lang[bi, ti, 0] = base_tok + (ti % 4)
            tlang[bi, ti] = base_tok + (ti % 4)
        tlang[bi, min(8, seq_len - 1)] = 3
    return {
        "video": vid,
        "audio": aud,
        "movement": mov,
        "action": p_act,
        "target_action": t_act,
        "has_true_action": np.ones((n_clips,), dtype=np.float32),
        "lang_tokens": lang,
        "target_lang_tokens": tlang,
        "modality_mask": np.ones((n_clips, seq_len, 4), dtype=np.float32),
        "embodiment_id": np.zeros((n_clips,), dtype=np.int32),
        "source_id": (np.arange(n_clips, dtype=np.int32) % 2),
        "group_id": np.arange(100, 100 + n_clips, dtype=np.int64),
    }


def _load_eval_480p_clips(
    npz_path: str | Path | None,
    *,
    max_clips: int | None,
    seed: int,
    smoke: bool,
) -> dict[str, np.ndarray]:
    if not npz_path or not Path(npz_path).is_file():
        if not smoke:
            raise FileNotFoundError(f"Non-smoke Study X3 evaluation requires verified 480p shard: {npz_path!r}")
        return _build_synthetic_480p_eval_clips(seed=seed, n_clips=(max_clips or 12))
    bank = X3Multimodal480pBank(npz_path)
    if max_clips is None or bank.size <= max_clips:
        idx_arr = np.arange(bank.size, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        src_emb_all = bank.source_id.astype(np.int32) * 16 + bank.embodiment_id.astype(np.int32)
        unique_keys = sorted(set(int(x) for x in src_emb_all.tolist()))
        per_key = max(2, (max_clips + max(1, len(unique_keys)) - 1) // max(1, len(unique_keys)))
        by_k: dict[int, list[int]] = {}
        for sk in unique_keys:
            idxs = np.where(src_emb_all == sk)[0]
            if idxs.size == 0:
                continue
            # Always prioritize distinct recording groups first so every `(source, embodiment)` retains >= 2 groups!
            grp_reps: list[int] = []
            rem_pool: list[int] = []
            for ug in sorted(set(int(x) for x in bank.group_id[idxs].tolist())):
                g_idxs = idxs[bank.group_id[idxs] == ug]
                p_rep = int(rng.choice(g_idxs))
                grp_reps.append(p_rep)
                rem_pool.extend(int(x) for x in g_idxs.tolist() if int(x) != p_rep)
            rng.shuffle(grp_reps)
            rng.shuffle(rem_pool)
            ordered = grp_reps + rem_pool
            by_k[sk] = ordered[: min(len(ordered), max(2, per_key))]
        # Keep at least 2 distinct-group representatives per `(source, embodiment)` when trimming to `max_clips`
        mandatory: list[int] = []
        extras_by_k: dict[int, list[int]] = {}
        for sk in unique_keys:
            items = by_k.get(sk, [])
            mandatory.extend(items[:2])
            extras_by_k[sk] = items[2:]
        chosen = list(mandatory)
        while len(chosen) < max_clips and any(extras_by_k.values()):
            for sk in unique_keys:
                if extras_by_k[sk] and len(chosen) < max_clips:
                    chosen.append(extras_by_k[sk].pop(0))
        idx_arr = np.asarray(sorted(chosen), dtype=np.int64)

    vids = np.empty((idx_arr.size, 24, 480, 640, 3), dtype=np.uint8)
    for vi, ci in enumerate(idx_arr):
        vids[vi] = bank.decode_clip_frames(int(ci), 24)
    return {
        "video": vids,
        "audio": bank.audio[idx_arr, :24],
        "movement": bank.movement[idx_arr, :24],
        "action": bank.action[idx_arr, :24],
        "target_action": bank.target_action[idx_arr, :24],
        "has_true_action": bank.has_true_action[idx_arr],
        "lang_tokens": bank.lang_tokens[idx_arr, :24],
        "target_lang_tokens": bank.target_lang_tokens[idx_arr, :24],
        "modality_mask": bank.modality_mask[idx_arr, :24],
        "embodiment_id": bank.embodiment_id[idx_arr],
        "source_id": bank.source_id[idx_arr],
        "group_id": bank.group_id[idx_arr],
    }


def apply_unseen_480p_sensor_corruption(clean_vid_480p: jax.Array, kind: str) -> jax.Array:
    b, t, h, w, c = clean_vid_480p.shape
    is_corr = (jnp.arange(t)[None, :] >= 10)[:, :, None, None, None]
    if kind == "frozen_camera_frame":
        frozen = jnp.broadcast_to(clean_vid_480p[:, 9:10], clean_vid_480p.shape)
        return jnp.where(is_corr, frozen, clean_vid_480p)
    if kind == "cross_episode_replay_swap":
        swapped = jnp.roll(clean_vid_480p, shift=max(1, b // 2), axis=0)
        return jnp.where(is_corr, swapped, clean_vid_480p)
    if kind == "unseen_90deg_rotation":
        rot_hw = jnp.rot90(clean_vid_480p, k=1, axes=(2, 3))
        rot_480 = jax.image.resize(rot_hw.reshape(b * t, w, h, c).astype(jnp.float32), (b * t, h, w, c), method="nearest")
        rot_u8 = jnp.clip(rot_480, 0.0, 255.0).astype(jnp.uint8).reshape(b, t, h, w, c)
        return jnp.where(is_corr, rot_u8, clean_vid_480p)
    if kind == "half_occluded_camera":
        occ = clean_vid_480p.at[:, :, :, : w // 2, :].set(jnp.uint8(12))
        return jnp.where(is_corr, occ, clean_vid_480p)
    raise ValueError(f"Unknown 480p sensor corruption: {kind!r}")


def eval_x3_unseen_ood_480p(
    params: dict[str, jax.Array],
    spec: X3Spec,
    *,
    seed: int = 302404,
    n_worlds: int = 24,
    chunk_bs: int = 4,
) -> dict[str, Any]:
    """Zero-shot `480p` camera corruption AUROC + `PolicyCommitGate` selective routing (`0` future action leak)."""
    st = init_unseen_blocked_worlds(jax.random.key(seed), batch_size=n_worlds)
    st = {**st, "box_hx": jnp.zeros_like(st["box_hx"]), "box_hy": jnp.zeros_like(st["box_hy"])}
    ee_l, ob_l, act_l = [], [], []
    cur = st
    phase = jnp.linspace(0.0, 2.0 * np.pi, n_worlds, endpoint=False, dtype=jnp.float32)
    for ti in range(24):
        sw = jnp.stack([0.64 * jnp.cos(phase + 0.35 * ti), -0.22 - 0.45 * jnp.abs(jnp.sin(phase + 0.28 * ti))], axis=-1)
        u2 = jnp.clip(3.2 * (sw - cur["ee_xy"]), -1.0, 1.0)
        ee_l.append(cur["ee_xy"])
        ob_l.append(cur["obj_xy"])
        act_l.append(u2)
        cur, _ = step_unseen_layout_world(cur, u2)

    ee_bt = jnp.stack(ee_l, axis=1)
    ob_bt = jnp.stack(ob_l, axis=1)
    u2_bt = jnp.stack(act_l, axis=1)
    g_bt = jnp.where(
        (jnp.arange(24)[None, :, None] < 3),
        jnp.broadcast_to(st["targ_a"][:, None, :], (n_worlds, 24, 2)),
        9.0,
    )
    clean_vid = render_arena_480p_batch(ee_bt, ob_bt, g_bt, st["box_cx"], st["box_cy"], st["box_hx"], st["box_hy"])
    ee_vel = jnp.concatenate([jnp.zeros((n_worlds, 1, 2)), ee_bt[:, 1:] - ee_bt[:, :-1]], axis=1)
    mov_seq = jnp.concatenate([ee_bt, ee_bt, ee_vel, jnp.zeros((n_worlds, 24, 26))], axis=-1)
    u16_seq = jnp.concatenate([u2_bt, jnp.zeros((n_worlds, 24, 14))], axis=-1)
    prev_u16 = jnp.concatenate([jnp.zeros((n_worlds, 1, 16)), u16_seq[:, :-1]], axis=1)

    @jax.jit
    def _eval_sub(p: dict[str, jax.Array], v_sub: jax.Array, m_sub: jax.Array, pa_sub: jax.Array):
        bs = v_sub.shape[0]
        b_in = {
            "video": v_sub,
            "audio": jnp.zeros((bs, 24, 64), dtype=jnp.float32),
            "movement": m_sub,
            "action": pa_sub,
            "sym_tokens": jnp.zeros((bs, 24, 4), dtype=jnp.int32),
            "lang_tokens": jnp.zeros((bs, 24, 4), dtype=jnp.int32),
            "modality_mask": jnp.ones((bs, 24, 4), dtype=jnp.float32).at[:, :, 0].set(0.0).at[:, :, 3].set(0.0),
            "embodiment_id": jnp.zeros((bs,), dtype=jnp.int32),
            "valid": jnp.ones((bs, 24), dtype=bool),
        }
        z_top, f300, _, _ = forward_x3_world_model(p, spec, b_in)
        z_sym, z_task = fork_x3_streams(z_top)
        # Strict causal action: use `predict_x3_causal_action_from_ztask` (`0` ground-truth `u16_seq` leak!)
        u_pred = jax.lax.stop_gradient(predict_x3_causal_action_from_ztask(p, z_task, b_in))
        zn1, lv1 = predict_x3_latent_step(p, spec, z_task, u_pred)
        hds = decode_x3_any_to_any_heads(p, spec, z_sym, z_task, zn1, lv1, f300, b_in, decode_480p_frames=False)
        act4 = hds["pred_action"][..., :4]
        u_trust = jnp.zeros_like(act4)
        routed = route_proposals_with_commit_gate(
            hds["commit_logits"], act4, u_trust, gate_policy="learned_innovation"
        )
        p_nc = 1.0 - routed["mode_probs"][:, :, 0]
        raw_s = hds["anomaly_score"]
        pad = jnp.pad(raw_s, ((0, 0), (3, 0)), mode="edge")
        cmax = jnp.maximum(jnp.maximum(pad[:, 3:], pad[:, 2:-1]), jnp.maximum(pad[:, 1:-2], pad[:, :-3]))
        return 0.65 * raw_s + 0.35 * cmax, routed["mode_id"], p_nc

    def _run_all(vid_all: jax.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sc_l, md_l, pn_l = [], [], []
        cbs = min(chunk_bs, n_worlds)
        for i0 in range(0, n_worlds, cbs):
            s, m, pn = jax.device_get(
                _eval_sub(params, vid_all[i0 : i0 + cbs], mov_seq[i0 : i0 + cbs], prev_u16[i0 : i0 + cbs])
            )
            sc_l.append(np.asarray(s))
            md_l.append(np.asarray(m))
            pn_l.append(np.asarray(pn))
        return np.concatenate(sc_l, 0)[:, 10:], np.concatenate(md_l, 0)[:, 10:], np.concatenate(pn_l, 0)[:, 10:]

    clean_sc, clean_md, clean_pnc = _run_all(clean_vid)
    clean_commit_vec = np.mean(clean_md == 0, axis=1)
    aurocs, ep_auc_list, nc_vecs, abs_rates, fb_rates, corr_pncs, corr_sc_vecs, corr_sc_ticks = [], [], [], [], [], [], [], []
    by_corr: dict[str, Any] = {}
    for kind in ("frozen_camera_frame", "cross_episode_replay_swap", "unseen_90deg_rotation", "half_occluded_camera"):
        c_vid = apply_unseen_480p_sensor_corruption(clean_vid, kind)
        csc, cmd, cpnc = _run_all(c_vid)
        auc = _binary_auroc(csc, clean_sc)
        aurocs.append(auc)
        ep_auc_list.append([_binary_auroc(csc[i], clean_sc) for i in range(n_worlds)])
        corr_sc_vecs.append(np.mean(csc, axis=1))
        corr_sc_ticks.append(csc)
        nc_v = np.mean(cmd != 0, axis=1)
        nc_vecs.append(nc_v)
        abs_r = float(np.mean(cmd == 3))
        fb_r = float(np.mean((cmd == 1) | (cmd == 2)))
        abs_rates.append(abs_r)
        fb_rates.append(fb_r)
        corr_pncs.append(cpnc)
        by_corr[kind] = {
            "auroc": float(auc),
            "non_commit_rate": float(np.mean(nc_v)),
            "abstain_rate": abs_r,
            "fallback_rate": fb_r,
        }

    mean_nc_vec = np.mean(np.stack(nc_vecs, axis=0), axis=0)
    sel_vec = 0.5 * clean_commit_vec + 0.5 * mean_nc_vec
    ep_auroc_vec = np.mean(np.asarray(ep_auc_list, dtype=np.float64), axis=0)
    clean_w_vec = np.mean(clean_sc, axis=1)
    corr_w_vec = np.mean(np.stack(corr_sc_vecs, axis=0), axis=0)
    return {
        "mean_unseen_ood_corruption_auroc": float(np.mean(aurocs)),
        "ood_ep_auroc_vec": [float(x) for x in ep_auroc_vec],
        "ood_clean_world_scores_vec": [float(x) for x in clean_w_vec],
        "ood_corrupt_world_scores_vec": [float(x) for x in corr_w_vec],
        "ood_clean_tick_scores": [[float(v) for v in row] for row in clean_sc],
        "ood_corrupt_tick_scores": [[[float(v) for v in row] for row in mat] for mat in corr_sc_ticks],
        "clean_commit_full_rate": float(np.mean(clean_commit_vec)),
        "clean_abstain_rate": float(np.mean(clean_md == 3)),
        "ood_non_commit_rate": float(np.mean(mean_nc_vec)),
        "ood_abstain_rate": float(np.mean(abs_rates)),
        "ood_fallback_mitigated_rate": float(np.mean(fb_rates)),
        "commit_selective_routing_acc": float(np.mean(sel_vec)),
        "commit_selective_routing_vec": [float(x) for x in sel_vec],
        "commit_binary_ece": _balanced_binary_ece_10bin(clean_pnc, np.stack(corr_pncs, axis=0)),
        "by_corruption": by_corr,
    }


def eval_x3_any_to_any_composition(
    params: dict[str, jax.Array],
    spec: X3Spec,
    clips: dict[str, np.ndarray],
    *,
    inv_vocab: dict[int, str],
    ood_panel: dict[str, Any] | None = None,
    chunk_bs: int = 4,
    collect_showcase: bool = False,
    strict_within_source: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Leak-free 6-Way Any-to-Any Composition + Within-Source Disjoint-Group Counterfactuals.

    Zero ground-truth `target_action` ever enters `_forward_masked`, and ALL 6 directions permute
    strictly within `(source_id, embodiment_id)` (`0` cross-source missing-modality `mm=0` swaps):
      - When `use_obs_action=True` (Directions 2 & 3 only), `SLOT_ACTION` and `predict_x3_latent_step`
        receive causal `prev_action = clips["action"] * has_true_action`.
      - When `use_obs_action=False` (Directions 1, 4, 5, & 6), `SLOT_ACTION` is strictly zeroed (`0.0`)
        and `predict_x3_latent_step` uses `stop_grad(predict_x3_causal_action_from_ztask(params, z_task, b_in))`.
    """
    n = int(clips["video"].shape[0])
    t = int(clips["video"].shape[1])
    mm_full = clips["modality_mask"].astype(np.float32)
    gids = clips["group_id"]
    sids = clips["source_id"]
    embs = clips["embodiment_id"]
    has_true_act = clips["has_true_action"].astype(np.float32)
    causal_prev_act = clips["action"].astype(np.float32) * has_true_act[:, None, None]

    src_emb_key = sids.astype(np.int32) * 16 + embs.astype(np.int32)
    perm_within_src = _disjoint_group_perm(gids, same_source_ids=src_emb_key)
    if strict_within_source:
        if not np.all(src_emb_key[perm_within_src] == src_emb_key):
            raise RuntimeError("Strict within-(source, embodiment) permutation failed: singleton group detected!")
        if not np.all(gids[perm_within_src] != gids):
            raise RuntimeError("Strict disjoint-group permutation failed: self-group swap detected!")

    @functools.partial(jax.jit, static_argnums=(9,))
    def _forward_masked(
        p: dict[str, jax.Array],
        vid: jax.Array,
        aud: jax.Array,
        mov: jax.Array,
        act_in: jax.Array,
        lang: jax.Array,
        mm_in: jax.Array,
        emb: jax.Array,
        ctx_f0: jax.Array,
        use_obs_action: bool,
    ):
        bs = vid.shape[0]
        act_san = act_in if use_obs_action else jnp.zeros((bs, t, 16), dtype=jnp.float32)
        b_in = {
            "video": jnp.where(mm_in[:, :, 1:2, None, None] > 0.5, vid, 0),
            "audio": jnp.where(mm_in[:, :, 0:1] > 0.5, aud, 0.0),
            "movement": jnp.where(mm_in[:, :, 2:3] > 0.5, mov, 0.0),
            "action": act_san,
            "sym_tokens": jnp.zeros((bs, t, 4), dtype=jnp.int32),
            "lang_tokens": jnp.where(mm_in[:, :, 3:4] > 0.5, lang, 0),
            "modality_mask": mm_in,
            "embodiment_id": emb,
            "valid": jnp.ones((bs, t), dtype=bool),
        }
        z_top, f300, _, _ = forward_x3_world_model(p, spec, b_in)
        z_sym, z_task = fork_x3_streams(z_top)
        u_pred = jax.lax.stop_gradient(predict_x3_causal_action_from_ztask(p, z_task, b_in))
        has_a = jnp.sum(jnp.abs(act_san), axis=-1, keepdims=True) > 1e-5
        u_lat = jnp.where(has_a, act_san, u_pred) if use_obs_action else u_pred
        zn1, lv1 = predict_x3_latent_step(p, spec, z_task, u_lat)
        hds = decode_x3_any_to_any_heads(
            p, spec, z_sym, z_task, zn1, lv1, f300, b_in, decode_480p_frames=False
        )
        v_idx = jnp.array([0, min(5, t - 2), min(13, t - 2), max(0, t - 2)], dtype=jnp.int32)
        reg_sub = jnp.mean(z_task[:, v_idx, X3_SLOTS_REGISTER, :].astype(jnp.float32), axis=2)
        pred_v480 = decode_480p_video_from_latents(
            p, spec, zn1[:, v_idx, X3_SLOTS_VISUAL, :], reg_sub, context_frame_480p=ctx_f0
        )
        return {
            "pred_lang_logits": hds["pred_lang_logits"],
            "pred_v480": pred_v480,
            "pred_audio_mel": hds["pred_audio_mel"],
            "pred_mov_next": hds["pred_mov_next"],
            "pred_mov_step4": hds["pred_mov_step4"],
            "pred_action": hds["pred_action"],
            "contact_logits": hds["contact_logits"],
        }

    def _batched_call(
        vid_arr, aud_arr, mov_arr, act_arr, lang_arr, mm_arr, ctx0_arr, *,
        use_obs_action: bool, keep: tuple[str, ...], vid_idx: np.ndarray | None = None,
    ):
        """Chunked forward that retains ONLY the consumed head outputs (host-RAM bounded).

        `pred_v480` is quantized to uint8 per chunk with the exact same elementwise expression
        previously applied after concatenation, and `vid_idx` gathers permuted video per chunk
        instead of materializing a full permuted copy of the `[N, T, 480, 640, 3]` clip tensor.
        """
        outs: dict[str, list[np.ndarray]] = {}
        cbs = min(chunk_bs, n)
        for i0 in range(0, n, cbs):
            sl = slice(i0, i0 + cbs)
            vid_chunk = vid_arr[vid_idx[sl]] if vid_idx is not None else vid_arr[sl]
            res = jax.device_get(
                _forward_masked(
                    params,
                    jnp.asarray(vid_chunk),
                    jnp.asarray(aud_arr[sl]),
                    jnp.asarray(mov_arr[sl]),
                    jnp.asarray(act_arr[sl]),
                    jnp.asarray(lang_arr[sl]),
                    jnp.asarray(mm_arr[sl]),
                    jnp.asarray(clips["embodiment_id"][sl]),
                    jnp.asarray(ctx0_arr[sl]),
                    use_obs_action,
                )
            )
            for k in keep:
                v = np.asarray(res[k])
                if k == "pred_v480":
                    v = _pred480_to_u8(v)
                outs.setdefault(k, []).append(v)
            del res
        return {k: np.concatenate(v, axis=0) for k, v in outs.items()}

    ctx_zero = np.zeros((n, 1, 480, 640, 3), dtype=np.uint8)
    ctx_true0 = clips["video"][:, :1]
    z_act_np = np.zeros_like(causal_prev_act)

    # --- Direction 1: `Video / Audio -> Text` (`mm[txt] = 0`, `action = 0`, within-source disjoint-group shuffle!) ---
    mm_d1 = mm_full.copy()
    mm_d1[:, :, 3] = 0.0
    k1 = ("pred_lang_logits",)
    d1_true = _batched_call(
        clips["video"], clips["audio"], clips["movement"], z_act_np, clips["lang_tokens"], mm_d1, ctx_zero, use_obs_action=False, keep=k1
    )
    d1_shuf = _batched_call(
        clips["video"], clips["audio"][perm_within_src], clips["movement"][perm_within_src],
        z_act_np, clips["lang_tokens"], mm_d1[perm_within_src], ctx_zero, use_obs_action=False, keep=k1, vid_idx=perm_within_src,
    )
    tlang = clips["target_lang_tokens"]
    has_txt_ep = (mm_full[:, 0, 3] > 0.5) & (np.sum(tlang > 0, axis=1) > 0)
    if not np.any(has_txt_ep):
        has_txt_ep = np.ones((n,), dtype=bool)
    d1_ce_true, d1_ce_shuf, d1_top1_true, d1_top1_shuf, d1_top5_true, d1_f1 = [], [], [], [], [], []
    decoded_txts, gt_txts = [], []
    d1_gids: list[int] = []
    for i in np.where(has_txt_ep)[0]:
        valid_tok = tlang[i] > 0
        logits_t = d1_true["pred_lang_logits"][i, valid_tok]
        logits_s = d1_shuf["pred_lang_logits"][i, valid_tok]
        yt = np.clip(tlang[i, valid_tok], 0, spec.vocab_size - 1)
        lse_t = np.log(np.sum(np.exp(logits_t - np.max(logits_t, axis=-1, keepdims=True)), axis=-1)) + np.max(logits_t, axis=-1)
        lse_s = np.log(np.sum(np.exp(logits_s - np.max(logits_s, axis=-1, keepdims=True)), axis=-1)) + np.max(logits_s, axis=-1)
        ce_t = float(np.mean(lse_t - logits_t[np.arange(yt.size), yt]))
        ce_s = float(np.mean(lse_s - logits_s[np.arange(yt.size), yt]))
        pred1_t = np.argmax(logits_t, axis=-1)
        pred1_s = np.argmax(logits_s, axis=-1)
        top5_t = np.argsort(logits_t, axis=-1)[:, -5:]
        d1_ce_true.append(ce_t)
        d1_ce_shuf.append(ce_s)
        d1_top1_true.append(float(np.mean(pred1_t == yt)))
        d1_top1_shuf.append(float(np.mean(pred1_s == yt)))
        d1_top5_true.append(float(np.mean(np.any(top5_t == yt[:, None], axis=-1))))
        d1_gids.append(int(gids[i]))
        pset = set(int(x) for x in pred1_t.tolist() if x > 3)
        gset = set(int(x) for x in yt.tolist() if x > 3)
        inter = len(pset & gset)
        d1_f1.append(float((2.0 * inter / max(1, len(pset) + len(gset))) if (pset or gset) else 0.0))
        if len(decoded_txts) < 8:
            decoded_txts.append(decode_pred_tokens_to_text(np.argmax(d1_true["pred_lang_logits"][i], axis=-1), inv_vocab))
            gt_txts.append(decode_pred_tokens_to_text(tlang[i], inv_vocab))

    d1_lift_vec = np.asarray(d1_ce_shuf) - np.asarray(d1_ce_true)
    dir1_metrics = {
        "n_episodes": int(np.sum(has_txt_ep)),
        "nll_ce": float(np.mean(d1_ce_true)),
        "within_source_shuffled_nll_ce": float(np.mean(d1_ce_shuf)),
        "shuffle_lift_ce": float(np.mean(d1_lift_vec)),
        "shuffle_lift_vec": [float(x) for x in d1_lift_vec],
        "group_ids": d1_gids,
        "top1_token_acc": float(np.mean(d1_top1_true)),
        "within_source_shuffled_top1_acc": float(np.mean(d1_top1_shuf)),
        "shuffle_lift_top1_pp": float((np.mean(d1_top1_true) - np.mean(d1_top1_shuf)) * 100.0),
        "top5_token_acc": float(np.mean(d1_top5_true)),
        "token_set_f1": float(np.mean(d1_f1)),
        "sample_predictions": [{"gt": g, "pred": p} for g, p in zip(gt_txts[:5], decoded_txts[:5])],
    }

    # --- Direction 2: `Text (+ Context Frame 0 + Action) -> 480p Video` (FIXED `ctx_true0`, permute ONLY `(lang, action)`!) ---
    mm_d2 = mm_full.copy()
    mm_d2[:, :, 0] = 0.0
    mm_d2[:, 1:, 1] = 0.0
    mm_d2[:, 1:, 2] = 0.0
    k2 = ("pred_v480",)
    d2_true = _batched_call(
        clips["video"], clips["audio"], clips["movement"], causal_prev_act, clips["lang_tokens"], mm_d2, ctx_true0, use_obs_action=True, keep=k2
    )
    mm_d2_shuf = mm_d2.copy()
    mm_d2_shuf[:, :, 3] = mm_full[perm_within_src, :, 3]
    d2_shuf = _batched_call(
        clips["video"], clips["audio"], clips["movement"],
        causal_prev_act[perm_within_src], clips["lang_tokens"][perm_within_src], mm_d2_shuf, ctx_true0, use_obs_action=True, keep=k2,
    )
    d2_neg_u = _batched_call(
        clips["video"], clips["audio"], clips["movement"],
        -causal_prev_act, clips["lang_tokens"][perm_within_src], mm_d2_shuf, ctx_true0, use_obs_action=True, keep=k2,
    )
    v_t_eval = np.array([1, min(6, t - 1), min(14, t - 1), max(1, t - 1)], dtype=np.int64)
    has_vis_cond_ep = (mm_full[:, 0, 1] > 0.5) & ((mm_full[:, 0, 3] > 0.5) | (has_true_act > 0.5))
    if not np.any(has_vis_cond_ep):
        has_vis_cond_ep = mm_full[:, 0, 1] > 0.5
    if not np.any(has_vis_cond_ep):
        has_vis_cond_ep = np.ones((n,), dtype=bool)
    idx_vis = np.flatnonzero(has_vis_cond_ep).astype(np.int64)
    pred_v_u8 = d2_true["pred_v480"][idx_vis]
    shuf_v_u8 = d2_shuf["pred_v480"][idx_vis]
    negu_v_u8 = d2_neg_u["pred_v480"][idx_vis]
    del d2_true, d2_shuf, d2_neg_u
    targ_v_u8 = clips["video"][idx_vis[:, None], v_t_eval[None, :]]

    psnr_true, ssim_true, edge_rmse_true = _chunked_psnr_ssim_480p(pred_v_u8, targ_v_u8)
    psnr_shuf, _, _ = _chunked_psnr_ssim_480p(shuf_v_u8, targ_v_u8)
    psnr_negu, _, _ = _chunked_psnr_ssim_480p(negu_v_u8, targ_v_u8)
    mov_num, mov_den, cf_abs_sum = 0.0, 0.0, 0.0
    for c0 in range(0, idx_vis.size, _METRIC_CHUNK):
        csl = slice(c0, c0 + _METRIC_CHUNK)
        f0_ref = clips["video"][idx_vis[csl], :1].astype(np.float32) / 255.0
        pf_m = pred_v_u8[csl].astype(np.float32) / 255.0
        tf_m = targ_v_u8[csl].astype(np.float32) / 255.0
        mov_pix = (np.mean(np.abs(tf_m - f0_ref), axis=-1, keepdims=True) > (15.0 / 255.0)).astype(np.float32)
        mov_num += float(np.sum(((pf_m - tf_m) ** 2) * mov_pix, dtype=np.float64))
        mov_den += float(np.sum(mov_pix, dtype=np.float64))
        cf_abs_sum += float(np.sum(np.abs(pf_m - (negu_v_u8[csl].astype(np.float32) / 255.0)), dtype=np.float64))
    mov_mse_pix = mov_num / max(mov_den * 3.0, 1.0)
    mov_psnr = float(10.0 * np.log10(1.0 / max(float(mov_mse_pix), 1e-8)))
    cf_div_l1 = float(cf_abs_sum / max(1, pred_v_u8.size))
    d2_lift_vec = psnr_true - psnr_shuf

    dir2_metrics = {
        "n_episodes": int(np.sum(has_vis_cond_ep)),
        "resolution": [480, 640, 3],
        "psnr_480p_db": float(np.mean(psnr_true)),
        "psnr_480p_vec": [float(x) for x in psnr_true],
        "ssim_480p": float(np.mean(ssim_true)),
        "high_freq_edge_rmse_480p": float(np.mean(edge_rmse_true)),
        "edge_rmse_vec": [float(x) for x in edge_rmse_true],
        "moving_region_psnr_480p_db": mov_psnr,
        "fixed_ctx0_shuffled_cond_psnr_480p_db": float(np.mean(psnr_shuf)),
        "negated_action_lang_swap_psnr_480p_db": float(np.mean(psnr_negu)),
        "shuffle_lift_psnr_db": float(np.mean(d2_lift_vec)),
        "shuffle_lift_vec": [float(x) for x in d2_lift_vec],
        "group_ids": [int(x) for x in gids[has_vis_cond_ep]],
        "counterfactual_action_lang_lift_psnr_db": float(np.mean(psnr_true - psnr_negu)),
        "counterfactual_video_l1_divergence": cf_div_l1,
    }

    # --- Direction 3: `Video (+ Action) -> 16 kHz Audio` (`mm[aud]=0, mm[mov]=0, mm[txt]=0`) ---
    mm_d3 = np.zeros_like(mm_full)
    mm_d3[:, :, 1] = mm_full[:, :, 1]
    k3 = ("pred_audio_mel",)
    d3_true = _batched_call(
        clips["video"], clips["audio"], clips["movement"], causal_prev_act, clips["lang_tokens"], mm_d3, ctx_zero, use_obs_action=True, keep=k3
    )
    d3_shuf = _batched_call(
        clips["video"], clips["audio"], clips["movement"],
        causal_prev_act[perm_within_src], clips["lang_tokens"], mm_d3[perm_within_src], ctx_zero, use_obs_action=True,
        keep=k3, vid_idx=perm_within_src,
    )
    has_av_ep = (mm_full[:, 0, 0] > 0.5) & (mm_full[:, 0, 1] > 0.5)
    if not np.any(has_av_ep):
        has_av_ep = np.ones((n,), dtype=bool)
    pa_mel = d3_true["pred_audio_mel"][has_av_ep]
    sa_mel = d3_shuf["pred_audio_mel"][has_av_ep]
    ta_mel = clips["audio"][has_av_ep]
    mse_aud_t = np.mean((pa_mel - ta_mel) ** 2, axis=(1, 2))
    mse_aud_s = np.mean((sa_mel - ta_mel) ** 2, axis=(1, 2))
    lsd_true = np.mean(np.sqrt(np.mean(((pa_mel - ta_mel) * 25.24) ** 2, axis=-1) + 1e-8), axis=-1)
    lsd_shuf = np.mean(np.sqrt(np.mean(((sa_mel - ta_mel) * 25.24) ** 2, axis=-1) + 1e-8), axis=-1)
    env_p = np.mean(pa_mel, axis=-1)
    env_t = np.mean(ta_mel, axis=-1)
    env_corrs = [
        float(np.corrcoef(env_p[ei], env_t[ei])[0, 1]) if (np.std(env_p[ei]) > 1e-5 and np.std(env_t[ei]) > 1e-5) else 0.0
        for ei in range(env_p.shape[0])
    ]
    d3_lift_vec = lsd_shuf - lsd_true
    dir3_metrics = {
        "n_episodes": int(np.sum(has_av_ep)),
        "mel_mse": float(np.mean(mse_aud_t)),
        "shuffled_video_mel_mse": float(np.mean(mse_aud_s)),
        "shuffle_lift_mse": float(np.mean(mse_aud_s - mse_aud_t)),
        "log_spectral_distance_db": float(np.mean(lsd_true)),
        "shuffled_log_spectral_distance_db": float(np.mean(lsd_shuf)),
        "shuffle_lift_lsd_db": float(np.mean(d3_lift_vec)),
        "shuffle_lift_vec": [float(x) for x in d3_lift_vec],
        "group_ids": [int(x) for x in gids[has_av_ep]],
        "onset_envelope_corr": float(np.mean(env_corrs)),
    }

    # --- Direction 4: `Text / Video -> Physical 3D Movement` (`mm[mov]=0, action=0`, physical kinematics ONLY!) ---
    mm_d4 = mm_full.copy()
    mm_d4[:, :, 2] = 0.0
    k4 = ("pred_mov_next", "pred_mov_step4")
    d4_true = _batched_call(
        clips["video"], clips["audio"], clips["movement"], z_act_np, clips["lang_tokens"], mm_d4, ctx_zero, use_obs_action=False, keep=k4
    )
    d4_shuf = _batched_call(
        clips["video"], clips["audio"][perm_within_src], clips["movement"],
        z_act_np, clips["lang_tokens"][perm_within_src], mm_d4[perm_within_src], ctx_zero, use_obs_action=False,
        keep=k4, vid_idx=perm_within_src,
    )
    has_mov_ep = (mm_full[:, 0, 2] > 0.5) & ((mm_full[:, 0, 1] > 0.5) | (mm_full[:, 0, 3] > 0.5))
    if not np.any(has_mov_ep):
        has_mov_ep = np.ones((n,), dtype=bool)
    mov_gt = clips["movement"][has_mov_ep]
    mov_t1 = np.concatenate([mov_gt[:, 1:], mov_gt[:, -1:]], axis=1)
    sh4 = min(4, max(1, t - 1))
    mov_t4 = np.concatenate([mov_gt[:, sh4:], np.repeat(mov_gt[:, -1:], sh4, axis=1)], axis=1)
    pm1 = d4_true["pred_mov_next"][has_mov_ep]
    sm1 = d4_shuf["pred_mov_next"][has_mov_ep]
    pm4 = d4_true["pred_mov_step4"][has_mov_ep]
    mse_m_t = np.mean((pm1 - mov_t1) ** 2, axis=(1, 2))
    mse_m_s = np.mean((sm1 - mov_t1) ** 2, axis=(1, 2))
    ade_1 = np.mean(np.linalg.norm(pm1[:, :, :6] - mov_t1[:, :, :6], axis=-1), axis=-1)
    fde_4 = np.mean(np.linalg.norm(pm4[:, :, :6] - mov_t4[:, :, :6], axis=-1), axis=-1)
    d4_lift_vec = mse_m_s - mse_m_t
    dir4_metrics = {
        "n_episodes": int(np.sum(has_mov_ep)),
        "mov_mse": float(np.mean(mse_m_t)),
        "shuffled_cond_mov_mse": float(np.mean(mse_m_s)),
        "shuffle_lift_mse": float(np.mean(d4_lift_vec)),
        "shuffle_lift_vec": [float(x) for x in d4_lift_vec],
        "group_ids": [int(x) for x in gids[has_mov_ep]],
        "ade_1step_l2": float(np.mean(ade_1)),
        "fde_4step_l2": float(np.mean(fde_4)),
    }

    # --- Direction 5: `Audio + Video (+ Text) -> Continuous Robot Actions` (`has_true_action==1`, `action=0`, fixed `t=0` state!) ---
    mm_d5 = mm_full.copy()
    mm_d5[:, 1:, 2] = 0.0
    k5 = ("pred_action",)
    d5_true = _batched_call(
        clips["video"], clips["audio"], clips["movement"], z_act_np, clips["lang_tokens"], mm_d5, ctx_zero, use_obs_action=False, keep=k5
    )
    mm_d5_shuf = mm_d5.copy()
    mm_d5_shuf[:, :, 0] = mm_full[perm_within_src, :, 0]
    mm_d5_shuf[:, :, 1] = mm_full[perm_within_src, :, 1]
    mm_d5_shuf[:, :, 3] = mm_full[perm_within_src, :, 3]
    d5_shuf = _batched_call(
        clips["video"], clips["audio"][perm_within_src], clips["movement"],
        z_act_np, clips["lang_tokens"][perm_within_src], mm_d5_shuf, ctx_zero, use_obs_action=False,
        keep=k5, vid_idx=perm_within_src,
    )
    has_act_ep = has_true_act > 0.5
    if not np.any(has_act_ep):
        has_act_ep = np.ones((n,), dtype=bool)
    pa_t = d5_true["pred_action"][has_act_ep]
    pa_s = d5_shuf["pred_action"][has_act_ep]
    ta_gt = clips["target_action"][has_act_ep]
    act_mse_t = np.mean((pa_t - ta_gt) ** 2, axis=(1, 2))
    act_mse_s = np.mean((pa_s - ta_gt) ** 2, axis=(1, 2))
    act_l1_t = np.mean(np.abs(pa_t - ta_gt), axis=(1, 2))
    cos_num = np.sum(pa_t * ta_gt, axis=-1)
    cos_den = np.maximum(np.linalg.norm(pa_t, axis=-1) * np.linalg.norm(ta_gt, axis=-1), 1e-6)
    d5_lift_vec = act_mse_s - act_mse_t
    dir5_metrics = {
        "n_episodes": int(np.sum(has_act_ep)),
        "action_mse": float(np.mean(act_mse_t)),
        "action_l1": float(np.mean(act_l1_t)),
        "action_cosine": float(np.mean(cos_num / cos_den)),
        "fixed_t0_shuffled_avt_action_mse": float(np.mean(act_mse_s)),
        "shuffle_lift_mse": float(np.mean(d5_lift_vec)),
        "shuffle_lift_vec": [float(x) for x in d5_lift_vec],
        "group_ids": [int(x) for x in gids[has_act_ep]],
    }

    # --- Direction 6: `Blind Audio (+ Proprio) -> Occluded 480p Visual Motion/Impact` (`mm[vis]=0`, within-source `audio` swap!) ---
    mm_d6 = np.zeros_like(mm_full)
    mm_d6[:, :, 0] = mm_full[:, :, 0]
    mm_d6[:, :, 2] = mm_full[:, :, 2]
    k6 = ("contact_logits",)
    d6_true = _batched_call(
        clips["video"], clips["audio"], clips["movement"], z_act_np, clips["lang_tokens"], mm_d6, ctx_zero, use_obs_action=False, keep=k6
    )
    mm_d6_shuf = mm_d6.copy()
    mm_d6_shuf[:, :, 0] = mm_full[perm_within_src, :, 0]
    d6_shuf = _batched_call(
        clips["video"], clips["audio"][perm_within_src], clips["movement"],
        z_act_np, clips["lang_tokens"], mm_d6_shuf, ctx_zero, use_obs_action=False, keep=k6,
    )
    has_av_vis_ep = (mm_full[:, 0, 0] > 0.5) & (mm_full[:, 0, 1] > 0.5)
    if not np.any(has_av_vis_ep):
        has_av_vis_ep = mm_full[:, 0, 1] > 0.5
    if not np.any(has_av_vis_ep):
        has_av_vis_ep = np.ones((n,), dtype=bool)
    # Ground-truth is genuinely OCCLUDED 480p visual frame-difference motion (`||V_t - V_{t-1}||_1 > 0.025`), NEVER `audio_t`!
    idx_av_vis = np.flatnonzero(has_av_vis_ep).astype(np.int64)
    vis_mot_parts: list[np.ndarray] = []
    for c0 in range(0, idx_av_vis.size, _METRIC_CHUNK):
        vf_occ = clips["video"][idx_av_vis[c0 : c0 + _METRIC_CHUNK]].astype(np.float32) / 255.0
        vf_occ_prev = np.concatenate([vf_occ[:, :1], vf_occ[:, :-1]], axis=1)
        vis_mot_parts.append(np.mean(np.abs(vf_occ - vf_occ_prev), axis=(2, 3, 4)))
    vis_mot_gt = np.concatenate(vis_mot_parts, axis=0) if vis_mot_parts else np.zeros((0, t), dtype=np.float32)
    targ_impact = (vis_mot_gt > 0.025).astype(np.float32)
    logit_imp_t = d6_true["contact_logits"][has_av_vis_ep, :, 0]
    logit_imp_s = d6_shuf["contact_logits"][has_av_vis_ep, :, 0]
    bce_imp_t = np.mean(np.maximum(logit_imp_t, 0.0) - logit_imp_t * targ_impact + np.log1p(np.exp(-np.abs(logit_imp_t))), axis=-1)
    bce_imp_s = np.mean(np.maximum(logit_imp_s, 0.0) - logit_imp_s * targ_impact + np.log1p(np.exp(-np.abs(logit_imp_s))), axis=-1)
    d6_lift_vec = bce_imp_s - bce_imp_t
    p_imp = 1.0 / (1.0 + np.exp(-np.clip(logit_imp_t, -20.0, 20.0)))
    pos_i = p_imp[targ_impact > 0.5]
    neg_i = p_imp[targ_impact <= 0.5]
    blind_imp_auroc = _binary_auroc(pos_i, neg_i) if (pos_i.size > 0 and neg_i.size > 0) else 0.5

    dir6_metrics = {
        "n_episodes": int(np.sum(has_av_vis_ep)),
        "blind_audio_contact_bce": float(np.mean(bce_imp_t)),
        "fixed_proprio_shuffled_audio_contact_bce": float(np.mean(bce_imp_s)),
        "shuffle_lift_mse": float(np.mean(d6_lift_vec)),
        "shuffle_lift_vec": [float(x) for x in d6_lift_vec],
        "group_ids": [int(x) for x in gids[has_av_vis_ep]],
        "blind_acoustic_impact_auroc": float(blind_imp_auroc),
        "ood_commit_gate": ood_panel if ood_panel is not None else {},
    }

    lifts = [
        dir1_metrics["shuffle_lift_ce"],
        dir2_metrics["shuffle_lift_psnr_db"],
        dir3_metrics["shuffle_lift_lsd_db"],
        dir4_metrics["shuffle_lift_mse"],
        dir5_metrics["shuffle_lift_mse"],
        dir6_metrics["shuffle_lift_mse"],
    ]
    pos_dirs = int(sum(1 for x in lifts if x > 0.0))

    showcase_arrays: dict[str, np.ndarray] = {}
    if collect_showcase:
        # 1. Deterministic P05 / P25 / P50 / P75 / P95 percentile selection over `psnr_true` (`AGENTS.md` §13.2)!
        order = np.argsort(psnr_true)
        pct_ranks = [
            int(order[min(len(order) - 1, max(0, int(round(q * (len(order) - 1)))))])
            for q in (0.05, 0.25, 0.50, 0.75, 0.95)
        ]
        pct_idx = np.asarray(pct_ranks, dtype=np.int64)
        # 2. Locked cross-cohort common `vault` episode indices (`common_idx`, independent of model weights!)
        common_idx = np.linspace(0, max(0, len(psnr_true) - 1), min(6, len(psnr_true)), dtype=np.int64)
        cond_global_ids = np.flatnonzero(has_vis_cond_ep).astype(np.int64)
        n_aud_keep = min(3, pa_mel.shape[0])
        wav_pred_list = [mel64_to_waveform_griffin_lim_16khz(pa_mel[i]) for i in range(n_aud_keep)]
        wav_gt_list = [mel64_to_waveform_griffin_lim_16khz(ta_mel[i]) for i in range(n_aud_keep)]
        showcase_arrays = {
            "dir2_percentile_labels": np.array([5, 25, 50, 75, 95], dtype=np.int32),
            "dir2_percentile_ep_ids": cond_global_ids[pct_idx],
            "dir2_percentile_psnr_db": psnr_true[pct_idx].astype(np.float32),
            "dir2_obs_f0_480p_u8": clips["video"][idx_vis[pct_idx], :1],
            "dir2_targ_480p_u8": targ_v_u8[pct_idx],
            "dir2_pred_480p_u8": pred_v_u8[pct_idx],
            "dir2_negu_480p_u8": negu_v_u8[pct_idx],
            "dir2_common_ep_ids": cond_global_ids[common_idx],
            "dir2_common_psnr_db": psnr_true[common_idx].astype(np.float32),
            "dir2_common_obs_f0_480p_u8": clips["video"][idx_vis[common_idx], :1],
            "dir2_common_targ_480p_u8": targ_v_u8[common_idx],
            "dir2_common_pred_480p_u8": pred_v_u8[common_idx],
            "dir3_pred_mel64": pa_mel[:n_aud_keep],
            "dir3_targ_mel64": ta_mel[:n_aud_keep],
            "dir3_pred_wav_16khz": np.stack(wav_pred_list, axis=0) if wav_pred_list else np.zeros((0, 38400), dtype=np.float32),
            "dir3_targ_wav_16khz": np.stack(wav_gt_list, axis=0) if wav_gt_list else np.zeros((0, 38400), dtype=np.float32),
            "dir4_pred_mov": pm1[: min(4, pm1.shape[0]), :, :6],
            "dir4_targ_mov": mov_t1[: min(4, mov_t1.shape[0]), :, :6],
            "dir5_pred_act": pa_t[: min(4, pa_t.shape[0])],
            "dir5_targ_act": ta_gt[: min(4, ta_gt.shape[0])],
        }

    return (
        {
            "positive_shuffle_lift_directions": pos_dirs,
            "dir1_video_audio_to_text": dir1_metrics,
            "dir2_text_ctx_to_480p_video": dir2_metrics,
            "dir3_video_to_16khz_audio": dir3_metrics,
            "dir4_text_video_to_3d_movement": dir4_metrics,
            "dir5_av_text_to_robot_actions": dir5_metrics,
            "dir6_blind_audio_proprio_and_ood_gate": dir6_metrics,
        },
        showcase_arrays,
    )


def eval_x3_livestream_T1024_and_overwrite(
    params: dict[str, jax.Array],
    spec: X3Spec,
    *,
    seed: int = 302406,
    livestream_ticks: int = 1024,
    overwrite_episodes: int = 16,
    horizons: tuple[int, ...] = (64, 256, 1024),
    k_items_list: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    carry0 = init_x3_streaming_carry(spec, batch_size=1)
    bytes_t0 = _carry_nbytes(carry0)

    @jax.jit
    def _stream_single_tick(
        p: dict[str, jax.Array],
        carry: dict[str, Any],
        frame_u8: jax.Array,
        aud_t: jax.Array,
        mov_t: jax.Array,
        act_t: jax.Array,
        sym_t: jax.Array,
        lang_t: jax.Array,
        mm_t: jax.Array,
    ):
        b1 = {
            "video": frame_u8,
            "audio": aud_t,
            "movement": mov_t,
            "action": act_t,
            "sym_tokens": sym_t,
            "lang_tokens": lang_t,
            "modality_mask": mm_t,
            "embodiment_id": jnp.zeros((1,), dtype=jnp.int32),
            "valid": jnp.ones((1, 1), dtype=bool),
        }
        z_top, f300, beta_a, next_carry = forward_x3_world_model(p, spec, b1, streaming_carry=carry)
        z_sym, z_task = fork_x3_streams(z_top)
        u_pred = jax.lax.stop_gradient(predict_x3_causal_action_from_ztask(p, z_task, b1))
        zn1, lv1 = predict_x3_latent_step(p, spec, z_task, u_pred)
        hds = decode_x3_any_to_any_heads(
            p, spec, z_sym, z_task, zn1, lv1, f300, b1, decode_480p_frames=False, streaming_carry=carry
        )
        next_carry = {
            **next_carry,
            "prev_z_next": hds["next_prev_z_next"],
            "prev_log_var": hds["next_prev_log_var"],
        }
        mode_id = jnp.argmax(hds["commit_logits"][0, 0], axis=-1)
        return (
            next_carry,
            hds["pred_goal_xy"][0, 0],
            hds["pred_action"][0, 0, :2],
            jnp.argmax(hds["sym_val_logits"][0, 0]) + X3_ASSOC_VAL_OFFSET,
            mode_id,
            beta_a[0, 0, 0],
        )

    goals_4 = np.array([[0.52, 0.36], [-0.52, 0.36], [-0.52, -0.36], [0.52, -0.36]], dtype=np.float32)
    st_ls = init_unseen_blocked_worlds(jax.random.key(seed), batch_size=1)
    st_ls = {**st_ls, "box_hx": jnp.zeros_like(st_ls["box_hx"]), "box_hy": jnp.zeros_like(st_ls["box_hy"])}
    ee_ls, ob_ls = [], []
    cur_ls = st_ls
    for ti in range(64):
        g_cur = jnp.asarray(goals_4[(ti // 16) % 4])[None, :]
        u_ls = _proportional_push_action(cur_ls["ee_xy"], cur_ls["obj_xy"], g_cur, final_goal_xy=g_cur)[:, :2]
        ee_ls.append(cur_ls["ee_xy"])
        ob_ls.append(cur_ls["obj_xy"])
        cur_ls, _ = step_unseen_layout_world(cur_ls, u_ls)
    ee_bank = jnp.stack(ee_ls, axis=1)
    ob_bank = jnp.stack(ob_ls, axis=1)
    g_bank = jnp.full((1, 64, 2), 9.0, dtype=jnp.float32)
    vid_bank_64 = render_arena_480p_batch(
        ee_bank, ob_bank, g_bank, st_ls["box_cx"], st_ls["box_cy"], st_ls["box_hx"], st_ls["box_hy"]
    )

    z_aud = jnp.zeros((1, 1, 64), dtype=jnp.float32)
    z_act = jnp.zeros((1, 1, 16), dtype=jnp.float32)
    z_sym_tok = jnp.zeros((1, 1, 4), dtype=jnp.int32)
    z_lang_tok = jnp.zeros((1, 1, 4), dtype=jnp.int32)
    mm_vis_txt = jnp.array([[[0.0, 1.0, 1.0, 1.0]]], dtype=jnp.float32)
    mm_blackout = jnp.array([[[0.0, 0.0, 1.0, 0.0]]], dtype=jnp.float32)
    gq_1 = encode_goal_query_tokens((1,))[:, None, :]
    g_toks_4 = [encode_goal_tokens_2d(jnp.asarray(g)[None, :], is_distractor=False)[:, None, :] for g in goals_4]

    t_compile0 = time.perf_counter()
    w_carry, w_g, _, _, _, _ = _stream_single_tick(
        params, carry0, vid_bank_64[:, 0:1], z_aud, jnp.zeros((1, 1, 32)), z_act, z_sym_tok, g_toks_4[0], mm_vis_txt
    )
    jax.block_until_ready(w_g)
    cold_compile_ms = (time.perf_counter() - t_compile0) * 1000.0
    del w_carry

    carry = init_x3_streaming_carry(spec, batch_size=1)
    bytes_t1 = bytes_t0
    bytes_t512 = bytes_t0
    bytes_t1024 = bytes_t0
    tick_ms: list[float] = []
    goal_trace = np.zeros((livestream_ticks, 2), dtype=np.float32)
    targ_goal_trace = np.zeros((livestream_ticks, 2), dtype=np.float32)
    mode_trace = np.zeros((livestream_ticks,), dtype=np.int32)

    for ti in range(livestream_ticks):
        seg_id = min(3, ti // max(1, livestream_ticks // 4))
        t_seg_start = seg_id * max(1, livestream_ticks // 4)
        targ_goal_trace[ti] = goals_4[seg_id]
        is_goal_flash = (ti - t_seg_start) < 3
        is_blackout = (livestream_ticks >= 640) and (600 <= ti < 620)
        f_in = jnp.zeros_like(vid_bank_64[:, 0:1]) if is_blackout else vid_bank_64[:, (ti % 64) : (ti % 64) + 1]
        ee_now = ee_bank[:, (ti % 64) : (ti % 64) + 1]
        mov_in = jnp.concatenate([ee_now, ee_now, jnp.zeros((1, 1, 28), dtype=jnp.float32)], axis=-1)
        if is_goal_flash:
            l_in = g_toks_4[seg_id]
            mm_in = mm_vis_txt
        elif is_blackout:
            l_in = z_lang_tok
            mm_in = mm_blackout
        else:
            l_in = gq_1
            mm_in = mm_vis_txt

        t_tick0 = time.perf_counter()
        carry, pg_t, _, _, md_t, _ = _stream_single_tick(
            params, carry, f_in, z_aud, mov_in, z_act, z_sym_tok, l_in, mm_in
        )
        jax.block_until_ready(pg_t)
        dt_ms = (time.perf_counter() - t_tick0) * 1000.0
        tick_ms.append(dt_ms)
        goal_trace[ti] = np.asarray(pg_t)
        mode_trace[ti] = int(md_t)
        if ti == 0:
            bytes_t1 = _carry_nbytes(carry)
        elif ti == (livestream_ticks // 2) - 1:
            bytes_t512 = _carry_nbytes(carry)
        elif ti == livestream_ticks - 1:
            bytes_t1024 = _carry_nbytes(carry)

    tick_ms_arr = np.asarray(tick_ms, dtype=np.float64)
    total_stream_s = float(np.sum(tick_ms_arr) / 1000.0)
    seg_len = max(1, livestream_ticks // 4)
    quad_hits = []
    for si in range(4):
        s0 = si * seg_len + 4
        s1 = min(livestream_ticks, (si + 1) * seg_len)
        if s1 > s0:
            same_sign = np.all(np.sign(goal_trace[s0:s1]) == np.sign(goals_4[si][None, :]), axis=-1)
            quad_hits.append(float(np.mean(same_sign)))
    blackout_non_commit = float(np.mean(mode_trace[600:620] != 0)) if livestream_ticks >= 640 else 1.0
    recovery_commit = float(np.mean(mode_trace[622:640] == 0)) if livestream_ticks >= 640 else 1.0

    @jax.jit
    def _stream_chunk32(
        p: dict[str, jax.Array],
        carry_c: dict[str, Any],
        vid_c: jax.Array,
        aud_c: jax.Array,
        sym_c: jax.Array,
    ):
        bs, tc = vid_c.shape[:2]
        bc = {
            "video": vid_c,
            "audio": aud_c,
            "movement": jnp.zeros((bs, tc, 32), dtype=jnp.float32),
            "action": jnp.zeros((bs, tc, 16), dtype=jnp.float32),
            "sym_tokens": sym_c,
            "lang_tokens": jnp.zeros((bs, tc, 4), dtype=jnp.int32),
            "modality_mask": jnp.ones((bs, tc, 4), dtype=jnp.float32).at[:, :, 3].set(0.0),
            "embodiment_id": jnp.zeros((bs,), dtype=jnp.int32),
            "valid": jnp.ones((bs, tc), dtype=bool),
        }
        z_top, _, _, nxt_c = forward_x3_world_model(p, spec, bc, streaming_carry=carry_c)
        z_sym, _ = fork_x3_streams(z_top)
        sym_h = z_sym.astype(jnp.float32) @ p["head_sym_w"].astype(jnp.float32)
        vw = p["text_emb"].astype(jnp.float32)[X3_ASSOC_VAL_OFFSET : X3_ASSOC_VAL_OFFSET + X3_ASSOC_NUM_VALS].T
        logits = (sym_h @ vw) * (spec.d_model**-0.5)
        preds = jnp.argmax(logits, axis=-1) + X3_ASSOC_VAL_OFFSET
        return nxt_c, preds

    vid_dyn_4 = jnp.tile(vid_bank_64[:, :32], (min(4, overwrite_episodes), 1, 1, 1, 1))
    bs_ov = int(vid_dyn_4.shape[0])
    n_batches_ov = max(1, overwrite_episodes // bs_ov)
    curves: dict[str, Any] = {}
    flat_ov: dict[str, Any] = {}

    for horizon in horizons:
        h_entry: dict[str, Any] = {}
        for k_items in k_items_list:
            m_over = 1 if k_items <= 2 else (k_items // 2)
            t_ov = max(2 + k_items, horizon // 2)
            q_start = horizon - k_items
            all_c, ov_c, ret_c, st_c = [], [], [], []
            rng_ov = np.random.default_rng(seed + 1000 * horizon + 31 * k_items)

            for _ in range(n_batches_ov):
                sym_full = np.zeros((bs_ov, horizon, 4), dtype=np.int32)
                targ_q = np.zeros((bs_ov, k_items), dtype=np.int32)
                stale_q = np.zeros((bs_ov, k_items), dtype=np.int32)
                for bi in range(bs_ov):
                    pkeys = rng_ov.permutation(np.arange(X3_ASSOC_KEY_OFFSET, X3_ASSOC_KEY_OFFSET + X3_ASSOC_NUM_KEYS, dtype=np.int32))
                    vi = rng_ov.integers(X3_ASSOC_VAL_OFFSET, X3_ASSOC_VAL_OFFSET + X3_ASSOC_NUM_VALS, size=(k_items,), dtype=np.int32)
                    vo = X3_ASSOC_VAL_OFFSET + ((vi - X3_ASSOC_VAL_OFFSET + rng_ov.integers(1, X3_ASSOC_NUM_VALS, size=(k_items,))) % X3_ASSOC_NUM_VALS)
                    for j in range(k_items):
                        sym_full[bi, 1 + j, 1] = pkeys[j]
                        sym_full[bi, 1 + j, 2] = vi[j]
                    for j in range(m_over):
                        sym_full[bi, t_ov + j, 1] = pkeys[j]
                        sym_full[bi, t_ov + j, 2] = vo[j]
                    for j in range(k_items):
                        sym_full[bi, q_start + j, 1] = pkeys[j]
                        sym_full[bi, q_start + j, 3] = 2
                        targ_q[bi, j] = vo[j] if j < m_over else vi[j]
                        stale_q[bi, j] = vi[j]

                carry_b = init_x3_streaming_carry(spec, batch_size=bs_ov)
                preds_chunks = []
                for t0 in range(0, horizon, 32):
                    aud_c = jnp.asarray(0.03 * rng_ov.standard_normal((bs_ov, 32, 64)).astype(np.float32))
                    carry_b, pr_c = _stream_chunk32(params, carry_b, vid_dyn_4, aud_c, jnp.asarray(sym_full[:, t0 : t0 + 32]))
                    if t0 + 32 > q_start:
                        preds_chunks.append((t0, np.asarray(jax.device_get(pr_c))))
                pred_full_tail = np.zeros((bs_ov, horizon), dtype=np.int32)
                for t0, pr_c in preds_chunks:
                    pred_full_tail[:, t0 : t0 + 32] = pr_c
                pr_q = pred_full_tail[:, q_start : q_start + k_items]
                corr = (pr_q == targ_q).astype(np.float32)
                all_c.append(np.mean(corr, axis=-1))
                ov_c.append(np.mean(corr[:, :m_over], axis=-1))
                st_c.append(np.mean((pr_q[:, :m_over] == stale_q[:, :m_over]).astype(np.float32), axis=-1))
                if k_items > m_over:
                    ret_c.append(np.mean(corr[:, m_over:], axis=-1))

            ep_all = [float(x) for x in np.concatenate(all_c, axis=0)]
            ep_ov = [float(x) for x in np.concatenate(ov_c, axis=0)]
            ep_st = [float(x) for x in np.concatenate(st_c, axis=0)]
            ep_ret = [float(x) for x in np.concatenate(ret_c, axis=0)] if ret_c else None
            h_entry[f"K{k_items}"] = {
                "overall_acc": float(np.mean(ep_all)),
                "overwritten_acc": float(np.mean(ep_ov)),
                "retained_acc": float(np.mean(ep_ret)) if ep_ret is not None else None,
                "stale_capture_rate": float(np.mean(ep_st)),
                "ep_vec": ep_all,
                "overwritten_vec": ep_ov,
            }
            flat_ov[f"K{k_items}_T{horizon}_acc"] = float(np.mean(ep_all))
            flat_ov[f"K{k_items}_T{horizon}_overwritten_acc"] = float(np.mean(ep_ov))
            flat_ov[f"K{k_items}_T{horizon}_overwritten_vec"] = ep_ov
            flat_ov[f"K{k_items}_T{horizon}_vec"] = ep_all
        curves[f"T{horizon}"] = h_entry

    state_acct = compute_x3_streaming_state_bytes(spec, batch_size=1)
    return (
        {
            "livestream_ticks": int(livestream_ticks),
            "cold_compile_ms": round(cold_compile_ms, 2),
            "ttft_ms": round(float(tick_ms_arr[0]), 3),
            "sustained_fps": round(float(livestream_ticks / max(total_stream_s, 1e-6)), 2),
            "latency_p50_ms": round(float(np.percentile(tick_ms_arr, 50)), 3),
            "latency_p95_ms": round(float(np.percentile(tick_ms_arr, 95)), 3),
            "latency_p99_ms": round(float(np.percentile(tick_ms_arr, 99)), 3),
            "carry_bytes_t1": int(bytes_t1),
            "carry_bytes_t512": int(bytes_t512),
            "carry_bytes_t1024": int(bytes_t1024),
            "state_growth_bytes": int(bytes_t1024 - bytes_t1),
            "strict_system_state_mib": state_acct["strict_system_state_mib"],
            "core_state_mib": state_acct["core_state_mib"],
            "midstream_4way_goal_switch_quadrant_acc": float(np.mean(quad_hits)) if quad_hits else 0.0,
            "midstream_blackout_fallback_rate": blackout_non_commit,
            "midstream_post_blackout_recovery_commit_rate": recovery_commit,
            "multi_item_overwrite_capacity": {
                **flat_ov,
                "curves": curves,
            },
        },
        {
            "livestream_pred_goal_trace": goal_trace,
            "livestream_targ_goal_trace": targ_goal_trace,
            "livestream_mode_trace": mode_trace,
            "livestream_tick_ms": tick_ms_arr.astype(np.float32),
        },
    )


def _plan_x3_subgoal_mppi_tick(
    params: dict[str, jax.Array],
    spec: X3Spec,
    z_task_cur: jax.Array,
    vis_ee_xy: jax.Array,
    vis_obj_xy: jax.Array,
    goal_xy: jax.Array,
    occ_8x8: jax.Array,
    prev_wp: jax.Array,
    has_prev_wp: jax.Array,
    dyn_gain_pred: jax.Array,
    rng_key: jax.Array,
    *,
    horizon_h: int = 6,
    num_candidates: int = 24,
    use_jepa_rollout: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Learned `H = 6` latent JEPA subgoal + `SLOT_SYSID` adaptive push-gain MPPI."""
    b = z_task_cur.shape[0]
    v_grid = compute_learned_value_grid_8x8(occ_8x8, goal_xy)
    all_wp, v_wp, occ_wp, reach_ring = _propose_subgoal_candidates_shared(
        vis_obj_xy, goal_xy, occ_8x8, v_grid, num_candidates
    )
    kc = all_wp.shape[1]
    gain_mult = jnp.clip(1.0 + 0.45 * dyn_gain_pred, 0.55, 1.45)  # [B]

    if use_jepa_rollout:
        z0 = jnp.repeat(z_task_cur[:, 0, :, :], kc, axis=0)[:, None, :, :]
        ee_rep = jnp.repeat(vis_ee_xy, kc, axis=0)
        ob_rep = jnp.repeat(vis_obj_xy, kc, axis=0)
        wp_flat = all_wp.reshape(b * kc, 2)
        g_rep = jnp.repeat(goal_xy, kc, axis=0)
        gm_rep = jnp.repeat(gain_mult, kc, axis=0)[:, None]

        def _lat_step(carry_l, hi):
            zc, ee_c, ob_c = carry_l
            kn = jax.random.fold_in(rng_key, hi)
            u_nom = _proportional_push_action(ee_c, ob_c, wp_flat, final_goal_xy=g_rep)[:, :2]
            u2 = jnp.clip(u_nom * gm_rep + 0.05 * jax.random.normal(kn, u_nom.shape), -1.0, 1.0)
            u16 = jnp.concatenate([u2, jnp.zeros((b * kc, 14), dtype=jnp.float32)], axis=-1)[:, None, :]
            zn, lv = predict_x3_latent_step(params, spec, zc, u16)
            reg_c = jnp.mean(zn[:, 0, 56:59, :].astype(jnp.float32), axis=1)
            sys_c = zn[:, 0, 59, :].astype(jnp.float32)
            ctrl_h = jax.nn.silu(
                (0.55 * reg_c + 0.45 * sys_c) @ params["ctrl_trunk_w"].astype(jnp.float32)
                + params["ctrl_trunk_b"].astype(jnp.float32)
            )
            val_s = (ctrl_h @ params["head_val_w"].astype(jnp.float32) + params["head_val_b"].astype(jnp.float32))[..., 0]
            mov_s = (
                (0.65 * jnp.mean(zn[:, 0, 0:48, :].astype(jnp.float32), axis=1) + 0.35 * sys_c)
                @ params["head_mov_next_w"].astype(jnp.float32)
                + params["head_mov_next_b"].astype(jnp.float32)
            )
            unc_pen = 0.08 * jnp.mean(jnp.exp(0.5 * lv[:, 0, :, 0]), axis=-1)
            ee_n = jnp.clip(0.5 * (ee_c + 0.14 * u2) + 0.5 * mov_s[:, 0:2], -0.95, 0.95)
            ob_n = jnp.clip(0.65 * ob_c + 0.35 * mov_s[:, 2:4], -0.95, 0.95)
            return (zn, ee_n, ob_n), (val_s - unc_pen, ob_n)

        (_, _, ob_fin), (step_vals, _) = jax.lax.scan(_lat_step, (z0, ee_rep, ob_rep), jnp.arange(horizon_h))
        rollout_ret = jnp.mean(step_vals, axis=0).reshape(b, kc)
        fin_v = jnp.sqrt(jnp.sum((ob_fin.reshape(b, kc, 2) - goal_xy[:, None, :]) ** 2, axis=-1) + 1e-6)
    else:
        rollout_ret = jnp.zeros((b, kc), dtype=jnp.float32)
        fin_v = jnp.zeros((b, kc), dtype=jnp.float32)

    scores = _score_subgoal_candidates_shared(
        vis_obj_xy,
        goal_xy,
        all_wp,
        v_wp,
        occ_wp,
        reach_ring,
        occ_8x8,
        rollout_ret,
        fin_v,
        prev_wp,
        has_prev_wp,
        use_rollout=use_jepa_rollout,
    )
    best_i = jnp.argmax(scores, axis=-1)
    chosen_wp = jnp.take_along_axis(all_wp, best_i[:, None, None], axis=1)[:, 0, :]
    u_raw = _proportional_push_action(vis_ee_xy, vis_obj_xy, chosen_wp, final_goal_xy=goal_xy)[:, :2]
    u_cmd = jnp.clip(u_raw * gain_mult[:, None], -1.0, 1.0)
    return u_cmd, chosen_wp


def eval_x3_unseen_layout_mppi_480p(
    params: dict[str, jax.Array],
    spec: X3Spec,
    *,
    seed: int = 302401,
    n_worlds: int = 64,
    horizon_ticks: int = 116,
    chunk_bs: int = 8,
) -> dict[str, Any]:
    """De-privileged `480p` unseen-layout MPPI + true OOD `(2.20, 0.90)` `dynamics_shift` + `L0` `SysID` counterfactuals."""
    st_all = init_unseen_blocked_worlds(jax.random.key(seed), batch_size=n_worlds)
    # Enforce true held-out OOD `dynamics_shift` physics `(2.20, 0.90)`, strictly disjoint from `[0.55, 1.85] x [0.55, 0.86]`!
    is_dyn_w = st_all["layout_id"] == LAYOUT_DYNAMICS_SHIFT
    st_all = {
        **st_all,
        "mass_scale": jnp.where(is_dyn_w, 2.20, st_all["mass_scale"]),
        "damping": jnp.where(is_dyn_w, 0.90, st_all["damping"]),
    }

    @functools.partial(jax.jit, static_argnums=(0,))
    def _oracle_rollout(use_roll: bool):
        def _step_o(carry, t_i):
            st, pwp, rk = carry
            rk, _ = jax.random.split(rk)
            cs = -1.0 + (2.0 * (jnp.arange(8, dtype=jnp.float32) + 0.5) / 8.0)
            gy, gx = jnp.meshgrid(cs, cs, indexing="ij")
            occ_o = jnp.zeros((n_worlds, 8, 8), dtype=jnp.float32)
            for bi in range(3):
                act = (st["box_hx"][:, bi] > 1e-4)[:, None, None]
                hit = act & (jnp.abs(gx[None] - st["box_cx"][:, bi, None, None]) <= st["box_hx"][:, bi, None, None] + 0.12) & (
                    jnp.abs(gy[None] - st["box_cy"][:, bi, None, None]) <= st["box_hy"][:, bi, None, None] + 0.12
                )
                occ_o = jnp.maximum(occ_o, hit.astype(jnp.float32))
            vg = compute_learned_value_grid_8x8(occ_o, st["targ_a"])
            all_wp, v_wp, occ_wp, r_ring = _propose_subgoal_candidates_shared(st["obj_xy"], st["targ_a"], occ_o, vg, 24)
            kc = all_wp.shape[1]
            o_gain = jnp.where(st["layout_id"] == LAYOUT_DYNAMICS_SHIFT, 1.25, 1.0)[:, None]
            if use_roll:
                st_r = jax.tree.map(lambda x: jnp.repeat(x, kc, axis=0), st)
                wp_f = all_wp.reshape(n_worlds * kc, 2)
                gf = jnp.repeat(st["targ_a"], kc, axis=0)
                og_r = jnp.repeat(o_gain, kc, axis=0)

                def _osim(sc, _):
                    u2 = jnp.clip(_proportional_push_action(sc["ee_xy"], sc["obj_xy"], wp_f, final_goal_xy=gf)[:, :2] * og_r, -1.0, 1.0)
                    sn, _ = step_unseen_layout_world(sc, u2)
                    d = jnp.sqrt(jnp.sum((sn["obj_xy"] - gf) ** 2, axis=-1) + 1e-6)
                    return sn, -1.35 * d

                sfin, r_h = jax.lax.scan(_osim, st_r, jnp.arange(6))
                r_ret = jnp.mean(r_h, axis=0).reshape(n_worlds, kc)
                fv = jnp.sqrt(jnp.sum((sfin["obj_xy"].reshape(n_worlds, kc, 2) - st["targ_a"][:, None, :]) ** 2, axis=-1) + 1e-6)
            else:
                r_ret = jnp.zeros((n_worlds, kc), dtype=jnp.float32)
                fv = jnp.zeros((n_worlds, kc), dtype=jnp.float32)
            sc = _score_subgoal_candidates_shared(
                st["obj_xy"], st["targ_a"], all_wp, v_wp, occ_wp, r_ring, occ_o, r_ret, fv, pwp, t_i > 0, use_rollout=use_roll
            )
            bi = jnp.argmax(sc, axis=-1)
            cwp = jnp.take_along_axis(all_wp, bi[:, None, None], axis=1)[:, 0, :]
            u = jnp.clip(_proportional_push_action(st["ee_xy"], st["obj_xy"], cwp, final_goal_xy=st["targ_a"])[:, :2] * o_gain, -1.0, 1.0)
            sn, _ = step_unseen_layout_world(st, u)
            dg = jnp.sqrt(jnp.sum((sn["obj_xy"] - st["targ_a"]) ** 2, axis=-1) + 1e-8)
            return (sn, cwp, rk), dg

        _, dh = jax.lax.scan(_step_o, (st_all, st_all["targ_a"], jax.random.key(seed + 3)), jnp.arange(horizon_ticks))
        tw = min(16, horizon_ticks)
        return (jnp.min(dh[-tw:], axis=0) < 0.24).astype(jnp.float32)

    oracle_succ = np.asarray(jax.device_get(_oracle_rollout(True)))
    oracle_nojepa = np.asarray(jax.device_get(_oracle_rollout(False)))

    @functools.partial(jax.jit, static_argnums=(2,))
    def _eval_mode_chunk(p_use: dict[str, jax.Array], st_sub: dict[str, jax.Array], mode: str):
        bs = st_sub["ee_xy"].shape[0]
        carry0 = init_x3_streaming_carry(spec, batch_size=bs)
        g_tok = encode_goal_tokens_2d(st_sub["targ_a"], is_distractor=False)[:, None, :]
        q_tok = encode_goal_query_tokens((bs,))[:, None, :]
        occ0 = jnp.zeros((bs, 8, 8), dtype=jnp.float32)
        u0 = jnp.zeros((bs, 16), dtype=jnp.float32)

        def _body(carry, t_i):
            st, sc_carry, prev_u16, prev_wp, occ_st, prev_ee, rk = carry
            rk, pk = jax.random.split(rk)
            vid_t = render_arena_480p_batch(
                st["ee_xy"][:, None], st["obj_xy"][:, None],
                jnp.where(t_i < 4, st["targ_a"][:, None], jnp.full_like(st["targ_a"][:, None], 9.0)),
                st["box_cx"], st["box_cy"], st["box_hx"], st["box_hy"],
            )
            ee_v = jnp.where(t_i > 0, st["ee_xy"] - prev_ee, 0.0)
            mov_t = jnp.concatenate([st["ee_xy"], st["ee_xy"], ee_v, jnp.zeros((bs, 26))], axis=-1)[:, None, :]
            lt = jnp.where(t_i < 4, g_tok, q_tok)
            b1 = {
                "video": vid_t,
                "audio": jnp.zeros((bs, 1, 64), dtype=jnp.float32),
                "movement": mov_t,
                "action": prev_u16[:, None, :],
                "sym_tokens": jnp.zeros((bs, 1, 4), dtype=jnp.int32),
                "lang_tokens": lt,
                "modality_mask": jnp.array([[[0.0, 1.0, 1.0, 1.0]]], dtype=jnp.float32) * jnp.ones((bs, 1, 1)),
                "embodiment_id": jnp.zeros((bs,), dtype=jnp.int32),
                "valid": jnp.ones((bs, 1), dtype=bool),
            }
            s_mode = "zeroed" if mode == "sysid_zeroed" else ("swapped" if mode == "sysid_swapped" else "normal")
            # `init_unseen_blocked_worlds` cycles `layout_id = 1, 2, 3, 4` across consecutive indices, so `shift = 1`
            # swaps every `dynamics_shift` (`4`) world with an `off_center` (`1`) / standard-physics world at L0 input!
            swap_p = jnp.roll(jnp.arange(bs, dtype=jnp.int32), shift=1) if bs > 1 else jnp.arange(bs, dtype=jnp.int32)
            z_top, f300, _, nxt_c = forward_x3_world_model(
                p_use, spec, b1, streaming_carry=sc_carry, sysid_mode=s_mode, sysid_swap_perm=swap_p
            )
            z_sym, z_task = fork_x3_streams(z_top)
            zn1, lv1 = predict_x3_latent_step(p_use, spec, z_task, prev_u16[:, None, :])
            hds = decode_x3_any_to_any_heads(
                p_use, spec, z_sym, z_task, zn1, lv1, f300, b1, decode_480p_frames=False, streaming_carry=sc_carry
            )
            nxt_c = {**nxt_c, "prev_z_next": hds["next_prev_z_next"], "prev_log_var": hds["next_prev_log_var"]}
            vee = hds["vis_ee_xy"][:, 0]
            vobj = hds["vis_obj_xy"][:, 0]
            pg = jnp.where(t_i < 4, st["targ_a"], hds["pred_goal_xy"][:, 0])
            occ_nxt = update_self_filtered_occ_8x8(
                occ_st,
                hds["occ_logits_8x8"][:, 0],
                vee,
                vobj,
                t_i == 0,
                ext_goal_xy=jnp.where(t_i < 4, st["targ_a"], jnp.zeros_like(st["targ_a"])),
            )
            # Pure single-factor `No-JEPA`: keep `dyn_g` active; zero `dyn_g` ONLY in `sysid_zeroed`!
            dyn_g = jnp.zeros((bs,), dtype=jnp.float32) if mode == "sysid_zeroed" else hds["pred_action"][:, 0, 2]

            if mode == "open_loop":
                u2 = hds["pred_action"][:, 0, :2]
                cwp = prev_wp
            else:
                u2, cwp = _plan_x3_subgoal_mppi_tick(
                    p_use,
                    spec,
                    z_task,
                    vee,
                    vobj,
                    pg,
                    occ_nxt,
                    prev_wp,
                    t_i > 0,
                    dyn_g,
                    pk,
                    use_jepa_rollout=(mode != "nojepa"),
                )
            sn, _ = step_unseen_layout_world(st, u2)
            u16_n = jnp.concatenate([u2, jnp.zeros((bs, 14))], axis=-1)
            dg = jnp.sqrt(jnp.sum((sn["obj_xy"] - st["targ_a"]) ** 2, axis=-1) + 1e-8)
            oe = jnp.sqrt(jnp.sum((vobj - st["obj_xy"]) ** 2, axis=-1) + 1e-8)
            return (sn, nxt_c, u16_n, cwp, occ_nxt, st["ee_xy"], rk), (dg, oe)

        (_, _, _, _, _, _, _), (dh, oeh) = jax.lax.scan(
            _body, (st_sub, carry0, u0, st_sub["targ_a"], occ0, st_sub["ee_xy"], jax.random.key(seed + 19)), jnp.arange(horizon_ticks)
        )
        tw = min(16, horizon_ticks)
        return (jnp.min(dh[-tw:], axis=0) < 0.24).astype(jnp.float32), jnp.mean(oeh, axis=0)

    def _run_mode_all(p_in: dict[str, jax.Array], mode: str) -> tuple[np.ndarray, np.ndarray]:
        sl_s, sl_o = [], []
        cbs = min(chunk_bs, n_worlds)
        for i0 in range(0, n_worlds, cbs):
            sub_st = jax.tree.map(lambda x: x[i0 : i0 + cbs], st_all)
            s_c, o_c = jax.device_get(_eval_mode_chunk(p_in, sub_st, mode))
            sl_s.append(np.asarray(s_c))
            sl_o.append(np.asarray(o_c))
        return np.concatenate(sl_s, 0), np.concatenate(sl_o, 0)

    mppi_s, obj_err = _run_mode_all(params, "mppi")
    nojepa_s, _ = _run_mode_all(params, "nojepa")
    open_s, _ = _run_mode_all(params, "open_loop")
    nosysid_s, _ = _run_mode_all(params, "sysid_zeroed")
    swapsysid_s, _ = _run_mode_all(params, "sysid_swapped")

    lid = np.asarray(st_all["layout_id"])
    fams = {
        LAYOUT_OFF_CENTER: "off_center_wall",
        LAYOUT_DUAL_CHICANE: "dual_chicane",
        LAYOUT_MULTI_CLUTTER: "multi_block_clutter",
        LAYOUT_DYNAMICS_SHIFT: "dynamics_shift",
    }
    by_fam: dict[str, Any] = {}
    for code, fn in fams.items():
        m = lid == code
        by_fam[fn] = {
            "oracle_mppi_ceiling": float(np.mean(oracle_succ[m])) if np.any(m) else 0.0,
            "mppi_success": float(np.mean(mppi_s[m])) if np.any(m) else 0.0,
            "no_jepa_subgoal_success": float(np.mean(nojepa_s[m])) if np.any(m) else 0.0,
            "sysid_zeroed_mppi_success": float(np.mean(nosysid_s[m])) if np.any(m) else 0.0,
            "sysid_swapped_mppi_success": float(np.mean(swapsysid_s[m])) if np.any(m) else 0.0,
            "open_loop_success": float(np.mean(open_s[m])) if np.any(m) else 0.0,
        }

    dyn_m = lid == LAYOUT_DYNAMICS_SHIFT
    return {
        "oracle_perception_dynamics_mppi_ceiling": float(np.mean(oracle_succ)),
        "oracle_no_jepa_subgoal_ceiling": float(np.mean(oracle_nojepa)),
        "geometry_free_mppi_success": float(np.mean(mppi_s)),
        "geometry_free_mppi_succ_vec": [float(x) for x in mppi_s],
        "no_jepa_subgoal_success": float(np.mean(nojepa_s)),
        "no_jepa_subgoal_succ_vec": [float(x) for x in nojepa_s],
        "sysid_zeroed_mppi_success": float(np.mean(nosysid_s)),
        "sysid_zeroed_succ_vec": [float(x) for x in nosysid_s],
        "sysid_swapped_mppi_success": float(np.mean(swapsysid_s)),
        "sysid_swapped_succ_vec": [float(x) for x in swapsysid_s],
        "dynamics_shift_succ_vec": [float(x) for x in mppi_s[dyn_m]],
        "dynamics_shift_sysid_zeroed_vec": [float(x) for x in nosysid_s[dyn_m]],
        "dynamics_shift_sysid_swapped_vec": [float(x) for x in swapsysid_s[dyn_m]],
        "mppi_over_no_jepa_gain_pp": float((np.mean(mppi_s) - np.mean(nojepa_s)) * 100.0),
        "sysid_dynamics_shift_gain_pp": float(
            (by_fam["dynamics_shift"]["mppi_success"] - by_fam["dynamics_shift"]["sysid_zeroed_mppi_success"]) * 100.0
        ),
        "sysid_over_swapped_dynamics_shift_gain_pp": float(
            (by_fam["dynamics_shift"]["mppi_success"] - by_fam["dynamics_shift"]["sysid_swapped_mppi_success"]) * 100.0
        ),
        "open_loop_policy_success": float(np.mean(open_s)),
        "vis_decoded_obj_rmse_480p": float(np.mean(obj_err)),
        "by_unseen_family": by_fam,
    }


def eval_x3_T160_paired_switch_480p(
    params: dict[str, jax.Array],
    spec: X3Spec,
    *,
    seed: int = 302402,
    n_worlds: int = 32,
    chunk_bs: int = 8,
) -> dict[str, Any]:
    total_t = 160
    sw_all = init_shortcut_free_t160_switch_worlds(jax.random.key(seed), batch_size=n_worlds)

    @jax.jit
    def _run_pair_sub(p: dict[str, jax.Array], sw_sub: dict[str, jax.Array], instructed_gb: jax.Array):
        bs = instructed_gb.shape[0]
        ga = sw_sub["targ_a"]
        fdist = sw_sub["false_distractor_xy"]
        ree = sw_sub["reset_ee_xy"]
        robj = sw_sub["reset_obj_xy"]
        st0 = {k: v for k, v in sw_sub.items() if k not in ("reset_ee_xy", "reset_obj_xy", "targ_b_prime", "false_distractor_xy")}
        carry0 = init_x3_streaming_carry(spec, batch_size=bs)
        t_ga = encode_goal_tokens_2d(ga, is_distractor=False)[:, None, :]
        t_gb = encode_goal_tokens_2d(instructed_gb, is_distractor=False)[:, None, :]
        t_fd = encode_goal_tokens_2d(fdist, is_distractor=True)[:, None, :]
        t_qr = encode_goal_query_tokens((bs,))[:, None, :]
        z_tok = jnp.zeros((bs, 1, 4), dtype=jnp.int32)
        u0 = jnp.zeros((bs, 16), dtype=jnp.float32)

        def _step_t160(carry, t_i):
            st, sc_c, prev_u16, prev_wp, prev_ee, rk = carry
            rk, pk = jax.random.split(rk)
            is_ga = t_i < 4
            is_gb = (t_i >= 16) & (t_i < 20)
            is_fd = ((t_i >= 48) & (t_i < 52)) | ((t_i >= 76) & (t_i < 80))
            is_act = t_i >= 88
            shown_g = jnp.where(is_ga, ga, jnp.where(is_gb, instructed_gb, jnp.where(is_fd, fdist, jnp.full_like(ga, 9.0))))
            vid_t = render_arena_480p_batch(
                st["ee_xy"][:, None], st["obj_xy"][:, None], shown_g[:, None],
                st["box_cx"], st["box_cy"], st["box_hx"], st["box_hy"],
            )
            ee_v = jnp.where((t_i > 0) & (t_i != 88), st["ee_xy"] - prev_ee, 0.0)
            mov_t = jnp.concatenate([st["ee_xy"], st["ee_xy"], ee_v, jnp.zeros((bs, 26))], axis=-1)[:, None, :]
            lt = jnp.where(is_ga, t_ga, jnp.where(is_gb, t_gb, jnp.where(is_fd, t_fd, jnp.where(is_act, t_qr, z_tok))))
            txt_on = (is_ga | is_gb | is_fd | is_act).astype(jnp.float32)
            mm_t = jnp.array([[[0.0, 1.0, 1.0, 0.0]]], dtype=jnp.float32).at[:, :, 3].set(txt_on) * jnp.ones((bs, 1, 1))
            b1 = {
                "video": vid_t,
                "audio": jnp.zeros((bs, 1, 64), dtype=jnp.float32),
                "movement": mov_t,
                "action": prev_u16[:, None, :],
                "sym_tokens": z_tok,
                "lang_tokens": lt,
                "modality_mask": mm_t,
                "embodiment_id": jnp.zeros((bs,), dtype=jnp.int32),
                "valid": jnp.ones((bs, 1), dtype=bool),
            }
            z_top, f300, _, nxt_c = forward_x3_world_model(p, spec, b1, streaming_carry=sc_c)
            z_sym, z_task = fork_x3_streams(z_top)
            zn1, lv1 = predict_x3_latent_step(p, spec, z_task, prev_u16[:, None, :])
            hds = decode_x3_any_to_any_heads(p, spec, z_sym, z_task, zn1, lv1, f300, b1, decode_480p_frames=False, streaming_carry=sc_c)
            nxt_c = {**nxt_c, "prev_z_next": hds["next_prev_z_next"], "prev_log_var": hds["next_prev_log_var"]}
            vee = hds["vis_ee_xy"][:, 0]
            vobj = hds["vis_obj_xy"][:, 0]
            pg = hds["pred_goal_xy"][:, 0]

            u_sweep = forced_retention_sweep_action(st["ee_xy"], t_i, action_dim=2)
            u_mppi, cwp = _plan_x3_subgoal_mppi_tick(
                p, spec, z_task, vee, vobj, pg, jnp.zeros((bs, 8, 8)), prev_wp, t_i > 88, hds["pred_action"][:, 0, 2], pk, use_jepa_rollout=True
            )
            u2 = jnp.where(is_act, u_mppi, u_sweep)
            sn, _ = step_unseen_layout_world(st, u2)
            is_r = t_i == 87
            sn = {
                **sn,
                "ee_xy": jnp.where(is_r, ree, sn["ee_xy"]),
                "obj_xy": jnp.where(is_r, robj, sn["obj_xy"]),
                "ee_vel": jnp.where(is_r, 0.0, sn["ee_vel"]),
                "obj_vel": jnp.where(is_r, 0.0, sn["obj_vel"]),
            }
            u16_n = jnp.concatenate([u2, jnp.zeros((bs, 14))], axis=-1)
            obs_sig = jnp.mean(vid_t.astype(jnp.float32), axis=(1, 2, 3, 4)) + jnp.sum(mov_t[:, 0, :6], axis=-1)
            return (sn, nxt_c, u16_n, jnp.where(is_act[:, None] if jnp.ndim(is_act) > 0 else is_act, cwp, pg), sn["ee_xy"], rk), (
                sn["obj_xy"],
                pg,
                obs_sig,
            )

        (_, _, _, _, _, _), (obj_h, pg_h, obs_h) = jax.lax.scan(
            _step_t160, (st0, carry0, u0, ga, st0["ee_xy"], jax.random.key(seed + 41)), jnp.arange(total_t)
        )
        return jnp.swapaxes(obj_h, 0, 1), jnp.swapaxes(pg_h, 0, 1), jnp.swapaxes(obs_h, 0, 1)

    obj_b_l, obj_bp_l, pg_b_l, obs_diff_l = [], [], [], []
    cbs = min(chunk_bs, n_worlds)
    for i0 in range(0, n_worlds, cbs):
        sub_sw = jax.tree.map(lambda x: x[i0 : i0 + cbs], sw_all)
        ob_b, pgb, os_b = jax.device_get(_run_pair_sub(params, sub_sw, sub_sw["targ_b"]))
        ob_bp, _, os_bp = jax.device_get(_run_pair_sub(params, sub_sw, sub_sw["targ_b_prime"]))
        obj_b_l.append(np.asarray(ob_b))
        obj_bp_l.append(np.asarray(ob_bp))
        pg_b_l.append(np.asarray(pgb))
        obs_diff_l.append(float(np.max(np.abs(np.asarray(os_b)[:, 20:89] - np.asarray(os_bp)[:, 20:89]))))

    obj_b = np.concatenate(obj_b_l, 0)
    obj_bp = np.concatenate(obj_bp_l, 0)
    pg_b = np.concatenate(pg_b_l, 0)
    gb_np = np.asarray(sw_all["targ_b"])
    gbp_np = np.asarray(sw_all["targ_b_prime"])
    ga_np = np.asarray(sw_all["targ_a"])
    fd_np = np.asarray(sw_all["false_distractor_xy"])

    succ_b = np.min(np.linalg.norm(obj_b[:, -16:] - gb_np[:, None, :], axis=-1), axis=-1) < 0.24
    succ_bp = np.min(np.linalg.norm(obj_bp[:, -16:] - gbp_np[:, None, :], axis=-1), axis=-1) < 0.24
    paired = (succ_b & succ_bp).astype(np.float32)
    stale_a = np.min(np.linalg.norm(obj_b[:, -16:] - ga_np[:, None, :], axis=-1), axis=-1) < 0.24
    cap_fd = np.min(np.linalg.norm(obj_b[:, -16:] - fd_np[:, None, :], axis=-1), axis=-1) < 0.24
    latch_rmse = np.sqrt(np.mean((pg_b[:, 90:] - gb_np[:, None, :]) ** 2))

    return {
        "paired_causal_switch_success": float(np.mean(paired)),
        "paired_causal_switch_vec": [float(x) for x in paired],
        "instructed_goal_b_success": float(np.mean(succ_b)),
        "counterfactual_goal_b_prime_success": float(np.mean(succ_bp)),
        "stale_goal_a_capture": float(np.mean(stale_a)),
        "false_distractor_capture": float(np.mean(cap_fd)),
        "post_reset_goal_latch_rmse": float(latch_rmse),
        "obs_t20_88_max_abs_diff": float(max(obs_diff_l) if obs_diff_l else 0.0),
    }


def eval_x3_spoken_xarm6_grounding(
    params: dict[str, jax.Array],
    spec: X3Spec,
    eval_npz_path: str | Path | None,
    *,
    train_ref_npz_path: str | Path | None = None,
    split: str = "dev",
    max_episodes: int = 284,
    chunk_bs: int = 8,
    smoke: bool = False,
) -> dict[str, Any]:
    """Evaluate speaker-disjoint two-utterance `xArm6` v2 shard + symmetric 2-block `480p` localization & `beta_aud`."""
    if not eval_npz_path or not Path(eval_npz_path).is_file():
        if not smoke:
            raise FileNotFoundError(f"Non-smoke Study X3 evaluation requires verified xArm6 v2 shard: {eval_npz_path!r}")
        return {
            "available": False,
            "joint_disc_acc": 0.0,
            "spoken_audio_only_disc_acc": 0.0,
            "text_only_disc_acc": 0.0,
            "spoken_audio_only_scene_color_acc": 0.0,
            "spoken_audio_only_scene_color_lift_vs_shuffle": 0.0,
            "joint_scene_color_acc": 0.0,
            "paired_180_flip_dual_acc": 0.0,
            "scene_cluster_ids": [],
            "spoken_disc_vec": [],
            "spoken_color_lift_vec": [],
        }

    p = Path(eval_npz_path)
    with np.load(p, allow_pickle=False) as d:
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
        missing = [pfx + k for k in required_keys if (pfx + k) not in d.files]
        if missing:
            raise KeyError(f"Wave 8A v2 robot shard {p} missing required fields: {missing}")

        n_tot = int(d[pfx + "video"].shape[0])
        n = min(n_tot, max_episodes)
        vid64 = np.asarray(d[pfx + "video"][:n], dtype=np.uint8)
        aud = np.asarray(d[pfx + "spoken_audio"][:n], dtype=np.float32)
        waud = np.asarray(d[pfx + "wrong_spoken_audio"][:n], dtype=np.float32)
        wcaud = np.asarray(d[pfx + "wrong_color_spoken_audio"][:n], dtype=np.float32)
        ee = np.asarray(d[pfx + "ee_xy"][:n], dtype=np.float32)
        lengths = np.clip(np.asarray(d[pfx + "length"][:n], dtype=np.int32), 2, 32)
        tok16_tr = np.asarray(d[pfx + "true_text_ids"][:n], dtype=np.int32)
        tok16_wr = np.asarray(d[pfx + "wrong_text_ids"][:n], dtype=np.int32)
        tok16_wc = np.asarray(d[pfx + "wrong_color_text_ids"][:n], dtype=np.int32)
        tb_xy = np.asarray(d[pfx + "target_block_xy"][:n], dtype=np.float32)
        db_xy = np.asarray(d[pfx + "distractor_block_xy"][:n], dtype=np.float32)
        col_ids = np.asarray(d[pfx + "color_id"][:n], dtype=np.int32)
        dist_col_ids = np.asarray(d[pfx + "distractor_color_id"][:n], dtype=np.int32)
        cluster_ids = np.asarray(d[pfx + "scene_cluster_id"][:n], dtype=np.int64)

    t_r = int(aud.shape[1])
    txt = encode_wave8a_two_utterance_tokens_32x4(tok16_tr, t_r)
    wtxt = encode_wave8a_two_utterance_tokens_32x4(tok16_wr, t_r)
    wctxt = encode_wave8a_two_utterance_tokens_32x4(tok16_wc, t_r)

    ee_vel = np.zeros_like(ee)
    ee_vel[:, 1:] = ee[:, 1:] - ee[:, :-1]
    moving = (np.linalg.norm(ee_vel, axis=-1) > 0.015).astype(np.float32)
    prop0 = np.concatenate(
        [ee[:, :1], ee[:, :1], ee_vel[:, :1], np.zeros((n, 1, 1), dtype=np.float32), moving[:, :1, None], np.zeros((n, 1, 8), dtype=np.float32)],
        axis=-1,
    )
    end_raw = np.array([ee[bi, lengths[bi] - 1] for bi in range(n)], dtype=np.float32)
    disp_xy = np.clip((end_raw - ee[:, 0]) * 1.25, -0.95, 0.95).astype(np.float32)

    # Disjoint `scene_cluster_id` permutation (`cluster_ids[shuf_perm[i]] != cluster_ids[i]`)
    shuf_perm = _disjoint_group_perm(cluster_ids, same_source_ids=None)
    shuf_v = vid64[shuf_perm]

    # Symmetric `mu_train` color-prior baseline
    prior_acc = 0.5
    if train_ref_npz_path and Path(train_ref_npz_path).is_file():
        with np.load(Path(train_ref_npz_path), allow_pickle=False) as td:
            if "train_target_block_xy" in td.files and "train_color_id" in td.files:
                tr_xy = np.asarray(td["train_target_block_xy"], dtype=np.float32)
                tr_c = np.asarray(td["train_color_id"], dtype=np.int32)
                mu_c = {c: np.mean(tr_xy[tr_c == c], axis=0) for c in set(int(x) for x in tr_c.tolist())}
                pr_scores = []
                for i in range(n):
                    mc_t = mu_c.get(int(col_ids[i]), np.zeros((2,), dtype=np.float32))
                    mc_w = mu_c.get(int(dist_col_ids[i]), np.zeros((2,), dtype=np.float32))
                    sym_s, _ = _symmetric_two_block_grounding_score(mc_t, mc_w, tb_xy[i], db_xy[i])
                    pr_scores.append(sym_s)
                prior_acc = float(np.mean(pr_scores))

    @functools.partial(jax.jit, static_argnums=(5, 6, 7))
    def _eval_rob_sub(
        p_in: dict[str, jax.Array],
        v0_64: jax.Array,
        p0_16: jax.Array,
        a_in: jax.Array,
        t_in: jax.Array,
        keep_a: float,
        keep_t: float,
        flip_180: bool = False,
    ):
        bs = v0_64.shape[0]
        v_use = jnp.flip(v0_64, axis=(2, 3)) if flip_180 else v0_64
        v480 = jax.image.resize(v_use[:, 0].astype(jnp.float32), (bs, 480, 640, 3), method="linear").astype(jnp.uint8)[:, None]
        vid_r = jnp.broadcast_to(v480, (bs, t_r, 480, 640, 3))
        mov_r = jnp.broadcast_to(jnp.pad(p0_16, ((0, 0), (0, 0), (0, 16))), (bs, t_r, 32))
        has_lt = jnp.any(t_in > 0, axis=-1).astype(jnp.float32)
        mm = jnp.ones((bs, t_r, 4), dtype=jnp.float32).at[:, :, 0].set(keep_a).at[:, :, 3].set(keep_t * has_lt)
        rb = {
            "video": vid_r,
            "audio": a_in * keep_a,
            "movement": mov_r,
            "action": jnp.zeros((bs, t_r, 16), dtype=jnp.float32),
            "sym_tokens": jnp.zeros((bs, t_r, 4), dtype=jnp.int32),
            "lang_tokens": jnp.where(keep_t > 0.5, t_in, 0),
            "modality_mask": mm,
            "embodiment_id": jnp.full((bs,), 7, dtype=jnp.int32),
            "valid": jnp.ones((bs, t_r), dtype=bool),
        }
        zt, f300, ba, _ = forward_x3_world_model(p_in, spec, rb)
        _, ztk = fork_x3_streams(zt)
        rg_dir = jnp.mean(ztk[:, -1, 57:59, :].astype(jnp.float32), axis=1)
        pg = jnp.tanh(rg_dir @ p_in["head_goal_xy_w"].astype(jnp.float32) + p_in["head_goal_xy_b"].astype(jnp.float32))
        pb = decode_x3_target_block_xy(p_in, ztk[:, -1:], f300[:, -1:])[:, 0]
        return pg, pb, ba[:, :, 0]

    def _run_cond(a_arr, t_arr, ka: float, kt: float, v_override=None, flip: bool = False):
        v_src = vid64[:, :1] if v_override is None else v_override[:, :1]
        pg_l, pb_l, ba_l = [], [], []
        cbs = min(chunk_bs, n)
        for i0 in range(0, n, cbs):
            sl = slice(i0, i0 + cbs)
            pg, pb, ba = jax.device_get(
                _eval_rob_sub(params, jnp.asarray(v_src[sl]), jnp.asarray(prop0[sl]), jnp.asarray(a_arr[sl]), jnp.asarray(t_arr[sl]), ka, kt, flip)
            )
            pg_l.append(np.asarray(pg))
            pb_l.append(np.asarray(pb))
            ba_l.append(np.asarray(ba))
        return np.concatenate(pg_l, 0), np.concatenate(pb_l, 0), np.concatenate(ba_l, 0)

    out_modes: dict[str, Any] = {}
    speech_beta, silence_beta = 0.0, 0.0

    for mname, ka, kt in (("joint_audio_text", 1.0, 1.0), ("spoken_audio_only", 1.0, 0.0), ("text_only", 0.0, 1.0)):
        pg_tr, pb_tr, ba_tr = _run_cond(aud, txt, ka, kt)
        pg_wr, _, _ = _run_cond(waud, wtxt, ka, kt)
        _, pb_wc, _ = _run_cond(wcaud, wctxt, ka, kt)
        _, pb_shuf_tr, _ = _run_cond(aud, txt, ka, kt, v_override=shuf_v)
        _, pb_shuf_wc, _ = _run_cond(wcaud, wctxt, ka, kt, v_override=shuf_v)
        _, pb_flip_tr, _ = _run_cond(aud, txt, ka, kt, flip=True)
        _, pb_flip_wc, _ = _run_cond(wcaud, wctxt, ka, kt, flip=True)

        disc_vec = (np.sum((pg_tr - disp_xy) ** 2, axis=-1) < np.sum((pg_wr - disp_xy) ** 2, axis=-1)).astype(np.float32)
        sym_col_vec, sym_shuf_vec, dual_flip_vec = [], [], []
        for i in range(n):
            s_orig, d_orig = _symmetric_two_block_grounding_score(pb_tr[i], pb_wc[i], tb_xy[i], db_xy[i])
            s_shuf, _ = _symmetric_two_block_grounding_score(pb_shuf_tr[i], pb_shuf_wc[i], tb_xy[i], db_xy[i])
            _, d_flip = _symmetric_two_block_grounding_score(pb_flip_tr[i], pb_flip_wc[i], -tb_xy[i], -db_xy[i])
            sym_col_vec.append(s_orig)
            sym_shuf_vec.append(s_shuf)
            dual_flip_vec.append(float(d_orig > 0.5 and d_flip > 0.5))

        sc_arr = np.asarray(sym_col_vec, dtype=np.float64)
        sh_arr = np.asarray(sym_shuf_vec, dtype=np.float64)
        df_arr = np.asarray(dual_flip_vec, dtype=np.float64)

        if mname == "spoken_audio_only":
            max_b = np.max(aud, axis=-1)
            sp_m = max_b > -0.95
            si_m = max_b < -1.20
            speech_beta = float(np.mean(ba_tr[sp_m])) if np.any(sp_m) else float(np.mean(ba_tr))
            silence_beta = float(np.mean(ba_tr[si_m])) if np.any(si_m) else 0.0

        out_modes[mname] = {
            "disc_acc": float(np.mean(disc_vec)),
            "disc_vec": [float(x) for x in disc_vec],
            "scene_color_acc": float(np.mean(sc_arr)),
            "scene_color_vec": [float(x) for x in sc_arr],
            "disjoint_scene_shuffle_acc": float(np.mean(sh_arr)),
            "scene_color_lift_vs_shuffle": float(np.mean(sc_arr - sh_arr)),
            "scene_color_lift_vec": [float(x) for x in (sc_arr - sh_arr)],
            "paired_180_flip_dual_acc": float(np.mean(df_arr)),
            "paired_180_flip_dual_vec": [float(x) for x in df_arr],
        }

    return {
        "available": True,
        "n_episodes": n,
        "scene_cluster_ids": [int(x) for x in cluster_ids],
        "train_color_prior_baseline": prior_acc,
        "beta_aud_active_speech_mean": speech_beta,
        "beta_aud_trailing_silence_mean": silence_beta,
        "by_modality": out_modes,
        "joint_disc_acc": out_modes["joint_audio_text"]["disc_acc"],
        "spoken_audio_only_disc_acc": out_modes["spoken_audio_only"]["disc_acc"],
        "spoken_disc_vec": out_modes["spoken_audio_only"]["disc_vec"],
        "text_only_disc_acc": out_modes["text_only"]["disc_acc"],
        "joint_scene_color_acc": out_modes["joint_audio_text"]["scene_color_acc"],
        "spoken_audio_only_scene_color_acc": out_modes["spoken_audio_only"]["scene_color_acc"],
        "spoken_audio_only_scene_color_lift_vs_shuffle": out_modes["spoken_audio_only"]["scene_color_lift_vs_shuffle"],
        "spoken_color_lift_vec": out_modes["spoken_audio_only"]["scene_color_lift_vec"],
        "paired_180_flip_dual_acc": out_modes["joint_audio_text"]["paired_180_flip_dual_acc"],
    }


def run_x3_full_evaluation(
    params: dict[str, jax.Array],
    spec: X3Spec,
    *,
    dev_480p_path: str | Path | None = None,
    vault_480p_path: str | Path | None = None,
    grounded_robot_path: str | Path | None = None,
    vault_robot_path: str | Path | None = None,
    vocab_path: str | Path | None = None,
    showcase_out_npz: str | Path | None = None,
    smoke: bool = False,
    progress_cb: Any = None,
) -> tuple[dict[str, Any], str | None, int]:
    """Execute leak-free Panels X3-1, X3-2, & X3-3; fail closed on missing vault/robot shards when `smoke=False`."""
    t_stage0 = time.monotonic()

    def _stage(name: str) -> None:
        cur_gb, peak_gb = _host_rss_gb()
        print(
            f"[EVAL X3] stage={name} t={time.monotonic() - t_stage0:.1f}s host_rss={cur_gb:.2f}GB peak={peak_gb:.2f}GB",
            flush=True,
        )
        if progress_cb is not None:
            progress_cb(name)

    if not smoke:
        for role_name, role_p in (
            ("dev_480p_path", dev_480p_path),
            ("vault_480p_path", vault_480p_path),
            ("grounded_robot_path", grounded_robot_path),
            ("vault_robot_path", vault_robot_path),
        ):
            if not role_p or not Path(role_p).is_file():
                raise ValueError(f"Non-smoke Study X3 evaluation requires explicit verified {role_name}: {role_p!r}")

    inv_vocab = _load_id_to_token_map(vocab_path)
    dev_max_clips = 8 if smoke else 96
    vault_max_clips = 8 if smoke else None  # Evaluate 100% of vault clips when smoke=False!
    n_mppi = 8 if smoke else 64
    mppi_h = 24 if smoke else 116
    n_t160 = 4 if smoke else 32
    ls_ticks = 64 if smoke else 1024
    ov_eps = 4 if smoke else 32
    ov_horizons = (64, 128) if smoke else (64, 256, 1024)
    ov_ks = (1, 4) if smoke else (1, 2, 4, 8, 16)
    n_ood = 8 if smoke else 24

    _stage("start")
    ood_res = eval_x3_unseen_ood_480p(params, spec, n_worlds=n_ood, chunk_bs=4)
    _stage("ood_done")
    dev_clips = _load_eval_480p_clips(dev_480p_path, max_clips=dev_max_clips, seed=302411, smoke=smoke)
    comp_dev, _ = eval_x3_any_to_any_composition(
        params, spec, dev_clips, inv_vocab=inv_vocab, ood_panel=ood_res, chunk_bs=4,
        collect_showcase=False, strict_within_source=(not smoke),
    )
    del dev_clips
    gc.collect()
    _stage("any_to_any_dev_done")
    vault_clips = _load_eval_480p_clips(vault_480p_path, max_clips=vault_max_clips, seed=302412, smoke=smoke)
    _stage(f"vault_clips_loaded_n{int(vault_clips['video'].shape[0])}")
    comp_vault, sc_vault = eval_x3_any_to_any_composition(
        params, spec, vault_clips, inv_vocab=inv_vocab, ood_panel=ood_res, chunk_bs=4,
        collect_showcase=bool(showcase_out_npz), strict_within_source=(not smoke),
    )
    del vault_clips
    gc.collect()
    _stage("any_to_any_vault_done")

    ls_res, sc_ls = eval_x3_livestream_T1024_and_overwrite(
        params,
        spec,
        livestream_ticks=ls_ticks,
        overwrite_episodes=ov_eps,
        horizons=ov_horizons,
        k_items_list=ov_ks,
    )
    _stage("livestream_done")
    mppi_res = eval_x3_unseen_layout_mppi_480p(params, spec, n_worlds=n_mppi, horizon_ticks=mppi_h, chunk_bs=8)
    _stage("mppi_done")
    t160_res = eval_x3_T160_paired_switch_480p(params, spec, n_worlds=n_t160, chunk_bs=8)
    _stage("t160_done")
    rob_dev = eval_x3_spoken_xarm6_grounding(
        params, spec, grounded_robot_path, train_ref_npz_path=grounded_robot_path, split="dev", max_episodes=(16 if smoke else 128), smoke=smoke
    )
    _stage("xarm6_dev_done")
    rob_vault = eval_x3_spoken_xarm6_grounding(
        params, spec, vault_robot_path, train_ref_npz_path=grounded_robot_path, split="vault", max_episodes=(16 if smoke else 284), smoke=smoke
    )
    _stage("xarm6_vault_done")

    ov_dict = ls_res["multi_item_overwrite_capacity"]
    t_max_key = f"K1_T{ov_horizons[-1]}_overwritten_acc"
    k_max_key = f"K{ov_ks[-1]}_T{ov_horizons[-1]}_acc"

    summary = {
        "dyn_visual_assoc_T1024": float(ov_dict.get(t_max_key, 0.0)),
        "multi_item_Kmax_Tmax_acc": float(ov_dict.get(k_max_key, 0.0)),
        "T160_paired_causal_switch": float(t160_res["paired_causal_switch_success"]),
        "unseen_mppi_480p": float(mppi_res["geometry_free_mppi_success"]),
        "no_jepa_subgoal_480p": float(mppi_res["no_jepa_subgoal_success"]),
        "mppi_over_no_jepa_gain_pp": float(mppi_res["mppi_over_no_jepa_gain_pp"]),
        "unseen_dynamics_shift_mppi": float(mppi_res["by_unseen_family"]["dynamics_shift"]["mppi_success"]),
        "sysid_dynamics_shift_gain_pp": float(mppi_res["sysid_dynamics_shift_gain_pp"]),
        "sysid_over_swapped_dynamics_shift_gain_pp": float(mppi_res["sysid_over_swapped_dynamics_shift_gain_pp"]),
        "unseen_ood_auroc_480p": float(ood_res["mean_unseen_ood_corruption_auroc"]),
        "commit_selective_routing_acc": float(ood_res["commit_selective_routing_acc"]),
        "commit_binary_ece": float(ood_res["commit_binary_ece"]),
        "vault_spoken_audio_only_disc": float(rob_vault["spoken_audio_only_disc_acc"]),
        "vault_spoken_audio_only_scene_color": float(rob_vault["spoken_audio_only_scene_color_acc"]),
        "vault_spoken_audio_only_color_lift": float(rob_vault["spoken_audio_only_scene_color_lift_vs_shuffle"]),
        "vault_joint_scene_color": float(rob_vault["joint_scene_color_acc"]),
        "vault_paired_180_flip_dual": float(rob_vault["paired_180_flip_dual_acc"]),
        "vault_480p_video_psnr_db": float(comp_vault["dir2_text_ctx_to_480p_video"]["psnr_480p_db"]),
        "vault_480p_video_ssim": float(comp_vault["dir2_text_ctx_to_480p_video"]["ssim_480p"]),
        "vault_480p_edge_rmse": float(comp_vault["dir2_text_ctx_to_480p_video"]["high_freq_edge_rmse_480p"]),
        "vault_vid2aud_lsd_db": float(comp_vault["dir3_video_to_16khz_audio"]["log_spectral_distance_db"]),
        "vault_va2txt_top1_acc": float(comp_vault["dir1_video_audio_to_text"]["top1_token_acc"]),
        "vault_txt_vid2mov_ade": float(comp_vault["dir4_text_video_to_3d_movement"]["ade_1step_l2"]),
        "vault_av2act_mse": float(comp_vault["dir5_av_text_to_robot_actions"]["action_mse"]),
        "vault_positive_shuffle_lift_dirs": int(comp_vault["positive_shuffle_lift_directions"]),
        "livestream_sustained_fps": float(ls_res["sustained_fps"]),
        "livestream_ttft_ms": float(ls_res["ttft_ms"]),
        "livestream_p50_ms": float(ls_res["latency_p50_ms"]),
        "livestream_p99_ms": float(ls_res["latency_p99_ms"]),
        "livestream_state_growth_bytes": int(ls_res["state_growth_bytes"]),
    }

    showcase_sha = None
    showcase_bytes = 0
    if showcase_out_npz:
        sp_p = Path(showcase_out_npz)
        sp_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sp_p, **sc_vault, **sc_ls)
        showcase_sha = sha256_file(sp_p)
        showcase_bytes = int(sp_p.stat().st_size)

    return (
        {
            "summary": summary,
            "any_to_any_dev": comp_dev,
            "any_to_any_vault": comp_vault,
            "livestream_T1024": ls_res,
            "unseen_layout_mppi_480p": mppi_res,
            "beyond_window_T160_switch_480p": t160_res,
            "unseen_ood_480p": ood_res,
            "spoken_xarm6_dev": rob_dev,
            "spoken_xarm6_vault": rob_vault,
        },
        showcase_sha,
        showcase_bytes,
    )
