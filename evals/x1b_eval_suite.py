"""Frozen 7-panel evaluation suite & paired bootstrap CI statistics for Study X1B.

Engineering invariant:
    Every @jax.jit function takes `p` (params) as an explicit traced argument rather than closing
    over `params`, preventing multi-GB HLO constant serialization at 1.008B parameter scale.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from data.x1b_multimodal import (
    ASSOC_NUM_VALS,
    ASSOC_VAL_OFFSET,
    ROBOT_SCALE_MM,
    RealMultimodalBank,
    RealRobotBank,
    generate_synthetic_world_batch,
    render_scene_64x64,
    state_to_proprio,
    step_physics_world,
)
from evals.x1b_mpc import plan_latent_cem_mpc, plan_random_shooting_g8
from models.x1b_commit import route_proposals_with_commit_gate
from models.x1b_jepa import decode_world_heads, l2_normalize, predict_latent_step, sigreg_loss_and_pr
from models.x1b_world import (
    SLOT_PROPRIO,
    SLOTS_AUDIO,
    SLOTS_VISUAL,
    X1BSpec,
    forward_world_model,
    init_streaming_memories,
)


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray | None = None,
    *,
    n_boot: int = 4000,
    seed: int = 20261111,
) -> dict[str, float]:
    """Compute mean and 95% bootstrap CI for values_a (or paired difference values_a - values_b)."""
    arr = np.asarray(values_a, dtype=np.float64)
    if values_b is not None:
        arr = arr - np.asarray(values_b, dtype=np.float64)
    arr = arr.reshape(-1)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.mean(arr[idx], axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact rank-based AUROC without external sklearn dependency."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1) > 0.5
    n_pos = int(np.sum(y))
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    sum_pos_ranks = np.sum(ranks[y])
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def eval_real_multimodal_and_text(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    real_test_bank: RealMultimodalBank | None,
    *,
    use_residual: bool = True,
) -> dict[str, Any]:
    """Panel 1: Real GreatestHits-64 + STARSS23 + PyMunk-64 + FineWeb-Edu evaluation."""
    if real_test_bank is None:
        return {"available": False}

    n_eval = real_test_bank.size
    bs = 16
    vis_embs = []
    aud_embs = []
    av_flags = []
    contact_scores = []
    contact_labels = []
    text_nlls = []
    text_accs = []
    state_sq_model = []
    state_sq_persist = []
    prs = []

    @jax.jit
    def _eval_batch(p: dict[str, jax.Array], b_in: dict[str, jax.Array]) -> dict[str, jax.Array]:
        z, _ = forward_world_model(p, spec, b_in)
        u_exec = jnp.concatenate([b_in["action"][:, 1:], b_in["action"][:, -1:]], axis=1)
        z_next, lv_next = predict_latent_step(p, spec, z, u_exec, horizon_idx=0, residual=use_residual)
        heads = decode_world_heads(
            p, spec, z, z_next, lv_next, b_in["goal"], obs_proprio=b_in["proprio"], use_residual=use_residual
        )
        vp = jnp.mean(z[:, :, SLOTS_VISUAL, :], axis=2).astype(jnp.float32) @ p["align_vis_w"].astype(jnp.float32)
        ap = jnp.mean(z[:, :, SLOTS_AUDIO, :], axis=2).astype(jnp.float32) @ p["align_aud_w"].astype(jnp.float32)
        _, pr = sigreg_loss_and_pr(vp.reshape(-1, vp.shape[-1]))
        return {
            "v_clip": l2_normalize(jnp.mean(vp, axis=1)),
            "a_clip": l2_normalize(jnp.mean(ap, axis=1)),
            "contact_prob": jax.nn.sigmoid(heads["contact_logits"][:, :, 0]),
            "proprio_next": heads["proprio_next"],
            "text_logits": heads["text_logits"],
            "pr": pr,
        }

    for start in range(0, n_eval - bs + 1, bs):
        idx = np.arange(start, start + bs)
        raw = {
            "video": real_test_bank.video[idx],
            "audio": real_test_bank.audio[idx],
            "proprio": real_test_bank.proprio[idx],
            "text": real_test_bank.text[idx],
            "target_text": real_test_bank.target_text[idx],
            "action": real_test_bank.action[idx],
            "contacts": real_test_bank.contacts[idx],
            "goal": real_test_bank.goal[idx],
            "modality_mask": real_test_bank.mod_mask[idx],
            "valid": np.ones((bs, real_test_bank.video.shape[1]), dtype=bool),
        }
        out = jax.device_get(_eval_batch(params, {k: jnp.asarray(v) for k, v in raw.items()}))
        vis_embs.append(out["v_clip"])
        aud_embs.append(out["a_clip"])
        av_flags.append(real_test_bank.has_av[idx])
        prs.append(float(out["pr"]))

        av_mask = real_test_bank.has_av[idx] > 0.5
        if np.any(av_mask):
            contact_scores.append(out["contact_prob"][av_mask])
            contact_labels.append(raw["contacts"][av_mask, :, 0])

            p_true_next = raw["proprio"][av_mask, 1:, :6]
            p_persist = raw["proprio"][av_mask, :-1, :6]
            p_pred = out["proprio_next"][av_mask, :-1, :6]
            state_sq_model.append(np.mean((p_pred - p_true_next) ** 2, axis=(1, 2)))
            state_sq_persist.append(np.mean((p_persist - p_true_next) ** 2, axis=(1, 2)))

        txt_active = raw["modality_mask"][:, 0, 3] > 0.5
        if np.any(txt_active):
            t_targ = raw["target_text"][txt_active]
            t_logits = out["text_logits"][txt_active].astype(np.float64)
            t_max = np.max(t_logits, axis=-1, keepdims=True)
            lse = t_max[..., 0] + np.log(np.sum(np.exp(t_logits - t_max), axis=-1))
            chosen = np.take_along_axis(t_logits, np.clip(t_targ, 0, spec.vocab_size - 1)[..., None], axis=-1)[..., 0]
            nll = lse - chosen
            mask = t_targ > 0
            if np.any(mask):
                text_nlls.extend((np.sum(nll * mask, axis=-1) / np.maximum(np.sum(mask, axis=-1), 1)).tolist())
                last_pred = np.argmax(t_logits[:, -1, :], axis=-1)
                text_accs.extend((last_pred == t_targ[:, -1]).astype(np.float32).tolist())

    v_mat = np.concatenate(vis_embs, axis=0)
    a_mat = np.concatenate(aud_embs, axis=0)
    av_idx = np.where(np.concatenate(av_flags, axis=0) > 0.5)[0]
    if len(av_idx) >= 8:
        pool = av_idx[: min(64, len(av_idx))]
        sim = v_mat[pool] @ a_mat[pool].T
        ranks = np.argsort(-sim, axis=-1)
        r1_vec = (ranks[:, 0] == np.arange(len(pool))).astype(np.float32)
        r5_vec = np.any(ranks[:, :5] == np.arange(len(pool))[:, None], axis=-1).astype(np.float32)
    else:
        r1_vec = np.zeros((1,), dtype=np.float32)
        r5_vec = np.zeros((1,), dtype=np.float32)

    m_rmse = float(np.sqrt(np.mean(np.concatenate(state_sq_model)))) if state_sq_model else 0.0
    p_rmse = float(np.sqrt(np.mean(np.concatenate(state_sq_persist)))) if state_sq_persist else 1.0
    persist_skill = float(1.0 - m_rmse / max(p_rmse, 1e-6))

    mean_nll = float(np.mean(text_nlls)) if text_nlls else 0.0
    return {
        "available": True,
        "av_r1": paired_bootstrap_ci(r1_vec),
        "av_r5": paired_bootstrap_ci(r5_vec),
        "contact_auroc": binary_auroc(np.concatenate(contact_scores), np.concatenate(contact_labels)) if contact_scores else 0.5,
        "real_forecast_rmse": m_rmse,
        "persistence_rmse": p_rmse,
        "persistence_skill": persist_skill,
        "fineweb_nll": mean_nll,
        "fineweb_ppl": float(np.exp(min(mean_nll, 15.0))) if text_nlls else 0.0,
        "fineweb_final_acc": float(np.mean(text_accs)) if text_accs else 0.0,
        "participation_ratio": float(np.mean(prs)),
        "per_item_r1": r1_vec.tolist(),
        "per_item_sq_err": np.concatenate(state_sq_model).tolist() if state_sq_model else [],
    }


def eval_thin_barrier_and_persistence(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    pymunk_panels_path: str | None = None,
    *,
    use_residual: bool = True,
) -> dict[str, Any]:
    """Panel 2: Pre-impact thin-barrier visual detection (64x64 vs 16x16 degraded) & multi-step persistence skill."""
    key = jax.random.key(77101)
    bs = 64
    k1, k2 = jax.random.split(key)
    side = jnp.where(jax.random.bernoulli(k1, 0.5, (bs,)), 1.0, -1.0)
    obj_xy = jnp.stack([side * 0.30, jax.random.uniform(k2, (bs,), minval=-0.22, maxval=0.22)], axis=-1)
    ee_xy = obj_xy + jnp.stack([side * 0.20, jnp.zeros((bs,))], axis=-1)
    targ_xy = jnp.stack([-side * 0.42, obj_xy[:, 1]], axis=-1)
    has_wall = jnp.arange(bs) % 2 == 0  # Exactly 50% thin 1-px central barrier, 50% open!

    st = {
        "ee_xy": ee_xy,
        "obj_xy": obj_xy,
        "obj_vel": jnp.zeros((bs, 2), dtype=jnp.float32),
        "obj_theta": jnp.zeros((bs,), dtype=jnp.float32),
        "has_barrier": has_wall,
        "polarity": jnp.ones((bs,), dtype=jnp.float32),
    }
    u_drive = jnp.concatenate([jnp.stack([-side * 0.85, jnp.zeros((bs,))], axis=-1), jnp.zeros((bs, 2))], axis=-1)

    vids = []
    props = []
    next_props = []
    prev_us = []
    prev_u = jnp.zeros((bs, 4), dtype=jnp.float32)
    prev_con = jnp.zeros((bs, 2), dtype=jnp.float32)
    for _ in range(16):
        vids.append(render_scene_64x64(st["ee_xy"], st["obj_xy"], st["obj_theta"], targ_xy, has_wall, jnp.ones((bs,), bool)))
        props.append(state_to_proprio(st, prev_u[:, :2], prev_con))
        prev_us.append(prev_u)
        st, obs_n = step_physics_world(st, u_drive)
        next_props.append(obs_n["proprio"])
        prev_u = u_drive
        prev_con = obs_n["contacts"]

    vid64 = jnp.stack(vids, axis=1)
    # G8-style 16x16 stride-4 subsampling misses x=0 thin central barrier column completely!
    vid16_sub = vid64[:, :, 2::4, 2::4, :]
    vid16_deg = jnp.repeat(jnp.repeat(vid16_sub, 4, axis=2), 4, axis=3)

    prop_seq = jnp.stack(props, axis=1)
    next_prop_seq = jnp.stack(next_props, axis=1)
    prev_u_seq = jnp.stack(prev_us, axis=1)
    u_exec_seq = jnp.broadcast_to(u_drive[:, None, :], (bs, 16, 4))
    goal = jnp.concatenate([targ_xy - obj_xy, targ_xy], axis=-1)

    # Zero contact bits on pre-impact ticks 0..2 so pre-impact barrier AUROC relies PURELY on visual frames,
    # while preserving physically truthful post-impact contact state for ticks 3..15 forecasting!
    prop_blind = prop_seq.at[:, :3, 10:12].set(0.0)
    eval_inp64 = {
        "video": vid64,
        "audio": jnp.zeros((bs, 16, 64), dtype=jnp.float32) - 0.35,
        "proprio": prop_blind,
        "text": jnp.zeros((bs, 16, 4), dtype=jnp.int32),
        "action": prev_u_seq,
        "modality_mask": jnp.array([[[0.0, 1.0, 1.0, 0.0]]] * bs).repeat(16, axis=1),
        "valid": jnp.ones((bs, 16), dtype=bool),
    }
    eval_inp16 = dict(eval_inp64, video=vid16_deg)

    @jax.jit
    def _run_panel2_single(
        p: dict[str, jax.Array],
        inp_b: dict[str, jax.Array],
        u_seq: jax.Array,
        g_vec: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        z_f, _ = forward_world_model(p, spec, inp_b)
        zn_f, lv_f = predict_latent_step(p, spec, z_f, u_seq, residual=use_residual)
        h_f = decode_world_heads(
            p, spec, z_f, zn_f, lv_f, g_vec, obs_proprio=inp_b["proprio"], use_residual=use_residual
        )

        # 4-step open-loop rollout in latent space from ticks 0..11
        z_anchor = z_f[:, :-4]
        z_r = z_anchor
        for step_k in range(4):
            z_r, _ = predict_latent_step(p, spec, z_r, u_seq[:, step_k : 12 + step_k], residual=use_residual)
        zf_r = z_r.astype(jnp.float32)
        zf_0 = z_anchor.astype(jnp.float32)
        hw = p["head_proprio_w"].astype(jnp.float32)
        hb = p["head_proprio_b"].astype(jnp.float32)
        if use_residual and spec.arch != "legacy_scalar_hybrid":
            delta4 = (
                0.75 * (zf_r[:, :, SLOT_PROPRIO, :] - zf_0[:, :, SLOT_PROPRIO, :])
                + 0.25 * (jnp.mean(zf_r[:, :, SLOTS_VISUAL, :], axis=2) - jnp.mean(zf_0[:, :, SLOTS_VISUAL, :], axis=2))
            ) @ hw
            pred4 = inp_b["proprio"][:, :-4].astype(jnp.float32) + delta4
        else:
            pred4 = (0.75 * zf_r[:, :, SLOT_PROPRIO, :] + 0.25 * jnp.mean(zf_r[:, :, SLOTS_VISUAL, :], axis=2)) @ hw + hb

        return (
            jax.nn.sigmoid(h_f["contact_logits"][:, :3, 1]).mean(axis=1),
            h_f["proprio_next"],
            pred4,
        )

    s64, p_mod1_full, p_mod4_full = jax.device_get(_run_panel2_single(params, eval_inp64, u_exec_seq, goal))
    s16, _, _ = jax.device_get(_run_panel2_single(params, eval_inp16, u_exec_seq, goal))
    p_mod1_j = p_mod1_full[:, :, :4]
    p_mod4_j = p_mod4_full[:, :, :4]
    wall_labels = np.asarray(has_wall, dtype=np.float32)

    p_true1 = np.asarray(next_prop_seq[:, :, :4])
    p_pers1 = np.asarray(prop_seq[:, :, :4])
    rmse_mod1 = float(np.sqrt(np.mean((p_mod1_j - p_true1) ** 2)))
    rmse_pers1 = float(np.sqrt(np.mean((p_pers1 - p_true1) ** 2)))

    p_true4 = np.asarray(prop_seq[:, 4:, :4])
    p_pers4 = np.asarray(prop_seq[:, :-4, :4])
    rmse_mod4 = float(np.sqrt(np.mean((p_mod4_j - p_true4) ** 2)))
    rmse_pers4 = float(np.sqrt(np.mean((p_pers4 - p_true4) ** 2)))

    out = {
        "barrier_auroc_64x64": binary_auroc(s64, wall_labels),
        "barrier_auroc_16x16_subsampled": binary_auroc(s16, wall_labels),
        "step1_rmse": rmse_mod1,
        "step1_persistence_rmse": rmse_pers1,
        "step1_skill_vs_persistence": float(1.0 - rmse_mod1 / max(rmse_pers1, 1e-6)),
        "step4_rmse": rmse_mod4,
        "step4_persistence_rmse": rmse_pers4,
        "step4_skill_vs_persistence": float(1.0 - rmse_mod4 / max(rmse_pers4, 1e-6)),
    }

    # Score held-out G8 PyMunk 64x64 scenes (32 puck_barrier [wall=1] + 32 pusht [wall=0])
    if pymunk_panels_path is not None:
        with np.load(pymunk_panels_path, allow_pickle=False) as pf:
            pv_bar = pf["puck_barrier_video64"][:32]
            pn_bar = pf["puck_barrier_n"][:32]
            pv_open = pf["pusht_video64"][:32]
            pn_open = pf["pusht_n"][:32]
            pv = jnp.asarray(np.concatenate([pv_bar, pv_open], axis=0))
            pn = jnp.asarray(np.concatenate([pn_bar, pn_open], axis=0))
            pbar_lbl = np.concatenate([np.ones((32,), dtype=np.float32), np.zeros((32,), dtype=np.float32)])
            prop16 = jnp.pad(pn, ((0, 0), (0, 0), (0, 10)))

            @jax.jit
            def _score_pymunk(p: dict[str, jax.Array], v_in: jax.Array, pr_in: jax.Array) -> jax.Array:
                zp, _ = forward_world_model(
                    p,
                    spec,
                    {
                        "video": v_in,
                        "audio": jnp.zeros((64, 16, 64), dtype=jnp.float32) - 0.35,
                        "proprio": pr_in,
                        "text": jnp.zeros((64, 16, 4), dtype=jnp.int32),
                        "action": jnp.zeros((64, 16, 4), dtype=jnp.float32),
                        "modality_mask": jnp.array([[[0.0, 1.0, 1.0, 0.0]]] * 64).repeat(16, axis=1),
                        "valid": jnp.ones((64, 16), dtype=bool),
                    },
                )
                znp, lvp = predict_latent_step(p, spec, zp, jnp.zeros((64, 16, 4)), residual=use_residual)
                hp = decode_world_heads(
                    p, spec, zp, znp, lvp, jnp.zeros((64, 4)), obs_proprio=pr_in, use_residual=use_residual
                )
                return jax.nn.sigmoid(hp["contact_logits"][:, :8, 1]).mean(axis=1)

            sp_64 = np.asarray(jax.device_get(_score_pymunk(params, pv, prop16)))
            out["pymunk_puck_barrier_auroc_64"] = binary_auroc(sp_64, pbar_lbl)

    return out


def eval_long_horizon_associative_recall(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    horizons: tuple[int, ...] = (32, 128, 256, 512, 1024),
) -> dict[str, Any]:
    """Panel 3: Associative Key-Value Overwrite Recall from T=32 up to T=1024 (32x OOD = 24,576 slots!)."""
    results: dict[str, Any] = {}
    for h_len in horizons:
        total_n = 32 if h_len <= 128 else 16
        if h_len >= 1024:
            chunk_bs = 1 if spec.d_model >= 1152 else 2
        elif h_len >= 512:
            chunk_bs = 2 if spec.d_model >= 1536 else 4
        elif h_len >= 256:
            chunk_bs = 4 if spec.d_model >= 1536 else 8
        else:
            chunk_bs = 8 if spec.d_model >= 1536 else 16

        @jax.jit
        def _eval_assoc_chunk(
            p: dict[str, jax.Array],
            subkey: jax.Array,
            c_bs: int = chunk_bs,
            seq_t: int = h_len,
        ) -> tuple[jax.Array, jax.Array]:
            b_data = generate_synthetic_world_batch(subkey, c_bs, seq_len=seq_t, vocab_size=spec.vocab_size)
            z, _ = forward_world_model(p, spec, b_data)
            zn, lv = predict_latent_step(p, spec, z, b_data["next_action"])
            heads = decode_world_heads(p, spec, z, zn, lv, b_data["goal"], obs_proprio=b_data["proprio"])
            val_slice = slice(ASSOC_VAL_OFFSET, ASSOC_VAL_OFFSET + ASSOC_NUM_VALS)
            return heads["text_logits"][:, -1, val_slice], b_data["target_text"][:, -1]

        acc_chunks = []
        base_key = jax.random.key(88000 + h_len)
        for c_idx in range(total_n // chunk_bs):
            val_logits, true_val = jax.device_get(_eval_assoc_chunk(params, jax.random.fold_in(base_key, c_idx)))
            pred_val = np.argmax(val_logits, axis=-1) + ASSOC_VAL_OFFSET
            acc_chunks.append((pred_val == true_val).astype(np.float32))
        acc_vec = np.concatenate(acc_chunks, axis=0)
        results[f"T{h_len}"] = paired_bootstrap_ci(acc_vec)
        results[f"T{h_len}_vec"] = acc_vec.tolist()
    return results


def eval_transient_command_retention(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    *,
    use_residual: bool = True,
) -> dict[str, Any]:
    """Panel 4: Continuous Goal Cue vs 0.4s Transient Goal Cue (34-tick closed-loop retention)."""
    bs = 48
    t_ep = 34
    key = jax.random.key(66123)
    k1, k2, k3 = jax.random.split(key, 3)
    side = jnp.where(jax.random.bernoulli(k1, 0.5, (bs,)), 1.0, -1.0)
    obj_init = jnp.stack([side * 0.34, jax.random.uniform(k2, (bs,), minval=-0.30, maxval=0.30)], axis=-1)
    ee_init = jnp.clip(obj_init + jnp.stack([side * 0.20, jnp.zeros((bs,))], axis=-1), -0.88, 0.88)
    gy_sign = jnp.where(jax.random.bernoulli(k3, 0.5, (bs,)), 1.0, -1.0)
    targ_xy = jnp.stack([-side * 0.36, gy_sign * 0.24], axis=-1)
    quad_tok = 4 + (targ_xy[:, 0] >= 0.0).astype(jnp.int32) * 2 + (targ_xy[:, 1] >= 0.0).astype(jnp.int32)
    full_goal = jnp.concatenate([targ_xy - obj_init, targ_xy], axis=-1)

    @jax.jit
    def _roll_full_ep(p: dict[str, jax.Array], continuous_flag: jax.Array) -> jax.Array:
        st0 = {
            "ee_xy": ee_init,
            "obj_xy": obj_init,
            "obj_vel": jnp.zeros((bs, 2), dtype=jnp.float32),
            "obj_theta": jnp.zeros((bs,), dtype=jnp.float32),
            "has_barrier": jnp.zeros((bs,), dtype=bool),
            "polarity": jnp.ones((bs,), dtype=jnp.float32),
        }
        mem0 = init_streaming_memories(spec, bs, cache_len=64)
        u0 = jnp.zeros((bs, 4), dtype=jnp.float32)
        con0 = jnp.zeros((bs, 2), dtype=jnp.float32)
        aud0 = jnp.zeros((bs, 64), dtype=jnp.float32) - 0.35

        def _body(carry, tick_idx):
            st, mem, prev_u, prev_con, prev_aud = carry
            cue_bool = continuous_flag | (tick_idx < 4)
            vid_t = render_scene_64x64(
                st["ee_xy"],
                st["obj_xy"],
                st["obj_theta"],
                targ_xy,
                jnp.zeros((bs,), bool),
                jnp.broadcast_to(cue_bool, (bs,)),
            )
            prop_t = state_to_proprio(st, prev_u[:, :2], prev_con)
            txt_t = jnp.where(
                cue_bool,
                jnp.zeros((bs, 1, 4), dtype=jnp.int32).at[:, 0, 0].set(quad_tok),
                jnp.zeros((bs, 1, 4), dtype=jnp.int32),
            )
            goal_in = jnp.where(cue_bool, full_goal, jnp.zeros((bs, 4), dtype=jnp.float32))
            step_batch = {
                "video": vid_t[:, None],
                "audio": prev_aud[:, None],
                "proprio": prop_t[:, None],
                "text": txt_t,
                "action": prev_u[:, None],
                "modality_mask": jnp.ones((bs, 1, 4), dtype=jnp.float32),
                "valid": jnp.ones((bs, 1), dtype=bool),
            }
            z_t, next_mem = forward_world_model(p, spec, step_batch, memories=mem, offset=tick_idx)
            zn_t, lv_t = predict_latent_step(p, spec, z_t, prev_u[:, None], residual=use_residual)
            h_t = decode_world_heads(
                p, spec, z_t, zn_t, lv_t, goal_in, obs_proprio=prop_t[:, None], use_residual=use_residual
            )
            u_cmd = h_t["action_proposal"][:, 0, :]
            st_next, obs = step_physics_world(st, u_cmd)
            return (st_next, next_mem, u_cmd, obs["contacts"], obs["audio"]), None

        (st_end, _, _, _, _), _ = jax.lax.scan(
            _body, (st0, mem0, u0, con0, aud0), jnp.arange(t_ep, dtype=jnp.int32)
        )
        return st_end["obj_xy"]

    outcomes: dict[str, np.ndarray] = {}
    init_dist = np.asarray(jnp.sqrt(jnp.sum((obj_init - targ_xy) ** 2, axis=-1)))
    for mode in ("continuous", "transient_04s"):
        end_obj = jax.device_get(_roll_full_ep(params, jnp.asarray(mode == "continuous", dtype=bool)))
        final_dist = np.asarray(jnp.sqrt(jnp.sum((end_obj - targ_xy) ** 2, axis=-1)))
        outcomes[mode + "_success"] = (final_dist < 0.25).astype(np.float32)
        outcomes[mode + "_progress"] = np.clip(1.0 - final_dist / np.maximum(init_dist, 1e-4), -1.0, 1.0)

    cont_s = outcomes["continuous_success"]
    tran_s = outcomes["transient_04s_success"]
    return {
        "continuous_success": paired_bootstrap_ci(cont_s),
        "transient_04s_success": paired_bootstrap_ci(tran_s),
        "transient_drop_pp": paired_bootstrap_ci((cont_s - tran_s) * 100.0),
        "continuous_progress": float(np.mean(outcomes["continuous_progress"])),
        "transient_progress": float(np.mean(outcomes["transient_04s_progress"])),
    }


def eval_closed_loop_mpc_and_spoofing(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    *,
    use_residual: bool = True,
) -> dict[str, Any]:
    """Panels 5 & 6: Closed-Loop Manipulation (Latent-CEM-MPC vs G8 Random Shooting vs Open-Loop)
    AND 180-deg Wrong-Camera Spoofing / Hidden-Barrier / Unsupported-Query CommitGate Calibration.
    """
    bs = 96
    t_ep = 38
    key = jax.random.key(55109)
    k1, k2, k_plan = jax.random.split(key, 3)
    side = jnp.where(jax.random.bernoulli(k1, 0.5, (bs,)), 1.0, -1.0)
    obj_init = jnp.stack([side * 0.34, jax.random.uniform(k2, (bs,), minval=-0.28, maxval=0.28)], axis=-1)
    ee_init = jnp.clip(obj_init + jnp.stack([side * 0.20, jnp.zeros((bs,))], axis=-1), -0.88, 0.88)
    targ_xy = jnp.stack([-side * 0.34, obj_init[:, 1] * 0.5], axis=-1)
    has_bar = jnp.arange(bs) % 2 == 0  # Half Thin-Barrier, half open Push-T!
    goal_vec = jnp.concatenate([targ_xy - obj_init, targ_xy], axis=-1)

    planner_results: dict[str, Any] = {}
    chunk_bs = (
        24
        if (spec.arch in ("dense_transformer", "sliding_window_transformer") and spec.d_model >= 1152)
        else bs
    )

    for planner_name in ("latent_cem_mpc", "g8_random_shooting_3", "open_loop_policy"):
        @jax.jit
        def _roll_mpc(
            p: dict[str, jax.Array],
            ee_c: jax.Array,
            obj_c: jax.Array,
            targ_c: jax.Array,
            bar_c: jax.Array,
            goal_c: jax.Array,
            rk_c: jax.Array,
            p_kind: str = planner_name,
        ) -> jax.Array:
            c_n = ee_c.shape[0]
            st0 = {
                "ee_xy": ee_c,
                "obj_xy": obj_c,
                "obj_vel": jnp.zeros((c_n, 2), dtype=jnp.float32),
                "obj_theta": jnp.zeros((c_n,), dtype=jnp.float32),
                "has_barrier": bar_c,
                "polarity": jnp.ones((c_n,), dtype=jnp.float32),
            }
            mem0 = init_streaming_memories(spec, c_n, cache_len=64)
            u0 = jnp.zeros((c_n, 4), dtype=jnp.float32)
            con0 = jnp.zeros((c_n, 2), dtype=jnp.float32)
            aud0 = jnp.zeros((c_n, 64), dtype=jnp.float32) - 0.35

            def _mpc_body(carry, tick_idx):
                st, mem, prev_u, prev_con, prev_aud, rk = carry
                rk, subk = jax.random.split(rk)
                vid_t = render_scene_64x64(
                    st["ee_xy"], st["obj_xy"], st["obj_theta"], targ_c, bar_c, jnp.ones((c_n,), bool)
                )
                prop_t = state_to_proprio(st, prev_u[:, :2], prev_con)
                step_b = {
                    "video": vid_t[:, None],
                    "audio": prev_aud[:, None],
                    "proprio": prop_t[:, None],
                    "text": jnp.zeros((c_n, 1, 4), dtype=jnp.int32),
                    "action": prev_u[:, None],
                    "modality_mask": jnp.ones((c_n, 1, 4), dtype=jnp.float32),
                    "valid": jnp.ones((c_n, 1), dtype=bool),
                }
                z_t, next_mem = forward_world_model(p, spec, step_b, memories=mem, offset=tick_idx)
                if p_kind == "latent_cem_mpc":
                    u_cmd, _ = plan_latent_cem_mpc(
                        p,
                        spec,
                        z_t[:, 0],
                        goal_c,
                        subk,
                        obs_proprio=prop_t,
                        horizon=6,
                        num_candidates=48,
                        num_elites=8,
                        cem_iters=3,
                        residual=use_residual,
                        trust_proprio_only=jnp.zeros((c_n,), dtype=bool),
                    )
                elif p_kind == "g8_random_shooting_3":
                    u_cmd = plan_random_shooting_g8(
                        p,
                        spec,
                        z_t[:, 0],
                        goal_c,
                        subk,
                        obs_proprio=prop_t,
                        num_candidates=3,
                        residual=use_residual,
                    )
                else:
                    zn_t, lv_t = predict_latent_step(p, spec, z_t, prev_u[:, None], residual=use_residual)
                    h_t = decode_world_heads(
                        p, spec, z_t, zn_t, lv_t, goal_c, obs_proprio=prop_t[:, None], use_residual=use_residual
                    )
                    u_cmd = h_t["action_proposal"][:, 0]

                st_next, obs = step_physics_world(st, u_cmd)
                return (st_next, next_mem, u_cmd, obs["contacts"], obs["audio"], rk), None

            (st_end, _, _, _, _, _), _ = jax.lax.scan(
                _mpc_body, (st0, mem0, u0, con0, aud0, rk_c), jnp.arange(t_ep, dtype=jnp.int32)
            )
            return st_end["obj_xy"]

        if chunk_bs == bs:
            end_obj = jax.device_get(_roll_mpc(params, ee_init, obj_init, targ_xy, has_bar, goal_vec, k_plan))
        else:
            obj_parts = []
            for c_idx in range(bs // chunk_bs):
                sl = slice(c_idx * chunk_bs, (c_idx + 1) * chunk_bs)
                obj_parts.append(
                    jax.device_get(
                        _roll_mpc(
                            params,
                            ee_init[sl],
                            obj_init[sl],
                            targ_xy[sl],
                            has_bar[sl],
                            goal_vec[sl],
                            jax.random.fold_in(k_plan, c_idx),
                        )
                    )
                )
            end_obj = np.concatenate(obj_parts, axis=0)
        dist_end = np.asarray(jnp.sqrt(jnp.sum((end_obj - targ_xy) ** 2, axis=-1)))
        succ = (dist_end < 0.25).astype(np.float32)
        planner_results[planner_name] = {
            "success_rate": paired_bootstrap_ci(succ),
            "final_distance": paired_bootstrap_ci(dist_end),
            "pusht_open_success": float(np.mean(succ[~np.asarray(has_bar)])),
            "thin_barrier_success": float(np.mean(succ[np.asarray(has_bar)])),
            "succ_vec": succ.tolist(),
        }

    # Panel 6: Spoofing (G6), Hidden Barrier (G8) & Unsupported Output Audit across CommitGate policies
    @jax.jit
    def _audit_chunk(
        p: dict[str, jax.Array],
        subk: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        ab = generate_synthetic_world_batch(subk, 16, seq_len=24, vocab_size=spec.vocab_size)
        za, _ = forward_world_model(p, spec, ab)
        zna, lva = predict_latent_step(p, spec, za, ab["next_action"], residual=use_residual)
        ha = decode_world_heads(
            p, spec, za, zna, lva, ab["goal"], obs_proprio=ab["proprio"], use_residual=use_residual
        )
        return (
            ab["target_commit"][:, 6:],
            ab["target_action"][:, 6:],
            ha["commit_logits"][:, 6:],
            ha["action_proposal"][:, 6:],
            ha["action_proposal_trusted"][:, 6:],
        )

    tc_list, ue_list, lc_list, af_list, at_list = [], [], [], [], []
    audit_root = jax.random.key(44991)
    for c_i in range(10):
        tc_c, ue_c, lc_c, af_c, at_c = jax.device_get(_audit_chunk(params, jax.random.fold_in(audit_root, c_i)))
        tc_list.append(tc_c)
        ue_list.append(ue_c)
        lc_list.append(lc_c)
        af_list.append(af_c)
        at_list.append(at_c)

    true_commit = np.concatenate(tc_list, axis=0)
    u_expert = np.concatenate(ue_list, axis=0)
    logits_c = jnp.asarray(np.concatenate(lc_list, axis=0))
    act_full = jnp.asarray(np.concatenate(af_list, axis=0))
    act_trust = jnp.asarray(np.concatenate(at_list, axis=0))

    gate_comparison: dict[str, Any] = {}
    for pol in ("learned_innovation", "always_commit", "fixed_threshold"):
        routed = route_proposals_with_commit_gate(logits_c, act_full, act_trust, gate_policy=pol)
        u_comm = np.asarray(routed["committed_action"])
        abstained = np.asarray(routed["abstained"])
        mode_pred = np.asarray(routed["mode_id"])

        clean_mask = true_commit == 0
        spoof_mask = true_commit == 1
        hidden_mask = true_commit == 2
        unsupp_mask = true_commit == 3

        clean_useful = float(
            np.mean((~abstained[clean_mask]) & (np.linalg.norm(u_comm[clean_mask] - u_expert[clean_mask], axis=-1) < 0.55))
        )
        spoof_regret = float(np.mean(np.linalg.norm(u_comm[spoof_mask] - u_expert[spoof_mask], axis=-1)))
        spoof_detect = float(np.mean(mode_pred[spoof_mask] == 1))
        unsupported_claim_rate = float(np.mean(~abstained[unsupp_mask]))
        abstention_rate_overall = float(np.mean(abstained))
        hidden_probe_rate = float(np.mean(mode_pred[hidden_mask] == 2)) if np.any(hidden_mask) else 0.0

        probs = np.asarray(routed["mode_probs"])
        conf = np.max(probs, axis=-1).reshape(-1)
        correct = (mode_pred == true_commit).astype(np.float32).reshape(-1)
        ece = 0.0
        for b_i in range(10):
            lo, hi = b_i / 10.0, (b_i + 1) / 10.0
            in_b = (conf >= lo) & (conf < (hi if b_i < 9 else 1.01))
            if np.any(in_b):
                ece += (np.sum(in_b) / len(conf)) * abs(float(np.mean(conf[in_b]) - np.mean(correct[in_b])))

        gate_comparison[pol] = {
            "clean_usefulness": clean_useful,
            "spoofed_action_regret": spoof_regret,
            "spoof_detection_rate": spoof_detect,
            "hidden_barrier_probe_rate": hidden_probe_rate,
            "unsupported_claim_rate": unsupported_claim_rate,
            "overall_abstention_rate": abstention_rate_overall,
            "mode_accuracy": float(np.mean(correct)),
            "ece": float(ece),
        }

    return {
        "planners": planner_results,
        "commit_gate": gate_comparison,
    }


def eval_long_horizon_memory_mpc(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    *,
    use_residual: bool = True,
) -> dict[str, Any]:
    """Panel 8: Long-Horizon OOD (T=76 = 2.38x T_train) Transient-Goal Thin-Barrier MPC & Mid-Episode Goal-Switch MPC.

    Evaluates 64 held-out physical worlds (32 with 1-px central barrier, 32 open) with ZERO external goal vector:
      1) sustained_transient_barrier_mpc: 0.4s goal cue at ticks 0..3, extinguished for ticks 4..75.
      2) mid_episode_goal_switch_mpc: 0.4s Goal A at ticks 0..3, extinguished 4..35, new 0.4s Goal B overwrite at
         OOD ticks 36..39 (t > T_train=32!), extinguished 40..75; scored on reaching Goal B (and stale Goal-A trap).
    """
    bs = 64
    t_ep = 76
    t_sw = 36
    key = jax.random.key(99107)
    k1, k2, k3, k_plan = jax.random.split(key, 4)
    side = jnp.where(jax.random.bernoulli(k1, 0.5, (bs,)), 1.0, -1.0)
    gy_sign = jnp.where(jax.random.bernoulli(k3, 0.5, (bs,)), 1.0, -1.0)
    y_jit = jax.random.uniform(k2, (bs,), minval=-0.06, maxval=0.06)
    obj_init = jnp.stack([side * 0.36, gy_sign * 0.24 + y_jit], axis=-1)
    ee_init = jnp.clip(obj_init + jnp.stack([side * 0.20, jnp.zeros((bs,))], axis=-1), -0.88, 0.88)
    # Goal A is across the central barrier in (-side, +gy_sign); Goal B is across the barrier in (-side, -gy_sign)
    # so do-nothing (dist ~0.91), barrier-stall (~0.49), stale Goal A (0.58), and Goal A/B midpoint (0.29) ALL score 0%!
    targ_a = jnp.stack([-side * 0.36, gy_sign * 0.28], axis=-1)
    targ_b = jnp.stack([-side * 0.36, (-gy_sign) * 0.30], axis=-1)
    quad_a = 4 + (targ_a[:, 0] >= 0.0).astype(jnp.int32) * 2 + (targ_a[:, 1] >= 0.0).astype(jnp.int32)
    quad_b = 4 + (targ_b[:, 0] >= 0.0).astype(jnp.int32) * 2 + (targ_b[:, 1] >= 0.0).astype(jnp.int32)
    has_bar = jnp.arange(bs) % 2 == 0
    zero_goal = jnp.zeros((bs, 4), dtype=jnp.float32)

    chunk_bs = (
        16
        if (spec.arch in ("dense_transformer", "sliding_window_transformer") and spec.d_model >= 1152)
        else bs
    )

    @jax.jit
    def _roll_memory_mpc(
        p: dict[str, jax.Array],
        switch_flag: jax.Array,
        ee_c: jax.Array,
        obj_c: jax.Array,
        ta_c: jax.Array,
        tb_c: jax.Array,
        qa_c: jax.Array,
        qb_c: jax.Array,
        bar_c: jax.Array,
        rk_c: jax.Array,
    ) -> jax.Array:
        c_n = ee_c.shape[0]
        zg_c = jnp.zeros((c_n, 4), dtype=jnp.float32)
        st0 = {
            "ee_xy": ee_c,
            "obj_xy": obj_c,
            "obj_vel": jnp.zeros((c_n, 2), dtype=jnp.float32),
            "obj_theta": jnp.zeros((c_n,), dtype=jnp.float32),
            "has_barrier": bar_c,
            "polarity": jnp.ones((c_n,), dtype=jnp.float32),
        }
        mem0 = init_streaming_memories(spec, c_n, cache_len=80)
        u0 = jnp.zeros((c_n, 4), dtype=jnp.float32)
        con0 = jnp.zeros((c_n, 2), dtype=jnp.float32)
        aud0 = jnp.zeros((c_n, 64), dtype=jnp.float32) - 0.35

        def _body(carry, tick_idx):
            st, mem, prev_u, prev_con, prev_aud, rk = carry
            rk, subk = jax.random.split(rk)
            in_b_phase = switch_flag & (tick_idx >= t_sw)
            curr_targ = jnp.where(in_b_phase, tb_c, ta_c)
            curr_quad = jnp.where(in_b_phase, qb_c, qa_c)
            cue_on = (tick_idx < 4) | (switch_flag & (tick_idx >= t_sw) & (tick_idx < t_sw + 4))

            vid_t = render_scene_64x64(
                st["ee_xy"],
                st["obj_xy"],
                st["obj_theta"],
                curr_targ,
                bar_c,
                jnp.broadcast_to(cue_on, (c_n,)),
            )
            prop_t = state_to_proprio(st, prev_u[:, :2], prev_con)
            txt_t = jnp.where(
                cue_on,
                jnp.zeros((c_n, 1, 4), dtype=jnp.int32).at[:, 0, 0].set(curr_quad),
                jnp.zeros((c_n, 1, 4), dtype=jnp.int32),
            )
            step_b = {
                "video": vid_t[:, None],
                "audio": prev_aud[:, None],
                "proprio": prop_t[:, None],
                "text": txt_t,
                "action": prev_u[:, None],
                "modality_mask": jnp.ones((c_n, 1, 4), dtype=jnp.float32),
                "valid": jnp.ones((c_n, 1), dtype=bool),
            }
            z_t, next_mem = forward_world_model(p, spec, step_b, memories=mem, offset=tick_idx)
            u_cmd, _ = plan_latent_cem_mpc(
                p,
                spec,
                z_t[:, 0],
                zg_c,
                subk,
                obs_proprio=prop_t,
                horizon=6,
                num_candidates=48,
                num_elites=8,
                cem_iters=3,
                residual=use_residual,
                trust_proprio_only=jnp.zeros((c_n,), dtype=bool),
            )
            st_next, obs = step_physics_world(st, u_cmd)
            return (st_next, next_mem, u_cmd, obs["contacts"], obs["audio"], rk), None

        (st_end, _, _, _, _, _), _ = jax.lax.scan(
            _body, (st0, mem0, u0, con0, aud0, rk_c), jnp.arange(t_ep, dtype=jnp.int32)
        )
        return st_end["obj_xy"]

    def _exec_panel8(sw_bool: bool) -> np.ndarray:
        sw_arr = jnp.asarray(sw_bool, dtype=bool)
        if chunk_bs == bs:
            return np.asarray(
                jax.device_get(
                    _roll_memory_mpc(params, sw_arr, ee_init, obj_init, targ_a, targ_b, quad_a, quad_b, has_bar, k_plan)
                )
            )
        parts = []
        for c_idx in range(bs // chunk_bs):
            sl = slice(c_idx * chunk_bs, (c_idx + 1) * chunk_bs)
            parts.append(
                jax.device_get(
                    _roll_memory_mpc(
                        params,
                        sw_arr,
                        ee_init[sl],
                        obj_init[sl],
                        targ_a[sl],
                        targ_b[sl],
                        quad_a[sl],
                        quad_b[sl],
                        has_bar[sl],
                        jax.random.fold_in(k_plan, c_idx),
                    )
                )
            )
        return np.concatenate(parts, axis=0)

    end_sust = _exec_panel8(False)
    dist_sust = np.asarray(jnp.sqrt(jnp.sum((end_sust - targ_a) ** 2, axis=-1)))
    succ_sust = (dist_sust < 0.25).astype(np.float32)

    end_sw = _exec_panel8(True)
    dist_sw_b = np.asarray(jnp.sqrt(jnp.sum((end_sw - targ_b) ** 2, axis=-1)))
    dist_sw_a = np.asarray(jnp.sqrt(jnp.sum((end_sw - targ_a) ** 2, axis=-1)))
    succ_sw = (dist_sw_b < 0.25).astype(np.float32)
    stale_trap = (dist_sw_a < 0.25).astype(np.float32)

    bar_np = np.asarray(has_bar)
    return {
        "sustained_transient_barrier_mpc": {
            "success_rate": paired_bootstrap_ci(succ_sust),
            "final_distance": paired_bootstrap_ci(dist_sust),
            "open_success": float(np.mean(succ_sust[~bar_np])),
            "thin_barrier_success": float(np.mean(succ_sust[bar_np])),
            "succ_vec": succ_sust.tolist(),
        },
        "mid_episode_goal_switch_mpc": {
            "success_rate": paired_bootstrap_ci(succ_sw),
            "final_distance": paired_bootstrap_ci(dist_sw_b),
            "stale_goal_a_trap_rate": float(np.mean(stale_trap)),
            "open_success": float(np.mean(succ_sw[~bar_np])),
            "thin_barrier_success": float(np.mean(succ_sw[bar_np])),
            "succ_vec": succ_sw.tolist(),
        },
    }


def eval_real_robot_transfer(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    robot_test_bank: RealRobotBank | None,
    *,
    use_residual: bool = True,
) -> dict[str, Any]:
    """Panel 9: Zero-Leakage Physical xArm6 / Language-Table 64x64x3 Multi-Step & Counterfactual Evaluation.

    Evaluates on N=256 held-out frame-deduplicated, recording-disjoint physical robot episodes in millimeters (mm):
      1) 1-step & 4-step Cartesian effector rollout error (mm) vs zero-velocity persistence on unpadded active ticks.
      2) Causal motor-plan conditioning gain (mm): error under negated motor plan (-u) minus true executed plan (+u).
      3) Visual-linguistic instruction grounding at t=3 (0.4s header, BEFORE operator trajectory unfolds):
         predicted endpoint error (mm), pairwise discrimination accuracy (true vs recording-matched counterfactual
         `wrong` human instruction on the exact same camera image), policy cosine alignment, and novel color-pair
         compositional generalization.
      4) Transient 0.4s instruction working-memory retention across full physical trajectories.
    """
    if robot_test_bank is None:
        return {"available": False}

    n_eval = robot_test_bank.size
    bs = 16
    t_seq = robot_test_bank.video.shape[1]

    @jax.jit
    def _eval_robot_batch(
        p: dict[str, jax.Array],
        vid: jax.Array,
        prop: jax.Array,
        prev_u: jax.Array,
        u_exec: jax.Array,
        txt_true: jax.Array,
        txt_wrong: jax.Array,
        txt_tran: jax.Array,
    ) -> dict[str, jax.Array]:
        aud = jnp.full((bs, t_seq, 64), -0.35, dtype=jnp.float32)
        mm = jnp.broadcast_to(jnp.array([0.0, 1.0, 1.0, 1.0], dtype=jnp.float32), (bs, t_seq, 4))
        val_mask = jnp.ones((bs, t_seq), dtype=bool)
        z_goal = jnp.zeros((bs, 4), dtype=jnp.float32)

        def _forward_cond(txt_in: jax.Array):
            b_in = {
                "video": vid,
                "audio": aud,
                "proprio": prop,
                "text": txt_in,
                "action": prev_u,
                "modality_mask": mm,
                "valid": val_mask,
            }
            z, _ = forward_world_model(p, spec, b_in)
            zn, lv = predict_latent_step(p, spec, z, u_exec, residual=use_residual)
            h = decode_world_heads(p, spec, z, zn, lv, z_goal, obs_proprio=prop, use_residual=use_residual)
            return z, h

        z_true, h_true = _forward_cond(txt_true)
        _, h_wrong = _forward_cond(txt_wrong)
        _, h_tran = _forward_cond(txt_tran)

        # 4-step open-loop latent rollout under true motor plan (+u) vs counterfactual motor plan (-u)
        z_anchor = z_true[:, :-4]
        hw = p["head_proprio_w"].astype(jnp.float32)
        hb = p["head_proprio_b"].astype(jnp.float32)

        def _roll4(u_seq_4: jax.Array) -> jax.Array:
            zr = z_anchor
            for k_step in range(4):
                zr, _ = predict_latent_step(
                    p, spec, zr, u_seq_4[:, k_step : (t_seq - 4) + k_step], residual=use_residual
                )
            zfr = zr.astype(jnp.float32)
            zf0 = z_anchor.astype(jnp.float32)
            if use_residual and spec.arch != "legacy_scalar_hybrid":
                d4 = (
                    0.75 * (zfr[:, :, SLOT_PROPRIO, :] - zf0[:, :, SLOT_PROPRIO, :])
                    + 0.25 * (jnp.mean(zfr[:, :, SLOTS_VISUAL, :], axis=2) - jnp.mean(zf0[:, :, SLOTS_VISUAL, :], axis=2))
                ) @ hw
                return (prop[:, :-4].astype(jnp.float32) + d4)[:, :, 0:2]
            return ((0.75 * zfr[:, :, SLOT_PROPRIO, :] + 0.25 * jnp.mean(zfr[:, :, SLOTS_VISUAL, :], axis=2)) @ hw + hb)[
                :, :, 0:2
            ]

        pred4_true = _roll4(u_exec)
        pred4_neg = _roll4(-u_exec)

        return {
            "pred1_xy": h_true["proprio_next"][:, :, 0:2],
            "pred4_true_xy": pred4_true,
            "pred4_neg_xy": pred4_neg,
            "goal_true_xy": h_true["pred_goal_xy"],
            "goal_wrong_xy": h_wrong["pred_goal_xy"],
            "goal_tran_xy": h_tran["pred_goal_xy"],
            "act_true_xy": h_true["action_proposal"][:, :, 0:2],
            "act_wrong_xy": h_wrong["action_proposal"][:, :, 0:2],
        }

    err1_mod_list: list[float] = []
    err1_per_list: list[float] = []
    err4_mod_list: list[float] = []
    err4_per_list: list[float] = []
    err4_neg_list: list[float] = []
    end_true_t3_list: list[float] = []
    end_wrong_t3_list: list[float] = []
    disc_t3_list: list[float] = []
    disc_tran_late_list: list[float] = []
    end_tran_late_list: list[float] = []
    cos_true_list: list[float] = []
    cos_wrong_list: list[float] = []
    novel_flags: list[bool] = []

    all_false = np.zeros((bs,), dtype=bool)
    all_true = np.ones((bs,), dtype=bool)

    for start in range(0, n_eval - bs + 1, bs):
        idx = np.arange(start, start + bs)
        txt_true_np = robot_test_bank.build_text_stream(robot_test_bank.text_header[idx], t_seq, all_false)
        txt_wrong_np = robot_test_bank.build_text_stream(robot_test_bank.wrong_header[idx], t_seq, all_false)
        txt_tran_np = robot_test_bank.build_text_stream(robot_test_bank.text_header[idx], t_seq, all_true)

        out = jax.device_get(
            _eval_robot_batch(
                params,
                jnp.asarray(robot_test_bank.video[idx]),
                jnp.asarray(robot_test_bank.proprio[idx]),
                jnp.asarray(robot_test_bank.prev_action[idx]),
                jnp.asarray(robot_test_bank.exec_action[idx]),
                jnp.asarray(txt_true_np),
                jnp.asarray(txt_wrong_np),
                jnp.asarray(txt_tran_np),
            )
        )

        prop_b = robot_test_bank.proprio[idx, :, 0:2]
        nprop_b = robot_test_bank.next_proprio[idx, :, 0:2]
        end_b = robot_test_bank.endpoint_xy[idx]
        eplen_b = robot_test_bank.ep_len[idx]
        nov_b = robot_test_bank.novel[idx]

        for i in range(bs):
            ln = int(max(5, min(eplen_b[i], t_seq)))
            e1_m = np.linalg.norm(out["pred1_xy"][i, :ln] - nprop_b[i, :ln], axis=-1) * ROBOT_SCALE_MM
            e1_p = np.linalg.norm(prop_b[i, :ln] - nprop_b[i, :ln], axis=-1) * ROBOT_SCALE_MM
            err1_mod_list.append(float(np.mean(e1_m)))
            err1_per_list.append(float(np.mean(e1_p)))

            ln4 = max(1, ln - 4)
            targ4 = prop_b[i, 4 : 4 + ln4]
            e4_m = np.linalg.norm(out["pred4_true_xy"][i, :ln4] - targ4, axis=-1) * ROBOT_SCALE_MM
            e4_p = np.linalg.norm(prop_b[i, :ln4] - targ4, axis=-1) * ROBOT_SCALE_MM
            e4_n = np.linalg.norm(out["pred4_neg_xy"][i, :ln4] - targ4, axis=-1) * ROBOT_SCALE_MM
            err4_mod_list.append(float(np.mean(e4_m)))
            err4_per_list.append(float(np.mean(e4_p)))
            err4_neg_list.append(float(np.mean(e4_n)))

            et = float(np.linalg.norm(out["goal_true_xy"][i, 3] - end_b[i]) * ROBOT_SCALE_MM)
            ew = float(np.linalg.norm(out["goal_wrong_xy"][i, 3] - end_b[i]) * ROBOT_SCALE_MM)
            end_true_t3_list.append(et)
            end_wrong_t3_list.append(ew)
            disc_t3_list.append(1.0 if et < ew - 1e-5 else (0.5 if abs(et - ew) <= 1e-5 else 0.0))

            t_late = min(ln - 1, 20)
            etr_late = float(np.linalg.norm(out["goal_tran_xy"][i, t_late] - end_b[i]) * ROBOT_SCALE_MM)
            ew_late = float(np.linalg.norm(out["goal_wrong_xy"][i, t_late] - end_b[i]) * ROBOT_SCALE_MM)
            end_tran_late_list.append(etr_late)
            disc_tran_late_list.append(
                1.0 if etr_late < ew_late - 1e-5 else (0.5 if abs(etr_late - ew_late) <= 1e-5 else 0.0)
            )

            disp = end_b[i] - prop_b[i, 3]
            disp_mm = float(np.linalg.norm(disp) * ROBOT_SCALE_MM)
            if disp_mm >= 15.0:
                d_unit = disp / (np.linalg.norm(disp) + 1e-8)
                at = out["act_true_xy"][i, 3]
                aw = out["act_wrong_xy"][i, 3]
                cos_true_list.append(float(np.dot(at, d_unit) / (np.linalg.norm(at) + 1e-6)))
                cos_wrong_list.append(float(np.dot(aw, d_unit) / (np.linalg.norm(aw) + 1e-6)))

            novel_flags.append(bool(nov_b[i]))

    e1_m_arr = np.asarray(err1_mod_list, dtype=np.float64)
    e1_p_arr = np.asarray(err1_per_list, dtype=np.float64)
    e4_m_arr = np.asarray(err4_mod_list, dtype=np.float64)
    e4_p_arr = np.asarray(err4_per_list, dtype=np.float64)
    e4_n_arr = np.asarray(err4_neg_list, dtype=np.float64)
    et_arr = np.asarray(end_true_t3_list, dtype=np.float64)
    ew_arr = np.asarray(end_wrong_t3_list, dtype=np.float64)
    disc_arr = np.asarray(disc_t3_list, dtype=np.float64)
    disc_tr_arr = np.asarray(disc_tran_late_list, dtype=np.float64)
    etr_arr = np.asarray(end_tran_late_list, dtype=np.float64)
    nov_mask = np.asarray(novel_flags, dtype=bool)
    cos_t_arr = np.asarray(cos_true_list, dtype=np.float64)
    cos_w_arr = np.asarray(cos_wrong_list, dtype=np.float64)

    return {
        "available": True,
        "n_episodes": int(len(e4_m_arr)),
        "n_novel_compositions": int(np.sum(nov_mask)),
        "step1_error_mm": paired_bootstrap_ci(e1_m_arr),
        "step1_persistence_mm": float(np.mean(e1_p_arr)),
        "step1_skill_vs_persistence": float(1.0 - np.mean(e1_m_arr) / max(float(np.mean(e1_p_arr)), 1e-6)),
        "step4_error_mm": paired_bootstrap_ci(e4_m_arr),
        "step4_persistence_mm": float(np.mean(e4_p_arr)),
        "step4_counterfactual_plan_mm": float(np.mean(e4_n_arr)),
        "step4_skill_vs_persistence": float(1.0 - np.mean(e4_m_arr) / max(float(np.mean(e4_p_arr)), 1e-6)),
        "causal_plan_gain_mm": paired_bootstrap_ci(e4_n_arr, e4_m_arr),
        "endpoint_true_t3_mm": paired_bootstrap_ci(et_arr),
        "endpoint_wrong_t3_mm": paired_bootstrap_ci(ew_arr),
        "counterfactual_endpoint_gain_mm": paired_bootstrap_ci(ew_arr, et_arr),
        "command_discrimination_acc": paired_bootstrap_ci(disc_arr),
        "transient_04s_late_endpoint_mm": paired_bootstrap_ci(etr_arr),
        "transient_04s_discrimination_acc": paired_bootstrap_ci(disc_tr_arr),
        "policy_cosine_true": float(np.mean(cos_t_arr)) if len(cos_t_arr) else 0.0,
        "policy_cosine_wrong": float(np.mean(cos_w_arr)) if len(cos_w_arr) else 0.0,
        "policy_cosine_gain": paired_bootstrap_ci(cos_t_arr, cos_w_arr) if len(cos_t_arr) else {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0},
        "novel_pair_step4_mm": paired_bootstrap_ci(e4_m_arr[nov_mask]) if np.any(nov_mask) else {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0},
        "novel_pair_discrimination_acc": paired_bootstrap_ci(disc_arr[nov_mask]) if np.any(nov_mask) else {"mean": 0.5, "ci95_low": 0.5, "ci95_high": 0.5},
        "step4_mm_vec": e4_m_arr.tolist(),
        "disc_vec": disc_arr.tolist(),
    }


def _streaming_memory_megabytes(spec: X1BSpec, cache_len: int) -> float:
    mem = init_streaming_memories(spec, batch_size=1, cache_len=cache_len)
    total_bytes = sum(int(x.size) * int(x.dtype.itemsize) for x in jax.tree.leaves(mem))
    return round(total_bytes / (1024.0 * 1024.0), 3)


def benchmark_context_scaling(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    lengths: tuple[int, ...] = (32, 128, 256, 512),
) -> dict[str, float]:
    """Panel 7: Compiled XLA forward throughput (slots/sec) & streaming state MB across sequence horizons."""
    timings: dict[str, float] = {
        "streaming_state_mb_T64": _streaming_memory_megabytes(spec, 64),
        "streaming_state_mb_T1024": _streaming_memory_megabytes(spec, 1024),
    }
    for seq_t in lengths:
        b = 1 if (seq_t >= 512 and spec.d_model >= 1536) else 2
        dummy = generate_synthetic_world_batch(jax.random.key(seq_t), b, seq_len=seq_t, vocab_size=spec.vocab_size)
        fn = jax.jit(lambda p, d_in: forward_world_model(p, spec, d_in)[0])
        out = fn(params, dummy)
        out.block_until_ready()
        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            out = fn(params, dummy)
            out.block_until_ready()
        dt = (time.perf_counter() - t0) / iters
        timings[f"T{seq_t}_ms"] = round((dt / b) * 2.0 * 1000.0, 2)
        timings[f"T{seq_t}_slots_per_s"] = round(float(b * seq_t * spec.slots_per_tick) / max(dt, 1e-6), 1)
    return timings


def run_full_evaluation(
    params: dict[str, jax.Array],
    spec: X1BSpec,
    *,
    objective_mode: str = "residual_ema_jepa_sigreg",
    real_bank_path: str | None = None,
    real_robot_path: str | None = None,
    pymunk_panels_path: str | None = None,
    include_scaling_bench: bool = True,
) -> dict[str, Any]:
    """Execute the complete frozen 9-panel evaluation suite for a trained X1B checkpoint."""
    use_res = objective_mode != "direct_jepa_no_residual"
    real_test = RealMultimodalBank(real_bank_path, split="test") if real_bank_path else None
    robot_test = RealRobotBank(real_robot_path, split="test") if real_robot_path else None

    eval_spec = X1BSpec(
        profile=spec.profile,
        arch=spec.arch,
        layers=spec.layers,
        d_model=spec.d_model,
        n_heads=spec.n_heads,
        head_dim=spec.head_dim,
        d_ff=spec.d_ff,
        slots_per_tick=spec.slots_per_tick,
        vocab_size=spec.vocab_size,
        dtype=spec.dtype,
        grad_checkpoint=False,
        rope_Window=1024,
    )

    p1 = eval_real_multimodal_and_text(params, eval_spec, real_test, use_residual=use_res)
    print("  [EVAL] Panel 1 (real multimodal & FineWeb) done", flush=True)
    p2 = eval_thin_barrier_and_persistence(params, eval_spec, pymunk_panels_path, use_residual=use_res)
    print("  [EVAL] Panel 2 (64x64 thin-barrier & persistence) done", flush=True)
    p3 = eval_long_horizon_associative_recall(params, eval_spec)
    print("  [EVAL] Panel 3 (T=32..1024 associative recall) done", flush=True)
    p4 = eval_transient_command_retention(params, eval_spec, use_residual=use_res)
    print("  [EVAL] Panel 4 (transient 0.4s command retention) done", flush=True)
    p56 = eval_closed_loop_mpc_and_spoofing(params, eval_spec, use_residual=use_res)
    print("  [EVAL] Panels 5 & 6 (N=96 Latent-CEM-MPC & CommitGate spoofing) done", flush=True)
    p8 = eval_long_horizon_memory_mpc(params, eval_spec, use_residual=use_res)
    print("  [EVAL] Panel 8 (T=76 OOD transient barrier & goal-switch MPC) done", flush=True)
    p9 = eval_real_robot_transfer(params, eval_spec, robot_test, use_residual=use_res)
    print("  [EVAL] Panel 9 (real physical xArm6 / Language-Table transfer) done", flush=True)
    p7 = benchmark_context_scaling(params, eval_spec) if include_scaling_bench else {}
    print("  [EVAL] Panel 7 (throughput & O(1) memory scaling) done", flush=True)

    return {
        "real_multimodal_and_text": p1,
        "thin_barrier_and_persistence": p2,
        "associative_recall_512": p3,
        "transient_goal_retention": p4,
        "closed_loop_mpc": p56["planners"],
        "long_horizon_memory_mpc": p8,
        "real_robot_transfer": p9,
        "spoofing_and_commit_gate": p56["commit_gate"],
        "throughput_scaling": p7,
    }
