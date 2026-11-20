"""SLWM-v3 Complete External SoTA Benchmark, Causal Intervention & Showcase Suite (`Waves A-D`).

Strictly evaluates frozen model weights (`no` online weight updates, `AGENTS.md §5 & §13.2`) across:
  1. **WikiText-103** (`validation` & sealed `test`): exact causal Bits-Per-Byte (BPB) & word-level PPL.
  2. **LAMBADA OpenAI** (`test`, 5,153 passages): exact final-word NLL, BPB, & greedy exact-word match accuracy.
  3. **HumanEval & MBPP** (`test`): exact canonical-solution causal BPB conditioned strictly on prompt bytes.
  4. **ESC-50 (`fold 5`) / LibriSpeech (`dev/test-clean`) / Clotho (`dev/test`)**:
       - Mimi 32-RVQ audio NLL, Audio -> Text byte BPB, and `AGENTS.md §13.2` condition-shuffling causal proof
         (`delta_shuffle_gain = L_shuffled - L_paired > 0`).
  5. **Synchronized Video <-> Audio <-> Text** (`GreatestHits`, `MSR-VTT`, `VGGSound`):
       - Video -> Audio Mimi RVQ NLL, Video+Audio -> Caption BPB, and condition-shuffling causal gain.
  6. **Embodied Robotics (`LeRobot ALOHA-Sim / PushT / LIBERO`)**:
       - Action-conditioned next-state proprio/visual latent prediction Huber error + `AGENTS.md §13.2`
         Action-Intervention Counterfactual Proof (`delta_action_intervention = L_shuffled_action - L_true_action > 0`).
  7. **Additive-RF Real-Speech-Overwrite (`dev` & sealed `test`, 167 s, 2,088 ticks > 1,776 additive RF)**:
       - Frozen-core decoupled probe counterfactual A/B flip + preservation accuracy on 100% real Mimi-encoded speech!
  8. **Deterministic Uncurated `inference_showcase/` Generator (`AGENTS.md §13.2`)**:
       - Pre-registered episode selection + deterministic `P05` (failure), `P25`, `P50` (median), `P75`, `P95`
         quantile ranking with explicit epistemic provenance (`observed`, `reconstructed`, `predicted`, `abstained`).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from data.v3_unified_shards import BYTE_EOS, BYTE_MOD_SEP, LoadedSparseShard, decode_utf8_bytes
from models.v3.any2any_heads import (
    compute_any2any_multimodal_loss,
    predict_any2any_heads,
)
from models.v3.core import init_core_state, step_core
from models.v3.fast_core import scan_core_fast
from models.v3.objectives import apply_decoupled_probe, init_decoupled_probe_params
from models.v3.spec import SLWMv3Spec


def evaluate_lambada_openai(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    shard_path: Path,
    *,
    max_passages: int | None = None,
    seq_len: int = 512,
    batch_size: int = 16,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
) -> dict[str, Any]:
    """Evaluate LAMBADA OpenAI exact final-word NLL, BPB, and greedy exact byte-level final-word match accuracy."""
    if not shard_path.is_file():
        return {"skipped": f"missing {shard_path}"}
    sh = LoadedSparseShard.load(shard_path)
    d = sh.data
    n_tot = sh.n_ticks
    rst = np.nonzero(d["reset_mask"])[0]
    ends = np.append(rst[1:], n_tot)
    n_ep = len(rst) if max_passages is None else min(int(max_passages), len(rst))

    @jax.jit
    def _fwd_logits(ap, batch):
        b = batch["valid_mask"].shape[1]
        z, _, _ = scan_core_fast(
            ap["core"],
            init_core_state(spec, b),
            batch,
            spec,
            remat_layers=False,
            matmul_dtype=matmul_dtype,
            use_wy_chunking=use_wy_chunking,
        )
        preds = predict_any2any_heads(ap["heads"], z, spec, matmul_dtype=matmul_dtype)
        return preds["text_logits"]  # [T, B, 264]

    word_nlls: list[float] = []
    word_exact: list[float] = []
    tot_target_bytes = 0
    tot_target_nats = 0.0

    # Group episodes into padded [seq_len, batch_size] windows aligned to each passage's final word
    for start_ep in range(0, n_ep, batch_size):
        chunk_ids = list(range(start_ep, min(start_ep + batch_size, n_ep)))
        cur_b = len(chunk_ids)
        tb_mat = np.full((seq_len, batch_size), 256, dtype=np.int32)
        vld_mat = np.zeros((seq_len, batch_size), dtype=np.float32)
        rst_mat = np.zeros((seq_len, batch_size), dtype=bool)
        rst_mat[0, :] = True
        target_spans: list[tuple[int, int]] = []  # (start_idx, end_idx_excl) in tb_mat for final word

        for bi, ei in enumerate(chunk_ids):
            s, e = int(rst[ei]), int(ends[ei])
            ep_bytes = d["text_byte"][s:e].astype(np.int32)
            # Strip trailing EOS if present, locate last whitespace separating context from final word
            raw_len = len(ep_bytes)
            while raw_len > 2 and ep_bytes[raw_len - 1] >= 256:
                raw_len -= 1
            ep_clean = ep_bytes[:raw_len]
            if len(ep_clean) > seq_len:
                ep_clean = ep_clean[-seq_len:]
            L = len(ep_clean)
            tb_mat[:L, bi] = ep_clean
            vld_mat[:L, bi] = 1.0
            # Find last space byte (32) or newline (10)
            ws_pos = np.where((ep_clean == 32) | (ep_clean == 10))[0]
            if len(ws_pos) > 0 and int(ws_pos[-1]) + 1 < L:
                w_start = int(ws_pos[-1]) + 1
            else:
                w_start = max(1, L - 4)
            target_spans.append((w_start, L))

        batch = {
            "modality_id": jnp.full((seq_len, batch_size), 2, dtype=jnp.int32),
            "dt_s": jnp.full((seq_len, batch_size), 0.02, dtype=jnp.float32),
            "valid_mask": jnp.asarray(vld_mat),
            "reliability": jnp.ones((seq_len, batch_size), dtype=jnp.float32),
            "reset_mask": jnp.asarray(rst_mat),
            "spatial_pos": jnp.zeros((seq_len, batch_size), dtype=jnp.int32),
            "text_byte": jnp.asarray(tb_mat),
            "has_true_action": jnp.zeros((seq_len, batch_size), dtype=jnp.float32),
            "video_feat": jnp.zeros((seq_len, batch_size, spec.video_patch_dim), dtype=jnp.float32),
            "audio_codes": jnp.zeros((seq_len, batch_size, spec.audio_codebooks), dtype=jnp.int32),
            "proprio_feat": jnp.zeros((seq_len, batch_size, spec.proprio_dim), dtype=jnp.float32),
            "action_feat": jnp.zeros((seq_len, batch_size, spec.action_dim), dtype=jnp.float32),
            "probe_feat": jnp.zeros((seq_len, batch_size, spec.probe_dim), dtype=jnp.float32),
        }
        logits = np.asarray(_fwd_logits(all_params, batch), dtype=np.float32)  # [T, B, 264]
        for bi in range(cur_b):
            ws, we = target_spans[bi]
            # Logit at position t-1 predicts byte at position t!
            pred_logits = logits[ws - 1 : we - 1, bi, :]  # [wlen, 264]
            true_bytes = tb_mat[ws:we, bi]  # [wlen]
            # Stable log-softmax
            m_l = np.max(pred_logits, axis=-1, keepdims=True)
            logp = pred_logits - m_l - np.log(np.sum(np.exp(pred_logits - m_l), axis=-1, keepdims=True))
            nll_bytes = -logp[np.arange(len(true_bytes)), true_bytes]
            w_nll = float(np.sum(nll_bytes))
            word_nlls.append(w_nll)
            tot_target_nats += w_nll
            tot_target_bytes += len(true_bytes)
            pred_ids = np.argmax(pred_logits, axis=-1)
            word_exact.append(1.0 if np.array_equal(pred_ids, true_bytes) else 0.0)

    mean_word_nll = float(np.mean(word_nlls)) if word_nlls else float("nan")
    return {
        "shard": shard_path.name,
        "evaluated_passages": int(len(word_nlls)),
        "final_word_nll": round(mean_word_nll, 5),
        "final_word_ppl": round(math.exp(min(20.0, mean_word_nll)), 3),
        "final_word_bpb": round((tot_target_nats / max(tot_target_bytes, 1)) / math.log(2.0), 5),
        "final_word_exact_match_acc": round(float(np.mean(word_exact)), 4) if word_exact else 0.0,
    }


def evaluate_code_solution_bpb(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    shard_path: Path,
    *,
    seq_len: int = 512,
    batch_size: int = 16,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
) -> dict[str, Any]:
    """Evaluate exact HumanEval / MBPP canonical-solution BPB strictly AFTER `BYTE_MOD_SEP` (prompt -> solution)."""
    if not shard_path.is_file():
        return {"skipped": f"missing {shard_path}"}
    sh = LoadedSparseShard.load(shard_path)
    d = sh.data
    n_tot = sh.n_ticks
    rst = np.nonzero(d["reset_mask"])[0]
    ends = np.append(rst[1:], n_tot)

    @jax.jit
    def _fwd_logits(ap, batch):
        b = batch["valid_mask"].shape[1]
        z, _, _ = scan_core_fast(
            ap["core"],
            init_core_state(spec, b),
            batch,
            spec,
            remat_layers=False,
            matmul_dtype=matmul_dtype,
            use_wy_chunking=use_wy_chunking,
        )
        return predict_any2any_heads(ap["heads"], z, spec, matmul_dtype=matmul_dtype)["text_logits"]

    tot_nats = 0.0
    tot_bytes = 0
    n_problems = 0
    for start_ep in range(0, len(rst), batch_size):
        chunk_ids = list(range(start_ep, min(start_ep + batch_size, len(rst))))
        tb_mat = np.full((seq_len, batch_size), 256, dtype=np.int32)
        vld_mat = np.zeros((seq_len, batch_size), dtype=np.float32)
        rst_mat = np.zeros((seq_len, batch_size), dtype=bool)
        rst_mat[0, :] = True
        sol_spans: list[tuple[int, int]] = []

        for bi, ei in enumerate(chunk_ids):
            ep_full = d["text_byte"][int(rst[ei]) : int(ends[ei])].astype(np.int32)
            # Ensure BYTE_MOD_SEP + solution bytes stay inside window even for long HumanEval prompts!
            sep_all = np.where(ep_full == BYTE_MOD_SEP)[0]
            if len(ep_full) > seq_len and len(sep_all) > 0:
                sep_pos = int(sep_all[0])
                # Keep prompt tail leading into separator + solution up to seq_len
                win_start = max(0, sep_pos - (seq_len // 2))
                ep = ep_full[win_start : win_start + seq_len]
            else:
                ep = ep_full[:seq_len]
            L = len(ep)
            tb_mat[:L, bi] = ep
            vld_mat[:L, bi] = 1.0
            sep = np.where(ep == BYTE_MOD_SEP)[0]
            if len(sep) > 0 and int(sep[0]) + 1 < L:
                sol_spans.append((int(sep[0]) + 1, L))
            else:
                sol_spans.append((L, L))

        batch = {
            "modality_id": jnp.full((seq_len, batch_size), 2, dtype=jnp.int32),
            "dt_s": jnp.full((seq_len, batch_size), 0.02, dtype=jnp.float32),
            "valid_mask": jnp.asarray(vld_mat),
            "reliability": jnp.ones((seq_len, batch_size), dtype=jnp.float32),
            "reset_mask": jnp.asarray(rst_mat),
            "spatial_pos": jnp.zeros((seq_len, batch_size), dtype=jnp.int32),
            "text_byte": jnp.asarray(tb_mat),
            "has_true_action": jnp.zeros((seq_len, batch_size), dtype=jnp.float32),
            "video_feat": jnp.zeros((seq_len, batch_size, spec.video_patch_dim), dtype=jnp.float32),
            "audio_codes": jnp.zeros((seq_len, batch_size, spec.audio_codebooks), dtype=jnp.int32),
            "proprio_feat": jnp.zeros((seq_len, batch_size, spec.proprio_dim), dtype=jnp.float32),
            "action_feat": jnp.zeros((seq_len, batch_size, spec.action_dim), dtype=jnp.float32),
            "probe_feat": jnp.zeros((seq_len, batch_size, spec.probe_dim), dtype=jnp.float32),
        }
        logits = np.asarray(_fwd_logits(all_params, batch), dtype=np.float32)
        for bi, (ss, se) in enumerate(sol_spans):
            if se <= ss:
                continue
            pl = logits[ss - 1 : se - 1, bi, :]
            yb = tb_mat[ss:se, bi]
            ml = np.max(pl, axis=-1, keepdims=True)
            logp = pl - ml - np.log(np.sum(np.exp(pl - ml), axis=-1, keepdims=True))
            tot_nats += float(np.sum(-logp[np.arange(len(yb)), yb]))
            tot_bytes += len(yb)
            n_problems += 1

    mean_nat = tot_nats / max(tot_bytes, 1)
    return {
        "shard": shard_path.name,
        "evaluated_problems": int(n_problems),
        "evaluated_solution_bytes": int(tot_bytes),
        "solution_ce_nat_per_byte": round(mean_nat, 5),
        "solution_bits_per_byte": round(mean_nat / math.log(2.0), 5),
    }


def _build_episode_aligned_batch(
    sh: LoadedSparseShard,
    ep_slices: list[tuple[int, int]],
    spec: SLWMv3Spec,
    seq_len: int,
) -> dict[str, np.ndarray]:
    """Pack complete episode prefixes starting cleanly at `reset_mask` into `[seq_len, B, ...]`."""
    d = sh.data
    b_sz = len(ep_slices)
    mod = np.full((seq_len, b_sz), 2, dtype=np.int32)
    dt = np.zeros((seq_len, b_sz), dtype=np.float32)
    vld = np.zeros((seq_len, b_sz), dtype=np.float32)
    rel = np.ones((seq_len, b_sz), dtype=np.float32)
    rst = np.zeros((seq_len, b_sz), dtype=bool)
    rst[0, :] = True
    sp = np.zeros((seq_len, b_sz), dtype=np.int32)
    tb = np.full((seq_len, b_sz), 256, dtype=np.int32)
    ha = np.zeros((seq_len, b_sz), dtype=np.float32)
    vid = np.zeros((seq_len, b_sz, spec.video_patch_dim), dtype=np.float32)
    aud = np.zeros((seq_len, b_sz, spec.audio_codebooks), dtype=np.int32)
    prop = np.zeros((seq_len, b_sz, spec.proprio_dim), dtype=np.float32)
    act = np.zeros((seq_len, b_sz, spec.action_dim), dtype=np.float32)

    for bi, (s, e) in enumerate(ep_slices):
        L = min(seq_len, e - s)
        idx = np.arange(s, s + L, dtype=np.int64)
        mod[:L, bi] = d["modality_id"][idx].astype(np.int32)
        dt[:L, bi] = d["dt_s"][idx].astype(np.float32)
        vld[:L, bi] = d["valid_mask"][idx].astype(np.float32)
        rel[:L, bi] = d["reliability"][idx].astype(np.float32)
        sp[:L, bi] = d["spatial_pos"][idx].astype(np.int32)
        tb[:L, bi] = d["text_byte"][idx].astype(np.int32)
        ha[:L, bi] = d["has_true_action"][idx].astype(np.float32)
        vi = d["video_idx"][idx]
        vm = vi >= 0
        if np.any(vm) and d["video_feat"].shape[0] > 0:
            vid[:L, bi][vm] = d["video_feat"][vi[vm]].astype(np.float32)
        ai = d["audio_idx"][idx]
        am = ai >= 0
        if np.any(am) and d["audio_codes"].shape[0] > 0:
            aud[:L, bi][am] = d["audio_codes"][ai[am]].astype(np.int32)
        pi = d["act_idx"][idx]
        pm = pi >= 0
        if np.any(pm) and d["proprio_feat"].shape[0] > 0:
            prop[:L, bi][pm] = d["proprio_feat"][pi[pm]].astype(np.float32)
            act[:L, bi][pm] = d["action_feat"][pi[pm]].astype(np.float32)

    return {
        "modality_id": mod,
        "dt_s": dt,
        "valid_mask": vld,
        "target_valid_mask": vld.copy(),
        "reliability": rel,
        "reset_mask": rst,
        "spatial_pos": sp,
        "text_byte": tb,
        "has_true_action": ha,
        "video_feat": vid,
        "audio_codes": aud,
        "proprio_feat": prop,
        "action_feat": act,
        "probe_feat": np.zeros((seq_len, b_sz, spec.probe_dim), dtype=np.float32),
    }


def evaluate_cross_modal_condition_shuffle(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    shard_path: Path,
    *,
    source_mod: int,
    target_mod: int,
    seq_len: int = 768,
    batch_size: int = 8,
    max_episodes: int = 64,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
    seed: int = 8801,
) -> dict[str, Any]:
    """Zero-padding-free episode-aligned AGENTS.md §13.2 Causal Conditioning Proof.

    Swaps ONLY real `source_mod` payloads (including `spatial_pos` for video) from a distinct donor
    episode into each recipient episode's `source_mod` ticks while holding ALL recipient target
    tokens, timestamps, masks, and target-history strictly fixed — mathematically guaranteeing
    `causal_shuffle_gain == 0.0` for any source-blind model!
    """
    del seed
    if not shard_path.is_file():
        return {"skipped": f"missing {shard_path}"}
    sh = LoadedSparseShard.load(shard_path)
    d = sh.data
    n_tot = sh.n_ticks
    rst = np.nonzero(d["reset_mask"])[0]
    if len(rst) == 0 or rst[0] != 0:
        rst = np.insert(rst, 0, 0)
    ends = np.append(rst[1:], n_tot)

    # Filter to episodes where BOTH source_mod and target_mod exist inside `eval_len` and source precedes target!
    eval_len = int(seq_len)
    valid_eps: list[tuple[int, int]] = []
    for s, e in zip(rst, ends):
        s_i, e_i = int(s), int(e)
        # If episode is longer than eval_len, align window so both source tail and target head fit!
        if e_i - s_i > eval_len:
            m_full = d["modality_id"][s_i:e_i]
            t_pos = np.where(m_full == target_mod)[0]
            s_pos = np.where(m_full == source_mod)[0]
            if len(t_pos) < 4 or len(s_pos) < 4:
                continue
            # Anchor window ending after target_mod starts, keeping preceding source_mod!
            first_t = int(t_pos[0])
            if first_t >= eval_len - 16:
                win_s = max(0, first_t - (eval_len * 3 // 4))
                s_i = s_i + win_s
                e_i = min(e_i, s_i + eval_len)
            else:
                e_i = s_i + eval_len
        m_ep = d["modality_id"][s_i:e_i]
        s_pos = np.where(m_ep == source_mod)[0]
        t_pos = np.where(m_ep == target_mod)[0]
        if len(s_pos) >= 4 and len(t_pos) >= 4 and int(s_pos[0]) < int(t_pos[-1]):
            valid_eps.append((s_i, e_i))
        if len(valid_eps) >= max_episodes:
            break

    if len(valid_eps) < 2:
        return {"shard": shard_path.name, "n_evaluated_episodes": 0, "causal_conditioning_verified": False}

    @jax.jit
    def _eval_per_item(ap, batch_in, batch_target):
        b = batch_in["valid_mask"].shape[1]
        z, _, _ = scan_core_fast(
            ap["core"],
            init_core_state(spec, b),
            batch_in,
            spec,
            remat_layers=False,
            matmul_dtype=matmul_dtype,
            use_wy_chunking=use_wy_chunking,
        )
        preds = predict_any2any_heads(ap["heads"], z[:-1], spec, matmul_dtype=matmul_dtype)
        # Strictly evaluate against UNMODIFIED `batch_target`!
        vld_pair = batch_target["valid_mask"][:-1] * batch_target["valid_mask"][1:]
        m_tgt = vld_pair * (batch_target["modality_id"][1:] == target_mod).astype(jnp.float32)  # [T-1, B]
        if target_mod == 2:
            y = jnp.clip(batch_target["text_byte"][1:].astype(jnp.int32), 0, spec.text_vocab_size - 1)
            per_tok = optax.softmax_cross_entropy_with_integer_labels(preds["text_logits"], y)
        elif target_mod == 1:
            k_aud = preds["audio_logits"].shape[2]
            y = jnp.clip(batch_target["audio_codes"][1:, :, :k_aud].astype(jnp.int32), 0, spec.audio_vocab_size - 1)
            per_tok = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(preds["audio_logits"], y), axis=-1)
        elif target_mod == 0:
            diff = preds["video_pred"] - batch_target["video_feat"][1:].astype(jnp.float32)
            ab = jnp.abs(diff)
            per_tok = jnp.mean(jnp.where(ab <= 1.0, 0.5 * diff * diff, ab - 0.5), axis=-1)
        else:
            diff = preds["action_pred"][..., spec.action_dim :] - batch_target["proprio_feat"][1:].astype(jnp.float32)
            ab = jnp.abs(diff)
            per_tok = jnp.mean(jnp.where(ab <= 1.0, 0.5 * diff * diff, ab - 0.5), axis=-1)
        ep_loss = jnp.sum(per_tok * m_tgt, axis=0) / jnp.maximum(jnp.sum(m_tgt, axis=0), 1.0)
        ep_cnt = jnp.sum(m_tgt, axis=0)
        return ep_loss, ep_cnt

    paired_ep_losses: list[float] = []
    shuf_ep_losses: list[float] = []
    bsz = max(2, int(batch_size))
    for st in range(0, len(valid_eps), bsz):
        chunk = valid_eps[st : st + bsz]
        if len(chunk) < 2:
            break
        b_pair_np = _build_episode_aligned_batch(sh, chunk, spec, eval_len)
        b_shuf_np = {k: v.copy() for k, v in b_pair_np.items()}
        cur_b = len(chunk)
        # Swap REAL source-modality stream from donor `(bi + 1) % cur_b` into recipient `bi`'s source ticks (zero padding!)
        for bi in range(cur_b):
            donor = (bi + 1) % cur_b
            rec_idx = np.where((b_pair_np["modality_id"][:, bi] == source_mod) & (b_pair_np["valid_mask"][:, bi] > 0.5))[0]
            don_idx = np.where((b_pair_np["modality_id"][:, donor] == source_mod) & (b_pair_np["valid_mask"][:, donor] > 0.5))[0]
            if len(rec_idx) == 0 or len(don_idx) == 0:
                continue
            mapped = don_idx[np.arange(len(rec_idx)) % len(don_idx)]
            if source_mod == 0:
                b_shuf_np["video_feat"][rec_idx, bi] = b_pair_np["video_feat"][mapped, donor]
                b_shuf_np["spatial_pos"][rec_idx, bi] = b_pair_np["spatial_pos"][mapped, donor]
            elif source_mod == 1:
                b_shuf_np["audio_codes"][rec_idx, bi] = b_pair_np["audio_codes"][mapped, donor]
            elif source_mod == 2:
                b_shuf_np["text_byte"][rec_idx, bi] = b_pair_np["text_byte"][mapped, donor]

        b_pair = {k: jnp.asarray(v) for k, v in b_pair_np.items()}
        b_shuf = {k: jnp.asarray(v) for k, v in b_shuf_np.items()}
        lp, cp = _eval_per_item(all_params, b_pair, b_pair)
        ls, _ = _eval_per_item(all_params, b_shuf, b_pair)
        lp_np, ls_np, cp_np = np.asarray(lp, dtype=np.float64), np.asarray(ls, dtype=np.float64), np.asarray(cp, dtype=np.float64)
        for bi in range(cur_b):
            if cp_np[bi] >= 1.0:
                paired_ep_losses.append(float(lp_np[bi]))
                shuf_ep_losses.append(float(ls_np[bi]))

    if not paired_ep_losses:
        return {"shard": shard_path.name, "n_evaluated_episodes": 0, "causal_conditioning_verified": False}
    deltas = np.asarray(shuf_ep_losses) - np.asarray(paired_ep_losses)
    mean_d = float(np.mean(deltas))
    se_d = float(np.std(deltas, ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
    key_map = {0: "video_patch_huber", 1: "audio_rvq_ce_nat", 2: "text_ce_nat", 3: "proprio_huber"}
    return {
        "shard": shard_path.name,
        "source_mod": int(source_mod),
        "target_mod": int(target_mod),
        "metric": key_map[target_mod],
        "n_evaluated_episodes": int(len(deltas)),
        "paired_loss": round(float(np.mean(paired_ep_losses)), 5),
        "shuffled_source_loss": round(float(np.mean(shuf_ep_losses)), 5),
        "causal_shuffle_gain": round(mean_d, 5),
        "causal_shuffle_gain_se": round(se_d, 5),
        "causal_shuffle_ci95_low": round(mean_d - 1.96 * se_d, 5),
        "frac_episodes_improved": round(float(np.mean(deltas > 0.0)), 4),
        "causal_conditioning_verified": bool(mean_d > 0.005 and (mean_d - 1.96 * se_d) > 0.0),
    }


def evaluate_robotics_action_intervention(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    shard_path: Path,
    *,
    seq_len: int = 512,
    batch_size: int = 8,
    max_episodes: int = 32,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
    seed: int = 8901,
) -> dict[str, Any]:
    """Zero-False-Positive AGENTS.md §13.2 Action-Intervention Proof.

    Intervenes ONLY on INPUT `action_feat` at real `modality_id == 3` ticks using real physical motor
    commands from a distinct donor episode, and scores predicted **next proprioception (`action_pred[..., 14:]`)**
    and **next visual frame (`video_pred`)** strictly against the **UNMODIFIED PAIRED targets**!
    Mathematically guarantees `gain == 0.0` for any action-blind model!
    """
    del seed
    if not shard_path.is_file():
        return {"skipped": f"missing {shard_path}"}
    sh = LoadedSparseShard.load(shard_path)
    d = sh.data
    n_tot = sh.n_ticks
    rst = np.nonzero(d["reset_mask"])[0]
    if len(rst) == 0 or rst[0] != 0:
        rst = np.insert(rst, 0, 0)
    ends = np.append(rst[1:], n_tot)
    ep_slices = []
    for s, e in zip(rst, ends):
        s_i, e_i = int(s), int(e)
        if np.sum((d["modality_id"][s_i : min(e_i, s_i + seq_len)] == 3) & (d["has_true_action"][s_i : min(e_i, s_i + seq_len)] > 0)) >= 4:
            ep_slices.append((s_i, e_i))
        if len(ep_slices) >= max_episodes:
            break
    if len(ep_slices) < 2:
        return {"shard": shard_path.name, "n_evaluated_episodes": 0, "causal_action_conditioning_verified": False}

    @jax.jit
    def _eval_targets(ap, batch_in, batch_target):
        b = batch_in["valid_mask"].shape[1]
        z, _, _ = scan_core_fast(
            ap["core"],
            init_core_state(spec, b),
            batch_in,
            spec,
            remat_layers=False,
            matmul_dtype=matmul_dtype,
            use_wy_chunking=use_wy_chunking,
        )
        preds = predict_any2any_heads(ap["heads"], z[:-1], spec, matmul_dtype=matmul_dtype)
        vld_pair = batch_target["valid_mask"][:-1] * batch_target["valid_mask"][1:]
        # Score ONLY next proprioception (dims `spec.action_dim:`) against UNMODIFIED paired `proprio_feat[1:]`!
        m_prop = vld_pair * (batch_target["modality_id"][1:] == 3).astype(jnp.float32) * (batch_target["has_true_action"][1:] > 0.5)
        p_diff = preds["action_pred"][..., spec.action_dim :] - batch_target["proprio_feat"][1:].astype(jnp.float32)
        p_abs = jnp.abs(p_diff)
        p_hub = jnp.mean(jnp.where(p_abs <= 1.0, 0.5 * p_diff * p_diff, p_abs - 0.5), axis=-1)
        ep_prop_loss = jnp.sum(p_hub * m_prop, axis=0) / jnp.maximum(jnp.sum(m_prop, axis=0), 1.0)
        ep_prop_cnt = jnp.sum(m_prop, axis=0)

        # Score next visual frame against UNMODIFIED paired `video_feat[1:]`!
        m_vid = vld_pair * (batch_target["modality_id"][1:] == 0).astype(jnp.float32)
        v_diff = preds["video_pred"] - batch_target["video_feat"][1:].astype(jnp.float32)
        v_abs = jnp.abs(v_diff)
        v_hub = jnp.mean(jnp.where(v_abs <= 1.0, 0.5 * v_diff * v_diff, v_abs - 0.5), axis=-1)
        ep_vid_loss = jnp.sum(v_hub * m_vid, axis=0) / jnp.maximum(jnp.sum(m_vid, axis=0), 1.0)
        ep_vid_cnt = jnp.sum(m_vid, axis=0)
        return ep_prop_loss, ep_prop_cnt, ep_vid_loss, ep_vid_cnt

    p_prop, s_prop, p_vid, s_vid = [], [], [], []
    bsz = max(2, int(batch_size))
    for st in range(0, len(ep_slices), bsz):
        chunk = ep_slices[st : st + bsz]
        if len(chunk) < 2:
            break
        b_pair_np = _build_episode_aligned_batch(sh, chunk, spec, seq_len)
        b_int_np = {k: v.copy() for k, v in b_pair_np.items()}
        cur_b = len(chunk)
        for bi in range(cur_b):
            donor = (bi + 1) % cur_b
            rec_a = np.where((b_pair_np["modality_id"][:, bi] == 3) & (b_pair_np["has_true_action"][:, bi] > 0.5))[0]
            don_a = np.where((b_pair_np["modality_id"][:, donor] == 3) & (b_pair_np["has_true_action"][:, donor] > 0.5))[0]
            if len(rec_a) > 0 and len(don_a) > 0:
                mapped = don_a[np.arange(len(rec_a)) % len(don_a)]
                b_int_np["action_feat"][rec_a, bi] = b_pair_np["action_feat"][mapped, donor]

        b_pair = {k: jnp.asarray(v) for k, v in b_pair_np.items()}
        b_int = {k: jnp.asarray(v) for k, v in b_int_np.items()}
        pp, pc, vp, vc = _eval_targets(all_params, b_pair, b_pair)
        pi, _, vi, _ = _eval_targets(all_params, b_int, b_pair)
        for bi in range(cur_b):
            if float(pc[bi]) >= 1.0:
                p_prop.append(float(pp[bi]))
                s_prop.append(float(pi[bi]))
            if float(vc[bi]) >= 1.0:
                p_vid.append(float(vp[bi]))
                s_vid.append(float(vi[bi]))

    prop_deltas = np.asarray(s_prop) - np.asarray(p_prop) if p_prop else np.zeros((0,))
    vid_deltas = np.asarray(s_vid) - np.asarray(p_vid) if p_vid else np.zeros((0,))
    prop_gain = float(np.mean(prop_deltas)) if len(prop_deltas) else 0.0
    vid_gain = float(np.mean(vid_deltas)) if len(vid_deltas) else 0.0
    prop_se = float(np.std(prop_deltas, ddof=1) / math.sqrt(len(prop_deltas))) if len(prop_deltas) > 1 else 0.0
    return {
        "shard": shard_path.name,
        "n_evaluated_episodes": int(len(p_prop)),
        "paired_next_proprio_huber": round(float(np.mean(p_prop)), 5) if p_prop else None,
        "intervened_next_proprio_huber": round(float(np.mean(s_prop)), 5) if s_prop else None,
        "action_intervention_proprio_gain": round(prop_gain, 5),
        "action_intervention_proprio_se": round(prop_se, 5),
        "paired_next_video_huber": round(float(np.mean(p_vid)), 5) if p_vid else None,
        "intervened_next_video_huber": round(float(np.mean(s_vid)), 5) if s_vid else None,
        "action_intervention_video_gain": round(vid_gain, 5),
        "targets_strictly_held_fixed": True,
        "causal_action_conditioning_verified": bool(prop_gain > 1e-4 and (prop_gain - 1.96 * prop_se) > 0.0),
    }


def evaluate_real_speech_overwrite_frozen_core(
    core_params: dict[str, Any],
    spec: SLWMv3Spec,
    train_npz: Path,
    eval_npz: Path,
    *,
    probe_steps: int = 200,
    eval_micro_bsz: int = 8,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
    seed: int = 9101,
) -> dict[str, Any]:
    """Evaluate strictly frozen core on 100% real Mimi-encoded additive-RF Real-Speech-Overwrite (`167 s`, `2,088` ticks)!"""
    if (not train_npz.is_file()) or (not eval_npz.is_file()):
        return {"skipped": f"missing {train_npz} or {eval_npz}"}

    tr_d = np.load(train_npz, allow_pickle=False)
    ev_d = np.load(eval_npz, allow_pickle=False)
    tr_ca = tr_d["audio_codes_a"] if "audio_codes_a" in tr_d.files else tr_d["codes_a"]
    tr_cb = tr_d["audio_codes_b"] if "audio_codes_b" in tr_d.files else tr_d["codes_b"]
    ev_ca = ev_d["audio_codes_a"] if "audio_codes_a" in ev_d.files else ev_d["codes_a"]
    ev_cb = ev_d["audio_codes_b"] if "audio_codes_b" in ev_d.files else ev_d["codes_b"]
    vocab_size = int(max(np.max(tr_d["y_a"]), np.max(tr_d["y_b"]), np.max(ev_d["y_a"]), np.max(ev_d["y_b"]))) + 1

    @jax.jit
    def _extract_final_z(cp, audio_codes_tb32, dt_tb):
        t_len, b_sz = dt_tb.shape
        pkts = {
            "modality_id": jnp.ones((t_len, b_sz), dtype=jnp.int32),
            "dt_s": dt_tb,
            "valid_mask": jnp.ones((t_len, b_sz), dtype=jnp.float32),
            "reliability": jnp.ones((t_len, b_sz), dtype=jnp.float32),
            "reset_mask": jnp.zeros((t_len, b_sz), dtype=jnp.bool_).at[0].set(True),
            "spatial_pos": jnp.zeros((t_len, b_sz), dtype=jnp.int32),
            "text_byte": jnp.full((t_len, b_sz), 256, dtype=jnp.int32),
            "has_true_action": jnp.zeros((t_len, b_sz), dtype=jnp.float32),
            "video_feat": jnp.zeros((t_len, b_sz, spec.video_patch_dim), dtype=jnp.float32),
            "audio_codes": audio_codes_tb32,
            "proprio_feat": jnp.zeros((t_len, b_sz, spec.proprio_dim), dtype=jnp.float32),
            "action_feat": jnp.zeros((t_len, b_sz, spec.action_dim), dtype=jnp.float32),
            "probe_feat": jnp.zeros((t_len, b_sz, spec.probe_dim), dtype=jnp.float32),
        }
        z, _, d_m = scan_core_fast(
            cp,
            init_core_state(spec, b_sz),
            pkts,
            spec,
            remat_layers=False,
            matmul_dtype=matmul_dtype,
            assume_all_valid=True,
            use_wy_chunking=use_wy_chunking,
        )
        # Return pure JAX arrays (NEVER call Python `float()` inside `@jax.jit`!)
        return jax.lax.stop_gradient(z[-1]), jnp.mean(d_m["mean_nowrite"])

    def _run_extract(codes_nt32: np.ndarray) -> tuple[np.ndarray, float]:
        n_ep, t_len, _ = codes_nt32.shape
        zs = []
        nws = []
        for s in range(0, n_ep, eval_micro_bsz):
            e = min(s + eval_micro_bsz, n_ep)
            # Transpose [N, T, 32] -> [T, B, 32]
            c_tb = jnp.asarray(np.transpose(codes_nt32[s:e], (1, 0, 2)), dtype=jnp.int32)
            dt_tb = jnp.full((t_len, e - s), 0.08, dtype=jnp.float32)  # 12.5 Hz Mimi = 0.08 s/tick
            zb, nw = _extract_final_z(core_params, c_tb, dt_tb)
            zs.append(np.asarray(zb, dtype=np.float32))
            nws.append(float(nw))
        return np.concatenate(zs, axis=0), float(np.mean(nws)) if nws else 0.0

    z_tr_a, _ = _run_extract(tr_ca)
    z_tr_b, _ = _run_extract(tr_cb)
    z_tr = jnp.asarray(np.concatenate([z_tr_a, z_tr_b], axis=0))
    y_tr = jnp.asarray(np.concatenate([tr_d["y_a"], tr_d["y_b"]], axis=0), dtype=jnp.int32)

    probe_p = init_decoupled_probe_params(jax.random.PRNGKey(seed), spec.d_model, out_dim=2 * vocab_size, hidden_dim=128)
    tx_p = optax.adamw(3e-3, weight_decay=1e-4)
    opt_p = tx_p.init(probe_p)

    @jax.jit
    def _pstep(pp, ost):
        def _l(pr):
            lg = apply_decoupled_probe(pr, z_tr).reshape(-1, 2, vocab_size)
            return 0.5 * (
                optax.softmax_cross_entropy_with_integer_labels(lg[:, 0], y_tr[:, 0]).mean()
                + optax.softmax_cross_entropy_with_integer_labels(lg[:, 1], y_tr[:, 1]).mean()
            )

        lv, g = jax.value_and_grad(_l)(pp)
        u, nost = tx_p.update(g, ost, pp)
        return optax.apply_updates(pp, u), nost, lv

    for _ in range(probe_steps):
        probe_p, opt_p, _ = _pstep(probe_p, opt_p)

    za_np, nw_a = _run_extract(ev_ca)
    zb_np, nw_b = _run_extract(ev_cb)
    za = jnp.asarray(za_np)
    zb = jnp.asarray(zb_np)
    pa = np.asarray(jnp.argmax(apply_decoupled_probe(probe_p, za).reshape(-1, 2, vocab_size), axis=-1))
    pb = np.asarray(jnp.argmax(apply_decoupled_probe(probe_p, zb).reshape(-1, 2, vocab_size), axis=-1))
    ya, yb = np.asarray(ev_d["y_a"]), np.asarray(ev_d["y_b"])
    ua = np.asarray(ev_d["update_which"] if "update_which" in ev_d.files else ev_d["upd_attr"], dtype=np.int32)
    is_catch = (
        np.asarray(ev_d["is_catch"], dtype=bool)
        if "is_catch" in ev_d.files
        else (ua < 0)
    )
    ow_mask = (~is_catch) & (ua >= 0) & (ua <= 1)
    c_mask = ~ow_mask

    # Overwrite episodes (`update_which in {0, 1}`): counterfactual flip on updated attr + preserve other attr!
    ow_idx = np.nonzero(ow_mask)[0]
    ow_ua = ua[ow_idx]
    upd_ok_ow = (pa[ow_idx, ow_ua] == ya[ow_idx, ow_ua]) & (pb[ow_idx, ow_ua] == yb[ow_idx, ow_ua]) if len(ow_idx) else np.zeros((0,), dtype=bool)
    pres_ok_ow = (pa[ow_idx, 1 - ow_ua] == ya[ow_idx, 1 - ow_ua]) & (pb[ow_idx, 1 - ow_ua] == yb[ow_idx, 1 - ow_ua]) if len(ow_idx) else np.zeros((0,), dtype=bool)
    ab_joint_ow = upd_ok_ow & pres_ok_ow

    # Catch trials (`update_which == -1`, zero correction): BOTH attributes must match initial state on A & B!
    c_idx = np.nonzero(c_mask)[0]
    catch_ok = (
        (pa[c_idx, 0] == ya[c_idx, 0])
        & (pa[c_idx, 1] == ya[c_idx, 1])
        & (pb[c_idx, 0] == yb[c_idx, 0])
        & (pb[c_idx, 1] == yb[c_idx, 1])
    ) if len(c_idx) else np.zeros((0,), dtype=bool)

    overall_vec = np.zeros((len(ya),), dtype=np.int32)
    overall_vec[ow_idx] = ab_joint_ow.astype(np.int32)
    overall_vec[c_idx] = catch_ok.astype(np.int32)

    n_spks = int(len({str(x) for x in ev_d["primary_speaker_id"].tolist()})) if "primary_speaker_id" in ev_d.files else None
    return {
        "eval_shard": eval_npz.name,
        "n_pairs": int(len(ya)),
        "n_overwrite_pairs": int(len(ow_idx)),
        "n_catch_pairs": int(len(c_idx)),
        "n_unique_primary_speakers": n_spks,
        "seq_len_ticks_12p5hz": int(ev_ca.shape[1]),
        "counterfactual_ab_flip_and_preserve": round(float(np.mean(ab_joint_ow)), 4) if len(ab_joint_ow) else 0.0,
        "updated_attr_ab_flip_acc": round(float(np.mean(upd_ok_ow)), 4) if len(upd_ok_ow) else 0.0,
        "preserved_attr_ab_acc": round(float(np.mean(pres_ok_ow)), 4) if len(pres_ok_ow) else 0.0,
        "catch_dual_preserve_acc": round(float(np.mean(catch_ok)), 4) if len(catch_ok) else 0.0,
        "overall_joint_acc": round(float(np.mean(overall_vec)), 4),
        "mean_nowrite_on_eval": round(0.5 * (nw_a + nw_b), 4),
        "episode_overall_joint_vector": [int(x) for x in overall_vec],
    }


def write_deterministic_inference_showcase(
    out_dir: Path,
    *,
    split: str,
    model_name: str,
    episodes: list[dict[str, Any]],
    metric_key: str,
    higher_is_better: bool = True,
) -> Path:
    """Write AGENTS.md §13.2 deterministic uncurated `inference_showcase/` manifest selecting P05/P25/P50/P75/P95."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not episodes:
        raise ValueError("Cannot build inference_showcase from empty episode list.")
    ordered = sorted(
        episodes,
        key=lambda e: (float(e[metric_key]), str(e.get("episode_id", ""))),
        reverse=not higher_is_better,
    )
    n = len(ordered)
    quantiles = {}
    for q_name, q_frac in (("P05_failure", 0.05), ("P25", 0.25), ("P50_median", 0.50), ("P75", 0.75), ("P95_best", 0.95)):
        qi = min(n - 1, max(0, int(round(q_frac * (n - 1)))))
        ep = dict(ordered[qi])
        ep["selected_quantile"] = q_name
        ep["rank_index"] = int(qi)
        quantiles[q_name] = ep

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "AGENTS.md §13.2 Deterministic Uncurated Quantile Showcase (Frozen Weights, Zero Best-of-N)",
        "split": split,
        "model_name": model_name,
        "ranking_metric": metric_key,
        "higher_is_better": bool(higher_is_better),
        "total_ranked_episodes": int(n),
        "selected_quantiles": quantiles,
    }
    p = out_dir / f"showcase_manifest_{split}_{model_name}.json"
    p.write_text(json.dumps(manifest, indent=2) + "\n")
    return p


def generate_cross_modal_text_showcase(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    shard_path: Path,
    *,
    out_dir: Path,
    split: str,
    model_name: str,
    n_episodes: int = 32,
    seq_len: int = 512,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
) -> dict[str, Any]:
    """Generate AGENTS.md §13.2 uncurated P05/P25/P50/P75/P95 qualitative cross-modal caption/text completions."""
    if not shard_path.is_file():
        return {"skipped": f"missing {shard_path}"}
    sh = LoadedSparseShard.load(shard_path)
    d = sh.data
    n_tot = sh.n_ticks
    rst = np.nonzero(d["reset_mask"])[0]
    ends = np.append(rst[1:], n_tot)
    n_ep = min(n_episodes, len(rst))

    @jax.jit
    def _fwd_one(ap, batch):
        z, st_end, dm = scan_core_fast(
            ap["core"],
            init_core_state(spec, 1),
            batch,
            spec,
            remat_layers=False,
            matmul_dtype=matmul_dtype,
            use_wy_chunking=use_wy_chunking,
        )
        logits = predict_any2any_heads(ap["heads"], z, spec, matmul_dtype=matmul_dtype)["text_logits"]
        # Return pure JAX arrays (NEVER call Python `float()` inside `@jax.jit`!)
        return logits[:, 0, :], st_end, jnp.mean(dm["mean_nowrite"])

    @jax.jit
    def _free_run_32(ap, st_prefix, start_byte):
        """Strict free-running O(1) autoregressive greedy rollout (`32` bytes) on frozen `step_core`!"""
        def _gen_step(carry, _):
            st_c, cur_b = carry
            pkt = {
                "modality_id": jnp.full((1,), 2, dtype=jnp.int32),
                "dt_s": jnp.full((1,), 0.02, dtype=jnp.float32),
                "valid_mask": jnp.ones((1,), dtype=jnp.float32),
                "reliability": jnp.ones((1,), dtype=jnp.float32),
                "reset_mask": jnp.zeros((1,), dtype=jnp.bool_),
                "spatial_pos": jnp.zeros((1,), dtype=jnp.int32),
                "text_byte": cur_b.reshape((1,)),
                "has_true_action": jnp.zeros((1,), dtype=jnp.float32),
                "video_feat": jnp.zeros((1, spec.video_patch_dim), dtype=jnp.float32),
                "audio_codes": jnp.zeros((1, spec.audio_codebooks), dtype=jnp.int32),
                "proprio_feat": jnp.zeros((1, spec.proprio_dim), dtype=jnp.float32),
                "action_feat": jnp.zeros((1, spec.action_dim), dtype=jnp.float32),
                "probe_feat": jnp.zeros((1, spec.probe_dim), dtype=jnp.float32),
            }
            z_t, st_next, _ = step_core(ap["core"], st_c, pkt, spec)
            lg = predict_any2any_heads(ap["heads"], z_t[None, :, :], spec, matmul_dtype=matmul_dtype)["text_logits"][0, 0]
            nxt = jnp.argmax(lg).astype(jnp.int32)
            return (st_next, nxt), nxt

        _, gen_bytes = jax.lax.scan(_gen_step, (st_prefix, start_byte.astype(jnp.int32)), None, length=32)
        return gen_bytes

    ep_records: list[dict[str, Any]] = []
    for ei in range(n_ep):
        s, e = int(rst[ei]), int(ends[ei])
        L = min(seq_len, e - s)
        if L < 12:
            continue
        idx = np.arange(s, s + L, dtype=np.int64)[:, None]
        pad_n = seq_len - L
        def _pad(arr, val=0):
            if pad_n == 0:
                return arr
            return np.concatenate([arr, np.full((pad_n,) + arr.shape[1:], val, dtype=arr.dtype)], axis=0)

        mod = _pad(d["modality_id"][idx].astype(np.int32), 2)
        dt = _pad(d["dt_s"][idx].astype(np.float32), 0.0)
        vld = np.zeros((seq_len, 1), dtype=np.float32)
        vld[:L, 0] = d["valid_mask"][idx[:, 0]].astype(np.float32)
        rel = _pad(d["reliability"][idx].astype(np.float32), 1.0)
        rst_a = np.zeros((seq_len, 1), dtype=bool)
        rst_a[0, 0] = True
        sp = _pad(d["spatial_pos"][idx].astype(np.int32), 0)
        tb = _pad(d["text_byte"][idx].astype(np.int32), 256)
        v_idx = d["video_idx"][idx]
        a_idx = d["audio_idx"][idx]
        vid = np.zeros((seq_len, 1, spec.video_patch_dim), dtype=np.float32)
        vm = v_idx >= 0
        if np.any(vm) and len(d["video_feat"]) > 0:
            vid[:L][vm] = d["video_feat"][v_idx[vm]].astype(np.float32)
        aud = np.zeros((seq_len, 1, spec.audio_codebooks), dtype=np.int32)
        am = a_idx >= 0
        if np.any(am) and len(d["audio_codes"]) > 0:
            aud[:L][am] = d["audio_codes"][a_idx[am]].astype(np.int32)

        batch = {
            "modality_id": jnp.asarray(mod),
            "dt_s": jnp.asarray(dt),
            "valid_mask": jnp.asarray(vld),
            "reliability": jnp.asarray(rel),
            "reset_mask": jnp.asarray(rst_a),
            "spatial_pos": jnp.asarray(sp),
            "text_byte": jnp.asarray(tb),
            "has_true_action": jnp.zeros((seq_len, 1), dtype=jnp.float32),
            "video_feat": jnp.asarray(vid),
            "audio_codes": jnp.asarray(aud),
            "proprio_feat": jnp.zeros((seq_len, 1, spec.proprio_dim), dtype=jnp.float32),
            "action_feat": jnp.zeros((seq_len, 1, spec.action_dim), dtype=jnp.float32),
            "probe_feat": jnp.zeros((seq_len, 1, spec.probe_dim), dtype=jnp.float32),
        }
        logits, _, mean_nw_arr = _fwd_one(all_params, batch)
        mean_nw = float(mean_nw_arr)
        logits_np = np.asarray(logits[:L], dtype=np.float32)
        mod_1d = mod[:L, 0]
        tb_1d = tb[:L, 0]
        txt_pos = np.where((mod_1d == 2) & (np.arange(L) > 0))[0]
        if len(txt_pos) < 4:
            continue
        pl = logits_np[txt_pos - 1]
        yb = tb_1d[txt_pos]
        ml = np.max(pl, axis=-1, keepdims=True)
        logp = pl - ml - np.log(np.sum(np.exp(pl - ml), axis=-1, keepdims=True))
        bpb = float(np.mean(-logp[np.arange(len(yb)), yb])) / math.log(2.0)
        pred_b = np.argmax(pl, axis=-1)

        # Free-running greedy continuation from prefix up to first 16 text bytes (keeps source audio/video intact!)
        p_cut = int(txt_pos[min(15, len(txt_pos) - 1)])
        vld_pref = np.zeros((seq_len, 1), dtype=np.float32)
        vld_pref[:p_cut, 0] = vld[:p_cut, 0]
        dt_pref = dt.copy()
        dt_pref[p_cut:, 0] = 0.0
        pref_batch = {**batch, "valid_mask": jnp.asarray(vld_pref), "dt_s": jnp.asarray(dt_pref)}
        _, st_pref, _ = _fwd_one(all_params, pref_batch)
        gen_bytes = np.asarray(_free_run_32(all_params, st_pref, jnp.asarray(int(tb_1d[p_cut]), dtype=jnp.int32)))

        ep_records.append(
            {
                "episode_id": f"{shard_path.stem}_ep{ei:04d}",
                "observed_modalities": sorted({int(x) for x in mod_1d}),
                "channel_provenance": {
                    "input_context": "observed",
                    "free_running_generation": "predicted_autoregressive_frozen_core",
                    "teacher_forced_readout": "predicted_diagnostic_forced",
                },
                "text_bpb": round(bpb, 4),
                "mean_nowrite": round(mean_nw, 4),
                "ground_truth_text_preview": decode_utf8_bytes(yb)[:160],
                "free_running_32b_continuation": decode_utf8_bytes(gen_bytes),
                "top1_next_byte_preview": decode_utf8_bytes(pred_b)[:160],
            }
        )

    if not ep_records:
        return {"skipped": f"no valid text episodes in {shard_path.name}"}
    manifest_p = write_deterministic_inference_showcase(
        out_dir,
        split=split,
        model_name=f"{model_name}_{shard_path.stem}",
        episodes=ep_records,
        metric_key="text_bpb",
        higher_is_better=False,
    )
    return {"manifest_path": str(manifest_p), "n_ranked_episodes": len(ep_records)}
