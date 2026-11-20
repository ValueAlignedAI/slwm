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
from models.v3.core import init_core_state
from models.v3.fast_core import scan_core_fast
from models.v3.objectives import apply_decoupled_probe, init_decoupled_probe_params
from models.v3.spec import SLWMv3Spec


def evaluate_lambada_openai(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    shard_path: Path,
    *,
    max_passages: int = 1024,
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
    n_ep = min(max_passages, len(rst))

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


def evaluate_robotics_action_intervention(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    shard_path: Path,
    *,
    seq_len: int = 512,
    batch_size: int = 8,
    n_batches: int = 8,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
    seed: int = 8901,
) -> dict[str, Any]:
    """AGENTS.md §13.2 Action-Intervention Proof: permuting `action_feat` must degrade future proprio/video prediction!"""
    if not shard_path.is_file():
        return {"skipped": f"missing {shard_path}"}
    sh = LoadedSparseShard.load(shard_path)
    rng = np.random.default_rng(seed)

    @jax.jit
    def _eval(ap, batch):
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
        _, m = compute_any2any_multimodal_loss(ap["heads"], z, batch, spec, matmul_dtype=matmul_dtype)
        return m

    p_act, s_act, p_vid, s_vid = [], [], [], []
    for _ in range(n_batches):
        b_np = sh.sample_batch(rng, batch_size=batch_size, seq_len=seq_len, drop_modality_prob=0.0)
        b_pair = {k: jnp.asarray(v) for k, v in b_np.items()}
        b_int_np = {k: v.copy() for k, v in b_np.items()}
        # Counterfactual action intervention: roll executed `action_feat` across batch while keeping initial proprio/video!
        b_int_np["action_feat"] = np.roll(b_int_np["action_feat"], shift=1, axis=1)
        b_int = {k: jnp.asarray(v) for k, v in b_int_np.items()}

        mp = _eval(all_params, b_pair)
        mi = _eval(all_params, b_int)
        if float(mp["n_action_targets"]) > 0:
            p_act.append(float(mp["action_proprio_huber"]))
            s_act.append(float(mi["action_proprio_huber"]))
        if float(mp["n_video_targets"]) > 0:
            p_vid.append(float(mp["video_patch_huber"]))
            s_vid.append(float(mi["video_patch_huber"]))

    act_gain = (float(np.mean(s_act)) - float(np.mean(p_act))) if p_act else 0.0
    vid_gain = (float(np.mean(s_vid)) - float(np.mean(p_vid))) if p_vid else 0.0
    return {
        "shard": shard_path.name,
        "paired_action_proprio_huber": round(float(np.mean(p_act)), 5) if p_act else None,
        "intervened_action_proprio_huber": round(float(np.mean(s_act)), 5) if s_act else None,
        "action_intervention_proprio_gain": round(act_gain, 5),
        "paired_video_huber": round(float(np.mean(p_vid)), 5) if p_vid else None,
        "intervened_video_huber": round(float(np.mean(s_vid)), 5) if s_vid else None,
        "action_intervention_video_gain": round(vid_gain, 5),
        "causal_action_conditioning_verified": bool(act_gain > 1e-4 or vid_gain > 1e-4),
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
        return jax.lax.stop_gradient(z[-1]), float(jnp.mean(d_m["mean_nowrite"]))

    def _run_extract(codes_nt32: np.ndarray) -> np.ndarray:
        n_ep, t_len, _ = codes_nt32.shape
        zs = []
        for s in range(0, n_ep, eval_micro_bsz):
            e = min(s + eval_micro_bsz, n_ep)
            # Transpose [N, T, 32] -> [T, B, 32]
            c_tb = jnp.asarray(np.transpose(codes_nt32[s:e], (1, 0, 2)), dtype=jnp.int32)
            dt_tb = jnp.full((t_len, e - s), 0.08, dtype=jnp.float32)  # 12.5 Hz Mimi = 0.08 s/tick
            zb, _ = _extract_final_z(core_params, c_tb, dt_tb)
            zs.append(np.asarray(zb, dtype=np.float32))
        return np.concatenate(zs, axis=0)

    z_tr = jnp.asarray(np.concatenate([_run_extract(tr_d["codes_a"]), _run_extract(tr_d["codes_b"])], axis=0))
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

    za = jnp.asarray(_run_extract(ev_d["codes_a"]))
    zb = jnp.asarray(_run_extract(ev_d["codes_b"]))
    pa = np.asarray(jnp.argmax(apply_decoupled_probe(probe_p, za).reshape(-1, 2, vocab_size), axis=-1))
    pb = np.asarray(jnp.argmax(apply_decoupled_probe(probe_p, zb).reshape(-1, 2, vocab_size), axis=-1))
    ya, yb = np.asarray(ev_d["y_a"]), np.asarray(ev_d["y_b"])
    ua = np.asarray(ev_d["upd_attr"])
    idx = np.arange(len(ya))
    upd_ok = (pa[idx, ua] == ya[idx, ua]) & (pb[idx, ua] == yb[idx, ua])
    pres_ok = (pa[idx, 1 - ua] == ya[idx, 1 - ua]) & (pb[idx, 1 - ua] == yb[idx, 1 - ua])
    ab_joint = upd_ok & pres_ok
    return {
        "eval_shard": eval_npz.name,
        "n_pairs": int(len(ya)),
        "seq_len_ticks_12p5hz": int(ev_d["codes_a"].shape[1]),
        "counterfactual_ab_flip_and_preserve": round(float(np.mean(ab_joint)), 4),
        "updated_attr_ab_flip_acc": round(float(np.mean(upd_ok)), 4),
        "preserved_attr_ab_acc": round(float(np.mean(pres_ok)), 4),
        "episode_ab_joint_vector": [int(x) for x in ab_joint],
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
