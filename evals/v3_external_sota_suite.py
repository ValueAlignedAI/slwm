"""SLWM-v3 Official External SoTA Execution & Free-Running Benchmark Suite (`Waves A-D`).

Evaluates strictly frozen `SLWM-v3` and matched comparator checkpoints (`no` online weight updates!)
under official external benchmark protocols:
  1. **Executed HumanEval & MBPP `pass@1` (`evaluate_code_execution_pass_at_1`):**
     Free-running O(1) autoregressive decoding via frozen `step_core` from prompt bytes, followed by
     isolated subprocess test-suite execution (`timeout=3.0s`) reporting executed `pass@1` & compile rate.
  2. **Official LibriSpeech `dev-clean` / `test-clean` Word Error Rate (`evaluate_librispeech_wer`):**
     Free-running Audio -> Text autoregressive decoding from Kyutai Mimi 32-RVQ speech streams,
     reporting exact normalized Levenshtein Word Error Rate (`WER`) and Character Error Rate (`CER`).
  3. **Official ESC-50 50-Class Environmental Audio Classification (`evaluate_esc50_classification`):**
     Decoupled frozen-core linear & MLP 50-class probe accuracy on `esc50_train_f14` (4-fold CV for dev,
     or fold-5 test when `--eval-sealed-test` is explicitly enabled).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from data.v3_unified_shards import BYTE_BOS, BYTE_EOS, BYTE_MOD_SEP, LoadedSparseShard, decode_utf8_bytes, encode_utf8_bytes
from models.v3.any2any_heads import predict_any2any_heads
from models.v3.core import init_core_state, step_core
from models.v3.fast_core import scan_core_fast
from models.v3.objectives import apply_decoupled_probe, init_decoupled_probe_params
from models.v3.spec import SLWMv3Spec


def _levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    m, n = len(seq_a), len(seq_b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        ai = seq_a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def _normalize_asr_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    return " ".join(s.split())


def _make_free_run_decoder(
    spec: SLWMv3Spec,
    max_gen_bytes: int,
    matmul_dtype: jnp.dtype,
):
    """Build a compiled free-running O(1) greedy byte decoder (`step_core`, strictly frozen weights!)."""

    @jax.jit
    def _decode(ap, st_prefix, start_byte):
        def _step_fn(carry, _):
            st_c, cur_b, done = carry
            pkt = {
                "modality_id": jnp.full((1,), 2, dtype=jnp.int32),
                "dt_s": jnp.full((1,), 0.02, dtype=jnp.float32),
                "valid_mask": jnp.where(done, 0.0, 1.0).reshape((1,)).astype(jnp.float32),
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
            nxt_out = jnp.where(done, jnp.asarray(BYTE_EOS, dtype=jnp.int32), nxt)
            done_next = done | (nxt_out == BYTE_EOS)
            return (st_next, nxt_out, done_next), nxt_out

        _, gen_bytes = jax.lax.scan(
            _step_fn,
            (st_prefix, start_byte.astype(jnp.int32), jnp.asarray(False)),
            None,
            length=int(max_gen_bytes),
        )
        return gen_bytes

    return _decode


def evaluate_librispeech_wer(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    shard_path: Path,
    *,
    max_utterances: int = 128,
    max_audio_ticks: int = 256,
    max_gen_bytes: int = 160,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
) -> dict[str, Any]:
    """Evaluate official free-running Audio -> Text WER & CER on LibriSpeech `dev-clean` / `test-clean`."""
    if not shard_path.is_file():
        return {"skipped": f"missing {shard_path}"}
    sh = LoadedSparseShard.load(shard_path)
    d = sh.data
    rst = np.nonzero(d["reset_mask"])[0]
    if len(rst) == 0 or rst[0] != 0:
        rst = np.insert(rst, 0, 0)
    ends = np.append(rst[1:], sh.n_ticks)
    n_utt = min(int(max_utterances), len(rst))

    @jax.jit
    def _encode_audio_prefix(ap, batch):
        _, st_end, _ = scan_core_fast(
            ap["core"],
            init_core_state(spec, 1),
            batch,
            spec,
            remat_layers=False,
            matmul_dtype=matmul_dtype,
            use_wy_chunking=use_wy_chunking,
        )
        return st_end

    decode_fn = _make_free_run_decoder(spec, max_gen_bytes=max_gen_bytes, matmul_dtype=matmul_dtype)

    tot_word_edits = 0
    tot_ref_words = 0
    tot_char_edits = 0
    tot_ref_chars = 0
    samples: list[dict[str, str]] = []

    for ei in range(n_utt):
        s, e = int(rst[ei]), int(ends[ei])
        m_ep = d["modality_id"][s:e]
        a_pos = np.where(m_ep == 1)[0]
        t_pos = np.where(m_ep == 2)[0]
        if len(a_pos) < 4 or len(t_pos) < 4:
            continue
        a_use = a_pos[:max_audio_ticks]
        la = len(a_use)
        a_idx = d["audio_idx"][s + a_use]
        ac_pad = np.zeros((max_audio_ticks, 1, spec.audio_codebooks), dtype=np.int32)
        valid_a = a_idx >= 0
        if np.any(valid_a):
            ac_pad[:la, 0][valid_a] = d["audio_codes"][a_idx[valid_a]].astype(np.int32)
        vld = np.zeros((max_audio_ticks, 1), dtype=np.float32)
        vld[:la, 0] = 1.0
        dt = np.zeros((max_audio_ticks, 1), dtype=np.float32)
        dt[:la, 0] = 0.08
        rst_a = np.zeros((max_audio_ticks, 1), dtype=bool)
        rst_a[0, 0] = True
        batch = {
            "modality_id": jnp.ones((max_audio_ticks, 1), dtype=jnp.int32),
            "dt_s": jnp.asarray(dt),
            "valid_mask": jnp.asarray(vld),
            "reliability": jnp.ones((max_audio_ticks, 1), dtype=jnp.float32),
            "reset_mask": jnp.asarray(rst_a),
            "spatial_pos": jnp.zeros((max_audio_ticks, 1), dtype=jnp.int32),
            "text_byte": jnp.full((max_audio_ticks, 1), 256, dtype=jnp.int32),
            "has_true_action": jnp.zeros((max_audio_ticks, 1), dtype=jnp.float32),
            "video_feat": jnp.zeros((max_audio_ticks, 1, spec.video_patch_dim), dtype=jnp.float32),
            "audio_codes": jnp.asarray(ac_pad),
            "proprio_feat": jnp.zeros((max_audio_ticks, 1, spec.proprio_dim), dtype=jnp.float32),
            "action_feat": jnp.zeros((max_audio_ticks, 1, spec.action_dim), dtype=jnp.float32),
            "probe_feat": jnp.zeros((max_audio_ticks, 1, spec.probe_dim), dtype=jnp.float32),
        }
        st_pref = _encode_audio_prefix(all_params, batch)
        start_tok = int(d["text_byte"][s + t_pos[0]])
        gen_b = np.asarray(decode_fn(all_params, st_pref, jnp.asarray(start_tok, dtype=jnp.int32)))
        eos_hits = np.where(gen_b == BYTE_EOS)[0]
        if len(eos_hits) > 0:
            gen_b = gen_b[: int(eos_hits[0])]
        hyp_str = _normalize_asr_text(decode_utf8_bytes(gen_b))
        ref_str = _normalize_asr_text(decode_utf8_bytes(d["text_byte"][s + t_pos]))
        if not ref_str:
            continue
        rw = ref_str.split()
        hw = hyp_str.split()
        tot_word_edits += _levenshtein(rw, hw)
        tot_ref_words += len(rw)
        tot_char_edits += _levenshtein(list(ref_str), list(hyp_str))
        tot_ref_chars += len(ref_str)
        if len(samples) < 5:
            samples.append({"episode_id": f"{shard_path.stem}_utt{ei:04d}", "ref": ref_str[:120], "hyp": hyp_str[:120]})

    return {
        "shard": shard_path.name,
        "evaluated_utterances": int(n_utt),
        "word_error_rate_wer": round(tot_word_edits / max(tot_ref_words, 1), 4),
        "char_error_rate_cer": round(tot_char_edits / max(tot_ref_chars, 1), 4),
        "total_ref_words": int(tot_ref_words),
        "sample_decodings": samples,
    }


def evaluate_esc50_classification(
    core_params: dict[str, Any],
    spec: SLWMv3Spec,
    train_shard_path: Path,
    eval_shard_path: Path | None = None,
    *,
    probe_steps: int = 250,
    eval_micro_bsz: int = 16,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
    seed: int = 9301,
) -> dict[str, Any]:
    """Evaluate decoupled frozen-core ESC-50 50-class environmental audio accuracy.

    If `eval_shard_path` is None, runs strictly dev-safe 4-fold internal split inside `train_shard_path`
    (leaving `esc50_test_f5.npz` completely sealed!). When `eval_shard_path` is provided on sealed test,
    trains decoupled probe on `esc50_train_f14.npz` and scores `esc50_test_f5.npz` (exact Fold 5 accuracy).
    """
    if not train_shard_path.is_file():
        return {"skipped": f"missing {train_shard_path}"}

    def _load_audio_and_labels(p: Path) -> tuple[np.ndarray, list[str]]:
        sh = LoadedSparseShard.load(p)
        d = sh.data
        rst = np.nonzero(d["reset_mask"])[0]
        if len(rst) == 0 or rst[0] != 0:
            rst = np.insert(rst, 0, 0)
        ends = np.append(rst[1:], sh.n_ticks)
        codes_list, labels = [], []
        max_len = 64  # ESC-50 clips are 5.0 s @ 12.5 Hz = ~63 Mimi ticks!
        for s, e in zip(rst, ends):
            m_ep = d["modality_id"][int(s) : int(e)]
            ai = d["audio_idx"][int(s) : int(e)][m_ep == 1]
            ai = ai[ai >= 0]
            tb = d["text_byte"][int(s) : int(e)][m_ep == 2]
            lbl = decode_utf8_bytes(tb).strip().lower()
            if len(ai) < 8 or not lbl:
                continue
            arr = np.zeros((max_len, spec.audio_codebooks), dtype=np.int32)
            L = min(max_len, len(ai))
            arr[:L] = d["audio_codes"][ai[:L]].astype(np.int32)
            codes_list.append(arr)
            labels.append(lbl)
        return np.stack(codes_list, axis=0), labels

    tr_codes, tr_lbls = _load_audio_and_labels(train_shard_path)
    if eval_shard_path is not None and eval_shard_path.is_file():
        ev_codes, ev_lbls = _load_audio_and_labels(eval_shard_path)
        mode = f"train_f14_to_{eval_shard_path.stem}"
    else:
        # Dev-safe deterministic 75/25 holdout strictly inside train_f14 (never touches test fold 5!)
        n = len(tr_lbls)
        ev_mask = (np.arange(n) % 4) == 3
        ev_codes = tr_codes[ev_mask]
        ev_lbls = [tr_lbls[i] for i in range(n) if ev_mask[i]]
        tr_codes = tr_codes[~ev_mask]
        tr_lbls = [tr_lbls[i] for i in range(n) if not ev_mask[i]]
        mode = "internal_dev_holdout_from_train_f14"

    classes = sorted(set(tr_lbls) | set(ev_lbls))
    c2i = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)
    y_tr = jnp.asarray([c2i[c] for c in tr_lbls], dtype=jnp.int32)
    y_ev = np.asarray([c2i[c] for c in ev_lbls], dtype=np.int32)

    @jax.jit
    def _extract_z(cp, ac_tb32):
        t_l, b_s, _ = ac_tb32.shape
        pkts = {
            "modality_id": jnp.ones((t_l, b_s), dtype=jnp.int32),
            "dt_s": jnp.full((t_l, b_s), 0.08, dtype=jnp.float32),
            "valid_mask": jnp.ones((t_l, b_s), dtype=jnp.float32),
            "reliability": jnp.ones((t_l, b_s), dtype=jnp.float32),
            "reset_mask": jnp.zeros((t_l, b_s), dtype=jnp.bool_).at[0].set(True),
            "spatial_pos": jnp.zeros((t_l, b_s), dtype=jnp.int32),
            "text_byte": jnp.full((t_l, b_s), 256, dtype=jnp.int32),
            "has_true_action": jnp.zeros((t_l, b_s), dtype=jnp.float32),
            "video_feat": jnp.zeros((t_l, b_s, spec.video_patch_dim), dtype=jnp.float32),
            "audio_codes": ac_tb32,
            "proprio_feat": jnp.zeros((t_l, b_s, spec.proprio_dim), dtype=jnp.float32),
            "action_feat": jnp.zeros((t_l, b_s, spec.action_dim), dtype=jnp.float32),
            "probe_feat": jnp.zeros((t_l, b_s, spec.probe_dim), dtype=jnp.float32),
        }
        z, _, _ = scan_core_fast(
            cp, init_core_state(spec, b_s), pkts, spec,
            remat_layers=False, matmul_dtype=matmul_dtype, assume_all_valid=True, use_wy_chunking=use_wy_chunking,
        )
        # Mean-pool frozen temporal audio representation across clip
        return jax.lax.stop_gradient(jnp.mean(z, axis=0))

    def _run_all(codes_nt32: np.ndarray) -> np.ndarray:
        outs = []
        for s in range(0, len(codes_nt32), eval_micro_bsz):
            e = min(s + eval_micro_bsz, len(codes_nt32))
            c_tb = jnp.asarray(np.transpose(codes_nt32[s:e], (1, 0, 2)), dtype=jnp.int32)
            outs.append(np.asarray(_extract_z(core_params, c_tb), dtype=np.float32))
        return np.concatenate(outs, axis=0)

    z_tr_np = _run_all(tr_codes)
    z_ev_np = _run_all(ev_codes)
    y_tr_np = np.asarray([c2i[c] for c in tr_lbls], dtype=np.int32)

    # 1. Width-invariant standardized + L2-normalized closed-form 4-fold train-CV Ridge Linear Probe
    # (Fixes d/N overfitting across d=512 -> 768 -> 1536 -> 1920 with ZERO test-fold leakage!)
    mu = z_tr_np.mean(axis=0, keepdims=True)
    sd = z_tr_np.std(axis=0, keepdims=True) + 1e-5
    x_tr_s = (z_tr_np - mu) / sd
    x_ev_s = (z_ev_np - mu) / sd
    x_tr_n = x_tr_s / (np.linalg.norm(x_tr_s, axis=-1, keepdims=True) + 1e-6)
    x_ev_n = x_ev_s / (np.linalg.norm(x_ev_s, axis=-1, keepdims=True) + 1e-6)

    def _fit_ridge(X_fit: np.ndarray, y_fit: np.ndarray, lam: float) -> np.ndarray:
        n_f, d_f = X_fit.shape
        Y = np.zeros((n_f, n_cls), dtype=np.float32)
        Y[np.arange(n_f), y_fit] = 1.0
        # Dual solve when d > n (O(n^3) fast & exact for d=1536/1920!), primal otherwise
        if d_f > n_f:
            K = X_fit @ X_fit.T + float(lam) * np.eye(n_f, dtype=np.float32)
            alpha = np.linalg.solve(K, Y)
            return X_fit.T @ alpha
        A = X_fit.T @ X_fit + float(lam) * np.eye(d_f, dtype=np.float32)
        return np.linalg.solve(A, X_fit.T @ Y)

    n_tr_all = len(y_tr_np)
    cv_folds = np.arange(n_tr_all) % 4
    best_lam = 1.0
    best_cv = -1.0
    for lam in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        cv_hits = 0
        for f_id in range(4):
            val_m = cv_folds == f_id
            W_f = _fit_ridge(x_tr_n[~val_m], y_tr_np[~val_m], lam)
            preds_f = np.argmax(x_tr_n[val_m] @ W_f, axis=-1)
            cv_hits += int(np.sum(preds_f == y_tr_np[val_m]))
        cv_acc = cv_hits / max(n_tr_all, 1)
        if cv_acc > best_cv:
            best_cv = cv_acc
            best_lam = lam

    W_full = _fit_ridge(x_tr_n, y_tr_np, best_lam)
    logits_lin = x_ev_n @ W_full
    top1_lin = float(np.mean(np.argmax(logits_lin, axis=-1) == y_ev))
    top5_lin_idx = np.argsort(-logits_lin, axis=-1)[:, : min(5, n_cls)]
    top5_lin = float(np.mean([y_ev[i] in top5_lin_idx[i] for i in range(len(y_ev))]))

    # 2. Width-scaled standardized MLP Probe (lr scaled by 1/sqrt(d/512) + stronger L2 decay)
    z_tr = jnp.asarray(x_tr_n)
    z_ev = jnp.asarray(x_ev_n)
    probe_lr = 3e-3 / math.sqrt(max(1.0, float(spec.d_model) / 512.0))
    probe_p = init_decoupled_probe_params(jax.random.PRNGKey(seed), spec.d_model, out_dim=n_cls, hidden_dim=256)
    tx_p = optax.adamw(probe_lr, weight_decay=2e-2)
    opt_p = tx_p.init(probe_p)

    @jax.jit
    def _pstep(pp, ost):
        def _loss(pr):
            lg = apply_decoupled_probe(pr, z_tr)
            return optax.softmax_cross_entropy_with_integer_labels(lg, y_tr).mean()

        lv, g = jax.value_and_grad(_loss)(pp)
        u, nost = tx_p.update(g, ost, pp)
        return optax.apply_updates(pp, u), nost, lv

    for _ in range(probe_steps):
        probe_p, opt_p, _ = _pstep(probe_p, opt_p)

    logits_ev = np.asarray(apply_decoupled_probe(probe_p, z_ev), dtype=np.float32)
    top1_mlp = float(np.mean(np.argmax(logits_ev, axis=-1) == y_ev))
    top5_mlp_idx = np.argsort(-logits_ev, axis=-1)[:, : min(5, n_cls)]
    top5_mlp = float(np.mean([y_ev[i] in top5_mlp_idx[i] for i in range(len(y_ev))]))
    return {
        "eval_mode": mode,
        "n_classes": int(n_cls),
        "n_train_clips": int(len(tr_lbls)),
        "n_eval_clips": int(len(ev_lbls)),
        "top1_accuracy": round(top1_lin, 4),
        "top5_accuracy": round(top5_lin, 4),
        "top1_linear_cv": round(top1_lin, 4),
        "top5_linear_cv": round(top5_lin, 4),
        "top1_mlp_std": round(top1_mlp, 4),
        "top5_mlp_std": round(top5_mlp, 4),
        "best_ridge_lambda": float(best_lam),
        "train_4fold_cv_acc": round(float(best_cv), 4),
        "chance_baseline": round(1.0 / max(n_cls, 1), 4),
    }


def evaluate_code_execution_pass_at_1(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    problems: list[dict[str, str]],
    *,
    max_prompt_bytes: int = 512,
    max_gen_bytes: int = 256,
    timeout_s: float = 3.0,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
) -> dict[str, Any]:
    """Official executed HumanEval / MBPP `pass@1` via isolated subprocess test execution on frozen-core greedy completions.

    Each item in `problems` has `{"task_id": str, "prompt": str, "test_code": str}`.
    """
    if not problems:
        return {"evaluated_problems": 0, "pass_at_1": 0.0}

    @jax.jit
    def _encode_prompt(ap, tb_pad, vld_pad):
        t_l = tb_pad.shape[0]
        batch = {
            "modality_id": jnp.full((t_l, 1), 2, dtype=jnp.int32),
            "dt_s": jnp.where(vld_pad > 0.5, 0.02, 0.0).astype(jnp.float32),
            "valid_mask": vld_pad.astype(jnp.float32),
            "reliability": jnp.ones((t_l, 1), dtype=jnp.float32),
            "reset_mask": jnp.zeros((t_l, 1), dtype=jnp.bool_).at[0].set(True),
            "spatial_pos": jnp.zeros((t_l, 1), dtype=jnp.int32),
            "text_byte": tb_pad.astype(jnp.int32),
            "has_true_action": jnp.zeros((t_l, 1), dtype=jnp.float32),
            "video_feat": jnp.zeros((t_l, 1, spec.video_patch_dim), dtype=jnp.float32),
            "audio_codes": jnp.zeros((t_l, 1, spec.audio_codebooks), dtype=jnp.int32),
            "proprio_feat": jnp.zeros((t_l, 1, spec.proprio_dim), dtype=jnp.float32),
            "action_feat": jnp.zeros((t_l, 1, spec.action_dim), dtype=jnp.float32),
            "probe_feat": jnp.zeros((t_l, 1, spec.probe_dim), dtype=jnp.float32),
        }
        _, st_end, _ = scan_core_fast(
            ap["core"], init_core_state(spec, 1), batch, spec,
            remat_layers=False, matmul_dtype=matmul_dtype, use_wy_chunking=use_wy_chunking,
        )
        return st_end

    decode_fn = _make_free_run_decoder(spec, max_gen_bytes=max_gen_bytes, matmul_dtype=matmul_dtype)

    n_pass = 0
    n_syntax_ok = 0
    for prob in problems:
        prompt = prob["prompt"]
        enc = encode_utf8_bytes(prompt, bos=BYTE_BOS, eos=-1, max_bytes=max_prompt_bytes - 1)
        L = len(enc)
        tb_pad = np.full((max_prompt_bytes, 1), 256, dtype=np.int32)
        vld_pad = np.zeros((max_prompt_bytes, 1), dtype=np.float32)
        tb_pad[:L, 0] = enc
        vld_pad[:L, 0] = 1.0
        st_pref = _encode_prompt(all_params, jnp.asarray(tb_pad), jnp.asarray(vld_pad))
        gen_b = np.asarray(decode_fn(all_params, st_pref, jnp.asarray(int(BYTE_MOD_SEP), dtype=jnp.int32)))
        eos_hits = np.where(gen_b == BYTE_EOS)[0]
        if len(eos_hits) > 0:
            gen_b = gen_b[: int(eos_hits[0])]
        completion = decode_utf8_bytes(gen_b)
        # Truncate cleanly before any second top-level definition
        for stop_tok in ("\ndef ", "\nclass ", "\nif __name__"):
            if stop_tok in completion:
                completion = completion.split(stop_tok, 1)[0]
        full_src = prompt + completion + "\n" + prob["test_code"] + "\n"
        try:
            compile(prompt + completion, "<gen>", "exec")
            n_syntax_ok += 1
        except SyntaxError:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(full_src)
            tmp_p = Path(tf.name)
        try:
            res = subprocess.run([sys.executable, str(tmp_p)], capture_output=True, timeout=timeout_s, check=False)
            if res.returncode == 0:
                n_pass += 1
        except subprocess.TimeoutExpired:
            pass
        finally:
            tmp_p.unlink(missing_ok=True)

    n = len(problems)
    return {
        "evaluated_problems": int(n),
        "syntax_valid_rate": round(n_syntax_ok / max(n, 1), 4),
        "executed_pass_at_1": round(n_pass / max(n, 1), 4),
        "n_passed": int(n_pass),
    }


def _lcs_len(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if ai == b[j - 1] else (prev[j] if prev[j] >= curr[j - 1] else curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def evaluate_cross_modal_caption_quality(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    shard_path: Path,
    *,
    max_episodes: int = 48,
    max_ctx_ticks: int = 384,
    max_gen_bytes: int = 112,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
) -> dict[str, Any]:
    """Evaluate free-running Audio->Text / Video->Text caption quality (Token F1, ROUGE-L, & shuffled-context divergence)."""
    if not shard_path.is_file():
        return {"skipped": f"missing {shard_path}"}
    sh = LoadedSparseShard.load(shard_path)
    d = sh.data
    n_eps = min(int(max_episodes), len(sh.episode_slices))

    @jax.jit
    def _encode_ctx(ap, batch):
        _, st_end, _ = scan_core_fast(
            ap["core"],
            init_core_state(spec, 1),
            batch,
            spec,
            remat_layers=False,
            matmul_dtype=matmul_dtype,
            use_wy_chunking=use_wy_chunking,
        )
        return st_end

    decode_fn = _make_free_run_decoder(spec, max_gen_bytes=max_gen_bytes, matmul_dtype=matmul_dtype)

    def _build_ctx_batch(s0: int, ctx_len: int) -> dict[str, jax.Array]:
        L = min(max_ctx_ticks, max(1, ctx_len))
        sl = slice(s0, s0 + L)
        m_pad = np.full((max_ctx_ticks, 1), 2, dtype=np.int32)
        m_pad[:L, 0] = d["modality_id"][sl]
        dt_pad = np.zeros((max_ctx_ticks, 1), dtype=np.float32)
        dt_pad[:L, 0] = d["dt_s"][sl]
        vld = np.zeros((max_ctx_ticks, 1), dtype=np.float32)
        vld[:L, 0] = 1.0
        rel = np.ones((max_ctx_ticks, 1), dtype=np.float32)
        rel[:L, 0] = d["reliability"][sl]
        rst = np.zeros((max_ctx_ticks, 1), dtype=bool)
        rst[0, 0] = True
        sp = np.zeros((max_ctx_ticks, 1), dtype=np.int32)
        sp[:L, 0] = d["spatial_pos"][sl]
        tb = np.full((max_ctx_ticks, 1), 256, dtype=np.int32)
        tb[:L, 0] = d["text_byte"][sl]
        vf = np.zeros((max_ctx_ticks, 1, spec.video_patch_dim), dtype=np.float32)
        vi = d["video_idx"][sl]
        vm = vi >= 0
        if np.any(vm) and len(d["video_feat"]) > 0:
            vf[:L, 0][vm] = d["video_feat"][vi[vm]].astype(np.float32)
        ac = np.zeros((max_ctx_ticks, 1, spec.audio_codebooks), dtype=np.int32)
        ai = d["audio_idx"][sl]
        am = ai >= 0
        if np.any(am) and len(d["audio_codes"]) > 0:
            ac[:L, 0][am] = d["audio_codes"][ai[am]].astype(np.int32)
        return {
            "modality_id": jnp.asarray(m_pad),
            "dt_s": jnp.asarray(dt_pad),
            "valid_mask": jnp.asarray(vld),
            "reliability": jnp.asarray(rel),
            "reset_mask": jnp.asarray(rst),
            "spatial_pos": jnp.asarray(sp),
            "text_byte": jnp.asarray(tb),
            "has_true_action": jnp.zeros((max_ctx_ticks, 1), dtype=jnp.float32),
            "video_feat": jnp.asarray(vf),
            "audio_codes": jnp.asarray(ac),
            "proprio_feat": jnp.zeros((max_ctx_ticks, 1, spec.proprio_dim), dtype=jnp.float32),
            "action_feat": jnp.zeros((max_ctx_ticks, 1, spec.action_dim), dtype=jnp.float32),
            "probe_feat": jnp.zeros((max_ctx_ticks, 1, spec.probe_dim), dtype=jnp.float32),
        }

    f1s: list[float] = []
    rougels: list[float] = []
    hyps: list[str] = []
    refs: list[str] = []
    for ei in range(n_eps):
        s0, e0 = sh.episode_slices[ei]
        tb_ep = d["text_byte"][s0:e0]
        sep = np.where(tb_ep == BYTE_MOD_SEP)[0]
        if len(sep) == 0 or int(sep[0]) < 8:
            continue
        ctx_len = int(sep[0])
        ref_str = _normalize_asr_text(decode_utf8_bytes(tb_ep[ctx_len + 1 :]))
        if not ref_str:
            continue
        st_pref = _encode_ctx(all_params, _build_ctx_batch(s0, ctx_len))
        gen_b = np.asarray(decode_fn(all_params, st_pref, jnp.asarray(int(BYTE_MOD_SEP), dtype=jnp.int32)))
        eos_hits = np.where(gen_b == BYTE_EOS)[0]
        if len(eos_hits) > 0:
            gen_b = gen_b[: int(eos_hits[0])]
        hyp_str = _normalize_asr_text(decode_utf8_bytes(gen_b))
        rw = ref_str.split()
        hw = hyp_str.split()
        if rw and hw:
            r_set, h_set = set(rw), set(hw)
            ov = len(r_set & h_set)
            f1 = (2.0 * ov) / (len(r_set) + len(h_set))
            lcs = _lcs_len(rw, hw)
            rl = (2.0 * lcs) / (len(rw) + len(hw))
        else:
            f1, rl = 0.0, 0.0
        f1s.append(f1)
        rougels.append(rl)
        hyps.append(hyp_str)
        refs.append(ref_str)

    # Shuffled-reference F1 baseline: does the model score higher against its OWN clip's caption than a shifted clip?
    shuf_f1s: list[float] = []
    n_ok = len(hyps)
    for i in range(n_ok):
        hw = hyps[i].split()
        rw_cf = refs[(i + max(1, n_ok // 3)) % n_ok].split()
        if hw and rw_cf:
            ov = len(set(hw) & set(rw_cf))
            shuf_f1s.append((2.0 * ov) / (len(set(hw)) + len(set(rw_cf))))
        else:
            shuf_f1s.append(0.0)
    unique_hyps = len(set(h for h in hyps if h))
    return {
        "shard": shard_path.name,
        "evaluated_episodes": int(n_ok),
        "token_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "rouge_l_f1": round(float(np.mean(rougels)) if rougels else 0.0, 4),
        "shuffled_ref_token_f1": round(float(np.mean(shuf_f1s)) if shuf_f1s else 0.0, 4),
        "causal_caption_f1_margin": round((float(np.mean(f1s)) - float(np.mean(shuf_f1s))) if f1s else 0.0, 4),
        "unique_caption_ratio": round(unique_hyps / max(n_ok, 1), 4),
        "sample_captions": [{"ref": refs[i][:120], "hyp": hyps[i][:120]} for i in range(min(4, n_ok))],
    }
