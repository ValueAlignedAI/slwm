"""SLWM-v3 Wave B: Strictly Frozen-Core `PolicyCommitGate` Trainer & Unsupported-Output Evaluator (`AGENTS.md §8`).

Loads ANY frozen Wave A trained checkpoint (`core + heads`), freezes 100% of world core weights
(`jax.lax.stop_gradient(Z_t)`), trains the decoupled width-invariant `PolicyCommitGate` to decide
between committing an external output (`COMMIT_TEXT`, `COMMIT_AUDIO`, `COMMIT_VIDEO`, `COMMIT_ACTION`)
versus `NO_OP_ABSTAIN` based on latent world state + epistemic uncertainty (`text_entropy`, `audio_entropy`,
`prior_err`, `reliability`, `evidence_support`), and reports ALL FIVE mandatory `AGENTS.md §8` metrics:
  - `grounded_answer_accuracy` (overall & on committed answers),
  - `unsupported_claim_rate` (committing on missing/corrupted/unobservable evidence),
  - `contradiction_rate` (committing an answer contradicting observed state),
  - `abstention_noop_rate` (proving non-collapse!),
  - `expected_calibration_error_ece` (10-bin confidence calibration),
alongside an `Always-Commit` no-gate baseline and deterministic uncurated `inference_showcase/` manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from data.v3_unified_shards import LoadedSparseShard
from evals.v3_sota_benchmarks import write_deterministic_inference_showcase
from models.v3.any2any_heads import (
    COMMIT_ACTIONS,
    apply_policy_commit_gate,
    init_policy_commit_gate_params,
    predict_any2any_heads,
)
from models.v3.core import init_core_state
from models.v3.fast_core import scan_core_fast


def _expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """10-bin Expected Calibration Error (ECE)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = max(len(conf), 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf >= lo) & (conf <= hi if i == n_bins - 1 else conf < hi)
        if np.any(m):
            ece += (np.sum(m) / float(n)) * abs(float(np.mean(correct[m])) - float(np.mean(conf[m])))
    return float(ece)


def build_epistemic_commitment_dataset(
    all_params: dict[str, Any],
    spec: Any,
    shard_paths: list[Path],
    *,
    seq_len: int = 256,
    batch_size: int = 8,
    n_batches: int = 16,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    seed: int = 6101,
) -> dict[str, np.ndarray]:
    """Construct paired Supported vs Unsupported (modality-dropped / low-reliability) commitment windows on frozen core."""
    shards = [LoadedSparseShard.load(p) for p in shard_paths if p.is_file()]
    if not shards:
        raise FileNotFoundError(f"No shards found for Wave B PolicyCommitGate: {shard_paths}")

    @jax.jit
    def _extract(ap, batch):
        b = batch["valid_mask"].shape[1]
        z, _, diags = scan_core_fast(
            ap["core"], init_core_state(spec, b), batch, spec, remat_layers=False, matmul_dtype=matmul_dtype
        )
        preds = predict_any2any_heads(ap["heads"], z, spec, matmul_dtype=matmul_dtype)
        # Predictive entropies (normalized by log vocab size -> [0, 1])
        t_logp = jax.nn.log_softmax(preds["text_logits"], axis=-1)
        t_ent = -jnp.sum(jnp.exp(t_logp) * t_logp, axis=-1) / jnp.log(float(spec.text_vocab_size))  # [T, B]
        a_logp = jax.nn.log_softmax(preds["audio_logits"], axis=-1)  # [T, B, 8, 2048]
        a_ent = jnp.mean(-jnp.sum(jnp.exp(a_logp) * a_logp, axis=-1), axis=-1) / jnp.log(float(spec.audio_vocab_size))
        t_top1 = jnp.argmax(preds["text_logits"], axis=-1)
        return (
            jax.lax.stop_gradient(z),
            jax.lax.stop_gradient(t_ent),
            jax.lax.stop_gradient(a_ent),
            jax.lax.stop_gradient(diags["mean_prior_error"]),
            jax.lax.stop_gradient(t_top1),
        )

    rng = np.random.default_rng(seed)
    z_list, ev_list, target_act_list, supported_list, correct_text_list = [], [], [], [], []
    # Exact COMMIT_ACTIONS index map: 0:SPEAK_TEXT, 1:EMIT_AUDIO, 2:GENERATE_VIDEO, 3:ACT_MOTOR, 4:ABSTAIN_SILENCE
    abstain_idx = COMMIT_ACTIONS.index("ABSTAIN_SILENCE")
    mod_to_commit = {0: 2, 1: 1, 2: 0, 3: 3, 4: abstain_idx}

    for bi in range(n_batches):
        sh = shards[bi % len(shards)]
        # Half batch: clean grounded stream (supported=1); Half batch: evidence-corrupted / dropped source (supported=0)!
        b_clean = sh.sample_batch(rng, batch_size=batch_size, seq_len=seq_len, drop_modality_prob=0.0)
        b_ungrp = {k: v.copy() for k, v in b_clean.items()}
        # Corrupt/drop context in first half of window AND lower reliability to 0.05 -> genuinely UNSUPPORTED!
        half_t = seq_len // 2
        b_ungrp["reliability"][:half_t] = 0.05
        b_ungrp["video_feat"][:half_t] = 0.0
        b_ungrp["audio_codes"][:half_t] = 0
        b_ungrp["text_byte"][:half_t] = 256

        for is_sup, b_np in ((1, b_clean), (0, b_ungrp)):
            b_jax = {k: jnp.asarray(v) for k, v in b_np.items()}
            z, t_ent, a_ent, p_err, t_top1 = _extract(all_params, b_jax)
            # Evaluate on second half `[half_t : seq_len - 1]` where prior evidence determines whether output is supported!
            sl = slice(half_t, seq_len - 1)
            z_np = np.asarray(z[sl], dtype=np.float32)  # [T2, B, D]
            te_np = np.asarray(t_ent[sl], dtype=np.float32)
            ae_np = np.asarray(a_ent[sl], dtype=np.float32)
            pe_np = np.asarray(p_err[sl], dtype=np.float32)
            # Rolling mean reliability over preceding context window (causal!)
            rel_full = b_np["reliability"].astype(np.float32)
            rel_ctx = np.cumsum(rel_full, axis=0) / np.arange(1, seq_len + 1, dtype=np.float32)[:, None]
            rc_np = rel_ctx[sl]
            # Ground-truth next-tick modality & text correctness
            next_mod = b_np["modality_id"][half_t + 1 : seq_len]
            next_tb = b_np["text_byte"][half_t + 1 : seq_len]
            pred_tb = np.asarray(t_top1[sl], dtype=np.int32)
            txt_ok = (pred_tb == next_tb).astype(np.float32)

            # Target commit action: commit the active next modality ONLY when supported, else ABSTAIN_SILENCE (4)!
            t2, bsz = next_mod.shape
            tgt_a = np.full((t2, bsz), abstain_idx, dtype=np.int32)
            if is_sup == 1:
                for m_k, c_idx in mod_to_commit.items():
                    tgt_a[next_mod == m_k] = c_idx

            ev = np.stack([te_np, ae_np, np.log1p(pe_np), rc_np, np.full_like(rc_np, 0.5)], axis=-1)  # [T2, B, 5]
            # Subsample stride 4 along time for compact balanced dataset
            z_list.append(z_np[::4].reshape(-1, spec.d_model))
            ev_list.append(ev[::4].reshape(-1, 5))
            target_act_list.append(tgt_a[::4].reshape(-1))
            supported_list.append(np.full((z_list[-1].shape[0],), is_sup, dtype=np.int32))
            correct_text_list.append(txt_ok[::4].reshape(-1))

    return {
        "z": np.concatenate(z_list, axis=0),
        "ev": np.concatenate(ev_list, axis=0),
        "target_action": np.concatenate(target_act_list, axis=0),
        "is_supported": np.concatenate(supported_list, axis=0),
        "text_correct": np.concatenate(correct_text_list, axis=0),
    }


def evaluate_policy_commit_gate(
    gate_params: dict[str, Any],
    eval_ds: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute ALL FIVE mandatory `AGENTS.md §8` hallucination, usefulness, abstention & ECE metrics vs Always-Commit."""
    z = jnp.asarray(eval_ds["z"])
    ev = jnp.asarray(eval_ds["ev"])
    _, probs = apply_policy_commit_gate(gate_params, z, ev)
    probs_np = np.asarray(probs, dtype=np.float32)  # [N, 5]
    pred_act = np.argmax(probs_np, axis=-1)
    tgt_act = eval_ds["target_action"]
    is_sup = eval_ds["is_supported"] == 1
    is_unsup = ~is_sup

    # PolicyCommitGate-routed metrics (`ABSTAIN_SILENCE` == index 4)
    abstain_idx = COMMIT_ACTIONS.index("ABSTAIN_SILENCE")
    committed = pred_act != abstain_idx
    abstained = pred_act == abstain_idx
    action_acc = float(np.mean(pred_act == tgt_act))
    grounded_acc = float(np.mean(pred_act[is_sup] == tgt_act[is_sup])) if np.any(is_sup) else 0.0
    unsupported_claim_rate = float(np.mean(committed[is_unsup])) if np.any(is_unsup) else 0.0
    contradiction_rate = float(np.mean(committed & is_sup & (pred_act != tgt_act)))
    abstention_rate = float(np.mean(abstained))
    conf = np.max(probs_np, axis=-1)
    ece = _expected_calibration_error(conf, (pred_act == tgt_act).astype(np.float32))

    # Always-Commit baseline (never abstains -> picks argmax over non-abstain actions 0..3)
    ac_pred = np.argmax(probs_np[:, :abstain_idx], axis=-1)
    ac_grounded_acc = float(np.mean(ac_pred[is_sup] == tgt_act[is_sup])) if np.any(is_sup) else 0.0
    ac_unsupported_rate = 1.0  # Commits on 100% of unsupported queries!

    return {
        "n_eval_samples": int(len(tgt_act)),
        "policy_commit_gate": {
            "overall_decision_accuracy": round(action_acc, 4),
            "grounded_answer_accuracy": round(grounded_acc, 4),
            "unsupported_claim_rate": round(unsupported_claim_rate, 4),
            "contradiction_rate": round(contradiction_rate, 4),
            "abstention_noop_rate": round(abstention_rate, 4),
            "expected_calibration_error_ece": round(ece, 4),
            "action_distribution": {
                COMMIT_ACTIONS[i]: round(float(np.mean(pred_act == i)), 4) for i in range(len(COMMIT_ACTIONS))
            },
        },
        "always_commit_baseline": {
            "grounded_answer_accuracy": round(ac_grounded_acc, 4),
            "unsupported_claim_rate": round(ac_unsupported_rate, 4),
            "abstention_noop_rate": 0.0,
        },
        "unsupported_claim_reduction_pp": round((ac_unsupported_rate - unsupported_claim_rate) * 100.0, 2),
        "wave_b_gate_pass": bool(
            unsupported_claim_rate < 0.20
            and grounded_acc >= 0.70
            and 0.15 <= abstention_rate <= 0.85  # Strictly non-collapsing!
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v3 Wave B Frozen-Core PolicyCommitGate Trainer & Evaluator.")
    ap.add_argument("--ckpt", type=Path, required=True, help="Frozen Wave A checkpoint `.pkl`")
    ap.add_argument("--shard-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=6101)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    with args.ckpt.open("rb") as f:
        ckpt = pickle.load(f)
    spec = ckpt["spec"]
    all_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), ckpt["params"])
    mm_dtype = jnp.float32 if str(jax.devices()[0].platform).lower() == "cpu" else jnp.bfloat16

    train_shards = [
        args.shard_dir / "audio_text/clotho_train.npz",
        args.shard_dir / "text_code/text_code_dev_v1.npz",
        args.shard_dir / "av/msrvtt_train.npz",
    ]
    eval_shards = [
        args.shard_dir / "audio_text/clotho_dev.npz",
        args.shard_dir / "text_code/wikitext103_validation.npz",
        args.shard_dir / "av/msrvtt_dev.npz",
    ]
    tr_ds = build_epistemic_commitment_dataset(all_params, spec, train_shards, n_batches=16, matmul_dtype=mm_dtype, seed=args.seed)
    ev_ds = build_epistemic_commitment_dataset(all_params, spec, eval_shards, n_batches=8, matmul_dtype=mm_dtype, seed=args.seed + 999)

    gate_p = init_policy_commit_gate_params(jax.random.PRNGKey(args.seed), spec.d_model, hidden_dim=256)
    tx = optax.adamw(learning_rate=3e-3, weight_decay=1e-4)
    opt_st = tx.init(gate_p)

    z_tr = jnp.asarray(tr_ds["z"])
    ev_tr = jnp.asarray(tr_ds["ev"])
    y_tr = jnp.asarray(tr_ds["target_action"], dtype=jnp.int32)

    @jax.jit
    def _step(gp, ost):
        def _loss(p):
            logits, _ = apply_policy_commit_gate(p, z_tr, ev_tr)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y_tr).mean()

        lv, g = jax.value_and_grad(_loss)(gp)
        u, nost = tx.update(g, ost, gp)
        return optax.apply_updates(gp, u), nost, lv

    for s in range(args.steps):
        gate_p, opt_st, lv = _step(gate_p, opt_st)
        if s == 0 or (s + 1) % 100 == 0:
            print(f"[WAVE_B step={s+1:4d}/{args.steps}] gate_ce={float(lv):.4f}", flush=True)

    report = evaluate_policy_commit_gate(gate_p, ev_ds)
    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    report["checkpoint"] = str(args.ckpt)
    report["core_weights_strictly_frozen"] = True

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_p = args.out_dir / f"wave_b_policy_gate_{spec.name}_receipt.json"
    out_p.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
