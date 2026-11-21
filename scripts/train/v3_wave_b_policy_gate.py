"""SLWM-v3 Wave B: Strictly Frozen-Core Shortcut-Free `PolicyCommitGate` Trainer & Evaluator (`AGENTS.md §8`).

Loads ANY frozen Wave A checkpoint (`core + heads`), freezes 100% of world core weights
(`jax.lax.stop_gradient(Z_t)`), trains the decoupled width-invariant `PolicyCommitGate` strictly on
TRAIN shards (never dev!), and evaluates on DEV shards across TWO distinct regimes:

  1. **PRIMARY ZERO-SHORTCUT REGIME (`blind_reliability_unsupported`):**
     `reliability == 1.0` IDENTICALLY on BOTH supported and unsupported episodes!
     - Supported (`is_supported = 1`): genuine paired Audio/Video -> Caption or coherent context -> target.
     - Unsupported (`is_supported = 0`): silent ungrounded/conflicting source (source modality swapped from
       an unrelated episode OR masked without lowering `reliability`, keeping `reliability == 1.0`!).
     A trivial `reliability`-threshold baseline gets 0% separation here — `PolicyCommitGate` MUST rely on
     frozen core `Z_t`, predictive entropies (`text_entropy`, `audio_entropy`), and `prior_err`!
  2. **SECONDARY REGIME (`degraded_sensor_outage`):**
     Explicit sensor outage (`reliability = 0.05` + blank source).

Reports ALL FIVE mandatory `AGENTS.md §8` metrics (scoring ACTUAL content correctness & contradiction,
not merely modality ID!) against BOTH `Always-Commit` AND `Reliability-Only Threshold` baselines,
plus a deterministic `AGENTS.md §13.2` `inference_showcase/` manifest.
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

from data.v3_unified_shards import BYTE_PAD, LoadedSparseShard
from evals.v3_sota_benchmarks import _build_episode_aligned_batch, write_deterministic_inference_showcase
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


def make_blind_unsupported_batch(
    b_clean: dict[str, np.ndarray],
    qp_per_b: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct zero-shortcut `blind_reliability_unsupported` batch via modality-matched modulo donor wrapping.

    Holds `reliability == 1.0` identical to `b_clean` and NEVER injects `BYTE_PAD` or all-zero `audio_codes`
    onto active source ticks `[0 : qp_b]`, swapping matching real payloads (plus `spatial_pos`) from distinct donors!
    """
    b_unsup = {k: v.copy() for k, v in b_clean.items()}
    cur_b = b_clean["valid_mask"].shape[1]
    b_unsup["reliability"][:] = 1.0
    for b_i in range(cur_b):
        qp = int(qp_per_b[b_i])
        if qp <= 0:
            continue
        donor = (b_i + 1) % cur_b
        for m_src in (0, 1, 2):
            rec_idx = np.where((b_clean["modality_id"][:qp, b_i] == m_src) & (b_clean["valid_mask"][:qp, b_i] > 0.5))[0]
            if len(rec_idx) == 0:
                continue
            don_idx = np.zeros((0,), dtype=np.int64)
            chosen_don = donor
            for off in range(1, cur_b):
                cand = (b_i + off) % cur_b
                ci = np.where((b_clean["modality_id"][:, cand] == m_src) & (b_clean["valid_mask"][:, cand] > 0.5))[0]
                if len(ci) > 0:
                    don_idx = ci
                    chosen_don = cand
                    break
            if len(don_idx) == 0:
                continue
            mapped = don_idx[np.arange(len(rec_idx)) % len(don_idx)]
            if m_src == 0:
                b_unsup["video_feat"][rec_idx, b_i] = b_clean["video_feat"][mapped, chosen_don]
                b_unsup["spatial_pos"][rec_idx, b_i] = b_clean["spatial_pos"][mapped, chosen_don]
            elif m_src == 1:
                b_unsup["audio_codes"][rec_idx, b_i] = b_clean["audio_codes"][mapped, chosen_don]
            elif m_src == 2:
                b_unsup["text_byte"][rec_idx, b_i] = b_clean["text_byte"][mapped, chosen_don]
    return b_unsup


def build_epistemic_commitment_dataset(
    all_params: dict[str, Any],
    spec: Any,
    shard_paths: list[Path],
    *,
    regime: str = "blind_reliability_unsupported",
    seq_len: int = 256,
    batch_size: int = 8,
    n_batches: int = 16,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
    seed: int = 6101,
) -> dict[str, np.ndarray]:
    """Construct paired Supported vs Unsupported commitment windows on frozen core.

    `regime == "blind_reliability_unsupported"`: keeps `reliability == 1.0` IDENTICALLY on both
    supported and unsupported queries (eliminating the metadata-threshold shortcut!). Unsupported
    queries receive a silently mismatched/blanked source prefix while `reliability` stays `1.0`.
    `regime == "degraded_sensor_outage"`: lowers `reliability = 0.05` on blanked source prefix.
    """
    shards = [LoadedSparseShard.load(p) for p in shard_paths if p.is_file()]
    if not shards:
        raise FileNotFoundError(f"No shards found for Wave B PolicyCommitGate: {shard_paths}")

    @jax.jit
    def _extract(ap, batch):
        b = batch["valid_mask"].shape[1]
        z, _, diags = scan_core_fast(
            ap["core"],
            init_core_state(spec, b),
            batch,
            spec,
            remat_layers=False,
            matmul_dtype=matmul_dtype,
            use_wy_chunking=use_wy_chunking,
        )
        preds = predict_any2any_heads(ap["heads"], z, spec, matmul_dtype=matmul_dtype)
        t_logp = jax.nn.log_softmax(preds["text_logits"], axis=-1)
        t_ent = -jnp.sum(jnp.exp(t_logp) * t_logp, axis=-1) / jnp.log(float(spec.text_vocab_size))
        a_logp = jax.nn.log_softmax(preds["audio_logits"], axis=-1)
        a_ent = jnp.mean(-jnp.sum(jnp.exp(a_logp) * a_logp, axis=-1), axis=-1) / jnp.log(float(spec.audio_vocab_size))
        t_top1 = jnp.argmax(preds["text_logits"], axis=-1)
        t_maxp = jnp.max(jnp.exp(t_logp), axis=-1)
        a_top1_cb0 = jnp.argmax(preds["audio_logits"][:, :, 0, :], axis=-1)
        v_pred = preds["video_pred"]
        return (
            jax.lax.stop_gradient(z),
            jax.lax.stop_gradient(t_ent),
            jax.lax.stop_gradient(a_ent),
            jax.lax.stop_gradient(diags["mean_prior_error"]),
            jax.lax.stop_gradient(t_top1),
            jax.lax.stop_gradient(t_maxp),
            jax.lax.stop_gradient(a_top1_cb0),
            jax.lax.stop_gradient(v_pred),
        )

    rng = np.random.default_rng(seed)
    z_list, ev_list, target_act_list, supported_list, correct_content_list, ep_id_list = [], [], [], [], [], []
    abstain_idx = COMMIT_ACTIONS.index("ABSTAIN_SILENCE")
    mod_to_commit = {0: 2, 1: 1, 2: 0, 3: 3, 4: abstain_idx}

    for bi in range(n_batches):
        sh = shards[bi % len(shards)]
        d = sh.data
        rst = np.nonzero(d["reset_mask"])[0]
        if len(rst) == 0 or rst[0] != 0:
            rst = np.insert(rst, 0, 0)
        ends = np.append(rst[1:], sh.n_ticks)
        cur_b = min(int(batch_size), len(rst))
        if cur_b < 2:
            continue
        # Sample WITHOUT replacement so `donor != recipient` is strictly guaranteed!
        ep_indices = rng.choice(len(rst), size=(cur_b,), replace=False)
        chunk_slices: list[tuple[int, int]] = []
        for i in ep_indices:
            s_i, e_i = int(rst[i]), int(ends[i])
            # Target-anchored windowing for long AV clips (e.g. MSR-VTT where text starts at tick ~645 > seq_len!)
            if e_i - s_i > seq_len:
                m_full = d["modality_id"][s_i:e_i]
                t_pos = np.where(m_full == 2)[0]
                if len(t_pos) >= 4 and int(t_pos[0]) >= seq_len - 32:
                    win_s = max(0, int(t_pos[0]) - (seq_len * 3 // 4))
                    s_i = s_i + win_s
                    e_i = min(e_i, s_i + seq_len)
                else:
                    e_i = s_i + seq_len
            chunk_slices.append((s_i, e_i))
        b_clean = _build_episode_aligned_batch(sh, chunk_slices, spec, seq_len)

        # Build UNSUPPORTED counterpart using per-episode boundary `q_pos_b` (handles short AND long episodes!):
        qp_per_b = np.zeros((cur_b,), dtype=np.int32)
        eval_mask_tb = np.zeros((seq_len - 1, cur_b), dtype=bool)
        for b_i in range(cur_b):
            vld_len = int(np.sum(b_clean["valid_mask"][:, b_i] > 0.5))
            if vld_len < 8:
                continue
            mod_b = b_clean["modality_id"][:vld_len, b_i]
            txt_idx = np.where(mod_b == 2)[0]
            non_txt_idx = np.where(mod_b != 2)[0]
            if len(non_txt_idx) >= 4 and len(txt_idx) >= 4 and int(txt_idx[0]) > int(non_txt_idx[0]):
                qp = int(txt_idx[0])
            else:
                qp = max(4, vld_len // 2)
            qp = min(qp, vld_len - 2)
            qp_per_b[b_i] = qp
            q_end = min(qp + 16, vld_len - 1)
            eval_mask_tb[qp:q_end, b_i] = True

        if regime == "blind_reliability_unsupported":
            b_clean["reliability"][:] = 1.0
            b_unsup = make_blind_unsupported_batch(b_clean, qp_per_b)
        else:
            b_unsup = {k: v.copy() for k, v in b_clean.items()}
            for b_i in range(cur_b):
                qp = int(qp_per_b[b_i])
                if qp > 0:
                    b_unsup["reliability"][:qp, b_i] = 0.05
                    b_unsup["video_feat"][:qp, b_i] = 0.0
                    b_unsup["audio_codes"][:qp, b_i] = 0
                    b_unsup["text_byte"][:qp, b_i] = BYTE_PAD

        for is_sup, b_np in ((1, b_clean), (0, b_unsup)):
            b_jax = {k: jnp.asarray(v) for k, v in b_np.items()}
            z, t_ent, a_ent, p_err, t_top1, t_maxp, a_top1_cb0, v_pred = _extract(all_params, b_jax)
            z_np = np.asarray(z[:-1], dtype=np.float32)
            te_np = np.asarray(t_ent[:-1], dtype=np.float32)
            ae_np = np.asarray(a_ent[:-1], dtype=np.float32)
            pe_np = np.asarray(p_err[:-1], dtype=np.float32)
            mp_np = np.asarray(t_maxp[:-1], dtype=np.float32)
            rel_full = b_np["reliability"].astype(np.float32)
            rel_ctx = np.cumsum(rel_full, axis=0) / np.arange(1, seq_len + 1, dtype=np.float32)[:, None]
            rc_np = rel_ctx[:-1]

            vld_sl = eval_mask_tb & (b_np["valid_mask"][:-1] > 0.5) & (b_np["valid_mask"][1:] > 0.5)
            next_mod = b_np["modality_id"][1:]
            next_tb = b_np["text_byte"][1:]
            pred_tb = np.asarray(t_top1[:-1], dtype=np.int32)
            next_aud0 = b_np["audio_codes"][1:, :, 0]
            pred_aud0 = np.asarray(a_top1_cb0[:-1], dtype=np.int32)
            vp_np = np.asarray(v_pred[:-1], dtype=np.float32)
            vt_np = b_np["video_feat"][1:].astype(np.float32)
            v_cos = np.sum(vp_np * vt_np, axis=-1) / (
                np.linalg.norm(vp_np, axis=-1) * np.linalg.norm(vt_np, axis=-1) + 1e-8
            )
            # Modality-matched content correctness (NEVER grade video/audio ticks with text entropy!)
            content_ok = np.where(
                next_mod == 2,
                (pred_tb == next_tb).astype(np.float32),
                np.where(
                    next_mod == 1,
                    (pred_aud0 == next_aud0).astype(np.float32),
                    (v_cos > 0.5).astype(np.float32),
                ),
            )

            t2, bsz = next_mod.shape
            tgt_a = np.full((t2, bsz), abstain_idx, dtype=np.int32)
            if is_sup == 1:
                for m_k, c_idx in mod_to_commit.items():
                    tgt_a[next_mod == m_k] = c_idx

            # Evidence vector: [text_entropy, audio_entropy, log1p(prior_err), rel_ctx, top1_prob]
            ev = np.stack([te_np, ae_np, np.log1p(pe_np), rc_np, mp_np], axis=-1)
            flat_vld = vld_sl.reshape(-1)
            if not np.any(flat_vld):
                continue
            z_f = z_np.reshape(-1, spec.d_model)[flat_vld]
            ev_f = ev.reshape(-1, 5)[flat_vld]
            ta_f = tgt_a.reshape(-1)[flat_vld]
            ck_f = content_ok.reshape(-1)[flat_vld]
            z_list.append(z_f)
            ev_list.append(ev_f)
            target_act_list.append(ta_f)
            supported_list.append(np.full((len(ta_f),), is_sup, dtype=np.int32))
            correct_content_list.append(ck_f)
            ep_id_list.append(np.full((len(ta_f),), bi * 2 + (1 - is_sup), dtype=np.int32))

    return {
        "z": np.concatenate(z_list, axis=0),
        "ev": np.concatenate(ev_list, axis=0),
        "target_action": np.concatenate(target_act_list, axis=0),
        "is_supported": np.concatenate(supported_list, axis=0),
        "content_correct": np.concatenate(correct_content_list, axis=0),
        "episode_group": np.concatenate(ep_id_list, axis=0),
    }


def evaluate_policy_commit_gate(
    gate_params: dict[str, Any],
    eval_ds: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute ALL FIVE mandatory `AGENTS.md §8` metrics vs BOTH `Always-Commit` & `Reliability-Only` baselines!"""
    z = jnp.asarray(eval_ds["z"])
    ev = jnp.asarray(eval_ds["ev"])
    _, probs = apply_policy_commit_gate(gate_params, z, ev)
    probs_np = np.asarray(probs, dtype=np.float32)  # [N, 5]
    pred_act = np.argmax(probs_np, axis=-1)
    tgt_act = eval_ds["target_action"]
    is_sup = eval_ds["is_supported"] == 1
    is_unsup = ~is_sup
    content_ok = eval_ds["content_correct"] > 0.5

    abstain_idx = COMMIT_ACTIONS.index("ABSTAIN_SILENCE")
    committed = pred_act != abstain_idx
    abstained = pred_act == abstain_idx

    action_acc = float(np.mean(pred_act == tgt_act))
    grounded_channel_acc = float(np.mean(pred_act[is_sup] == tgt_act[is_sup])) if np.any(is_sup) else 0.0
    # Actual content accuracy strictly on committed supported outputs + contradiction rate (wrong committed content!)
    sup_comm = is_sup & committed
    grounded_content_acc = float(np.mean(content_ok[sup_comm])) if np.any(sup_comm) else 0.0
    contradiction_rate = float(np.mean(committed & is_sup & ((pred_act != tgt_act) | (~content_ok))))
    unsupported_claim_rate = float(np.mean(committed[is_unsup])) if np.any(is_unsup) else 0.0
    abstention_rate = float(np.mean(abstained))
    conf = np.max(probs_np, axis=-1)
    ece = _expected_calibration_error(conf, (pred_act == tgt_act).astype(np.float32))

    # Baseline 1: Always-Commit (never abstains)
    ac_pred = np.argmax(probs_np[:, :abstain_idx], axis=-1)
    ac_grounded_channel_acc = float(np.mean(ac_pred[is_sup] == tgt_act[is_sup])) if np.any(is_sup) else 0.0
    ac_grounded_content_acc = float(np.mean(content_ok[is_sup])) if np.any(is_sup) else 0.0
    ac_unsupported_rate = 1.0

    # Baseline 2: Trivial Reliability-Only Threshold (`rel_ctx < 0.75 -> ABSTAIN`, else commit argmax 0..3)
    rel_ctx = eval_ds["ev"][:, 3]
    ro_abstain = rel_ctx < 0.75
    ro_pred = np.where(ro_abstain, abstain_idx, ac_pred)
    ro_comm = ro_pred != abstain_idx
    ro_action_acc = float(np.mean(ro_pred == tgt_act))
    ro_unsupported_rate = float(np.mean(ro_comm[is_unsup])) if np.any(is_unsup) else 0.0
    ro_abstention_rate = float(np.mean(ro_abstain))

    return {
        "n_eval_samples": int(len(tgt_act)),
        "policy_commit_gate": {
            "overall_decision_accuracy": round(action_acc, 4),
            "grounded_channel_accuracy": round(grounded_channel_acc, 4),
            "grounded_content_accuracy_on_committed": round(grounded_content_acc, 4),
            "unsupported_claim_rate": round(unsupported_claim_rate, 4),
            "contradiction_rate": round(contradiction_rate, 4),
            "abstention_noop_rate": round(abstention_rate, 4),
            "expected_calibration_error_ece": round(ece, 4),
            "action_distribution": {
                COMMIT_ACTIONS[i]: round(float(np.mean(pred_act == i)), 4) for i in range(len(COMMIT_ACTIONS))
            },
        },
        "reliability_only_threshold_baseline": {
            "overall_decision_accuracy": round(ro_action_acc, 4),
            "unsupported_claim_rate": round(ro_unsupported_rate, 4),
            "abstention_noop_rate": round(ro_abstention_rate, 4),
        },
        "always_commit_baseline": {
            "grounded_channel_accuracy": round(ac_grounded_channel_acc, 4),
            "grounded_content_accuracy": round(ac_grounded_content_acc, 4),
            "unsupported_claim_rate": round(ac_unsupported_rate, 4),
            "abstention_noop_rate": 0.0,
        },
        "unsupported_claim_reduction_vs_always_commit_pp": round((ac_unsupported_rate - unsupported_claim_rate) * 100.0, 2),
        "decision_acc_gain_over_reliability_only_pp": round((action_acc - ro_action_acc) * 100.0, 2),
        "non_collapsing_abstention": bool(0.10 <= abstention_rate <= 0.90),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v3 Wave B Frozen-Core Shortcut-Free PolicyCommitGate.")
    ap.add_argument("--ckpt", type=Path, required=True, help="Frozen Wave A checkpoint `.pkl`")
    ap.add_argument("--shard-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded"))
    ap.add_argument("--tag", type=str, default="", help="Optional unique receipt tag (e.g. profile_arm_s4201).")
    ap.add_argument("--use-wy-chunking", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=6101)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    with args.ckpt.open("rb") as f:
        ckpt = pickle.load(f)
    spec = ckpt["spec"]
    all_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), ckpt["params"])
    mm_dtype = jnp.float32 if str(jax.devices()[0].platform).lower() == "cpu" else jnp.bfloat16

    # STRICT TRAIN-ONLY SHARDS (zero dev leakage!)
    tc_tr = args.shard_dir / "text_code/text_code_train_v1.npz"
    if not tc_tr.is_file():
        tc_tr = args.shard_dir / "text_code/text_code_train_24m_balanced.npz"
    train_shards = [
        args.shard_dir / "audio_text/clotho_train.npz",
        tc_tr,
        args.shard_dir / "av/msrvtt_train.npz",
    ]
    eval_shards = [
        args.shard_dir / "audio_text/clotho_dev.npz",
        args.shard_dir / "text_code/wikitext103_validation.npz",
        args.shard_dir / "av/msrvtt_dev.npz",
    ]

    # Train jointly on BOTH blind-reliability (primary) AND degraded-sensor-outage so gate handles both!
    tr_blind = build_epistemic_commitment_dataset(
        all_params, spec, train_shards, regime="blind_reliability_unsupported", n_batches=12,
        matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking, seed=args.seed,
    )
    tr_outage = build_epistemic_commitment_dataset(
        all_params, spec, train_shards, regime="degraded_sensor_outage", n_batches=6,
        matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking, seed=args.seed + 77,
    )
    z_tr = jnp.asarray(np.concatenate([tr_blind["z"], tr_outage["z"]], axis=0))
    ev_tr = jnp.asarray(np.concatenate([tr_blind["ev"], tr_outage["ev"]], axis=0))
    y_tr = jnp.asarray(np.concatenate([tr_blind["target_action"], tr_outage["target_action"]], axis=0), dtype=jnp.int32)

    ev_blind = build_epistemic_commitment_dataset(
        all_params, spec, eval_shards, regime="blind_reliability_unsupported", n_batches=8,
        matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking, seed=args.seed + 999,
    )
    ev_outage = build_epistemic_commitment_dataset(
        all_params, spec, eval_shards, regime="degraded_sensor_outage", n_batches=8,
        matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking, seed=args.seed + 1999,
    )

    gate_p = init_policy_commit_gate_params(jax.random.PRNGKey(args.seed), spec.d_model, hidden_dim=256)
    gate_lr = 3e-3 / max(1.0, float(spec.width_mult))
    tx = optax.adamw(learning_rate=gate_lr, weight_decay=1e-4)
    opt_st = tx.init(gate_p)

    @jax.jit
    def _step(gp, ost, z_b, ev_b, y_b):
        def _loss(p):
            logits, _ = apply_policy_commit_gate(p, z_b, ev_b)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y_b).mean()

        lv, g = jax.value_and_grad(_loss)(gp)
        u, nost = tx.update(g, ost, gp)
        return optax.apply_updates(gp, u), nost, lv

    for s in range(args.steps):
        gate_p, opt_st, lv = _step(gate_p, opt_st, z_tr, ev_tr, y_tr)
        if s == 0 or (s + 1) % 100 == 0:
            print(f"[WAVE_B step={s+1:4d}/{args.steps}] gate_ce={float(lv):.4f}", flush=True)

    rep_blind = evaluate_policy_commit_gate(gate_p, ev_blind)
    rep_outage = evaluate_policy_commit_gate(gate_p, ev_outage)
    tag = args.tag or spec.name

    # Deterministic AGENTS.md §13.2 Wave B PolicyCommitGate Showcase Manifest
    _, probs_b = apply_policy_commit_gate(gate_p, jnp.asarray(ev_blind["z"]), jnp.asarray(ev_blind["ev"]))
    probs_b_np = np.asarray(probs_b, dtype=np.float32)
    abstain_idx = COMMIT_ACTIONS.index("ABSTAIN_SILENCE")
    ep_groups = np.unique(ev_blind["episode_group"])
    showcase_eps = []
    for g_id in ep_groups[:32]:
        m = ev_blind["episode_group"] == g_id
        p_act = np.argmax(probs_b_np[m], axis=-1)
        t_act = ev_blind["target_action"][m]
        is_s = int(ev_blind["is_supported"][m][0])
        acc_g = float(np.mean(p_act == t_act))
        abs_g = float(np.mean(p_act == abstain_idx))
        showcase_eps.append(
            {
                "episode_id": f"wave_b_blind_ep{int(g_id):04d}",
                "ground_truth_supported": bool(is_s == 1),
                "channel_provenance": {
                    "input_context": "observed_blind_reliability_1p0",
                    "commit_decision": "policy_commit_gate_routed",
                },
                "decision_accuracy": round(acc_g, 4),
                "abstention_rate": round(abs_g, 4),
                "dominant_action": COMMIT_ACTIONS[int(np.bincount(p_act).argmax())],
            }
        )
    showcase_p = write_deterministic_inference_showcase(
        args.out_dir / "inference_showcase",
        split="dev",
        model_name=f"wave_b_{tag}",
        episodes=showcase_eps,
        metric_key="decision_accuracy",
        higher_is_better=True,
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "checkpoint": str(args.ckpt),
        "core_weights_strictly_frozen": True,
        "train_shards_strictly_disjoint_from_dev": True,
        "primary_blind_reliability_regime": rep_blind,
        "secondary_degraded_sensor_outage_regime": rep_outage,
        "wave_b_showcase_manifest": str(showcase_p),
        "wave_b_blind_beats_reliability_only": bool(
            rep_blind["decision_acc_gain_over_reliability_only_pp"] > 5.0
            and rep_blind["non_collapsing_abstention"]
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gate_ckpt_p = args.out_dir / f"wave_b_{tag}_ckpt.pkl"
    with gate_ckpt_p.open("wb") as f:
        pickle.dump(
            {
                "tag": tag,
                "commit_actions": list(COMMIT_ACTIONS),
                "gate_params": jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float32), gate_params),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    out_p = args.out_dir / f"wave_b_policy_gate_{tag}_receipt.json"
    out_p.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
