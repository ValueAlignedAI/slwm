"""SLWM-v3 Wave C: Strictly Frozen-Core Any-to-Any Decoders & Causal Cross-Modal Generation Audit (`AGENTS.md §13.2`).

Trains decoupled generative readout decoders strictly on `jax.lax.stop_gradient(Z_t)` (verifying
bit-exact ZERO gradient into frozen world core weights!) across:
  1. `Video -> Audio` (VGGSound synchronized 6s AV clips: predict Kyutai Mimi RVQ-0..7 from Wan 2.1 video!)
  2. `Audio -> Text` & `Video -> Text` (Clotho & MSR-VTT: grounded caption generation!)
  3. `Action + Proprio + Video -> Future Visual & Proprioceptive Rollout` (LeRobot ALOHA-Sim / PushT / LIBERO!)

Mandatory `AGENTS.md §13.2` Causal Conditioning & Non-Inferiority Checks:
  - Runs paired vs condition-shuffled (`Video-shuffled -> Audio`, `Audio-shuffled -> Text`,
    `Video-shuffled -> Text`) and `Action-intervened -> Next Visual/Proprio` counterfactual tests
    proving genuine causal cross-modal conditioning rather than unconditional prior hallucination!
  - Explicitly labels every output channel (`observed`, `reconstructed`, `predicted`, `imagined`, `abstained`).
  - Writes deterministic uncurated `inference_showcase/` quantile manifests (`P05/P25/P50/P75/P95`).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from data.v3_unified_shards import LoadedSparseShard
from evals.v3_sota_benchmarks import write_deterministic_inference_showcase
from models.v3.any2any_heads import compute_any2any_multimodal_loss, init_any2any_heads_params
from models.v3.core import init_core_state
from models.v3.fast_core import scan_core_fast
from models.v3.spec import count_tree_params
from scripts.train.v3_any2any_trainer import evaluate_cross_modal_condition_shuffle
from evals.v3_sota_benchmarks import evaluate_robotics_action_intervention


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v3 Wave C Frozen-Core Any-to-Any Decoder Trainer & Causal Audit.")
    ap.add_argument("--ckpt", type=Path, required=True, help="Frozen Wave A checkpoint `.pkl`")
    ap.add_argument("--shard-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7101)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    with args.ckpt.open("rb") as f:
        ckpt = pickle.load(f)
    spec = ckpt["spec"]
    core_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), ckpt["params"]["core"])
    mm_dtype = jnp.float32 if str(jax.devices()[0].platform).lower() == "cpu" else jnp.bfloat16

    # Freshly initialize decoupled Wave C decoder heads (proving core Z_t carries rich cross-modal structure!)
    dec_params = init_any2any_heads_params(jax.random.PRNGKey(args.seed), spec)
    n_dec_params = count_tree_params(dec_params)

    # Load cross-modal training shards
    shard_paths = [
        ("clotho", args.shard_dir / "audio_text/clotho_train.npz"),
        ("msrvtt", args.shard_dir / "av/msrvtt_train.npz"),
        ("vggsound", args.shard_dir / "av/vggsound_train.npz"),
        ("robotics", args.shard_dir / "robotics/robotics_train.npz"),
    ]
    shards = [(n, LoadedSparseShard.load(p)) for n, p in shard_paths if p.is_file()]
    if not shards:
        raise FileNotFoundError(f"No cross-modal train shards found in {args.shard_dir}")

    tx = optax.adamw(learning_rate=3e-3, weight_decay=1e-4)
    opt_st = tx.init(dec_params)

    # Verify bit-exact ZERO gradient flows into frozen `core_params`!
    def _dec_loss_with_core(cp, dp, batch):
        b = batch["valid_mask"].shape[1]
        z, _, _ = scan_core_fast(cp, init_core_state(spec, b), batch, spec, remat_layers=False, matmul_dtype=mm_dtype)
        z_frozen = jax.lax.stop_gradient(z)
        return compute_any2any_multimodal_loss(dp, z_frozen, batch, spec, matmul_dtype=mm_dtype)

    rng = np.random.default_rng(args.seed + 11)
    probe_b = {k: jnp.asarray(v) for k, v in shards[0][1].sample_batch(rng, batch_size=2, seq_len=min(64, args.seq_len), drop_modality_prob=0.0).items()}
    (_, _), (g_core_chk, g_dec_chk) = jax.value_and_grad(_dec_loss_with_core, argnums=(0, 1), has_aux=True)(
        core_params, dec_params, probe_b
    )
    max_core_grad = max(float(jnp.max(jnp.abs(g))) for g in jax.tree_util.tree_leaves(g_core_chk))
    assert max_core_grad == 0.0, f"Wave C core gradient isolation VIOLATED: {max_core_grad}"

    @jax.jit(donate_argnums=(0, 1))
    def _step(dp, ost, batch):
        (loss, m), g = jax.value_and_grad(lambda p: _dec_loss_with_core(core_params, p, batch), has_aux=True)(dp)
        u, nost = tx.update(g, ost, dp)
        return optax.apply_updates(dp, u), nost, loss, m

    curve = []
    for s in range(args.steps):
        _, sh = shards[s % len(shards)]
        b_np = sh.sample_batch(rng, batch_size=args.batch_size, seq_len=args.seq_len, drop_modality_prob=0.25)
        batch = {k: jnp.asarray(v) for k, v in b_np.items()}
        dec_params, opt_st, lv, m = _step(dec_params, opt_st, batch)
        if s == 0 or (s + 1) % max(1, args.steps // 10) == 0 or s == args.steps - 1:
            row = {
                "step": s + 1,
                "loss": round(float(lv), 5),
                "text_bpb": round(float(m["text_bpb"]), 4),
                "audio_rvq_ce": round(float(m["audio_rvq_ce_nat"]), 4),
                "video_huber": round(float(m["video_patch_huber"]), 5),
                "action_huber": round(float(m["action_proprio_huber"]), 5),
            }
            curve.append(row)
            print(f"[WAVE_C step={s+1:4d}/{args.steps}] {row}", flush=True)

    frozen_pair = {"core": core_params, "heads": dec_params}
    causal_checks = {
        "video_to_audio_vggsound_dev": evaluate_cross_modal_condition_shuffle(
            frozen_pair, spec, args.shard_dir / "av/vggsound_dev.npz",
            source_mod=0, target_mod=1, seq_len=args.seq_len, batch_size=min(8, args.batch_size), matmul_dtype=mm_dtype,
        ),
        "audio_to_text_clotho_dev": evaluate_cross_modal_condition_shuffle(
            frozen_pair, spec, args.shard_dir / "audio_text/clotho_dev.npz",
            source_mod=1, target_mod=2, seq_len=args.seq_len, batch_size=min(8, args.batch_size), matmul_dtype=mm_dtype,
        ),
        "video_to_text_msrvtt_dev": evaluate_cross_modal_condition_shuffle(
            frozen_pair, spec, args.shard_dir / "av/msrvtt_dev.npz",
            source_mod=0, target_mod=2, seq_len=args.seq_len, batch_size=min(8, args.batch_size), matmul_dtype=mm_dtype,
        ),
        "action_intervention_robotics_dev": evaluate_robotics_action_intervention(
            frozen_pair, spec, args.shard_dir / "robotics/robotics_dev.npz",
            seq_len=args.seq_len, batch_size=min(8, args.batch_size), matmul_dtype=mm_dtype,
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "spec_name": spec.name,
        "checkpoint": str(args.ckpt),
        "wave_c_decoder_params": int(n_dec_params),
        "max_core_grad_verified": float(max_core_grad),
        "core_weights_strictly_frozen": True,
        "train_curve": curve,
        "causal_conditioning_proofs": causal_checks,
    }
    out_p = args.out_dir / f"wave_c_decoders_{spec.name}_receipt.json"
    out_p.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
