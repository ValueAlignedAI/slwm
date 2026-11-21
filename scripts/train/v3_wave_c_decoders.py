"""SLWM-v3 Wave C: Strictly Frozen-Core Any-to-Any Decoders & Causal Cross-Modal Generation Audit (`AGENTS.md §13.2`).

Trains decoupled generative readout decoders strictly on `jax.lax.stop_gradient(Z_t)` (verifying
bit-exact ZERO gradient into frozen world core weights, passing `core_params` as a traced JIT
argument so `700M/1B` cores NEVER blow up XLA HLO constants, and using exact `1 / m_d` muP readout LR!):
  1. `Video <-> Audio` (VGGSound synchronized 6s AV clips),
  2. `Audio -> Text` & `Video -> Text` (Clotho & MSR-VTT),
  3. `Action + Proprio + Video -> Next Proprio & Visual Rollout` (LeRobot ALOHA-Sim / PushT / LIBERO).

Mandatory `AGENTS.md §13.2` Zero-False-Positive Causal Conditioning & Showcase Checks:
  - Episode-aligned zero-padding-free condition-shuffling (`Video->Audio`, `Audio->Video`, `Audio->Text`, `Video->Text`),
  - Target-held-fixed Action-Intervention proof (`Action -> Next Proprio & Video`),
  - Deterministic uncurated `inference_showcase/` quantile manifests (`P05/P25/P50/P75/P95`).
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
from evals.v3_sota_benchmarks import (
    evaluate_cross_modal_condition_shuffle,
    evaluate_robotics_action_intervention,
    generate_cross_modal_text_showcase,
)
from models.v3.any2any_heads import compute_any2any_multimodal_loss, init_any2any_heads_params
from models.v3.core import init_core_state
from models.v3.fast_core import scan_core_fast
from models.v3.spec import count_tree_params


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v3 Wave C Frozen-Core Any-to-Any Decoder Trainer & Causal Audit.")
    ap.add_argument("--ckpt", type=Path, required=True, help="Frozen Wave A checkpoint `.pkl`")
    ap.add_argument("--shard-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded"))
    ap.add_argument("--tag", type=str, default="", help="Optional unique receipt tag (e.g. profile_arm_s4201).")
    ap.add_argument("--use-wy-chunking", action="store_true")
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

    dec_params = init_any2any_heads_params(jax.random.PRNGKey(args.seed), spec)
    n_dec_params = count_tree_params(dec_params)

    shard_paths = [
        ("clotho", args.shard_dir / "audio_text/clotho_train.npz"),
        ("msrvtt", args.shard_dir / "av/msrvtt_train.npz"),
        ("vggsound", args.shard_dir / "av/vggsound_train.npz"),
        ("robotics", args.shard_dir / "robotics/robotics_train.npz"),
    ]
    shards = [(n, LoadedSparseShard.load(p)) for n, p in shard_paths if p.is_file()]
    if not shards:
        raise FileNotFoundError(f"No cross-modal train shards found in {args.shard_dir}")

    # Exact muP readout LR scaling (`3e-3 / m_d`) so Wave C transfers cleanly across 42M -> 138M -> 703M -> 1.07B!
    readout_lr = 3e-3 / max(1.0, float(spec.width_mult))
    tx = optax.adamw(learning_rate=readout_lr, weight_decay=1e-4)
    opt_st = tx.init(dec_params)

    # Pass `cp` as explicit traced argument (NEVER capture 700M/1B weights as XLA closure constant!)
    def _dec_loss_with_core(cp, dp, batch):
        b = batch["valid_mask"].shape[1]
        z, _, _ = scan_core_fast(
            cp,
            init_core_state(spec, b),
            batch,
            spec,
            remat_layers=False,
            matmul_dtype=mm_dtype,
            assume_all_valid=True,
            use_wy_chunking=args.use_wy_chunking,
        )
        z_frozen = jax.lax.stop_gradient(z)
        return compute_any2any_multimodal_loss(dp, z_frozen, batch, spec, matmul_dtype=mm_dtype)

    rng = np.random.default_rng(args.seed + 11)
    probe_b = {
        k: jnp.asarray(v)
        for k, v in shards[0][1].sample_batch(rng, batch_size=2, seq_len=min(64, args.seq_len), drop_modality_prob=0.0).items()
    }
    (_, _), (g_core_chk, _) = jax.value_and_grad(_dec_loss_with_core, argnums=(0, 1), has_aux=True)(
        core_params, dec_params, probe_b
    )
    max_core_grad = max(float(jnp.max(jnp.abs(g))) for g in jax.tree_util.tree_leaves(g_core_chk))
    assert max_core_grad == 0.0, f"Wave C core gradient isolation VIOLATED: {max_core_grad}"

    @jax.jit(donate_argnums=(1, 2))
    def _step(cp, dp, ost, batch):
        (loss, m), g = jax.value_and_grad(lambda p: _dec_loss_with_core(cp, p, batch), has_aux=True)(dp)
        u, nost = tx.update(g, ost, dp)
        return optax.apply_updates(dp, u), nost, loss, m

    curve = []
    for s in range(args.steps):
        _, sh = shards[s % len(shards)]
        b_np = sh.sample_batch(rng, batch_size=args.batch_size, seq_len=args.seq_len, drop_modality_prob=0.25)
        batch = {k: jnp.asarray(v) for k, v in b_np.items()}
        dec_params, opt_st, lv, m = _step(core_params, dec_params, opt_st, batch)
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
    eval_cm_seq = max(args.seq_len, 768)
    eval_bsz = min(8, args.batch_size)
    causal_checks = {
        "video_to_audio_vggsound_dev": evaluate_cross_modal_condition_shuffle(
            frozen_pair, spec, args.shard_dir / "av/vggsound_dev.npz",
            source_mod=0, target_mod=1, seq_len=eval_cm_seq, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        ),
        "audio_to_video_vggsound_dev": evaluate_cross_modal_condition_shuffle(
            frozen_pair, spec, args.shard_dir / "av/vggsound_dev.npz",
            source_mod=1, target_mod=0, seq_len=eval_cm_seq, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        ),
        "audio_to_text_clotho_dev": evaluate_cross_modal_condition_shuffle(
            frozen_pair, spec, args.shard_dir / "audio_text/clotho_dev.npz",
            source_mod=1, target_mod=2, seq_len=eval_cm_seq, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        ),
        "video_to_text_msrvtt_dev": evaluate_cross_modal_condition_shuffle(
            frozen_pair, spec, args.shard_dir / "av/msrvtt_dev.npz",
            source_mod=0, target_mod=2, seq_len=eval_cm_seq, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        ),
        "action_intervention_robotics_dev": evaluate_robotics_action_intervention(
            frozen_pair, spec, args.shard_dir / "robotics/robotics_dev.npz",
            seq_len=args.seq_len, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        ),
    }

    tag = args.tag or spec.name
    showcase_dir = args.out_dir / "inference_showcase"
    showcases = {
        "clotho_dev": generate_cross_modal_text_showcase(
            frozen_pair, spec, args.shard_dir / "audio_text/clotho_dev.npz",
            out_dir=showcase_dir, split="dev", model_name=f"wave_c_{tag}",
            seq_len=args.seq_len, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        ),
        "msrvtt_dev": generate_cross_modal_text_showcase(
            frozen_pair, spec, args.shard_dir / "av/msrvtt_dev.npz",
            out_dir=showcase_dir, split="dev", model_name=f"wave_c_{tag}",
            seq_len=args.seq_len, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dec_ckpt_p = args.out_dir / f"wave_c_{tag}_ckpt.pkl"
    with dec_ckpt_p.open("wb") as f:
        pickle.dump(
            {
                "tag": tag,
                "spec": spec,
                "dec_params": jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float32), dec_params),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "spec_name": spec.name,
        "checkpoint": str(args.ckpt),
        "wave_c_decoder_params": int(n_dec_params),
        "readout_mup_lr": round(readout_lr, 6),
        "max_core_grad_verified": float(max_core_grad),
        "core_weights_strictly_frozen": True,
        "train_curve": curve,
        "causal_conditioning_proofs": causal_checks,
        "wave_c_showcases": showcases,
    }
    out_p = args.out_dir / f"wave_c_decoders_{tag}_receipt.json"
    out_p.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
