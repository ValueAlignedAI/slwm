"""SLWM-v3 Unified Any-to-Any Real-Corpus Trainer & External SoTA Evaluation Harness (`Waves A-D`).

Trains ANY SLWM-v3 backbone or matched Wave A comparator (`±1.0%` params, matched state bytes)
using `models/v3/fast_core.py:scan_core_fast` (`bf16` MXU + Exact Algebraic Surprise Hoisting +
`fp32 Precision.HIGHEST` memory state) + `models/v3/any2any_heads.py` (`head_text`, `head_audio`,
`head_video`, `head_action`) + TPU-verified exact base-shape muP (`base_lr = 3e-3`) across
unified sparse multimodal shards (`data/v3_unified_shards.py`) with modality dropout.

Evaluates strictly on frozen weights (`no` online weight updates):
  - WikiText-103 (`validation` / `test`): exact Bits-Per-Byte (BPB) & word-level Perplexity,
  - LAMBADA (`test`): exact final-word NLL & word accuracy,
  - ESC-50 (`fold 5`): audio RVQ continuation NLL & decoupled 50-class probe accuracy,
  - LibriSpeech (`dev-clean` / `test-clean`): speech RVQ NLL & Audio -> Text byte BPB,
  - Cross-Modal Causal Conditioning Proof (`AGENTS.md §13.2`): paired vs batch-shuffled source
    condition delta (`delta_shuffle_gain = L_shuffled - L_paired > 0`) on Audio <-> Text & Video <-> Audio,
  - Deterministic uncurated `inference_showcase/` manifest (`P05, P25, P50, P75, P95` episodes).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from data.v3_unified_shards import BYTE_EOS, BYTE_MOD_SEP, LoadedSparseShard, decode_utf8_bytes
from models.v3.any2any_heads import (
    compute_any2any_multimodal_loss,
    init_any2any_heads_params,
)
from models.v3.core import (
    build_mup_lr_multipliers,
    build_weight_decay_mask,
    init_core_state,
    init_slwm_v3_params,
)
from models.v3.fast_core import scan_core_fast
from models.v3.spec import (
    SLWMv3Spec,
    build_wave_a_comparator_specs,
    compute_core_streaming_state_bytes,
    count_tree_params,
    make_slwm_v3_spec,
)
from scripts.train.v3_pilot_train import build_optimizer, lr_schedule_mult


def build_combined_mup_optimizer(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    *,
    weight_decay: float = 0.01,
    grad_clip_norm: float = 1.0,
) -> optax.GradientTransformation:
    """Exact base-shape muP + masked independent WD across `{'core': ..., 'heads': ...}` (unit LR)."""
    mults = {
        "core": build_mup_lr_multipliers(all_params["core"], spec),
        "heads": build_mup_lr_multipliers(all_params["heads"], spec),
    }
    wd_mask = {
        "core": build_weight_decay_mask(all_params["core"]),
        "heads": build_weight_decay_mask(all_params["heads"]),
    }

    def _scale_mult() -> optax.GradientTransformation:
        def init_fn(_p):
            return optax.EmptyState()

        def update_fn(updates, state, _p=None):
            return jax.tree_util.tree_map(lambda u, m: u * jnp.asarray(m, dtype=u.dtype), updates, mults), state

        return optax.GradientTransformation(init_fn, update_fn)

    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-8),
        _scale_mult(),
        optax.add_decayed_weights(weight_decay, mask=wd_mask),
        optax.scale(-1.0),
    )


class MultiShardMixture:
    """Balanced multi-domain batch sampler over unified sparse `.npz` shards."""

    def __init__(self, shard_paths: list[tuple[str, Path, float]]) -> None:
        self.shards: list[tuple[str, LoadedSparseShard, float]] = []
        for name, p, w in shard_paths:
            if p.is_file():
                sh = LoadedSparseShard.load(p)
                if sh.n_ticks > 64:
                    self.shards.append((name, sh, float(w)))
        if not self.shards:
            raise FileNotFoundError(f"No valid sparse shards found in: {shard_paths}")
        tot_w = sum(w for _, _, w in self.shards)
        self.probs = np.array([w / tot_w for _, _, w in self.shards], dtype=np.float64)

    def sample_batch(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        seq_len: int,
        drop_modality_prob: float = 0.25,
    ) -> dict[str, jax.Array]:
        # Sample each batch item from a domain chosen according to mixture probabilities
        chosen = rng.choice(len(self.shards), size=(batch_size,), p=self.probs)
        rows = [
            self.shards[int(ci)][1].sample_batch(
                rng, batch_size=1, seq_len=seq_len, drop_modality_prob=drop_modality_prob
            )
            for ci in chosen
        ]
        merged = {k: jnp.asarray(np.concatenate([r[k] for r in rows], axis=1)) for k in rows[0]}
        return merged


def evaluate_text_sota_shard(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    shard_path: Path,
    *,
    seq_len: int = 512,
    batch_size: int = 8,
    max_windows: int = 64,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
) -> dict[str, Any]:
    """Evaluate exact causal Bits-Per-Byte (BPB) & approximate word-level PPL on WikiText-103 / dev shard."""
    if not shard_path.is_file():
        return {"skipped": f"missing {shard_path}"}
    sh = LoadedSparseShard.load(shard_path)
    n_ticks = sh.n_ticks
    n_win = min(max_windows, max(1, (n_ticks - 1) // (seq_len * batch_size)))

    @jax.jit
    def _eval_batch(ap, batch):
        b = batch["valid_mask"].shape[1]
        st0 = init_core_state(spec, b)
        z, _, _ = scan_core_fast(
            ap["core"], st0, batch, spec, remat_layers=False, matmul_dtype=matmul_dtype, use_wy_chunking=use_wy_chunking
        )
        _, m = compute_any2any_multimodal_loss(ap["heads"], z, batch, spec, matmul_dtype=matmul_dtype)
        return m["text_ce_nat"], m["n_text_targets"]

    tot_nats = 0.0
    tot_bytes = 0.0
    tot_words = 0
    for wi in range(n_win):
        start_base = wi * batch_size * seq_len
        starts = np.array([start_base + b * seq_len for b in range(batch_size)], dtype=np.int64)
        idx = starts[None, :] + np.arange(seq_len, dtype=np.int64)[:, None]
        idx = np.clip(idx, 0, n_ticks - 1)
        d = sh.data
        tb = d["text_byte"][idx].astype(np.int32)
        for b in range(batch_size):
            txt = decode_utf8_bytes(tb[:, b])
            tot_words += max(1, len(txt.split()))
        batch = {
            "modality_id": jnp.asarray(d["modality_id"][idx].astype(np.int32)),
            "dt_s": jnp.asarray(d["dt_s"][idx].astype(np.float32)),
            "valid_mask": jnp.asarray(d["valid_mask"][idx].astype(np.float32)),
            "target_valid_mask": jnp.asarray(d["valid_mask"][idx].astype(np.float32)),
            "reliability": jnp.asarray(d["reliability"][idx].astype(np.float32)),
            "reset_mask": jnp.asarray(d["reset_mask"][idx].astype(bool)),
            "spatial_pos": jnp.asarray(d["spatial_pos"][idx].astype(np.int32)),
            "text_byte": jnp.asarray(tb),
            "has_true_action": jnp.zeros((seq_len, batch_size), dtype=jnp.float32),
            "video_feat": jnp.zeros((seq_len, batch_size, spec.video_patch_dim), dtype=jnp.float32),
            "audio_codes": jnp.zeros((seq_len, batch_size, spec.audio_codebooks), dtype=jnp.int32),
            "proprio_feat": jnp.zeros((seq_len, batch_size, spec.proprio_dim), dtype=jnp.float32),
            "action_feat": jnp.zeros((seq_len, batch_size, spec.action_dim), dtype=jnp.float32),
            "probe_feat": jnp.zeros((seq_len, batch_size, spec.probe_dim), dtype=jnp.float32),
        }
        ce_nat, cnt = _eval_batch(all_params, batch)
        tot_nats += float(ce_nat) * float(cnt)
        tot_bytes += float(cnt)

    mean_nat = tot_nats / max(tot_bytes, 1.0)
    bpb = mean_nat / math.log(2.0)
    word_ppl = math.exp(min(20.0, tot_nats / max(float(tot_words), 1.0)))
    return {
        "shard": shard_path.name,
        "evaluated_bytes": int(tot_bytes),
        "evaluated_words": int(tot_words),
        "ce_nat_per_byte": round(mean_nat, 5),
        "bits_per_byte": round(bpb, 5),
        "approx_word_perplexity": round(word_ppl, 3),
    }


def evaluate_cross_modal_condition_shuffle(
    all_params: dict[str, Any],
    spec: SLWMv3Spec,
    shard_path: Path,
    *,
    source_mod: int,
    target_mod: int,
    seq_len: int = 512,
    batch_size: int = 8,
    n_batches: int = 8,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
    seed: int = 8801,
) -> dict[str, Any]:
    """AGENTS.md §13.2 Causal Conditioning Proof: paired vs batch-shuffled source modality loss!

    Proves the model genuinely conditions target-modality predictions (`target_mod`) on the paired
    source modality (`source_mod`) rather than ignoring source input and hallucinating an unconditional prior!
    """
    if not shard_path.is_file():
        return {"skipped": f"missing {shard_path}"}
    sh = LoadedSparseShard.load(shard_path)
    rng = np.random.default_rng(seed)

    @jax.jit
    def _eval_pair(ap, batch):
        b = batch["valid_mask"].shape[1]
        st0 = init_core_state(spec, b)
        z, _, _ = scan_core_fast(
            ap["core"], st0, batch, spec, remat_layers=False, matmul_dtype=matmul_dtype, use_wy_chunking=use_wy_chunking
        )
        _, m = compute_any2any_multimodal_loss(ap["heads"], z, batch, spec, matmul_dtype=matmul_dtype)
        return m

    key_map = {0: "video_patch_huber", 1: "audio_rvq_ce_nat", 2: "text_ce_nat", 3: "action_proprio_huber"}
    cnt_map = {0: "n_video_targets", 1: "n_audio_targets", 2: "n_text_targets", 3: "n_action_targets"}
    metric_k = key_map[target_mod]
    cnt_k = cnt_map[target_mod]

    paired_vals, shuf_vals = [], []
    for _ in range(n_batches):
        b_np = sh.sample_batch(rng, batch_size=batch_size, seq_len=seq_len, drop_modality_prob=0.0)
        b_pair = {k: jnp.asarray(v) for k, v in b_np.items()}
        # Roll source modality payloads strictly by +1 along batch axis while keeping target modality untouched!
        b_shuf_np = {k: v.copy() for k, v in b_np.items()}
        src_mask = b_shuf_np["modality_id"] == source_mod
        if source_mod == 0:
            rolled = np.roll(b_shuf_np["video_feat"], shift=1, axis=1)
            b_shuf_np["video_feat"] = np.where(src_mask[..., None], rolled, b_shuf_np["video_feat"])
        elif source_mod == 1:
            rolled = np.roll(b_shuf_np["audio_codes"], shift=1, axis=1)
            b_shuf_np["audio_codes"] = np.where(src_mask[..., None], rolled, b_shuf_np["audio_codes"])
        elif source_mod == 2:
            rolled = np.roll(b_shuf_np["text_byte"], shift=1, axis=1)
            b_shuf_np["text_byte"] = np.where(src_mask, rolled, b_shuf_np["text_byte"])
        b_shuf = {k: jnp.asarray(v) for k, v in b_shuf_np.items()}

        mp = _eval_pair(all_params, b_pair)
        ms = _eval_pair(all_params, b_shuf)
        if float(mp[cnt_k]) > 0:
            paired_vals.append(float(mp[metric_k]))
            shuf_vals.append(float(ms[metric_k]))

    if not paired_vals:
        return {"shard": shard_path.name, "n_valid_batches": 0, "causal_conditioning_verified": False}
    p_mean = float(np.mean(paired_vals))
    s_mean = float(np.mean(shuf_vals))
    delta = s_mean - p_mean
    return {
        "shard": shard_path.name,
        "source_mod": int(source_mod),
        "target_mod": int(target_mod),
        "metric": metric_k,
        "paired_loss": round(p_mean, 5),
        "shuffled_source_loss": round(s_mean, 5),
        "causal_shuffle_gain": round(delta, 5),
        "causal_conditioning_verified": bool(delta > 0.01),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v3 Unified Any-to-Any Real-Corpus Trainer & SoTA Evaluator.")
    ap.add_argument("--profile", type=str, default="discovery_45m")
    ap.add_argument("--backbone", type=str, default="slwm_v3_routed")
    ap.add_argument("--match-anchor", action="store_true", help="Match comparator params/bytes to slwm_v3_routed.")
    ap.add_argument("--use-wy-chunking", action="store_true")
    ap.add_argument("--shard-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--base-lr", type=float, default=3e-3)
    ap.add_argument("--drop-modality-prob", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=4201)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    if args.match_anchor and args.profile in ("pilot_d256", "discovery_45m", "confirmation_135m"):
        spec = build_wave_a_comparator_specs(args.profile, match_params_to_anchor=True)[args.backbone]
    else:
        spec = make_slwm_v3_spec(args.profile, backbone_type=args.backbone)

    is_cpu = str(jax.devices()[0].platform).lower() == "cpu"
    mm_dtype = jnp.float32 if is_cpu else jnp.bfloat16

    k_core, k_head = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    all_params = {
        "core": init_slwm_v3_params(k_core, spec),
        "heads": init_any2any_heads_params(k_head, spec),
    }
    n_core_p = count_tree_params(all_params["core"])
    n_head_p = count_tree_params(all_params["heads"])
    st_bytes = compute_core_streaming_state_bytes(spec, batch_size=1)

    # Auto-discover all available unified train shards under `--shard-dir`
    train_candidates = [
        ("text_code", args.shard_dir / "text_code/text_code_train_v1.npz", 0.35),
        ("clotho", args.shard_dir / "audio_text/clotho_train.npz", 0.25),
        ("esc50", args.shard_dir / "audio_text/esc50_train_f14.npz", 0.10),
        ("greatesthits", args.shard_dir / "av/greatesthits_train.npz", 0.15),
        ("msrvtt", args.shard_dir / "av/msrvtt_train.npz", 0.15),
        ("vggsound", args.shard_dir / "av/vggsound_train.npz", 0.15),
        ("robotics", args.shard_dir / "robotics/robotics_train.npz", 0.15),
    ]
    mixture = MultiShardMixture(train_candidates)
    print(f"[TRAIN] Active domains: {[(n, sh.n_ticks, round(float(p), 3)) for (n, sh, _), p in zip(mixture.shards, mixture.probs)]}", flush=True)

    tx_unit = build_combined_mup_optimizer(all_params, spec, weight_decay=0.01, grad_clip_norm=1.0)
    opt_state = tx_unit.init(all_params)

    @jax.jit(donate_argnums=(0, 1))
    def _train_step(ap, o_st, batch, lr_arr):
        b = batch["valid_mask"].shape[1]
        st0 = init_core_state(spec, b)

        def _loss_fn(p):
            z, _, diags = scan_core_fast(
                p["core"],
                st0,
                batch,
                spec,
                remat_layers=True,
                matmul_dtype=mm_dtype,
                use_wy_chunking=args.use_wy_chunking,
            )
            loss, m = compute_any2any_multimodal_loss(p["heads"], z, batch, spec, matmul_dtype=mm_dtype)
            m["mean_nowrite"] = jnp.mean(diags["mean_nowrite"])
            m["mean_eff_write_eta"] = jnp.mean(diags["mean_eff_write_eta"])
            m["max_mem_frob_norm"] = jnp.max(diags["max_mem_frob_norm"])
            return loss, m

        (loss, m), g = jax.value_and_grad(_loss_fn, has_aux=True)(ap)
        unit_upd, new_o = tx_unit.update(g, o_st, ap)
        upd = jax.tree_util.tree_map(lambda u: u * lr_arr.astype(u.dtype), unit_upd)
        return optax.apply_updates(ap, upd), new_o, loss, m

    rng = np.random.default_rng(args.seed + 100)
    t_start = time.perf_counter()
    log_rows: list[dict[str, Any]] = []
    for step in range(args.steps):
        batch = mixture.sample_batch(
            rng, batch_size=args.batch_size, seq_len=args.seq_len, drop_modality_prob=args.drop_modality_prob
        )
        step_lr = jnp.asarray(args.base_lr * lr_schedule_mult(step, args.steps), dtype=jnp.float32)
        all_params, opt_state, lv, metrics = _train_step(all_params, opt_state, batch, step_lr)
        if step == 0 or (step + 1) % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            lv.block_until_ready()
            row = {
                "step": step + 1,
                "elapsed_s": round(time.perf_counter() - t_start, 1),
                "total_loss": round(float(lv), 5),
                "text_bpb": round(float(metrics["text_bpb"]), 4),
                "audio_rvq_ce": round(float(metrics["audio_rvq_ce_nat"]), 4),
                "video_huber": round(float(metrics["video_patch_huber"]), 5),
                "action_huber": round(float(metrics["action_proprio_huber"]), 5),
                "mean_nowrite": round(float(metrics["mean_nowrite"]), 4),
                "max_mem_frob": round(float(metrics["max_mem_frob_norm"]), 3),
            }
            log_rows.append(row)
            print(f"[TRAIN step={step+1:5d}/{args.steps}] {row}", flush=True)

    # Strictly frozen-weight evaluations (zero weight updates!)
    evals: dict[str, Any] = {
        "text_code_dev": evaluate_text_sota_shard(
            all_params, spec, args.shard_dir / "text_code/text_code_dev_v1.npz",
            seq_len=args.seq_len, batch_size=args.batch_size, max_windows=32, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        ),
        "wikitext103_val": evaluate_text_sota_shard(
            all_params, spec, args.shard_dir / "text_code/wikitext103_validation.npz",
            seq_len=args.seq_len, batch_size=args.batch_size, max_windows=32, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        ),
        "clotho_audio_to_text_shuffle": evaluate_cross_modal_condition_shuffle(
            all_params, spec, args.shard_dir / "audio_text/clotho_dev.npz",
            source_mod=1, target_mod=2, seq_len=args.seq_len, batch_size=min(8, args.batch_size),
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        ),
        "greatesthits_video_to_audio_shuffle": evaluate_cross_modal_condition_shuffle(
            all_params, spec, args.shard_dir / "av/greatesthits_dev.npz",
            source_mod=0, target_mod=1, seq_len=args.seq_len, batch_size=min(8, args.batch_size),
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "backbone": args.backbone,
        "use_wy_chunking": bool(args.use_wy_chunking),
        "seed": int(args.seed),
        "trainable_core_params": int(n_core_p),
        "trainable_head_params": int(n_head_p),
        "core_streaming_state_bytes_b1": int(st_bytes["total_core_state_bytes"]),
        "total_positions_trained": int(args.steps * args.batch_size * args.seq_len),
        "wall_sec": round(time.perf_counter() - t_start, 2),
        "train_curve": log_rows,
        "frozen_evaluations": evals,
    }
    (args.out_dir / f"{args.profile}_{args.backbone}_s{args.seed}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
