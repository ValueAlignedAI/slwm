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
import gc
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
from evals.v3_external_sota_suite import (
    evaluate_code_execution_pass_at_1,
    evaluate_esc50_classification,
    evaluate_librispeech_wer,
)
from evals.v3_sota_benchmarks import (
    evaluate_code_solution_bpb,
    evaluate_cross_modal_condition_shuffle,
    evaluate_lambada_openai,
    evaluate_real_speech_overwrite_frozen_core,
    evaluate_robotics_action_intervention,
    generate_cross_modal_text_showcase,
)
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
        anchor_transition_prob: float = 0.50,
    ) -> dict[str, jax.Array]:
        # Sample each batch item from a domain chosen according to mixture probabilities
        chosen = rng.choice(len(self.shards), size=(batch_size,), p=self.probs)
        rows = [
            self.shards[int(ci)][1].sample_batch(
                rng,
                batch_size=1,
                seq_len=seq_len,
                drop_modality_prob=drop_modality_prob,
                anchor_transition_prob=anchor_transition_prob,
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
    max_windows: int = 256,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
) -> dict[str, Any]:
    """Evaluate exact causal Bits-Per-Byte (BPB) & approximate word-level PPL without tail duplication."""
    if not shard_path.is_file():
        return {"skipped": f"missing {shard_path}"}
    sh = LoadedSparseShard.load(shard_path)
    n_ticks = sh.n_ticks
    # Strictly non-overlapping full windows + optional masked final tail (ZERO `np.clip` tail duplication!)
    stride = seq_len * batch_size
    n_win = min(max_windows, max(1, (n_ticks + stride - 1) // stride))

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
        start_base = wi * stride
        starts = np.array([start_base + b * seq_len for b in range(batch_size)], dtype=np.int64)
        idx_raw = starts[None, :] + np.arange(seq_len, dtype=np.int64)[:, None]
        in_bounds = idx_raw < n_ticks
        idx = np.where(in_bounds, idx_raw, 0)
        d = sh.data
        tb = np.where(in_bounds, d["text_byte"][idx].astype(np.int32), 256)
        vld = np.where(in_bounds, d["valid_mask"][idx].astype(np.float32), 0.0)
        rst = np.where(in_bounds, d["reset_mask"][idx].astype(bool), False)
        rst[0, :] = True
        for b in range(batch_size):
            # Count words strictly on scored target positions (`in_bounds[1:]`)
            m_b = in_bounds[:, b]
            if np.sum(m_b) >= 2:
                txt = decode_utf8_bytes(tb[m_b, b][1:])
                tot_words += max(1, len(txt.split()))
        batch = {
            "modality_id": jnp.asarray(np.where(in_bounds, d["modality_id"][idx].astype(np.int32), 2)),
            "dt_s": jnp.asarray(np.where(in_bounds, d["dt_s"][idx].astype(np.float32), 0.0)),
            "valid_mask": jnp.asarray(vld),
            "target_valid_mask": jnp.asarray(vld),
            "reliability": jnp.asarray(np.where(in_bounds, d["reliability"][idx].astype(np.float32), 1.0)),
            "reset_mask": jnp.asarray(rst),
            "spatial_pos": jnp.asarray(np.where(in_bounds, d["spatial_pos"][idx].astype(np.int32), 0)),
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
        "coverage_frac": round(min(1.0, tot_bytes / max(float(n_ticks), 1.0)), 4),
        "ce_nat_per_byte": round(mean_nat, 5),
        "bits_per_byte": round(bpb, 5),
        "approx_word_perplexity": round(word_ppl, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v3 Unified Any-to-Any Real-Corpus Trainer & SoTA Evaluator.")
    ap.add_argument("--profile", type=str, default="discovery_45m")
    ap.add_argument("--backbone", type=str, default="slwm_v3_routed")
    ap.add_argument("--match-anchor", action="store_true", help="Match comparator params/bytes to slwm_v3_routed.")
    ap.add_argument("--representation-mode", type=str, default="frozen_codec_edge", choices=("frozen_codec_edge", "scratch_edge"))
    ap.add_argument("--use-wy-chunking", action="store_true")
    ap.add_argument("--surprise-chunk-size", type=int, default=1, help="1=exact tick-by-tick; C>1=exact causal Chunk-Anchor Surprise WY.")
    ap.add_argument("--shard-dir", type=Path, default=Path("/mnt/slwm-data/SLWM/v3_preencoded"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--base-lr", type=float, default=2e-3)
    ap.add_argument("--drop-modality-prob", type=float, default=0.25)
    ap.add_argument("--anchor-transition-prob", type=float, default=0.50, help="Fraction of cross-modal windows anchored on source->target transition.")
    ap.add_argument("--multi-horizon-weight", type=float, default=0.15, help="Weight on multi-horizon latent predictive objective (k in {1, 8, 32}).")
    ap.add_argument("--eval-sealed-test", action="store_true", help="Unlock sealed `*_test*.npz` benchmarks (ONLY for frozen final eval!).")
    ap.add_argument("--seed", type=int, default=4201)
    ap.add_argument("--tag", type=str, default="", help="Optional unique output tag (default: `<profile>_<backbone>_s<seed>`).")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    run_tag = args.tag.strip() or f"{args.profile}_{args.backbone}_s{args.seed}"

    if args.match_anchor and args.profile in ("pilot_d256", "discovery_45m", "confirmation_135m", "scale_700m", "scale_1b"):
        spec = build_wave_a_comparator_specs(
            args.profile, representation_mode=args.representation_mode, match_params_to_anchor=True
        )[args.backbone]
    else:
        spec = make_slwm_v3_spec(
            args.profile, backbone_type=args.backbone, representation_mode=args.representation_mode
        )
    if int(args.surprise_chunk_size) > 1 and spec.use_surprise_router:
        from dataclasses import replace as dc_replace
        spec = dc_replace(spec, surprise_chunk_size=int(args.surprise_chunk_size))

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
    # Balanced weights: 45% Text/Code (so byte-level LM & code converge strongly!), 15% LibriSpeech ASR,
    # 10% Clotho, 10% MSR-VTT, 10% VGGSound, 7% Robotics, 3% ESC-50 (auto-renormalized over available shards!).
    tc_shard = args.shard_dir / "text_code/text_code_train_v1.npz"
    if not tc_shard.is_file():
        tc_shard = args.shard_dir / "text_code/text_code_train_24m_balanced.npz"
    train_candidates = [
        ("text_code", tc_shard, 0.45),
        ("librispeech_train", args.shard_dir / "audio_text/librispeech_train_clean_100.npz", 0.15),
        ("clotho", args.shard_dir / "audio_text/clotho_train.npz", 0.10),
        ("esc50", args.shard_dir / "audio_text/esc50_train_f14.npz", 0.03),
        ("greatesthits", args.shard_dir / "av/greatesthits_train.npz", 0.05),
        ("msrvtt", args.shard_dir / "av/msrvtt_train.npz", 0.10),
        ("vggsound", args.shard_dir / "av/vggsound_train.npz", 0.10),
        ("robotics", args.shard_dir / "robotics/robotics_train.npz", 0.07),
    ]
    mixture = MultiShardMixture(train_candidates)
    print(f"[TRAIN] Active domains: {[(n, sh.n_ticks, round(float(p), 3)) for (n, sh, _), p in zip(mixture.shards, mixture.probs)]}", flush=True)

    tx_unit = build_combined_mup_optimizer(all_params, spec, weight_decay=0.01, grad_clip_norm=1.0)
    opt_state = tx_unit.init(all_params)
    grad_accum = max(1, int(args.grad_accum))

    mh_weight = float(args.multi_horizon_weight)

    def _micro_loss(p, mb):
        b = mb["valid_mask"].shape[1]
        st0 = init_core_state(spec, b)
        z, _, diags = scan_core_fast(
            p["core"],
            st0,
            mb,
            spec,
            remat_layers=True,
            matmul_dtype=mm_dtype,
            assume_all_valid=True,
            use_wy_chunking=args.use_wy_chunking,
        )
        loss, m = compute_any2any_multimodal_loss(p["heads"], z, mb, spec, matmul_dtype=mm_dtype)
        # Registered SLWM Multi-Horizon Latent Predictive Objective (`k in {1, 8, 32}`):
        # Forces long-range dynamics (> local attention window W=32) through multi-timescale matrix memory!
        if mh_weight > 0.0:
            from models.v3.core import predict_future_latents
            from models.v3.fast_core import embed_packet_seq
            t_l = z.shape[0]
            ep_id = jnp.cumsum(mb["reset_mask"].astype(jnp.int32), axis=0)
            x_targ_full = jax.lax.stop_gradient(embed_packet_seq(p["core"]["edge"], mb, spec, matmul_dtype=mm_dtype))
            # Project target embedding cleanly to `target_feat_dim` (first 64 dims normalized -> zero extra params!)
            y_lat = x_targ_full[..., : spec.target_feat_dim]
            y_lat = y_lat / jnp.sqrt(jnp.mean(y_lat * y_lat, axis=-1, keepdims=True) + 1e-6)
            preds_mh = predict_future_latents(p["core"], z, mb["action_feat"], mb["has_true_action"], spec)
            mh_terms = []
            for hi, h_step in enumerate((1, 8, 32)):
                if h_step < t_l and hi < preds_mh.shape[2]:
                    same_e = (ep_id[:-h_step] == ep_id[h_step:]).astype(jnp.float32)
                    mv_h = mb["valid_mask"][:-h_step] * mb["target_valid_mask"][h_step:] * same_e
                    diff = preds_mh[:-h_step, :, hi, :] - y_lat[h_step:]
                    ad = jnp.abs(diff)
                    hub = jnp.mean(jnp.where(ad <= 1.0, 0.5 * diff * diff, ad - 0.5), axis=-1)
                    mh_terms.append(jnp.sum(hub * mv_h) / jnp.maximum(jnp.sum(mv_h), 1.0))
            if mh_terms:
                mh_loss = jnp.mean(jnp.stack(mh_terms, axis=0))
                loss = loss + mh_weight * mh_loss
                m["multi_horizon_huber"] = mh_loss
            else:
                m["multi_horizon_huber"] = jnp.asarray(0.0, dtype=jnp.float32)
        else:
            m["multi_horizon_huber"] = jnp.asarray(0.0, dtype=jnp.float32)
        m["mean_nowrite"] = jnp.mean(diags["mean_nowrite"])
        m["mean_eff_write_eta"] = jnp.mean(diags["mean_eff_write_eta"])
        m["max_mem_frob_norm"] = jnp.max(diags["max_mem_frob_norm"])
        return loss, m

    @jax.jit(donate_argnums=(0, 1))
    def _train_step_accum(ap, o_st, micro_batches, lr_arr):
        def _acc_body(carry, mb):
            g_acc, l_acc, m_acc = carry
            (lv, mv), gm = jax.value_and_grad(_micro_loss, has_aux=True)(ap, mb)
            g_acc = jax.tree_util.tree_map(lambda a, b: a + b, g_acc, gm)
            m_acc = jax.tree_util.tree_map(lambda a, b: a + b, m_acc, mv)
            return (g_acc, l_acc + lv, m_acc), None

        g0 = jax.tree_util.tree_map(jnp.zeros_like, ap)
        m0 = {
            "total_loss": jnp.asarray(0.0, dtype=jnp.float32),
            "text_ce_nat": jnp.asarray(0.0, dtype=jnp.float32),
            "text_bpb": jnp.asarray(0.0, dtype=jnp.float32),
            "audio_rvq_ce_nat": jnp.asarray(0.0, dtype=jnp.float32),
            "video_patch_huber": jnp.asarray(0.0, dtype=jnp.float32),
            "action_proprio_huber": jnp.asarray(0.0, dtype=jnp.float32),
            "n_text_targets": jnp.asarray(0.0, dtype=jnp.float32),
            "n_audio_targets": jnp.asarray(0.0, dtype=jnp.float32),
            "n_video_targets": jnp.asarray(0.0, dtype=jnp.float32),
            "n_action_targets": jnp.asarray(0.0, dtype=jnp.float32),
            "multi_horizon_huber": jnp.asarray(0.0, dtype=jnp.float32),
            "mean_nowrite": jnp.asarray(0.0, dtype=jnp.float32),
            "mean_eff_write_eta": jnp.asarray(0.0, dtype=jnp.float32),
            "max_mem_frob_norm": jnp.asarray(0.0, dtype=jnp.float32),
        }
        (g_sum, l_sum, m_sum), _ = jax.lax.scan(_acc_body, (g0, jnp.asarray(0.0, dtype=jnp.float32), m0), micro_batches)
        inv_ga = 1.0 / float(grad_accum)
        g_mean = jax.tree_util.tree_map(lambda x: x * inv_ga, g_sum)
        m_mean = jax.tree_util.tree_map(lambda x: x * inv_ga, m_sum)
        unit_upd, new_o = tx_unit.update(g_mean, o_st, ap)
        upd = jax.tree_util.tree_map(lambda u: u * lr_arr.astype(u.dtype), unit_upd)
        return optax.apply_updates(ap, upd), new_o, l_sum * inv_ga, m_mean

    rng = np.random.default_rng(args.seed + 100)
    t_start = time.perf_counter()
    log_rows: list[dict[str, Any]] = []
    for step in range(args.steps):
        mbs = [
            mixture.sample_batch(
                rng,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                drop_modality_prob=args.drop_modality_prob,
                anchor_transition_prob=args.anchor_transition_prob,
            )
            for _ in range(grad_accum)
        ]
        stacked_mbs = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *mbs)
        step_lr = jnp.asarray(args.base_lr * lr_schedule_mult(step, args.steps), dtype=jnp.float32)
        all_params, opt_state, lv, metrics = _train_step_accum(all_params, opt_state, stacked_mbs, step_lr)
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

    # CRITICAL FIX: Persist frozen checkpoint IMMEDIATELY after training BEFORE running evals!
    # Keep full float32 precision for <= 150M models (and never truncate to float16 which underflows 1/m_d muP weights!).
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import pickle
    ckpt_path = args.out_dir / f"{run_tag}_ckpt.pkl"
    save_dtype_2d = np.float32 if (n_core_p + n_head_p) < 250_000_000 else jnp.bfloat16
    np_params = jax.tree_util.tree_map(
        lambda x: np.asarray(x, dtype=np.float32) if (save_dtype_2d is np.float32 or x.ndim < 2) else jax.device_get(x.astype(jnp.bfloat16)),
        all_params,
    )
    with ckpt_path.open("wb") as f:
        pickle.dump({"spec": spec, "params": np_params, "seed": int(args.seed)}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[CKPT] Saved frozen checkpoint BEFORE evals: {ckpt_path} ({ckpt_path.stat().st_size / 1024**2:.1f} MiB)", flush=True)

    # Free optimizer state (`2x` param HBM = ~9.2 GiB at 1B!) before reloading frozen params for evals!
    del opt_state, all_params
    gc.collect()
    jax.clear_caches()

    # Reload exact saved checkpoint weights into fp32 for frozen evaluation (ensures eval == saved checkpoint!)
    all_params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), np_params)

    def _safe_eval(name: str, fn):
        try:
            return fn()
        except Exception as exc:
            print(f"[EVAL WARN] {name} failed: {exc!r}", flush=True)
            return {"error": repr(exc)}

    eval_cm_seq = max(args.seq_len, 768)
    eval_bsz = min(8, args.batch_size)
    # Strictly DEV / VALIDATION frozen evaluations (ALWAYS safe during Discovery & Hyperparameter selection!)
    def _rso_path(split: str) -> Path:
        if spec.n_layers >= 16:
            p16 = args.shard_dir / f"real_speech_overwrite/real_speech_overwrite_{split}_L16_Weq128_additive.npz"
            if p16.is_file():
                return p16
        return args.shard_dir / f"real_speech_overwrite/real_speech_overwrite_{split}_L12_Weq64_additive.npz"

    evals: dict[str, Any] = {
        "text_code_dev": _safe_eval("text_code_dev", lambda: evaluate_text_sota_shard(
            all_params, spec, args.shard_dir / "text_code/text_code_dev_v1.npz",
            seq_len=args.seq_len, batch_size=args.batch_size, max_windows=64, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "wikitext103_val": _safe_eval("wikitext103_val", lambda: evaluate_text_sota_shard(
            all_params, spec, args.shard_dir / "text_code/wikitext103_validation.npz",
            seq_len=args.seq_len, batch_size=args.batch_size, max_windows=256, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "clotho_dev_audio_to_text_shuffle": _safe_eval("clotho_dev_audio_to_text", lambda: evaluate_cross_modal_condition_shuffle(
            all_params, spec, args.shard_dir / "audio_text/clotho_dev.npz",
            source_mod=1, target_mod=2, seq_len=eval_cm_seq, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "librispeech_dev_clean_audio_to_text_shuffle": _safe_eval("librispeech_dev", lambda: evaluate_cross_modal_condition_shuffle(
            all_params, spec, args.shard_dir / "audio_text/librispeech_dev_clean.npz",
            source_mod=1, target_mod=2, seq_len=eval_cm_seq, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "msrvtt_dev_video_to_text_shuffle": _safe_eval("msrvtt_dev_v2t", lambda: evaluate_cross_modal_condition_shuffle(
            all_params, spec, args.shard_dir / "av/msrvtt_dev.npz",
            source_mod=0, target_mod=2, seq_len=eval_cm_seq, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "msrvtt_dev_audio_to_text_shuffle": _safe_eval("msrvtt_dev_a2t", lambda: evaluate_cross_modal_condition_shuffle(
            all_params, spec, args.shard_dir / "av/msrvtt_dev.npz",
            source_mod=1, target_mod=2, seq_len=eval_cm_seq, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "vggsound_dev_video_to_audio_shuffle": _safe_eval("vggsound_dev_v2a", lambda: evaluate_cross_modal_condition_shuffle(
            all_params, spec, args.shard_dir / "av/vggsound_dev.npz",
            source_mod=0, target_mod=1, seq_len=eval_cm_seq, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "vggsound_dev_audio_to_video_shuffle": _safe_eval("vggsound_dev_a2v", lambda: evaluate_cross_modal_condition_shuffle(
            all_params, spec, args.shard_dir / "av/vggsound_dev.npz",
            source_mod=1, target_mod=0, seq_len=eval_cm_seq, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "robotics_dev_action_intervention": _safe_eval("robotics_dev_act", lambda: evaluate_robotics_action_intervention(
            all_params, spec, args.shard_dir / "robotics/robotics_dev.npz",
            seq_len=args.seq_len, batch_size=eval_bsz,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "real_speech_overwrite_dev_additive_rf": _safe_eval("rso_dev", lambda: evaluate_real_speech_overwrite_frozen_core(
            all_params["core"],
            spec,
            _rso_path("train"),
            _rso_path("dev"),
            matmul_dtype=mm_dtype,
            use_wy_chunking=args.use_wy_chunking,
        )),
        "esc50_dev_50class_probe": _safe_eval("esc50_dev", lambda: evaluate_esc50_classification(
            all_params["core"], spec, args.shard_dir / "audio_text/esc50_train_f14.npz",
            eval_shard_path=None, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "librispeech_dev_clean_wer": _safe_eval("librispeech_dev_wer", lambda: evaluate_librispeech_wer(
            all_params, spec, args.shard_dir / "audio_text/librispeech_dev_clean.npz",
            max_utterances=64, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
    }

    # STRICT SEALED-TEST LOCK (AGENTS.md §12 Gate 2 & §13.2): ONLY unlocked when `--eval-sealed-test` is passed!
    if args.eval_sealed_test:
        def _load_jsonl(p: Path) -> list[dict[str, str]]:
            if not p.is_file():
                return []
            return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]

        evals["sealed_test"] = {
            "wikitext103_test": _safe_eval("wikitext103_test", lambda: evaluate_text_sota_shard(
                all_params, spec, args.shard_dir / "text_code/wikitext103_test.npz",
                seq_len=args.seq_len, batch_size=args.batch_size, max_windows=256, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "lambada_openai_test_full": _safe_eval("lambada_test", lambda: evaluate_lambada_openai(
                all_params, spec, args.shard_dir / "text_code/lambada_openai_test.npz",
                max_passages=None, seq_len=args.seq_len, batch_size=args.batch_size, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "humaneval_test_bpb": _safe_eval("humaneval_test", lambda: evaluate_code_solution_bpb(
                all_params, spec, args.shard_dir / "text_code/humaneval_test.npz",
                seq_len=args.seq_len, batch_size=args.batch_size, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "humaneval_executed_pass_at_1": _safe_eval("humaneval_exec", lambda: evaluate_code_execution_pass_at_1(
                all_params, spec, _load_jsonl(args.shard_dir / "text_code/humaneval_exec.jsonl"),
                matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "mbpp_sanitized_test_bpb": _safe_eval("mbpp_test", lambda: evaluate_code_solution_bpb(
                all_params, spec, args.shard_dir / "text_code/mbpp_sanitized_test.npz",
                seq_len=args.seq_len, batch_size=args.batch_size, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "mbpp_executed_pass_at_1": _safe_eval("mbpp_exec", lambda: evaluate_code_execution_pass_at_1(
                all_params, spec, _load_jsonl(args.shard_dir / "text_code/mbpp_exec.jsonl"),
                matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "clotho_test_audio_to_text_shuffle": _safe_eval("clotho_test", lambda: evaluate_cross_modal_condition_shuffle(
                all_params, spec, args.shard_dir / "audio_text/clotho_test.npz",
                source_mod=1, target_mod=2, seq_len=eval_cm_seq, batch_size=eval_bsz,
                matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "librispeech_test_clean_audio_to_text_shuffle": _safe_eval("librispeech_test", lambda: evaluate_cross_modal_condition_shuffle(
                all_params, spec, args.shard_dir / "audio_text/librispeech_test_clean.npz",
                source_mod=1, target_mod=2, seq_len=eval_cm_seq, batch_size=eval_bsz,
                matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "esc50_test_f5_audio_to_text_shuffle": _safe_eval("esc50_test", lambda: evaluate_cross_modal_condition_shuffle(
                all_params, spec, args.shard_dir / "audio_text/esc50_test_f5.npz",
                source_mod=1, target_mod=2, seq_len=eval_cm_seq, batch_size=eval_bsz,
                matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "msrvtt_test_video_to_text_shuffle": _safe_eval("msrvtt_test", lambda: evaluate_cross_modal_condition_shuffle(
                all_params, spec, args.shard_dir / "av/msrvtt_test.npz",
                source_mod=0, target_mod=2, seq_len=eval_cm_seq, batch_size=eval_bsz,
                matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "vggsound_test_video_to_audio_shuffle": _safe_eval("vggsound_test_v2a", lambda: evaluate_cross_modal_condition_shuffle(
                all_params, spec, args.shard_dir / "av/vggsound_test.npz",
                source_mod=0, target_mod=1, seq_len=eval_cm_seq, batch_size=eval_bsz,
                matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "robotics_test_action_intervention": _safe_eval("robotics_test", lambda: evaluate_robotics_action_intervention(
                all_params, spec, args.shard_dir / "robotics/robotics_test.npz",
                seq_len=args.seq_len, batch_size=eval_bsz,
                matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "esc50_test_f5_50class_probe": _safe_eval("esc50_test_50cls", lambda: evaluate_esc50_classification(
                all_params["core"], spec, args.shard_dir / "audio_text/esc50_train_f14.npz",
                eval_shard_path=args.shard_dir / "audio_text/esc50_test_f5.npz",
                matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "librispeech_test_clean_wer": _safe_eval("librispeech_test_wer", lambda: evaluate_librispeech_wer(
                all_params, spec, args.shard_dir / "audio_text/librispeech_test_clean.npz",
                max_utterances=128, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "real_speech_overwrite_test_additive_rf": _safe_eval("rso_test", lambda: evaluate_real_speech_overwrite_frozen_core(
                all_params["core"],
                spec,
                _rso_path("train"),
                _rso_path("test"),
                matmul_dtype=mm_dtype,
                use_wy_chunking=args.use_wy_chunking,
            )),
        }

    # Deterministic uncurated `inference_showcase/` gallery (AGENTS.md §13.2: P05/P25/P50/P75/P95, frozen weights!)
    showcase_dir = args.out_dir / "inference_showcase"
    evals["inference_showcase"] = {
        "clotho_dev": _safe_eval("showcase_clotho", lambda: generate_cross_modal_text_showcase(
            all_params, spec, args.shard_dir / "audio_text/clotho_dev.npz",
            out_dir=showcase_dir, split="dev", model_name=run_tag,
            seq_len=args.seq_len, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "msrvtt_dev": _safe_eval("showcase_msrvtt", lambda: generate_cross_modal_text_showcase(
            all_params, spec, args.shard_dir / "av/msrvtt_dev.npz",
            out_dir=showcase_dir, split="dev", model_name=run_tag,
            seq_len=args.seq_len, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
        "wikitext103_val": _safe_eval("showcase_wt103", lambda: generate_cross_modal_text_showcase(
            all_params, spec, args.shard_dir / "text_code/wikitext103_validation.npz",
            out_dir=showcase_dir, split="dev", model_name=run_tag,
            seq_len=args.seq_len, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )),
    }
    if args.eval_sealed_test:
        evals["inference_showcase_sealed_test"] = {
            "clotho_test": _safe_eval("showcase_clotho_test", lambda: generate_cross_modal_text_showcase(
                all_params, spec, args.shard_dir / "audio_text/clotho_test.npz",
                out_dir=showcase_dir, split="test", model_name=run_tag,
                seq_len=args.seq_len, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "msrvtt_test": _safe_eval("showcase_msrvtt_test", lambda: generate_cross_modal_text_showcase(
                all_params, spec, args.shard_dir / "av/msrvtt_test.npz",
                out_dir=showcase_dir, split="test", model_name=run_tag,
                seq_len=args.seq_len, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
            "wikitext103_test": _safe_eval("showcase_wt103_test", lambda: generate_cross_modal_text_showcase(
                all_params, spec, args.shard_dir / "text_code/wikitext103_test.npz",
                out_dir=showcase_dir, split="test", model_name=run_tag,
                seq_len=args.seq_len, matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
            )),
        }

    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tag": run_tag,
        "profile": args.profile,
        "backbone": args.backbone,
        "representation_mode": args.representation_mode,
        "use_wy_chunking": bool(args.use_wy_chunking),
        "surprise_chunk_size": int(spec.surprise_chunk_size),
        "multi_horizon_weight": float(mh_weight),
        "anchor_transition_prob": float(args.anchor_transition_prob),
        "eval_sealed_test_enabled": bool(args.eval_sealed_test),
        "seed": int(args.seed),
        "trainable_core_params": int(n_core_p),
        "trainable_head_params": int(n_head_p),
        "core_streaming_state_bytes_b1": int(st_bytes["total_core_state_bytes"]),
        "grad_accum": int(grad_accum),
        "effective_batch_size": int(args.batch_size * grad_accum),
        "total_positions_trained": int(args.steps * args.batch_size * grad_accum * args.seq_len),
        "base_lr": float(args.base_lr),
        "wall_sec": round(time.perf_counter() - t_start, 2),
        "train_curve": log_rows,
        "frozen_evaluations": evals,
    }
    (args.out_dir / f"{run_tag}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
