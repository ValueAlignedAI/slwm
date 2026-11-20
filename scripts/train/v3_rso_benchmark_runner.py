"""SLWM-v3 Real-Speech Selective Overwrite Multi-Seed Benchmark (`100% Real Kyutai Mimi 32-RVQ Speech`).

Tests the primary SLWM-v3 research hypothesis on 100% real, speaker-disjoint & background-disjoint
multi-minute speech streams (`167 s` / `2,088` ticks for Discovery `L=12, W_eq=64`; `274 s` / `3,425`
ticks for Confirmation `L=16, W_eq=128`) where counterfactual A/B streams are verified BIT-EXACT
IDENTICAL (`0.000000` mismatch) across the ENTIRE accessible SlidingWindow-GQA receptive field!

Protocol (strictly matched `±1.0%` core params, matched streaming state bytes, zero task opcodes):
  - Trains each backbone from scratch (or optional pretrained checkpoint) on `real_speech_overwrite_train`
    using causal final-state readout (`dir` + `dig` attributes) + auxiliary self-supervised next-packet
    Mimi RVQ-0..3 acoustic prediction (`aux_mimi_weight = 0.05`) under base-shape muP (`lr <= 1e-3`).
  - Evaluates strictly frozen weights (`no` online updates!) on speaker-disjoint `dev` (and sealed `test`
    ONLY when `--eval-sealed-test` is passed!), scoring:
      * Overwrite trials (`~is_catch`): counterfactual A/B flip on updated attr AND preserve other attr!
      * Catch trials (`is_catch`, zero correction): dual preserve of BOTH initial attributes on A & B!
      * Per-speaker-cluster accuracy vector for exact crossed-seed & speaker-cluster 95% CIs!
"""

from __future__ import annotations

import argparse
from dataclasses import replace as dc_replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models.v3.core import (
    build_mup_lr_multipliers,
    build_weight_decay_mask,
    init_core_state,
    init_slwm_v3_params,
)
from models.v3.fast_core import scan_core_fast
from models.v3.memory import init_mup_weight, rms_norm
from models.v3.spec import (
    build_wave_a_comparator_specs,
    compute_core_streaming_state_bytes,
    count_tree_params,
    make_slwm_v3_spec,
)
from scripts.train.v3_pilot_train import lr_schedule_mult


def _init_rso_heads(key: jax.Array, spec, vocab_size: int = 4) -> dict[str, Any]:
    k1, k2 = jax.random.split(key, 2)
    d = spec.d_model
    m_d = spec.width_mult
    return {
        "head_state": {
            "norm_w": jnp.ones((d,), dtype=jnp.float32),
            "w": init_mup_weight(k1, (d, 2 * vocab_size), role="readout", width_mult=m_d, zero_init=True),
            "bias": jnp.zeros((2 * vocab_size,), dtype=jnp.float32),
        },
        "head_mimi_aux": {
            "norm_w": jnp.ones((d,), dtype=jnp.float32),
            "w": init_mup_weight(k2, (d, 2 * spec.audio_vocab_size), role="readout", width_mult=m_d, zero_init=True),
            "bias": jnp.zeros((2, spec.audio_vocab_size), dtype=jnp.float32),
        },
    }


def _make_mimi_packets(codes_tb32: jax.Array, spec) -> dict[str, jax.Array]:
    t_len, b_sz, _ = codes_tb32.shape
    return {
        "modality_id": jnp.ones((t_len, b_sz), dtype=jnp.int32),
        "dt_s": jnp.full((t_len, b_sz), 0.08, dtype=jnp.float32),  # 12.5 Hz Kyutai Mimi = 0.08 s/tick
        "valid_mask": jnp.ones((t_len, b_sz), dtype=jnp.float32),
        "reliability": jnp.ones((t_len, b_sz), dtype=jnp.float32),
        "reset_mask": jnp.zeros((t_len, b_sz), dtype=jnp.bool_).at[0].set(True),
        "spatial_pos": jnp.zeros((t_len, b_sz), dtype=jnp.int32),
        "text_byte": jnp.full((t_len, b_sz), 256, dtype=jnp.int32),
        "has_true_action": jnp.zeros((t_len, b_sz), dtype=jnp.float32),
        "video_feat": jnp.zeros((t_len, b_sz, spec.video_patch_dim), dtype=jnp.float32),
        "audio_codes": codes_tb32,
        "proprio_feat": jnp.zeros((t_len, b_sz, spec.proprio_dim), dtype=jnp.float32),
        "action_feat": jnp.zeros((t_len, b_sz, spec.action_dim), dtype=jnp.float32),
        "probe_feat": jnp.zeros((t_len, b_sz, spec.probe_dim), dtype=jnp.float32),
    }


def _build_rso_optimizer(all_params: dict[str, Any], spec, weight_decay: float = 0.01) -> optax.GradientTransformation:
    mults = {
        "core": build_mup_lr_multipliers(all_params["core"], spec),
        "heads": build_mup_lr_multipliers(all_params["heads"], spec),
    }
    wd_mask = {
        "core": build_weight_decay_mask(all_params["core"]),
        "heads": build_weight_decay_mask(all_params["heads"]),
    }

    def _scale_mult():
        return optax.GradientTransformation(
            lambda _p: optax.EmptyState(),
            lambda u, s, _p=None: (jax.tree_util.tree_map(lambda ui, mi: ui * jnp.asarray(mi, dtype=ui.dtype), u, mults), s),
        )

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(b1=0.9, b2=0.95, eps=1e-8),
        _scale_mult(),
        optax.add_decayed_weights(weight_decay, mask=wd_mask),
        optax.scale(-1.0),
    )


class DynamicRealSpeechClipbank:
    """Zero-memorization dynamic real-speech stream synthesizer strictly from `train`-split SpeechCommands Mimi clips."""

    def __init__(self, npz_path: Path) -> None:
        z = np.load(npz_path, allow_pickle=False)
        self.dir_codes = np.asarray(z["dir_codes"], dtype=np.int32)  # [4, M, 25, 32]
        self.dig_codes = np.asarray(z["dig_codes"], dtype=np.int32)  # [4, M, 25, 32]
        self.dist_codes = np.asarray(z["dist_codes"], dtype=np.int32)  # [K, 25, 32]
        self.bg_codes = np.asarray(z["bg_codes"], dtype=np.int32)  # [B, 125, 32]
        self.clip_len = int(self.dir_codes.shape[2])
        self.bg_len = int(self.bg_codes.shape[1])

    def sample_batch(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        seq_len: int,
        catch_prob: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(codes [T, B, 32], y_running [T, B, 2], loss_w [T, B])` with infinite unique distractor suffixes."""
        c_len = self.clip_len
        codes = np.zeros((seq_len, batch_size, 32), dtype=np.int32)
        y_run = np.zeros((seq_len, batch_size, 2), dtype=np.int32)
        lw = np.zeros((seq_len, batch_size), dtype=np.float32)

        for b in range(batch_size):
            # 1. Tile random pure train-background beds across [0, T)
            pos = 0
            while pos < seq_len:
                bg = self.bg_codes[int(rng.integers(0, len(self.bg_codes)))]
                off = int(rng.integers(0, max(1, self.bg_len - 25)))
                seg = bg[off:]
                take = min(seq_len - pos, len(seg))
                codes[pos : pos + take, b] = seg[:take]
                pos += take

            d_init = int(rng.integers(0, 4))
            n_init = int(rng.integers(0, 4))
            order = int(rng.integers(0, 2))
            is_catch = bool(rng.random() < catch_prob)

            # 2. Place initial Direction + Digit spoken words with random jitter
            t1 = int(rng.integers(8, min(32, max(9, seq_len // 10))))
            t2 = t1 + c_len + int(rng.integers(4, 24))
            if t2 + c_len + 40 >= seq_len:
                t1 = 4
                t2 = t1 + c_len + 4

            c_dir = self.dir_codes[d_init, int(rng.integers(0, self.dir_codes.shape[1]))]
            c_dig = self.dig_codes[n_init, int(rng.integers(0, self.dig_codes.shape[1]))]
            if order == 0:
                codes[t1 : t1 + c_len, b] = c_dir
                codes[t2 : t2 + c_len, b] = c_dig
            else:
                codes[t1 : t1 + c_len, b] = c_dig
                codes[t2 : t2 + c_len, b] = c_dir

            # 3. Place optional selective overwrite (or matched-energy catch distractor)
            max_corr = min(seq_len - c_len - 16, max(t2 + c_len + 36, min(420, seq_len // 3)))
            min_corr = min(max_corr - 1, t2 + c_len + 12)
            t_corr = int(rng.integers(min_corr, max_corr + 1))

            d_fin, n_fin = d_init, n_init
            if is_catch:
                codes[t_corr : t_corr + c_len, b] = self.dist_codes[int(rng.integers(0, len(self.dist_codes)))]
            else:
                uw = int(rng.integers(0, 2))
                old_v = d_init if uw == 0 else n_init
                new_v = (old_v + int(rng.integers(1, 4))) % 4
                if uw == 0:
                    d_fin = new_v
                    codes[t_corr : t_corr + c_len, b] = self.dir_codes[new_v, int(rng.integers(0, self.dir_codes.shape[1]))]
                else:
                    n_fin = new_v
                    codes[t_corr : t_corr + c_len, b] = self.dig_codes[new_v, int(rng.integers(0, self.dig_codes.shape[1]))]

            # 4. Fill ALL non-overlapping gaps (mid + long post suffix!) with fresh random matched-energy distractors
            occupied = [(t1, t1 + c_len), (t2, t2 + c_len), (t_corr, t_corr + c_len)]
            cur = 2
            while cur + c_len < seq_len - 4:
                step_gap = int(rng.integers(c_len + 6, c_len + 34))
                if not any(not (cur + c_len + 2 <= s or cur >= e + 2) for s, e in occupied):
                    codes[cur : cur + c_len, b] = self.dist_codes[int(rng.integers(0, len(self.dist_codes)))]
                cur += step_gap

            # 5. Strictly causal running-state targets & probe mask (IDENTICAL across all compared backbones!)
            y_run[:, b, 0] = d_init
            y_run[:, b, 1] = n_init
            y_run[t_corr + c_len :, b, 0] = d_fin
            y_run[t_corr + c_len :, b, 1] = n_fin

            # Probe 1: right after initial pair (teaches spoken word recognition!)
            p1_s = min(seq_len - 1, t2 + c_len)
            p1_e = min(t_corr, p1_s + 8)
            if p1_e > p1_s:
                lw[p1_s:p1_e, b] = 0.5 / float(p1_e - p1_s)
            # Probe 2: right after overwrite/catch (teaches selective overwrite & preservation!)
            p2_s = min(seq_len - 1, t_corr + c_len)
            p2_e = min(seq_len, p2_s + 8)
            if p2_e > p2_s:
                lw[p2_s:p2_e, b] = 1.0 / float(p2_e - p2_s)
            # Probe 3: final tail post-distractors (forces long-horizon survival > 756/2032 ticks!)
            p3_s = max(p2_e, seq_len - 32)
            if seq_len > p3_s:
                lw[p3_s:seq_len, b] = 3.0 / float(seq_len - p3_s)

        return codes, y_run, lw


def evaluate_rso_split(
    all_params: dict[str, Any],
    spec,
    npz_path: Path,
    *,
    vocab_size: int = 4,
    eval_bsz: int = 8,
    eval_tick: int | None = None,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    use_wy_chunking: bool = False,
) -> dict[str, Any]:
    """Strictly frozen-weight paired counterfactual A/B evaluation with speaker-cluster provenance."""
    d = np.load(npz_path, allow_pickle=False)
    ca = d["audio_codes_a"]
    cb = d["audio_codes_b"]
    if eval_tick is not None and eval_tick < ca.shape[1]:
        ca = ca[:, :eval_tick, :]
        cb = cb[:, :eval_tick, :]
    ya = np.asarray(d["y_a"], dtype=np.int32)
    yb = np.asarray(d["y_b"], dtype=np.int32)
    ua = np.asarray(d["update_which"], dtype=np.int32)
    is_catch = np.asarray(d["is_catch"], dtype=bool)
    spks = [str(x) for x in d["primary_speaker_id"].tolist()]

    @jax.jit
    def _pred_batch(ap, codes_tb32):
        b_sz = codes_tb32.shape[1]
        pkts = _make_mimi_packets(codes_tb32, spec)
        z, _, dm = scan_core_fast(
            ap["core"],
            init_core_state(spec, b_sz),
            pkts,
            spec,
            remat_layers=False,
            matmul_dtype=matmul_dtype,
            assume_all_valid=True,
            use_wy_chunking=use_wy_chunking,
        )
        hs = ap["heads"]["head_state"]
        # Read out from final tick `z[-1]` strictly causally!
        zf = rms_norm(z[-1], hs["norm_w"])
        lg = jnp.matmul(zf, hs["w"]) + hs["bias"]
        return lg.reshape(b_sz, 2, vocab_size), jnp.mean(dm["mean_nowrite"]), jnp.mean(dm["mean_eff_write_eta"])

    def _run(codes_nt32: np.ndarray) -> tuple[np.ndarray, float, float]:
        preds, nws, etas = [], [], []
        for s in range(0, len(codes_nt32), eval_bsz):
            e = min(s + eval_bsz, len(codes_nt32))
            c_tb = jnp.asarray(np.transpose(codes_nt32[s:e], (1, 0, 2)), dtype=jnp.int32)
            lg, nw, et = _pred_batch(all_params, c_tb)
            preds.append(np.asarray(jnp.argmax(lg, axis=-1), dtype=np.int32))
            nws.append(float(nw))
            etas.append(float(et))
        return np.concatenate(preds, axis=0), float(np.mean(nws)), float(np.mean(etas))

    pa, nw_a, eta_a = _run(ca)
    pb, nw_b, eta_b = _run(cb)

    ow_idx = np.nonzero((~is_catch) & (ua >= 0) & (ua <= 1))[0]
    c_idx = np.nonzero(is_catch | (ua < 0))[0]
    ow_ua = ua[ow_idx]

    upd_ok = (pa[ow_idx, ow_ua] == ya[ow_idx, ow_ua]) & (pb[ow_idx, ow_ua] == yb[ow_idx, ow_ua])
    pres_ok = (pa[ow_idx, 1 - ow_ua] == ya[ow_idx, 1 - ow_ua]) & (pb[ow_idx, 1 - ow_ua] == yb[ow_idx, 1 - ow_ua])
    ab_joint_ow = upd_ok & pres_ok
    catch_ok = (
        (pa[c_idx, 0] == ya[c_idx, 0])
        & (pa[c_idx, 1] == ya[c_idx, 1])
        & (pb[c_idx, 0] == yb[c_idx, 0])
        & (pb[c_idx, 1] == yb[c_idx, 1])
    )
    # Single-stream marginal accuracy on stream A (diagnostic)
    marg_a = (pa[:, 0] == ya[:, 0]) & (pa[:, 1] == ya[:, 1])

    overall_vec = np.zeros((len(ya),), dtype=np.int32)
    overall_vec[ow_idx] = ab_joint_ow.astype(np.int32)
    overall_vec[c_idx] = catch_ok.astype(np.int32)

    # Speaker-cluster bootstrap 95% CI on overall joint accuracy
    spk_to_vals: dict[str, list[int]] = {}
    for spk, v in zip(spks, overall_vec.tolist()):
        spk_to_vals.setdefault(spk, []).append(int(v))
    spk_keys = sorted(spk_to_vals.keys())
    rng_b = np.random.default_rng(2026)
    boot_means = []
    for _ in range(1000):
        draw = rng_b.integers(0, len(spk_keys), size=(len(spk_keys),))
        vals = [v for di in draw for v in spk_to_vals[spk_keys[int(di)]]]
        boot_means.append(float(np.mean(vals)))
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    return {
        "shard": npz_path.name,
        "n_pairs": int(len(ya)),
        "n_overwrite_pairs": int(len(ow_idx)),
        "n_catch_pairs": int(len(c_idx)),
        "n_unique_primary_speakers": len(spk_keys),
        "seq_len_ticks_12p5hz": int(ca.shape[1]),
        "overwrite_ab_flip_and_preserve": round(float(np.mean(ab_joint_ow)), 4) if len(ab_joint_ow) else 0.0,
        "overwrite_updated_attr_flip_acc": round(float(np.mean(upd_ok)), 4) if len(upd_ok) else 0.0,
        "overwrite_preserved_attr_acc": round(float(np.mean(pres_ok)), 4) if len(pres_ok) else 0.0,
        "catch_dual_preserve_acc": round(float(np.mean(catch_ok)), 4) if len(catch_ok) else 0.0,
        "overall_counterfactual_joint_acc": round(float(np.mean(overall_vec)), 4),
        "overall_speaker_cluster_ci95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "single_stream_a_joint_acc": round(float(np.mean(marg_a)), 4),
        "mean_nowrite": round(0.5 * (nw_a + nw_b), 4),
        "mean_eff_write_eta": round(0.5 * (eta_a + eta_b), 4),
        "episode_overall_joint_vector": [int(x) for x in overall_vec],
        "episode_primary_speaker_ids": spks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SLWM-v3 Real-Speech Selective Overwrite Multi-Seed Benchmark.")
    ap.add_argument("--profile", type=str, default="discovery_45m")
    ap.add_argument("--backbone", type=str, default="slwm_v3_routed")
    ap.add_argument("--surprise-chunk-size", type=int, default=1)
    ap.add_argument("--use-wy-chunking", action="store_true")
    ap.add_argument("--rso-dir", type=Path, required=True)
    ap.add_argument("--shard-tag", type=str, default="L12_Weq64_additive")
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--phase-a-steps", type=int, default=250, help="Short-horizon word+overwrite grounding steps when `--train-clipbank` is set.")
    ap.add_argument("--phase-a-len", type=int, default=256, help="Short-horizon sequence length for Phase A grounding.")
    ap.add_argument("--train-clipbank", type=Path, default=None, help="Optional train-speaker Mimi clipbank for zero-memorization dynamic synthesis.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--base-lr", type=float, default=1e-3)
    ap.add_argument("--aux-mimi-weight", type=float, default=0.05)
    ap.add_argument("--eval-sealed-test", action="store_true")
    ap.add_argument("--seed", type=int, default=4201)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    spec = build_wave_a_comparator_specs(args.profile, match_params_to_anchor=True)[args.backbone]
    if int(args.surprise_chunk_size) > 1 and spec.use_surprise_router:
        spec = dc_replace(spec, surprise_chunk_size=int(args.surprise_chunk_size))

    is_cpu = str(jax.devices()[0].platform).lower() == "cpu"
    mm_dtype = jnp.float32 if is_cpu else jnp.bfloat16

    tr_p = args.rso_dir / f"real_speech_overwrite_train_{args.shard_tag}.npz"
    dv_p = args.rso_dir / f"real_speech_overwrite_dev_{args.shard_tag}.npz"
    te_p = args.rso_dir / f"real_speech_overwrite_test_{args.shard_tag}.npz"
    tr_d = np.load(tr_p, allow_pickle=False)
    tr_codes = np.concatenate([tr_d["audio_codes_a"], tr_d["audio_codes_b"]], axis=0)  # [2*N, T, 32]
    tr_y = np.concatenate([tr_d["y_a"], tr_d["y_b"]], axis=0).astype(np.int32)  # [2*N, 2]
    n_tr = len(tr_codes)
    target_seq_len = int(tr_codes.shape[1])
    vocab_size = int(np.max(tr_y)) + 1
    clipbank = DynamicRealSpeechClipbank(args.train_clipbank) if (args.train_clipbank and args.train_clipbank.is_file()) else None
    aux_w = float(args.aux_mimi_weight)

    k_c, k_h = jax.random.split(jax.random.PRNGKey(args.seed), 2)
    all_params = {
        "core": init_slwm_v3_params(k_c, spec),
        "heads": _init_rso_heads(k_h, spec, vocab_size=vocab_size),
    }
    n_core_p = count_tree_params(all_params["core"])
    st_bytes = compute_core_streaming_state_bytes(spec, batch_size=1)
    tx_unit = _build_rso_optimizer(all_params, spec, weight_decay=0.01)
    opt_st = tx_unit.init(all_params)
    grad_accum = max(1, int(args.grad_accum))

    def _micro_loss_running(ap, codes_tb32, y_tb2, lw_tb):
        b_sz = codes_tb32.shape[1]
        pkts = _make_mimi_packets(codes_tb32, spec)
        z, _, dm = scan_core_fast(
            ap["core"],
            init_core_state(spec, b_sz),
            pkts,
            spec,
            remat_layers=True,
            matmul_dtype=mm_dtype,
            assume_all_valid=True,
            use_wy_chunking=args.use_wy_chunking,
        )
        hs = ap["heads"]["head_state"]
        z_n = rms_norm(z, hs["norm_w"])  # [T, B, D]
        lg = (jnp.matmul(z_n.astype(mm_dtype), hs["w"].astype(mm_dtype)).astype(jnp.float32) + hs["bias"]).reshape(
            z.shape[0], b_sz, 2, vocab_size
        )
        ce0 = optax.softmax_cross_entropy_with_integer_labels(lg[:, :, 0, :], y_tb2[:, :, 0])
        ce1 = optax.softmax_cross_entropy_with_integer_labels(lg[:, :, 1, :], y_tb2[:, :, 1])
        ce_tb = 0.5 * (ce0 + ce1)
        state_loss = jnp.sum(ce_tb * lw_tb) / jnp.maximum(jnp.sum(lw_tb), 1e-6)
        if aux_w > 0.0:
            hm = ap["heads"]["head_mimi_aux"]
            z_sub = rms_norm(z[:-1:8], hm["norm_w"])
            t_sub = codes_tb32[1::8, :, :2]
            m_lg = (jnp.matmul(z_sub.astype(mm_dtype), hm["w"].astype(mm_dtype)).astype(jnp.float32)).reshape(
                z_sub.shape[0], b_sz, 2, spec.audio_vocab_size
            ) + hm["bias"][None, None, :, :]
            aux_loss = optax.softmax_cross_entropy_with_integer_labels(m_lg, t_sub).mean()
        else:
            aux_loss = jnp.asarray(0.0, dtype=jnp.float32)
        tot = state_loss + aux_w * aux_loss
        return tot, {
            "state_ce": state_loss,
            "aux_mimi_ce": aux_loss,
            "mean_nowrite": jnp.mean(dm["mean_nowrite"]),
            "mean_eta": jnp.mean(dm["mean_eff_write_eta"]),
            "max_frob": jnp.max(dm["max_mem_frob_norm"]),
        }

    @jax.jit(donate_argnums=(0, 1))
    def _step_running(ap, ost, mb_codes, mb_y, mb_lw, lr_arr):
        def _acc(carry, xs):
            ga, la, ma = carry
            c_tb, y_tb, w_tb = xs
            (lv, mv), gm = jax.value_and_grad(_micro_loss_running, has_aux=True)(ap, c_tb, y_tb, w_tb)
            ga = jax.tree_util.tree_map(lambda a, b: a + b, ga, gm)
            ma = jax.tree_util.tree_map(lambda a, b: a + b, ma, mv)
            return (ga, la + lv, ma), None

        g0 = jax.tree_util.tree_map(jnp.zeros_like, ap)
        m0 = {k: jnp.asarray(0.0, dtype=jnp.float32) for k in ("state_ce", "aux_mimi_ce", "mean_nowrite", "mean_eta", "max_frob")}
        (gs, ls, ms), _ = jax.lax.scan(_acc, (g0, jnp.asarray(0.0, dtype=jnp.float32), m0), (mb_codes, mb_y, mb_lw))
        inv = 1.0 / float(grad_accum)
        gm = jax.tree_util.tree_map(lambda x: x * inv, gs)
        mm = jax.tree_util.tree_map(lambda x: x * inv, ms)
        u, nost = tx_unit.update(gm, ost, ap)
        u = jax.tree_util.tree_map(lambda x: x * lr_arr.astype(x.dtype), u)
        return optax.apply_updates(ap, u), nost, ls * inv, mm

    rng = np.random.default_rng(args.seed + 300)
    t0 = time.perf_counter()
    curve = []
    if clipbank is not None:
        # Phase A: Short-horizon spoken-word + selective-overwrite grounding (identical for ALL comparators!)
        pa_steps = max(0, int(args.phase_a_steps))
        for s in range(pa_steps):
            c_mbs, y_mbs, w_mbs = [], [], []
            for _ in range(grad_accum):
                c_tb, y_tb, w_tb = clipbank.sample_batch(rng, batch_size=args.batch_size, seq_len=int(args.phase_a_len))
                c_mbs.append(c_tb)
                y_mbs.append(y_tb)
                w_mbs.append(w_tb)
            lr_v = jnp.asarray(args.base_lr * lr_schedule_mult(s, max(pa_steps + args.steps, 1)), dtype=jnp.float32)
            all_params, opt_st, lv, met = _step_running(
                all_params, opt_st,
                jnp.asarray(np.stack(c_mbs, axis=0)),
                jnp.asarray(np.stack(y_mbs, axis=0)),
                jnp.asarray(np.stack(w_mbs, axis=0)),
                lr_v,
            )
            if s == 0 or (s + 1) % max(1, pa_steps // 5) == 0 or s == pa_steps - 1:
                lv.block_until_ready()
                row = {
                    "phase": "A_short", "step": s + 1, "elapsed_s": round(time.perf_counter() - t0, 1),
                    "total_loss": round(float(lv), 4), "state_ce": round(float(met["state_ce"]), 4),
                    "mean_nowrite": round(float(met["mean_nowrite"]), 4), "mean_eta": round(float(met["mean_eta"]), 4),
                    "max_frob": round(float(met["max_frob"]), 2),
                }
                curve.append(row)
                print(f"[RSO-A step={s+1:4d}/{pa_steps}] {row}", flush=True)

        # Phase B: Full additive-RF horizon (`T=2088` / `3425`) dynamic zero-memorization distractor survival!
        for s in range(args.steps):
            c_mbs, y_mbs, w_mbs = [], [], []
            for _ in range(grad_accum):
                c_tb, y_tb, w_tb = clipbank.sample_batch(rng, batch_size=args.batch_size, seq_len=target_seq_len)
                c_mbs.append(c_tb)
                y_mbs.append(y_tb)
                w_mbs.append(w_tb)
            lr_v = jnp.asarray(args.base_lr * lr_schedule_mult(pa_steps + s, max(pa_steps + args.steps, 1)), dtype=jnp.float32)
            all_params, opt_st, lv, met = _step_running(
                all_params, opt_st,
                jnp.asarray(np.stack(c_mbs, axis=0)),
                jnp.asarray(np.stack(y_mbs, axis=0)),
                jnp.asarray(np.stack(w_mbs, axis=0)),
                lr_v,
            )
            if s == 0 or (s + 1) % max(1, args.steps // 15) == 0 or s == args.steps - 1:
                lv.block_until_ready()
                row = {
                    "phase": "B_long", "step": s + 1, "elapsed_s": round(time.perf_counter() - t0, 1),
                    "total_loss": round(float(lv), 4), "state_ce": round(float(met["state_ce"]), 4),
                    "mean_nowrite": round(float(met["mean_nowrite"]), 4), "mean_eta": round(float(met["mean_eta"]), 4),
                    "max_frob": round(float(met["max_frob"]), 2),
                }
                curve.append(row)
                print(f"[RSO-B step={s+1:4d}/{args.steps}] {row}", flush=True)
    else:
        for s in range(args.steps):
            c_mbs, y_mbs, w_mbs = [], [], []
            for _ in range(grad_accum):
                idx = rng.integers(0, n_tr, size=(args.batch_size,))
                c_tb = np.transpose(tr_codes[idx], (1, 0, 2)).astype(np.int32)
                y_tb = np.broadcast_to(tr_y[idx][None, :, :], (target_seq_len, args.batch_size, 2)).copy()
                w_tb = np.zeros((target_seq_len, args.batch_size), dtype=np.float32)
                w_tb[-8:, :] = 1.0 / 8.0
                c_mbs.append(c_tb)
                y_mbs.append(y_tb)
                w_mbs.append(w_tb)
            lr_v = jnp.asarray(args.base_lr * lr_schedule_mult(s, args.steps), dtype=jnp.float32)
            all_params, opt_st, lv, met = _step_running(
                all_params, opt_st,
                jnp.asarray(np.stack(c_mbs, axis=0)),
                jnp.asarray(np.stack(y_mbs, axis=0)),
                jnp.asarray(np.stack(w_mbs, axis=0)),
                lr_v,
            )
            if s == 0 or (s + 1) % max(1, args.steps // 15) == 0 or s == args.steps - 1:
                lv.block_until_ready()
                row = {
                    "step": s + 1, "elapsed_s": round(time.perf_counter() - t0, 1),
                    "total_loss": round(float(lv), 4), "state_ce": round(float(met["state_ce"]), 4),
                    "aux_mimi_ce": round(float(met["aux_mimi_ce"]), 4),
                    "mean_nowrite": round(float(met["mean_nowrite"]), 4), "mean_eta": round(float(met["mean_eta"]), 4),
                    "max_frob": round(float(met["max_frob"]), 2),
                }
                curve.append(row)
                print(f"[RSO step={s+1:4d}/{args.steps}] {row}", flush=True)

    dev_eval = evaluate_rso_split(
        all_params, spec, dv_p, vocab_size=vocab_size, eval_bsz=8,
        matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
    )
    short_tick = 320 if target_seq_len <= 2400 else 384
    dev_short_diag = evaluate_rso_split(
        all_params, spec, dv_p, vocab_size=vocab_size, eval_bsz=8, eval_tick=short_tick,
        matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
    )
    evals: dict[str, Any] = {
        "dev": dev_eval,
        "dev_short_horizon_inside_gqa_rf_diag": {
            "eval_tick": short_tick,
            "overwrite_ab_flip_and_preserve": dev_short_diag["overwrite_ab_flip_and_preserve"],
            "overall_counterfactual_joint_acc": dev_short_diag["overall_counterfactual_joint_acc"],
            "single_stream_a_joint_acc": dev_short_diag["single_stream_a_joint_acc"],
        },
    }
    if args.eval_sealed_test and te_p.is_file():
        evals["sealed_test"] = evaluate_rso_split(
            all_params, spec, te_p, vocab_size=vocab_size, eval_bsz=8,
            matmul_dtype=mm_dtype, use_wy_chunking=args.use_wy_chunking,
        )

    tag = args.tag or f"{args.profile}_{args.backbone}_c{args.surprise_chunk_size}_s{args.seed}"
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "profile": args.profile,
        "backbone": args.backbone,
        "surprise_chunk_size": int(args.surprise_chunk_size),
        "use_wy_chunking": bool(args.use_wy_chunking),
        "shard_tag": args.shard_tag,
        "seed": int(args.seed),
        "trainable_core_params": int(n_core_p),
        "core_streaming_state_bytes_b1": int(st_bytes["total_core_state_bytes"]),
        "steps": int(args.steps),
        "effective_batch_size": int(args.batch_size * grad_accum),
        "base_lr": float(args.base_lr),
        "wall_sec": round(time.perf_counter() - t0, 2),
        "train_curve": curve,
        "frozen_evaluations": evals,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_p = args.out_dir / f"rso_{tag}_receipt.json"
    out_p.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "train_curve"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
