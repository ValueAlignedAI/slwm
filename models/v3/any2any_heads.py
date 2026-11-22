"""SLWM-v3 Unified Any-to-Any Multimodal Predictive Heads & Frozen-Core PolicyCommitGate.

Operates on top of the shared latent world field `Z_seq: float32[T, B, D]` produced by ANY
SLWM-v3 backbone or matched comparator, providing exact base-shape muP readout heads (`head_*`
prefix -> automatic `"readout"` role, `zero_init=True`, LR mult `1 / m_d`):
  1. `head_text`:   Lossless ByT5 UTF-8 byte logits `[T, B, 264]` (`Text/Code -> Text/Code`,
                    `Audio -> Text`, `Video+Audio -> Text/QA`; directly scores WikiText-103 BPB/PPL,
                    LAMBADA accuracy, HumanEval/MBPP code synthesis, LibriSpeech ASR, Clotho/MSR-VTT/StreamingBench).
  2. `head_audio`:  Kyutai Mimi semantic + coarse acoustic RVQ logits `[T, B, K_rvq, 2048]` (`K_rvq = 8`
                    core codebooks covering semantic RVQ-0 + primary acoustic RVQ-1..7; `Audio -> Audio`,
                    `Video -> Impact/Ambient Audio` on GreatestHits & VGGSound, `Text -> Speech/Audio`).
  3. `head_video`:  Wan 2.1 Causal VAE latent patch predictor `[T, B, 1024]` (`Video -> Video`,
                    `Audio -> Video`, `Video+Action+Text -> Future Rollout`).
  4. `head_action`: Continuous motor action + next proprioception predictor `[T, B, 14 + 16]`
                    (strictly conditioned/masked on `has_true_action == 1` across xArm6, ALOHA, PushT, LIBERO).
  5. `PolicyCommitGate` (Wave B — strictly `stop_gradient(Z_t)`): evidence- and uncertainty-aware
     commitment gate selecting `{SPEAK_TEXT, EMIT_AUDIO, GENERATE_VIDEO, ACT_MOTOR, ABSTAIN_SILENCE}`
     without ever backpropagating into the frozen world core.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import optax

from models.v3.memory import init_mup_weight, rms_norm
from models.v3.spec import SLWMv3Spec

Array = jax.Array
PyTree = dict[str, Any]

COMMIT_ACTIONS = (
    "SPEAK_TEXT",
    "EMIT_AUDIO",
    "GENERATE_VIDEO",
    "ACT_MOTOR",
    "ABSTAIN_SILENCE",
)


def _init_head_block(
    key: Array, d: int, out_shape: tuple[int, ...], m_d: float, *, use_swiglu_mlp: bool = False
) -> PyTree:
    out_flat = 1
    for s in out_shape:
        out_flat *= int(s)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    blk: PyTree = {
        "norm_w": jnp.ones((d,), dtype=jnp.float32),
        "w": init_mup_weight(k1, (d, out_flat), role="readout", width_mult=m_d, zero_init=True),
        "bias": jnp.zeros(out_shape, dtype=jnp.float32),
    }
    if use_swiglu_mlp:
        h_dec = min(2048, max(512, 2 * d))
        blk["w_g"] = init_mup_weight(k2, (d, h_dec), role="hidden", width_mult=m_d, base_std=0.5 / math.sqrt(float(d)))
        blk["w_u"] = init_mup_weight(k3, (d, h_dec), role="hidden", width_mult=m_d, base_std=0.5 / math.sqrt(float(d)))
        blk["w_d"] = init_mup_weight(k4, (h_dec, out_flat), role="readout", width_mult=m_d, zero_init=True)
    return blk


def _apply_head_block(blk: PyTree, z: Array, matmul_dtype: jnp.dtype) -> Array:
    zn = rms_norm(z, blk["norm_w"])
    out = _mm(zn, blk["w"], matmul_dtype)
    if "w_g" in blk:
        hid = jax.nn.silu(_mm(zn, blk["w_g"], matmul_dtype)) * _mm(zn, blk["w_u"], matmul_dtype)
        out = out + _mm(hid, blk["w_d"], matmul_dtype)
    return out


def init_video_spatial_refiner_params(key: Array, patch_dim: int = 1024, bottleneck: int = 256) -> PyTree:
    """Strictly causal intra-frame `3x5` spatial patch refiner (`dt_s == 0` across all 15 patches of one frame group).

    Removes `3x5` Wan 2.1 patch tile seams with ZERO future temporal lookahead (operates strictly inside one frame group).
    Zero-initialized output projection starts strictly at identity.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "norm_w": jnp.ones((patch_dim,), dtype=jnp.float32),
        "w_down": init_mup_weight(k1, (patch_dim, bottleneck), role="input", width_mult=1.0, base_std=0.5 / math.sqrt(float(patch_dim))),
        "conv_k": init_mup_weight(k2, (3, 3, bottleneck, bottleneck), role="hidden", width_mult=1.0, base_std=0.35 / math.sqrt(9.0 * float(bottleneck))),
        "conv_b": jnp.zeros((bottleneck,), dtype=jnp.float32),
        "w_up": init_mup_weight(k3, (bottleneck, patch_dim), role="readout", width_mult=1.0, zero_init=True),
    }


def apply_video_spatial_refiner(ref_params: PyTree, patches_n_15_d: Array, matmul_dtype: jnp.dtype = jnp.bfloat16) -> Array:
    """Refine `[N, 15, 1024]` intra-frame Wan 2.1 patches across their `3x5` spatial grid (zero temporal lookahead)."""
    n_g = patches_n_15_d.shape[0]
    d_p = patches_n_15_d.shape[-1]
    x_norm = rms_norm(patches_n_15_d.astype(jnp.float32), ref_params["norm_w"])
    h = _mm(x_norm, ref_params["w_down"], matmul_dtype).reshape(n_g, 3, 5, -1)  # [N, H=3, W=5, C]
    # 2D spatial 3x3 convolution strictly within each frame group (`SAME` zero padding, NHWC)
    h_conv = jax.lax.conv_general_dilated(
        h.astype(jnp.float32),
        ref_params["conv_k"].astype(jnp.float32),
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    ) + ref_params["conv_b"][None, None, None, :]
    h_act = jax.nn.silu(h_conv).reshape(n_g, 15, -1)
    delta = _mm(h_act, ref_params["w_up"], matmul_dtype).reshape(n_g, 15, d_p)
    return patches_n_15_d + delta


def init_any2any_heads_params(
    key: Array,
    spec: SLWMv3Spec,
    *,
    n_pred_audio_codebooks: int = 8,
    use_swiglu_mlp: bool = False,
) -> PyTree:
    """Initialize all 4 unified any-to-any readout heads under exact base-shape muP (`zero_init=True`)."""
    keys = jax.random.split(key, 8)
    d = spec.d_model
    m_d = spec.width_mult
    k_aud = int(n_pred_audio_codebooks)

    head_audio = (
        _init_head_block(keys[1], d, (k_aud, spec.audio_vocab_size), m_d, use_swiglu_mlp=use_swiglu_mlp)
        if spec.representation_mode == "frozen_codec_edge"
        else _init_head_block(keys[1], d, (spec.audio_cont_dim,), m_d, use_swiglu_mlp=use_swiglu_mlp)
    )

    out: PyTree = {
        "head_text": _init_head_block(keys[0], d, (spec.text_vocab_size,), m_d, use_swiglu_mlp=use_swiglu_mlp),
        "head_audio": head_audio,
        "head_video": _init_head_block(keys[2], d, (spec.video_patch_dim,), m_d, use_swiglu_mlp=use_swiglu_mlp),
        "head_action": _init_head_block(keys[3], d, (spec.action_dim + spec.proprio_dim,), m_d, use_swiglu_mlp=use_swiglu_mlp),
    }
    if use_swiglu_mlp:
        out["video_spatial_refiner"] = init_video_spatial_refiner_params(keys[4], patch_dim=spec.video_patch_dim)
    return out


def _mm(x: Array, w: Array, matmul_dtype: jnp.dtype) -> Array:
    if matmul_dtype == jnp.float32:
        return jnp.matmul(x.astype(jnp.float32), w.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST)
    return jnp.matmul(x.astype(matmul_dtype), w.astype(matmul_dtype)).astype(jnp.float32)


def predict_any2any_heads(
    heads_params: PyTree,
    z_seq: Array,
    spec: SLWMv3Spec,
    *,
    n_pred_audio_codebooks: int = 8,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, Array]:
    """Compute all 4 any-to-any readout predictions (`text_logits`, `audio_logits`, `video_pred`, `action_pred`)."""
    t_len, b_sz, _ = z_seq.shape
    k_aud = int(n_pred_audio_codebooks)
    ht = heads_params["head_text"]
    ha = heads_params["head_audio"]
    hv = heads_params["head_video"]
    hact = heads_params["head_action"]
    aud_out = (
        _apply_head_block(ha, z_seq, matmul_dtype).reshape(t_len, b_sz, k_aud, spec.audio_vocab_size)
        + ha["bias"][None, None, :, :]
        if spec.representation_mode == "frozen_codec_edge"
        else _apply_head_block(ha, z_seq, matmul_dtype) + ha["bias"]
    )
    return {
        "text_logits": _apply_head_block(ht, z_seq, matmul_dtype) + ht["bias"],
        "audio_logits": aud_out,
        "video_pred": _apply_head_block(hv, z_seq, matmul_dtype) + hv["bias"],
        "action_pred": _apply_head_block(hact, z_seq, matmul_dtype) + hact["bias"],
    }


def compute_any2any_multimodal_loss(
    heads_params: PyTree,
    z_seq: Array,
    batch: dict[str, Array],
    spec: SLWMv3Spec,
    *,
    n_pred_audio_codebooks: int = 8,
    matmul_dtype: jnp.dtype = jnp.bfloat16,
    loss_weights: dict[str, float] | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Compute strictly causal next-packet any-to-any loss across ALL present modalities in `batch`.

    For every position `t in 0..T-2` where `t` and `t+1` belong to the same episode (`ep[t] == ep[t+1]`)
    and target tick `t+1` is valid (`target_valid_mask[t+1] == 1`):
      - If `modality_id[t+1] == 2` (text/code): ByT5 next-byte cross-entropy (nat/byte -> exact BPB!).
      - If `modality_id[t+1] == 1` (audio):     Mimi RVQ-0..K-1 next-packet cross-entropy.
      - If `modality_id[t+1] == 0` (video):     Wan 2.1 next-patch Huber loss (1024-dim latent patch).
      - If `modality_id[t+1] == 3` (action):    Continuous action + next-proprio Huber loss (masked by `has_true_action`).
    Works identically under modality dropout (because `target_valid_mask` supervises the full target stream
    even when source inputs had a modality masked out with `valid_mask == 0`, training true cross-modal prediction!).
    """
    lw = loss_weights or {"text": 1.0, "audio": 1.0, "video": 1.0, "action": 1.0}
    t_len, b_sz, _ = z_seq.shape
    rst = batch.get("reset_mask", jnp.zeros((t_len, b_sz), dtype=jnp.bool_)).astype(jnp.bool_)
    ep_id = jnp.cumsum(rst.astype(jnp.int32), axis=0)
    same_ep = (ep_id[:-1] == ep_id[1:]).astype(jnp.float32)  # [T-1, B]
    src_vld = batch["valid_mask"][:-1].astype(jnp.float32)  # Never supervise t -> t+1 from zeroed valid_mask[t] == 0!
    targ_vld = batch.get("target_valid_mask", batch["valid_mask"]).astype(jnp.float32)[1:] * src_vld * same_ep  # [T-1, B]

    z_src = z_seq[:-1]  # [T-1, B, D]
    next_mod = batch["modality_id"][1:].astype(jnp.int32)  # [T-1, B]

    # 1. Text / Code Next-Byte Cross-Entropy (modality_id == 2)
    m_txt = targ_vld * (next_mod == 2).astype(jnp.float32)
    ht = heads_params["head_text"]
    txt_logits = _apply_head_block(ht, z_src, matmul_dtype) + ht["bias"]
    txt_targets = jnp.clip(batch["text_byte"][1:].astype(jnp.int32), 0, spec.text_vocab_size - 1)
    txt_ce_per = optax.softmax_cross_entropy_with_integer_labels(txt_logits, txt_targets)
    txt_cnt = jnp.sum(m_txt)
    txt_loss = jnp.sum(txt_ce_per * m_txt) / jnp.maximum(txt_cnt, 1.0)
    txt_has = (txt_cnt > 0.5).astype(jnp.float32)

    # 2. Audio Loss (modality_id == 1): Mimi RVQ-0..K-1 CE on `frozen_codec_edge` OR 64-bin Log-Mel Huber on `scratch_edge`!
    k_aud = int(n_pred_audio_codebooks)
    m_aud = targ_vld * (next_mod == 1).astype(jnp.float32)
    ha_p = heads_params["head_audio"]
    aud_cnt = jnp.sum(m_aud)
    aud_has = (aud_cnt > 0.5).astype(jnp.float32)

    if spec.representation_mode == "frozen_codec_edge":
        aud_targets = jnp.clip(batch["audio_codes"][1:, :, :k_aud].astype(jnp.int32), 0, spec.audio_vocab_size - 1)

        def _chunked_audio_ce(_: None) -> Array:
            tm1 = t_len - 1
            c_sz = min(64, max(1, tm1))
            rem = tm1 % c_sz
            pad_n = (c_sz - rem) if rem != 0 else 0
            n_ch = (tm1 + pad_n) // c_sz
            if pad_n > 0:
                zp = jnp.concatenate([z_src, jnp.zeros((pad_n, b_sz, z_src.shape[-1]), dtype=z_src.dtype)], axis=0)
                tp = jnp.concatenate([aud_targets, jnp.zeros((pad_n, b_sz, k_aud), dtype=jnp.int32)], axis=0)
                mp = jnp.concatenate([m_aud, jnp.zeros((pad_n, b_sz), dtype=jnp.float32)], axis=0)
            else:
                zp, tp, mp = z_src, aud_targets, m_aud
            zc = zp.reshape(n_ch, c_sz, b_sz, z_src.shape[-1])
            tc = tp.reshape(n_ch, c_sz, b_sz, k_aud)
            mc = mp.reshape(n_ch, c_sz, b_sz)

            @jax.checkpoint
            def _ce_chunk(acc: Array, xs: tuple[Array, Array, Array]) -> tuple[Array, None]:
                z_i, t_i, m_i = xs
                lg = _apply_head_block(ha_p, z_i, matmul_dtype).reshape(
                    c_sz, b_sz, k_aud, spec.audio_vocab_size
                ) + ha_p["bias"][None, None, :, :]
                ce = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(lg, t_i), axis=-1)
                return acc + jnp.sum(ce * m_i), None

            tot_ce, _ = jax.lax.scan(_ce_chunk, jnp.asarray(0.0, dtype=jnp.float32), (zc, tc, mc))
            return tot_ce / jnp.maximum(aud_cnt, 1.0)

        aud_loss = jax.lax.cond(aud_cnt > 0.5, _chunked_audio_ce, lambda _: jnp.asarray(0.0, dtype=jnp.float32), None)
    else:
        aud_pred = _apply_head_block(ha_p, z_src, matmul_dtype) + ha_p["bias"]
        aud_targ_c = jax.lax.stop_gradient(batch["audio_cont"][1:].astype(jnp.float32))
        ad_c = jnp.abs(aud_pred - aud_targ_c)
        a_hub = jnp.mean(jnp.where(ad_c <= 1.0, 0.5 * (aud_pred - aud_targ_c) ** 2, ad_c - 0.5), axis=-1)
        aud_loss = jnp.sum(a_hub * m_aud) / jnp.maximum(aud_cnt, 1.0)

    # 3. Video Wan 2.1 Latent Patch Huber Loss (modality_id == 0) + optional strictly-causal intra-frame 3x5 spatial refiner!
    m_vid = targ_vld * (next_mod == 0).astype(jnp.float32)
    hv = heads_params["head_video"]
    vid_pred = _apply_head_block(hv, z_src, matmul_dtype) + hv["bias"]
    vid_targ = jax.lax.stop_gradient(batch["video_feat"][1:].astype(jnp.float32))
    v_diff = vid_pred - vid_targ
    v_abs = jnp.abs(v_diff)
    v_huber = jnp.mean(jnp.where(v_abs <= 1.0, 0.5 * v_diff * v_diff, v_abs - 0.5), axis=-1)
    vid_cnt = jnp.sum(m_vid)
    vid_loss = jnp.sum(v_huber * m_vid) / jnp.maximum(vid_cnt, 1.0)
    vid_has = (vid_cnt > 0.5).astype(jnp.float32)

    # Strictly causal intra-frame 3x5 spatial refiner loss when `video_spatial_refiner` is present (Wave C!):
    # Finds EVERY complete contiguous 15-patch frame group (`spatial_pos == 0..14` within the same episode,
    # regardless of interleaved audio/action tick alignment!) and refines them jointly with
    # `Huber + 0.15 * spatial_gradient_Huber` (eliminates 3x5 tile seams with ZERO future lookahead!).
    if "video_spatial_refiner" in heads_params:
        sp_next = batch["spatial_pos"][1:].astype(jnp.int32)  # [T-1, B]
        ep_next = ep_id[1:]  # [T-1, B]
        tm1 = t_len - 1
        n_w = tm1 - 14
        k_max = max(1, (tm1 // 15) * b_sz)
        if n_w > 0:
            offs = jnp.arange(15, dtype=jnp.int32)
            sp_win = jnp.stack([sp_next[k : n_w + k] for k in range(15)], axis=-1)  # [n_w, B, 15]
            mv_win = jnp.stack([m_vid[k : n_w + k] for k in range(15)], axis=-1)    # [n_w, B, 15]
            ep_win = jnp.stack([ep_next[k : n_w + k] for k in range(15)], axis=-1)  # [n_w, B, 15]
            is_grp_start = (
                (jnp.min(mv_win, axis=-1) > 0.5)
                & jnp.all(sp_win == offs[None, None, :], axis=-1)
                & jnp.all(ep_win == ep_win[..., :1], axis=-1)
            )  # [n_w, B]
            n_found = jnp.sum(is_grp_start)
            t_starts, b_starts = jnp.nonzero(is_grp_start, size=k_max, fill_value=0)
            fm = (jnp.arange(k_max, dtype=jnp.int32) < n_found).astype(jnp.float32)
            t_idx = t_starts[:, None] + offs[None, :]  # [k_max, 15]
            b_idx = b_starts[:, None]                  # [k_max, 1]
            vp_g = vid_pred[t_idx, b_idx]              # [k_max, 15, 1024]
            vt_g = vid_targ[t_idx, b_idx]              # [k_max, 15, 1024]
            vp_ref = apply_video_spatial_refiner(heads_params["video_spatial_refiner"], vp_g, matmul_dtype=matmul_dtype)
            rd = vp_ref - vt_g
            ra = jnp.abs(rd)
            r_hub = jnp.mean(jnp.where(ra <= 1.0, 0.5 * rd * rd, ra - 0.5), axis=(1, 2))
            # Spatial boundary continuity across the 3x5 grid (matches inter-patch vertical & horizontal differences!)
            pr_35 = vp_ref.reshape(-1, 3, 5, spec.video_patch_dim)
            gt_35 = vt_g.reshape(-1, 3, 5, spec.video_patch_dim)
            dh = (pr_35[:, :, 1:, :] - pr_35[:, :, :-1, :]) - (gt_35[:, :, 1:, :] - gt_35[:, :, :-1, :])
            dv = (pr_35[:, 1:, :, :] - pr_35[:, :-1, :, :]) - (gt_35[:, 1:, :, :] - gt_35[:, :-1, :, :])
            g_loss = 0.5 * (jnp.mean(jnp.abs(dh), axis=(1, 2, 3)) + jnp.mean(jnp.abs(dv), axis=(1, 2, 3)))
            fm_cnt = jnp.sum(fm)
            ref_loss = jnp.sum((r_hub + 0.15 * g_loss) * fm) / jnp.maximum(fm_cnt, 1.0)
            vid_loss = jnp.where(fm_cnt > 0.5, 0.5 * vid_loss + 0.5 * ref_loss, vid_loss)

    # 4. Continuous Action & Proprioception Huber Loss (modality_id == 3, masked by has_true_action)
    m_act = targ_vld * (next_mod == 3).astype(jnp.float32) * (batch["has_true_action"][1:].astype(jnp.float32) > 0.5)
    hact = heads_params["head_action"]
    act_pred = _apply_head_block(hact, z_src, matmul_dtype) + hact["bias"]
    act_targ = jax.lax.stop_gradient(
        jnp.concatenate(
            [batch["action_feat"][1:].astype(jnp.float32), batch["proprio_feat"][1:].astype(jnp.float32)],
            axis=-1,
        )
    )
    a_diff = act_pred - act_targ
    a_abs = jnp.abs(a_diff)
    a_huber = jnp.mean(jnp.where(a_abs <= 1.0, 0.5 * a_diff * a_diff, a_abs - 0.5), axis=-1)
    act_cnt = jnp.sum(m_act)
    act_loss = jnp.sum(a_huber * m_act) / jnp.maximum(act_cnt, 1.0)
    act_has = (act_cnt > 0.5).astype(jnp.float32)

    # Active-modality weighted average (never divides by missing modality counts!)
    w_sum = (
        lw["text"] * txt_has
        + lw["audio"] * aud_has
        + lw["video"] * vid_has
        + lw["action"] * act_has
    )
    total_loss = (
        lw["text"] * txt_loss * txt_has
        + lw["audio"] * aud_loss * aud_has
        + lw["video"] * vid_loss * vid_has
        + lw["action"] * act_loss * act_has
    ) / jnp.maximum(w_sum, 1.0)

    metrics = {
        "total_loss": total_loss,
        "text_ce_nat": txt_loss,
        "text_bpb": txt_loss / jnp.log(2.0),
        "audio_rvq_ce_nat": aud_loss,
        "video_patch_huber": vid_loss,
        "action_proprio_huber": act_loss,
        "n_text_targets": txt_cnt,
        "n_audio_targets": aud_cnt,
        "n_video_targets": vid_cnt,
        "n_action_targets": act_cnt,
    }
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Wave B: Strictly Frozen-Core Evidence-Aware PolicyCommitGate
# ---------------------------------------------------------------------------


def init_policy_commit_gate_params(key: Array, d_model: int, hidden_dim: int = 256) -> PyTree:
    """Initialize standalone Wave B PolicyCommitGate operating strictly on `stop_gradient(Z_t)` with width-invariant epistemic branch."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    n_ev = 5  # [text_entropy, audio_entropy, prior_err, reliability, abstention_prior]
    ev_hid = 32
    n_act = len(COMMIT_ACTIONS)
    return {
        "norm_w": jnp.ones((d_model,), dtype=jnp.float32),
        # Content branch on `RMSNorm(Z_t)`
        "w1": init_mup_weight(k1, (d_model, hidden_dim), role="hidden", width_mult=1.0, base_std=1.0 / math.sqrt(float(d_model))),
        "b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "w2": init_mup_weight(k2, (hidden_dim, n_act), role="readout", width_mult=1.0, zero_init=True),
        # Width-invariant epistemic evidence branch (NEVER diluted O(1/d) as d -> 1920!)
        "ev_skip": init_mup_weight(k3, (n_ev, n_act), role="input", width_mult=1.0, base_std=0.25 / math.sqrt(float(n_ev))),
        "ev_w1": init_mup_weight(k4, (n_ev, ev_hid), role="input", width_mult=1.0, base_std=1.0 / math.sqrt(float(n_ev))),
        "ev_b1": jnp.zeros((ev_hid,), dtype=jnp.float32),
        "ev_w2": init_mup_weight(k5, (ev_hid, n_act), role="input", width_mult=1.0, base_std=0.25 / math.sqrt(float(ev_hid))),
        "b2": jnp.zeros((n_act,), dtype=jnp.float32),
    }


def apply_policy_commit_gate(
    gate_params: PyTree,
    z_t: Array,
    evidence_feats: Array,
) -> tuple[Array, Array]:
    """Select committed output behavior strictly on `stop_gradient(Z_t)` (zero gradient to world core!).

    Args:
      z_t:            `[..., D]` latent world field (wrapped in `jax.lax.stop_gradient`).
      evidence_feats: `[..., 5]` epistemic uncertainty & evidence features.
    Returns:
      logits: `[..., 5]` over `COMMIT_ACTIONS`
      probs:  `[..., 5]` calibrated commitment probabilities
    """
    z_sg = jax.lax.stop_gradient(z_t)
    ev_sg = jax.lax.stop_gradient(evidence_feats.astype(jnp.float32))
    zn = rms_norm(z_sg, gate_params["norm_w"])
    hz = jax.nn.silu(jnp.matmul(zn, gate_params["w1"]) + gate_params["b1"])
    hev = jax.nn.silu(jnp.matmul(ev_sg, gate_params["ev_w1"]) + gate_params["ev_b1"])
    logits = (
        jnp.matmul(hz, gate_params["w2"])
        + jnp.matmul(ev_sg, gate_params["ev_skip"])
        + jnp.matmul(hev, gate_params["ev_w2"])
        + gate_params["b2"]
    )
    return logits, jax.nn.softmax(logits, axis=-1)
