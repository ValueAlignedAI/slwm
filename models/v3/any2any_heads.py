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


def init_any2any_heads_params(
    key: Array,
    spec: SLWMv3Spec,
    *,
    n_pred_audio_codebooks: int = 8,
) -> PyTree:
    """Initialize all 4 unified any-to-any readout heads under exact base-shape muP (`zero_init=True`)."""
    keys = jax.random.split(key, 8)
    d = spec.d_model
    m_d = spec.width_mult
    k_aud = int(n_pred_audio_codebooks)

    return {
        "head_text": {
            "norm_w": jnp.ones((d,), dtype=jnp.float32),
            "w": init_mup_weight(keys[0], (d, spec.text_vocab_size), role="readout", width_mult=m_d, zero_init=True),
            "bias": jnp.zeros((spec.text_vocab_size,), dtype=jnp.float32),
        },
        "head_audio": {
            "norm_w": jnp.ones((d,), dtype=jnp.float32),
            "w": init_mup_weight(
                keys[1], (d, k_aud * spec.audio_vocab_size), role="readout", width_mult=m_d, zero_init=True
            ),
            "bias": jnp.zeros((k_aud, spec.audio_vocab_size), dtype=jnp.float32),
        },
        "head_video": {
            "norm_w": jnp.ones((d,), dtype=jnp.float32),
            "w": init_mup_weight(keys[2], (d, spec.video_patch_dim), role="readout", width_mult=m_d, zero_init=True),
            "bias": jnp.zeros((spec.video_patch_dim,), dtype=jnp.float32),
        },
        "head_action": {
            "norm_w": jnp.ones((d,), dtype=jnp.float32),
            "w": init_mup_weight(
                keys[3], (d, spec.action_dim + spec.proprio_dim), role="readout", width_mult=m_d, zero_init=True
            ),
            "bias": jnp.zeros((spec.action_dim + spec.proprio_dim,), dtype=jnp.float32),
        },
    }


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
    return {
        "text_logits": _mm(rms_norm(z_seq, ht["norm_w"]), ht["w"], matmul_dtype) + ht["bias"],
        "audio_logits": _mm(rms_norm(z_seq, ha["norm_w"]), ha["w"], matmul_dtype).reshape(
            t_len, b_sz, k_aud, spec.audio_vocab_size
        )
        + ha["bias"][None, None, :, :],
        "video_pred": _mm(rms_norm(z_seq, hv["norm_w"]), hv["w"], matmul_dtype) + hv["bias"],
        "action_pred": _mm(rms_norm(z_seq, hact["norm_w"]), hact["w"], matmul_dtype) + hact["bias"],
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
    targ_vld = batch.get("target_valid_mask", batch["valid_mask"]).astype(jnp.float32)[1:] * same_ep  # [T-1, B]

    z_src = z_seq[:-1]  # [T-1, B, D]
    next_mod = batch["modality_id"][1:].astype(jnp.int32)  # [T-1, B]

    # 1. Text / Code Next-Byte Cross-Entropy (modality_id == 2)
    m_txt = targ_vld * (next_mod == 2).astype(jnp.float32)
    ht = heads_params["head_text"]
    txt_logits = _mm(rms_norm(z_src, ht["norm_w"]), ht["w"], matmul_dtype) + ht["bias"]
    txt_targets = jnp.clip(batch["text_byte"][1:].astype(jnp.int32), 0, spec.text_vocab_size - 1)
    txt_ce_per = optax.softmax_cross_entropy_with_integer_labels(txt_logits, txt_targets)
    txt_cnt = jnp.sum(m_txt)
    txt_loss = jnp.sum(txt_ce_per * m_txt) / jnp.maximum(txt_cnt, 1.0)
    txt_has = (txt_cnt > 0.5).astype(jnp.float32)

    # 2. Audio Mimi RVQ-0..K-1 Cross-Entropy (modality_id == 1)
    k_aud = int(n_pred_audio_codebooks)
    m_aud = targ_vld * (next_mod == 1).astype(jnp.float32)
    ha_p = heads_params["head_audio"]
    aud_logits = _mm(rms_norm(z_src, ha_p["norm_w"]), ha_p["w"], matmul_dtype).reshape(
        t_len - 1, b_sz, k_aud, spec.audio_vocab_size
    ) + ha_p["bias"][None, None, :, :]
    aud_targets = jnp.clip(batch["audio_codes"][1:, :, :k_aud].astype(jnp.int32), 0, spec.audio_vocab_size - 1)
    aud_ce_per = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(aud_logits, aud_targets), axis=-1)
    aud_cnt = jnp.sum(m_aud)
    aud_loss = jnp.sum(aud_ce_per * m_aud) / jnp.maximum(aud_cnt, 1.0)
    aud_has = (aud_cnt > 0.5).astype(jnp.float32)

    # 3. Video Wan 2.1 Latent Patch Huber Loss (modality_id == 0)
    m_vid = targ_vld * (next_mod == 0).astype(jnp.float32)
    hv = heads_params["head_video"]
    vid_pred = _mm(rms_norm(z_src, hv["norm_w"]), hv["w"], matmul_dtype) + hv["bias"]
    vid_targ = jax.lax.stop_gradient(batch["video_feat"][1:].astype(jnp.float32))
    v_diff = vid_pred - vid_targ
    v_abs = jnp.abs(v_diff)
    v_huber = jnp.mean(jnp.where(v_abs <= 1.0, 0.5 * v_diff * v_diff, v_abs - 0.5), axis=-1)
    vid_cnt = jnp.sum(m_vid)
    vid_loss = jnp.sum(v_huber * m_vid) / jnp.maximum(vid_cnt, 1.0)
    vid_has = (vid_cnt > 0.5).astype(jnp.float32)

    # 4. Continuous Action & Proprioception Huber Loss (modality_id == 3, masked by has_true_action)
    m_act = targ_vld * (next_mod == 3).astype(jnp.float32) * (batch["has_true_action"][1:].astype(jnp.float32) > 0.5)
    hact = heads_params["head_action"]
    act_pred = _mm(rms_norm(z_src, hact["norm_w"]), hact["w"], matmul_dtype) + hact["bias"]
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
