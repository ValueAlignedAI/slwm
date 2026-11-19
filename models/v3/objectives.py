"""SLWM-v3 Wave A Single Coherent Predictive Objective & Strictly Decoupled Downstream Heads.

Enforces two core SLWM-v3 scientific contracts (`slwm_v3_evolution_proposal.md` & `AGENTS.md §12 Gate 3`):
  1. **Single Coherent Multi-Horizon Frozen-Target Predictive Objective:**
     Replaces Study X3's 17-loss soup + moving EMA validation probe with ONE masked multi-horizon
     loss predicting fixed, train-normalized future packet features (`target_feats[t + h]`).
     Zero EMA moving targets in the default Wave A objective; validation loss compares every
     architecture and seed on the exact same frozen target ruler.
  2. **Strictly Decoupled Diagnostic / Downstream Readout Heads (`stop_gradient` + Zero Core Update):**
     Downstream readout probes (overwrite fact classifiers, `xArm6` block XY localization, `HoloAssist`
     4-way choice, future Wave B/C heads) ALWAYS apply `jax.lax.stop_gradient(Z_t)` and own separate
     parameter trees so training or tuning a readout head produces **bit-exact zero gradient and zero
     parameter change** on the shared world core.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from models.v3.core import predict_future_latents, scan_core
from models.v3.memory import init_mup_weight
from models.v3.spec import SLWMv3Spec

Array = jax.Array
PyTree = dict[str, Any]


def masked_multi_horizon_predictive_loss(
    core_params: PyTree,
    init_state: list[PyTree],
    packets_seq: PyTree,
    target_feats_seq: Array,
    target_valid_seq: Array,
    spec: SLWMv3Spec,
    *,
    remat_step: bool = False,
    remat_chunk_size: int = 64,
    return_z_seq: bool = False,
) -> tuple[Array, dict[str, Any]]:
    """Compute the single coherent Wave A masked multi-horizon frozen-target predictive loss.

    Inputs:
      - `packets_seq`:      dict of `[T, B, ...]` causal input packets
      - `target_feats_seq`: `float32[T, B, target_feat_dim]` fixed train-normalized target features
      - `target_valid_seq`: `float32[T, B]` in `{0.0, 1.0}` marking valid supervised future targets
    Causal alignment:
      For each horizon `h` in `spec.pred_horizons`, prediction made at time `t` (`preds[t, :, hi]`)
      is supervised ONLY against `target_feats_seq[t + h]` (requiring BOTH step `t` valid AND
      step `t + h` valid, strictly `t + h < T`). Action conditioning uses ONLY action at step `t`
      when `has_true_action[t] == 1`.
    """
    z_seq, final_state, seq_diags = scan_core(
        core_params,
        init_state,
        packets_seq,
        spec,
        remat_step=remat_step,
        remat_chunk_size=remat_chunk_size,
    )
    t_len, b_sz, _ = z_seq.shape
    action_seq = packets_seq.get("action_feat", jnp.zeros((t_len, b_sz, spec.action_dim), dtype=jnp.float32))
    has_act_seq = packets_seq.get("has_true_action", jnp.zeros((t_len, b_sz), dtype=jnp.float32))
    src_valid = packets_seq["valid_mask"].astype(jnp.float32)  # [T, B]
    if "reset_mask" in packets_seq:
        ep_id = jnp.cumsum(packets_seq["reset_mask"].astype(jnp.int32), axis=0)  # [T, B]
    else:
        ep_id = jnp.zeros((t_len, b_sz), dtype=jnp.int32)

    preds = predict_future_latents(core_params, z_seq, action_seq, has_act_seq, spec)  # [T, B, H, F]

    horizon_losses = []
    per_h_metrics: dict[str, Array] = {}
    for hi, h in enumerate(spec.pred_horizons):
        if h >= t_len:
            h_loss = jnp.asarray(0.0, dtype=jnp.float32)
        else:
            p_h = preds[:-h, :, hi, :]  # [T-h, B, F]
            y_h = jax.lax.stop_gradient(target_feats_seq[h:, :, :].astype(jnp.float32))
            same_ep = (ep_id[:-h, :] == ep_id[h:, :]).astype(jnp.float32)
            m_h = src_valid[:-h, :] * target_valid_seq[h:, :].astype(jnp.float32) * same_ep  # [T-h, B]
            # Smooth L1 / Huber loss (robust to rare sensory bursts while preserving exact gradients)
            diff = p_h - y_h
            abs_d = jnp.abs(diff)
            huber = jnp.where(abs_d <= 1.0, 0.5 * diff * diff, abs_d - 0.5)
            per_pos = jnp.mean(huber, axis=-1)  # [T-h, B]
            h_loss = jnp.sum(per_pos * m_h) / jnp.maximum(jnp.sum(m_h), 1.0)
        horizon_losses.append(h_loss)
        per_h_metrics[f"pred_loss_h{h}"] = h_loss

    total_loss = jnp.mean(jnp.stack(horizon_losses, axis=0))
    metrics: dict[str, Any] = {
        "total_loss": total_loss,
        **per_h_metrics,
        "z_rms": jnp.sqrt(jnp.mean(z_seq * z_seq) + 1e-8),
        "mean_nowrite": jnp.mean(seq_diags["mean_nowrite"]),
        "mean_eff_write_eta": jnp.mean(seq_diags["mean_eff_write_eta"]),
        "mean_eff_decay": jnp.mean(seq_diags["mean_eff_decay"]),
        "mean_state_delta_norm": jnp.mean(seq_diags["mean_state_delta_norm"]),
        "mean_mem_out_rms": jnp.mean(seq_diags["mean_mem_out_rms"]),
        "mean_prior_error": jnp.mean(seq_diags["mean_prior_error"]),
        "mean_key_norm": jnp.mean(seq_diags["mean_key_norm"]),
    }
    if return_z_seq:
        metrics["z_seq"] = z_seq
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Strictly Decoupled Readout / Probe Head (Zero Gradient to World Core!)
# ---------------------------------------------------------------------------


def init_decoupled_probe_params(key: Array, d_model: int, out_dim: int, hidden_dim: int = 128) -> PyTree:
    """Initialize a standalone decoupled probe head operating strictly on `stop_gradient(Z_t)`."""
    k1, k2 = jax.random.split(key, 2)
    return {
        "probe_w1": init_mup_weight(k1, (d_model, hidden_dim), role="hidden", width_mult=1.0, base_std=0.02),
        "probe_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "probe_w2": init_mup_weight(k2, (hidden_dim, out_dim), role="readout", width_mult=1.0, base_std=0.02, zero_init=True),
        "probe_b2": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def apply_decoupled_probe(probe_params: PyTree, z_t: Array) -> Array:
    """Apply decoupled downstream head with mandatory `jax.lax.stop_gradient(z_t)` isolation."""
    z_frozen = jax.lax.stop_gradient(z_t.astype(jnp.float32))
    h = jax.nn.silu(jnp.matmul(z_frozen, probe_params["probe_w1"]) + probe_params["probe_b1"])
    return jnp.matmul(h, probe_params["probe_w2"]) + probe_params["probe_b2"]


def decoupled_probe_mse_loss(
    core_params: PyTree,
    probe_params: PyTree,
    init_state: list[PyTree],
    packets_seq: PyTree,
    targets_final: Array,
    spec: SLWMv3Spec,
) -> Array:
    """Example downstream probe training loss guaranteed to produce ZERO gradient on `core_params`."""
    z_seq, _, _ = scan_core(core_params, init_state, packets_seq, spec)
    preds = apply_decoupled_probe(probe_params, z_seq[-1])
    return jnp.mean((preds - targets_final.astype(jnp.float32)) ** 2)
