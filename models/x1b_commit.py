"""Learned Innovation-Calibrated PolicyCommitGate for SLWM-X1B (AGENTS.md §2.3, §2.4, §8.5).

Commitment Modes:
    0: commit_full            — Commit fused multimodal proposal (clean sensors, supported query).
    1: commit_trusted_proprio — Suppress spoofed/corrupted visual channel; commit trusted proprio/memory plan (defeats G6 wrong-camera attack without paralysis).
    2: probe_cautious         — Issue bounded low-amplitude exploratory probe action under hidden-constraint surprise (defeats G8 hidden barrier trap).
    3: noop_abstain           — Commit NO_OP / abstain on unsupported queries or ungrounded state.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

COMMIT_MODES = (
    "commit_full",
    "commit_trusted_proprio",
    "probe_cautious",
    "noop_abstain",
)


def route_proposals_with_commit_gate(
    commit_logits: jax.Array | np.ndarray,
    action_full: jax.Array | np.ndarray,
    action_trusted: jax.Array | np.ndarray,
    *,
    exploration_mode: bool = False,
    gate_policy: str = "learned_innovation",
    score_threshold: float = 0.55,
) -> dict[str, Any]:
    """Route action/text/diagnostic proposals through the learned PolicyCommitGate.

    Args:
        commit_logits : float32[..., 4] logits over COMMIT_MODES
        action_full   : float32[..., 4] fused multimodal action proposal
        action_trusted: float32[..., 4] visual-suppressed proprio/register action proposal
        exploration_mode: If True, marks diagnostic probes inspectable without external commitment.
        gate_policy   :
            - "learned_innovation": argmax / calibrated probability over learned 4-mode gate
            - "always_commit": always selects Mode 0 (G6/G8 baseline without safety gate)
            - "fixed_threshold": abstains (Mode 3) whenever max prob < score_threshold, else Mode 0

    Returns:
        Dictionary with committed_action, mode_id, mode_probs, abstained_mask, spoof_mitigated_mask,
        probe_mask, and structured status summary.
    """
    logits = jnp.asarray(commit_logits, dtype=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    act_f = jnp.asarray(action_full, dtype=jnp.float32)
    act_t = jnp.asarray(action_trusted, dtype=jnp.float32)

    if gate_policy == "always_commit":
        mode = jnp.zeros(logits.shape[:-1], dtype=jnp.int32)
    elif gate_policy == "fixed_threshold":
        conf = jnp.max(probs[..., :2], axis=-1)
        mode = jnp.where(conf >= score_threshold, 0, 3).astype(jnp.int32)
    elif gate_policy == "learned_innovation":
        mode = jnp.argmax(probs, axis=-1).astype(jnp.int32)
    else:
        raise ValueError(f"Unknown gate_policy {gate_policy!r}")

    # Mode 2 (probe_cautious) issues a orthogonal/damped exploratory probe step to reveal obstacles
    perp = jnp.stack([-act_f[..., 1], act_f[..., 0], act_f[..., 2] * 0.25, act_f[..., 3] * 0.25], axis=-1)
    probe_act = 0.35 * act_f + 0.35 * perp
    noop_act = jnp.zeros_like(act_f)

    m_exp = mode[..., None]
    committed_action = jnp.where(
        m_exp == 0,
        act_f,
        jnp.where(
            m_exp == 1,
            act_t,
            jnp.where(m_exp == 2, probe_act, noop_act),
        ),
    )
    if exploration_mode:
        committed_action = noop_act

    return {
        "committed_action": committed_action,
        "mode_id": mode,
        "mode_probs": probs,
        "abstained": (mode == 3),
        "spoof_mitigated": (mode == 1),
        "probing": (mode == 2),
        "status": "diagnostic-only" if exploration_mode else "committed",
    }
