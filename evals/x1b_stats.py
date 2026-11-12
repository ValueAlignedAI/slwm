"""Seed-Aware Hierarchical Cluster Bootstrap & Leave-One-Out Attribution Statistics.

Addresses the peer-review statistical calibration requirement:
  - Episode-pooled bootstrap measures test-world variability conditional on the trained checkpoints.
  - Algorithm-level claims across stochastic training runs require two-level hierarchical resampling
    (resampling training seeds S with replacement at level 1, and evaluation worlds E within each sampled
    seed at level 2) alongside exact paired seed-level mean/SD/min/max summaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def hierarchical_seed_bootstrap_ci(
    seed_episode_matrix: np.ndarray,
    ref_seed_episode_matrix: np.ndarray | None = None,
    *,
    n_boot: int = 10_000,
    rng_seed: int = 20261112,
) -> dict[str, Any]:
    """Compute both two-level seed-aware hierarchical bootstrap CIs and episode-pooled CIs.

    Args:
        seed_episode_matrix: float64 array of shape [num_seeds, num_episodes].
        ref_seed_episode_matrix: optional paired baseline array of shape [num_seeds, num_episodes].
    """
    mat = np.asarray(seed_episode_matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D [num_seeds, num_episodes] array, got shape {mat.shape}")
    if ref_seed_episode_matrix is not None:
        ref = np.asarray(ref_seed_episode_matrix, dtype=np.float64)
        if ref.shape != mat.shape:
            raise ValueError(f"Shape mismatch: {mat.shape} vs {ref.shape}")
        mat = mat - ref

    n_seeds, n_eps = mat.shape
    seed_means = np.mean(mat, axis=1)
    grand_mean = float(np.mean(seed_means))
    between_seed_sd = float(np.std(seed_means, ddof=1)) if n_seeds > 1 else 0.0

    rng = np.random.default_rng(rng_seed)
    # Level 1: sample S seed indices [n_boot, n_seeds]
    s_idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    # Level 2: sample E episode indices within each drawn seed [n_boot, n_seeds, n_eps]
    e_idx = rng.integers(0, n_eps, size=(n_boot, n_seeds, n_eps))
    hier_draws = np.mean(mat[s_idx[:, :, None], e_idx], axis=(1, 2))

    # Episode-pooled comparison (for side-by-side transparency)
    flat = mat.reshape(-1)
    p_idx = rng.integers(0, flat.size, size=(n_boot, flat.size))
    pooled_draws = np.mean(flat[p_idx], axis=1)

    return {
        "mean": grand_mean,
        "seed_means": [float(x) for x in seed_means],
        "between_seed_sd": between_seed_sd,
        "seed_min": float(np.min(seed_means)),
        "seed_max": float(np.max(seed_means)),
        "hierarchical_ci95": [float(np.percentile(hier_draws, 2.5)), float(np.percentile(hier_draws, 97.5))],
        "hierarchical_ci99": [float(np.percentile(hier_draws, 0.5)), float(np.percentile(hier_draws, 99.5))],
        "hierarchical_p_gt_zero": float(np.mean(hier_draws > 0.0)),
        "episode_pooled_ci95": [float(np.percentile(pooled_draws, 2.5)), float(np.percentile(pooled_draws, 97.5))],
        "episode_pooled_ci99": [float(np.percentile(pooled_draws, 0.5)), float(np.percentile(pooled_draws, 99.5))],
        "n_seeds": int(n_seeds),
        "n_episodes_per_seed": int(n_eps),
    }
