"""Seed-Aware Hierarchical Cluster Bootstrap & Leave-One-Out Attribution Statistics.

Addresses the peer-review statistical calibration requirement:
  - Episode-pooled bootstrap measures test-world variability conditional on the trained checkpoints.
  - Algorithm-level claims across stochastic training runs require two-level hierarchical resampling
    (resampling training seeds S with replacement at level 1, and evaluation worlds E — or recording-session
    scene clusters C when multiple clips share a physical scene — within each sampled seed at level 2)
    alongside exact paired seed-level mean/SD/min/max summaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def hierarchical_seed_bootstrap_ci(
    seed_episode_matrix: np.ndarray,
    ref_seed_episode_matrix: np.ndarray | None = None,
    *,
    cluster_ids: np.ndarray | list[int] | None = None,
    n_boot: int = 10_000,
    rng_seed: int = 20261112,
) -> dict[str, Any]:
    """Compute both two-level seed-aware hierarchical bootstrap CIs and episode-pooled CIs.

    Args:
        seed_episode_matrix: float64 array of shape [num_seeds, num_episodes].
        ref_seed_episode_matrix: optional paired baseline array of shape [num_seeds, num_episodes].
        cluster_ids: optional 1D array/list of length `num_episodes` grouping episodes into shared
            recording sessions / physical scenes for cluster-aware Level-2 resampling.
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
    s_idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))

    if cluster_ids is not None:
        c_arr = np.asarray(cluster_ids)
        if c_arr.shape != (n_eps,):
            raise ValueError(f"Expected cluster_ids shape ({n_eps},), got {c_arr.shape}")
        uniq_c, inv = np.unique(c_arr, return_inverse=True)
        n_clusters = int(uniq_c.shape[0])
        cluster_sums = np.zeros((n_seeds, n_clusters), dtype=np.float64)
        cluster_counts = np.bincount(inv, minlength=n_clusters).astype(np.float64)
        for si in range(n_seeds):
            cluster_sums[si] = np.bincount(inv, weights=mat[si], minlength=n_clusters)
        c_idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
        drawn_sums = np.sum(cluster_sums[s_idx[:, :, None], c_idx[:, None, :]], axis=2)  # [n_boot, n_seeds]
        drawn_cnts = np.maximum(np.sum(cluster_counts[c_idx], axis=1), 1.0)[:, None]  # [n_boot, 1]
        hier_draws = np.mean(drawn_sums / drawn_cnts, axis=1)
    else:
        n_clusters = int(n_eps)
        e_idx = rng.integers(0, n_eps, size=(n_boot, n_eps))
        hier_draws = np.mean(mat[s_idx[:, :, None], e_idx[:, None, :]], axis=(1, 2))

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
        "n_level2_clusters": int(n_clusters),
        "scene_clustered": bool(cluster_ids is not None),
    }


def _world_pair_mann_whitney_matrix(corr_worlds: np.ndarray, clean_worlds: np.ndarray) -> np.ndarray:
    """Exact tie-consistent world-by-world Mann-Whitney comparison matrix `W[u, v]` (`[M, M]`).

    For any paired-world bootstrap multiplicity distribution `p` (`p_w = count(w) / M`), the exact
    tie-consistent Mann-Whitney AUROC between the resampled corrupted tick multiset and the jointly
    resampled clean tick multiset equals the quadratic form `p^T W p` without approximation.
    """
    corr = np.asarray(corr_worlds, dtype=np.float64)
    clean = np.asarray(clean_worlds, dtype=np.float64)
    if corr.ndim == 1:
        corr = corr[:, None]
    if clean.ndim == 1:
        clean = clean[:, None]
    if corr.shape != clean.shape or corr.ndim != 2:
        raise ValueError(f"Expected matching 2D [n_worlds, n_ticks] score arrays, got {corr.shape} vs {clean.shape}")
    diff = corr[:, :, None, None] - clean[None, None, :, :]
    return np.mean((diff > 0.0).astype(np.float64) + 0.5 * (diff == 0.0).astype(np.float64), axis=(1, 3))


def hierarchical_paired_auroc_bootstrap_ci(
    seed_corr_scores: np.ndarray,
    seed_clean_scores: np.ndarray,
    ref_seed_corr_scores: np.ndarray | None = None,
    ref_seed_clean_scores: np.ndarray | None = None,
    *,
    n_boot: int = 10_000,
    rng_seed: int = 20261112,
) -> dict[str, Any]:
    """Two-level hierarchical bootstrap recomputing exact Mann-Whitney AUROC under joint paired-world resampling.

    Resamples training seeds `S` with replacement at Level 1 and paired evaluation worlds `w_1..w_M`
    simultaneously for BOTH corrupted and clean score pools at Level 2, capturing clean-pool and
    corrupted-pool uncertainty jointly.
    """
    corr_a = np.asarray(seed_corr_scores, dtype=np.float64)
    clean_a = np.asarray(seed_clean_scores, dtype=np.float64)
    if corr_a.ndim == 3:
        corr_a = corr_a[:, None, :, :]
    if clean_a.ndim == 2:
        clean_a = clean_a[:, :, None]
    if corr_a.ndim != 4 or clean_a.ndim != 3:
        raise ValueError(f"Unexpected AUROC score shapes: corr={corr_a.shape}, clean={clean_a.shape}")
    n_seeds, n_kinds, n_worlds, n_ticks = corr_a.shape
    if clean_a.shape != (n_seeds, n_worlds, n_ticks):
        raise ValueError(f"Clean score shape {clean_a.shape} does not match {(n_seeds, n_worlds, n_ticks)}")

    w_mat_a = np.zeros((n_seeds, n_worlds, n_worlds), dtype=np.float64)
    for si in range(n_seeds):
        w_mat_a[si] = np.mean(
            [_world_pair_mann_whitney_matrix(corr_a[si, ki], clean_a[si]) for ki in range(n_kinds)],
            axis=0,
        )

    if (ref_seed_corr_scores is None) != (ref_seed_clean_scores is None):
        raise ValueError("Provide both ref_seed_corr_scores and ref_seed_clean_scores, or neither.")

    if ref_seed_corr_scores is not None and ref_seed_clean_scores is not None:
        corr_b = np.asarray(ref_seed_corr_scores, dtype=np.float64)
        clean_b = np.asarray(ref_seed_clean_scores, dtype=np.float64)
        if corr_b.ndim == 3:
            corr_b = corr_b[:, None, :, :]
        if clean_b.ndim == 2:
            clean_b = clean_b[:, :, None]
        if corr_b.shape != corr_a.shape or clean_b.shape != clean_a.shape:
            raise ValueError("Reference AUROC score shapes must match primary cohort score shapes.")
        w_mat_b = np.zeros((n_seeds, n_worlds, n_worlds), dtype=np.float64)
        for si in range(n_seeds):
            w_mat_b[si] = np.mean(
                [_world_pair_mann_whitney_matrix(corr_b[si, ki], clean_b[si]) for ki in range(n_kinds)],
                axis=0,
            )
        target_w = w_mat_a - w_mat_b
    else:
        target_w = w_mat_a

    # Exact unresampled seed-level headline AUROC (or delta-AUROC) equals mean over all (u, v) pairs
    seed_means = np.mean(target_w, axis=(1, 2))
    grand_mean = float(np.mean(seed_means))
    between_seed_sd = float(np.std(seed_means, ddof=1)) if n_seeds > 1 else 0.0

    rng = np.random.default_rng(rng_seed)
    s_idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    w_idx = rng.integers(0, n_worlds, size=(n_boot, n_worlds))
    eye_w = np.eye(n_worlds, dtype=np.float64)
    p_weights = np.mean(eye_w[w_idx], axis=1)  # [n_boot, n_worlds] shared across resampled seeds
    drawn_w = target_w[s_idx]  # [n_boot, n_seeds, n_worlds, n_worlds]
    hier_draws = np.mean(np.einsum("bu,bsuv,bv->bs", p_weights, drawn_w, p_weights), axis=1)

    # Pooled paired-world draws across seeds (conditioning on trained seeds, varying worlds jointly)
    w_pool_idx = rng.integers(0, n_worlds, size=(n_boot, n_worlds))
    p_pool = np.mean(eye_w[w_pool_idx], axis=1)  # [n_boot, n_worlds]
    mean_w_across_seeds = np.mean(target_w, axis=0)  # [n_worlds, n_worlds]
    pooled_draws = np.einsum("bu,uv,bv->b", p_pool, mean_w_across_seeds, p_pool)

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
        "n_episodes_per_seed": int(n_worlds),
        "n_level2_clusters": int(n_worlds),
        "scene_clustered": False,
        "paired_world_auroc_recomputed": True,
    }
