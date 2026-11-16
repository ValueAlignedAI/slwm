"""Anti-Collapse Latent Geometry Regularizers: Exact LeJEPA Epps-Pulley SIGReg vs VICReg Variance-Covariance.

Separates two distinct isotropic anti-collapse objectives identified during peer review:
  1. `vicreg_var_cov_loss`: Hinge standard-deviation + off-diagonal covariance penalty (Bardes et al., ICLR 2022;
     used under the historical `sigreg` config key in Waves 1-5).
  2. `epps_pulley_sigreg_loss`: True Sketched Isotropic Gaussian Regularization (SIGReg; Balestriero & LeCun,
     LeJEPA, arXiv:2511.08544), projecting embeddings onto `num_slices` unit random 1D directions and testing
     univariate Gaussianity via the closed-form Epps-Pulley characteristic-function integral against N(0, 1).

Both functions accept an optional `mask` (`[N]` bool/float32) so padded ticks (`mask == 0`) are strictly
excluded from centering, covariance, Participation Ratio, and Epps-Pulley empirical characteristic-function sums.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _normalize_row_mask(n: int, mask: jax.Array | None) -> tuple[jax.Array, jax.Array]:
    if mask is None:
        w = jnp.ones((n,), dtype=jnp.float32)
    else:
        w = (mask.reshape(n).astype(jnp.float32) > 0.5).astype(jnp.float32)
    n_eff = jnp.maximum(jnp.sum(w), 1.0)
    return w, n_eff


def participation_ratio(z_samples: jax.Array, *, mask: jax.Array | None = None) -> jax.Array:
    """Compute normalized covariance Participation Ratio PR in [1/D, 1] over valid (`mask > 0.5`) rows."""
    zf = z_samples.astype(jnp.float32)
    n, d = zf.shape
    w, n_eff = _normalize_row_mask(n, mask)
    mean_z = jnp.sum(zf * w[:, None], axis=0, keepdims=True) / n_eff
    zc = (zf - mean_z) * w[:, None]
    cov = (zc.T @ zc) / jnp.maximum(n_eff - 1.0, 1.0)
    tr_c = jnp.trace(cov)
    tr_c2 = jnp.sum(cov * cov)
    return (tr_c * tr_c) / jnp.maximum(float(d) * tr_c2, 1e-8)


def vicreg_var_cov_loss(
    z_samples: jax.Array,
    cov_weight: float = 4.0,
    *,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """VICReg hinge-variance + off-diagonal Frobenius covariance loss (Bardes et al., 2022)."""
    zf = z_samples.astype(jnp.float32)
    n, d = zf.shape
    w, n_eff = _normalize_row_mask(n, mask)
    mean_z = jnp.sum(zf * w[:, None], axis=0, keepdims=True) / n_eff
    zc = (zf - mean_z) * w[:, None]
    std = jnp.sqrt(jnp.sum(zc * zc, axis=0) / n_eff + 1e-4)
    var_loss = jnp.mean(jnp.maximum(0.0, 1.0 - std))

    cov = (zc.T @ zc) / jnp.maximum(n_eff - 1.0, 1.0)
    off_diag = cov - jnp.diag(jnp.diag(cov))
    cov_loss = jnp.sum(off_diag * off_diag) / float(max(1, d * (d - 1)))

    tr_c = jnp.trace(cov)
    tr_c2 = jnp.sum(cov * cov)
    pr = (tr_c * tr_c) / jnp.maximum(float(d) * tr_c2, 1e-8)
    return var_loss + cov_weight * cov_loss, pr


def epps_pulley_sigreg_loss(
    z_samples: jax.Array,
    rng_key: jax.Array,
    *,
    num_slices: int = 64,
    bandwidth: float = 1.0,
    scale_anchor_weight: float = 0.25,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Exact Sketched Isotropic Gaussian Regularization (SIGReg; Balestriero & LeCun, LeJEPA 2025).

    Projects valid `z_samples[N, D]` rows (`mask > 0.5`, count `N_eff`) onto `num_slices` unit-norm random
    directions `U[D, M]` and evaluates the closed-form Epps-Pulley L2 characteristic-function distance to N(0, 1):
        T_EP(y) = (1/N_eff^2) sum_{i,j in V} exp(-b^2 (y_i - y_j)^2 / 2)
                  - (2 / (N_eff * sqrt(1 + b^2))) sum_{i in V} exp(-b^2 y_i^2 / (2*(1 + b^2)))
                  + 1 / sqrt(1 + 2*b^2).
    """
    zf = z_samples.astype(jnp.float32)
    n, d = zf.shape
    w, n_eff = _normalize_row_mask(n, mask)
    mean_z = jnp.sum(zf * w[:, None], axis=0, keepdims=True) / n_eff
    zc = (zf - mean_z) * w[:, None]

    k_rand, k_axes = jax.random.split(rng_key)
    m_rand = max(1, num_slices // 2)
    m_axes = max(1, num_slices - m_rand)
    raw_dirs = jax.random.normal(k_rand, (d, m_rand), dtype=jnp.float32)
    rand_dirs = raw_dirs * jax.lax.rsqrt(jnp.sum(raw_dirs * raw_dirs, axis=0, keepdims=True) + 1e-8)
    axis_idx = jax.random.randint(k_axes, (m_axes,), 0, d)
    axis_dirs = jax.nn.one_hot(axis_idx, d, dtype=jnp.float32).T
    unit_dirs = jnp.concatenate([rand_dirs, axis_dirs], axis=1)

    proj = zc @ unit_dirs  # [N, M], strictly 0.0 on invalid rows
    b2 = float(bandwidth * bandwidth)

    diff = proj[:, None, :] - proj[None, :, :]  # [N, N, M]
    w_pair = w[:, None] * w[None, :]  # [N, N]
    term_pair = jnp.sum(jnp.exp(-0.5 * b2 * (diff * diff)) * w_pair[:, :, None], axis=(0, 1)) / jnp.maximum(
        n_eff * n_eff, 1.0
    )
    term_cross = (2.0 / jnp.sqrt(1.0 + b2)) * (
        jnp.sum(jnp.exp(-0.5 * b2 * (proj * proj) / (1.0 + b2)) * w[:, None], axis=0) / n_eff
    )
    term_ref = 1.0 / jnp.sqrt(1.0 + 2.0 * b2)
    ep_stat = jnp.maximum(0.0, term_pair - term_cross + term_ref)

    slice_std = jnp.sqrt(jnp.sum((proj * proj) * w[:, None], axis=0) / n_eff + 1e-4)
    scale_pen = jnp.mean((slice_std - 1.0) ** 2)

    loss = n_eff * jnp.mean(ep_stat) + scale_anchor_weight * scale_pen
    pr = participation_ratio(zf, mask=w)
    return loss, pr
