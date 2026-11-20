"""Pre-Scale-Up (`L=24`, `703M/1.072B` depth) Long-Horizon & `dt=0` Perturbation Stability Tests.

Verifies the peer-reviewer pre-1B stability advisory before launching `L=24` (`scale_700m` / `scale_1b`):
  1. `L=24` CompleteP residual scaling (`residual_base_gain * L_base / L`) keeps activations, memory
     Frobenius norms, and gradients strictly finite and bounded over `T=2048` ticks.
  2. Adversarial `dt_s == 0.0` burst streams (simultaneous multimodal packets / zero-elapsed-time
     byte bursts where `exp(-dt / tau) == 1.0` identically across all 4 timescale banks) remain
     strictly contractive (`||M||_F` bounded, zero NaNs/Infs, finite gradients) under BOTH `C=1`
     and `C=16` Chunk-Anchor Surprise WY.
  3. Perturbing `dt_s` at future ticks (`t >= t_split`) produces strictly `0.0` change in `z[:t_split]`
     at `L=24` (`C=16` WY), proving zero future leakage at full 1B depth.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from models.v3.core import init_core_state, init_slwm_v3_params
from models.v3.fast_core import scan_core_fast
from models.v3.spec import SLWMv3Spec


def _make_l24_proxy_spec(*, surprise_chunk_size: int = 1) -> SLWMv3Spec:
    return SLWMv3Spec(
        name=f"l24_proxy_c{surprise_chunk_size}",
        backbone_type="slwm_v3_routed",
        d_model=64,
        d_base=64,
        n_layers=24,
        n_base_layers=12,
        residual_base_gain=0.25,
        ffn_mult=2.0,
        window_size=32,
        n_attn_heads=4,
        n_kv_heads=2,
        attn_head_dim=16,
        n_banks=4,
        n_mem_heads=1,
        mem_key_dim=16,
        mem_val_dim=16,
        mem_base_dim=64,
        ssm_state_dim=4,
        surprise_chunk_size=surprise_chunk_size,
    )


def _make_burst_batch(spec: SLWMv3Spec, *, seq_len: int, batch_size: int, dt_zero: bool, seed: int) -> dict[str, jax.Array]:
    rng = np.random.default_rng(seed)
    dt = np.zeros((seq_len, batch_size), dtype=np.float32) if dt_zero else rng.uniform(0.0, 0.25, size=(seq_len, batch_size)).astype(np.float32)
    rst = np.zeros((seq_len, batch_size), dtype=bool)
    rst[0, :] = True
    return {
        "modality_id": jnp.asarray(rng.integers(0, 4, size=(seq_len, batch_size), dtype=np.int32)),
        "dt_s": jnp.asarray(dt),
        "valid_mask": jnp.ones((seq_len, batch_size), dtype=jnp.float32),
        "reliability": jnp.ones((seq_len, batch_size), dtype=jnp.float32),
        "reset_mask": jnp.asarray(rst),
        "spatial_pos": jnp.zeros((seq_len, batch_size), dtype=jnp.int32),
        "text_byte": jnp.asarray(rng.integers(32, 126, size=(seq_len, batch_size), dtype=np.int32)),
        "video_feat": jnp.asarray(rng.standard_normal((seq_len, batch_size, spec.video_patch_dim)).astype(np.float32) * 0.5),
        "audio_codes": jnp.asarray(rng.integers(0, spec.audio_vocab_size, size=(seq_len, batch_size, spec.audio_codebooks), dtype=np.int32)),
        "proprio_feat": jnp.asarray(rng.standard_normal((seq_len, batch_size, spec.proprio_dim)).astype(np.float32) * 0.5),
        "action_feat": jnp.asarray(rng.standard_normal((seq_len, batch_size, spec.action_dim)).astype(np.float32) * 0.5),
        "has_true_action": jnp.ones((seq_len, batch_size), dtype=jnp.float32),
        "probe_feat": jnp.zeros((seq_len, batch_size, spec.probe_dim), dtype=jnp.float32),
    }


def test_l24_t2048_and_dt_zero_stability() -> None:
    """Verify `L=24` stability over `T=2048` and `dt_s == 0.0` burst perturbation for `C=1` and `C=16` WY."""
    for sc, use_wy in ((1, False), (16, True)):
        spec = _make_l24_proxy_spec(surprise_chunk_size=sc)
        params = init_slwm_v3_params(jax.random.PRNGKey(2400 + sc), spec)
        st0 = init_core_state(spec, 1)

        # 1) Long-horizon T=2048 mixed-dt stability
        batch_long = _make_burst_batch(spec, seq_len=2048, batch_size=1, dt_zero=False, seed=11 + sc)
        z_long, _, diags_long = scan_core_fast(
            params, st0, batch_long, spec, remat_layers=True, matmul_dtype=jnp.float32, assume_all_valid=True, use_wy_chunking=use_wy
        )
        assert bool(jnp.all(jnp.isfinite(z_long))), f"Non-finite z_long at L=24 C={sc}"
        max_z = float(jnp.max(jnp.abs(z_long)))
        max_m = float(jnp.max(diags_long["max_mem_frob_norm"]))
        assert max_z < 50.0, f"Unbounded activation at L=24 T=2048 C={sc}: {max_z}"
        assert np.isfinite(max_m) and max_m < 200.0, f"Unbounded memory norm at L=24 T=2048 C={sc}: {max_m}"

        # 2) Adversarial dt=0 burst forward + backward gradient finiteness
        batch_dt0 = _make_burst_batch(spec, seq_len=256, batch_size=2, dt_zero=True, seed=22 + sc)
        st0_b2 = init_core_state(spec, 2)

        def _loss_fn(p):
            z, _, dg = scan_core_fast(
                p, st0_b2, batch_dt0, spec, remat_layers=True, matmul_dtype=jnp.float32, assume_all_valid=True, use_wy_chunking=use_wy
            )
            return jnp.mean(z * z), jnp.max(dg["max_mem_frob_norm"])

        (lv, m_dt0), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params)
        assert np.isfinite(float(lv)) and np.isfinite(float(m_dt0)) and float(m_dt0) < 200.0
        g_norm = float(jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(grads))))
        assert np.isfinite(g_norm) and g_norm > 0.0, f"Bad grad norm under dt=0 at L=24 C={sc}: {g_norm}"

        # 3) Strict causal invariance under future dt=0 / payload perturbation at L=24
        t_split = 83
        batch_pert = {k: v for k, v in batch_dt0.items()}
        batch_pert["dt_s"] = batch_dt0["dt_s"].at[t_split:].set(5.0)
        batch_pert["text_byte"] = batch_dt0["text_byte"].at[t_split:].set(90)
        z_a, _, _ = scan_core_fast(params, st0_b2, batch_dt0, spec, remat_layers=False, matmul_dtype=jnp.float32, assume_all_valid=True, use_wy_chunking=use_wy)
        z_b, _, _ = scan_core_fast(params, st0_b2, batch_pert, spec, remat_layers=False, matmul_dtype=jnp.float32, assume_all_valid=True, use_wy_chunking=use_wy)
        leak = float(jnp.max(jnp.abs(z_a[:t_split] - z_b[:t_split])))
        assert leak == 0.0, f"Future causal leak at L=24 C={sc}: {leak}"


if __name__ == "__main__":
    test_l24_t2048_and_dt_zero_stability()
    print("L24_DEPTH_STABILITY_AND_DT0_TESTS_OK")
