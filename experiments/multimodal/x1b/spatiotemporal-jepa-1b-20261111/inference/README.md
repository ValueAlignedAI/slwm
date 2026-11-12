# SLWM-X1B (`1.008B`) Standalone Inference & Checkpoint Streaming

This directory tracks the cryptographic manifest (`manifest.json`, generated post-verification by [`scripts/eval/x1b_infer.py`](../../../../../scripts/eval/x1b_infer.py)) for the `1,008,454,410`-parameter Wave 4 (`X1B-v4`) world models:

- **`SLWM-X1B` (`slwm_x1b`)**: 3:1 Multi-Head Gated Delta Matrix Recurrence + Ring-Buffer Causal RoPE (`O(1)` streaming state = `14.91 MB` at both `T=64` and `T=1024`), `S=24` spatiotemporal field (`16` visual `64x64` patches + `2` spectral audio + `1` proprio + `1` text + `1` action + `3` persistent working-memory registers), 4-step unrolled Residual-EMA-JEPA + `SIGReg`, decoupled `head_vis_pose` / `head_proprio` / `head_goal` readouts, and Mahalanobis `PolicyCommitGate`.
- **`DenseTransformer-X1B` (`dense_transformer`)**: Matched `1,008,435,114`-parameter (`-0.0019%` delta) all-RoPE causal transformer baseline (`O(T)` KV state = `207.03 MB` at `T=64`, `3,312.54 MB` at `T=1024` per single `batch_size=1` stream — `222.2×` larger than `SLWM-X1B`).
- **`LegacyScalarHybrid-X1B` (`legacy_scalar_hybrid`)**: Matched `1,008,435,114`-parameter (`-0.0019%` delta) G5/G10-style scalar-decay + block-diagonal modality baseline.

## Quickstart Commands

Replay a closed-loop OOD `T=76` mid-episode goal-switch & `1-px` thin-barrier mission (`Goal A` shown at `t=0..3`, extinguished `t=4..35`, overwritten by `Goal B` at `t=36..39 > T_train=32`, extinguished `t=40..75`) using SHA-256-verified weights streamed from Fedora (`db@192.168.1.200`) or loaded from a local `.npz` file:

```bash
# Stream portable_int8.npz from Fedora dual-HDD archive & verify SHA-256 against receipt.json:
.venv/bin/python scripts/eval/x1b_infer.py \
  --tag x1b-flagship-slwm-1b-s2001 \
  --mission ood_goal_switch_barrier_mpc

# Or run offline from a local portable_int8.npz / weights_bf16.npz checkpoint:
.venv/bin/python scripts/eval/x1b_infer.py \
  --tag x1b-flagship-slwm-1b-s2001 \
  --checkpoint /path/to/portable_int8.npz \
  --mission thin_barrier_mpc

# Regenerate the full 1.008B head-to-head replay pack, manifest.json, and 64x64 filmstrips:
.venv/bin/python scripts/eval/x1b_infer.py --build-study-pack
```
