# SLWM-124M Inference Modes — Sprint I3

**Sprint:** I3 — output heads and policy stubs  
**Status:** API and metadata contract only; no trained generation quality claim.

## 1. Canonical Input

All inference modes start from the canonical latent contract:

```python
Z_world: FloatTensor[B,T,D]
mask: BoolTensor[B,T]
```

In the current NumPy smoke wrapper, `NumpySLWMCore.forward(...)` runs:

```text
adapters → latent field → signal processor → diagnostic heads → output heads → policy gate
```

## 2. Supported Modes

| Mode | Purpose | External output? | I3 behavior |
|---|---|---:|---|
| `perception` | process current context | no | return `z_world` and auxiliary metadata |
| `predict` | latent/future prediction | no by default | `LatentPredictionHead` remains diagnostic/internal |
| `reconstruct` | decode hidden/missing signal | no by default | reconstruction/probe outputs are internal-only |
| `commit` | select external behavior | yes, if policy commits | policy returns `commitments`, `suppressed`, gates, no-op probability |
| `explore` | diagnostic probing | no | all probe outputs are tagged `diagnostic-only` |

I3 does not hard-code one meaning of inference. The same processed latent field may be inspected by probes or routed through the policy gate, but probe outputs are not external behavior unless a later trained policy explicitly commits them under a documented protocol.

## 3. Commit Path

Example commit-mode call:

```python
output = model.forward(
    batch,
    policy_goal={"commit_heads": ["text_decoder", "audio_decoder"]},
)
```

Expected structured result:

```python
{
  "z_world": FloatTensor[B,T,D],
  "output_heads": {
    "text": {"proposal": ..., "text_logits": TensorSpec[B,T,V]},
    "audio": {"proposal": ..., "audio_latents": TensorSpec[B,T,A]},
    "visual": {"proposal": ..., "visual_latents": TensorSpec[B,T,Vd]},
    "noop": {"proposal": ...},
  },
  "policy": {
    "commitments": [...],
    "suppressed": [...],
    "diagnostic_only": [...],
    "gates": {...},
    "noop_probability": float,
    "metadata": {...},
  },
}
```

If no external head is requested, the default policy commits no-op rather than decoding every head.

## 4. Explore / Probe Path

Example diagnostic call:

```python
output = model.forward(
    batch,
    output_metadata={"mode": "explore"},
    policy_goal={"mode": "explore", "commit_noop": False},
)
```

In this mode:

- all output proposals are marked `diagnostic-only`;
- `policy["commitments"]` is empty;
- outputs may be inspected as probes but are not committed external behavior.

## 5. No-Op / Wait Behavior

No-op is first-class and uses the stable data-contract ID:

```python
{"modality": "noop", "modality_id": 0, "channel": "none"}
```

The default no-op path protects the research contract: internal predictions, reconstructions, or imagined outputs are not automatically expressed externally.

## 6. Limitations

- I3 does not implement natural-language generation, audio synthesis, visual synthesis, or action execution.
- I3 policy scores are routing metadata only, not calibrated confidence.
- I3 does not evaluate unsupported-output reduction, usefulness, grounding, or hallucination.
- Later training/evaluation sprints must compare fixed-rule, always-text, always-no-op, and learned policy variants before making behavior claims.
