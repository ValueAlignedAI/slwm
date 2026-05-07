# SLWM-124M Policy Commitment — Sprint I3

**Sprint:** I3 — output heads and policy stubs  
**Status:** deterministic proposal/routing API stubs only; no trained policy, no behavior-quality claim.

## 1. Scope

Sprint I3 implements the minimal commitment boundary between processed latent world fields and external outputs:

```text
Z_world[B,T,D]
  → output heads produce proposals
  → policy/commit gate scores/routes proposals
  → selected commitments become external behavior contracts
```

The gate does **not** train or optimize a complex agentic policy. Its purpose is to make no-op, text, audio-latent, visual-latent, and diagnostic/probe paths explicit and testable.

## 2. Proposal Contract

Every output head returns:

```python
{
  "proposal": {
    "head": str,
    "channel": "text" | "audio" | "visual" | "none" | "internal",
    "modality": "text_code" | "audio" | "visual_video" | "noop" | None,
    "modality_id": int | None,
    "latent_shape": [B, T, D],
    "score": float,
    "source_tag": "observed" | "reconstructed" | "predicted" | "inferred" | "imagined" | "unknown" | "unsupported",
    "diagnostic_only": bool,
    "internal_only": bool,
    "committed": False,
    "status": "proposal" | "diagnostic-only",
    "intention": str,
    "metadata": dict,
  },
  "metadata": dict,
  ... optional shape-only output specs ...
}
```

The I3 heads are:

| Head | Channel | Modality ID | Output key | Default role |
|---|---|---:|---|---|
| `TextDecoderHead` | `text` | `text_code=1` | `text_logits` | text/code proposal |
| `AudioDecoderHead` | `audio` | `audio=2` | `audio_latents` | audio-latent proposal |
| `VisualDecoderHead` | `visual` | `visual_video=3` | `visual_latents` | visual/video-latent proposal |
| `NoOpHead` | `none` | `noop=0` | none | observe/wait/no-op proposal |

Internal/probe heads such as `LatentPredictionHead`, `ReconstructionHead`, and `UncertaintyHead` remain diagnostic by default.

## 3. Policy Gate Outputs

`PolicyCommitGate` is currently an alias of the deterministic fixed-rule baseline:

```python
PolicyCommitGate(z_world, proposals, uncertainty=None, goal=None) -> {
  "commitments": list,
  "suppressed": list,
  "diagnostic_only": list,
  "decisions": list,
  "gates": dict[str, float],
  "noop_probability": float,
  "metadata": dict,
}
```

Each proposal is tagged as one of:

- `committed`: selected external behavior;
- `suppressed`: valid proposal not selected for external output;
- `diagnostic-only`: internal/probe output, inspectable but not externally committed.

Default behavior remains conservative: if no external head is requested, the policy commits no-op rather than decoding all heads.

## 4. Fixed-Rule Baseline

`FixedRulePolicyCommitGate` supports:

- default no-op commitment;
- single-head commitment with `goal={"commit_head": "text_decoder"}`;
- multi-head commitment with `goal={"commit_heads": ["text_decoder", "audio_decoder"]}`;
- zero-commitment internal-only routing with `goal={"commit_heads": [], "commit_noop": False}`;
- optional score threshold routing for baseline experiments.

No-op is a valid output, but by default it is not committed alongside external text/audio/visual heads.

## 5. Learned Policy Stub

`LearnedPolicyCommitGateStub` exposes the same API and records deterministic pseudo-logits from proposal scores. It has no trainable parameters and reports:

```python
metadata["learned_policy_stub"] == True
metadata["training_enabled"] == False
```

This preserves the future learned-policy interface without claiming learned behavior in I3.

## 6. Acceptance Status

I3 acceptance is limited to API behavior:

1. text/audio/visual/no-op proposals are generated;
2. the policy can route no-op, single-head, multi-head, and zero-head cases;
3. diagnostic/probe outputs are marked internal-only;
4. unselected heads are suppressed, preventing default decode-all behavior;
5. commit metadata is returned in structured form.

No hallucination, grounding, usefulness, or policy-quality claim is made from these stubs.
