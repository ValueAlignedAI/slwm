# SLWM Architecture — Sprint I2/I3

**Sprint:** I2 — SLWM core processor and adapters; I3 — output heads and policy stubs  
**Status:** tiny NumPy forward/backward implementation for adapters, shared latent field, processor blocks, latent prediction head, uncertainty/source head, plus I3 shape-only output-head proposal and policy/commit APIs.  
**Out of scope:** trained policy behavior, exploration dashboards, datasets/preprocessing, large training runs, and capability claims.

## 1. Canonical Contract

All Sprint I2 modules preserve the project-wide latent tensor contract:

```python
Z: FloatTensor[B, T, D]
mask: BoolTensor[B, T]
```

The I2 tiny smoke config uses `T=12`, `D=8`; the 124M-style config records the intended GPT-2-scale shape (`T=1024`, `D=768`) without running a large model.

## 2. Adapter Modules

Implemented adapters:

- `TextSignalAdapter`: integer text/code IDs at the edge codec → `Z[B,T,D]` via token + position embeddings.
- `AudioSignalAdapter`: provided audio latents/features `FloatTensor[B,T_audio,A]` → `Z[B,T,D]` via trainable projection + position signal.
- `VisualSignalAdapter`: provided visual/video latents/features `FloatTensor[B,T_visual,V]` → `Z[B,T,D]` via trainable projection + position signal.

Adapters return the data contract packet:

```python
{"z": FloatTensor[B,T,D], "mask": BoolTensor[B,T], "metadata": dict}
```

Metadata includes stable modality IDs from `docs/data_contract.md` and `observed=True`.

## 3. Shared Latent Field

`LatentSignalField.from_adapter_outputs(...)` validates adapter packets and packs real NumPy arrays by concatenating modalities in packet order, then padding/truncating to the fixed context length. The backward path splits `grad_context[B,T,D]` back into packet-shaped gradients for adapter smoke tests.

## 4. Processor Block

`SignalWorldProcessor` stacks `SignalProcessorBlock` instances. Each block preserves shape:

```text
Z[B,T,D]
  → LayerNorm
  → optional local temporal depthwise convolution
  → optional DCT-like spectral mixer
  → optional long-conv depthwise temporal mixer
  → optional gated MLP
  → residual Z'[B,T,D]
```

Every novel block has a config flag for ablation:

| Component | Config flag |
|---|---|
| local temporal mixer | `use_local_temporal_mixer` |
| spectral mixer | `use_spectral_mixer` |
| long convolution | `use_long_conv` |
| gated MLP | `use_gated_mlp` |
| latent prediction head | `use_latent_prediction_head` |
| uncertainty/source head | `use_uncertainty_head` |

## 5. I2 Heads in Scope

I2 implements only the heads required by the sprint gate:

- `LatentPredictionHead`: trainable `D → D` projection with MSE helper for dummy latent-prediction backward tests.
- `UncertaintyHead`: trainable scalar uncertainty and source-tag logits over the controlled source tag set:
  `observed`, `reconstructed`, `predicted`, `inferred`, `imagined`, `unknown`, `unsupported`.

No text/audio/visual external decoder behavior is added in I2 beyond the existing I0 shape stubs.

## 6. I3 Output Heads and Policy Stubs

Sprint I3 extends the shape-contract stubs with a proposal/commit interface:

```text
Z_world[B,T,D]
  → TextDecoderHead / AudioDecoderHead / VisualDecoderHead / NoOpHead
  → proposal dictionaries with modality IDs, source tags, scores, and status
  → PolicyCommitGate
  → committed, suppressed, or diagnostic-only decisions
```

Implemented routing classes:

| Class | Purpose | Trainable parameters |
|---|---|---:|
| `PolicyCommitGate` | default deterministic I3 gate | 0 |
| `FixedRulePolicyCommitGate` | explicit fixed-rule baseline | 0 |
| `LearnedPolicyCommitGateStub` | future learned-policy API placeholder | 0 |

The default policy commits no-op if no external head is requested. It supports single-head, multi-head, and zero-head routing through `goal` metadata. Diagnostic/probe outputs are marked `diagnostic-only` and are not external behavior.

## 7. Parameter Accounting

`NumpySLWMCore.parameter_count_breakdown()` reports exact instantiated counts split by:

- adapters: `text_code`, `audio`, `visual_video`, plus adapter total,
- processor,
- heads: `latent_prediction`, `uncertainty`, plus head total,
- policy: `0` for I2/I3 stubs,
- strict total.

## 8. Validation Gate

Sprint I2 is accepted only as an implementation smoke result when:

1. all required adapters produce `Z[B,T,D]` and masks,
2. the processor preserves the canonical shape with and without ablation flags,
3. a dummy text/audio/visual batch runs forward and backward through latent prediction,
4. gradients are finite and non-zero for trainable parameters participating in the dummy objective,
5. parameter counts are reported by adapter, processor, and head.

No model-quality, grounding, hallucination, or policy claims are made from I2 smoke tests.

I3 acceptance adds API-level tests that verify:

1. text, audio, visual, and no-op heads produce scored proposals,
2. default no-op does not decode all external heads,
3. single-head, multi-head, and zero-head policy routes work,
4. diagnostic/probe outputs remain internal-only,
5. `NumpySLWMCore.forward(...)` wires proposals through the policy gate.

No behavior-quality, usefulness, grounding, or hallucination claim is made from I3 stubs.
