# SLWM-124M Model Spec — Sprint I0

**Sprint:** I0 — repository skeleton and contracts  
**Status:** shape-contract specification only; no real architecture or training logic is implemented.  
**Primary gate:** a dummy batch passes through adapter → latent field → processor → head → policy with correct shapes.

## 1. Scope

I0 establishes stable module boundaries and tensor contracts for later sprints. It intentionally does **not** implement the GPT-2 baseline, the SLWM processor, spectral/SSM/long-conv blocks, real modality codecs, learned output heads, training loops, or evaluation claims.

## 2. Canonical Tensor Contract

The shared core contract is:

```python
Z: FloatTensor[B, T, D]
mask: BoolTensor[B, T]
```

Recommended GPT-2-scale defaults retained for later implementation:

```yaml
context_length_T: 1024
latent_dim_D: 768
processor_layers: 12  # later sprint, not I0
parameter_budget: ~124M total for strict comparison  # later sprint, not I0
```

I0 uses dependency-free `TensorSpec` objects to validate shape and dtype metadata because no Python dependency stack existed before this sprint.

## 3. Stable Module Interfaces

### Modality adapters

```python
adapter(sample) -> {
  "z": FloatTensor[B,T,D],
  "mask": BoolTensor[B,T],
  "metadata": {
    "modality": "text_code" | "audio" | "visual_video",
    "modality_id": int
  }
}
```

I0 stubs:

- `TextSignalAdapter`
- `AudioSignalAdapter`
- `VisualSignalAdapter`

### Shared latent field

```python
LatentSignalField.from_adapter_outputs(packets) -> {
  "z": FloatTensor[B,T_context,D],
  "mask": BoolTensor[B,T_context],
  "metadata": dict
}
```

### Processor

```python
SignalWorldProcessor(z, mask=None, state=None) -> {
  "z_world": FloatTensor[B,T,D],
  "aux": dict
}
```

I0 processor behavior is shape-preserving only.

### Output / diagnostic heads

```python
OutputHead(z_world, query=None, metadata=None) -> {
  "proposal": dict,
  "metadata": dict,
  ... optional TensorSpec outputs ...
}
```

I0 stubs include latent prediction, reconstruction, uncertainty/source, text, audio, visual, and no-op heads. These are proposal and shape stubs only.

### Policy / commit gate

```python
PolicyCommitGate(z_world, proposals, uncertainty=None, goal=None) -> {
  "commitments": list,
  "gates": dict,
  "noop_probability": float,
  "metadata": dict
}
```

The I0 policy stub defaults to no-op so that diagnostic/proposal outputs are not confused with committed external behavior.

## 4. Parameter Accounting

I0 stubs define no trainable parameters.

| Component | I0 trainable parameters |
|---|---:|
| adapters | 0 |
| latent field | 0 |
| processor | 0 |
| heads | 0 |
| policy | 0 |

Future experiment reports must distinguish strict accounting (`adapters + processor + heads + policy`) from core-only accounting (`processor` only).

## 5. Out-of-Scope Until Later Sprints

- I1: GPT-2, vanilla multimodal Transformer, Perceiver/null baselines.
- I2: real text/audio/visual adapters, processor blocks, spectral mixer, long-conv/SSM, trainable latent prediction and uncertainty heads.
- I3: real output-head routing and policy/commitment mechanisms.
- T/E/X stages: training, evaluation, exploration dashboards, and claims.
