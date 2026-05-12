# SLWM Sprint R1 — Design Decisions

**Sprint:** R1 — Literature-to-design mapping  
**Primary purpose:** state what SLWM will implement now, what it will not implement now, and what remains future-phase.  
**Status:** pre-implementation decision record; no empirical success is claimed here.

---

## 0. Decision Rules

Every current-phase design decision must satisfy at least one of:

1. It directly tests an R0 hypothesis or guardrail from `docs/research/hypotheses.md`.
2. It is required by the architecture/evaluation contract in `AGENTS.md`.
3. It is justified by a mapped source in `docs/research/literature_map.md`.

Every non-trivial component must have an ablation or control. If a component cannot be ablated cleanly, it cannot support an architecture claim.

No decision below is evidence that SLWM works. Evidence requires registered experiments in `docs/experiments/experiment_registry.md`.

---

## 1. Implement Now — Current-Phase Decisions

| ID | Decision | Why / reference | Affected module(s) / eval(s) | Required ablation or control | Linked R0 item |
|---|---|---|---|---|---|
| DD-R1-001 | Build a GPT-2-small-style text/code baseline before claiming SLWM text or code quality. | GPT-2; Hugging Face GPT-2 implementation reference; project baseline contract. | GPT-2 baseline, text/code evals, `TextDecoderHead`. | Same tokenizer/data/split/compute where possible; SLWM text-only; no-spectral text ablation. | G-R0-1 |
| DD-R1-002 | Use BPE tokenization as the first text/code edge codec for comparability, while preserving a later byte-level ablation. | GPT-2 comparison fairness; research plan text adapter recommendation. | `TextSignalAdapter`, `TextDecoderHead`, text/code configs. | BPE vs byte-level text adapter; report tokenizer and data budgets. | G-R0-1 |
| DD-R1-003 | Implement a vanilla multimodal Transformer baseline with comparable adapters/latents where feasible. | Required project baseline; isolates SLWM signal-processor value. | Baseline model, multimodal evals. | Same adapters/latent length; same data mix; parameter accounting. | H-R0-1, H-R0-2, H-R0-3 |
| DD-R1-004 | Implement or plan a Perceiver-style latent bottleneck baseline where feasible. | Perceiver IO; tests whether fixed latent arrays alone explain gains. | Perceiver baseline, `LatentSignalField`, query/readout heads. | Same adapters; comparable latent size; latent-size sweep. | H-R0-2, H-R0-5 |
| DD-R1-005 | Use one common latent field contract `Z: FloatTensor[B,T,D]` for required modalities. | Perceiver IO; ImageBind/CLIP shared embedding precedent; AGENTS architecture contract. | `LatentSignalField`, all adapters, processor, heads. | Shared core vs separate cores; no shared core; shuffled-pair controls. | H-R0-2 |
| DD-R1-006 | Keep modality-specific adapters at the edges and a shared processor in the middle. | Research-plan architecture; ImageBind/CLIP alignment precedent; R0 shared-core hypothesis. | `TextSignalAdapter`, `AudioSignalAdapter`, `VisualSignalAdapter`, `SignalWorldProcessor`. | Separate modality cores; single-modality training; modality dropout. | H-R0-2 |
| DD-R1-007 | Start audio with simple log-mel or lightweight latent features, then compare EnCodec-style latents. | EnCodec motivates compressed audio latents; log-mel reduces first-pass complexity. | `AudioSignalAdapter`, audio latent prediction/reconstruction evals. | Log-mel vs EnCodec; frozen vs trainable codec; raw waveform small control if feasible. | H-R0-1, H-R0-2 |
| DD-R1-008 | Start visual/video with patch or tube-patch latents and masked/future latent prediction tasks. | VideoMAE, I-JEPA, V-JEPA. | `VisualSignalAdapter`, `ReconstructionHead`, `LatentPredictionHead`, video evals. | Patch vs tube; reconstruction-only vs latent prediction; shuffled temporal order. | H-R0-1, H-R0-2 |
| DD-R1-009 | Make latent prediction a primary training objective family, not raw reconstruction alone. | data2vec, JEPA/I-JEPA/V-JEPA. | `LatentPredictionHead`, masked completion, future latent prediction evals. | No latent prediction; reconstruction-only; latent-only vs mixed losses. | H-R0-1 |
| DD-R1-010 | Retain reconstruction heads for decodability checks and diagnostics, but do not let reconstruction define the whole objective. | VideoMAE/EnCodec reconstruction use cases; R0 probe requirements. | `ReconstructionHead`, `AudioDecoderHead`, `VisualDecoderHead`, diagnostic probes. | Reconstruction-only; no reconstruction; forced diagnostic decode with null/random controls. | H-R0-1, H-R0-5 |
| DD-R1-011 | Implement spectral processing only as an independently disableable processor subpath. | FNet; SIREN/Fourier signal precedent; R0 spectral hypothesis. | `SpectralMixer`, `SignalWorldProcessor`, synthetic/audio/video signal evals. | No spectral mixer; spectral-only; local/windowed spectral variant; vanilla baseline. | H-R0-3 |
| DD-R1-012 | Include a local temporal mixer/filter path for short-range signal structure. | H-R0-3 and the research-plan hybrid processor contract; complements FNet/Hyena/Mamba long-range and spectral paths with an ablatable short-range temporal path. | Processor block, adapters, temporal signal evals. | No local temporal mixer; local-only vs hybrid. | H-R0-3 |
| DD-R1-013 | Include either `LongConv` or `SSMBlock` as a replaceable long-range temporal path, depending on implementation feasibility. | Hyena for long convolution; Mamba for SSM. | `LongConv`, `SSMBlock`, `SignalWorldProcessor`. | No long-conv/SSM; LongConv vs SSM; attention-only; throughput/memory logging. | H-R0-3 |
| DD-R1-014 | Keep attention or binding capacity in the processor/baseline instead of using a pure Fourier/SSM-only core. | FNet limitations; research-plan hybrid block; multimodal binding risk. | `SignalWorldProcessor`, multimodal binding evals. | Remove attention/binding; spectral-only; SSM-only; retrieval/grounding comparison. | H-R0-2, H-R0-3 |
| DD-R1-015 | Add uncertainty/source outputs as first-class prediction heads before unsupported-output claims. | TruthfulQA, HaluEval, POPE/MME protocols, `docs/exploration/exploration.md`. | `UncertaintyHead`, hallucination evals, probe reports. | No uncertainty head; calibration ECE; source-tag coverage. | H-R0-4, H-R0-5 |
| DD-R1-016 | Implement policy/commitment as a gate over proposals, with no-op/wait as valid outputs, when policy sprint scope begins. | R0 H-R0-4; hallucination evals require answer vs abstain/no-op; AGENTS policy contract. | `PolicyCommitGate`, `NoOpHead`, output heads, policy evals. | No policy; fixed router; always-answer; always-no-op; no no-op head; no uncertainty. | H-R0-4 |
| DD-R1-017 | Require diagnostic probes to be internal-only and source/uncertainty-tagged. | `docs/exploration/exploration.md`; linear probing precedent; R0 H-R0-5. | Diagnostic text/audio/visual probes, `UncertaintyHead`, exploration logs. | Random latent; null probe; shuffled modality; frozen random head; no shared core. | H-R0-5 |
| DD-R1-018 | Use CLIP/ImageBind-style retrieval as grounding evaluation precedent, not as unaccounted proof of SLWM performance. | CLIP and ImageBind. | Cross-modal retrieval evals, alignment loss diagnostics. | Shuffled pairs; random retrieval; no alignment loss; separate cores; frozen reference labeled separately. | H-R0-2 |
| DD-R1-019 | Gate hallucination/unsupported-output claims on usefulness, accuracy, contradiction, abstention/no-op, and calibration. | TruthfulQA, HaluEval, POPE/MME; AGENTS evaluation rules. | Hallucination evals, policy evals, registry metric bundle. | Always-no-op; always-answer; no-policy; no-uncertainty; fixed router. | H-R0-4 |
| DD-R1-020 | Register every experiment before it can change a claim state. | R0 `docs/experiments/experiment_registry.md`; project evidence rules. | Training/eval/exploration workflows. | Reject unregistered results as evidence. | all R0 items |

---

## 2. Do Not Implement Now — Current-Phase Exclusions

| ID | Not-now decision | Why not now | What would be required later |
|---|---|---|---|
| DN-R1-001 | Do not add optional sensor, robot, proprioception, or tool-use modalities in the current phase. | R1 and R0 are scoped to required modalities: text/code, audio, visual/video. Optional modalities would create sprint drift. | Required modality gates complete; separate hypothesis/eval plan; action/sensor data contract. |
| DN-R1-002 | Do not implement main-weight continual learning or online weight updates. | AGENTS memory policy forbids it in the initial phase; privacy/rollback/forgetting risks are unresolved. | Isolated continual-learning phase with replay, rollback, forgetting tests, and privacy controls. |
| DN-R1-003 | Do not default to decoding all heads during inference. | Violates policy-selected output contract and confuses internal prediction with external behavior. | Policy/commit gate with explicit committed/suppressed/diagnostic metadata. |
| DN-R1-004 | Do not claim hallucination reduction before E2-style evaluation. | R0 and AGENTS require unsupported rate plus usefulness/accuracy, abstention/no-op, contradiction, and calibration. | Registered E2 evals with baselines and ablations. |
| DN-R1-005 | Do not treat diagnostic probe outputs as proof of understanding or grounded latent representations. | Probe outputs can reflect decoder priors or probe training shortcuts. | Random/null/shuffled controls, source tags, failure cases, causal/intervention tests. |
| DN-R1-006 | Do not use frozen CLIP/ImageBind/V-JEPA/wav2vec/EnCodec results as fair direct wins unless external pretraining and parameter/data accounting are labeled. | Pretraining budgets are incomparable to SLWM-124M. | Strict/core accounting labels and reference-only interpretation, or matched training budget. |
| DN-R1-007 | Do not prioritize raw waveform/video generation as a primary deliverable. | Current goal is latent signal prediction, alignment, and controlled evaluation, not demo-quality generation. | Decoder-focused sprint after latent and grounding gates pass. |
| DN-R1-008 | Do not merge architecture, training, evaluation, and exploration into one sprint. | Violates no-sprint-drift rules and blocks falsification. | Follow playbook gates: I0/I1/I2 before T0/T1/T2/E0/X0 as applicable. |
| DN-R1-009 | Do not claim SLWM is broadly novel or brain-like. | R1 justifies a testable combination of known ideas; novelty claims require careful publication-level review and evidence. | Literature review beyond R1 plus empirical evidence and precise claim wording. |

---

## 3. Future-Phase Candidates

| ID | Candidate | Trigger to revisit | Required evidence/control |
|---|---|---|---|
| FP-R1-001 | Trainable neural audio codec or EnCodec-style adapter fine-tuning. | Log-mel/simple latent path works and audio latent prediction beats null/shuffled baselines. | Frozen vs trainable codec, parameter accounting, audio reconstruction/continuation metrics. |
| FP-R1-002 | Stronger video representation targets or V-JEPA-style frozen teacher. | Patch/tube visual path works and video latent prediction is nontrivial. | Frozen teacher labeled reference; trainable adapter control; temporal-shuffle control. |
| FP-R1-003 | Deeper SSM or Hyena-style backbone exploration. | Initial processor ablations show temporal block value and implementation budget allows. | LongConv vs SSM vs attention-only; throughput/memory; signal and text guardrails. |
| FP-R1-004 | Learned policy optimization beyond supervised answer/abstain/no-op. | I3 policy stubs and E2 metrics show supervised policy value without abstention collapse. | Fixed-router/always/no-op baselines; usefulness-adjusted hallucination; calibration. |
| FP-R1-005 | Action/affordance experiments. | Required text/audio/visual gates pass and action hypotheses/evals are pre-registered. | No-op/action controls; safety/uncertainty metrics; clear current vs future separation. |
| FP-R1-006 | External episodic memory. | Working-memory-only context shows a measured limitation and memory task is defined. | Retrieval vs no retrieval; privacy/leakage controls; no online weight update. |

---

## 4. P0 Traceability Matrix

| P0 item | Current decision source | Trace status |
|---|---|---|
| `TextSignalAdapter` | GPT-2, BPE edge-codec decision, G-R0-1 | Covered |
| `AudioSignalAdapter` | EnCodec/log-mel decision, data2vec/audio latent prediction, H-R0-1/H-R0-2 | Covered |
| `VisualSignalAdapter` | VideoMAE, JEPA/V-JEPA, CLIP grounding evals | Covered |
| `LatentSignalField` | Perceiver IO, ImageBind/CLIP, H-R0-2 | Covered |
| `SignalWorldProcessor` | GPT/Transformer baseline, FNet, Hyena, Mamba, H-R0-3 | Covered |
| `SpectralMixer` | FNet, SIREN signal precedent, H-R0-3 | Covered |
| `LongConv` or `SSMBlock` | Hyena or Mamba, H-R0-3 | Covered |
| `LatentPredictionHead` | data2vec, JEPA/V-JEPA, H-R0-1 | Covered |
| `ReconstructionHead` | VideoMAE, EnCodec, H-R0-1/H-R0-5 | Covered |
| `UncertaintyHead` | TruthfulQA/HaluEval/POPE/MME metrics, H-R0-4/H-R0-5 | Covered |
| `PolicyCommitGate` | R0 policy hypothesis, hallucination eval protocols, AGENTS contract | Covered |
| `TextDecoderHead` | GPT-2, LAMBADA, HumanEval/MBPP, grounded QA | Covered |
| `AudioDecoderHead` | EnCodec/log-mel diagnostic reconstruction, exploration controls | Covered |
| `VisualDecoderHead` | VideoMAE/I-JEPA diagnostics, POPE/MME grounding controls | Covered |
| `NoOpHead` | H-R0-4, abstention/no-op gate, always-no-op baseline | Covered |
| `ActionHead` | AGENTS architecture lists it only if action experiments are enabled; optional/future | Future-only covered |
| Baselines | GPT-2, vanilla multimodal, Perceiver, ablated SLWM variants | Covered |
| Evals | LAMBADA, HumanEval/MBPP, TruthfulQA, HaluEval, POPE, MME, retrieval/correspondence | Covered |
| Registry/evidence | R0 experiment registry schema | Covered |

---

## 5. Required Configuration Flags for Later Implementation

These flags are design requirements, not implemented code in R1.

```yaml
processor:
  use_local_temporal_mixer: true
  use_spectral_mixer: true
  use_longconv: true
  use_ssm: false
  use_attention_binding: true

objectives:
  use_latent_prediction: true
  use_reconstruction: true
  use_cross_modal_alignment: false
  use_uncertainty_loss: false

policy:
  use_policy_commit_gate: false
  use_noop_head: false

adapters:
  text_codec: bpe
  audio_features: log_mel
  visual_features: patch_or_tube_patch
```

Rules for these flags:

- Each true flag must be disableable for ablation before large-scale training.
- If both `use_longconv` and `use_ssm` are enabled in an experiment, that experiment cannot isolate which temporal mechanism caused a gain unless additional ablations are run.
- Policy/no-op flags should stay off until the policy sprint unless the sprint explicitly requires stubs.

---

## 6. Acceptance Checklist

- [x] Current-phase decisions are separated from not-now and future-phase decisions.
- [x] Every current design choice has a reference or R0 hypothesis link.
- [x] Every reference maps to a concrete module or evaluation through `docs/research/literature_map.md`.
- [x] Every non-trivial component has an ablation/control requirement.
- [x] P0 modules are traceable.
- [x] No empirical success claim is made.
- [x] Optional modalities and continual learning are kept future-phase.
- [x] Hallucination/unsupported-output claims are gated by usefulness and abstention/no-op metrics.

---

## 7. Decision State

R1 justifies the design space and current-phase boundaries. It does not authorize skipping baselines, ablations, or registered evaluations. The next implementation sprint should use this file only as traceability input, not as evidence that any component will improve metrics.
