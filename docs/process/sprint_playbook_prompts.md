# sprint_playbook_prompts.md

# SLWM Sprint Playbook and Agent Prompts

**Project:** Signal-Latent World Models, anchored by GPT-2-size `SLWM-124M` and extending only through explicitly labeled larger profiles  
**Purpose:** define clear implementation details, sprint boundaries, KPIs, success criteria, and prompts for research, implementation, training, evaluation, and exploration agents.  
**Rule:** do not merge sprints, invent new scope mid-sprint, or claim success without the stated KPIs.

---

## 0. Operating Principle

SLWM is a research system, not a demo-first product. `SLWM-124M` remains the comparison anchor; larger profiles must be labeled by parameter accounting mode, data budget, and compute budget.

Every sprint must answer one of these questions:

1. **Representation:** does the shared latent signal field work?
2. **Prediction:** does latent signal prediction improve over baselines?
3. **Grounding:** does multimodal signal training reduce unsupported outputs while preserving usefulness?
4. **Policy:** can the model choose when to speak, act, wait, or remain internal?
5. **Exploration:** can we inspect the model's latent world view without confusing probes with committed behavior?

If a task does not support one of those questions, it is out of scope.

---

## 1. Non-Negotiable Anti-Drift Rules

### 1.1 Sprint Isolation

Each sprint has one owner, one objective, one artifact set, and one success gate.

Do **not** glue together:

- architecture + training + eval in the same sprint,
- data preprocessing + model redesign in the same sprint,
- exploration visualizations + claims about hallucination reduction,
- policy behavior + core latent prediction,
- GPT-2 comparison + multimodal comparison unless the sprint explicitly requires both.

### 1.2 No Unmeasured Claims

Forbidden claims unless backed by a registered eval:

- “more brain-like,”
- “less hallucination,”
- “better reasoning,”
- “understands video,”
- “learns primitive signals,”
- “world model emerged.”

Allowed wording before measurement:

- “implements latent signal prediction,”
- “improves metric X on dataset Y,”
- “reduces unsupported-claim rate under eval Z,”
- “probe suggests modality alignment, pending causal validation.”

### 1.3 Experiment Registry Required

Every training/eval run must have:

```yaml
experiment_id:
git_commit:
config_path:
model_variant:
parameter_count:
dataset_mix:
train_tokens_or_samples:
random_seed:
checkpoint_path:
eval_script:
metrics:
notes:
```

No registry entry means the result is not part of the research record.

### 1.4 Baseline Before Novelty

No new module is accepted unless compared against at least one baseline.

Minimum baselines by topic:

| Topic                | Required baseline                                                              |
| -------------------- | ------------------------------------------------------------------------------ |
| Text/code            | GPT-2-small-style decoder                                                      |
| Latent bottleneck    | Perceiver-style latent processor                                               |
| Spectral processor   | same model with spectral mixer removed                                         |
| Policy gate          | fixed rule router / always-text / always-no-op baselines                       |
| Multimodal grounding | vanilla multimodal Transformer or CLIP/ImageBind-style frozen encoder baseline |
| Exploration probes   | random-latent probe and shuffled-modality probe                                |

---

## 2. Canonical System Interfaces

All agents must preserve these boundaries.

```text
raw/compressed modality signals
        ↓
modality adapters
        ↓
shared latent signal field
        ↓
signal world processor
        ↓
updated latent world field
        ↓
heads:
    latent prediction
    reconstruction
    uncertainty
    text/code
    audio
    visual/video
    policy/commitment
        ↓
committed external outputs or internal-only probes
```

### 2.1 Core Tensor Contract

Use one common latent form:

```python
Z: FloatTensor[B, T, D]
```

Recommended GPT-2-size target:

```yaml
context_length_T: 1024
latent_dim_D: 768
processor_layers: 12
heads_or_mixing_groups: 12
parameter_budget: ~124M total for strict comparison
```

Optional spatial/video form before pooling/projection:

```python
Z_video: FloatTensor[B, T, P, D]
```

where `P` is patch or region count. It must be projected into the shared field before the core processor unless the experiment explicitly tests a structured spatial latent.

### 2.2 Required Module APIs

```python
class ModalityAdapter(nn.Module):
    def forward(self, sample: dict) -> dict:
        # returns {"z": Tensor[B,T,D], "mask": Tensor[B,T], "metadata": dict}
        ...

class SignalWorldProcessor(nn.Module):
    def forward(self, z, mask=None, state=None) -> dict:
        # returns {"z_world": Tensor[B,T,D], "aux": dict}
        ...

class OutputHead(nn.Module):
    def forward(self, z_world, query=None, metadata=None) -> dict:
        # returns modality-specific logits/latents/proposals
        ...

class PolicyCommitGate(nn.Module):
    def forward(self, z_world, proposals, uncertainty, goal=None) -> dict:
        # returns commitments, gates, no-op probability, and rationale metadata
        ...
```

### 2.3 Inference Modes

Do not hard-code one inference meaning.

| Mode          | Meaning                      | Required output                           |
| ------------- | ---------------------------- | ----------------------------------------- |
| `perception`  | process current full context | updated latent world field                |
| `predict`     | rollout future latent state  | future latent prediction + uncertainty    |
| `reconstruct` | decode hidden/missing signal | reconstructed modality latent/output      |
| `commit`      | decide external behavior     | selected heads + no-op/speak/act decision |
| `explore`     | diagnostic probing only      | probe outputs marked internal-only        |

---

## 3. KPI Dashboard

These KPIs are project-wide. Individual sprints select a subset.

### 3.1 Engineering KPIs

| KPI                                         |                                         Target |
| ------------------------------------------- | ---------------------------------------------: |
| Reproducible config runs                    |                                100% registered |
| Unit test pass rate                         |                                         >= 95% |
| Shape-contract tests for all adapters/heads |                                           100% |
| Checkpoint load/save parity                 |           exact tensor equality for smoke test |
| Deterministic tiny run                      | same loss curve within tolerance across 2 runs |
| GPU memory regression                       |         <= 10% unexplained increase per sprint |

### 3.2 Training KPIs

| KPI                             |                                                         Target |
| ------------------------------- | -------------------------------------------------------------: |
| Tiny overfit batch              | loss decreases by >= 90% or reaches near-zero where applicable |
| Synthetic periodic continuation |        beats vanilla Transformer baseline by registered metric |
| Multimodal batch utilization    |               >= 90% valid samples after preprocessing filters |
| NaN/Inf rate                    |                            0 tolerated in accepted checkpoints |
| Training throughput logging     |                   tokens/s or samples/s recorded for every run |
| Parameter count accounting      |                                         exact, split by module |

### 3.3 Evaluation KPIs

| KPI                                      |                                                     Target |
| ---------------------------------------- | ---------------------------------------------------------: |
| Baseline available before comparison     |                                                        yes |
| Random/shuffled sanity checks            |                                                        yes |
| Confidence calibration reported          |                         ECE or equivalent where applicable |
| Hallucination result includes usefulness |                                                        yes |
| Eval reproducibility                     | same checkpoint + seed gives same metrics within tolerance |
| Eval scripts versioned                   |                                                        yes |

### 3.4 Research KPIs

| KPI                                     | Target |
| --------------------------------------- | -----: |
| Hypothesis is falsifiable               |    yes |
| Success/failure gate defined before run |    yes |
| Ablation isolates one variable          |    yes |
| Negative result recorded                |    yes |
| Finding links to exact experiment IDs   |    yes |

### 3.5 Exploration KPIs

| KPI                                           |                     Target |
| --------------------------------------------- | -------------------------: |
| Probe marked diagnostic, not committed output |                       100% |
| Probe baseline included                       | random/shuffled/null probe |
| Cross-modal mapping tested both directions    |           where applicable |
| Human-readable visualization saved            |                        yes |
| Probe confidence/uncertainty reported         |                        yes |

---

## 4. Sprint Plan Overview

| Stage          | Sprint | Name                           | Primary artifact                      | Gate                                 |
| -------------- | -----: | ------------------------------ | ------------------------------------- | ------------------------------------ |
| Research       |     R0 | Hypotheses and falsification   | `docs/research/hypotheses.md`         | claims are measurable                |
| Research       |     R1 | Literature-to-design mapping   | `docs/research/literature_map.md`     | design choices justified             |
| Implementation |     I0 | Repo skeleton and contracts    | code + `docs/model_spec.md`           | shape tests pass                     |
| Implementation |     I1 | Baselines                      | GPT-2 + vanilla baselines             | baseline metrics logged              |
| Implementation |     I2 | SLWM core                      | processor + adapters                  | tiny forward/backward works          |
| Implementation |     I3 | Output heads and policy stubs  | heads + gate APIs                     | no-op/text/probe paths work          |
| Training       |     T0 | Synthetic signal pretraining   | training loop + dataset               | overfit + baseline comparison        |
| Training       |     T1 | Text/code baseline training    | GPT-2-size run                        | PPL/loss logged                      |
| Training       |     T2 | Audio/visual latent training   | multimodal latent runs                | prediction metrics logged            |
| Training       |     T3 | Joint SLWM training            | mixed-modal run                       | stable training + registered ckpts   |
| Evaluation     |     E0 | Eval harness                   | `evals.md` + scripts                  | all baselines runnable               |
| Evaluation     |     E1 | Signal/text/code evals         | metric tables                         | pass/fail vs baselines               |
| Evaluation     |     E2 | Multimodal/hallucination evals | grounded reports                      | unsupported/usefulness both reported |
| Exploration    |     X0 | Probe harness                  | `exploration_evals.md` + scripts      | probes + null baselines              |
| Exploration    |     X1 | Worldview maps                 | saved visual/audio/text probe reports | cross-modal maps interpretable       |
| Integration    |     G0 | Findings report                | `findings.md`                         | claim matrix completed               |

`G0` is not a glue sprint. It only writes findings from completed sprint outputs.

---

# 5. Research Stage

## R0 — Hypotheses and Falsification

### Objective

Convert the project idea into falsifiable claims.

### Scope

Write exact hypotheses for:

1. latent signal prediction,
2. multimodal grounding,
3. spectral/temporal processing,
4. policy/commitment,
5. exploration probes.

### Deliverables

- `docs/research/hypotheses.md`
- `docs/research/risks_and_assumptions.md`
- initial `docs/experiments/experiment_registry.md` schema

### KPIs

| KPI                                      | Success criterion |
| ---------------------------------------- | ----------------- |
| Every hypothesis has a metric            | 100%              |
| Every hypothesis has a baseline          | 100%              |
| Every hypothesis has a failure condition | 100%              |
| Claims separated from speculation        | yes               |

### Success Gate

Accepted only if each hypothesis can be answered with `support`, `partial support`, or `not supported`.

### Agent Prompt

```text
You are the Research Agent for SLWM Sprint R0.
Your task is to convert the project vision into falsifiable hypotheses.
Do not propose implementation changes.
Do not add new modalities.
For each hypothesis, write: claim, metric, dataset, baseline, ablation, success threshold, failure threshold, and expected interpretation.
Also write explicit assumptions and risks.
Output only `docs/research/hypotheses.md`, `docs/research/risks_and_assumptions.md`, and `docs/experiments/experiment_registry.md` schema sections.
If a claim cannot be measured at GPT-2 scale, mark it as future-phase and remove it from current success criteria.
```

---

## R1 — Literature-to-Design Mapping

### Objective

Map existing methods to specific SLWM design decisions.

### Scope

Summarize only papers that affect architecture, training, or evaluation.

Required topic buckets:

- GPT-2 / decoder baseline,
- Perceiver IO / latent bottleneck,
- data2vec / JEPA latent prediction,
- FNet / Fourier mixing,
- Hyena / long convolutions,
- Mamba / SSM,
- SIREN / periodic activations,
- ImageBind / multimodal shared embeddings,
- EnCodec / audio codecs,
- VideoMAE or equivalent visual/video latent learning,
- hallucination/grounding evals.

### Deliverables

- `docs/research/literature_map.md`
- `docs/research/design_decisions.md`

### KPIs

| KPI                                     | Success criterion |
| --------------------------------------- | ----------------- |
| Each design choice has a reference      | yes               |
| Each reference maps to a module or eval | yes               |
| No unsupported novelty claims           | yes               |
| Current-phase vs future-phase separated | yes               |

### Success Gate

Accepted only if the implementation team can trace every P0 module to either a project hypothesis or a literature-backed design choice.

### Agent Prompt

```text
You are the Research Agent for SLWM Sprint R1.
Create a literature-to-design map.
Do not write a general survey.
For each source, extract only: what it contributes, what module or eval it affects, what risk it introduces, and what ablation should test it.
End with a `docs/research/design_decisions.md` section that states what SLWM will implement now, what it will not implement now, and why.
Keep all design claims testable.
```

---

# 6. Implementation Stage

## I0 — Repository Skeleton and Contracts

### Objective

Create the minimal repo structure and stable interfaces.

### Scope

Implement no novel model logic beyond stubs and shape contracts.

### Required Repo Layout

```text
slwm/
  configs/
  data/
  models/
    adapters/
    processor/
    heads/
    policy/
    baselines/
  training/
  evals/
  exploration/
  utils/
  tests/
  docs/
```

### Deliverables

- `model_spec.md`
- `data_contract.md`
- module stubs
- shape-contract tests
- config loader
- experiment registry writer

### KPIs

| KPI                                        | Success criterion |
| ------------------------------------------ | ----------------- |
| All stubs import                           | yes               |
| Unit tests pass                            | >= 95%            |
| Shape tests for `Z[B,T,D]`                 | yes               |
| Config round trip works                    | yes               |
| Experiment registry writes valid YAML/JSON | yes               |

### Success Gate

A dummy batch passes through adapter → processor → head → policy with correct shapes and no training logic.

### Agent Prompt

```text
You are the Implementation Agent for SLWM Sprint I0.
Build only the repository skeleton, type contracts, configs, and tests.
Do not implement the real architecture.
Do not train anything.
Do not add undocumented dependencies.
Every module must accept and return the canonical dict interface.
Add tests for tensor shapes, masks, modality IDs, and experiment registry output.
Success is a clean test run and a dummy end-to-end forward pass.
```

---

## I1 — Baselines

### Objective

Implement required baselines before SLWM novelty.

### Scope

Implement:

1. GPT-2-small-style decoder-only baseline,
2. vanilla multimodal Transformer baseline,
3. minimal Perceiver-style latent bottleneck baseline if feasible,
4. null/random baselines for probes.

### Deliverables

- `docs/evaluation/baselines.md`
- baseline model code
- baseline configs
- baseline smoke tests
- baseline parameter-count report

### KPIs

| KPI                             | Success criterion           |
| ------------------------------- | --------------------------- |
| GPT-2-style baseline runs       | yes                         |
| Parameter count close to target | within documented tolerance |
| Baseline forward/backward works | yes                         |
| Tiny overfit works              | loss drops materially       |
| Baseline metrics logged         | yes                         |

### Success Gate

At least one text baseline and one multimodal/latent baseline can train for a tiny run and produce logged metrics.

### Agent Prompt

```text
You are the Implementation Agent for SLWM Sprint I1.
Implement baselines only.
Do not implement SLWM novelty modules except shared utilities needed by baselines.
Report exact parameter counts, model configs, context length, tokenizer/codec choices, and training smoke-test metrics.
A baseline is not accepted until it can overfit a tiny batch and write an experiment registry entry.
```

---

## I2 — SLWM Core Processor and Adapters

### Objective

Implement the first real SLWM architecture without policy complexity.

### Scope

Implement:

- `TextSignalAdapter`,
- `AudioSignalAdapter`,
- `VisualSignalAdapter`,
- `SignalWorldProcessor`,
- spectral mixer block,
- long-conv or SSM block,
- gated MLP,
- latent prediction head,
- uncertainty head.

Do **not** implement full policy behavior yet.

### Minimal Processor Block

```text
Z
↓
norm
↓
local temporal mixer
↓
spectral or long-range mixer
↓
gated MLP
↓
residual output
```

### Deliverables

- `docs/architecture/architecture.md`
- implemented model modules
- processor configs
- shape tests
- tiny forward/backward tests
- parameter-count breakdown

### KPIs

| KPI                                              | Success criterion                          |
| ------------------------------------------------ | ------------------------------------------ |
| Text/audio/visual adapters produce `Z[B,T,D]`    | yes                                        |
| Processor preserves canonical shape              | yes                                        |
| Latent prediction head trains on dummy targets   | yes                                        |
| Uncertainty head returns calibrated-form outputs | interface ready                            |
| Parameter budget report exists                   | yes                                        |
| Ablation flags exist                             | spectral/no-spectral, longconv/no-longconv |

### Success Gate

One SLWM core config runs forward/backward on synthetic multimodal batches and can be ablated through config flags.

### Agent Prompt

```text
You are the Implementation Agent for SLWM Sprint I2.
Implement the SLWM core and adapters only.
Preserve the canonical latent tensor contract Z[B,T,D].
Do not implement advanced policy, exploration dashboards, or large training.
Every novel block must have a config flag to disable it for ablation.
Write parameter counts by adapter, processor, and head.
Success is a stable forward/backward pass on text, audio-latent, and visual-latent dummy batches plus passing shape tests.
```

---

## I3 — Output Heads and Policy Stubs

### Objective

Implement output head interfaces and a minimal policy/commitment mechanism.

### Scope

Implement:

- text/code head,
- audio latent head,
- visual latent head,
- no-op head,
- proposal interface,
- policy commit gate,
- fixed-rule policy baseline,
- learned policy stub.

Do not train a complex agentic policy yet.

### Deliverables

- `docs/architecture/policy_commitment.md`
- `docs/architecture/inference.md`
- output head code
- policy code
- commit/no-op tests

### KPIs

| KPI                                            | Success criterion |
| ---------------------------------------------- | ----------------- |
| Each head can produce a proposal               | yes               |
| Policy can select zero, one, or multiple heads | yes               |
| No-op is valid output                          | yes               |
| Diagnostic probe outputs marked internal-only  | yes               |
| Commit metadata saved                          | yes               |

### Success Gate

Given a processed latent field, the system can produce candidate outputs and route them through a policy gate without confusing probe outputs with committed external outputs.

### Agent Prompt

```text
You are the Implementation Agent for SLWM Sprint I3.
Implement output heads and policy/commit interfaces.
Do not optimize behavior quality yet.
The policy must support no-op, single-head commitment, and multi-head commitment.
Every output must be tagged as committed, suppressed, or diagnostic-only.
Include a fixed-rule policy baseline and a learned policy stub.
Success is a test where text, audio, visual, and no-op proposals are generated, scored, and either committed or suppressed with metadata.
```

---

# 7. Training Stage

## T0 — Synthetic Signal Pretraining

### Objective

Prove the signal architecture helps on controlled signal tasks.

### Scope

Train on synthetic:

- sine mixtures,
- chirps,
- phase shifts,
- noisy periodic signals,
- missing-span reconstruction,
- long-horizon extrapolation.

Compare to:

- vanilla Transformer baseline,
- no-spectral SLWM ablation,
- random/no-op predictor.

### Deliverables

- `training.md` initial version
- synthetic datasets
- training loop
- baseline comparison table
- plots of prediction vs target

### KPIs

| KPI                                                          | Success criterion           |
| ------------------------------------------------------------ | --------------------------- |
| Tiny overfit                                                 | yes                         |
| No NaN/Inf                                                   | yes                         |
| Beats random predictor                                       | yes                         |
| Beats or matches vanilla baseline on at least 2 signal tasks | required for moving forward |
| Spectral ablation measured                                   | yes                         |
| Phase/frequency metrics reported                             | yes                         |

### Success Gate

Proceed only if SLWM core shows measurable value on at least one periodic/frequency task and does not collapse on others.

### Agent Prompt

```text
You are the Training Agent for SLWM Sprint T0.
Train only on synthetic signal tasks.
Do not use text/code/audio/video datasets yet.
Compare SLWM, vanilla Transformer, and no-spectral ablation.
Record MSE, spectral loss, phase error, frequency recovery error, throughput, and stability.
If SLWM does not beat the baseline on any controlled signal task, stop and write a failure report instead of proceeding.
```

---

## T1 — Text/Code Baseline Training

### Objective

Establish text/code baselines at GPT-2-size or smaller pilot scale.

### Scope

Train/evaluate:

- GPT-2-style decoder,
- SLWM text adapter + text head in text-only mode,
- no-spectral SLWM text ablation.

Datasets should focus on English text and code. Use small pilot subsets before full runs.

### Deliverables

- text/code training configs
- tokenizer decision record
- loss/perplexity reports
- checkpoint registry entries

### KPIs

| KPI                                     | Success criterion |
| --------------------------------------- | ----------------- |
| GPT-2 baseline trains stably            | yes               |
| SLWM text-only trains stably            | yes               |
| Perplexity/loss reported for same split | yes               |
| Decode samples saved with seed/settings | yes               |
| Parameter budget comparison documented  | yes               |

### Success Gate

Do not claim text improvement unless SLWM beats the GPT-2 baseline under the same data/compute budget. Otherwise record it as representation tradeoff.

### Agent Prompt

```text
You are the Training Agent for SLWM Sprint T1.
Train text/code baselines and SLWM text-only variants.
Do not add audio or visual data in this sprint.
Use the same tokenizer, splits, optimizer settings where possible, and parameter accounting.
Report validation loss/perplexity, sample generations, throughput, memory use, and exact decoding settings.
If SLWM underperforms GPT-2 on text, do not hide it; record the tradeoff and continue only if signal/multimodal hypotheses remain viable.
```

---

## T2 — Audio/Visual Latent Training

### Objective

Train SLWM on compressed audio and visual/video latents.

### Scope

Use frozen or precomputed encoders/codecs initially.

Train tasks:

- audio latent continuation,
- visual/video latent continuation,
- missing-span reconstruction,
- audio-video correspondence.

Do not train raw waveform/video generation yet unless already supported by a lightweight decoder.

### Deliverables

- `docs/training/preprocessing.md`
- latent dataset pipeline
- audio/visual configs
- prediction metric reports
- checkpoint registry entries

### KPIs

| KPI                                   | Success criterion |
| ------------------------------------- | ----------------- |
| Audio latent dataset loads            | yes               |
| Visual/video latent dataset loads     | yes               |
| Batch format matches data contract    | yes               |
| Prediction loss decreases             | yes               |
| Shuffled-modality baseline included   | yes               |
| Cross-modal alignment metric reported | yes               |

### Success Gate

Proceed only if the model learns non-trivial audio/visual latent prediction and beats shuffled/null baselines.

### Agent Prompt

```text
You are the Training Agent for SLWM Sprint T2.
Train on audio and visual/video latents only, plus optional alignment pairs.
Use frozen/precomputed codecs unless the sprint explicitly says otherwise.
Do not mix in text generation objectives yet except labels/captions needed for alignment metadata.
Report latent prediction loss, spectral/audio metrics where available, video latent error, alignment accuracy, shuffled baseline, and throughput.
Reject any result that does not beat shuffled or null baselines.
```

---

## T3 — Joint Multimodal SLWM Training

### Objective

Train the first combined text/code/audio/visual SLWM model.

### Scope

Mix modalities under a fixed data schedule.

Tasks:

- latent prediction,
- hidden/missing reconstruction,
- cross-modal alignment,
- text/code output head training,
- uncertainty estimation.

Policy training remains limited to supervised or fixed-rule targets unless separately scheduled.

### Deliverables

- joint training config
- dataset mixture card
- loss-weight schedule
- checkpoint registry entries
- stability report

### KPIs

| KPI                                   | Success criterion             |
| ------------------------------------- | ----------------------------- |
| Joint training stable                 | no NaN/Inf, no loss explosion |
| All modalities contribute batches     | logged distribution           |
| No single loss dominates silently     | loss scales monitored         |
| Validation metrics for every modality | yes                           |
| Ablations scheduled                   | yes                           |

### Success Gate

Accepted if joint model trains stably, beats null/shuffled multimodal baselines, and does not catastrophically degrade text/code relative to its own text-only SLWM variant beyond a documented threshold.

### Agent Prompt

```text
You are the Training Agent for SLWM Sprint T3.
Train the first joint multimodal SLWM.
Do not change architecture during this sprint.
Use the frozen model config from I2/I3.
Use a documented dataset mixture and fixed loss weights or scheduled loss weights.
Log per-modality losses, batch proportions, gradient norms, throughput, memory, and validation metrics.
If one modality collapses or dominates, stop and report instead of patching the architecture mid-run.
```

---

# 8. Evaluation Stage

## E0 — Evaluation Harness

### Objective

Build the evaluation framework before interpreting results.

### Scope

Implement common runner for:

- model loading,
- baseline loading,
- dataset split loading,
- deterministic decoding/probing,
- metric aggregation,
- report generation.

### Deliverables

- `evals.md`
- eval runner scripts
- metric modules
- eval report template
- baseline sanity checks

### KPIs

| KPI                                     | Success criterion |
| --------------------------------------- | ----------------- |
| Same checkpoint gives reproducible eval | yes               |
| Baselines and SLWM use same splits      | yes               |
| Random/shuffled sanity checks included  | yes               |
| Metric outputs saved as JSON/CSV        | yes               |
| Report template auto-filled             | yes               |

### Success Gate

No model claim is accepted until the eval harness can run baselines and SLWM variants through the same registered protocol.

### Agent Prompt

```text
You are the Evaluation Agent for SLWM Sprint E0.
Build the eval harness only.
Do not interpret model quality yet.
Support deterministic loading, seeded decoding, shared splits, metric aggregation, and report generation.
Include random, shuffled, and fixed-rule sanity baselines.
Success is running the harness on dummy and tiny checkpoints with consistent output files.
```

---

## E1 — Text, Code, and Signal Evaluation

### Objective

Evaluate unimodal and controlled-signal competence.

### Scope

Evaluate:

- validation loss/perplexity,
- continuation tasks,
- code generation if trained,
- synthetic signal continuation,
- audio latent continuation,
- visual latent prediction.

### Deliverables

- `text_code_evals.md`
- `signal_evals.md`
- metric tables
- plots
- ablation comparisons

### KPIs

| KPI                                         | Success criterion |
| ------------------------------------------- | ----------------- |
| GPT-2 comparison included                   | yes               |
| SLWM text-only vs joint comparison          | yes               |
| Synthetic signal eval included              | yes               |
| Spectral/no-spectral ablation included      | yes               |
| Code eval only claimed if code head trained | yes               |

### Success Gate

Accepted only if every improvement claim has a matching baseline and ablation.

### Agent Prompt

```text
You are the Evaluation Agent for SLWM Sprint E1.
Evaluate text, code, and signal metrics only.
Do not evaluate hallucination or policy behavior in this sprint.
Compare GPT-2 baseline, SLWM text-only, SLWM joint, no-spectral ablation, and relevant null baselines.
Report both quality and compute/throughput.
Do not claim general reasoning or grounding from these metrics.
```

---

## E2 — Multimodal Grounding and Hallucination Evaluation

### Objective

Measure whether multimodal latent training improves grounding and reduces unsupported outputs.

### Scope

Evaluate:

- image/video-text grounding,
- audio-video alignment,
- context-grounded QA,
- unanswerable questions,
- unsupported claim rate,
- contradiction rate,
- abstention/no-op rate,
- usefulness/accuracy.

### Deliverables

- `multimodal_evals.md`
- `hallucination_evals.md`
- grounded QA reports
- unsupported-claim annotation protocol
- calibration report

### KPIs

| KPI                                               | Success criterion |
| ------------------------------------------------- | ----------------- |
| Unsupported output measured                       | yes               |
| Accuracy/usefulness measured                      | yes               |
| Abstention rate measured                          | yes               |
| Confidence calibration measured                   | yes               |
| Context-grounded and unanswerable splits included | yes               |
| Text-only and multimodal baselines included       | yes               |

### Success Gate

Do not claim hallucination reduction unless unsupported-claim rate decreases **and** usefulness/accuracy does not collapse.

### Agent Prompt

```text
You are the Evaluation Agent for SLWM Sprint E2.
Evaluate multimodal grounding and hallucination behavior.
Every hallucination result must include unsupported-claim rate, contradiction rate, abstention/no-op rate, and usefulness/accuracy.
Compare against text-only GPT-2-style baseline, vanilla multimodal baseline, and SLWM ablations.
Separate observed facts, inferred facts, and generated/imaginative outputs where metadata allows.
If the model reduces hallucination only by refusing everything, mark the result as failed.
```

---

# 9. Exploration Stage

## X0 — Probe Harness

### Objective

Build tools to inspect latent world states without turning probes into behavior claims.

### Scope

Implement diagnostic probes for:

- latent-to-text descriptions,
- latent-to-visual reconstructions,
- latent-to-audio/spectrogram reconstructions,
- cross-modal retrieval maps,
- uncertainty/source maps,
- policy proposal maps.

### Deliverables

- `exploration_evals.md`
- probe runner
- null/random/shuffled probe baselines
- saved probe artifacts
- internal-only output tagging

### KPIs

| KPI                                 | Success criterion        |
| ----------------------------------- | ------------------------ |
| Probe outputs marked diagnostic     | 100%                     |
| Null/random probe baseline exists   | yes                      |
| Shuffled-modality probe exists      | yes                      |
| Probe reproducibility               | same seed, same artifact |
| Probe confidence/uncertainty logged | yes                      |

### Success Gate

Accepted when probes can visualize or decode latent states and prove they are not just reflecting decoder priors via null/shuffled baselines.

### Agent Prompt

```text
You are the Exploration Agent for SLWM Sprint X0.
Build diagnostic probe infrastructure only.
Do not claim the model understands the world.
Every probe must have a random or shuffled control.
Every probe output must be tagged as diagnostic-only, not committed behavior.
Save artifacts, configs, seeds, and checkpoint references.
Success is a repeatable probe harness that can inspect text, audio, visual, uncertainty, and policy proposal latents.
```

---

## X1 — Worldview Mapping

### Objective

Explore how latent states map across modalities.

### Scope

Run experiments such as:

- audio latent → text description,
- audio latent → visual imagination/probe,
- visual latent → text description,
- text latent → visual probe,
- video/audio context → action affordance probe,
- uncertainty/source map over output claims.

### Deliverables

- worldview map report
- cross-modal probe galleries
- failure case gallery
- uncertainty/source visualizations
- hypothesis updates for next research cycle

### KPIs

| KPI                                               | Success criterion |
| ------------------------------------------------- | ----------------- |
| At least 3 cross-modal directions tested          | yes               |
| At least 1 reverse direction tested               | yes               |
| Null/shuffled controls shown beside real probes   | yes               |
| Failure cases included                            | yes               |
| No committed-output claims from diagnostic probes | yes               |

### Success Gate

Accepted if the report identifies concrete latent correspondences, failure modes, and next experiments without overstating subjective interpretations.

### Agent Prompt

```text
You are the Exploration Agent for SLWM Sprint X1.
Map the model's latent world view through diagnostic heads.
Test audio-to-text, visual-to-text, text-to-visual, and at least one bidirectional mapping.
Always include null or shuffled controls.
Include failure cases.
Do not claim consciousness, understanding, or reasoning from visualizations alone.
Your output is an exploration report with artifacts, controls, uncertainty maps, and follow-up hypotheses.
```

---

# 10. Integration and Findings Stage

## G0 — Findings Report

### Objective

Summarize what was learned from completed sprints only.

### Scope

No new training, no new evals, no new architecture.

Inputs:

- experiment registry,
- eval reports,
- exploration reports,
- failure reports,
- ablation tables.

### Deliverables

- `findings.md`
- claim matrix
- next-phase recommendation

### Claim Matrix Format

```markdown
| Claim                                              | Evidence   | Baseline            | Result                       | Status        | Next action                |
| -------------------------------------------------- | ---------- | ------------------- | ---------------------------- | ------------- | -------------------------- |
| Spectral mixer helps synthetic periodic prediction | EXP-T0-003 | vanilla Transformer | -18% spectral error          | supported     | test audio latent          |
| SLWM reduces hallucination                         | EXP-E2-002 | GPT-2 text baseline | unsupported ↓ but accuracy ↓ | not supported | improve policy/uncertainty |
```

### KPIs

| KPI                                 | Success criterion |
| ----------------------------------- | ----------------- |
| Every claim links to experiment IDs | yes               |
| Negative results included           | yes               |
| Unsupported claims removed          | yes               |
| Next phase is gated by evidence     | yes               |

### Success Gate

The findings report must clearly say which hypotheses are supported, partially supported, not supported, or untested.

### Agent Prompt

```text
You are the Research Lead for SLWM Sprint G0.
Write findings.md using only completed experiments and reports.
Do not run new experiments.
Do not introduce new architecture.
For each original hypothesis, state supported, partially supported, not supported, or untested.
Include negative results and failure cases.
Recommend the next phase only if evidence supports it.
```

---

## 11. Success Criteria by Project Phase

### Phase A — Prototype Validity

Success means:

- repo contracts are stable,
- baselines train,
- SLWM core trains on synthetic signals,
- spectral/temporal components show measurable value or are rejected.

Failure means:

- no stable training,
- no baseline comparison,
- no improvement on any signal task,
- architecture cannot be ablated cleanly.

### Phase B — GPT-2-Scale Viability

Success means:

- GPT-2-size baseline is registered,
- SLWM trains stably at the registered scale,
- text/code tradeoff is quantified,
- audio/visual latent prediction beats null/shuffled baselines,
- parameter accounting is honest.

Failure means:

- SLWM only works by using more parameters/compute without reporting it,
- multimodal training destroys text/code capability beyond acceptable threshold,
- results cannot be reproduced.

### Phase C — Grounding and Policy

Success means:

- policy can choose speak/wait/no-op/decode with measurable behavior,
- unsupported claims decrease without usefulness collapse,
- uncertainty and source metadata improve decisions,
- exploration probes reveal testable latent structure.

Failure means:

- lower hallucination is only higher refusal,
- policy is a hard-coded router pretending to be learned behavior,
- probes are decoder artifacts with no null-control difference.

---

## 12. Stop Conditions

Stop or pause the current line of work if any occur:

1. SLWM fails to beat baselines on all synthetic signal tasks after reasonable tuning.
2. Training instability persists across two architecture simplifications.
3. Multimodal results disappear under shuffled controls.
4. Hallucination reduction is entirely explained by abstention.
5. Exploration probes fail random/shuffled controls.
6. Parameter accounting shows SLWM is not comparable to baselines.
7. The team cannot reproduce a claimed result from the registry.

A stop condition does not kill the project. It forces a findings report and redesign decision.

---

## 13. Minimal First Milestone

The first milestone is not a full multimodal thinking system.

The first milestone is:

```text
baseline GPT-2-style model
vs
small SLWM core
on
synthetic periodic signals + text-only pilot
with
registered metrics and ablations
```

Required before moving to full multimodal:

- I0 complete,
- I1 complete,
- I2 complete,
- T0 complete,
- E0 complete,
- initial findings recorded.

---

## 14. Practical Agent Start Prompt

Use this when starting any new agent session.

```text
You are working on SLWM, a Signal-Latent World Model research project anchored by `SLWM-124M` and clearly labeled larger scale profiles.
Read `AGENTS.md`, `docs/research/signal_latent_world_model_research_plan.md`, `docs/exploration/exploration.md`, `docs/research/research_impl_eval_docs.md`, and `docs/process/sprint_playbook_prompts.md` before acting.
Your current sprint is: <SPRINT_ID>.
Do only the scope of that sprint.
Do not merge sprints.
Do not change research goals without recording a finding or design decision.
All experiments must be registered with config, seed, checkpoint, dataset mix, metrics, and git commit.
All claims must reference baselines and metrics.
If blocked, write a blocker report with the smallest next action, not a broad redesign.
```

---

## 15. Blocker Report Template

```markdown
# Blocker Report

## Sprint

## Blocker

## Evidence

## Smallest possible next action

## What was not changed

## Risk if ignored

## Recommendation
```

---

## 16. Experiment Report Template

```markdown
# Experiment Report

## Experiment ID

## Sprint

## Question

## Model variant

## Baseline

## Dataset / split

## Config

## Parameter count

## Training budget

## Metrics

## Results

## Success gate

## Passed?

## Interpretation

## Failure cases

## Next action
```

---

## 17. Final Rule

The project succeeds by producing reliable evidence, not by preserving the original idea.

If the signal-latent approach works, the evidence should show it.

If it fails, the evidence should show exactly where and why.
