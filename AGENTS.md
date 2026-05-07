# AGENTS.md — SLWM-124M Research Project

## 0. Mission

Build and evaluate **SLWM-124M**, a GPT-2-scale **Signal-Latent World Model** that learns from multimodal signals rather than treating tokens as the core substrate.

The model receives compressed or encoded **text/code, audio, and visual/video signals**, maps them into a fixed-length shared latent signal field, processes that field into an updated latent world field, and decodes only through specialized heads selected by a learned policy/commitment mechanism.

Primary research question:

> Can a GPT-2-size latent signal world model learn reusable, grounded, multimodal representations that improve signal prediction, cross-modal transfer, and unsupported-output control compared with token-only or vanilla multimodal baselines?

---

## 1. Required Reading Order

Before implementing or evaluating, agents must read these docs in order:

1. `signal_latent_world_model_research_plan.md` — scientific modelation and research definition.
2. `research_impl_eval_docs.md` — document map and required references.
3. `sprint_playbook_prompts.md` — sprint boundaries, KPIs, agent prompts, success gates.
4. `exploration.md` — diagnostic output-head and latent-worldview exploration plan.
5. `AGENTS.md` — operational constraints for agents working in the repo.

Do not invent new project direction before checking these documents.

---

## 2. Core Design Principles

### 2.1 Signals first, tokens at the edge

- Text and code are supported modalities, not the entire modeling substrate.
- Tokens may be used inside text/code adapters or decoders.
- The shared core must operate on latent signal fields, not raw token IDs.

### 2.2 Shared latent world field

- Modality-specific encoders/codecs map inputs into a common latent format.
- The processor core updates a full latent context field.
- Future prediction, reconstruction, alignment, and decoding are heads/objectives, not the only behavior of the core.

### 2.3 Prediction before expression

- The model may internally predict, reconstruct, simulate, or imagine latent states.
- Internal predictions are not automatically external outputs.
- External output requires policy/commitment selection.

### 2.4 Policy-selected output heads

- Do not naively decode all modalities.
- Heads may propose outputs; the policy/commit gate decides what is committed.
- Valid committed behaviors include speak/write, act/move, generate media, ask, wait, silence, no-op, or internal-only processing.

### 2.5 No sprint drift

- Each sprint has one objective, one deliverable class, and explicit KPIs.
- Do not glue adjacent sprints together.
- Do not add new datasets, architectures, or evaluation suites unless the current sprint gate is passed or the sprint explicitly requires it.

---

## 3. Architecture Contract

Implement the project as separable modules:

```text
modality encoders/codecs
        ↓
shared latent signal field
        ↓
signal world processor
        ↓
proposal / prediction / uncertainty heads
        ↓
policy / commit gate
        ↓
selected output decoders
```

Expected modules:

- `TextSignalAdapter`
- `AudioSignalAdapter`
- `VisualSignalAdapter`
- `LatentSignalField`
- `SignalWorldProcessor`
- `SpectralMixer`
- `LongConv` or `SSMBlock`
- `LatentPredictionHead`
- `ReconstructionHead`
- `UncertaintyHead`
- `PolicyCommitGate`
- `TextDecoderHead`
- `AudioDecoderHead`
- `VisualDecoderHead`
- `ActionHead` if action experiments are enabled
- `NoOpHead`

The core operation should be described as:

```text
Z_context → Z_processed
```

Training heads may then optimize:

```text
Z_processed → hidden latent reconstruction
Z_processed → future latent prediction
Z_processed → cross-modal alignment
Z_processed → uncertainty/source estimates
```

Inference heads may then produce:

```text
Z_processed → text/audio/visual/action/no-op/internal-only behavior
```

---

## 4. Model Scale Contract

Target scale:

```yaml
name: SLWM-124M
reference_baseline: GPT-2 small
approx_params: 124M
initial_context_length: 1024
minimum_modalities:
  - english_text_and_code
  - audio
  - visual_or_video
```

Maintain two parameter accounting modes:

1. **Strict comparison**: adapters + processor + policy + heads ≈ 124M.
2. **Core comparison**: processor ≈ 124M, adapters/decoders counted separately.

Every experiment must report which accounting mode was used.

---

## 5. Memory Policy

The initial project uses **working memory**, not unsafe online weight updates.

Definitions:

- **Working memory**: current full latent context field.
- **Episodic memory**: optional external latent memory store, retrieved into context.
- **Semantic/procedural memory**: trained model weights.
- **Adaptive memory**: future LoRA/adapters or continual-learning modules.

Rules:

- Do not implement continual learning into main weights in the initial phase.
- Do not update model weights during normal inference.
- If persistent memory is required, use external latent memory retrieval first.
- Continual learning belongs to a later, isolated research phase with rollback, replay, forgetting tests, and privacy controls.

---

## 6. Required Modalities

Minimum supported modalities:

- **Text/code**: English-focused natural language and code.
- **Audio**: speech and general audio, preferably compressed into learned audio latents.
- **Visual/video**: images and short video clips, preferably patch or video-latent encoded.

Optional later modalities:

- sensors,
- robot trajectories,
- proprioception,
- action logs,
- tool-use traces.

Do not begin optional modalities until the required modality gates are complete.

---

## 7. Training Stages

Follow `sprint_playbook_prompts.md`. Do not merge stages unless explicitly authorized.

### Stage 1 — Infrastructure and baselines

Deliver:

- deterministic data pipeline,
- config system,
- GPT-2-size text baseline,
- vanilla multimodal baseline if feasible,
- unit tests for shapes and masks.

Gate:

- baselines train and evaluate reproducibly.

### Stage 2 — Signal adapters and latent field

Deliver:

- text/code, audio, and visual adapters,
- shared latent field construction,
- adapter reconstruction/alignment smoke tests.

Gate:

- all required modalities map into a common latent shape and back through diagnostic heads.

### Stage 3 — Signal world processor

Deliver:

- spectral/temporal processor blocks,
- latent update pass,
- future/hidden latent prediction objectives.

Gate:

- processor beats or matches controlled baselines on synthetic/audio/visual signal prediction tasks.

### Stage 4 — Multimodal training

Deliver:

- mixed-modality training loop,
- modality dropout,
- cross-modal alignment and reconstruction losses,
- uncertainty head.

Gate:

- measurable cross-modal transfer above unimodal or shuffled-pair baselines.

### Stage 5 — Policy and commitment

Deliver:

- proposal heads,
- policy/commit gate,
- no-op/silence/wait behavior,
- uncertainty-aware output selection.

Gate:

- policy improves committed-output quality or reduces unsupported outputs without collapsing to abstention.

### Stage 6 — Exploration and diagnostics

Deliver:

- latent probes,
- output-head visualizations,
- cross-modal decoding tests,
- world-view inspection reports.

Gate:

- diagnostics reveal interpretable mappings and failure modes without being treated as proof of understanding.

### Stage 7 — Final evaluation and findings

Deliver:

- full baseline comparison,
- ablation matrix,
- hallucination/unsupported-output results,
- final research report.

Gate:

- claims are supported by metrics, ablations, and failure-case analysis.

---

## 8. Evaluation Requirements

Always compare against:

- GPT-2-size decoder-only text baseline,
- vanilla multimodal Transformer baseline,
- Perceiver-style latent bottleneck baseline if feasible,
- ablated SLWM variants.

Evaluate at minimum:

### Text/code

- validation loss / perplexity,
- LAMBADA-style continuation,
- HumanEval/MBPP-style code generation if code decoder is trained.

### Audio/signal

- waveform or latent prediction error,
- spectral loss,
- phase/coherence error,
- continuation stability.

### Visual/video

- image/video latent prediction,
- retrieval/alignment metrics,
- temporal consistency,
- object/action grounding.

### Multimodal grounding

- audio-video correspondence,
- image/video question answering,
- caption grounding,
- cross-modal retrieval.

### Unsupported output / hallucination

Track:

- unsupported claim rate,
- contradiction rate,
- abstention/no-op rate,
- grounded answer accuracy,
- confidence calibration.

Never report hallucination reduction without also reporting usefulness, answer accuracy, and abstention rate.

---

## 9. Required Ablations

At minimum, run:

- no spectral mixer,
- no long-conv/SSM layer,
- no shared core,
- separate modality cores,
- no latent prediction,
- reconstruction only,
- no uncertainty head,
- no policy/commit gate,
- no no-op head,
- text-only training,
- signal-only pretraining then text fine-tuning,
- different context lengths,
- different latent bottleneck sizes.

Ablations must be planned before large training runs, not added after results are known.

---

## 10. Implementation Rules

- Keep modules small, typed, and independently replaceable.
- Add shape annotations in docstrings for every model component.
- Add unit tests for every tensor transformation that changes shape.
- Log parameter counts for every run.
- Log modality mix, dataset versions, context length, latent size, and compute budget.
- Save configs with every checkpoint.
- Make all training/eval scripts runnable from config files.
- Never compare runs with different data budgets without labeling the comparison as approximate.
- Avoid train/validation/test leakage, especially for paired multimodal data.
- Treat diagnostic decoders as inspection tools, not proof of internal understanding.

---

## 11. Repository Expectations

Recommended structure:

```text
configs/
  model/
  data/
  train/
  eval/

src/
  adapters/
  core/
  heads/
  policy/
  losses/
  data/
  eval/
  memory/
  utils/

experiments/
  baselines/
  ablations/
  multimodal/
  policy/
  exploration/

reports/
  findings/
  figures/
  tables/

scripts/
  train/
  eval/
  explore/

tests/
  unit/
  integration/
```

---

## 12. Agent Roles

### Research Agent

Responsibilities:

- maintain hypotheses,
- define baselines,
- select datasets/evals,
- write experiment briefs,
- prevent scope drift.

Must output:

- clear hypothesis,
- expected failure modes,
- required ablations,
- success gate.

### Implementation Agent

Responsibilities:

- implement only the current sprint deliverables,
- preserve modularity,
- add tests,
- avoid speculative features.

Must output:

- changed files,
- test coverage,
- shape contracts,
- known limitations.

### Training Agent

Responsibilities:

- run configured training jobs,
- track reproducibility,
- log compute and data budgets,
- stop bad runs early.

Must output:

- config hash,
- checkpoint path,
- metric curves,
- run anomalies.

### Evaluation Agent

Responsibilities:

- run frozen evaluations,
- compare against baselines,
- produce tables and failure cases,
- avoid cherry-picking.

Must output:

- metric tables,
- confidence intervals where feasible,
- qualitative failure examples,
- pass/fail against sprint KPIs.

### Exploration Agent

Responsibilities:

- inspect latent spaces and output heads,
- build probes and visualizations,
- map cross-modal behavior,
- clearly label diagnostics as exploratory.

Must output:

- what was probed,
- what was decoded,
- what was observed,
- what cannot be concluded.

---

## 13. Reporting Standard

Every experiment report must include:

- sprint ID,
- hypothesis,
- model/config hash,
- parameter count,
- datasets and sample counts,
- modality mixture,
- training objective,
- compute budget,
- baselines,
- metrics,
- ablations,
- failure cases,
- interpretation,
- next allowed step.

Use cautious language. Do not claim “thinking,” “understanding,” “worldview,” or “reduced hallucination” unless supported by specific evaluations and ablations.

---

## 14. Non-Goals

Do not:

- optimize for demos before controlled evaluation,
- build an unconstrained autonomous agent before world model, uncertainty, and commit gate are measurable,
- treat raw pixels, raw waveforms, bytes, or tokens as automatically more primitive than learned latent signals,
- collapse all modalities into text-only supervision,
- implement main-weight continual learning in the initial phase,
- decode all heads by default during inference,
- treat beautiful visualizations as evidence of grounded understanding.

---

## 15. Current Success Criteria

The project is promising if SLWM-124M shows:

1. clear advantage over GPT-2-style baselines on periodic/audio/visual signal prediction,
2. competitive but not necessarily superior text/code performance,
3. measurable cross-modal transfer,
4. lower unsupported-output rate at similar usefulness,
5. interpretable policy behavior for speak/act/wait/no-op choices,
6. diagnostic evidence that latent fields preserve useful cross-modal structure,
7. ablation evidence that the shared latent signal design matters.

The project is not successful if improvements only appear in demos, require unfair parameter/data budgets, collapse to abstention, or disappear under ablation.

## 16. Version Control Branches

research
implementation
training
evaluation
exploration
integration

Use one branch per sprint. Merge completed sprint work through a PR into integration
