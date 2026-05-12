# SLWM Experiment Registry Schema

**Sprint:** R0 — initial registry schema  
**Purpose:** define the minimum metadata required before any experiment, evaluation, or exploration output can support a project claim.  
**Status:** schema plus completed I1/T0/T1/T2 artifact references. Sprint T1 includes dependency-light pilot evidence and a GPT-2-BPE 124M-scale limited-step benchmark; T2 includes a generated-fixture mechanics smoke. None is converged model-quality evidence.

---

## 0. Registry Rule

No training, evaluation, or exploration result counts as project evidence unless it has a registry entry with enough information to reproduce the run or clearly state why reproduction is not possible.

At minimum, every entry must identify:

- hypothesis or guardrail ID,
- model variant and ablations,
- parameter accounting mode,
- config path/hash,
- dataset mix and split/version,
- seed(s),
- compute budget,
- checkpoint or artifact path,
- eval/probe script path/version,
- metrics,
- known failures,
- interpretation and allowed claim status.

---

## 1. Experiment ID Convention

Use stable IDs:

```text
EXP-<SPRINT>-<NNN>
```

Examples:

- `EXP-T0-001` — synthetic signal training/evaluation run.
- `EXP-T1-001` — text/code baseline run.
- `EXP-E2-001` — multimodal grounding/hallucination evaluation.
- `EXP-X0-001` — diagnostic probe run.

Status values:

```text
planned | running | completed | failed | superseded | invalidated
```

Claim status values:

```text
untested | support | partial_support | not_supported | guardrail_pass | guardrail_fail
```

---

## 2. Required YAML Schema

Each experiment entry must satisfy this shape.

```yaml
experiment_id: EXP-<SPRINT>-<NNN>
status: planned
created_at: YYYY-MM-DD
updated_at: YYYY-MM-DD

sprint:
  id: R0|R1|I0|I1|I2|I3|T0|T1|T2|T3|E0|E1|E2|X0|X1|G0
  name: "<sprint name>"
  owner_role: research|implementation|training|evaluation|exploration|integration

claim_trace:
  hypothesis_ids:
    - H-R0-1
  guardrail_ids: []
  research_questions:
    - RQ2
  expected_decision: support|partial_support|not_supported|guardrail_pass|guardrail_fail|untested

repository:
  git_commit: "<required when repo is under git>"
  working_tree_state: clean|dirty|not_a_git_repo
  code_diff_ref: "<link or patch reference, if applicable>"
  docs_read:
    - AGENTS.md
    - docs/research/signal_latent_world_model_research_plan.md
    - docs/research/research_impl_eval_docs.md
    - docs/process/sprint_playbook_prompts.md
    - docs/exploration/exploration.md

config:
  config_path: "configs/.../<config>.yaml"
  config_hash: "sha256:<hash>"
  seed: 0
  deterministic: true
  precision: fp32|fp16|bf16|mixed
  context_length: 1024
  latent_length: 1024
  latent_dim: 768

model:
  name: "<model/run name>"
  variant: gpt2_baseline|vanilla_multimodal_transformer|perceiver_baseline|slwm|slwm_ablation|null_baseline|probe
  parameter_accounting_mode: strict|core_only|approximate
  total_trainable_parameters: null
  core_trainable_parameters: null
  frozen_parameters: null
  module_parameter_counts:
    adapters: null
    processor: null
    heads: null
    policy: null
    decoders: null
  enabled_modalities:
    - text_code
    - audio
    - visual_video
  architecture_flags:
    spectral_mixer: true
    local_temporal_mixer: true
    longconv_or_ssm: true
    shared_core: true
    latent_prediction: true
    reconstruction: true
    cross_modal_alignment: false
    uncertainty_head: false
    policy_commit_gate: false
    noop_head: false

ablation:
  is_ablation: false
  ablation_of: null
  changed_variable: null
  held_constant:
    - parameter_budget
    - dataset_split
    - optimizer
    - seed

data:
  dataset_mix:
    text_code: null
    audio: null
    visual_video: null
    synthetic: null
  datasets:
    - name: "<dataset name>"
      version_or_snapshot: "<version/hash/date>"
      split: train|validation|test|custom
      sample_count: null
      tokens: null
      audio_hours: null
      video_hours: null
      license_notes: "<license/filter notes>"
      leakage_checks: "<duplicate/split contamination checks>"
  preprocessing:
    text_codec: null
    audio_codec_or_features: null
    visual_codec_or_features: null
    sample_schema_version: null

training:
  objective:
    - latent_prediction
  optimizer: null
  learning_rate_schedule: null
  batch_size: null
  total_steps: null
  train_tokens_or_samples: null
  wall_clock_time: null
  hardware: null
  total_flops_estimate: null
  checkpoint_path: null
  save_config_with_checkpoint: true
  anomalies:
    nan_or_inf: false
    loss_explosion: false
    modality_collapse: false
    notes: ""

evaluation:
  eval_script: "scripts/eval/..."
  eval_script_hash: "sha256:<hash>"
  checkpoint_path: null
  seeds:
    - 0
  decoding_or_probe_settings:
    temperature: null
    top_p: null
    max_new_tokens: null
    diagnostic_only: null
  metrics:
    primary:
      name: null
      value: null
      higher_is_better: null
      confidence_interval: null
    secondary: []
    required_bundles:
      hallucination_or_policy_claim:
        required_when_claiming_reduction: true
        unsupported_claim_rate: null
        contradiction_rate: null
        grounded_accuracy_or_usefulness: null
        abstention_or_noop_rate: null
        calibration_metric: null
  baselines_compared:
    - experiment_id: null
      name: null
      comparison_notes: null
  controls:
    random_or_null: false
    shuffled_pairs: false
    fixed_router: false
    always_noop: false
    no_policy: false

interpretation:
  result_summary: ""
  hypothesis_decision: untested
  failure_modes_observed: []
  limitations: []
  next_allowed_step: ""
  claim_language_allowed: "No claim beyond registered metrics."
```

---

## 3. Metric Blocks by Hypothesis

Use these metric names consistently where applicable.

### H-R0-1 — Latent signal prediction

```yaml
metrics:
  synthetic_mse: null
  spectral_magnitude_error: null
  phase_or_coherence_error: null
  frequency_recovery_error: null
  audio_latent_prediction_error: null
  video_latent_prediction_error: null
  multi_step_rollout_drift: null
```

### H-R0-2 — Multimodal grounding

```yaml
metrics:
  retrieval_r1: null
  retrieval_r5: null
  retrieval_r10: null
  audio_video_correspondence_accuracy: null
  audio_video_correspondence_auroc: null
  grounded_qa_accuracy: null
  caption_grounding_precision: null
  shuffled_control_score: null
```

### H-R0-3 — Spectral/temporal processing

```yaml
metrics:
  no_spectral_delta: null
  no_longconv_or_ssm_delta: null
  spectral_error_delta: null
  phase_error_delta: null
  long_horizon_error_delta: null
  throughput_delta_percent: null
  memory_delta_percent: null
```

### H-R0-4 — Policy/commitment

```yaml
metrics:
  unsupported_claim_rate: null
  contradiction_rate: null
  grounded_answer_accuracy: null
  usefulness_or_task_success: null
  abstention_or_noop_rate: null
  expected_calibration_error: null
  usefulness_adjusted_hallucination: null
  false_commitment_rate: null
  unnecessary_silence_rate: null
```

### H-R0-5 — Exploration probes

```yaml
metrics:
  probe_accuracy_or_f1: null
  probe_retrieval_r_at_k: null
  cross_head_consistency: null
  unsupported_diagnostic_claim_rate: null
  source_uncertainty_tag_coverage: null
  random_latent_control_score: null
  shuffled_modality_control_score: null
  diagnostic_outputs_tagged_percent: null
```

### G-R0-1 — Text/code guardrail

```yaml
metrics:
  text_validation_loss: null
  gpt2_baseline_text_validation_loss: null
  text_loss_relative_delta_percent: null
  code_validation_loss: null
  lambada_score: null
  humaneval_pass_at_1: null
  mbpp_pass_at_1: null
```

---

## 4. Minimal Planned Registry Skeleton

The first concrete entries should be added when their sprint starts. Initial planned placeholders:

| Experiment ID | Sprint | Purpose | Hypothesis trace | Status |
|---|---|---|---|---|
| EXP-T0-001 | T0 | Synthetic periodic signal baseline comparison | H-R0-1, H-R0-3 | planned |
| EXP-T1-001 | T1 | GPT-2-style text/code baseline | G-R0-1 | planned |
| EXP-T1-002 | T1 | SLWM text-only tradeoff run | G-R0-1, H-R0-3 | planned |
| EXP-T2-001 | T2 | Audio/visual latent prediction pilot | H-R0-1, H-R0-3 | planned |
| EXP-T2-002 | T2 | Audio-video correspondence with shuffled control | H-R0-2 | planned |
| EXP-E2-001 | E2 | Grounded QA and unsupported-output evaluation | H-R0-2, H-R0-4 | planned |
| EXP-X0-001 | X0 | Diagnostic probe harness with random/shuffled controls | H-R0-5 | planned |

These rows are planning anchors, not results.

---

## 5. Result Acceptance Checklist

Before a result can change a hypothesis status, verify:

- [ ] Experiment has an ID and complete registry entry.
- [ ] Config path and hash are recorded.
- [ ] Dataset versions, splits, and sample counts are recorded.
- [ ] Parameter counts and accounting mode are recorded.
- [ ] Baselines are referenced by experiment ID or explicitly marked pending.
- [ ] Required ablations/controls are present or the interpretation is limited to `partial_support` / `untested`.
- [ ] Seeds and deterministic settings are recorded.
- [ ] Checkpoint/artifact path is recorded.
- [ ] Metrics include confidence intervals/error bars where feasible.
- [ ] Hallucination/unsupported-output claims include unsupported rate, contradiction rate, grounded accuracy/usefulness, abstention/no-op rate, and calibration.
- [ ] Known failures and anomalies are recorded.
- [ ] Claim language is limited to what the metrics prove.

---

## 6. Current Registry State

Completed Sprint I1 implementation-smoke registry entries are stored as JSON
artifacts under `experiments/baselines/`:

| Experiment ID | Sprint | Purpose | Registry artifact | Status | Claim state |
|---|---|---|---|---|---|
| `EXP-I1-001` | I1 | GPT-2-style tiny overfit baseline smoke run | `experiments/baselines/EXP-I1-001.json` | completed | implementation readiness only; no hypothesis support |
| `EXP-I1-002` | I1 | Vanilla multimodal Transformer tiny overfit baseline smoke run | `experiments/baselines/EXP-I1-002.json` | completed | implementation readiness only; no hypothesis support |

Completed Sprint T0 synthetic-signal artifact:

| Experiment ID | Sprint | Purpose | Registry artifact | Status | Claim state |
|---|---|---|---|---|---|
| `EXP-T0-001` | T0 | Synthetic signal pilot comparison | `experiments/synthetic/t0/EXP-T0-001/registry.json` | completed | controlled synthetic signal pilot only |

Completed Sprint T1 dependency-light text/code pilot artifacts are stored under
`experiments/text/t1/`. `EXP-T1-001` through `EXP-T1-003` use the same inline
text/code split and byte fallback tokenizer for all variants. They are useful for
pipeline validation and tradeoff recording, but **not** for GPT-2-scale
language-quality claims:

| Experiment ID | Sprint | Purpose | Registry artifact | Status | Claim state |
|---|---|---|---|---|---|
| `EXP-T1-001` | T1 | GPT-2-style text/code tiny pilot baseline | `experiments/text/t1/EXP-T1-001/registry.json` | completed | baseline anchor for local pilot only |
| `EXP-T1-002` | T1 | SLWM text-only tiny pilot | `experiments/text/t1/EXP-T1-002/registry.json` | completed | text tradeoff recorded; guardrail failed in pilot vs `EXP-T1-001` |
| `EXP-T1-003` | T1 | SLWM no-spectral text-only tiny pilot | `experiments/text/t1/EXP-T1-003/registry.json` | completed | text tradeoff recorded; guardrail failed in pilot vs `EXP-T1-001` |

Completed Sprint T1 PyTorch/MPS GPT-2-BPE limited benchmark artifacts are stored
under `experiments/text/t1/EXP-T1-101` through `EXP-T1-103`. These use the same
prepared GPT-2-BPE text/code corpus, split hashes, optimizer family, seed,
sequence length, and 40,960-token train budget per model. They are useful as
124M-scale mechanics and initial guardrail evidence, but **not** converged
GPT-2-quality claims:

| Experiment ID | Sprint | Purpose | Registry artifact | Status | Claim state |
|---|---|---|---|---|---|
| `EXP-T1-101` | T1 | GPT-2-style 124M PyTorch/MPS GPT-2-BPE baseline | `experiments/text/t1/EXP-T1-101/registry.json` | completed | baseline anchor for limited GPT-2-BPE benchmark |
| `EXP-T1-102` | T1 | SLWM text-only 124M PyTorch/MPS GPT-2-BPE run | `experiments/text/t1/EXP-T1-102/registry.json` | completed | limited benchmark guardrail passed vs `EXP-T1-101`; not converged quality evidence |
| `EXP-T1-103` | T1 | SLWM no-spectral 124M PyTorch/MPS GPT-2-BPE ablation | `experiments/text/t1/EXP-T1-103/registry.json` | completed | limited benchmark guardrail passed vs `EXP-T1-101`; not converged quality evidence |

Completed Sprint T2 generated-fixture smoke artifact is stored under
`experiments/multimodal/t2/EXP-T2-901`. This run uses a project-generated
audio/visual latent fixture and is useful for pipeline validation only; it is
**not** evidence of real audio/video model quality or multimodal grounding:

| Experiment ID | Sprint | Purpose | Registry artifact | Status | Claim state |
|---|---|---|---|---|---|
| `EXP-T2-901` | T2 | Audio/visual latent preparation + tiny smoke training | `experiments/multimodal/t2/EXP-T2-901/registry.json` | completed | T2 mechanics only; hypothesis state remains untested for external audio/video data |

Core R0 hypotheses in `docs/research/hypotheses.md` remain `untested` until future
training/evaluation entries use full modality-specific data, fair baseline
comparisons, required controls, and reviewed metrics. G-R0-1 has pilot and
limited-benchmark guardrail readings, but not converged text/code quality
evidence. T2 currently has mechanics evidence only from a generated latent
fixture; external audio/video results must use curated corpora, split hashes,
and shuffled/null controls before changing H-R0-1/H-R0-2/H-R0-3 states.
