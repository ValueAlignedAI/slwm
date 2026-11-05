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


## 7. Real audiovisual cycles — 2026-11-04

| Experiment IDs | Cycle | Data and budget | Status and decision |
|---|---|---|---|
| `EXP-T2-201`–`EXP-T2-209` | `experiments/multimodal/t2/real-av-20261104` | CREMA-D, 1,019 train / 210 validation / 210 test; 600×32 clips per arm; 4.47M full-model trainable parameters | Completed on RTX 3090. Audio beats matched Transformer in both seeds; paired-modality gate fails; video loses to persistence. Do not scale. |
| `EXP-T2-211`–`EXP-T2-219` | `experiments/multimodal/t2/real-av-residual-20261104` | Same train/validation/preprocessing; 210 clips from seven fresh test people; residual-to-persistence prediction; same budget | Completed on RTX 3090. Paired-modality and visual-dynamics gates fail. Do not scale or advance to T3 on this recipe. |

Each nine-arm matrix includes full SLWM, vanilla Transformer, no spectral,
no long convolution, shuffled-pair training, audio-only, video-only, and
second-seed full/vanilla. Exact counts, configurations, hashes, two-seed
comparisons, person-cluster intervals, failure cases, checkpoint paths and
source snapshots are in each cycle. All 18 best checkpoints are retained
locally and on Fedora. See `docs/research/findings.md`.

Metric correction: historical T2 code computed retrieval from already fused
audio/video representations. Those retrieval values cannot establish
independent cross-modal alignment. The new runner and the shared T2 loss
encode each modality with its other stream absent, then evaluate against the
full candidate set with shuffled-label and within-person controls. Historical
generated-fixture records remain unchanged and continue to support mechanics
only.

An explicit accounting erratum removes the unused 513,000-parameter ImageNet
classification head from the deployed frozen encoder count; original hashed
cards/registries are preserved. Exploratory temporal-mean and fixed-ridge
controls are separately labeled and do not modify preregistered decisions.


## 8. Motion alignment and X0 conditional outputs — 2026-11-04

`EXP-T2-301`–`EXP-T2-310`, under
`experiments/multimodal/t2/motion-alignment-20261104`, complete ten registered
1,600×32-window RTX 3090 runs. Exact neural sizes are 4,741,089 and 15,232,225
parameters, with training-only PCA/statistics separate and no pretrained neural
encoder. Compare full/vanilla, legacy packing, no local alignment, shuffled pairs,
two larger full/vanilla seeds and a larger no-spectral arm.

Physical-time synchronization passes on eight fresh test people: small 95.5%,
larger 93.6/94.1%, energy/motion control 36.1%, chance 14.3%. Wrong/reversed/zeroed
audio checks establish input dependence. Video→audio feature prediction passes;
audio→motion fails. Increasing size does not improve timing; generation remains
limited. These are narrow speech-domain results, not general world-model claims.

`exploration/registry.json` separately registers X0-M1 frozen recognition probes
and actual conditional media, with model/codec/source hashes. The 12-sentence
readout is closed-set recognition. A later fixed-ridge decoder diagnosis and an
external, fallible speech recognizer are explicitly exploratory and never alter
primary checkpoints, thresholds, or outputs. All 18 gallery examples, original,
codec, model-reconstruction and null controls are retained. See the current
findings and full report for all results, failures and parameter/time accounting.

## 9. Frozen-world generation and signal preservation — 2026-11-04

| IDs | Cycle | Accounting and budget | Decision |
|---|---|---|---|
| `EXP-T2-401`–`405`, `411`–`416` | T2-G1, `conditional-generation-20261104` | 548,352 trained audio / 213,699 trained visual weights; 4,741,089 frozen core where used; 1,600 updates, batch 32 audio / 16 visual | Complete. Reconstruction improves within CREMA, but core transfer degrades on RAVDESS. Time-only control explains visual improvement. Cross-modal speech and visual gates fail. |
| `EXP-T2-421`–`424` | T2-G2, `conditional-residual-20261104` | Two raw/core correction seeds; 548,352 trained weights + fixed signal skip; core separate; 1,600×32 windows each | Complete. Raw reconstruction gate passes both seeds, ~15% lower mel error CREMA / ~16% reused external vs interpolation. Core-benefit and frozen-core external gates fail. |

G1 reserves 64 new RAVDESS clips from four people; G2 explicitly reuses them
after observing G1's failure. No external fitting or selection. Every checkpoint,
optimizer, source/config/data hash, curve, per-clip metric and actual media is
retained. A separately labeled EXP-T2-415 evaluation correction disables the
shuffled-training flag at inference, retaining original metrics/media and fixed
selection. The independent recognizer uses 48 preregistered cases and never
supplies generation. All 104 tests and real-file CPU isolation checks pass.

## 10. Shared-field restoration and content conservation — 2026-11-05

| IDs | Cycle | Data / accounting | Decision |
|---|---|---|---|
| `EXP-T2-501`–`509` | T2-R1, `signal-restoration-20261105` | 1,600×32 updates each; 24 CREMA + 8 Libri windows; signal core+head 5,083,841 weights, raw controls smaller, corresponding M1 core warmstarts disclosed | Complete. Clean output/timing gates pass; repair advantage over raw-AV AND vanilla and positive-video gates fail. Field-only readout exposes a bypass incentive problem. |
| `EXP-T2-521`–`524` | T2-R2, `content-conservation-20261105` | Same data/windows/updates/warmstart, +16,448 auxiliary readout weights; .5 two seeds, zero and 2.0 controls | Complete. Both primary seeds pass content, clean-output, repair and timing guardrails. Fresh harder test-other sound-readout MSE improves ~81–83% over matched R1 full cores. This does not establish useful video-driven repair. |

New R1 holdouts: 48 Libri test-clean clips/12 people and 128 RAVDESS clips/eight
people. R2 explicitly reuses these after examining R1, and reserves 48 test-other
clips/12 different Libri people as a fresh harder condition. No external fitting
or checkpoint selection. Codec statistics remain CREMA-training-only. All best
optimizer states, last weights, curves, hashes, per-clip scores and actual media
are retained. Field-only linear readouts fit training clips only. The original
fine-rate silence diagnostic is richer than model observations; the separately
saved coarse-silence baseline corrects that comparison without replacing it.
See current findings and full reports for intervals and all failed gates.


## 11. True waveform gaps — video contribution, 2026-11-05

| IDs | Role and fixed budget | Decision |
|---|---|---|
| `EXP-T2-601`–`602` | Two audio anchors;1,600×32 windows each, frozen R2 expert521/522 | Complete; adapt to physical-gap support masks. |
| `EXP-T2-611`–`612`, `619` | Extra audio capacity;2,400×32, with619 doubled to4,800 | Complete; validation selects619 for seed61 and612 for62 before final scoring. |
| `EXP-T2-613`–`618` | PCA video, two primary CNN seeds, ungated, shuffled-pair training, no-negative training;2,400×32 each | Complete. CREMA paired video/audio gains9.30% [6.71,11.79] and9.11% [6.50,11.55]; fresh RAV effects uncertain. Full feature and speech gates fail. |

Artifacts: `experiments/multimodal/t2/video-repair-20261105/`. Config and frozen
protocol: `configs/t2/video_repair.json` and
`docs/research/video_repair_protocol_20261105.md`. Exact source/config/data
hashes, counts, pretraining references, checkpoints and selected steps accompany
every run. The primary model has7,990,547 neural weights:6,567,522 frozen
expert+anchor and1,423,025 newly trainable video weights. Audio graph-connected
capacity is within about1%; raw/PCA ablations and doubled exposure are labeled.
The fixed fitted codec/PCA values are counted separately from neural weights.
This is a small head/adaptation comparison, not a124M strict/core claim.

Data: CREMA5,061 train/210 validation/240 reused test; RAV240 train/128
repurposed validation/128 fresh-person test. Primary shifted controls use202
CREMA and128 RAV length≥50 clips; all-clip and160/480ms diagnostics are retained.
Training uses24 CREMA+8 RAV windows per update. Eleven runs total27,200
updates/870,400 positive windows and16.18 measured GPU-process minutes including
validation; five visual arms also see384,000 negative windows. Historical
R2 training is separate and disclosed. No external generator is used.

Keep all11 best optimizer checkpoints and11 final weights, plus a usable
audio/video bundle. The fixed32-case recognizer diagnostic includes448 actual
WAVs. CREMA WER improves in point estimates; the second seed interval crosses
zero. Fresh RAV WER matches audio alone and loses to silence. Wrong/shifted
video can harm either set despite the trained gate. No robust general speech
repair, calibrated rejection, learned policy or world-model claim is allowed.

The first render-only attempt failed on terminal subframe padding. Preserve
its original source snapshot/log/partial outputs. Separately hashed revision2
applies the existing edge rule and adds a short-tail waveform test; no trained
weight, selected ID, gate or feature score changes. All124 tests and trained
artifact audits pass, including6,007 raw-record hashes, frozen weights, actual
CPU file inference, hidden-waveform/target/soundtrack isolation and exact
observed PCM. See `report.md`, `failure_analysis.md`, `audit.json`,
`render_correction.json` and `delivery_manifest.json` for complete evidence.

## 12. Fresh-person content and calibrated timing — 2026-11-05

| IDs | Role / fixed budget | Decision |
|---|---|---|
| `EXP-T2-701`–`704` | Unit CE 0/.2, seeds 71/72; 4,000×32 updates; 24 GRID +4 CREMA +4 Libri windows | Complete. Both content seeds pass field-word gain and spectral/clean guardrails. Both controls have the timed-text auxiliary interface. |
| `EXP-T2-711`–`712` | Strong additional-audio candidates; 3,000×32; 24 GRID +8 CREMA | Complete. Exact frozen fallback for corresponding video candidates. |
| `EXP-T2-713`–`716` | Primary video two seeds, no-unit-CE and shuffled-pair controls; 3,000×32 plus384,000 negative windows total | Complete. Fresh-person spectral-video and calibrated ±240ms rejection/coverage gates pass both seeds. Practical speech-recognition gate fails. |

Artifacts: `experiments/multimodal/t2/content-sync-20261105/`. Frozen protocol
`docs/research/content_sync_protocol_20261105.md`; current scoring report
`report_v2.md`. GRID train12,704/16people; selection395/4; calibration278/3;
primary test749/8, with complete sentence combinations also disjoint. Only51
lexical words; no open-vocabulary claim. The strict source alarm and90 truncated
training-person sources are documented with exact exclusions and failed logs.
Historical R2 warmstarts and CREMA/Libri retention data are disclosed separately.

Bases5,133,942 weights; matched audio6,428,512 / video7,713,820 deployed totals.
Audio/video newly trained branches1,294,570 /1,285,308; extra video negatives
are additional compute. The strong audio fallback is frozen and exact, including
word/latent state. Fitted codecs are separate. Total34,000updates,
1,088,000positive windows and25.68 measured training minutes including validation.
All ten best optimizer/RNG checkpoints and final weights remain saved.

Paired field lexical-word gains9.72pp [8.21,10.97] /12.43pp [10.95,13.92].
Calibrated video spectral gains12.26% [9.65,14.86] /11.24% [8.24,14.21], with
aligned coverage81.84% /83.85% and each ±240ms admission rate below1%.
Wrong-utterance rejection remains incomplete (~3.5% spectral harm). WER is mixed
and fails the practical gate. Revision2 excludes19 short-pause frames from word
scoring using the original fitted probes; all original results/sources retained,
with no weight, threshold, waveform or spectral change. Audits verify14,636 raw
records and640 actual WAVs with exact observed PCM. Full suite133 passes.

## 13. Early GPT-2-scale text comparison — 2026-11-05

| IDs | Variant / exposure | Decision |
|---|---|---|
| `EXP-T1-201`, `203` | GPT-2-style124,439,808 weights, seeds81/82,16,777,216 tokens each | Complete. Random initialization; validation selection and frozen test evaluation. |
| `EXP-T1-202`, `204` | Causal SLWM125,131,008 weights, paired initialization/windows/exposure | Complete. Early loss guardrail passes both seeds; ~3% lower NLL at equal tokens with ~40% more time/token. Not converged superiority. |

Artifacts: `experiments/text/t1/scale-20261105/`; protocol
`docs/research/text_scale_protocol_20261105.md`. Prepared pinned FineWeb-Edu
corpus2.600B train /14.57M validation /10.01M test tokens, with disjoint registrable
domains and global exact/prefix/URL deduplication. No guaranteed near-duplicate
freedom. Available20.9tokens/parameter differs from actual0.13seen/parameter.
Each run1024updates, global32×512; total67,108,864 training tokens and38.59
training minutes including validation. Both-seed optimizer/RNG and final weights,
fixed raw continuations, compute diagnostic and all data/source hashes retained.
These are separate text-only weights; no jointly trained124M multimodal claim.
