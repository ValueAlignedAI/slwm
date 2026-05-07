# research_impl_eval_docs.md

# SLWM-124M Project Documentation Map

**Purpose:** define every document that a research, implementation, or evaluation agent should read or maintain for the **Signal-Latent World Model** project.

This file is a **navigation and responsibility map**. It does not replace the main research plan. It tells agents what to read, what to create, and which external references are required for grounded implementation and evaluation.

---

## 0. Reading Order

Every agent should read these first, in order:

1. [`AGENTS.md`](./AGENTS.md)  
   Agent behavior, coding style, repository expectations, research constraints.

2. [`signal_latent_world_model_research_plan.md`](./signal_latent_world_model_research_plan.md)  
   Main scientific specification: hypothesis, architecture, training, baselines, evaluations, and findings template.

3. [`exploration.md`](./exploration.md)  
   Diagnostic exploration of output heads, latent probes, cross-modal mappings, and the model's latent world view.

4. `research_impl_eval_docs.md`  
   This file. Defines the full documentation set and external reading list.

After reading these, agents should follow the role-specific tracks below.

---

## 1. Existing Project Documents

| Document | Status | Owner | Purpose |
|---|---:|---|---|
| `AGENTS.md` | Exists | All agents | Operating instructions for implementation/research/eval agents. |
| `signal_latent_world_model_research_plan.md` | Exists | Research lead | Main scientific modelation and project plan. |
| `exploration.md` | Exists | Research + eval | Explore output heads and latent world-view diagnostics. |
| `research_impl_eval_docs.md` | Exists | Documentation agent | Canonical doc map and reading list. |

---

## 2. Documents to Create Next

These should become repository-level `.md` files. Keep each file focused. Do not let one document become the entire project.

### 2.1 Implementation Documents

| Document | Priority | Purpose |
|---|---:|---|
| `architecture.md` | P0 | Concrete architecture: encoders, shared latent field, processor blocks, policy/commitment, heads. |
| `model_spec.md` | P0 | Exact GPT-2-scale model configs, tensor shapes, parameter budget, block definitions. |
| `data_contract.md` | P0 | Defines the unified signal sample format: text/code, audio, image/video, masks, timestamps, modality IDs, uncertainty channels. |
| `preprocessing.md` | P0 | How each modality becomes compressed/encoded latent signals. Includes audio/video/text/code pipelines. |
| `training.md` | P0 | Losses, batch construction, curriculum, optimizer, checkpoints, mixed precision, distributed training. |
| `inference.md` | P0 | No-grad inference modes: perception update, latent rollout, committed output, diagnostic probing. |
| `policy_commitment.md` | P0 | Learned policy/arbiter that decides speak/act/wait/decode/no-op. Separates imagination from committed behavior. |
| `memory.md` | P1 | Working memory, external episodic memory, retrieval into context, later adapter/continual-learning phases. |
| `baselines.md` | P0 | GPT-2 decoder, vanilla multimodal transformer, FNet-like, Hyena-like, Mamba-like, ablated SLWM variants. |
| `ablations.md` | P0 | Required ablations: no spectral mixer, no policy, no cross-modal loss, no uncertainty, no signal pretraining. |
| `engineering.md` | P1 | Repo layout, config system, logging, determinism, experiment tracking, failure handling. |

### 2.2 Research Documents

| Document | Priority | Purpose |
|---|---:|---|
| `hypotheses.md` | P0 | Testable claims and falsification criteria. |
| `literature_map.md` | P0 | Summary of related work and why each paper matters. |
| `research_questions.md` | P0 | Open questions: latent signal universality, policy selection, hallucination, transfer, memory. |
| `experiment_registry.md` | P0 | Every experiment ID, config, dataset mix, seed, checkpoint, metric, and result link. |
| `findings.md` | P0 | Running scientific findings, negative results, interpretation, and next decisions. |
| `risks_and_assumptions.md` | P1 | Technical risks, naive assumptions, data risks, eval risks, safety risks. |
| `phase_plan.md` | P0 | Phase 0 prototype → Phase 1 GPT-2-scale text/audio/visual → Phase 2 memory/action. |

### 2.3 Evaluation Documents

| Document | Priority | Purpose |
|---|---:|---|
| `evals.md` | P0 | Master evaluation protocol. Defines required metrics and pass/fail gates. |
| `text_code_evals.md` | P0 | Perplexity, LAMBADA, MMLU subset, HumanEval, MBPP, code perplexity. |
| `signal_evals.md` | P0 | Synthetic periodic signals, audio continuation, video latent prediction, reconstruction. |
| `multimodal_evals.md` | P0 | Image/video-text retrieval, audio-video alignment, text-to-visual latent probing, visual QA. |
| `hallucination_evals.md` | P0 | TruthfulQA, HaluEval, POPE, context-grounded QA, unanswerable questions, source-grounding. |
| `policy_evals.md` | P0 | Speak/move/wait/no-op decision accuracy, over-output rate, unsafe commit rate, uncertainty-aware abstention. |
| `exploration_evals.md` | P1 | Diagnostic head activation, latent probes, cross-modal mapping maps, worldview visualizations. |
| `eval_report_template.md` | P0 | Standard report format for comparing models and runs. |

---

## 3. Role-Specific Reading Tracks

### 3.1 Implementation Agent Track

Read in this order:

1. `AGENTS.md`
2. `signal_latent_world_model_research_plan.md`
3. `architecture.md`
4. `model_spec.md`
5. `data_contract.md`
6. `preprocessing.md`
7. `training.md`
8. `inference.md`
9. `policy_commitment.md`
10. `baselines.md`
11. `engineering.md`

Implementation agents should not change the research hypothesis or evaluation criteria casually. If implementation constraints force a design change, record it in `experiment_registry.md` and `findings.md`.

### 3.2 Research Agent Track

Read in this order:

1. `signal_latent_world_model_research_plan.md`
2. `exploration.md`
3. `hypotheses.md`
4. `literature_map.md`
5. `research_questions.md`
6. `phase_plan.md`
7. `risks_and_assumptions.md`
8. `experiment_registry.md`
9. `findings.md`
10. All evaluation docs relevant to the current phase.

Research agents must convert vague claims into measurable hypotheses.

### 3.3 Evaluation Agent Track

Read in this order:

1. `signal_latent_world_model_research_plan.md`
2. `evals.md`
3. `hallucination_evals.md`
4. `text_code_evals.md`
5. `signal_evals.md`
6. `multimodal_evals.md`
7. `policy_evals.md`
8. `exploration_evals.md`
9. `baselines.md`
10. `eval_report_template.md`

Evaluation agents must be stricter than implementation agents. Do not accept a model improvement unless the baseline, dataset split, random seeds, and decoding settings are documented.

---

## 4. Required External Reading: Architecture and Modeling

These sources define the architecture space SLWM-124M is exploring.

| Topic | Source | Why agents should read it |
|---|---|---|
| GPT-2 baseline | Radford et al., **Language Models are Unsupervised Multitask Learners** — https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | Defines the token-only decoder baseline and GPT-2 framing. |
| GPT-2 implementation reference | Hugging Face GPT-2 docs — https://huggingface.co/docs/transformers/model_doc/gpt2 | Practical reference for config, tokenizer, generation, baseline loading. |
| Arbitrary input/output latent processor | **Perceiver IO** — https://arxiv.org/abs/2107.14795 | Important because SLWM uses shared latents and output queries/heads. |
| Cross-modal latent self-supervision | **data2vec** — https://arxiv.org/abs/2202.03555 | Predicting contextual latent representations across speech, vision, language. |
| Latent video world model | **V-JEPA / V-JEPA 2** — https://ai.meta.com/research/vjepa/ and https://arxiv.org/abs/2506.09985 | Central reference for latent-space prediction instead of pixel reconstruction. |
| Shared multimodal embedding | **ImageBind** — https://arxiv.org/abs/2305.05665 | Shows binding images, text, audio, depth, thermal, and IMU in one embedding space. |
| Fourier token mixing | **FNet** — https://arxiv.org/abs/2105.03824 | Reference for replacing attention-like mixing with Fourier transforms. |
| Long convolution / gating | **Hyena Hierarchy** — https://arxiv.org/abs/2302.10866 | Reference for long-range signal/sequence mixing without dense attention. |
| State-space backbone | **Mamba** — https://arxiv.org/abs/2312.00752 | Reference for selective state-space sequence modeling and long contexts. |
| Periodic activations | **SIREN** — https://arxiv.org/abs/2006.09661 | Reference for sinusoidal activations and continuous signal representations. |
| Audio neural codec | **EnCodec** — https://arxiv.org/abs/2210.13438 | Reference for compressed audio latents and codec-style modality adapters. |
| Speech representation learning | **wav2vec 2.0** — https://arxiv.org/abs/2006.11477 | Reference for latent speech modeling from raw audio. |
| Video masked modeling | **VideoMAE** — https://arxiv.org/abs/2203.12602 | Reference for masked video representation learning. |
| Vision-language contrastive grounding | **CLIP** — https://arxiv.org/abs/2103.00020 | Reference for image-text alignment and zero-shot transfer. |

---

## 5. Required External Reading: Datasets

Use these as candidates. Dataset use must be checked for license, availability, contamination risk, and storage/compute requirements before training.

| Modality | Dataset | Source | Use |
|---|---|---|---|
| English text | FineWeb | https://huggingface.co/datasets/HuggingFaceFW/fineweb | English web text pretraining candidate. |
| Code | The Stack v2 | https://huggingface.co/datasets/bigcode/the-stack-v2 | Code pretraining candidate; use license-aware filters. |
| English speech | LibriSpeech | https://www.openslr.org/12 | Speech/audio-text alignment and ASR-style evaluation. |
| Audio events | AudioSet | https://research.google.com/audioset/ | General audio event signals. |
| Audio-video | VGGSound | https://www.robots.ox.ac.uk/~vgg/data/vggsound/ | Audio-visual correspondence and grounding. |
| Images/captions | COCO | https://cocodataset.org/ | Image-caption grounding, object/caption evaluation. |
| Video/text | MSR-VTT | https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/ | Video-caption alignment and retrieval. |
| Egocentric video | Ego4D | https://ego4d-data.org/ | Later-stage world/action/first-person perception research. |

Minimum Phase 1 dataset mix:

```text
text/code: FineWeb subset + The Stack v2 filtered subset
audio: LibriSpeech + small AudioSet subset
visual: COCO + small VGGSound or MSR-VTT subset
synthetic: generated periodic/multifrequency/chirp/phase tasks
```

Do not start with all large datasets at full scale. Start with small curated subsets that fit reproducible iteration.

---

## 6. Required External Reading: Evaluations

| Evaluation area | Source | Why it matters |
|---|---|---|
| Truthfulness | TruthfulQA — https://arxiv.org/abs/2109.07958 | Measures whether models reproduce common falsehoods. |
| Text hallucination | HaluEval — https://aclanthology.org/2023.emnlp-main.397/ | Hallucination recognition and generated/human-annotated hallucinated samples. |
| Visual object hallucination | POPE — https://arxiv.org/abs/2305.10355 | Object hallucination evaluation for vision-language outputs. |
| Multimodal perception/cognition | MME — https://arxiv.org/abs/2306.13394 | Broad MLLM evaluation across perception and cognition subtasks. |
| Code generation | HumanEval — https://github.com/openai/human-eval | Standard code generation benchmark harness. |
| Basic Python coding | MBPP — https://arxiv.org/abs/2108.07732 | Entry-level Python synthesis benchmark. |
| General knowledge/problem solving | MMLU — https://arxiv.org/abs/2009.03300 | Multitask text understanding benchmark across 57 subjects. |
| Broad context language modeling | LAMBADA — https://arxiv.org/abs/1606.06031 | Long-context final-word prediction requiring discourse understanding. |

SLWM-specific evaluations must add signal-native metrics:

```text
latent reconstruction error
future latent prediction error
spectral magnitude error
phase/coherence error
cross-modal retrieval accuracy
unsupported-claim rate
uncertainty calibration
policy over-output rate
commitment suppression accuracy
```

---

## 7. Required External Reading: Engineering and Experimentation

| Topic | Source | Use |
|---|---|---|
| PyTorch distributed training | https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html | Multi-GPU training basics. |
| PyTorch DDP notes | https://docs.pytorch.org/docs/stable/notes/ddp.html | Deeper details for distributed implementation. |
| Hugging Face Transformers | https://huggingface.co/docs/transformers/index | Token baselines, GPT-2 tokenizer/model utilities. |
| W&B sweeps | https://docs.wandb.ai/models/sweeps | Experiment tracking and hyperparameter sweeps. |

These are not mandatory if the implementation chooses another stack, but the chosen alternatives must be documented in `engineering.md`.

---

## 8. Documentation Rules for Agents

### 8.1 Every experiment must be traceable

Each experiment entry should include:

```yaml
experiment_id:
model_variant:
commit_hash:
config_path:
dataset_mix:
dataset_versions:
train_tokens_or_samples:
seeds:
compute:
checkpoint_paths:
metrics:
known_failures:
interpretation:
next_action:
```

### 8.2 Every architecture change needs an ablation plan

If an agent adds a component, also define how to remove it:

```text
component added: spectral mixer
ablation: same model without spectral mixer
comparison: text PPL, synthetic signal error, audio latent prediction, throughput
```

### 8.3 Every output head needs a diagnostic probe

For each output head:

```text
head: text / audio / visual / action / no-op
normal use: policy-selected committed behavior
diagnostic use: forced probe from latent field
metrics: reconstruction/prediction quality + unsupported output rate
failure modes: overcommit, hallucination, modality confusion, instability
```

### 8.4 Separate internal prediction from external behavior

Never document internal predictions as external actions unless the policy committed them.

Use this distinction:

```text
latent imagination: uncommitted internal rollout
proposal: candidate behavior
action: committed external behavior
no-op: valid committed behavior
```

---

## 9. Minimal Repository Documentation Layout

Recommended final layout:

```text
/
  AGENTS.md
  README.md
  research_impl_eval_docs.md

/docs
  signal_latent_world_model_research_plan.md
  architecture.md
  model_spec.md
  data_contract.md
  preprocessing.md
  training.md
  inference.md
  policy_commitment.md
  memory.md
  exploration.md
  baselines.md
  ablations.md
  engineering.md

/docs/research
  hypotheses.md
  literature_map.md
  research_questions.md
  risks_and_assumptions.md
  phase_plan.md
  experiment_registry.md
  findings.md

/docs/evals
  evals.md
  text_code_evals.md
  signal_evals.md
  multimodal_evals.md
  hallucination_evals.md
  policy_evals.md
  exploration_evals.md
  eval_report_template.md
```

---

## 10. Immediate Next Documentation Tasks

1. Create `architecture.md` from the current research plan.
2. Create `data_contract.md` before writing dataset loaders.
3. Create `model_spec.md` before implementing model classes.
4. Create `evals.md` before claiming any improvement.
5. Create `experiment_registry.md` before running the first training job.
6. Create `findings.md` and update it after every meaningful result, including failed runs.

---

## 11. Definition of Done for Documentation

A documentation set is good enough for Phase 1 when a new implementation agent can answer:

```text
What is the model?
What tensors does it consume and emit?
Which outputs are internal predictions versus committed behavior?
Which datasets are used and why?
Which baselines are required?
Which metrics decide success or failure?
How are hallucinations measured?
How are policy/no-op decisions evaluated?
How are experiments reproduced?
Where are findings recorded?
```

If any answer is unclear, add or revise a doc before scaling training.
