# Signal-Latent World Model at GPT-2 Scale

**Scientific modelation, implementation protocol, evaluation plan, and findings template**  
**Version:** 0.1  
**Date:** 2026-05-07  
**Working name:** **SLWM-124M** — Signal-Latent World Model, GPT-2-small scale

---

## 0. Executive Summary

This document defines a research program for testing a **GPT-2-size multimodal signal model** whose core representation is not tokens, bytes, pixels, or waveform samples, but a **shared latent signal field**. The model receives compressed or encoded signals from multiple modalities, processes them in a fixed-length latent context, updates a full latent signal field, and is trained with objectives that include future prediction, hidden-state reconstruction, multimodal alignment, and uncertainty estimation latent signal states, and only decodes into text, audio, video, or actions through explicit output heads selected by a learned policy/commitment mechanism.

The central hypothesis is:

> A fixed-context, multimodal, latent-signal world model with spectral/temporal processing and learned output commitment can learn more primitive predictive representations than a token-only decoder, while remaining testable at GPT-2-small scale.

The first goal is **not** to beat frontier multimodal systems. The first goal is to determine whether the architecture learns useful, reusable, and less hallucination-prone latent representations when compared with:

1. a GPT-2-size text decoder baseline,
2. a parameter-matched vanilla multimodal Transformer baseline,
3. a Perceiver-style latent bottleneck baseline,
4. ablated versions of the proposed model.

The project should be considered successful if SLWM-124M:

- performs competitively with GPT-2-size baselines on English text/code modeling,
- substantially outperforms token-only baselines on periodic, audio, and audio-visual signal prediction,
- improves grounding and reduces unsupported claims in multimodal factual tasks,
- produces interpretable latent probes for visual, audio, textual, and action/affordance understanding,
- shows that output selection can be learned as a policy rather than hard-coded as a decoder switch.

---

## 1. Scientific Framing

### 1.1 Motivation

Standard decoder-only language models learn by predicting the next token. This is powerful but constrains the substrate of learning to a symbolic stream. The proposed model instead treats text, code, audio, video, and sensor/action traces as **different projections of underlying signal dynamics**.

The core idea is:

```text
world signals → modality encoders/codecs → shared latent signal field → latent world processor → future/hidden latent states → policy-selected decoders
```

Text is not removed. Text becomes one **interface** to the latent state, not the substrate of thought.

### 1.2 Relation to Existing Work

This project combines ideas from several families of research:

- **GPT-2 / decoder-only language modeling:** strong text-only baseline at approximately 124M parameters.
- **Perceiver IO:** arbitrary input/output arrays mapped through a fixed-size latent bottleneck.
- **data2vec:** common self-supervised latent prediction framework across speech, vision, and language.
- **JEPA / I-JEPA / V-JEPA:** prediction in abstract embedding space instead of raw reconstruction.
- **FNet:** Fourier token mixing as a lightweight alternative to self-attention in encoder architectures.
- **Hyena / long convolution models:** long-range sequence processing through implicit filters and gating.
- **Mamba / selective state-space models:** efficient recurrent-like processing for long sequences.
- **ImageBind:** joint embedding alignment across multiple modalities.
- **VQ-VAE / EnCodec / VideoMAE:** learned compressed signal representations for audio, visual, and temporal data.

The proposed system is closest in spirit to a **latent predictive world model**, but deliberately scaled down so it can be tested against GPT-2-size baselines.

---

## 2. Core Research Questions

### RQ1 — Representation

Can a shared latent signal field support text, code, audio, and visual understanding without collapsing into modality-specific islands?

### RQ2 — Prediction

Does predicting future/hidden latent signals produce better general representations than reconstructing raw pixels/audio/text or predicting next tokens only?

### RQ3 — Frequency/temporal processing

Do spectral mixers, filter banks, long convolutions, or state-space layers improve performance on periodic and multimodal temporal data at the same parameter budget?

### RQ4 — Output commitment

Can the model learn when to speak, act, wait, or stay silent using a policy/commit gate rather than naively decoding every modality?

### RQ5 — Hallucination and grounding

Does a signal-grounded latent state reduce hallucination, especially in visual/audio/text grounded QA, compared with a text-only GPT-2-style baseline and a vanilla multimodal baseline?

---

## 3. Hypotheses

### H1 — Signal advantage

SLWM-124M will outperform GPT-2-style token decoders on periodic synthetic tasks, audio continuation, audio-visual correspondence, and video temporal prediction.

### H2 — Text tradeoff

A GPT-2-style decoder may initially outperform SLWM-124M on pure text perplexity. SLWM-124M is considered promising if it remains within a controlled degradation range while gaining multimodal and signal advantages.

Suggested initial tolerance:

```text
SLWM text validation loss ≤ GPT-2 baseline loss + 15–25%
```

This threshold should be tightened if the model scales well.

### H3 — Latent prediction beats raw reconstruction

JEPA-style latent prediction should produce more semantic and transferable representations than raw reconstruction alone, especially for video and audio-visual understanding.

### H4 — Shared core beats separate cores

A shared latent processor should outperform separate modality-specific processors on cross-modal transfer tasks, retrieval, and grounded question answering.

### H5 — Commitment reduces hallucination

A learned policy/commit gate with uncertainty access should reduce unsupported external outputs compared with models that always decode a text answer.

---

## 4. Model Overview

### 4.1 High-Level Architecture

```text
                 ┌────────────────────┐
video frames ───→│ video codec/adapter │──┐
audio waveform ─→│ audio codec/adapter │──┤
text/code ──────→│ text codec/adapter  │──┤
sensors/actions →│ sensor/action adapter│─┤
                 └────────────────────┘  │
                                          ↓
                              shared latent signal field
                                          ↓
                              signal world processor
                                          ↓
                              predicted latent field(s)
                                          ↓
                   ┌──────────────────────┴──────────────────────┐
                   ↓                                             ↓
          internal probes/decoders                      policy / commit gate
                   ↓                                             ↓
       visual/audio/text/action probes          selected external output heads
```

### 4.2 Important Distinction

The model may internally predict many things:

- future visual state,
- future audio state,
- future text/speech state,
- possible action outcomes,
- uncertainty,
- affordances.

But it should externally emit only what the policy commits to:

- speak/text,
- move/act,
- generate visual output,
- generate audio output,
- ask for information,
- wait/no-op.

Therefore:

```text
internal prediction ≠ external behavior
latent imagination ≠ committed action
uncertainty ≠ hallucination, unless decoded as fact
```

---

## 5. Proposed GPT-2-Scale Configuration

### 5.1 Parameter Budget

Two versions should be maintained.

#### Strict model-size comparison

```text
Total trainable parameters: 124M ± 5%
Includes: adapters + processor + policy + output heads
```

This is the fairest comparison to GPT-2-small.

#### Core-size comparison

```text
Core processor: ~124M parameters
Adapters/decoders: counted separately
```

This is useful when using modality-specific codecs that make strict matching impractical.

All headline results must state which budget was used.

### 5.2 Baseline GPT-2-Small Configuration

Approximate GPT-2-small configuration:

```yaml
model: GPT2-small-style
parameters: ~124M
layers: 12
hidden_size: 768
attention_heads: 12
context_length: 1024
objective: next-token cross-entropy
```

### 5.3 SLWM-124M Initial Configuration

```yaml
model: SLWM-124M
latent_length: 1024
hidden_size: 768
layers: 12
processor_type: hybrid_signal_transformer
modalities:
  - text
  - code
  - audio
  - image/video
optional_modalities:
  - action
  - IMU/sensors
precision: bf16/fp16 mixed precision
context: fixed-length latent field
training_style: self-supervised multimodal latent prediction
```

If adapters make the strict 124M budget impossible, reduce width to 640 or 704 and match the total parameter count by measurement rather than formula.

---

## 6. Modality Handling

### 6.1 General Principle

Each modality gets a dedicated **edge adapter**, but the core processor is shared.

```text
modality-specific at the edges
shared in the middle
policy-selected at output
```

### 6.2 Text and Code

Text/code should not be the core representation, but an edge codec is still necessary.

Initial practical options:

#### Option A — UTF-8 byte signal

```text
UTF-8 bytes → byte embeddings → temporal signal adapter → latent field
```

Pros:

- domain-neutral,
- avoids subword assumptions,
- handles code naturally,
- simple.

Cons:

- longer sequences,
- harder language modeling,
- may underperform GPT-2 on text initially.

#### Option B — BPE as compression interface

```text
BPE tokens → token embedding → continuous latent signal adapter → latent field
```

Pros:

- fairer GPT-2 comparison,
- more compute-efficient,
- easier evaluation.

Cons:

- less pure signal-first design,
- tokenization artifacts remain.

Recommended initial choice:

```text
Use BPE for baseline comparability, but treat BPE only as an external text codec.
Also run byte-level ablation.
```

### 6.3 Audio

Initial audio adapter choices:

1. log-mel spectrogram + convolutional adapter,
2. EnCodec discrete/continuous latent codes,
3. raw waveform convolutional frontend for small experiments.

Recommended initial path:

```text
Phase 1: log-mel spectrograms for simplicity
Phase 2: EnCodec latents for compressed signal modeling
Phase 3: compare frozen vs trainable audio codec
```

### 6.4 Images and Video

Initial visual adapter choices:

1. image patches with ViT-style projection,
2. VideoMAE-style tube patches,
3. VQ-VAE/VQGAN-style discrete visual latents,
4. frozen pretrained visual embeddings for bootstrap experiments.

Recommended initial path:

```text
Phase 1: image/video patch adapter
Phase 2: video tube adapter
Phase 3: learned/frozen visual codec comparison
```

### 6.5 Actions and Movement

Actions should be included early as a **conceptual output type**, even if real robotics is deferred.

Initial proxy action sources:

- Something-Something V2 action labels,
- Ego4D action/forecasting annotations,
- synthetic control tasks,
- simulated 2D/3D environments later.

Action is represented as:

```text
action history → action adapter → latent world state
latent world state → action proposal head → policy commit gate
```

No-op/wait must be a first-class action.

---

## 7. Shared Latent Signal Field

### 7.1 Shape

The shared field should be a fixed-size latent context:

```text
Z ∈ R[L, D]
```

where:

```text
L = latent context length, e.g. 1024
D = channel width, e.g. 768
```

Optional structured variants:

```text
Z ∈ R[T, S, D]
```

where `T` is time and `S` is spatial or object/slot structure.

### 7.2 Metadata Channels

Each latent packet should include or be conditioned on:

- modality ID,
- time position,
- source ID,
- confidence/noise estimate,
- observed vs predicted flag,
- internal vs external flag,
- uncertainty estimate,
- valid/missing flag.

This prevents confusion between:

```text
silence vs missing audio
black frame vs missing video
unknown answer vs no answer required
imagined state vs observed state
```

---

## 8. Processor Core

### 8.1 Hybrid Signal Block

Each processor block combines local, spectral, long-range, and binding operations.

```text
Z
↓
normalization
↓
local temporal convolution / filter bank
↓
spectral mixer, e.g. FFT/DCT/STFT-style filter
↓
state-space or long-convolution update
↓
sparse attention / cross-channel binding
↓
gated MLP
↓
Z'
```

### 8.2 Why Not Pure Fourier?

Pure global FFT is too blunt for language and video because it ignores localized events unless combined with windows, phase, or local filters. The proposed model should use Fourier/spectral processing as one component, not as the entire architecture.

Recommended spectral variants:

- global FFT for low-frequency/global structure,
- DCT for non-periodic sequence boundaries,
- local STFT/wavelet-like filters for localized events,
- learned complex filters with amplitude/phase preservation.

### 8.3 Activation Functions

Recommended activations by subsystem:

```text
signal adapters: sine/cosine/Fourier features, SiLU
processor core: SwiGLU / SiLU / gated tanh
spectral complex states: phase-preserving operations where possible
policy heads: sigmoid/softmax gates with entropy regularization
```

Avoid using sine activations everywhere. Periodic activations are useful near signals and implicit representations; the core still needs gating, selection, memory, and compositional binding.

---

## 9. Policy and Commitment Mechanism

### 9.1 Problem

The model should not naively decode every possible output. It must decide whether to speak, move, wait, ask, generate an image, or do nothing.

### 9.2 Behavior Proposal Layer

Each output family produces proposals:

```text
speech/text proposal
visual output proposal
audio output proposal
action/motor proposal
look/attention proposal
wait/no-op proposal
ask-for-information proposal
```

### 9.3 Policy Arbiter

The policy evaluates proposals using:

- current latent world state,
- goal/task state,
- uncertainty,
- source grounding,
- safety constraints,
- available output channels,
- predicted consequences,
- energy/latency cost.

### 9.4 Commitment Output

The policy does not output just one label. It emits a behavior contract.

Example:

```yaml
behavior_contract:
  channel: text
  intention: answer_question
  confidence: 0.78
  groundedness_required: true
  uncertainty_allowed: true
  interruptible: true
  duration_budget: short
```

For action:

```yaml
behavior_contract:
  channel: action
  intention: avoid_obstacle
  safety_margin: high
  confidence: 0.91
  interruptible: true
  no_speech: true
```

For no-op:

```yaml
behavior_contract:
  channel: none
  intention: observe_wait
  reason: insufficient_confidence
  duration_budget: short
```

### 9.5 Commitment Losses

Initial supervised objectives:

- speak vs no-speak classification,
- answer vs abstain,
- ask-for-more-information vs answer,
- act vs no-op,
- multimodal response selection.

Later objectives:

- reinforcement learning in simulated environments,
- preference learning over behavior choices,
- cost-sensitive decision-making,
- uncertainty-calibrated abstention.

---

## 10. Training Objectives

The model should not rely on one loss. Use a mixture.

### 10.1 Latent Prediction Loss

Predict target latent representations from context latent representations.

```text
L_latent = distance(predicted_target_latent, target_encoder_latent)
```

Useful distances:

- cosine distance,
- smooth L1 / MSE,
- variance/covariance regularization,
- contrastive alignment loss.

### 10.2 Future Rollout Loss

```text
Z_t → Z_{t+1} → Z_{t+2} → ... → Z_{t+k}
```

Loss should measure:

- one-step prediction,
- multi-step drift,
- uncertainty growth,
- modality-specific decode quality where available.

### 10.3 Masked Signal Completion

Mask parts of the latent signal field and predict them:

- missing audio segment,
- missing video frames,
- missing text span,
- missing code region,
- missing modality entirely.

### 10.4 Cross-Modal Alignment

Train paired representations to agree:

```text
video ↔ audio
image ↔ text
audio ↔ text
video ↔ caption
speech ↔ transcript
code ↔ docstring
```

### 10.5 Reconstruction / Edge Decode Losses

Use only at decoders, not as the whole training objective.

Text/code:

```text
cross-entropy over byte/BPE output codec
```

Audio:

```text
multi-scale STFT loss
mel loss
codec-token cross-entropy if using EnCodec-like tokens
```

Image/video:

```text
patch reconstruction loss
perceptual loss
latent-code prediction loss
```

Action:

```text
behavior cloning loss
trajectory loss
classification loss for action labels
```

### 10.6 Uncertainty and Grounding Loss

Train the model to distinguish:

```text
observed
inferred
predicted
imagined
unknown
```

Suggested losses:

- calibration loss / expected calibration error tracking,
- abstention loss on unanswerable questions,
- source-attribution/grounding loss where labels exist,
- contrastive negative examples for unsupported statements.

---

## 11. Data Plan

### 11.1 Dataset Principles

Use data that provides at least one of:

- temporal continuity,
- cross-modal alignment,
- grounding between signal and language,
- action or affordance information,
- clean evaluation labels.

Avoid starting with massive raw streams. Use compressed or preprocessed signals first.

### 11.2 Text and Code Data

Recommended sources:

| Dataset                   |                      Use | Notes                                                                                             |
| ------------------------- | -----------------------: | ------------------------------------------------------------------------------------------------- |
| FineWeb / FineWeb-Edu     | English text pretraining | High-quality English web text subset; use small curated slices first.                             |
| OpenWebText               |         GPT-2 comparison | Useful for reproducing GPT-2-style baseline.                                                      |
| The Stack / StarCoderData |                     Code | Use permissively licensed code; prefer Python, JavaScript/TypeScript, Markdown, shell, JSON/YAML. |
| RedPajama subsets         |            Text/code mix | Useful for transparent dataset composition experiments.                                           |

Initial mix:

```yaml
text_english: 70%
code: 15%
markdown_docs: 10%
math_structured_text: 5%
```

For code focus, shift to:

```yaml
text_english: 55%
code: 30%
markdown_docs: 10%
math_structured_text: 5%
```

### 11.3 Audio Data

Recommended sources:

| Dataset              |                                 Use | Notes                                                         |
| -------------------- | ----------------------------------: | ------------------------------------------------------------- |
| LibriSpeech          | English speech/transcript alignment | Clean read speech, good initial audio-text alignment.         |
| Common Voice English |                    Speech diversity | More speakers and conditions; check current access/licensing. |
| AudioSet             |          environmental audio labels | Large-scale weak labels from YouTube clips.                   |
| VGGSound             |           audio-visual sound events | Good for audio-visual correspondence.                         |

### 11.4 Visual and Video Data

Recommended sources:

| Dataset                |                                   Use | Notes                                           |
| ---------------------- | ------------------------------------: | ----------------------------------------------- |
| COCO Captions          |                  image-text grounding | Strong first image-caption benchmark.           |
| MSR-VTT                |                  video-text grounding | 10K clips, many captions, retrieval/captioning. |
| VGGSound               |                 audio-video grounding | Sound source often visually present.            |
| Something-Something V2 |                  action understanding | Fine-grained human-object action labels.        |
| Ego4D                  | egocentric world/action understanding | Useful later; more complex and heavier.         |

### 11.5 Synthetic Signal Data

Use synthetic signals before multimodal pretraining to verify architecture claims.

Tasks:

- sine continuation,
- multi-frequency decomposition,
- phase shift detection,
- chirp extrapolation,
- noisy signal denoising,
- missing-span reconstruction,
- cross-frequency coupling prediction,
- simple physical simulation traces.

These tasks test whether the spectral/temporal design actually helps.

---

## 12. Training Stages

### Stage 0 — Synthetic Signal Sanity Check

Goal:

```text
Prove the signal processor beats vanilla Transformer/GPT blocks on signal-native tasks.
```

Models:

- vanilla GPT-style decoder,
- vanilla bidirectional encoder,
- FNet-style Fourier mixer,
- Hyena/Mamba-style sequence mixer if available,
- SLWM signal block.

Success criterion:

```text
SLWM wins on phase, spectral, and long-horizon extrapolation metrics at matched parameter count.
```

### Stage 1 — Text/Code Edge Baseline

Goal:

```text
Verify that the model can process English and code through a text codec without collapsing.
```

Train:

- GPT-2 baseline,
- SLWM text-only,
- SLWM text+code.

Evaluate:

- validation loss/perplexity,
- LAMBADA,
- HumanEval/MBPP small-scale,
- TruthfulQA generation/multiple-choice,
- calibration/abstention.

### Stage 2 — Audio + Text

Goal:

```text
Learn speech/audio signal grounding.
```

Data:

- LibriSpeech,
- Common Voice English,
- AudioSet subset.

Tasks:

- speech latent prediction,
- transcript prediction through text decoder,
- audio continuation,
- audio classification,
- speech-text retrieval.

### Stage 3 — Image/Video + Text

Goal:

```text
Learn visual grounding and visual-text alignment.
```

Data:

- COCO Captions,
- MSR-VTT,
- Something-Something V2.

Tasks:

- image/video latent prediction,
- captioning,
- image/video-text retrieval,
- action classification,
- object hallucination evaluation.

### Stage 4 — Audio-Visual-Text Multimodal Training

Goal:

```text
Test shared latent world-state learning across visual, audio, and text/code.
```

Data:

- VGGSound,
- MSR-VTT,
- AudioSet subset,
- LibriSpeech/Common Voice,
- text/code corpora.

Tasks:

- video → audio prediction,
- audio → video class prediction,
- video/audio → text caption,
- text → retrieve audio/video,
- missing modality prediction.

### Stage 5 — Policy and Commitment Training

Goal:

```text
Learn when to speak, act, wait, ask, or no-op.
```

Initial proxy tasks:

- answerable vs unanswerable QA,
- grounded vs unsupported caption QA,
- action label vs no-action classification,
- ask-for-clarification vs answer,
- response suppression under uncertainty.

Later:

- simulated environments,
- behavior cloning,
- preference learning,
- reinforcement learning.

---

## 13. Baselines

### 13.1 Text Baseline

```text
GPT-2-small-style decoder, 124M parameters, trained on same text/code data.
```

### 13.2 Serialized Multimodal GPT Baseline

Encode modalities into discrete tokens/latents and serialize them into a GPT-style decoder.

```text
[AUDIO_LATENTS] [VIDEO_LATENTS] [TEXT] → next-token/next-latent prediction
```

This tests whether the shared signal field is better than simply tokenizing everything.

### 13.3 Vanilla Multimodal Transformer Encoder

Same adapters, same latent length, but no spectral/SSM/signal-specific blocks.

### 13.4 Perceiver IO Baseline

Use cross-attention into fixed latent array and query-based output decoding.

### 13.5 FNet / Fourier Mixer Baseline

Replace attention-like blocks with Fourier token mixing.

### 13.6 Hyena/Mamba-Inspired Baseline

Use long convolution or SSM sequence blocks where feasible.

---

## 14. Evaluation Suite

### 14.1 Text and Language

| Metric / Benchmark                    | Purpose                                                            |
| ------------------------------------- | ------------------------------------------------------------------ |
| validation cross-entropy / perplexity | basic language modeling quality                                    |
| LAMBADA                               | long-range discourse dependency                                    |
| MMLU subset                           | knowledge/problem-solving proxy; expect low absolute score at 124M |
| TruthfulQA                            | falsehood imitation / truthfulness                                 |
| HaluEval                              | hallucination recognition and task-specific hallucination examples |
| retrieval-grounded QA                 | source-faithfulness under provided context                         |
| unanswerable QA                       | abstention and unsupported claim rate                              |

### 14.2 Code

| Metric / Benchmark          | Purpose                      |
| --------------------------- | ---------------------------- |
| code validation loss        | code modeling quality        |
| HumanEval pass@1/pass@k     | functional Python generation |
| MBPP pass@1/pass@k          | simple Python synthesis      |
| docstring ↔ code retrieval | code/text alignment          |
| syntax error rate           | decoder correctness          |

### 14.3 Audio

| Metric / Benchmark               | Purpose                              |
| -------------------------------- | ------------------------------------ |
| audio continuation spectral loss | future signal prediction             |
| multi-scale STFT loss            | reconstruction/continuation fidelity |
| speech WER on LibriSpeech        | speech-text grounding                |
| AudioSet mAP                     | audio event recognition              |
| audio-text retrieval R@K         | cross-modal alignment                |

### 14.4 Visual / Video

| Metric / Benchmark              | Purpose                                         |
| ------------------------------- | ----------------------------------------------- |
| COCO retrieval R@1/R@5/R@10     | image-text grounding                            |
| COCO caption metrics            | visual-to-text readout                          |
| MSR-VTT retrieval R@K           | video-text grounding                            |
| Something-Something V2 accuracy | action/temporal understanding                   |
| video future latent loss        | rollout stability                               |
| POPE                            | object hallucination in vision-language outputs |
| MME                             | multimodal perception/cognition evaluation      |

### 14.5 Multimodal Signal Understanding

Custom tests should measure:

- missing modality prediction,
- audio-video correspondence,
- text-video-action alignment,
- modality dropout robustness,
- cross-modal retrieval,
- cross-modal generation consistency,
- internal latent probe consistency.

### 14.6 Policy / Commitment

Metrics:

```text
speak/no-speak accuracy
answer vs abstain accuracy
ask vs answer accuracy
unsafe-action suppression
unnecessary-output rate
unsupported-claim rate
latency/cost per committed output
```

Important derived metric:

```text
usefulness-adjusted hallucination = unsupported_claim_rate / task_success_rate
```

This prevents a useless always-silent model from looking good.

---

## 15. Hallucination Evaluation Protocol

### 15.1 Definitions

A hallucination is an external output that:

- contradicts the provided source signal,
- asserts information not present in the source when the task requires grounding,
- confuses predicted/imaged content with observed content,
- expresses uncertainty as fact.

### 15.2 Text-Only Hallucination

Use:

- TruthfulQA,
- HaluEval,
- closed-book factual QA,
- retrieval-grounded QA,
- unanswerable context QA.

Track:

```text
accuracy
truthfulness
unsupported specificity
abstention rate
calibration
```

### 15.3 Visual Hallucination

Use:

- POPE object hallucination,
- MME perception subtasks,
- COCO/VG-style object-presence QA,
- adversarial prompts asking about absent objects.

Track:

```text
object false positive rate
object false negative rate
unsupported object mentions
caption precision
caption recall
```

### 15.4 Audio Hallucination

Create tests where the model must distinguish:

- heard sound vs absent sound,
- speech content vs noise,
- visible object with no sound,
- sound with no visible source.

Track:

```text
audio event false positive rate
audio-text contradiction rate
confidence calibration
```

### 15.5 Imagined vs Observed Separation

Specific test:

```text
Input: partial video/audio context
Task A: describe what is observed
Task B: predict what may happen next
```

The model must not mix these modes.

Metrics:

```text
observed-claim precision
future-prediction plausibility
observed/future confusion rate
```

---

## 16. Ablation Plan

Run these ablations systematically:

| Ablation                          | Question                                                  |
| --------------------------------- | --------------------------------------------------------- |
| remove spectral mixer             | Are frequency operations useful?                          |
| remove local conv/filter bank     | Does local signal processing matter?                      |
| remove SSM/long convolution       | Does recurrent-like temporal memory matter?               |
| remove attention                  | Is symbolic binding still needed?                         |
| remove policy gate                | Does commitment reduce hallucination/unnecessary outputs? |
| raw reconstruction only           | Is latent prediction superior?                            |
| latent prediction only            | Is reconstruction needed for decodability?                |
| text-only training                | Does multimodal training improve grounding?               |
| no uncertainty channels           | Do uncertainty signals affect hallucination?              |
| separate modality cores           | Does shared core help transfer?                           |
| frozen codecs vs trainable codecs | Are representations bottlenecked by adapters?             |
| byte text vs BPE text             | Is tokenization helping too much?                         |

Ablations should be run at smaller scale first, then repeated on the best candidates at 124M.

---

## 17. Implementation Plan

### 17.1 Repository Structure

```text
slwm/
  configs/
    gpt2_baseline.yaml
    slwm_124m_text.yaml
    slwm_124m_audio_text.yaml
    slwm_124m_multimodal.yaml
  data/
    prepare_text.py
    prepare_code.py
    prepare_audio.py
    prepare_video.py
    pack_multimodal_samples.py
  models/
    adapters/
      text_adapter.py
      audio_adapter.py
      video_adapter.py
      action_adapter.py
    processors/
      signal_block.py
      spectral_mixer.py
      ssm_block.py
      attention_block.py
    policy/
      proposal_heads.py
      commit_gate.py
      behavior_contract.py
    decoders/
      text_decoder.py
      audio_decoder.py
      video_decoder.py
      action_decoder.py
    slwm.py
    gpt2_baseline.py
  training/
    losses.py
    schedules.py
    train.py
    eval.py
  evals/
    text_eval.py
    code_eval.py
    audio_eval.py
    vision_eval.py
    hallucination_eval.py
    policy_eval.py
  reports/
    experiment_template.md
```

### 17.2 Data Sample Format

Use a unified multimodal sample schema.

```json
{
  "sample_id": "...",
  "streams": {
    "text": { "data": "...", "start": 0.0, "end": 5.0 },
    "audio": { "path": "...", "start": 0.0, "end": 5.0 },
    "video": { "path": "...", "fps": 8, "start": 0.0, "end": 5.0 },
    "action": { "labels": ["..."], "trajectory": null }
  },
  "targets": {
    "future_text": null,
    "future_audio": null,
    "future_video": null,
    "caption": "...",
    "action_label": "...",
    "answerability": "answerable"
  },
  "metadata": {
    "dataset": "...",
    "license": "...",
    "language": "en",
    "split": "train"
  }
}
```

### 17.3 Pseudocode

```python
class SLWM(nn.Module):
    def __init__(self, adapters, processor, proposal_heads, policy, decoders):
        super().__init__()
        self.adapters = adapters
        self.processor = processor
        self.proposal_heads = proposal_heads
        self.policy = policy
        self.decoders = decoders

    def forward(self, batch, task_state, mode="train"):
        latent_packets = []

        for modality, stream in batch.streams.items():
            if stream is not None:
                z_m = self.adapters[modality](stream)
                latent_packets.append(z_m)

        z = pack_to_latent_field(latent_packets, task_state)
        z_world = self.processor(z)

        proposals = {
            name: head(z_world, task_state)
            for name, head in self.proposal_heads.items()
        }

        commitments = self.policy(z_world, proposals, task_state)

        outputs = {}
        for channel, commitment in commitments.items():
            if commitment.active:
                outputs[channel] = self.decoders[channel](z_world, commitment)

        return {
            "z_world": z_world,
            "proposals": proposals,
            "commitments": commitments,
            "outputs": outputs,
        }
```

---

## 18. Compute Plan

### 18.1 Small-Scale Debug

```yaml
parameters: 10M-30M
latent_length: 256-512
modalities: synthetic + text + small audio
hardware: 1-4 consumer/prosumer GPUs
purpose: debug architecture, losses, packing, evals
```

### 18.2 Mid-Scale Validation

```yaml
parameters: 60M-80M
latent_length: 512-1024
modalities: text/code/audio/image
purpose: ablation pruning
```

### 18.3 GPT-2-Scale Run

```yaml
parameters: 124M ± 5%
latent_length: 1024
modalities: text/code/audio/video
purpose: headline comparison
```

### 18.4 Compute Discipline

Every run must log:

- parameter count,
- training tokens/samples/hours,
- modality mix,
- optimizer settings,
- wall-clock time,
- GPU type/count,
- total FLOPs estimate where possible,
- validation curves by modality.

---

## 19. Success Criteria

### Minimum Success

SLWM-124M:

- trains stably,
- supports at least text, code, audio, and visual adapters,
- reaches nontrivial text/code performance,
- beats vanilla baselines on synthetic periodic tasks,
- shows meaningful cross-modal retrieval above chance,
- produces usable latent probes.

### Strong Success

SLWM-124M:

- comes within 15–25% of GPT-2-small text validation loss,
- beats GPT-2-style serialized multimodal baseline on audio/video prediction,
- improves grounded hallucination metrics under multimodal context,
- shows better calibration/abstention than text-only GPT-2 baseline,
- uses policy gating to reduce unnecessary or unsupported outputs.

### Very Strong Success

SLWM-124M:

- matches or beats GPT-2-small on selected text/code benchmarks,
- significantly improves multimodal grounding,
- demonstrates stable latent rollout across several future steps,
- learns useful action/affordance representations,
- shows scaling promise from 30M → 80M → 124M.

---

## 20. Expected Failure Modes

### 20.1 Modality Collapse

The model may ignore audio/video and rely on text captions.

Mitigation:

- modality dropout,
- missing-modality prediction,
- audio/video-only tasks,
- contrastive hard negatives.

### 20.2 Decoder Dominance

Strong text decoder may hallucinate from language priors rather than latent evidence.

Mitigation:

- grounding loss,
- uncertainty conditioning,
- observed-vs-imagined flags,
- answerability training,
- policy commit gate.

### 20.3 Raw Signal Overload

Raw pixels/audio may consume capacity without improving abstraction.

Mitigation:

- start with compressed learned latents,
- compare frozen vs trainable codecs,
- use bottlenecks.

### 20.4 Spectral Shortcut Failure

FFT-style mixers may help synthetic signals but not text/video semantics.

Mitigation:

- combine spectral mixers with local filters, SSMs, and attention,
- ablate each component,
- use local time-frequency windows.

### 20.5 No-Op Degeneracy

The policy may learn to abstain too often to avoid errors.

Mitigation:

- usefulness-adjusted hallucination metric,
- reward correct action/answer,
- penalize unnecessary silence.

### 20.6 Unclear Fairness Against GPT-2

Multimodal models and text-only models do not have identical input/output interfaces.

Mitigation:

- compare text-only mode separately,
- compare total parameter count and core parameter count separately,
- report compute and data differences explicitly,
- avoid claiming direct superiority unless task definitions match.

---

## 21. Experiment Tracking Template

Use this template for every experiment.

```markdown
# Experiment: <name>

## Purpose

## Hypothesis

## Model

- architecture:
- parameter count:
- latent length:
- modalities:
- adapters:
- processor blocks:
- policy enabled: yes/no

## Data

- datasets:
- modality mix:
- sample count:
- token/audio/video hours:
- preprocessing:

## Training

- optimizer:
- learning rate schedule:
- batch size:
- precision:
- hardware:
- wall-clock:
- total steps:

## Evaluation

- text:
- code:
- audio:
- visual:
- multimodal:
- hallucination:
- policy:

## Results

## Interpretation

## Failure Modes

## Next Action
```

---

## 22. Findings Report Template

```markdown
# Findings: SLWM-124M Research Cycle <N>

## Summary

## What Improved

## What Failed

## Best Model Variant

## Comparison to GPT-2 Baseline

## Comparison to Multimodal Baselines

## Hallucination/Grounding Findings

## Policy/Commitment Findings

## Signal Processing Findings

## Ablation Conclusions

## Limitations

## Recommended Next Experiments
```

---

## 23. Recommended First 10 Experiments

1. **Synthetic periodic benchmark:** GPT block vs spectral block vs SLWM block.
2. **Text-only SLWM:** BPE codec, no audio/video, compare to GPT-2-small-like baseline.
3. **Byte-vs-BPE ablation:** test whether signal-purity costs too much.
4. **Audio-only latent prediction:** LibriSpeech/log-mel or EnCodec latent continuation.
5. **Image-text grounding:** COCO retrieval with shared latent field.
6. **Video-text grounding:** MSR-VTT retrieval/captioning.
7. **Audio-video grounding:** VGGSound correspondence prediction.
8. **Multimodal missing-modality prediction:** predict audio from video, text from audio, etc.
9. **Policy/no-op training:** answerable vs unanswerable, speak vs abstain.
10. **Hallucination comparison:** TruthfulQA, HaluEval, POPE/MME where applicable.

---

## 24. Reference Sources

### Architecture and Modeling

- GPT-2: Radford et al., _Language Models are Unsupervised Multitask Learners_ — https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Perceiver IO: Jaegle et al., _Perceiver IO: A General Architecture for Structured Inputs & Outputs_ — https://arxiv.org/abs/2107.14795
- data2vec: Baevski et al., _data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language_ — https://arxiv.org/abs/2202.03555
- I-JEPA: Assran et al., _Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture_ — https://arxiv.org/abs/2301.08243
- V-JEPA 2: Meta AI, _Video Joint Embedding Predictive Architecture 2_ — https://ai.meta.com/research/vjepa/
- FNet: Lee-Thorp et al., _FNet: Mixing Tokens with Fourier Transforms_ — https://arxiv.org/abs/2105.03824
- Hyena: Poli et al., _Hyena Hierarchy: Towards Larger Convolutional Language Models_ — https://arxiv.org/abs/2302.10866
- Mamba: Gu and Dao, _Mamba: Linear-Time Sequence Modeling with Selective State Spaces_ — https://arxiv.org/abs/2312.00752
- VQ-VAE: van den Oord et al., _Neural Discrete Representation Learning_ — https://arxiv.org/abs/1711.00937
- EnCodec: Défossez et al., _High Fidelity Neural Audio Compression_ — https://arxiv.org/abs/2210.13438
- VideoMAE: Tong et al., _Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training_ — https://arxiv.org/abs/2203.12602
- ImageBind: Girdhar et al., _ImageBind: One Embedding Space To Bind Them All_ — https://arxiv.org/abs/2305.05665

### Datasets

- FineWeb — https://huggingface.co/datasets/HuggingFaceFW/fineweb
- OpenWebText — https://huggingface.co/datasets/Skylion007/openwebtext
- The Stack — https://huggingface.co/datasets/bigcode/the-stack
- StarCoderData — https://huggingface.co/datasets/bigcode/starcoderdata
- RedPajama — https://github.com/togethercomputer/RedPajama-Data
- LibriSpeech — https://www.openslr.org/12
- Common Voice — https://commonvoice.mozilla.org/
- AudioSet — https://research.google.com/audioset/
- VGGSound — https://www.robots.ox.ac.uk/~vgg/data/vggsound/
- COCO — https://cocodataset.org/
- MSR-VTT — https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/
- Something-Something V2 — https://www.qualcomm.com/developer/software/something-something-v-2-dataset
- Ego4D — https://ego4d-data.org/

### Evaluation

- TruthfulQA — https://arxiv.org/abs/2109.07958
- HaluEval — https://aclanthology.org/2023.emnlp-main.397/
- POPE — https://arxiv.org/abs/2305.10355
- MME — https://arxiv.org/abs/2306.13394
- HumanEval — https://arxiv.org/abs/2107.03374
- MBPP — https://arxiv.org/abs/2108.07732
- MMLU — https://arxiv.org/abs/2009.03300
- LAMBADA — https://arxiv.org/abs/1606.06031

---

## 25. Final Research Position

This project should not be sold as “a transformer that thinks like a brain.” That claim is too broad and not testable.

A better scientific claim is:

> A GPT-2-scale model with shared latent signal processing, multimodal predictive training, and learned output commitment can be tested as a candidate substrate for domain-general predictive representations.

The key experiment is not whether it immediately beats GPT-2 on text. The key experiment is whether it learns a latent state that is:

- predictive across modalities,
- robust under missing information,
- inspectable through probes,
- less likely to externalize unsupported predictions,
- useful for both language and non-language signal tasks.

If those properties appear at 124M scale, then the idea deserves larger-scale testing.
