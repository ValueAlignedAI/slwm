# SLWM-124M Sprint R1 — Literature-to-Design Mapping

**Sprint:** R1 — Literature-to-design mapping  
**Primary artifact:** `literature_map.md`  
**Gate:** design choices are justified by references, R0 hypotheses, and testable ablations.  
**Status:** pre-implementation research specification; no empirical result is claimed here.

---

## 0. Scope and Source Notes

This document maps prior work to concrete SLWM-124M architecture, training, evaluation, and ablation decisions. It is **not** a general literature survey. A source is included only if it affects a module, objective, baseline, evaluation, risk, or control.

Read context:

- `signal_latent_world_model_research_plan.md` — current canonical research plan in this repository.
- `hypotheses.md` — Sprint R0 falsifiable hypotheses and guardrails.
- `risks_and_assumptions.md` — Sprint R0 assumptions and risks.
- `experiment_registry.md` — Sprint R0 evidence/registry schema.
- `sprint_playbook_prompts.md` — Sprint R1 scope, deliverables, KPIs, and success gate.
- `exploration.md` — diagnostic-probe controls and source/uncertainty tagging requirements.
- `AGENTS.md` — architecture contract, modality constraints, evaluation requirements, and required ablations.

Current repository note: earlier R0 docs recorded a missing canonical research-plan filename. The current repository contains `signal_latent_world_model_research_plan.md`; this R1 document treats it as canonical.

Terminology note: any terms inherited from `exploration.md` such as “world view” or “thinks” are treated only as informal labels for diagnostic probe outputs. They are not cognition, understanding, or grounding claims without registered evaluations and controls.

R1 acceptance criteria:

- Each design choice has at least one reference or R0 hypothesis link.
- Each reference maps to a concrete module or evaluation.
- No unsupported novelty claims are made.
- Current-phase and future-phase decisions are separated.
- Every P0 module traces to either an R0 hypothesis or a literature-backed design choice.

---

## 1. Required Topic Coverage

| R1 bucket | Covered source(s) | Main SLWM decision impact |
|---|---|---|
| GPT-2 / decoder baseline | GPT-2, Hugging Face GPT-2 docs | Text/code baseline, text guardrail, BPE interface for fair first comparison. |
| Perceiver IO / latent bottleneck | Perceiver IO | Fixed latent-array baseline and query/readout comparison. |
| data2vec / JEPA latent prediction | data2vec, I-JEPA, V-JEPA | Latent prediction objectives and hidden/future latent prediction ablations. |
| FNet / Fourier mixing | FNet | Spectral mixer as an ablatable processor subpath, not a full architecture claim. |
| Hyena / long convolutions | Hyena Hierarchy | Long-convolution temporal mixer candidate and ablation. |
| Mamba / SSM | Mamba | SSMBlock candidate alternative to long convolution. |
| SIREN / periodic activations | SIREN | Fourier/sine features near signal adapters and synthetic signal tests only. |
| ImageBind / multimodal shared embeddings | ImageBind, CLIP | Cross-modal alignment/retrieval protocols and shuffled-pair controls. |
| EnCodec / audio codecs | EnCodec, wav2vec 2.0 | Audio latent/codec pathway and codec-vs-feature ablations. |
| VideoMAE or equivalent visual/video latent learning | VideoMAE, V-JEPA | Patch/tube masking and visual/video latent prediction protocols. |
| Hallucination/grounding evals | TruthfulQA, HaluEval, POPE, MME, grounded QA controls | Unsupported-output metrics, usefulness/abstention gate, visual hallucination controls. |

---

## 2. Literature-to-Design Map

| Source | Contribution relevant to SLWM | Design implication | Affected module(s) / eval(s) | Risk or limit | Required ablation / control | Phase | R0 link |
|---|---|---|---|---|---|---|---|
| **GPT-2** — Radford et al., 2019, [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Establishes decoder-only Transformer text baseline, next-token objective, 1024-token context precedent, and GPT-2-small-scale comparison target. | Implement a GPT-2-small-style text/code baseline before claiming SLWM gains. Use text validation loss/perplexity and continuation tasks as guardrails. | GPT-2 baseline, `TextSignalAdapter`, `TextDecoderHead`, text/code evals. | SLWM may lose text quality; unfair tokenizer/data/compute comparisons can invalidate conclusions. | Matched tokenizer/data/split/compute where possible; SLWM text-only; no-spectral SLWM text ablation; strict/core parameter accounting. | Current | G-R0-1, H-R0-1 |
| **Hugging Face GPT-2 docs** — [Transformers GPT-2 model docs](https://huggingface.co/docs/transformers/model_doc/gpt2) | Practical implementation reference for GPT-2 config, tokenizer, generation settings, and reproducible baseline behavior. | Use as engineering reference for baseline configuration and generation/eval settings; record exact tokenizer and decoding configuration. | Baseline configs, text evaluation harness, experiment registry. | Library defaults can silently change generation/eval behavior. | Pin config, tokenizer, seed, and generation settings in registry. | Current | G-R0-1 |
| **Perceiver IO** — Jaegle et al., 2021, [arXiv:2107.14795](https://arxiv.org/abs/2107.14795) | Maps arbitrary inputs into a fixed latent array and uses query-based outputs. | Use a Perceiver-style latent bottleneck as a baseline for the shared latent field, not as proof that SLWM’s processor is necessary. | `LatentSignalField`, packing/query heads, Perceiver-style baseline, retrieval/probe evals. | Latent bottleneck strength depends heavily on latent length and cross-attention budget. | Same adapters and comparable latent size/parameter budget; different latent bottleneck sizes; no shared SLWM processor. | Current | H-R0-2, H-R0-5 |
| **data2vec** — Baevski et al., 2022, [arXiv:2202.03555](https://arxiv.org/abs/2202.03555) | Defines a modality-general self-supervised objective that predicts contextual latent targets across speech, vision, and language. | Make future/hidden latent prediction a primary objective family, with reconstruction as auxiliary/diagnostic rather than sole target. | `LatentPredictionHead`, masked completion, latent prediction evals for text/audio/video. | Teacher/target design can cause collapse or make gains depend on pretrained encoders. | No latent prediction; reconstruction-only; modality-specific objective controls. | Current | H-R0-1, H-R0-2 |
| **I-JEPA** — Assran et al., 2023, [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) | Shows image representation learning by predicting abstract embeddings rather than reconstructing pixels. | Prefer latent visual prediction for representation claims; keep pixel/patch reconstruction as a decodability check. | `VisualSignalAdapter`, `LatentPredictionHead`, visual latent evals. | Abstract prediction can be less inspectable and may lose fine detail. | Raw/patch reconstruction only; latent prediction only; prediction+reconstruction mix. | Current | H-R0-1 |
| **V-JEPA / V-JEPA 2** — Meta AI, [project page](https://ai.meta.com/research/vjepa/) | Uses predictive embedding objectives over video context and targets, supporting temporal latent prediction. | Use video future/hidden latent prediction and temporal consistency metrics as core visual/video tests. Strong pretrained V-JEPA-like systems may be reference baselines only unless budgets are comparable. | `VisualSignalAdapter`, `SignalWorldProcessor`, video future latent loss, temporal consistency eval. | Pretrained systems can make unfair comparisons; video latent targets may hide lost details. | Frozen reference labeled separately; trainable patch/tube adapter control; shuffled temporal order. | Current reference; future stronger baseline | H-R0-1, H-R0-2 |
| **FNet** — Lee-Thorp et al., 2021, [arXiv:2105.03824](https://arxiv.org/abs/2105.03824) | Demonstrates Fourier token mixing as a lightweight alternative to attention in encoder-style models. | Include `SpectralMixer` as one ablatable subpath inside a hybrid processor, not as a pure replacement for attention/binding. | `SpectralMixer`, synthetic periodic evals, audio spectral/phase metrics. | Global FFT can miss localized events and text/video semantics. | No spectral mixer; spectral-only variant; local/windowed spectral variant; vanilla Transformer baseline. | Current | H-R0-3 |
| **Hyena Hierarchy** — Poli et al., 2023, [arXiv:2302.10866](https://arxiv.org/abs/2302.10866) | Uses implicit long convolutions and gating for long-range sequence modeling. | Provide `LongConv` as one temporal-memory option in the processor and compare against attention/SSM variants. | `LongConv`, `SignalWorldProcessor`, audio/video continuation, long-horizon rollout evals. | Implementation complexity and hardware behavior can obscure quality gains. | No long-conv; vanilla Transformer; SSM/Mamba alternative; throughput/memory logging. | Current if feasible | H-R0-3 |
| **Mamba** — Gu & Dao, 2023, [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | Selective state-space sequence model with input-dependent linear-time recurrence. | Treat `SSMBlock` as a replaceable alternative to `LongConv`, not as a mandatory first implementation if complexity is too high. | `SSMBlock`, `SignalWorldProcessor`, throughput/memory evals, temporal prediction. | Specialized kernels and implementation complexity can dominate early sprint effort; SSM alone may not bind modalities. | No SSM; long-conv alternative; attention-only/vanilla baseline. | Current if feasible; otherwise future | H-R0-3 |
| **SIREN** — Sitzmann et al., 2020, [arXiv:2006.09661](https://arxiv.org/abs/2006.09661) | Shows sinusoidal activations can represent high-frequency signals and derivatives in implicit neural representations. | Use sine/Fourier features near signal adapters and synthetic probes where justified; do not use sine activations everywhere. | Signal adapter features, synthetic signal tasks, positional/time encoding options. | Can overfit toy periodic tasks or destabilize general multimodal training. | SiLU/SwiGLU adapter baseline; Fourier features vs learned embeddings; no-sine control; real audio/video confirmation. | Current limited use | H-R0-1, H-R0-3 |
| **CLIP** — Radford et al., 2021, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | Contrastive image-text alignment and retrieval protocol. | Use retrieval R@K, hard negatives, and shuffled-pair controls for image/video-text grounding. Frozen CLIP is a reference, not a fair direct baseline unless budget-labeled. | `TextSignalAdapter`, `VisualSignalAdapter`, cross-modal alignment evals. | Massive pretraining makes direct comparison unfair; retrieval can overstate generation grounding. | Shuffled captions/images; random retrieval; no shared core; frozen CLIP labeled reference-only. | Current eval/reference | H-R0-2 |
| **ImageBind** — Girdhar et al., 2023, [arXiv:2305.05665](https://arxiv.org/abs/2305.05665) | Aligns image, text, audio, and other modalities in one embedding space, largely through image-paired supervision. | Use as a precedent for shared multimodal embedding and cross-modal retrieval/correspondence evals. Do not claim equivalent ImageBind capability. | `LatentSignalField`, `TextSignalAdapter`, `AudioSignalAdapter`, `VisualSignalAdapter`, alignment evals. | Image-hub supervision can bias modality binding; pretrained scale is incomparable. | Shuffled-pair controls; modality dropout; separate modality cores; no alignment loss; frozen reference labeled separately. | Current reference | H-R0-2, H-R0-5 |
| **EnCodec** — Défossez et al., 2022, [arXiv:2210.13438](https://arxiv.org/abs/2210.13438) | Neural audio codec producing compressed audio latents/tokens. | Start with log-mel for simplicity, then compare EnCodec-style latents for compressed audio modeling. | `AudioSignalAdapter`, audio latent prediction/reconstruction, codec-token CE or latent loss. | Codec artifacts and external pretraining may dominate; raw audio information may be lost. | Log-mel vs EnCodec; frozen vs trainable codec; small raw-waveform control if feasible. | Current simple path; future codec comparison | H-R0-1, H-R0-2 |
| **wav2vec 2.0** — Baevski et al., 2020, [arXiv:2006.11477](https://arxiv.org/abs/2006.11477) | Learns speech representations through contrastive latent prediction from raw audio. | Use as audio representation-learning precedent and optional reference for speech/text alignment, not as the core SLWM objective. | `AudioSignalAdapter`, speech-text retrieval/ASR-style evals. | Speech-specific features may not generalize to general audio; pretrained scale fairness issues. | Speech-only baseline; log-mel/codec comparison; frozen reference labeled separately. | Future/reference | H-R0-1, H-R0-2 |
| **VideoMAE** — Tong et al., 2022, [arXiv:2203.12602](https://arxiv.org/abs/2203.12602) | Masked autoencoding for video with tube masking and efficient video pretraining. | Use patch/tube masking for visual/video adapter smoke tests and reconstruction/latent-prediction comparisons. | `VisualSignalAdapter`, `ReconstructionHead`, masked video completion eval. | Reconstruction can dominate semantics; decoder quality may be mistaken for latent quality. | VideoMAE-style reconstruction-only vs JEPA latent prediction; image patches vs tube patches. | Current | H-R0-1, H-R0-3 |
| **TruthfulQA** — Lin et al., 2021, [arXiv:2109.07958](https://arxiv.org/abs/2109.07958) | Tests whether language models imitate common human falsehoods. | Use as text-only truthfulness/unsupported-output baseline with abstention and calibration metrics. | `TextDecoderHead`, `UncertaintyHead`, `PolicyCommitGate`, text hallucination eval. | Closed-book truthfulness may reflect missing knowledge, not multimodal grounding; 124M models may score low. | GPT-2 baseline; SLWM text-only; always-answer vs abstain-capable policy; no-uncertainty ablation. | Current eval | H-R0-4, G-R0-1 |
| **HaluEval** — Li et al., 2023, [ACL Anthology](https://aclanthology.org/2023.emnlp-main.397/) | Provides hallucination examples and recognition/evaluation settings. | Use for unsupported/contradictory output recognition, with usefulness and abstention/no-op reported together. | `UncertaintyHead`, `PolicyCommitGate`, hallucination eval. | Generated examples may have detectable artifacts; hallucination reduction can be refusal collapse. | Human-labeled subset where possible; no-policy/no-uncertainty; always-answer; always-no-op. | Current eval | H-R0-4 |
| **POPE** — Li et al., 2023, [arXiv:2305.10355](https://arxiv.org/abs/2305.10355) | Polling-based object-presence evaluation for visual object hallucination. | Use once visual-to-text readout exists; measure object false positives/negatives and unsupported object mentions. | `VisualSignalAdapter`, `TextDecoderHead`, visual hallucination eval. | Object-only; prompt format can matter; not proof of event grounding. | Shuffled image-question pairs; blind/text-only prior; always-yes/no; no-visual ablation. | Current after visual path | H-R0-2, H-R0-4 |
| **MME** — Fu et al., 2023, [arXiv:2306.13394](https://arxiv.org/abs/2306.13394) | Broad multimodal perception/cognition benchmark with concise instructions. | Use selected perception subtasks for visual grounding after visual adapter/readout works; broader cognition subtasks are future-phase. | Visual/text evals, policy evals. | Full suite may overreach GPT-2-scale capability and invite reasoning overclaims. | Vanilla multimodal baseline; image-shuffled/null controls; prompt-invariance checks. | Current subset; future broader | H-R0-2, H-R0-4 |
| **LAMBADA** — Paperno et al., 2016, [arXiv:1606.06031](https://arxiv.org/abs/1606.06031) | Last-word prediction benchmark requiring discourse context. | Use as text guardrail beyond validation loss/perplexity. | `TextSignalAdapter`, `TextDecoderHead`, text eval. | Text-only; does not show multimodal grounding. | GPT-2-small-style baseline; same tokenizer/split; SLWM text-only. | Current eval | G-R0-1 |
| **HumanEval** — Chen et al., 2021, [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) | Functional Python synthesis benchmark using pass@k. | Use only if a code decoder is trained; run in sandbox and record sampling budget. | `TextSignalAdapter`, `TextDecoderHead`, code eval. | Contamination and sampling-budget sensitivity; 124M may underperform. | GPT-2/code baseline; decontamination checks; fixed prompt/sampling settings. | Current if code head trained | G-R0-1 |
| **MBPP** — Austin et al., 2021, [arXiv:2108.07732](https://arxiv.org/abs/2108.07732) | Entry-level Python synthesis tasks complementing HumanEval. | Use pass@1/pass@k and syntax/error taxonomy if code decoder is trained. | Code evals. | Prompt sensitivity and contamination risk. | GPT-2/code baseline; same prompt templates; syntax/null baseline. | Current if code head trained | G-R0-1 |
| **Linear probing / diagnostic classifiers** — Alain & Bengio, 2016, [arXiv:1610.01644](https://arxiv.org/abs/1610.01644) | Probes information available in learned representations. | Use frozen-core probes for diagnostic labels/retrieval; never treat probe success as proof of understanding. | Exploration probes, `UncertaintyHead`, source maps. | Probes can learn task shortcuts or decoder priors. | Random-latent, null, shuffled-modality, frozen random head, no shared core. | Current diagnostics | H-R0-5 |
| **`exploration.md` protocol** | Defines diagnostic-only head activation, source/uncertainty labels, cross-head consistency, and controls. | All exploration outputs must be internal-only and tagged as observed/reconstructed/predicted/inferred/imagined/unknown/unsupported. | Diagnostic text/audio/visual/action probes, `PolicyCommitGate`, `UncertaintyHead`. | Beautiful decodes may be confused with grounded representations. | Policy enabled/disabled; forced head; input ablations; random/shuffled/null latent; no-uncertainty/source. | Current diagnostics | H-R0-5, H-R0-4 |

---

## 3. P0 Module Traceability

| P0 module / artifact | Literature-backed reason | R0 hypothesis / guardrail | Required test or ablation |
|---|---|---|---|
| GPT-2-small-style baseline | GPT-2; HF GPT-2 docs | G-R0-1 | Matched tokenizer/data/split/compute; text loss/perplexity; LAMBADA. |
| Vanilla multimodal Transformer baseline | Research-plan baseline requirement; Transformer family control | H-R0-1, H-R0-2, H-R0-3 | Same adapters/latents where feasible; compare signal and cross-modal metrics. |
| Perceiver-style latent baseline | Perceiver IO | H-R0-2, H-R0-5 | Same adapters and comparable latent length/parameter accounting. |
| `TextSignalAdapter` | GPT-2 baseline, BPE/token interface as edge codec | G-R0-1 | BPE vs byte ablation; text-only SLWM vs GPT-2. |
| `AudioSignalAdapter` | EnCodec, wav2vec 2.0, data2vec | H-R0-1, H-R0-2 | Log-mel vs codec latent; frozen vs trainable; audio latent prediction. |
| `VisualSignalAdapter` | VideoMAE, I-JEPA/V-JEPA, CLIP | H-R0-1, H-R0-2 | Patch vs tube; reconstruction-only vs latent prediction; shuffled frames/images. |
| `LatentSignalField` | Perceiver IO, ImageBind/CLIP shared embeddings | H-R0-2, H-R0-5 | Shared core vs separate cores; shuffled-pair retrieval controls. |
| `SignalWorldProcessor` | GPT-style Transformer baseline plus Hyena/Mamba/FNet alternatives | H-R0-1, H-R0-3 | Vanilla processor baseline; component ablations. |
| `SpectralMixer` | FNet; SIREN as signal-frequency precedent | H-R0-3 | No spectral mixer; spectral-only; real audio/video confirmation. |
| `LongConv` | Hyena | H-R0-3 | No long-conv; SSM alternative; throughput/memory logging. |
| `SSMBlock` | Mamba | H-R0-3 | No SSM; LongConv alternative; attention-only baseline. |
| `LatentPredictionHead` | data2vec, JEPA, V-JEPA | H-R0-1 | No latent prediction; reconstruction-only; latent-only vs mixed objective. |
| `ReconstructionHead` | VideoMAE, EnCodec reconstruction/codec losses | H-R0-1, H-R0-5 | Reconstruction-only vs latent prediction; diagnostic decode controls. |
| `UncertaintyHead` | TruthfulQA/HaluEval/POPE protocol needs calibration/source tags; `exploration.md` | H-R0-4, H-R0-5 | No uncertainty head; calibration ECE; source-tag coverage. |
| `PolicyCommitGate` | HaluEval/TruthfulQA/POPE metrics require answer vs abstain/no-op decisions | H-R0-4 | No policy; fixed router; always-answer; always-no-op; no-uncertainty. |
| `TextDecoderHead` | GPT-2, LAMBADA, HumanEval/MBPP, grounding evals | G-R0-1, H-R0-4 | Same decoding settings; text/code baseline; forced vs committed output. |
| `AudioDecoderHead` | EnCodec/log-mel reconstruction and audio continuation | H-R0-1, H-R0-5 | Diagnostic-only forced decode; random-latent and shuffled controls. |
| `VisualDecoderHead` | VideoMAE/I-JEPA/V-JEPA diagnostics | H-R0-1, H-R0-5 | Diagnostic-only forced decode; reconstruction vs latent prediction controls. |
| `NoOpHead` | R0 policy guardrail; abstention/no-op metrics from hallucination protocol | H-R0-4 | Always-no-op baseline; no-no-op ablation; usefulness-adjusted hallucination. |
| `ActionHead` | Project architecture allows it but current required modalities exclude embodied/action experiments | Future only; not current R1 success criterion | Do not implement now except interface placeholder if I3 explicitly requires it. |

---

## 4. Current-Phase vs Future-Phase Boundary

### Current phase: literature-backed and allowed

- GPT-2-small-style text/code baseline.
- Vanilla multimodal and Perceiver-style baselines where feasible.
- Text/code, audio, and visual/video adapters to common `Z[B,T,D]` latent fields.
- Latent prediction with reconstruction/alignment as controlled objectives.
- Spectral, local temporal, long-conv, and/or SSM processor paths only if independently ablatable.
- Log-mel or simple audio features first; EnCodec comparison later when feasible.
- Patch/tube visual/video latents and masked/latent prediction evaluations.
- Cross-modal retrieval/correspondence with shuffled/null controls.
- Unsupported-output evaluation with usefulness, accuracy, abstention/no-op, contradiction, and calibration.
- Diagnostic probes marked internal-only with random/null/shuffled controls.

### Not current phase

- Claims of general understanding, consciousness, or emergent world modeling.
- Main-weight continual learning during inference.
- Optional sensor/robot/action datasets before required modality gates.
- Raw waveform or raw video generation as a primary goal.
- Treating frozen CLIP/ImageBind/V-JEPA/wav2vec systems as fair wins without accounting for external pretraining.
- Reporting hallucination reduction without usefulness/accuracy and abstention/no-op.

### Future phase if evidence justifies it

- Larger or pretrained codec/encoder comparisons with explicit accounting.
- More advanced trainable audio/video codecs.
- Policy optimization beyond supervised answer/abstain/no-op proxy tasks.
- Action/affordance experiments after required modality gates.
- Continual/adaptive memory only in a separate phase with privacy, rollback, replay, and forgetting tests.

---

## 5. R1 Acceptance Checklist

- [x] All required R1 topic buckets are covered.
- [x] Each listed source maps to at least one module or evaluation.
- [x] Each design choice is tied to a reference and/or R0 hypothesis.
- [x] Risks and limits are recorded for every major source family.
- [x] Required ablations or controls are listed.
- [x] Current-phase and future-phase choices are separated.
- [x] No empirical success claim is made.
- [x] Diagnostic probes are not treated as proof of understanding.
- [x] Hallucination/grounding claims are framed as future measurable evaluations with usefulness and abstention gates.

---

## 6. R1 Claim State

This document justifies what to build and evaluate next. It does **not** show that any design choice works. All R0 hypotheses remain `untested` until future registered experiments produce evidence.
