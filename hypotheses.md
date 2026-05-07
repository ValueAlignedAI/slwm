# SLWM-124M Sprint R0 — Hypotheses and Falsification

**Sprint:** R0 — Hypotheses and falsification  
**Primary artifact:** `hypotheses.md`  
**Gate:** claims are measurable before implementation, training, or evaluation claims are accepted.  
**Status:** pre-experiment research specification; no results are claimed here.

---

## 0. Source and Scope Notes

This document converts the SLWM-124M project vision into falsifiable, GPT-2-scale research claims.

Read context:

- `signal_latent_world_model_research_plan.md` — canonical scientific/modelation plan available in this repository.
- `research_impl_eval_docs.md` — project documentation map and required reference/evaluation areas.
- `sprint_playbook_prompts.md` — sprint R0 scope, KPIs, and success gate.
- `exploration.md` — diagnostic probe and latent-worldview exploration protocol.
- `AGENTS.md` — architecture, modality, evaluation, ablation, and reporting constraints.

Repository note: at initial R0 drafting, the canonical research-plan file was reported missing and a similarly named modelation file was used as the effective source. The current repository contains `signal_latent_world_model_research_plan.md`; it is now treated as the canonical main research plan.

Current-phase scope:

- Required modalities only: English text/code, audio, and visual/video.
- No new modalities are introduced here.
- No implementation, training, or evaluation result is claimed here.
- Diagnostic probes are treated as inspection tools, not proof of understanding.

---

## 1. Claim Status Definitions

Every hypothesis below must be answerable after registered experiments as one of:

| Status | Meaning |
|---|---|
| `support` | The primary success threshold is met against the registered baseline(s), required controls pass, and the result is not explained by a required ablation or unfair budget difference. |
| `partial support` | Some planned metrics improve, but support is incomplete because the result is limited to pilot scale, one modality/task family, a weaker baseline, or a narrower threshold. |
| `not supported` | The failure threshold is met, improvement is negligible, the result vanishes under controls/ablations, or the model gains only by unfair parameter/data/compute budget. |
| `untested` | No registered experiment has produced the required evidence yet. Untested claims must not be used as findings. |

General evidence rules:

1. Results require an `experiment_registry.md` entry with config, dataset split, seed, parameter count, compute/data budget, checkpoint, eval script, and metrics.
2. Comparisons must state whether they use strict parameter accounting or core-only parameter accounting.
3. Runs with different data/compute budgets may be reported only as approximate comparisons.
4. Hallucination or unsupported-output improvements must report unsupported claim rate, contradiction rate, abstention/no-op rate, grounded accuracy/usefulness, and calibration.
5. Probe outputs must be tagged diagnostic/internal-only and compared against random, null, or shuffled controls.

---

## 2. Hypothesis Summary Matrix

| ID | R0 area | Short claim | Primary baseline family | Required decision |
|---|---|---|---|---|
| H-R0-1 | Latent signal prediction | Latent prediction improves signal-native continuation and missing-signal completion. | GPT-style/vanilla sequence baselines, reconstruction-only, null predictors | `support` / `partial support` / `not supported` |
| H-R0-2 | Multimodal grounding | A shared latent core improves cross-modal grounding over separate or shuffled alternatives. | Vanilla multimodal Transformer, separate modality cores, shuffled-pair controls | `support` / `partial support` / `not supported` |
| H-R0-3 | Spectral/temporal processing | Spectral + local + long-range temporal mixing improves periodic and temporal signal metrics. | No-spectral, no-longconv/SSM, vanilla Transformer, FNet/Hyena/Mamba-style where feasible | `support` / `partial support` / `not supported` |
| H-R0-4 | Policy/commitment | Uncertainty-aware commitment reduces unsupported outputs without collapsing usefulness. | Always-text, fixed-router, always-no-op, no-policy ablations | `support` / `partial support` / `not supported` |
| H-R0-5 | Exploration probes | Diagnostic probes reveal latent cross-modal structure beyond decoder priors and controls. | Random-latent, shuffled-modality, null/frozen random heads | `support` / `partial support` / `not supported` |
| G-R0-1 | Guardrail | Text/code degradation remains quantified and within an acceptable tradeoff range. | GPT-2-small-style text/code baseline | pass / fail guardrail |

---

## 3. H-R0-1 — Latent Signal Prediction

### Claim

SLWM-124M variants trained with future/hidden **latent signal prediction** will improve signal-native continuation and missing-signal completion compared with token-only or reconstruction-only baselines at matched or explicitly reported parameter/data/compute budgets.

### Source mapping

- Research plan: RQ2, H1, H3.
- `AGENTS.md`: signal/audio/visual evaluation requirements; required ablations for no latent prediction and reconstruction only.
- `sprint_playbook_prompts.md`: R0 scope item 1 and T0/T2/T3 gates.

### Metrics

Primary metrics:

- Synthetic signal continuation MSE or smooth L1.
- Spectral magnitude error.
- Phase/coherence error.
- Frequency recovery error on periodic/chirp tasks.
- Audio latent prediction error or codec-token cross-entropy where applicable.
- Video/image latent future prediction error.
- Multi-step rollout drift.

Secondary metrics:

- Missing-span reconstruction error.
- Stability: NaN/Inf rate, loss explosion rate, continuation degradation over horizon.
- Throughput and memory for compute-aware interpretation.

### Dataset / evaluation family

- Synthetic sine mixtures, chirps, phase shifts, noisy periodic signals, missing-span reconstruction.
- Audio latent continuation using log-mel or codec latents from approved audio datasets such as LibriSpeech or AudioSet subsets.
- Visual/video latent continuation using approved image/video datasets such as COCO, MSR-VTT, or VGGSound subsets.

### Baselines

- Null/random predictor and persistence predictor where applicable.
- GPT-style serialized signal baseline where signals are represented as tokens/latents.
- Vanilla Transformer sequence baseline with comparable context and parameter budget.
- Reconstruction-only SLWM variant.
- SLWM variant with latent prediction disabled.

### Required ablations

- No latent prediction.
- Reconstruction only.
- No spectral mixer.
- No long-conv/SSM component.
- Different latent bottleneck sizes for sensitivity when feasible.

### Success threshold

`support` requires all of:

1. At least **10% relative reduction** in the primary prediction error versus the strongest registered non-SLWM or ablated baseline on synthetic signal continuation.
2. At least **5% relative reduction** in the primary prediction error on at least one real latent modality family: audio or visual/video.
3. The improvement remains directionally positive under at least two random seeds or has reported confidence intervals/error bars where feasible.
4. The gain does not disappear when compared with a parameter/data/compute-matched baseline.

`partial support` may be assigned if the synthetic signal threshold is met but real audio/visual evidence is missing or below threshold, or if only one seed/pilot run exists.

### Failure threshold

`not supported` if any of the following occur:

- Improvement is **≤ 2% relative** or worse than the strongest baseline on all registered signal prediction tasks.
- Improvements appear only against null/random baselines but not against vanilla Transformer or targeted ablations.
- Reconstruction-only matches or beats latent-prediction variants within noise on the same tasks.
- Results require materially larger unreported parameter, data, or compute budgets.

### Expected interpretation

- `support`: latent prediction is a viable core objective for signal-native modeling at this scale.
- `partial support`: latent prediction may help controlled signals but needs stronger modality evidence before scaling.
- `not supported`: prioritize baseline/architecture review before multimodal scaling; do not proceed by assuming latent prediction is valuable.

---

## 4. H-R0-2 — Multimodal Grounding and Shared Core

### Claim

A **shared latent signal core** will improve cross-modal grounding and transfer compared with separate modality cores, shuffled-pair controls, or vanilla multimodal baselines.

### Source mapping

- Research plan: RQ1, RQ5, H4.
- `AGENTS.md`: required modality support, multimodal grounding evaluation requirements, no-shared-core/separate-core ablations.
- `sprint_playbook_prompts.md`: R0 scope item 2 and multimodal grounding gates.

### Metrics

Primary metrics:

- Cross-modal retrieval R@1/R@5/R@10.
- Audio-video correspondence accuracy or AUROC.
- Image/video-text retrieval score.
- Caption grounding precision/recall where annotations support it.
- Grounded QA accuracy on answerable examples.

Required negative-control metrics:

- Shuffled-pair retrieval/correspondence score.
- Modality dropout robustness.
- Unsupported cross-modal claim rate for generated/readout text.

### Dataset / evaluation family

- Image-text grounding: COCO-style image/caption subsets.
- Video-text grounding: MSR-VTT-style video/caption subsets.
- Audio-video grounding: VGGSound-style correspondence subsets.
- Speech-text alignment: LibriSpeech-style speech/transcript pairs where used.

### Baselines

- Vanilla multimodal Transformer with the same adapters or input latents where feasible.
- Separate modality cores with comparable total parameter accounting.
- Shuffled-pair and null-pair baselines.
- Perceiver-style latent bottleneck baseline if feasible.
- Frozen CLIP/ImageBind-style encoder baseline may be included as a reference, but must be labeled if parameter/data budgets are not comparable.

### Required ablations

- No shared core.
- Separate modality cores.
- No cross-modal alignment loss.
- No modality dropout.
- Text-only or single-modality training variant where relevant.

### Success threshold

`support` requires all of:

1. Shared-core SLWM improves the primary cross-modal metric by at least **5 absolute percentage points** or **10% relative** over the strongest comparable registered baseline on at least two cross-modal task families.
2. Shared-core SLWM beats shuffled-pair/null controls by a large enough margin to rule out dataset priors; target: at least **20 absolute percentage points** on binary correspondence tasks or statistically clear retrieval separation.
3. Gains remain after controlling for parameter count, modality mix, and dataset split.
4. Generated/readout grounding does not show increased unsupported claim rate relative to the best comparable baseline.

`partial support` may be assigned if one cross-modal family meets threshold but another is untested or inconclusive.

### Failure threshold

`not supported` if any of the following occur:

- Shared-core SLWM is equivalent to or worse than separate-core or vanilla multimodal baselines on registered cross-modal metrics.
- Apparent gains vanish against shuffled-pair controls.
- One modality dominates such that cross-modal transfer fails when captions/text are removed or corrupted.
- Results rely on external frozen encoders with incomparable unreported pretraining and are presented as direct SLWM superiority.

### Expected interpretation

- `support`: the shared latent field is a useful design choice for multimodal transfer.
- `partial support`: shared latent representations may help specific pairs but need tighter modality-collapse tests.
- `not supported`: prefer simpler modality-specific or baseline multimodal designs until a revised shared-core hypothesis is justified.

---

## 5. H-R0-3 — Spectral/Temporal Processing

### Claim

Combining local temporal processing, spectral mixing, and long-range convolution or SSM-style updates will improve periodic, audio, and video temporal prediction metrics compared with variants that remove those components.

### Source mapping

- Research plan: RQ3, H1, processor-core sections, ablation plan.
- `AGENTS.md`: required `SpectralMixer`, `LongConv` or `SSMBlock`, and required ablations.
- `sprint_playbook_prompts.md`: R0 scope item 3 and T0 synthetic signal gate.

### Metrics

Primary metrics:

- Spectral magnitude error.
- Phase error or phase/coherence error.
- Frequency recovery error.
- Long-horizon extrapolation error.
- Audio latent continuation error.
- Video temporal consistency / future latent error.

Secondary metrics:

- Throughput and memory overhead.
- Training stability.
- Text/code validation loss impact in text-only or joint settings.

### Dataset / evaluation family

- Controlled synthetic periodic and chirp datasets.
- Audio continuation on approved audio latent subsets.
- Video latent temporal prediction on approved visual/video subsets.
- Optional text/code pilot only as a guardrail, not as proof of spectral benefit.

### Baselines

- Same model without spectral mixer.
- Same model without long-conv/SSM component.
- Same model without local temporal mixer/filter bank.
- Vanilla Transformer sequence baseline.
- FNet/Hyena/Mamba-style baselines where feasible and clearly documented.

### Required ablations

- No spectral mixer.
- No local conv/filter bank.
- No long-conv/SSM layer.
- Spectral-only variant if feasible, to test whether the full hybrid block is needed.

### Success threshold

`support` requires all of:

1. Full spectral/temporal SLWM improves synthetic spectral or phase metrics by at least **10% relative** over no-spectral and vanilla Transformer baselines.
2. Full spectral/temporal SLWM improves at least one audio or video temporal metric by at least **5% relative** over the strongest comparable ablation.
3. The component does not impose an unexplained throughput or memory regression larger than **20%** without a corresponding quality gain.
4. Text/code guardrail metrics do not catastrophically degrade when the component is enabled in text-capable settings.

`partial support` may be assigned if controlled synthetic gains are strong but audio/video transfer is untested or below threshold.

### Failure threshold

`not supported` if any of the following occur:

- No-spectral or no-longconv/SSM ablations match or beat the full variant on registered signal metrics.
- Gains appear only on toy synthetic tasks and disappear on audio/video latent prediction.
- The component causes instability, NaN/Inf failures, or unacceptable compute overhead without clear quality gains.
- The architecture cannot isolate the component through clean ablation flags.

### Expected interpretation

- `support`: spectral/temporal modules are justified for signal-native modeling.
- `partial support`: keep the component under ablation while gathering real-modality evidence.
- `not supported`: remove or redesign the component before scaling.

---

## 6. H-R0-4 — Policy and Commitment

### Claim

A learned policy/commitment gate with uncertainty and no-op/wait options will reduce unsupported external outputs while preserving task usefulness, compared with models that always decode or use fixed routing.

### Source mapping

- Research plan: RQ4, RQ5, H5.
- `AGENTS.md`: policy-selected output heads, hallucination/unsupported-output evaluation requirements, no-policy/no-no-op/no-uncertainty ablations.
- `sprint_playbook_prompts.md`: R0 scope item 4, E2 gate, policy/no-op gates.

### Metrics

Primary metrics:

- Unsupported claim rate.
- Contradiction rate against provided source signals/context.
- Grounded answer accuracy or task success.
- Usefulness score or answer completion rate.
- Abstention/no-op rate.
- Expected calibration error or equivalent confidence calibration.

Derived metric:

- Usefulness-adjusted hallucination: unsupported claim rate divided by task success rate.

Policy-specific metrics:

- Speak/no-speak accuracy.
- Ask-vs-answer accuracy.
- False commitment rate.
- Unnecessary silence/no-op rate.

### Dataset / evaluation family

- Retrieval-grounded QA and unanswerable context QA.
- Visual object-presence and object-hallucination evaluations where a visual head is trained.
- Audio event presence/absence tests where an audio pathway is trained.
- Multimodal contexts that require answer, abstain/no-op, or ask-for-information decisions.

### Baselines

- Always-text / always-answer baseline.
- Fixed rule router.
- Always-no-op / always-abstain baseline.
- Same model without policy/commit gate.
- Same model without uncertainty head.
- Same model without no-op head.

### Required ablations

- No policy/commit gate.
- No uncertainty head.
- No no-op head.
- Forced output head.
- Fixed-router baseline.

### Success threshold

`support` requires all of:

1. Unsupported claim rate decreases by at least **15% relative** versus the strongest comparable output baseline.
2. Grounded answer accuracy/usefulness is at least **95% of the strongest comparable baseline**, or absolute degradation is no more than **3 percentage points** on the registered task.
3. Abstention/no-op rate does not explain the entire improvement; always-no-op must not win usefulness-adjusted metrics.
4. Calibration improves or remains no worse than the no-policy/no-uncertainty ablation.
5. Results include contradiction rate, unsupported claim rate, abstention/no-op rate, grounded accuracy/usefulness, and calibration.

`partial support` may be assigned if unsupported outputs drop but usefulness declines modestly, or if policy works only in supervised proxy settings.

### Failure threshold

`not supported` if any of the following occur:

- Unsupported-output reduction is entirely explained by higher refusal/no-op behavior.
- Accuracy/usefulness drops by more than **10% relative** without explicit task tradeoff justification.
- Fixed routing matches or beats the learned policy on registered metrics.
- The model confuses internal predictions or diagnostic probes with committed external outputs.

### Expected interpretation

- `support`: policy/commitment is a measurable control mechanism, not just a decoder switch.
- `partial support`: commitment behavior is promising but needs better calibration or task utility.
- `not supported`: policy training/evaluation must be redesigned before claims about unsupported-output control.

---

## 7. H-R0-5 — Exploration Probes and Latent Diagnostics

### Claim

Diagnostic probes can reveal cross-modal latent structure and failure modes beyond decoder priors, provided they are marked internal-only and compared against null, random, and shuffled controls.

### Source mapping

- `exploration.md`: diagnostic modes, controls, logging, and failure modes.
- Research plan: exploration and latent probe success criteria.
- `AGENTS.md`: diagnostic decoders are inspection tools, not proof of internal understanding.
- `sprint_playbook_prompts.md`: R0 scope item 5 and X0/X1 gates.

### Metrics

Primary metrics:

- Probe task accuracy or F1 for observable labels.
- Cross-modal retrieval R@K from probe latents.
- Cross-head consistency score.
- Unsupported diagnostic claim rate.
- Source/uncertainty tagging coverage.
- Difference from random-latent, shuffled-modality, and null probe baselines.

Required tagging metric:

- **100%** of diagnostic outputs must be marked as diagnostic/internal-only and labeled as observed, reconstructed, predicted, inferred, imagined, unknown, or unsupported where applicable.

### Dataset / evaluation family

Minimum diagnostic paths from `exploration.md`:

- Video → latent → text.
- Audio → latent → text.
- Text → latent → visual diagnostic output.
- Video → latent → audio diagnostic output.
- Latent → policy gates.

### Baselines

- Random latent probe.
- Shuffled-modality probe.
- Null probe.
- Frozen random head.
- Same architecture without shared latent core.
- Same architecture without uncertainty/source head.

### Required ablations

- No shared core.
- No uncertainty/source head.
- Shuffled input modality.
- Probe trained on frozen core versus unfrozen core where feasible.

### Success threshold

`support` requires all of:

1. Probes outperform random/null/shuffled controls by at least **10 absolute percentage points** or a statistically clear retrieval/consistency margin on at least **three** diagnostic paths.
2. Probe outputs are **100%** tagged diagnostic/internal-only.
3. Failure cases and unsupported diagnostic claims are logged, not hidden.
4. The report states what cannot be concluded from probes.

`partial support` may be assigned if fewer than three paths meet threshold or if controls are incomplete but early evidence is directionally positive.

### Failure threshold

`not supported` if any of the following occur:

- Random/shuffled/null controls perform similarly to real probes.
- Diagnostic outputs are used as committed behavior or factual claims.
- Probe quality appears to come from decoder priors rather than latent input.
- Source/uncertainty labels are missing for diagnostic outputs.

### Expected interpretation

- `support`: probes are useful scientific instruments for inspecting latent representations.
- `partial support`: probes may be useful but need stronger controls.
- `not supported`: exploration outputs should not be used to motivate claims or scaling decisions.

---

## 8. G-R0-1 — Text/Code Tradeoff Guardrail

### Guardrail claim

SLWM-124M does not need to beat GPT-2-small on pure text/code in early phases, but any degradation must be quantified and bounded so signal/multimodal gains are not purchased by total text collapse.

### Source mapping

- Research plan: H2 and success criteria.
- `AGENTS.md`: required text/code evaluations.
- `sprint_playbook_prompts.md`: T1 text/code gate.

### Metrics

- Text validation cross-entropy / perplexity.
- Code validation loss.
- LAMBADA-style continuation if trained/evaluated.
- HumanEval/MBPP pass@k only if a code decoder is trained and the eval harness is ready.

### Baselines

- GPT-2-small-style decoder-only baseline under the same tokenizer/data/split/compute budget where possible.
- SLWM text-only variant.
- SLWM no-spectral text ablation.

### Pass threshold

Guardrail passes if:

```text
SLWM text validation loss ≤ GPT-2 baseline text validation loss + 15–25%
```

The acceptable point within 15–25% must be registered before the run and tightened if scaling evidence supports it.

### Fail threshold

Guardrail fails if:

- SLWM text validation loss exceeds the registered tolerance.
- Text/code performance collapses enough to make downstream language readout unusable.
- Comparisons use different data, tokenizer, or compute budgets without being labeled approximate.

### Interpretation

- Pass: text/code remains viable as an edge modality while signal hypotheses are tested.
- Fail: record as a tradeoff or blocker; do not hide text degradation behind multimodal demos.

---

## 9. Baseline and Ablation Coverage Matrix

| Baseline / ablation | H-R0-1 | H-R0-2 | H-R0-3 | H-R0-4 | H-R0-5 | G-R0-1 |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2-small-style text decoder | conditional | conditional | guardrail only | yes for text-only output behavior | no | required |
| Vanilla multimodal Transformer | yes | required | yes | conditional | conditional | no |
| Perceiver-style latent bottleneck | conditional | conditional | no | no | conditional | no |
| Null/random predictor | required | required | required | no | required | no |
| Shuffled-pair/modal controls | no | required | no | conditional | required | no |
| No spectral mixer | required | conditional | required | no | conditional | conditional |
| No long-conv/SSM | required | conditional | required | no | no | no |
| No shared core / separate cores | no | required | no | no | required | no |
| No latent prediction / reconstruction only | required | conditional | conditional | no | no | no |
| No uncertainty head | no | conditional | no | required | required | no |
| No policy/commit gate | no | no | no | required | conditional | no |
| No no-op head / always-output | no | no | no | required | no | no |

`conditional` means required only when the experiment’s architecture/data path includes that component or when the relevant comparison is feasible at the current sprint scale.

---

## 10. R0 Acceptance Checklist

The R0 hypothesis set is acceptable only if each item is true:

- [x] Each required R0 hypothesis area has a claim.
- [x] Each hypothesis has at least one metric.
- [x] Each hypothesis has baseline(s).
- [x] Each hypothesis has required ablation(s) or controls.
- [x] Each hypothesis has success and failure thresholds.
- [x] Each hypothesis can resolve to `support`, `partial support`, or `not supported`.
- [x] Claims are separated from speculation.
- [x] Diagnostic probes are explicitly not treated as proof of understanding.
- [x] Hallucination/unsupported-output claims require usefulness, accuracy, abstention/no-op, and calibration metrics.
- [x] Text/code tradeoff is tracked as a guardrail rather than hidden.

---

## 11. Current Claim State

All hypotheses are currently `untested`.

No implementation, training, evaluation, or exploration evidence has been registered in this document. Future findings must cite experiment IDs from `experiment_registry.md` before any hypothesis status changes.
