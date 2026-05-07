# SLWM-124M Sprint R0 — Risks and Assumptions

**Sprint:** R0 — Hypotheses and falsification  
**Purpose:** record the assumptions and risks that could invalidate the R0 hypotheses or make later results misleading.  
**Status:** pre-experiment; no empirical claims are made here.

---

## 0. Source and Scope Notes

This file supports `hypotheses.md` and should be updated whenever a sprint discovers a new blocker, invalid assumption, or result that changes the project risk profile.

Repository note: at initial R0 drafting, `signal_latent_world_model_research_plan.md` was reported missing and a similarly named modelation file was used as the effective source. The current repository contains `signal_latent_world_model_research_plan.md`; it is now treated as the canonical main research plan. No `docs/rules/*.md` or `CONTRIBUTING.md` files were present at the time this R0/R1 documentation was drafted.

R0 scope limits:

- Documentation/research only.
- Required modalities only: English text/code, audio, visual/video.
- No new architecture, datasets, or evaluation suites beyond what the current project docs already identify.
- All claims remain untested until linked to registered experiments.

---

## 1. Core Assumptions

| ID | Assumption | Why it matters | If false |
|---|---|---|---|
| A-R0-1 | `signal_latent_world_model_research_plan.md` is the effective main research plan. | R0/R1 need a canonical source for RQ/H mappings. | If the canonical file diverges from prior notes, revise traceability links before implementation. |
| A-R0-2 | GPT-2-scale experiments are sufficient to reject or support early architecture claims. | The project is scoped around SLWM-124M and comparable baselines. | Mark claims as pilot-only or future-phase; avoid scaling claims. |
| A-R0-3 | Required modalities can be converted to common latent fields without unacceptable information loss. | Shared-core hypotheses depend on comparable latent representations. | Revisit adapter/data-contract design before interpreting shared-core failures. |
| A-R0-4 | Fair baseline comparisons are feasible with logged parameter, data, and compute budgets. | Most hypotheses depend on matched or clearly labeled comparisons. | Results become approximate only; do not claim superiority. |
| A-R0-5 | Synthetic signal tasks are useful early tests of spectral/temporal components. | T0 is expected to de-risk signal processing before full multimodal runs. | Treat synthetic wins as insufficient and require real audio/video confirmation. |
| A-R0-6 | Latent prediction, reconstruction, alignment, uncertainty, and policy losses can be isolated through ablations. | Falsification requires single-variable comparisons. | Implementation must add ablation flags or the relevant claim remains untestable. |
| A-R0-7 | Diagnostic probes can be controlled with random/null/shuffled baselines. | Probe claims depend on ruling out decoder priors. | Mark exploration findings as anecdotal, not evidence. |
| A-R0-8 | Unsupported-output behavior can be measured with usefulness and abstention, not only hallucination rate. | Policy claims can otherwise collapse to always-no-op. | Do not claim hallucination reduction. |

---

## 2. Risk Register

| Risk ID | Risk | Impact | Leading indicators | Mitigation | Stop / decision condition | Linked hypotheses |
|---|---|---|---|---|---|---|
| R-R0-1 | Hypotheses become broad slogans rather than falsifiable claims. | Later work can cherry-pick results or overclaim. | Claims lack metric, baseline, ablation, or failure threshold. | Enforce `hypotheses.md` checklist; reject any claim that cannot resolve to support/partial/not supported. | Pause next sprint until missing criteria are added. | all |
| R-R0-2 | Canonical research-plan filename drift causes traceability ambiguity. | Agents may read different sources or miss constraints. | References to stale aliases or divergent research-plan files. | Treat `signal_latent_world_model_research_plan.md` as canonical and update traceability notes when file names change. | If source docs diverge, block implementation until reconciled. | all |
| R-R0-3 | Baselines are weaker or unfairly budgeted. | Apparent SLWM gains may be invalid. | Missing parameter counts, data budgets, split details, context length, or compute logs. | Registry requires budget/accounting fields; compare strict and core accounting separately. | No result can change hypothesis status without fair or labeled-approximate baseline. | all |
| R-R0-4 | Modality collapse: model ignores audio/video and relies on text/captions. | Shared latent and grounding hypotheses become invalid. | High text-only performance but poor audio/video-only or modality-dropout metrics. | Use modality dropout, audio/video-only tasks, missing-modality prediction, and hard negatives. | If multimodal gains vanish when text is removed/corrupted, mark grounding claim not supported. | H-R0-2, H-R0-5 |
| R-R0-5 | Decoder dominance: output heads invent plausible content from priors. | Probe and hallucination results become misleading. | Random-latent or shuffled probes produce similar quality outputs. | Include random/null/shuffled controls and source/uncertainty labels. | If controls match real probes, exploration claims are not supported. | H-R0-4, H-R0-5 |
| R-R0-6 | Hallucination reduction is actually abstention collapse. | Policy looks safer but loses usefulness. | Unsupported rate drops while no-op/abstention rises sharply and accuracy/usefulness falls. | Always report usefulness, accuracy, abstention/no-op, contradiction, and calibration. | If always-no-op wins usefulness-adjusted metrics or usefulness collapses, H-R0-4 is not supported. | H-R0-4 |
| R-R0-7 | Spectral components help toy signals but not real modalities. | Architecture complexity may not be justified. | Large synthetic gains but no audio/video latent gain. | Require real audio or video confirmation for full support. | If real-modality gains remain absent after controlled tests, remove/redesign component. | H-R0-1, H-R0-3 |
| R-R0-8 | Text/code performance collapses. | Text becomes unusable as an edge modality and comparisons to GPT-2 are weak. | Text loss exceeds registered 15–25% tolerance; code syntax errors dominate. | Track text/code guardrail against GPT-2 baseline and SLWM text-only ablations. | If guardrail fails, record tradeoff/blocker before claiming broader viability. | G-R0-1 |
| R-R0-9 | Data leakage or split contamination in paired multimodal data. | Grounding metrics become inflated. | Duplicate media/captions across train/val/test; benchmark contamination. | Dataset cards, split hashes, duplicate checks, and source IDs in registry. | Results with unresolved leakage cannot support hypotheses. | H-R0-2, H-R0-4, H-R0-5 |
| R-R0-10 | Parameter accounting ambiguity hides unfair comparisons. | Claims against GPT-2 or baselines become invalid. | Adapters/codecs counted inconsistently; frozen encoders omitted without labeling. | Report strict and core-only accounting modes for every experiment. | If accounting cannot be reconstructed, mark comparison approximate only. | all |
| R-R0-11 | Evaluation harness lags behind training. | Results cannot be reproduced or compared. | Metrics generated by ad-hoc scripts without versioning, config, or seed. | E0 eval harness before model-quality claims; registry requires eval script path/version. | Do not update hypothesis status from unregistered ad-hoc results. | all |
| R-R0-12 | Loss interactions obscure which objective caused gains. | Hypotheses cannot be falsified cleanly. | Joint training improves one metric but ablations are missing or multiple variables changed. | Pre-register ablations and isolate one variable per comparison. | If ablation cannot isolate the variable, status remains untested/partial only. | H-R0-1, H-R0-2, H-R0-3 |
| R-R0-13 | Exploration outputs are overinterpreted as understanding or worldview evidence. | Scientific claims become unprofessional and unsupported. | Reports use terms like “understands” without metrics/controls. | Require diagnostic-only tags, controls, failure cases, and “what cannot be concluded.” | Reject exploration reports that treat probes as committed factual behavior. | H-R0-5 |
| R-R0-14 | Compute limits prevent GPT-2-scale validation. | Claims may remain pilot-scale. | Only 10M–30M runs exist; no scaling trend. | Label results by scale and treat pilot support as partial. | Do not claim SLWM-124M support without 124M or justified scale trend evidence. | all |
| R-R0-15 | Future-phase action/embodied claims leak into current success criteria. | Scope drift delays required modality gates. | Optional sensors/robotics become required before text/audio/visual gates pass. | Keep action as policy/no-op conceptual output unless explicitly enabled later. | Remove optional-modality claims from current R0 status. | H-R0-4, H-R0-5 |

---

## 3. Risk Severity Priorities

High-priority risks for immediate control:

1. **R-R0-3 baseline unfairness** — invalidates almost every gain claim.
2. **R-R0-6 abstention collapse** — most likely false-positive route for unsupported-output improvements.
3. **R-R0-5 decoder dominance** — most likely false-positive route for probe results.
4. **R-R0-9 data leakage** — especially important for paired multimodal data.
5. **R-R0-12 missing ablations** — prevents causal interpretation of architectural choices.

---

## 4. Assumption Review Gates

Before moving past R0/R1 into implementation and training, review:

- [x] Is the canonical research plan filename resolved as `signal_latent_world_model_research_plan.md`?
- [ ] Does every planned experiment have at least one fair baseline?
- [ ] Are strict and core-only parameter accounting modes defined for configs?
- [ ] Can each key component be disabled through an ablation flag or alternate config?
- [ ] Are text/code, audio, and visual/video all represented in the data contract plan?
- [ ] Are unsupported-output metrics tied to usefulness and abstention?
- [ ] Are diagnostic probes required to include random/null/shuffled controls?

---

## 5. Current Risk Conclusion

The project is scientifically viable only if it treats evidence as the product. The main risk is not that SLWM-124M fails; failure is informative if registered cleanly. The main risk is accepting unmeasured or unfair claims. R0 therefore prioritizes falsifiability, baselines, ablations, and negative-result recording over architectural enthusiasm.
