# SLWM Documentation Index

This directory contains the project documentation beyond the root `README.md` and `AGENTS.md`.

## Required Agent Reading

Agents should follow the reading order in `../AGENTS.md`:

1. `research/signal_latent_world_model_research_plan.md`
2. `research/research_impl_eval_docs.md`
3. `process/sprint_playbook_prompts.md`
4. `exploration/exploration.md`
5. `../AGENTS.md`

## Research

- `research/signal_latent_world_model_research_plan.md` — canonical scientific plan.
- `research/research_impl_eval_docs.md` — documentation map and reading tracks.
- `research/hypotheses.md` — falsifiable project hypotheses and guardrails.
- `research/risks_and_assumptions.md` — assumptions, risks, mitigations, and stop conditions.
- `research/literature_map.md` — literature-to-design mapping.
- `research/design_decisions.md` — design decisions and ablation hooks.
- `research/video_repair_protocol_20261105.md` — frozen true-gap video versus audio comparison.
- `research/findings.md` — current decisions, including mixed video-repair evidence.

## Architecture

- `architecture/architecture.md` — architecture and module boundaries.
- `architecture/inference.md` — inference modes and claim boundaries.
- `architecture/policy_commitment.md` — policy/commitment design.
- `model_spec.md` — I0 tensor and module shape contracts.
- `data_contract.md` — unified sample schema, modality IDs, and source tags.

## Training

- `training/preprocessing.md` — T2 audio/visual latent preprocessing and dataset plan.
- `training/t1_text_code_training.md` — T1 text/code training protocol and current evidence.
- `training/video_repair.md` — actual speech-repair checkpoints, waveform isolation and reproduction.
- `../training/README.md` — runnable training command reference.

## Evaluation And Experiments

- `evaluation/baselines.md` — baseline specifications and smoke artifacts.
- `experiments/experiment_registry.md` — registry schema and completed artifact references.
- `exploration/exploration.md` — diagnostic probe and latent-exploration protocol.

## Process

Latest T2 generation research: [findings](research/findings.md),
[conditional generation and standalone inference](training/conditional_generation.md),
[G1 preregistration](research/conditional_generation_protocol_20261104.md), and
[G2 residual follow-up](research/conditional_residual_followup_20261104.md).

New shared-core work: [signal restoration and content conservation](training/signal_restoration.md),
[R1 protocol](research/signal_restoration_protocol_20261105.md), and
[R2 protocol](research/content_conservation_protocol_20261105.md).

- `process/sprint_playbook_prompts.md` — sprint boundaries, KPIs, prompts, and gates.

## Scale Notes

`SLWM-124M` remains the GPT-2-small-scale anchor. Larger profiles such as the T2 700M+ fit-check are separate scale profiles and must report parameter accounting mode, data budget, compute budget, and claim limits before comparison.
