"""Experiment registry writer compatible with ``experiment_registry.md``.

I0 writes JSON records (valid YAML subset) and performs minimal schema checks.
It does not create evidence or claim success for any model result.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from utils.config import config_hash


REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "experiment_id",
    "status",
    "created_at",
    "updated_at",
    "sprint",
    "claim_trace",
    "repository",
    "config",
    "model",
    "ablation",
    "data",
    "training",
    "evaluation",
    "interpretation",
)


def make_i0_registry_entry(
    *,
    experiment_id: str = "EXP-I0-001",
    config_path: str = "configs/default_i0.json",
    config: Mapping[str, Any] | None = None,
    git_commit: str | None = None,
    working_tree_state: str = "dirty",
) -> dict[str, Any]:
    """Create a planned I0 registry entry following the existing schema.

    The entry documents a shape-contract validation artifact only. It must not
    be interpreted as training or evaluation evidence.
    """

    today = date.today().isoformat()
    cfg = dict(config or {})
    cfg_hash = config_hash(cfg) if cfg else "sha256:uncomputed"
    return {
        "experiment_id": experiment_id,
        "status": "planned",
        "created_at": today,
        "updated_at": today,
        "sprint": {"id": "I0", "name": "Repo skeleton and contracts", "owner_role": "implementation"},
        "claim_trace": {
            "hypothesis_ids": [],
            "guardrail_ids": ["I0-shape-contract"],
            "research_questions": [],
            "expected_decision": "untested",
        },
        "repository": {
            "git_commit": git_commit,
            "working_tree_state": working_tree_state,
            "code_diff_ref": None,
            "docs_read": [
                "AGENTS.md",
                "signal_latent_world_model_research_plan.md",
                "research_impl_eval_docs.md",
                "sprint_playbook_prompts.md",
                "exploration.md",
                "design_decisions.md",
                "experiment_registry.md",
            ],
        },
        "config": {
            "config_path": config_path,
            "config_hash": cfg_hash,
            "seed": cfg.get("runtime", {}).get("seed", 0) if cfg else 0,
            "deterministic": cfg.get("runtime", {}).get("deterministic", True) if cfg else True,
            "precision": cfg.get("runtime", {}).get("precision", "fp32") if cfg else "fp32",
            "context_length": cfg.get("model", {}).get("latent_length", 1024) if cfg else 1024,
            "latent_length": cfg.get("model", {}).get("latent_length", 1024) if cfg else 1024,
            "latent_dim": cfg.get("model", {}).get("latent_dim", 768) if cfg else 768,
        },
        "model": {
            "name": cfg.get("model", {}).get("name", "SLWM-I0-shape-contract-stub") if cfg else "SLWM-I0-shape-contract-stub",
            "variant": "slwm",
            "parameter_accounting_mode": "strict",
            "total_trainable_parameters": 0,
            "core_trainable_parameters": 0,
            "frozen_parameters": 0,
            "module_parameter_counts": {"adapters": 0, "processor": 0, "heads": 0, "policy": 0, "decoders": 0},
            "enabled_modalities": ["text_code", "audio", "visual_video"],
            "architecture_flags": cfg.get("model", {}).get("architecture_flags", {}) if cfg else {},
        },
        "ablation": {"is_ablation": False, "ablation_of": None, "changed_variable": None, "held_constant": []},
        "data": {
            "dataset_mix": cfg.get("data", {}).get("dataset_mix", {}) if cfg else {},
            "datasets": [],
            "preprocessing": {
                "text_codec": None,
                "audio_codec_or_features": None,
                "visual_codec_or_features": None,
                "sample_schema_version": cfg.get("data", {}).get("sample_schema_version", "i0.1") if cfg else "i0.1",
            },
        },
        "training": {
            "objective": [],
            "optimizer": None,
            "learning_rate_schedule": None,
            "batch_size": None,
            "total_steps": 0,
            "train_tokens_or_samples": 0,
            "wall_clock_time": None,
            "hardware": None,
            "total_flops_estimate": 0,
            "checkpoint_path": None,
            "save_config_with_checkpoint": True,
            "anomalies": {"nan_or_inf": False, "loss_explosion": False, "modality_collapse": False, "notes": "I0 has no training."},
        },
        "evaluation": {
            "eval_script": "tests/test_dummy_end_to_end.py",
            "eval_script_hash": "sha256:uncomputed",
            "checkpoint_path": None,
            "seeds": [0],
            "decoding_or_probe_settings": {"temperature": None, "top_p": None, "max_new_tokens": None, "diagnostic_only": True},
            "metrics": {"primary": {"name": "shape_contract_tests", "value": None, "higher_is_better": True, "confidence_interval": None}, "secondary": []},
            "baselines_compared": [],
            "controls": {"random_or_null": False, "shuffled_pairs": False, "fixed_router": False, "always_noop": False, "no_policy": False},
        },
        "interpretation": {
            "result_summary": "Planned I0 shape-contract validation; no empirical model claim.",
            "hypothesis_decision": "untested",
            "failure_modes_observed": [],
            "limitations": ["No real architecture, training, datasets, or evaluation metrics in I0."],
            "next_allowed_step": "Proceed to I1 only after I0 shape tests pass.",
            "claim_language_allowed": "Only repo skeleton and shape-contract readiness claims after tests pass.",
        },
    }


def make_i1_baseline_registry_entry(
    *,
    experiment_id: str,
    config_path: str,
    config: Mapping[str, Any],
    model_name: str,
    model_variant: str,
    parameter_count: int,
    module_parameter_counts: Mapping[str, int],
    enabled_modalities: list[str],
    metrics: Mapping[str, Any],
    training_steps: int,
    train_tokens_or_samples: int,
    checkpoint_path: str | None = None,
    git_commit: str | None = None,
    working_tree_state: str = "dirty",
) -> dict[str, Any]:
    """Create a completed Sprint I1 baseline registry entry.

    The entry is for tiny overfit/smoke evidence only. It can show that a
    baseline path runs, backpropagates, reduces tiny-batch loss, and logs
    metrics; it does not support SLWM quality claims.
    """

    today = date.today().isoformat()
    cfg = dict(config)
    cfg_hash = config_hash(cfg)
    runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), Mapping) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), Mapping) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), Mapping) else {}
    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), Mapping) else {}
    codecs = model_cfg.get("codecs", {}) if isinstance(model_cfg.get("codecs", {}), Mapping) else {}
    tokenizer_cfg = model_cfg.get("tokenizer", {}) if isinstance(model_cfg.get("tokenizer", {}), Mapping) else {}
    primary_metric_name = str(metrics.get("primary_metric", "loss_drop_percent"))
    primary_metric_value = metrics.get(primary_metric_name, metrics.get("loss_drop_percent"))
    initial_loss = metrics.get("initial_loss")
    final_loss = metrics.get("final_loss")
    text_validation_loss = final_loss if model_variant == "gpt2_baseline" else None
    perplexity = metrics.get("perplexity")
    objective = train_cfg.get("objective", ["tiny_overfit_cross_entropy"])
    if isinstance(objective, str):
        objective = [objective]

    return {
        "experiment_id": experiment_id,
        "status": "completed",
        "created_at": today,
        "updated_at": today,
        "sprint": {"id": "I1", "name": "Baselines", "owner_role": "implementation"},
        "claim_trace": {
            "hypothesis_ids": ["H-R0-2"] if model_variant == "vanilla_multimodal_transformer" else [],
            "guardrail_ids": ["G-R0-1"] if model_variant == "gpt2_baseline" else ["I1-baseline-smoke"],
            "research_questions": ["RQ1"] if model_variant == "vanilla_multimodal_transformer" else [],
            "expected_decision": "untested",
        },
        "repository": {
            "git_commit": git_commit,
            "working_tree_state": working_tree_state,
            "code_diff_ref": None,
            "docs_read": [
                "signal_latent_world_model_research_plan.md",
                "research_impl_eval_docs.md",
                "sprint_playbook_prompts.md",
                "exploration.md",
                "AGENTS.md",
                "README.md",
                "hypotheses.md",
                "design_decisions.md",
                "experiment_registry.md",
                "docs/model_spec.md",
                "docs/data_contract.md",
            ],
        },
        "config": {
            "config_path": config_path,
            "config_hash": cfg_hash,
            "seed": int(runtime.get("seed", model_cfg.get("seed", 0))),
            "deterministic": bool(runtime.get("deterministic", True)),
            "precision": str(runtime.get("precision", "fp32")),
            "context_length": int(model_cfg.get("context_length", model_cfg.get("latent_length", 1024))),
            "latent_length": int(model_cfg.get("latent_length", model_cfg.get("context_length", 1024))),
            "latent_dim": int(model_cfg.get("latent_dim", model_cfg.get("n_embd", 768))),
        },
        "model": {
            "name": model_name,
            "variant": model_variant,
            "parameter_accounting_mode": str(model_cfg.get("parameter_accounting_mode", "strict")),
            "total_trainable_parameters": int(parameter_count),
            "core_trainable_parameters": int(module_parameter_counts.get("processor", 0)),
            "frozen_parameters": 0,
            "module_parameter_counts": dict(module_parameter_counts),
            "enabled_modalities": enabled_modalities,
            "architecture_flags": model_cfg.get("architecture_flags", {}),
        },
        "ablation": {"is_ablation": False, "ablation_of": None, "changed_variable": None, "held_constant": ["seed", "tiny_batch"]},
        "data": {
            "dataset_mix": data_cfg.get("dataset_mix", {}),
            "datasets": data_cfg.get("datasets", []),
            "preprocessing": {
                "text_codec": tokenizer_cfg.get("type", codecs.get("text", "gpt2_bpe")),
                "audio_codec_or_features": codecs.get("audio", None),
                "visual_codec_or_features": codecs.get("visual", None),
                "sample_schema_version": data_cfg.get("sample_schema_version", "i1.baseline_smoke"),
            },
        },
        "training": {
            "objective": list(objective),
            "optimizer": train_cfg.get("optimizer", "adamw_numpy"),
            "learning_rate_schedule": train_cfg.get("learning_rate_schedule", "constant"),
            "batch_size": train_cfg.get("batch_size", None),
            "total_steps": int(training_steps),
            "train_tokens_or_samples": int(train_tokens_or_samples),
            "wall_clock_time": metrics.get("wall_clock_time_seconds"),
            "hardware": metrics.get("hardware", "local_cpu_numpy"),
            "total_flops_estimate": None,
            "checkpoint_path": checkpoint_path,
            "save_config_with_checkpoint": True,
            "anomalies": {
                "nan_or_inf": bool(metrics.get("nan_or_inf", False)),
                "loss_explosion": bool(metrics.get("loss_explosion", False)),
                "modality_collapse": False,
                "notes": "Sprint I1 tiny-batch overfit smoke run; no model-quality claim.",
            },
        },
        "evaluation": {
            "eval_script": "training/baseline_smoke.py",
            "eval_script_hash": metrics.get("eval_script_hash", "sha256:uncomputed"),
            "checkpoint_path": checkpoint_path,
            "seeds": [int(runtime.get("seed", model_cfg.get("seed", 0)))],
            "decoding_or_probe_settings": {"temperature": None, "top_p": None, "max_new_tokens": None, "diagnostic_only": False},
            "metrics": {
                "primary": {
                    "name": primary_metric_name,
                    "value": primary_metric_value,
                    "higher_is_better": True,
                    "confidence_interval": None,
                },
                "secondary": [
                    {"name": "initial_loss", "value": initial_loss, "higher_is_better": False},
                    {"name": "final_loss", "value": final_loss, "higher_is_better": False},
                    {"name": "perplexity", "value": perplexity, "higher_is_better": False},
                    {"name": "parameter_count", "value": int(parameter_count), "higher_is_better": None},
                ],
                "required_bundles": {
                    "hallucination_or_policy_claim": {
                        "required_when_claiming_reduction": True,
                        "unsupported_claim_rate": None,
                        "contradiction_rate": None,
                        "grounded_accuracy_or_usefulness": None,
                        "abstention_or_noop_rate": None,
                        "calibration_metric": None,
                    }
                },
            },
            "baselines_compared": [],
            "controls": {
                "random_or_null": bool(metrics.get("random_or_null_control", False)),
                "shuffled_pairs": bool(metrics.get("shuffled_pairs", False)),
                "fixed_router": False,
                "always_noop": False,
                "no_policy": True,
            },
        },
        "interpretation": {
            "result_summary": str(metrics.get("result_summary", "Tiny-batch baseline smoke run completed.")),
            "hypothesis_decision": "untested",
            "failure_modes_observed": list(metrics.get("failure_modes_observed", [])),
            "limitations": [
                "Tiny synthetic/in-memory data only.",
                "No SLWM comparison and no model-quality claim.",
                "No hallucination, grounding, or policy claim.",
            ],
            "next_allowed_step": "Use registered baselines as implementation readiness evidence; larger training belongs to Training/Evaluation sprints.",
            "claim_language_allowed": "Baseline path runs and tiny-batch loss decreased; no broader capability claim.",
        },
    }


def make_t0_synthetic_registry_entry(
    *,
    experiment_id: str,
    config_path: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    model_parameter_counts: Mapping[str, Mapping[str, int]],
    training_steps: int,
    train_samples: int,
    checkpoint_path: str | None = None,
    git_commit: str | None = None,
    working_tree_state: str = "dirty",
) -> dict[str, Any]:
    """Create a completed Sprint T0 synthetic-signal registry entry.

    The entry records controlled synthetic signal evidence only. It must not be
    interpreted as text/code/audio/video, multimodal grounding, hallucination, or
    policy evidence.
    """

    today = date.today().isoformat()
    cfg = dict(config)
    cfg_hash = config_hash(cfg)
    runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), Mapping) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), Mapping) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), Mapping) else {}
    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), Mapping) else {}

    aggregate = metrics.get("aggregate", {}) if isinstance(metrics.get("aggregate", {}), Mapping) else {}
    gate = metrics.get("success_gate", {}) if isinstance(metrics.get("success_gate", {}), Mapping) else {}
    tasks = list(data_cfg.get("tasks", metrics.get("tasks", []))) if isinstance(data_cfg.get("tasks", metrics.get("tasks", [])), list) else []
    slwm_counts = dict(model_parameter_counts.get("slwm", {})) if isinstance(model_parameter_counts.get("slwm", {}), Mapping) else {}
    total_params = int(slwm_counts.get("total", 0))

    return {
        "experiment_id": experiment_id,
        "status": "completed" if not gate.get("failure_report_written", False) else "failed",
        "created_at": today,
        "updated_at": today,
        "sprint": {"id": "T0", "name": "Synthetic signal pretraining", "owner_role": "training"},
        "claim_trace": {
            "hypothesis_ids": ["H-R0-1", "H-R0-3"],
            "guardrail_ids": ["T0-synthetic-only", "T0-baseline-comparison", "T0-stop-on-no-win"],
            "research_questions": ["RQ2", "RQ3"],
            "expected_decision": "support_or_fail_signal_processor_sanity",
        },
        "repository": {
            "git_commit": git_commit,
            "working_tree_state": working_tree_state,
            "code_diff_ref": None,
            "docs_read": [
                "signal_latent_world_model_research_plan.md",
                "research_impl_eval_docs.md",
                "sprint_playbook_prompts.md",
                "exploration.md",
                "AGENTS.md",
                "docs/model_spec.md",
                "docs/data_contract.md",
            ],
        },
        "config": {
            "config_path": config_path,
            "config_hash": cfg_hash,
            "seed": int(runtime.get("seed", model_cfg.get("seed", 0))),
            "deterministic": bool(runtime.get("deterministic", True)),
            "precision": str(runtime.get("precision", "float64_numpy")),
            "context_length": int(model_cfg.get("context_length", model_cfg.get("latent_length", data_cfg.get("context_length", 0)))),
            "latent_length": int(model_cfg.get("context_length", model_cfg.get("latent_length", data_cfg.get("context_length", 0)))),
            "latent_dim": int(model_cfg.get("latent_dim", model_cfg.get("n_embd", data_cfg.get("latent_dim", 0)))),
        },
        "model": {
            "name": str(model_cfg.get("name", "SLWM-T0-synthetic-signal")),
            "variant": "slwm_synthetic_signal_predictor",
            "parameter_accounting_mode": str(model_cfg.get("parameter_accounting_mode", "strict")),
            "total_trainable_parameters": total_params,
            "core_trainable_parameters": int(slwm_counts.get("processor", 0)),
            "frozen_parameters": 0,
            "module_parameter_counts": dict(slwm_counts),
            "enabled_modalities": ["synthetic_signal"],
            "architecture_flags": model_cfg.get("architecture_flags", {}),
        },
        "ablation": {
            "is_ablation": False,
            "ablation_of": None,
            "changed_variable": None,
            "held_constant": ["synthetic_tasks", "sample_count", "seed", "optimizer_family", "training_steps"],
            "included_ablation_variants": ["slwm_no_spectral"],
        },
        "data": {
            "dataset_mix": data_cfg.get("dataset_mix", {"synthetic_signal": 1.0}),
            "datasets": [
                {
                    "name": "synthetic_signal_t0",
                    "version_or_snapshot": data_cfg.get("dataset_version", "synthetic_signal_v0"),
                    "split": "generated_train_eval" if not train_cfg.get("overfit_batch", True) else "fixed_overfit_batch",
                    "sample_count": int(train_samples),
                    "tokens": None,
                    "audio_hours": None,
                    "video_hours": None,
                    "license_notes": "synthetic data generated in repo; no external dataset",
                    "leakage_checks": "same fixed batch intentionally used only when overfit_batch=true; otherwise split-specific seeds",
                    "tasks": tasks,
                }
            ],
            "preprocessing": {
                "text_codec": None,
                "audio_codec_or_features": None,
                "visual_codec_or_features": None,
                "sample_schema_version": data_cfg.get("sample_schema_version", "t0.synthetic_signal_v0"),
            },
        },
        "training": {
            "objective": train_cfg.get("objective", ["synthetic_latent_mse_prediction"]),
            "optimizer": train_cfg.get("optimizer", "adamw_numpy"),
            "learning_rate_schedule": train_cfg.get("learning_rate_schedule", "constant"),
            "batch_size": train_cfg.get("batch_size", None),
            "total_steps": int(training_steps),
            "train_tokens_or_samples": int(train_samples),
            "wall_clock_time": metrics.get("wall_clock_time_seconds"),
            "hardware": metrics.get("hardware", "local_cpu_numpy"),
            "total_flops_estimate": None,
            "checkpoint_path": checkpoint_path,
            "save_config_with_checkpoint": checkpoint_path is not None,
            "anomalies": {
                "nan_or_inf": bool(aggregate.get("nan_or_inf", False)),
                "loss_explosion": bool(aggregate.get("loss_explosion", False)),
                "modality_collapse": False,
                "notes": "Sprint T0 synthetic-signal run; no text/code/audio/video datasets used.",
            },
        },
        "evaluation": {
            "eval_script": "training/t0_synthetic_pretrain.py",
            "eval_script_hash": metrics.get("eval_script_hash", "sha256:uncomputed"),
            "checkpoint_path": checkpoint_path,
            "seeds": [int(runtime.get("seed", model_cfg.get("seed", 0)))],
            "decoding_or_probe_settings": {"temperature": None, "top_p": None, "max_new_tokens": None, "diagnostic_only": False},
            "metrics": {
                "primary": {
                    "name": "slwm_beats_vanilla_mse_on_any_task",
                    "value": bool(gate.get("slwm_beats_vanilla_on_any_task", False)),
                    "higher_is_better": True,
                    "confidence_interval": None,
                },
                "secondary": [
                    {"name": "synthetic_mse", "value": aggregate.get("synthetic_mse"), "higher_is_better": False},
                    {"name": "spectral_magnitude_error", "value": aggregate.get("spectral_magnitude_error"), "higher_is_better": False},
                    {"name": "phase_or_coherence_error", "value": aggregate.get("phase_or_coherence_error"), "higher_is_better": False},
                    {"name": "frequency_recovery_error", "value": aggregate.get("frequency_recovery_error"), "higher_is_better": False},
                    {"name": "throughput_samples_per_second", "value": aggregate.get("throughput_samples_per_second"), "higher_is_better": True},
                    {"name": "slwm_tasks_beating_vanilla_count", "value": gate.get("slwm_tasks_beating_vanilla_count"), "higher_is_better": True},
                    {"name": "slwm_tasks_beating_vanilla_any_metric_count", "value": gate.get("slwm_tasks_beating_vanilla_any_metric_count"), "higher_is_better": True},
                    {"name": "slwm_tasks_beating_vanilla_all_required_metrics_count", "value": gate.get("slwm_tasks_beating_vanilla_all_required_metrics_count"), "higher_is_better": True},
                    {"name": "slwm_tasks_beating_random_or_noop_count", "value": gate.get("slwm_tasks_beating_random_or_noop_count"), "higher_is_better": True},
                    {"name": "no_spectral_delta_mse", "value": aggregate.get("no_spectral_delta_mse"), "higher_is_better": True},
                ],
                "required_bundles": {
                    "hallucination_or_policy_claim": {
                        "required_when_claiming_reduction": True,
                        "unsupported_claim_rate": None,
                        "contradiction_rate": None,
                        "grounded_accuracy_or_usefulness": None,
                        "abstention_or_noop_rate": None,
                        "calibration_metric": None,
                    }
                },
            },
            "baselines_compared": ["vanilla_continuous_transformer", "slwm_no_spectral", "random_signal", "noop_signal"],
            "controls": {"random_or_null": True, "shuffled_pairs": False, "fixed_router": False, "always_noop": True, "no_policy": True},
        },
        "interpretation": {
            "result_summary": str(metrics.get("result_summary", "Sprint T0 synthetic comparison completed.")),
            "hypothesis_decision": "partial_support" if gate.get("slwm_beats_vanilla_on_any_task", False) else "not_supported",
            "failure_modes_observed": list(metrics.get("failure_modes_observed", [])),
            "limitations": [
                "Synthetic controlled signals only; no text/code/audio/video dataset evidence.",
                "NumPy smoke-scale implementation; not GPT-2-scale evidence.",
                "No hallucination, grounding, policy, or multimodal claim is supported by T0.",
            ],
            "next_allowed_step": str(metrics.get("next_allowed_step", "Do not proceed beyond T0 unless success gate passes.")),
            "claim_language_allowed": "Only controlled synthetic signal metric comparisons may be claimed.",
        },
    }


def make_t1_text_registry_entry(
    *,
    experiment_id: str,
    config_path: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    model_name: str,
    model_variant: str,
    parameter_count: int,
    module_parameter_counts: Mapping[str, int],
    training_steps: int,
    train_tokens: int,
    checkpoint_path: str | None,
    git_commit: str | None = None,
    working_tree_state: str = "dirty",
) -> dict[str, Any]:
    """Create a completed Sprint T1 text/code training registry entry.

    The entry records text/code-only validation loss/perplexity, generation
    settings, throughput, memory, parameter accounting, and guardrail context.
    It does not support multimodal, hallucination, or policy claims.
    """

    today = date.today().isoformat()
    cfg = dict(config)
    cfg_hash = config_hash(cfg)
    runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), Mapping) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), Mapping) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), Mapping) else {}
    train_cfg = cfg.get("training", {}) if isinstance(cfg.get("training", {}), Mapping) else {}
    tokenizer_cfg = model_cfg.get("tokenizer", {}) if isinstance(model_cfg.get("tokenizer", {}), Mapping) else {}
    generation_cfg = cfg.get("generation", {}) if isinstance(cfg.get("generation", {}), Mapping) else {}
    validation_loss = metrics.get("validation_loss")
    validation_perplexity = metrics.get("validation_perplexity")
    is_slwm = str(model_variant).startswith("slwm")

    return {
        "experiment_id": experiment_id,
        "status": "completed" if not metrics.get("nan_or_inf", False) else "failed",
        "created_at": today,
        "updated_at": today,
        "sprint": {"id": "T1", "name": "Text/code baseline training", "owner_role": "training"},
        "claim_trace": {
            "hypothesis_ids": ["H-R0-3"] if is_slwm else [],
            "guardrail_ids": ["G-R0-1", "T1-text-code-only", "DD-R1-001", "DD-R1-002", "DD-R1-020"],
            "research_questions": ["RQ3"] if is_slwm else [],
            "expected_decision": "guardrail_pass_or_tradeoff_record",
        },
        "repository": {
            "git_commit": git_commit,
            "working_tree_state": working_tree_state,
            "code_diff_ref": None,
            "docs_read": [
                "signal_latent_world_model_research_plan.md",
                "research_impl_eval_docs.md",
                "sprint_playbook_prompts.md",
                "exploration.md",
                "AGENTS.md",
                "hypotheses.md",
                "design_decisions.md",
                "experiment_registry.md",
                "docs/model_spec.md",
                "docs/data_contract.md",
                "docs/t1_text_code_training.md",
            ],
        },
        "config": {
            "config_path": config_path,
            "config_hash": cfg_hash,
            "seed": int(runtime.get("seed", model_cfg.get("seed", 0))),
            "deterministic": bool(runtime.get("deterministic", True)),
            "precision": str(runtime.get("precision", "float64_numpy")),
            "context_length": int(model_cfg.get("context_length", model_cfg.get("latent_length", train_cfg.get("sequence_length", 0)))),
            "latent_length": int(model_cfg.get("latent_length", model_cfg.get("context_length", train_cfg.get("sequence_length", 0)))),
            "latent_dim": int(model_cfg.get("latent_dim", model_cfg.get("n_embd", 0))),
        },
        "model": {
            "name": model_name,
            "variant": model_variant,
            "parameter_accounting_mode": str(model_cfg.get("parameter_accounting_mode", "strict")),
            "total_trainable_parameters": int(parameter_count),
            "core_trainable_parameters": int(module_parameter_counts.get("processor", 0)),
            "frozen_parameters": 0,
            "module_parameter_counts": dict(module_parameter_counts),
            "enabled_modalities": ["text_code"],
            "architecture_flags": model_cfg.get("architecture_flags", {}),
        },
        "ablation": cfg.get("ablation", {"is_ablation": False, "ablation_of": None, "changed_variable": None, "held_constant": []}),
        "data": {
            "dataset_mix": data_cfg.get("dataset_mix", {"text_code": 1.0, "audio": None, "visual_video": None}),
            "datasets": metrics.get("registry_datasets", data_cfg.get("datasets", [])),
            "preprocessing": {
                "text_codec": metrics.get("tokenizer", {}).get("effective_type", tokenizer_cfg.get("type", "unknown"))
                if isinstance(metrics.get("tokenizer", {}), Mapping)
                else tokenizer_cfg.get("type", "unknown"),
                "audio_codec_or_features": None,
                "visual_codec_or_features": None,
                "sample_schema_version": data_cfg.get("sample_schema_version", "t1.text_code_v0"),
                "split_digests": metrics.get("split_digests", {}),
            },
        },
        "training": {
            "objective": train_cfg.get("objective", ["next_token_cross_entropy"]),
            "optimizer": train_cfg.get("optimizer", "adamw_numpy"),
            "learning_rate_schedule": train_cfg.get("learning_rate_schedule", "constant"),
            "batch_size": train_cfg.get("batch_size", None),
            "total_steps": int(training_steps),
            "train_tokens_or_samples": int(train_tokens),
            "wall_clock_time": metrics.get("wall_clock_time_seconds"),
            "hardware": metrics.get("hardware", "local_cpu_numpy"),
            "total_flops_estimate": None,
            "checkpoint_path": checkpoint_path,
            "save_config_with_checkpoint": checkpoint_path is not None,
            "anomalies": {
                "nan_or_inf": bool(metrics.get("nan_or_inf", False)),
                "loss_explosion": bool(metrics.get("loss_explosion", False)),
                "modality_collapse": False,
                "notes": str(metrics.get("anomaly_notes", "Sprint T1 text/code-only run; no audio/visual data used.")),
            },
        },
        "evaluation": {
            "eval_script": "training/t1_text_baseline.py",
            "eval_script_hash": metrics.get("eval_script_hash", "sha256:uncomputed"),
            "checkpoint_path": checkpoint_path,
            "seeds": [int(runtime.get("seed", model_cfg.get("seed", 0)))],
            "decoding_or_probe_settings": {
                "temperature": generation_cfg.get("temperature"),
                "top_p": generation_cfg.get("top_p"),
                "top_k": generation_cfg.get("top_k"),
                "max_new_tokens": generation_cfg.get("max_new_tokens"),
                "diagnostic_only": False,
            },
            "metrics": {
                "primary": {
                    "name": "validation_loss",
                    "value": validation_loss,
                    "higher_is_better": False,
                    "confidence_interval": None,
                },
                "secondary": [
                    {"name": "validation_perplexity", "value": validation_perplexity, "higher_is_better": False},
                    {"name": "train_loss", "value": metrics.get("train_loss"), "higher_is_better": False},
                    {"name": "throughput_tokens_per_second", "value": metrics.get("throughput_tokens_per_second"), "higher_is_better": True},
                    {"name": "max_memory_mb", "value": metrics.get("max_memory_mb"), "higher_is_better": False},
                    {"name": "parameter_count", "value": int(parameter_count), "higher_is_better": None},
                    {"name": "text_loss_relative_delta_percent", "value": metrics.get("text_loss_relative_delta_percent"), "higher_is_better": False},
                ],
                "required_bundles": {
                    "hallucination_or_policy_claim": {
                        "required_when_claiming_reduction": True,
                        "unsupported_claim_rate": None,
                        "contradiction_rate": None,
                        "grounded_accuracy_or_usefulness": None,
                        "abstention_or_noop_rate": None,
                        "calibration_metric": None,
                    }
                },
            },
            "baselines_compared": metrics.get("baselines_compared", []),
            "controls": {"random_or_null": False, "shuffled_pairs": False, "fixed_router": False, "always_noop": False, "no_policy": True},
        },
        "interpretation": {
            "result_summary": str(metrics.get("result_summary", "Sprint T1 text/code run completed.")),
            "hypothesis_decision": str(metrics.get("hypothesis_decision", "guardrail_pass" if not is_slwm else "untested")),
            "failure_modes_observed": list(metrics.get("failure_modes_observed", [])),
            "limitations": list(
                metrics.get(
                    "limitations",
                    [
                        "Dependency-light local/pilot data unless config names a prepared external corpus.",
                        "No audio, visual, multimodal grounding, hallucination, or policy claim is supported by T1.",
                    ],
                )
            ),
            "next_allowed_step": str(metrics.get("next_allowed_step", "Compare registered T1 runs on the same tokenizer/split before changing G-R0-1 state.")),
            "claim_language_allowed": str(metrics.get("claim_language_allowed", "Only text/code validation loss, perplexity, throughput, memory, and sample-generation settings may be reported.")),
        },
    }


def validate_registry_entry(entry: Mapping[str, Any]) -> None:
    """Validate required top-level and I0-specific registry fields."""

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in entry]
    if missing:
        raise ValueError(f"Registry entry missing required top-level keys: {missing}")
    if not str(entry["experiment_id"]).startswith("EXP-"):
        raise ValueError("experiment_id must start with EXP-")
    sprint = entry.get("sprint", {})
    if not isinstance(sprint, Mapping) or "id" not in sprint or "owner_role" not in sprint:
        raise ValueError("sprint must contain id and owner_role")
    config = entry.get("config", {})
    if not isinstance(config, Mapping) or "config_path" not in config or "config_hash" not in config:
        raise ValueError("config must contain config_path and config_hash")


def write_registry_entry(entry: Mapping[str, Any], path: str | Path) -> Path:
    """Write a JSON registry entry and return the output path."""

    validate_registry_entry(entry)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(output_path)
    return output_path
