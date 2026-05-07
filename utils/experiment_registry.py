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
    output_path.write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path
