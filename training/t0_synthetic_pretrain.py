"""Config-driven Sprint T0 synthetic signal pretraining runner.

Scope guardrails:
* synthetic signal tasks only;
* no text/code/audio/video datasets;
* compare SLWM, vanilla continuous Transformer, no-spectral SLWM ablation,
  random predictor, and no-op predictor on identical synthetic batches;
* record MSE, spectral loss, phase error, frequency recovery error, throughput,
  and stability;
* write a failure report if SLWM does not beat the vanilla baseline on any
  controlled signal task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from models.baselines.continuous_transformer import (
    ContinuousSignalTransformerConfig,
    NumpyContinuousSignalTransformerBaseline,
)
from models.slwm_config import SLWMCoreConfig
from models.slwm_signal_predictor import NumpySLWMSignalPredictor
from training.synthetic_metrics import flatten_metric_summary, masked_mse, prediction_metric_bundle, summarize_stability
from training.synthetic_signals import SUPPORTED_SYNTHETIC_TASKS, SyntheticSignalBatch, make_batches_from_config
from utils.config import config_hash, load_config
from utils.experiment_registry import make_t0_synthetic_registry_entry, write_registry_entry


ARTIFACT_ROOT = Path("experiments/synthetic/t0")
MAX_T0_STEPS = 2_000
MAX_T0_BATCH = 64
MAX_T0_CONTEXT = 512
MAX_T0_WIDTH = 256
MAX_T0_LAYERS = 6
MAX_T0_PARAMETERS = 20_000_000
TRAINABLE_VARIANTS = ("slwm", "slwm_no_spectral", "vanilla_transformer")
CONTROL_VARIANTS = ("random_signal", "noop_signal")
REQUIRED_METRICS = ("mse", "spectral_loss", "phase_error", "frequency_recovery_error")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_artifact_path(raw_path: str | Path) -> Path:
    """Validate T0 artifact paths before writing.

    Only relative paths under ``experiments/synthetic/t0`` are allowed.  This
    keeps config-driven runs from writing outside the research artifact area.
    """

    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("T0 artifact paths must be relative")
    repo_root = Path.cwd().resolve()
    artifact_root = (repo_root / ARTIFACT_ROOT).resolve()
    resolved = (repo_root / path).resolve()
    if not _is_relative_to(resolved, artifact_root):
        raise ValueError(f"T0 synthetic artifacts must be written under {ARTIFACT_ROOT}")
    if resolved.exists() and resolved.is_symlink():
        raise ValueError("T0 artifact path must not be a symlink")
    return path


def _write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def _write_text(payload: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)
    return path


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _registry_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("registry", {}) if isinstance(config.get("registry", {}), Mapping) else {}


def _training_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("training", {}) if isinstance(config.get("training", {}), Mapping) else {}


def _model_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}


def _runtime_seed(config: Mapping[str, Any]) -> int:
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    return int(runtime.get("seed", 0))


def _experiment_id(config: Mapping[str, Any]) -> str:
    registry = _registry_config(config)
    return str(registry.get("experiment_id", "EXP-T0-001"))


def _artifact_dir(config: Mapping[str, Any], experiment_id: str) -> Path:
    registry = _registry_config(config)
    return _safe_artifact_path(str(registry.get("artifact_dir", f"experiments/synthetic/t0/{experiment_id}")))


def _artifact_paths(config: Mapping[str, Any], experiment_id: str) -> dict[str, Path]:
    registry = _registry_config(config)
    artifact_dir = _artifact_dir(config, experiment_id)
    return {
        "artifact_dir": artifact_dir,
        "metrics": _safe_artifact_path(str(registry.get("metrics_path", artifact_dir / "metrics.json"))),
        "registry": _safe_artifact_path(str(registry.get("path", artifact_dir / "registry.json"))),
        "comparison_csv": _safe_artifact_path(str(registry.get("comparison_path", artifact_dir / "comparison_table.csv"))),
        "failure_report": _safe_artifact_path(str(registry.get("failure_report_path", artifact_dir / "failure_report.md"))),
        "config_copy": _safe_artifact_path(str(registry.get("config_copy_path", artifact_dir / "config.json"))),
    }


def _tasks_from_config(config: Mapping[str, Any]) -> list[str]:
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), Mapping) else {}
    raw_tasks = data_cfg.get("tasks", list(SUPPORTED_SYNTHETIC_TASKS))
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("T0 config data.tasks must be a non-empty list")
    tasks = [str(task) for task in raw_tasks]
    unsupported = [task for task in tasks if task not in SUPPORTED_SYNTHETIC_TASKS]
    if unsupported:
        raise ValueError(f"Unsupported T0 synthetic tasks: {unsupported}")
    return tasks


def _slwm_config(config: Mapping[str, Any], *, use_spectral_mixer: bool, seed_offset: int) -> SLWMCoreConfig:
    model_cfg = dict(_model_config(config))
    flags = dict(model_cfg.get("architecture_flags", {})) if isinstance(model_cfg.get("architecture_flags", {}), Mapping) else {}
    runtime_seed = _runtime_seed(config)
    model_cfg.update(
        {
            "seed": runtime_seed + int(seed_offset),
            "use_spectral_mixer": bool(use_spectral_mixer),
            "use_latent_prediction_head": True,
            "use_uncertainty_head": False,
            "use_output_heads": False,
            "use_policy_gate": False,
        }
    )
    flags.update(
        {
            "use_spectral_mixer": bool(use_spectral_mixer),
            "use_latent_prediction_head": True,
            "use_uncertainty_head": False,
            "policy_commit_gate": False,
            "noop_head": False,
        }
    )
    model_cfg["architecture_flags"] = flags
    return SLWMCoreConfig.from_mapping({"model": model_cfg, "runtime": {"seed": runtime_seed + int(seed_offset)}})


def _vanilla_config(config: Mapping[str, Any], *, seed_offset: int) -> ContinuousSignalTransformerConfig:
    model_cfg = dict(_model_config(config))
    runtime_seed = _runtime_seed(config)
    model_cfg["seed"] = runtime_seed + int(seed_offset)
    return ContinuousSignalTransformerConfig.from_mapping({"model": model_cfg, "runtime": {"seed": runtime_seed + int(seed_offset)}})


def _validate_t0_limits(config: Mapping[str, Any], *, model_counts: Mapping[str, Mapping[str, int]]) -> None:
    train_cfg = _training_config(config)
    model_cfg = _model_config(config)
    steps = int(train_cfg.get("steps", 100))
    batch_size = int(train_cfg.get("batch_size", 4))
    context_length = int(model_cfg.get("context_length", model_cfg.get("latent_length", 32)))
    latent_dim = int(model_cfg.get("latent_dim", model_cfg.get("n_embd", 8)))
    n_layer = int(model_cfg.get("n_layer", model_cfg.get("processor_layers", 1)))
    if steps > MAX_T0_STEPS:
        raise ValueError(f"T0 steps must be <= {MAX_T0_STEPS}; got {steps}")
    if batch_size > MAX_T0_BATCH:
        raise ValueError(f"T0 batch_size must be <= {MAX_T0_BATCH}; got {batch_size}")
    if context_length > MAX_T0_CONTEXT:
        raise ValueError(f"T0 context_length must be <= {MAX_T0_CONTEXT}; got {context_length}")
    if latent_dim > MAX_T0_WIDTH:
        raise ValueError(f"T0 latent_dim must be <= {MAX_T0_WIDTH}; got {latent_dim}")
    if n_layer > MAX_T0_LAYERS:
        raise ValueError(f"T0 n_layer must be <= {MAX_T0_LAYERS}; got {n_layer}")
    for variant, counts in model_counts.items():
        total = int(counts.get("total", 0))
        if total > MAX_T0_PARAMETERS:
            raise ValueError(f"T0 {variant} parameter_count must be <= {MAX_T0_PARAMETERS}; got {total}")


def _evaluate_model(model: Any, batch: SyntheticSignalBatch) -> tuple[dict[str, float], np.ndarray]:
    output = model.forward(batch.input_latents, input_mask=batch.input_mask)
    prediction = np.asarray(output["prediction"], dtype=np.float64)
    return prediction_metric_bundle(prediction, batch.target_latents, loss_mask=batch.loss_mask), prediction


def _train_fixed_batch(
    *,
    model: Any,
    optimizer: Any,
    batch: SyntheticSignalBatch,
    eval_batch: SyntheticSignalBatch,
    steps: int,
) -> dict[str, Any]:
    """Run deterministic fixed-batch T0 overfit training for one model."""

    initial_loss = masked_mse(model.forward(eval_batch.input_latents, input_mask=eval_batch.input_mask)["prediction"], eval_batch.target_latents, mask=eval_batch.loss_mask)
    losses: list[float] = [float(initial_loss)]
    grad_norms: list[float] = []
    nan_or_inf = not np.isfinite(initial_loss)
    start = time.perf_counter()
    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss, _ = model.loss_and_backward(
            batch.input_latents,
            batch.target_latents,
            input_mask=batch.input_mask,
            loss_mask=batch.loss_mask,
        )
        if not np.isfinite(loss):
            nan_or_inf = True
            losses.append(float(loss))
            break
        grad_norm = optimizer.step()
        grad_norms.append(float(grad_norm))
        losses.append(float(loss))
    wall_clock = time.perf_counter() - start
    final_loss = masked_mse(model.forward(eval_batch.input_latents, input_mask=eval_batch.input_mask)["prediction"], eval_batch.target_latents, mask=eval_batch.loss_mask)
    losses.append(float(final_loss))
    if not np.isfinite(final_loss):
        nan_or_inf = True
    stability = summarize_stability(losses, grad_norms)
    stability["nan_or_inf"] = bool(stability["nan_or_inf"] or nan_or_inf)
    samples_seen = int(steps) * int(batch.input_latents.shape[0])
    return {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "loss_drop_percent": float(100.0 * (initial_loss - final_loss) / max(initial_loss, 1e-12)),
        "losses": losses,
        "grad_norms": grad_norms,
        "wall_clock_time_seconds": float(wall_clock),
        "throughput_samples_per_second": float(samples_seen / max(wall_clock, 1e-12)),
        "stability": stability,
    }


def _random_prediction(batch: SyntheticSignalBatch, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    target = np.asarray(batch.target_latents, dtype=np.float64)
    std = float(np.std(target)) or 1.0
    return rng.normal(0.0, std, size=target.shape).astype(np.float64)


def _noop_prediction(batch: SyntheticSignalBatch) -> np.ndarray:
    return np.asarray(batch.input_latents, dtype=np.float64).copy()


def _counts_for_variants(config: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    slwm = NumpySLWMSignalPredictor(_slwm_config(config, use_spectral_mixer=True, seed_offset=10))
    no_spectral = NumpySLWMSignalPredictor(_slwm_config(config, use_spectral_mixer=False, seed_offset=20))
    vanilla = NumpyContinuousSignalTransformerBaseline(_vanilla_config(config, seed_offset=30))
    return {
        "slwm": slwm.parameter_count_breakdown().registry_module_counts(),
        "slwm_no_spectral": no_spectral.parameter_count_breakdown().registry_module_counts(),
        "vanilla_transformer": vanilla.module_parameter_counts(),
        "random_signal": {"adapters": 0, "processor": 0, "heads": 0, "policy": 0, "decoders": 0, "embeddings": 0, "total": 0},
        "noop_signal": {"adapters": 0, "processor": 0, "heads": 0, "policy": 0, "decoders": 0, "embeddings": 0, "total": 0},
    }


def _comparison_for_task(task_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    slwm = task_metrics["slwm"]["final_metrics"]
    vanilla = task_metrics["vanilla_transformer"]["final_metrics"]
    no_spectral = task_metrics["slwm_no_spectral"]["final_metrics"]
    random_metrics = task_metrics["random_signal"]["final_metrics"]
    noop_metrics = task_metrics["noop_signal"]["final_metrics"]

    wins_vs_vanilla = [metric for metric in REQUIRED_METRICS if float(slwm[metric]) < float(vanilla[metric])]
    wins_vs_vanilla_mse = ["mse"] if float(slwm["mse"]) < float(vanilla["mse"]) else []
    wins_vs_vanilla_all = list(REQUIRED_METRICS) if len(wins_vs_vanilla) == len(REQUIRED_METRICS) else []
    wins_vs_no_spectral = [metric for metric in REQUIRED_METRICS if float(slwm[metric]) < float(no_spectral[metric])]
    wins_vs_random_or_noop = [
        metric
        for metric in REQUIRED_METRICS
        if float(slwm[metric]) < min(float(random_metrics[metric]), float(noop_metrics[metric]))
    ]
    return {
        "wins_vs_vanilla_metrics": wins_vs_vanilla,
        "wins_vs_vanilla_mse_metrics": wins_vs_vanilla_mse,
        "wins_vs_vanilla_all_required_metrics": wins_vs_vanilla_all,
        "wins_vs_no_spectral_metrics": wins_vs_no_spectral,
        "wins_vs_random_or_noop_metrics": wins_vs_random_or_noop,
        "mse_delta_vanilla_minus_slwm": float(vanilla["mse"] - slwm["mse"]),
        "mse_delta_no_spectral_minus_slwm": float(no_spectral["mse"] - slwm["mse"]),
        "spectral_delta_no_spectral_minus_slwm": float(no_spectral["spectral_loss"] - slwm["spectral_loss"]),
        "phase_delta_no_spectral_minus_slwm": float(no_spectral["phase_error"] - slwm["phase_error"]),
    }


def _run_one_task(config: Mapping[str, Any], *, task: str, task_index: int) -> tuple[dict[str, Any], dict[str, np.ndarray], SyntheticSignalBatch]:
    train_cfg = _training_config(config)
    steps = int(train_cfg.get("steps", 100))
    learning_rate = float(train_cfg.get("learning_rate", 0.003))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip_norm_raw = train_cfg.get("grad_clip_norm", 1.0)
    grad_clip_norm = None if grad_clip_norm_raw is None else float(grad_clip_norm_raw)
    overfit_batch = bool(train_cfg.get("overfit_batch", True))
    seed = _runtime_seed(config) + 1000 * (task_index + 1)

    train_batch = make_batches_from_config(config, task=task, split="train")
    eval_batch = train_batch if overfit_batch else make_batches_from_config(config, task=task, split="eval")

    variants: dict[str, Any] = {
        "slwm": NumpySLWMSignalPredictor(_slwm_config(config, use_spectral_mixer=True, seed_offset=10 + task_index)),
        "slwm_no_spectral": NumpySLWMSignalPredictor(_slwm_config(config, use_spectral_mixer=False, seed_offset=20 + task_index)),
        "vanilla_transformer": NumpyContinuousSignalTransformerBaseline(_vanilla_config(config, seed_offset=30 + task_index)),
    }
    task_metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for variant_name, model in variants.items():
        initial_metrics, _ = _evaluate_model(model, eval_batch)
        optimizer = model.make_optimizer(learning_rate=learning_rate, weight_decay=weight_decay, grad_clip_norm=grad_clip_norm)
        training = _train_fixed_batch(model=model, optimizer=optimizer, batch=train_batch, eval_batch=eval_batch, steps=steps)
        final_metrics, prediction = _evaluate_model(model, eval_batch)
        task_metrics[variant_name] = {
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "training": training,
            "parameter_count": model.parameter_count(),
        }
        predictions[variant_name] = prediction

    random_pred = _random_prediction(eval_batch, seed=seed + 99)
    noop_pred = _noop_prediction(eval_batch)
    predictions["random_signal"] = random_pred
    predictions["noop_signal"] = noop_pred
    task_metrics["random_signal"] = {
        "initial_metrics": None,
        "final_metrics": prediction_metric_bundle(random_pred, eval_batch.target_latents, loss_mask=eval_batch.loss_mask),
        "training": {"steps": 0, "throughput_samples_per_second": None, "stability": {"nan_or_inf": False, "loss_explosion": False}},
        "parameter_count": 0,
    }
    task_metrics["noop_signal"] = {
        "initial_metrics": None,
        "final_metrics": prediction_metric_bundle(noop_pred, eval_batch.target_latents, loss_mask=eval_batch.loss_mask),
        "training": {"steps": 0, "throughput_samples_per_second": None, "stability": {"nan_or_inf": False, "loss_explosion": False}},
        "parameter_count": 0,
    }
    task_metrics["comparison"] = _comparison_for_task(task_metrics)
    task_metrics["batch_metadata"] = eval_batch.metadata
    return task_metrics, predictions, eval_batch


def _aggregate_metrics(task_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = flatten_metric_summary(task_results)
    slwm_throughputs: list[float] = []
    all_nan_or_inf = False
    any_loss_explosion = False
    no_spectral_delta_mse: list[float] = []
    no_spectral_delta_spectral: list[float] = []
    vanilla_delta_mse: list[float] = []
    for task_metrics in task_results.values():
        slwm_training = task_metrics["slwm"]["training"]
        slwm_throughputs.append(float(slwm_training["throughput_samples_per_second"]))
        for variant in TRAINABLE_VARIANTS:
            stability = task_metrics[variant]["training"]["stability"]
            all_nan_or_inf = bool(all_nan_or_inf or stability.get("nan_or_inf", False))
            any_loss_explosion = bool(any_loss_explosion or stability.get("loss_explosion", False))
        comparison = task_metrics["comparison"]
        no_spectral_delta_mse.append(float(comparison["mse_delta_no_spectral_minus_slwm"]))
        no_spectral_delta_spectral.append(float(comparison["spectral_delta_no_spectral_minus_slwm"]))
        vanilla_delta_mse.append(float(comparison["mse_delta_vanilla_minus_slwm"]))
    aggregate.update(
        {
            "throughput_samples_per_second": float(np.mean(slwm_throughputs)) if slwm_throughputs else None,
            "nan_or_inf": bool(all_nan_or_inf),
            "loss_explosion": bool(any_loss_explosion),
            "no_spectral_delta_mse": float(np.mean(no_spectral_delta_mse)) if no_spectral_delta_mse else None,
            "spectral_error_delta": float(np.mean(no_spectral_delta_spectral)) if no_spectral_delta_spectral else None,
            "vanilla_delta_mse": float(np.mean(vanilla_delta_mse)) if vanilla_delta_mse else None,
        }
    )
    return aggregate


def _success_gate(task_results: Mapping[str, Mapping[str, Any]], aggregate: Mapping[str, Any]) -> dict[str, Any]:
    wins_vanilla_mse = [task for task, metrics in task_results.items() if metrics["comparison"]["wins_vs_vanilla_mse_metrics"]]
    wins_vanilla_any_metric = [task for task, metrics in task_results.items() if metrics["comparison"]["wins_vs_vanilla_metrics"]]
    wins_vanilla_all_required = [task for task, metrics in task_results.items() if metrics["comparison"]["wins_vs_vanilla_all_required_metrics"]]
    wins_random = [task for task, metrics in task_results.items() if metrics["comparison"]["wins_vs_random_or_noop_metrics"]]
    wins_no_spectral = [task for task, metrics in task_results.items() if metrics["comparison"]["wins_vs_no_spectral_metrics"]]
    slwm_loss_drops = [float(metrics["slwm"]["training"]["loss_drop_percent"]) for metrics in task_results.values()]
    return {
        "tiny_overfit": bool(slwm_loss_drops and max(slwm_loss_drops) > 0.0),
        "tiny_overfit_90_percent_any_task": bool(slwm_loss_drops and max(slwm_loss_drops) >= 90.0),
        "no_nan_or_inf": not bool(aggregate.get("nan_or_inf", True)),
        "no_loss_explosion": not bool(aggregate.get("loss_explosion", True)),
        "slwm_beats_vanilla_on_any_task": bool(wins_vanilla_mse),
        "slwm_tasks_beating_vanilla": wins_vanilla_mse,
        "slwm_tasks_beating_vanilla_count": len(wins_vanilla_mse),
        "slwm_beats_or_matches_vanilla_on_at_least_two_tasks": len(wins_vanilla_mse) >= 2,
        "slwm_beats_vanilla_any_metric_on_any_task": bool(wins_vanilla_any_metric),
        "slwm_tasks_beating_vanilla_any_metric": wins_vanilla_any_metric,
        "slwm_tasks_beating_vanilla_any_metric_count": len(wins_vanilla_any_metric),
        "slwm_tasks_beating_vanilla_all_required_metrics": wins_vanilla_all_required,
        "slwm_tasks_beating_vanilla_all_required_metrics_count": len(wins_vanilla_all_required),
        "slwm_tasks_beating_random_or_noop": wins_random,
        "slwm_tasks_beating_random_or_noop_count": len(wins_random),
        "slwm_tasks_beating_no_spectral": wins_no_spectral,
        "slwm_tasks_beating_no_spectral_count": len(wins_no_spectral),
        "spectral_ablation_measured": all("slwm_no_spectral" in metrics for metrics in task_results.values()),
        "phase_frequency_metrics_reported": all(
            "phase_error" in metrics["slwm"]["final_metrics"] and "frequency_recovery_error" in metrics["slwm"]["final_metrics"]
            for metrics in task_results.values()
        ),
        "failure_report_written": False,
    }


def _write_comparison_csv(task_results: Mapping[str, Mapping[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "variant", "mse", "spectral_loss", "phase_error", "frequency_recovery_error", "loss_drop_percent", "throughput_samples_per_second"])
        for task, metrics in task_results.items():
            for variant in (*TRAINABLE_VARIANTS, *CONTROL_VARIANTS):
                final = metrics[variant]["final_metrics"]
                training = metrics[variant]["training"]
                writer.writerow(
                    [
                        task,
                        variant,
                        final.get("mse"),
                        final.get("spectral_loss"),
                        final.get("phase_error"),
                        final.get("frequency_recovery_error"),
                        training.get("loss_drop_percent"),
                        training.get("throughput_samples_per_second"),
                    ]
                )
    tmp_path.replace(path)
    return path


def _points(values: np.ndarray, *, width: int, height: int, min_v: float, max_v: float) -> str:
    if values.size == 1:
        xs = np.array([0.0])
    else:
        xs = np.linspace(0.0, float(width), num=values.size)
    denom = max(max_v - min_v, 1e-9)
    ys = float(height) - ((values - min_v) / denom) * float(height)
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys, strict=True))


def _write_prediction_preview(
    *,
    artifact_dir: Path,
    task: str,
    batch: SyntheticSignalBatch,
    predictions: Mapping[str, np.ndarray],
) -> dict[str, str]:
    """Write lightweight CSV and SVG prediction-vs-target artifacts."""

    safe_task = task.replace("/", "_")
    csv_path = _safe_artifact_path(artifact_dir / f"prediction_preview_{safe_task}.csv")
    svg_path = _safe_artifact_path(artifact_dir / f"prediction_preview_{safe_task}.svg")
    series = {
        "input": np.asarray(batch.input_latents[0, :, 0], dtype=np.float64),
        "target": np.asarray(batch.target_latents[0, :, 0], dtype=np.float64),
        "slwm": np.asarray(predictions["slwm"][0, :, 0], dtype=np.float64),
        "slwm_no_spectral": np.asarray(predictions["slwm_no_spectral"][0, :, 0], dtype=np.float64),
        "vanilla_transformer": np.asarray(predictions["vanilla_transformer"][0, :, 0], dtype=np.float64),
        "noop_signal": np.asarray(predictions["noop_signal"][0, :, 0], dtype=np.float64),
    }
    random_series = np.asarray(predictions["random_signal"][0, :, 0], dtype=np.float64)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", *series.keys(), "random_signal"])
        for index in range(series["target"].size):
            writer.writerow([index, *(float(values[index]) for values in series.values()), float(random_series[index])])
    tmp_csv.replace(csv_path)

    all_values = np.concatenate([*series.values(), random_series])
    min_v = float(np.min(all_values))
    max_v = float(np.max(all_values))
    width = 640
    height = 240
    colors = {
        "input": "#777777",
        "target": "#111111",
        "slwm": "#1f77b4",
        "slwm_no_spectral": "#ff7f0e",
        "vanilla_transformer": "#2ca02c",
        "noop_signal": "#9467bd",
    }
    polylines = []
    for name, values in series.items():
        points = _points(values, width=width, height=height, min_v=min_v, max_v=max_v)
        polylines.append(f'<polyline points="{points}" fill="none" stroke="{colors[name]}" stroke-width="1.7" />')
    legend = " ".join(html.escape(name) for name in series)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 48}" viewBox="0 0 {width} {height + 48}">
  <rect x="0" y="0" width="{width}" height="{height + 48}" fill="white" />
  <text x="8" y="18" font-family="monospace" font-size="12">T0 {html.escape(task)} prediction preview: {legend}</text>
  <g transform="translate(0,32)">
    <line x1="0" y1="{height / 2:.2f}" x2="{width}" y2="{height / 2:.2f}" stroke="#dddddd" stroke-width="1" />
    {' '.join(polylines)}
  </g>
</svg>
"""
    _write_text(svg, svg_path)
    return {"csv": str(csv_path), "svg": str(svg_path)}


def _failure_report(experiment_id: str, task_results: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any]) -> str:
    rows = []
    for task, metrics in task_results.items():
        comparison = metrics["comparison"]
        rows.append(
            f"| {task} | {comparison['wins_vs_vanilla_mse_metrics']} | {comparison['wins_vs_vanilla_metrics']} | "
            f"{comparison['mse_delta_vanilla_minus_slwm']:.6g} | "
            f"{comparison['mse_delta_no_spectral_minus_slwm']:.6g} |"
        )
    return "\n".join(
        [
            f"# Sprint T0 Failure Report — {experiment_id}",
            "",
            "## Stop condition",
            "",
            "SLWM did not beat the vanilla continuous Transformer baseline on the primary MSE metric for any controlled synthetic signal task. Per Sprint T0 rules, do not proceed to later training stages from this result.",
            "",
            "## Evidence summary",
            "",
            "| Task | SLWM MSE win | SLWM any-metric wins | Vanilla MSE - SLWM MSE | No-spectral MSE - SLWM MSE |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Gate fields",
            "",
            "```json",
            json.dumps(gate, indent=2, sort_keys=True),
            "```",
            "",
            "## Smallest next action",
            "",
            "Review the T0 architecture/optimization configuration and rerun synthetic-only comparisons before using any text/code/audio/video data.",
        ]
    ) + "\n"


def run_t0_synthetic_pretraining(config_path: str | Path) -> dict[str, Any]:
    """Run Sprint T0 synthetic signal pretraining and write artifacts."""

    path = Path(config_path)
    config = load_config(path)
    experiment_id = _experiment_id(config)
    paths = _artifact_paths(config, experiment_id)
    tasks = _tasks_from_config(config)
    train_cfg = _training_config(config)
    steps = int(train_cfg.get("steps", 100))
    model_counts = _counts_for_variants(config)
    _validate_t0_limits(config, model_counts=model_counts)

    run_start = time.perf_counter()
    task_results: dict[str, Any] = {}
    preview_artifacts: dict[str, dict[str, str]] = {}
    for task_index, task in enumerate(tasks):
        metrics, predictions, batch = _run_one_task(config, task=task, task_index=task_index)
        task_results[task] = metrics
        preview_artifacts[task] = _write_prediction_preview(artifact_dir=paths["artifact_dir"], task=task, batch=batch, predictions=predictions)
    wall_clock = time.perf_counter() - run_start
    aggregate = _aggregate_metrics(task_results)
    aggregate["wall_clock_time_seconds"] = float(wall_clock)
    gate = _success_gate(task_results, aggregate)

    failure_modes: list[str] = []
    result_summary: str
    next_allowed_step: str
    if not gate["slwm_beats_vanilla_on_any_task"]:
        gate["failure_report_written"] = True
        failure_modes.append("SLWM did not beat the vanilla Transformer baseline on any controlled T0 task.")
        _write_text(_failure_report(experiment_id, task_results, gate), paths["failure_report"])
        result_summary = "Sprint T0 stop condition triggered on primary MSE; failure report written."
        next_allowed_step = "Stop before later training stages; revise and rerun synthetic-only T0."
    else:
        paths["failure_report"].unlink(missing_ok=True)
        result_summary = "Sprint T0 synthetic comparison completed; SLWM beat vanilla on primary MSE for at least one controlled task."
        next_allowed_step = "Proceed only to formal signal evaluation/finding review; do not claim multimodal or hallucination effects."

    metrics_payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "sprint": "T0",
        "config_hash": config_hash(config),
        "tasks": tasks,
        "model_parameter_counts": model_counts,
        "task_metrics": task_results,
        "aggregate": aggregate,
        "success_gate": gate,
        "preview_artifacts": preview_artifacts,
        "wall_clock_time_seconds": float(wall_clock),
        "hardware": "cpu_numpy",
        "eval_script_hash": _file_sha256(Path(__file__).resolve()),
        "result_summary": result_summary,
        "next_allowed_step": next_allowed_step,
        "failure_modes_observed": failure_modes,
        "docs_rules_status": {
            "docs_rules_found": False,
            "contributing_found": False,
            "note": "No docs/rules/*.md or CONTRIBUTING.md files were present when T0 was implemented; project docs and AGENTS.md were used.",
        },
    }
    _write_json(config, paths["config_copy"])
    _write_json(metrics_payload, paths["metrics"])
    _write_comparison_csv(task_results, paths["comparison_csv"])
    registry_entry = make_t0_synthetic_registry_entry(
        experiment_id=experiment_id,
        config_path=str(path),
        config=config,
        metrics=metrics_payload,
        model_parameter_counts=model_counts,
        training_steps=steps,
        train_samples=steps * int(train_cfg.get("batch_size", 4)) * len(tasks),
        working_tree_state="dirty",
    )
    write_registry_entry(registry_entry, paths["registry"])

    metrics_payload.update(
        {
            "metrics_path": str(paths["metrics"]),
            "registry_path": str(paths["registry"]),
            "comparison_path": str(paths["comparison_csv"]),
            "failure_report_path": str(paths["failure_report"]) if gate["failure_report_written"] else None,
            "config_copy_path": str(paths["config_copy"]),
        }
    )
    _write_json(metrics_payload, paths["metrics"])
    return metrics_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Sprint T0 synthetic signal pretraining")
    parser.add_argument("--config", required=True, help="Path to a T0 JSON config")
    args = parser.parse_args(argv)
    metrics = run_t0_synthetic_pretraining(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())


__all__ = ["run_t0_synthetic_pretraining"]
