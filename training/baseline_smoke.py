"""Config-driven tiny overfit smoke runs for Sprint I1 baselines.

The smoke runner proves only that baseline implementations can instantiate,
run forward/backward, reduce loss on a fixed tiny batch, and write a registry
entry. It does not perform large-scale training or make SLWM quality claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from models.baselines.gpt2_decoder import GPT2DecoderConfig, NumpyGPT2DecoderBaseline
from models.baselines.null_random import RandomLogitBaseline, UniformLogitBaseline, shuffled_targets
from models.baselines.numpy_nn import cross_entropy_loss
from models.baselines.parameter_count import (
    gpt2_module_counts_for_registry,
    multimodal_module_counts_for_registry,
)
from models.baselines.vanilla_multimodal_transformer import NumpyVanillaMultimodalTransformerBaseline, VanillaMultimodalConfig
from utils.config import config_hash, load_config
from utils.experiment_registry import make_i1_baseline_registry_entry, write_registry_entry


ARTIFACT_ROOT = Path("experiments/baselines")
MAX_SMOKE_PARAMETERS = 10_000_000
MAX_SMOKE_STEPS = 1_000
MAX_SMOKE_BATCH = 64
MAX_SMOKE_CONTEXT = 256
MAX_SMOKE_WIDTH = 256
MAX_SMOKE_LAYERS = 4
MAX_SMOKE_VOCAB = 8_192


def _runtime_seed(config: Mapping[str, Any]) -> int:
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    return int(runtime.get("seed", 0))


def _training_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("training", {}) if isinstance(config.get("training", {}), Mapping) else {}


def _registry_path(config: Mapping[str, Any], experiment_id: str) -> Path:
    registry = config.get("registry", {}) if isinstance(config.get("registry", {}), Mapping) else {}
    return _safe_artifact_path(str(registry.get("path", f"experiments/baselines/{experiment_id}.json")))


def _metrics_path(config: Mapping[str, Any], experiment_id: str) -> Path:
    registry = config.get("registry", {}) if isinstance(config.get("registry", {}), Mapping) else {}
    return _safe_artifact_path(str(registry.get("metrics_path", f"experiments/baselines/{experiment_id}.metrics.json")))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_artifact_path(raw_path: str) -> Path:
    """Validate baseline smoke artifact paths before writing.

    Only relative paths under ``experiments/baselines`` are allowed. This keeps
    config-driven smoke runs from clobbering arbitrary local files.
    """

    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("baseline smoke artifact paths must be relative")
    repo_root = Path.cwd().resolve()
    artifact_root = (repo_root / ARTIFACT_ROOT).resolve()
    resolved = (repo_root / path).resolve()
    if not _is_relative_to(resolved, artifact_root):
        raise ValueError(f"baseline smoke artifacts must be written under {ARTIFACT_ROOT}")
    if resolved.exists() and resolved.is_symlink():
        raise ValueError("baseline smoke artifact path must not be a symlink")
    return path


def _write_metrics(metrics: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _validate_common_smoke_limits(*, steps: int, batch_size: int, context_length: int, n_layer: int, n_embd: int, parameter_count: int) -> None:
    if steps > MAX_SMOKE_STEPS:
        raise ValueError(f"baseline smoke steps must be <= {MAX_SMOKE_STEPS}; got {steps}")
    if batch_size > MAX_SMOKE_BATCH:
        raise ValueError(f"baseline smoke batch_size must be <= {MAX_SMOKE_BATCH}; got {batch_size}")
    if context_length > MAX_SMOKE_CONTEXT:
        raise ValueError(f"baseline smoke context_length must be <= {MAX_SMOKE_CONTEXT}; got {context_length}")
    if n_layer > MAX_SMOKE_LAYERS:
        raise ValueError(f"baseline smoke n_layer must be <= {MAX_SMOKE_LAYERS}; got {n_layer}")
    if n_embd > MAX_SMOKE_WIDTH:
        raise ValueError(f"baseline smoke n_embd must be <= {MAX_SMOKE_WIDTH}; got {n_embd}")
    if parameter_count > MAX_SMOKE_PARAMETERS:
        raise ValueError(f"baseline smoke parameter_count must be <= {MAX_SMOKE_PARAMETERS}; got {parameter_count}")


def _validate_gpt2_smoke_limits(config: GPT2DecoderConfig, *, steps: int, batch_size: int) -> None:
    if config.vocab_size > MAX_SMOKE_VOCAB:
        raise ValueError(f"baseline smoke vocab_size must be <= {MAX_SMOKE_VOCAB}; got {config.vocab_size}")
    _validate_common_smoke_limits(
        steps=steps,
        batch_size=batch_size,
        context_length=config.context_length,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        parameter_count=config.parameter_breakdown().total,
    )


def _validate_multimodal_smoke_limits(config: VanillaMultimodalConfig, *, steps: int, batch_size: int) -> None:
    if config.text_vocab_size > MAX_SMOKE_VOCAB or config.target_vocab_size > MAX_SMOKE_VOCAB:
        raise ValueError("baseline smoke text/target vocab sizes exceed configured smoke cap")
    _validate_common_smoke_limits(
        steps=steps,
        batch_size=batch_size,
        context_length=config.context_length,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        parameter_count=config.parameter_breakdown().total,
    )


def make_gpt2_tiny_batch(config: GPT2DecoderConfig, *, batch_size: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic tiny text/code token IDs for overfit smoke.

    Shape contract:
        returns ``input_ids`` and ``target_ids`` with shape ``IntTensor[B,T]``.
    """

    if length > config.context_length:
        raise ValueError("tiny batch length exceeds model context_length")
    # Structured next-token pattern is intentionally simple: the smoke test checks
    # backprop/overfit mechanics, not language quality.
    full = (np.arange(batch_size * (length + 1), dtype=np.int64).reshape(batch_size, length + 1) + 1) % config.vocab_size
    return full[:, :length], full[:, 1:]


def make_multimodal_tiny_batch(config: VanillaMultimodalConfig, *, batch_size: int, seed: int) -> dict[str, np.ndarray]:
    """Create deterministic text/audio/visual features for multimodal overfit.

    Shape contract:
        returns text ``IntTensor[B,T_text]``, audio
        ``FloatTensor[B,T_audio,A]``, visual ``FloatTensor[B,T_visual,V]``, and
        target IDs ``IntTensor[B,T_total]``.
    """

    rng = np.random.default_rng(seed)
    text_len = 3
    audio_len = 3
    visual_len = 3
    if text_len + audio_len + visual_len > config.context_length:
        raise ValueError("tiny multimodal batch exceeds context_length")
    text_tokens = rng.integers(0, config.text_vocab_size, size=(batch_size, text_len), dtype=np.int64)
    audio_features = rng.normal(0.0, 1.0, size=(batch_size, audio_len, config.audio_feature_dim)).astype(np.float64)
    visual_features = rng.normal(0.0, 1.0, size=(batch_size, visual_len, config.visual_feature_dim)).astype(np.float64)

    text_targets = text_tokens % config.target_vocab_size
    audio_targets = (np.argmax(audio_features, axis=-1) + 3) % config.target_vocab_size
    visual_targets = (np.argmax(visual_features, axis=-1) + 7) % config.target_vocab_size
    target_ids = np.concatenate([text_targets, audio_targets, visual_targets], axis=1).astype(np.int64)
    return {
        "text_tokens": text_tokens,
        "audio_features": audio_features,
        "visual_features": visual_features,
        "target_ids": target_ids,
    }


def _train_loop(
    *,
    steps: int,
    optimizer: Any,
    eval_loss: Callable[[], float],
    backward_step: Callable[[], float],
) -> dict[str, Any]:
    """Run deterministic fixed-batch overfit training."""

    initial_loss = eval_loss()
    losses: list[float] = [initial_loss]
    grad_norms: list[float] = []
    nan_or_inf = not np.isfinite(initial_loss)
    loss_explosion = False
    for _ in range(int(steps)):
        optimizer.zero_grad()
        loss = backward_step()
        if not np.isfinite(loss):
            nan_or_inf = True
            break
        grad_norm = optimizer.step()
        grad_norms.append(float(grad_norm))
        losses.append(float(loss))
        if loss > initial_loss * 10.0:
            loss_explosion = True
    final_loss = eval_loss()
    losses.append(final_loss)
    if not np.isfinite(final_loss):
        nan_or_inf = True
    drop_percent = 100.0 * (initial_loss - final_loss) / max(initial_loss, 1e-12)
    return {
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "loss_drop_percent": float(drop_percent),
        "losses": losses,
        "grad_norms": grad_norms,
        "nan_or_inf": bool(nan_or_inf),
        "loss_explosion": bool(loss_explosion),
    }


def run_gpt2_smoke(config: Mapping[str, Any], *, config_path: str) -> dict[str, Any]:
    """Run the GPT-2-style tiny overfit smoke test and write registry output."""

    model_cfg_map = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    runtime_seed = _runtime_seed(config)
    model_cfg = GPT2DecoderConfig.from_mapping({**model_cfg_map, "seed": runtime_seed})
    train_cfg = _training_config(config)
    batch_size = int(train_cfg.get("batch_size", 4))
    length = int(train_cfg.get("sequence_length", min(8, model_cfg.context_length)))
    steps = int(train_cfg.get("steps", 200))
    learning_rate = float(train_cfg.get("learning_rate", 0.01))
    grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    grad_clip = None if grad_clip_norm is None else float(grad_clip_norm)
    experiment_id = str(config.get("registry", {}).get("experiment_id", "EXP-I1-001")) if isinstance(config.get("registry", {}), Mapping) else "EXP-I1-001"
    registry_path = _registry_path(config, experiment_id)
    metrics_path = _metrics_path(config, experiment_id)

    _validate_gpt2_smoke_limits(model_cfg, steps=steps, batch_size=batch_size)
    model = NumpyGPT2DecoderBaseline(model_cfg)
    optimizer = model.make_optimizer(learning_rate=learning_rate, weight_decay=float(train_cfg.get("weight_decay", 0.0)), grad_clip_norm=grad_clip)
    input_ids, target_ids = make_gpt2_tiny_batch(model_cfg, batch_size=batch_size, length=length)

    def eval_loss() -> float:
        logits = model.forward(input_ids)
        loss, _ = cross_entropy_loss(logits, target_ids)
        return loss

    def backward_step() -> float:
        loss, _ = model.loss_and_backward(input_ids, target_ids)
        return loss

    start = time.perf_counter()
    metrics = _train_loop(steps=steps, optimizer=optimizer, eval_loss=eval_loss, backward_step=backward_step)
    wall_clock = time.perf_counter() - start
    breakdown = model_cfg.parameter_breakdown()
    module_counts = gpt2_module_counts_for_registry(breakdown)
    instantiated_count = model.parameter_count()
    metrics.update(
        {
            "primary_metric": "loss_drop_percent",
            "perplexity": float(np.exp(min(metrics["final_loss"], 50.0))),
            "parameter_count": int(instantiated_count),
            "formula_parameter_count": int(breakdown.total),
            "parameter_breakdown": breakdown.as_dict(),
            "context_length": model_cfg.context_length,
            "tokenizer": model_cfg.tokenizer,
            "codec_choices": {"text": model_cfg.tokenizer, "audio": None, "visual": None},
            "config_hash": config_hash(config),
            "wall_clock_time_seconds": float(wall_clock),
            "hardware": "cpu_numpy",
            "eval_script_hash": _file_sha256(Path(__file__).resolve()),
            "result_summary": "GPT-2-style tiny overfit smoke run completed.",
        }
    )
    entry = make_i1_baseline_registry_entry(
        experiment_id=experiment_id,
        config_path=config_path,
        config=config,
        model_name=str(model_cfg_map.get("name", "gpt2_tiny_smoke")),
        model_variant="gpt2_baseline",
        parameter_count=int(instantiated_count),
        module_parameter_counts=module_counts,
        enabled_modalities=["text_code"],
        metrics=metrics,
        training_steps=steps,
        train_tokens_or_samples=int(batch_size * length),
        working_tree_state="dirty",
    )
    write_registry_entry(entry, registry_path)
    _write_metrics(metrics, metrics_path)
    metrics["registry_path"] = str(registry_path)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def run_multimodal_smoke(config: Mapping[str, Any], *, config_path: str) -> dict[str, Any]:
    """Run the vanilla multimodal tiny overfit smoke test and write registry output."""

    model_cfg_map = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    runtime_seed = _runtime_seed(config)
    model_cfg = VanillaMultimodalConfig.from_mapping({**model_cfg_map, "seed": runtime_seed})
    train_cfg = _training_config(config)
    batch_size = int(train_cfg.get("batch_size", 4))
    steps = int(train_cfg.get("steps", 250))
    learning_rate = float(train_cfg.get("learning_rate", 0.01))
    grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    grad_clip = None if grad_clip_norm is None else float(grad_clip_norm)
    experiment_id = str(config.get("registry", {}).get("experiment_id", "EXP-I1-002")) if isinstance(config.get("registry", {}), Mapping) else "EXP-I1-002"
    registry_path = _registry_path(config, experiment_id)
    metrics_path = _metrics_path(config, experiment_id)

    _validate_multimodal_smoke_limits(model_cfg, steps=steps, batch_size=batch_size)
    model = NumpyVanillaMultimodalTransformerBaseline(model_cfg)
    optimizer = model.make_optimizer(learning_rate=learning_rate, weight_decay=float(train_cfg.get("weight_decay", 0.0)), grad_clip_norm=grad_clip)
    batch = make_multimodal_tiny_batch(model_cfg, batch_size=batch_size, seed=runtime_seed)

    def eval_loss() -> float:
        logits = model.forward(
            text_tokens=batch["text_tokens"],
            audio_features=batch["audio_features"],
            visual_features=batch["visual_features"],
        )
        loss, _ = cross_entropy_loss(logits, batch["target_ids"])
        return loss

    def backward_step() -> float:
        loss, _ = model.loss_and_backward(
            text_tokens=batch["text_tokens"],
            audio_features=batch["audio_features"],
            visual_features=batch["visual_features"],
            target_ids=batch["target_ids"],
        )
        return loss

    start = time.perf_counter()
    metrics = _train_loop(steps=steps, optimizer=optimizer, eval_loss=eval_loss, backward_step=backward_step)
    wall_clock = time.perf_counter() - start
    breakdown = model_cfg.parameter_breakdown()
    module_counts = multimodal_module_counts_for_registry(breakdown)
    instantiated_count = model.parameter_count()
    uniform_loss = UniformLogitBaseline(model_cfg.target_vocab_size).evaluate(batch["target_ids"]).loss
    random_loss = RandomLogitBaseline(model_cfg.target_vocab_size, seed=runtime_seed).evaluate(batch["target_ids"]).loss
    shuffled_loss, _ = cross_entropy_loss(model.forward(
        text_tokens=batch["text_tokens"],
        audio_features=batch["audio_features"],
        visual_features=batch["visual_features"],
    ), shuffled_targets(batch["target_ids"], seed=runtime_seed + 1))
    metrics.update(
        {
            "primary_metric": "loss_drop_percent",
            "perplexity": float(np.exp(min(metrics["final_loss"], 50.0))),
            "parameter_count": int(instantiated_count),
            "formula_parameter_count": int(breakdown.total),
            "parameter_breakdown": breakdown.as_dict(),
            "context_length": model_cfg.context_length,
            "tokenizer": model_cfg.text_codec,
            "codec_choices": {
                "text": model_cfg.text_codec,
                "audio": model_cfg.audio_codec_or_features,
                "visual": model_cfg.visual_codec_or_features,
            },
            "uniform_null_loss": float(uniform_loss),
            "random_null_loss": float(random_loss),
            "shuffled_target_loss": float(shuffled_loss),
            "random_or_null_control": True,
            "shuffled_pairs": True,
            "config_hash": config_hash(config),
            "wall_clock_time_seconds": float(wall_clock),
            "hardware": "cpu_numpy",
            "eval_script_hash": _file_sha256(Path(__file__).resolve()),
            "result_summary": "Vanilla multimodal Transformer tiny overfit smoke run completed.",
        }
    )
    entry = make_i1_baseline_registry_entry(
        experiment_id=experiment_id,
        config_path=config_path,
        config=config,
        model_name=str(model_cfg_map.get("name", "vanilla_multimodal_tiny_smoke")),
        model_variant="vanilla_multimodal_transformer",
        parameter_count=int(instantiated_count),
        module_parameter_counts=module_counts,
        enabled_modalities=["text_code", "audio", "visual_video"],
        metrics=metrics,
        training_steps=steps,
        train_tokens_or_samples=int(batch_size * batch["target_ids"].shape[1]),
        working_tree_state="dirty",
    )
    write_registry_entry(entry, registry_path)
    _write_metrics(metrics, metrics_path)
    metrics["registry_path"] = str(registry_path)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def run_baseline_smoke(config_path: str | Path) -> dict[str, Any]:
    """Load a baseline config and run the selected tiny smoke path."""

    path = Path(config_path)
    config = load_config(path)
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    variant = str(model_cfg.get("variant", ""))
    if variant == "gpt2_baseline":
        return run_gpt2_smoke(config, config_path=str(path))
    if variant == "vanilla_multimodal_transformer":
        return run_multimodal_smoke(config, config_path=str(path))
    raise ValueError(f"Unsupported baseline smoke variant {variant!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Sprint I1 baseline tiny-overfit smoke training")
    parser.add_argument("--config", required=True, help="Path to a JSON baseline config")
    args = parser.parse_args(argv)
    metrics = run_baseline_smoke(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())


__all__ = [
    "make_gpt2_tiny_batch",
    "make_multimodal_tiny_batch",
    "run_baseline_smoke",
    "run_gpt2_smoke",
    "run_multimodal_smoke",
]
