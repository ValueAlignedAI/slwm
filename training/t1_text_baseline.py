"""Sprint T1 text/code baseline training runner.

This runner is intentionally NumPy/dependency-light so the repository can prove
the T1 plumbing: same tokenizer/splits, text-only data, next-token CE, validation
loss/PPL, sample generations, throughput, memory, checkpoints, and registry
entries.  Full GPT-2-size training remains a compute/dataset task; tiny pilot
configs are labelled as non-claim smoke evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from data.text_code import TextCodeDatasetBundle, TokenWindowDataset, build_text_code_lm_datasets
from data.tokenizer import TextTokenizer, build_text_tokenizer
from models.baselines.gpt2_decoder import GPT2DecoderConfig, NumpyGPT2DecoderBaseline
from models.baselines.numpy_nn import cross_entropy_loss, softmax
from models.baselines.parameter_count import gpt2_module_counts_for_registry
from models.slwm_config import SLWMCoreConfig
from models.slwm_core import NumpySLWMCore
from models.slwm_parameter_count import slwm_parameter_breakdown_from_config
from utils.config import config_hash, load_config, write_config
from utils.experiment_registry import make_t1_text_registry_entry, write_registry_entry


ARTIFACT_ROOT = Path("experiments/text/t1")
MAX_T1_PILOT_PARAMETERS = 15_000_000
MAX_T1_PILOT_STEPS = 2_000
MAX_T1_PILOT_BATCH = 64
MAX_T1_PILOT_CONTEXT = 256
MAX_T1_PILOT_WIDTH = 256
MAX_T1_PILOT_LAYERS = 4


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_artifact_path(raw_path: str) -> Path:
    """Validate T1 artifact paths before writing.

    Only relative paths under ``experiments/text/t1`` are allowed.  This mirrors
    the I1/T0 runners and prevents config-driven writes outside the research
    artifact tree.
    """

    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError("T1 artifact paths must be relative")
    repo_root = Path.cwd().resolve()
    artifact_root = (repo_root / ARTIFACT_ROOT).resolve()
    resolved = (repo_root / path).resolve()
    if not _is_relative_to(resolved, artifact_root):
        raise ValueError(f"T1 artifacts must be written under {ARTIFACT_ROOT}")
    if resolved.exists() and resolved.is_symlink():
        raise ValueError("T1 artifact path must not be a symlink")
    return path


def _write_json(payload: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _write_text(payload: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _max_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _train_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("training", {}) if isinstance(config.get("training", {}), Mapping) else {}


def _runtime_seed(config: Mapping[str, Any]) -> int:
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime", {}), Mapping) else {}
    model = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    return int(runtime.get("seed", model.get("seed", 0)))


def _registry_paths(config: Mapping[str, Any], experiment_id: str) -> dict[str, Path]:
    registry = config.get("registry", {}) if isinstance(config.get("registry", {}), Mapping) else {}
    artifact_dir = _safe_artifact_path(str(registry.get("artifact_dir", f"experiments/text/t1/{experiment_id}")))
    return {
        "artifact_dir": artifact_dir,
        "registry": _safe_artifact_path(str(registry.get("path", f"experiments/text/t1/{experiment_id}/registry.json"))),
        "metrics": _safe_artifact_path(str(registry.get("metrics_path", f"experiments/text/t1/{experiment_id}/metrics.json"))),
        "samples": _safe_artifact_path(str(registry.get("samples_path", f"experiments/text/t1/{experiment_id}/samples.json"))),
        "report": _safe_artifact_path(str(registry.get("report_path", f"experiments/text/t1/{experiment_id}/report.md"))),
        "checkpoint": _safe_artifact_path(str(registry.get("checkpoint_path", f"experiments/text/t1/{experiment_id}/checkpoint.npz"))),
        "config_copy": _safe_artifact_path(str(registry.get("config_copy_path", f"experiments/text/t1/{experiment_id}/config.json"))),
    }


def _validate_pilot_limits(*, steps: int, batch_size: int, context_length: int, n_layer: int, width: int, parameter_count: int) -> None:
    if steps > MAX_T1_PILOT_STEPS:
        raise ValueError(f"T1 pilot steps must be <= {MAX_T1_PILOT_STEPS}; got {steps}")
    if batch_size > MAX_T1_PILOT_BATCH:
        raise ValueError(f"T1 pilot batch_size must be <= {MAX_T1_PILOT_BATCH}; got {batch_size}")
    if context_length > MAX_T1_PILOT_CONTEXT:
        raise ValueError(f"T1 pilot context_length must be <= {MAX_T1_PILOT_CONTEXT}; got {context_length}")
    if n_layer > MAX_T1_PILOT_LAYERS:
        raise ValueError(f"T1 pilot n_layer must be <= {MAX_T1_PILOT_LAYERS}; got {n_layer}")
    if width > MAX_T1_PILOT_WIDTH:
        raise ValueError(f"T1 pilot model width must be <= {MAX_T1_PILOT_WIDTH}; got {width}")
    if parameter_count > MAX_T1_PILOT_PARAMETERS:
        raise ValueError(f"T1 pilot parameter_count must be <= {MAX_T1_PILOT_PARAMETERS}; got {parameter_count}")


def _model_variant(config: Mapping[str, Any]) -> str:
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    return str(model_cfg.get("variant", ""))


def _build_gpt2_model(config: Mapping[str, Any], tokenizer: TextTokenizer) -> tuple[NumpyGPT2DecoderBaseline, dict[str, int], int, str]:
    model_cfg_map = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    cfg = GPT2DecoderConfig.from_mapping({**model_cfg_map, "seed": _runtime_seed(config), "vocab_size": tokenizer.vocab_size})
    model = NumpyGPT2DecoderBaseline(cfg)
    breakdown = cfg.parameter_breakdown()
    return model, gpt2_module_counts_for_registry(breakdown), model.parameter_count(), str(model_cfg_map.get("name", "gpt2_t1_text"))


def _build_slwm_model(config: Mapping[str, Any], tokenizer: TextTokenizer) -> tuple[NumpySLWMCore, dict[str, int], int, str, SLWMCoreConfig]:
    model_cfg_map = config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else {}
    cfg = SLWMCoreConfig.from_mapping({**config, "model": {**model_cfg_map, "text_vocab_size": tokenizer.vocab_size, "vocab_size": tokenizer.vocab_size}})
    if not cfg.use_text_lm_head:
        raise ValueError("T1 SLWM text-only configs must set architecture_flags.use_text_lm_head=true")
    model = NumpySLWMCore(cfg)
    breakdown = model.parameter_count_breakdown()
    module_counts = breakdown.registry_module_counts()
    # T1 trains text_code only, but the current tiny SLWM core still instantiates
    # audio/visual adapter parameters. Keep them in strict total accounting while
    # labeling them inactive to avoid hiding parameter-budget ambiguity.
    module_counts["text_code_adapter"] = int(breakdown.adapters.get("text_code", 0))
    module_counts["inactive_audio_adapter"] = int(breakdown.adapters.get("audio", 0))
    module_counts["inactive_visual_video_adapter"] = int(breakdown.adapters.get("visual_video", 0))
    module_counts["inactive_adapter_parameters"] = module_counts["inactive_audio_adapter"] + module_counts["inactive_visual_video_adapter"]
    return model, module_counts, breakdown.total, str(model_cfg_map.get("name", "slwm_t1_text_only")), cfg


def _evaluate_loss(
    dataset: TokenWindowDataset,
    *,
    batch_size: int,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    losses: list[float] = []
    total_batches = max(1, math.ceil(dataset.sample_count / max(1, int(batch_size))))
    for step in range(total_batches):
        input_ids, target_ids = dataset.batch(batch_size=batch_size, step=step)
        losses.append(float(loss_fn(input_ids, target_ids)))
    return float(np.mean(losses))


def _softmax_sample(logits: np.ndarray, *, rng: np.random.Generator, temperature: float, top_k: int | None, top_p: float | None) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))
    scaled = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-8)
    if top_k is not None and int(top_k) > 0 and int(top_k) < scaled.size:
        keep = np.argpartition(scaled, -int(top_k))[-int(top_k):]
        masked = np.full_like(scaled, -1e12)
        masked[keep] = scaled[keep]
        scaled = masked
    probs = softmax(scaled[None, :], axis=-1)[0]
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep_count = max(1, int(np.searchsorted(cumulative, float(top_p), side="left") + 1))
        keep = order[:keep_count]
        filtered = np.zeros_like(probs)
        filtered[keep] = probs[keep]
        probs = filtered / max(float(np.sum(filtered)), 1e-12)
    return int(rng.choice(np.arange(probs.size), p=probs))


def _generate_samples(
    *,
    model: Any,
    variant: str,
    tokenizer: TextTokenizer,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    gen_cfg = config.get("generation", {}) if isinstance(config.get("generation", {}), Mapping) else {}
    prompts = gen_cfg.get("prompts", ["The text model"])
    if not isinstance(prompts, list) or not prompts:
        prompts = ["The text model"]
    max_new_tokens = int(gen_cfg.get("max_new_tokens", 24))
    temperature = float(gen_cfg.get("temperature", 0.0))
    top_k = gen_cfg.get("top_k")
    top_p = gen_cfg.get("top_p")
    stop_on_eos = bool(gen_cfg.get("stop_on_eos", True))
    seed = int(gen_cfg.get("seed", _runtime_seed(config) + 10_000))
    rng = np.random.default_rng(seed)
    context_length = int(config.get("model", {}).get("context_length", config.get("training", {}).get("sequence_length", 16)))

    samples: list[dict[str, Any]] = []
    for prompt in prompts:
        prompt_text = str(prompt)
        token_ids = tokenizer.encode(prompt_text, add_eos=False)
        if not token_ids:
            token_ids = [tokenizer.eos_token_id]
        generated: list[int] = []
        for _ in range(max_new_tokens):
            context = token_ids[-context_length:]
            input_array = np.asarray([context], dtype=np.int64)
            if variant == "gpt2_baseline":
                logits = model.forward(input_array)
            else:
                logits = model.text_lm_logits(input_array)
            next_token = _softmax_sample(
                logits[0, -1, :],
                rng=rng,
                temperature=temperature,
                top_k=None if top_k is None else int(top_k),
                top_p=None if top_p is None else float(top_p),
            )
            token_ids.append(next_token)
            generated.append(next_token)
            if stop_on_eos and next_token == tokenizer.eos_token_id:
                break
        samples.append(
            {
                "prompt": prompt_text,
                "prompt_token_count": len(tokenizer.encode(prompt_text, add_eos=False)),
                "generated_token_ids": generated,
                "generated_text": tokenizer.decode(generated),
                "full_text": tokenizer.decode(token_ids),
                "decoding_settings": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "seed": seed,
                    "stop_on_eos": stop_on_eos,
                },
            }
        )
    return samples


def _save_checkpoint(model: Any, path: Path, *, metadata: Mapping[str, Any]) -> Path:
    params = list(model.parameters())
    arrays: dict[str, np.ndarray] = {f"p{index:04d}_{param.name.replace('.', '_')}": param.value for index, param in enumerate(params)}
    arrays["metadata_json"] = np.asarray(json.dumps(dict(metadata), sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)
    return path


def _baseline_text_loss_from_refs(config: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Load the first referenced GPT-2 baseline validation loss if available."""

    refs = config.get("baselines_compared", [])
    if not isinstance(refs, list):
        return None, None
    for ref in refs:
        if not isinstance(ref, Mapping):
            continue
        experiment_id = str(ref.get("experiment_id", ""))
        if not experiment_id:
            continue
        metrics_path = Path("experiments/text/t1") / experiment_id / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if metrics.get("model_variant") == "gpt2_baseline" and metrics.get("validation_loss") is not None:
            return float(metrics["validation_loss"]), experiment_id
    return None, None


def _report_markdown(metrics: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"# Sprint T1 Report — {metrics['experiment_id']}",
            "",
            f"- Model variant: `{metrics['model_variant']}`",
            f"- Tokenizer: `{metrics['tokenizer']['effective_type']}` (intended: `{metrics['tokenizer'].get('intended_tokenizer')}`)",
            f"- Validation loss: `{metrics['validation_loss']}`",
            f"- Validation perplexity: `{metrics['validation_perplexity']}`",
            f"- Throughput tokens/s: `{metrics['throughput_tokens_per_second']}`",
            f"- Max memory MB: `{metrics['max_memory_mb']}`",
            f"- Checkpoint: `{metrics['checkpoint_path']}`",
            "",
            "## Scope",
            "Text/code only. No audio or visual data was loaded or trained in this sprint run.",
            "",
            "## Claim limits",
            metrics.get("claim_language_allowed", "Only registered T1 metrics may be reported."),
            "",
        ]
    )


def run_t1_text_training(config_path: str | Path) -> dict[str, Any]:
    """Run one registered Sprint T1 text/code training job."""

    path = Path(config_path)
    config = load_config(path)
    variant = _model_variant(config)
    experiment_id = str(config.get("registry", {}).get("experiment_id", "EXP-T1-001")) if isinstance(config.get("registry", {}), Mapping) else "EXP-T1-001"
    paths = _registry_paths(config, experiment_id)
    tokenizer = build_text_tokenizer(config.get("model", {}) if isinstance(config.get("model", {}), Mapping) else config)
    dataset = build_text_code_lm_datasets(config, tokenizer)
    train_cfg = _train_config(config)
    steps = int(train_cfg.get("steps", 50))
    batch_size = int(train_cfg.get("batch_size", 4))
    learning_rate = float(train_cfg.get("learning_rate", 0.005))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    grad_clip = None if grad_clip_norm is None else float(grad_clip_norm)

    if variant == "gpt2_baseline":
        model, module_counts, parameter_count, model_name = _build_gpt2_model(config, tokenizer)
        context_length = model.config.context_length
        width = model.config.n_embd
        n_layer = model.config.n_layer

        def loss_only(input_ids: np.ndarray, target_ids: np.ndarray) -> float:
            loss, _ = cross_entropy_loss(model.forward(input_ids), target_ids)
            return loss

        def backward_step(input_ids: np.ndarray, target_ids: np.ndarray) -> float:
            loss, _ = model.loss_and_backward(input_ids, target_ids)
            return loss

    elif variant in {"slwm_text_only", "slwm_text_only_no_spectral", "slwm_ablation"}:
        model, module_counts, parameter_count, model_name, slwm_cfg = _build_slwm_model(config, tokenizer)
        context_length = slwm_cfg.context_length
        width = slwm_cfg.latent_dim
        n_layer = slwm_cfg.n_layer

        def loss_only(input_ids: np.ndarray, target_ids: np.ndarray) -> float:
            loss, _ = cross_entropy_loss(model.text_lm_logits(input_ids), target_ids)
            return loss

        def backward_step(input_ids: np.ndarray, target_ids: np.ndarray) -> float:
            loss, _ = model.text_lm_loss_and_backward(input_ids, target_ids)
            return loss

    else:
        raise ValueError(f"Unsupported T1 model variant {variant!r}")

    _validate_pilot_limits(
        steps=steps,
        batch_size=batch_size,
        context_length=context_length,
        n_layer=n_layer,
        width=width,
        parameter_count=parameter_count,
    )
    if dataset.train.sequence_length > context_length:
        raise ValueError(f"dataset sequence length {dataset.train.sequence_length} exceeds model context_length {context_length}")

    optimizer = model.make_optimizer(learning_rate=learning_rate, weight_decay=weight_decay, grad_clip_norm=grad_clip)
    initial_train_loss = _evaluate_loss(dataset.train, batch_size=batch_size, loss_fn=loss_only)
    initial_validation_loss = _evaluate_loss(dataset.validation, batch_size=batch_size, loss_fn=loss_only)
    losses: list[float] = [float(initial_train_loss)]
    grad_norms: list[float] = []
    nan_or_inf = not (math.isfinite(initial_train_loss) and math.isfinite(initial_validation_loss))
    loss_explosion = False
    tokens_seen = 0
    start = time.perf_counter()
    for step in range(steps):
        input_ids, target_ids = dataset.train.batch(batch_size=batch_size, step=step)
        optimizer.zero_grad()
        loss = float(backward_step(input_ids, target_ids))
        if not math.isfinite(loss):
            nan_or_inf = True
            break
        grad_norm = float(optimizer.step())
        grad_norms.append(grad_norm)
        losses.append(loss)
        tokens_seen += int(input_ids.size)
        if loss > max(initial_train_loss, 1e-12) * 10.0:
            loss_explosion = True
    wall_clock = time.perf_counter() - start
    train_loss = _evaluate_loss(dataset.train, batch_size=batch_size, loss_fn=loss_only)
    validation_loss = _evaluate_loss(dataset.validation, batch_size=batch_size, loss_fn=loss_only)
    validation_perplexity = float(np.exp(min(float(validation_loss), 50.0)))
    samples = _generate_samples(model=model, variant=variant, tokenizer=tokenizer, config=config)
    tokenizer_metadata = tokenizer.metadata()
    checkpoint_metadata = {
        "experiment_id": experiment_id,
        "model_variant": variant,
        "config_hash": config_hash(config),
        "parameter_count": int(parameter_count),
        "tokenizer": tokenizer_metadata,
    }
    checkpoint_path = _save_checkpoint(model, paths["checkpoint"], metadata=checkpoint_metadata)
    write_config(config, paths["config_copy"])

    throughput = float(tokens_seen / wall_clock) if wall_clock > 0.0 else None
    metrics: dict[str, Any] = {
        "experiment_id": experiment_id,
        "sprint": "T1",
        "model_variant": variant,
        "model_name": model_name,
        "initial_train_loss": float(initial_train_loss),
        "initial_validation_loss": float(initial_validation_loss),
        "train_loss": float(train_loss),
        "validation_loss": float(validation_loss),
        "validation_perplexity": validation_perplexity,
        "losses": losses,
        "grad_norms": grad_norms,
        "nan_or_inf": bool(nan_or_inf),
        "loss_explosion": bool(loss_explosion),
        "parameter_count": int(parameter_count),
        "module_parameter_counts": dict(module_counts),
        "tokenizer": tokenizer_metadata,
        "split_digests": dataset.split_digests(),
        "registry_datasets": dataset.registry_datasets(),
        "train_windows": dataset.train.sample_count,
        "validation_windows": dataset.validation.sample_count,
        "train_tokens_seen": int(tokens_seen),
        "throughput_tokens_per_second": throughput,
        "wall_clock_time_seconds": float(wall_clock),
        "max_memory_mb": float(_max_rss_mb()),
        "hardware": f"cpu_numpy:{platform.platform()}",
        "config_hash": config_hash(config),
        "eval_script_hash": _file_sha256(Path(__file__).resolve()),
        "checkpoint_path": str(checkpoint_path),
        "config_copy_path": str(paths["config_copy"]),
        "samples_path": str(paths["samples"]),
        "claim_language_allowed": (
            "Tiny dependency-light T1 pilot: report validation loss/PPL, samples, throughput, memory, and exact settings only; "
            "do not claim GPT-2-scale text quality or SLWM superiority."
        ),
        "limitations": [
            "Dependency-free pilot tokenizer is byte fallback, not GPT-2 BPE, unless a future config uses an external tokenizer stack.",
            "Inline/local pilot corpus is for pipeline validation; full T1 evidence needs prepared FineWeb/FineWeb-Edu + license-filtered code splits.",
            "No audio or visual data was used; no multimodal, grounding, hallucination, or policy claim is supported.",
        ],
        "result_summary": f"Sprint T1 {variant} text/code pilot completed with validation loss {validation_loss:.6f}.",
        "hypothesis_decision": "guardrail_pass" if variant == "gpt2_baseline" else "untested",
        "next_allowed_step": "Run the companion T1 variants on the exact same tokenizer/splits and compare against EXP-T1-001 before updating G-R0-1.",
        "failure_modes_observed": [],
        "baselines_compared": config.get("baselines_compared", []),
    }
    tolerance = float(train_cfg.get("guardrail_tolerance_percent", 20.0))
    metrics["guardrail_tolerance_percent"] = tolerance
    if variant == "gpt2_baseline":
        metrics["gpt2_baseline_text_validation_loss"] = float(validation_loss)
        metrics["text_loss_relative_delta_percent"] = 0.0
        metrics["guardrail_status"] = "baseline_anchor"
    else:
        baseline_loss, baseline_experiment_id = _baseline_text_loss_from_refs(config)
        metrics["gpt2_baseline_text_validation_loss"] = baseline_loss
        metrics["gpt2_baseline_experiment_id"] = baseline_experiment_id
        if baseline_loss is not None and baseline_loss > 0.0:
            delta = 100.0 * (float(validation_loss) - baseline_loss) / baseline_loss
            metrics["text_loss_relative_delta_percent"] = float(delta)
            guardrail_pass = delta <= tolerance
            metrics["guardrail_status"] = "guardrail_pass" if guardrail_pass else "guardrail_fail"
            metrics["hypothesis_decision"] = metrics["guardrail_status"]
            if not guardrail_pass:
                metrics["failure_modes_observed"].append("text_loss_exceeds_registered_tolerance")
                metrics["result_summary"] += f" Text tradeoff recorded: {delta:.2f}% loss delta vs {baseline_experiment_id}."
                metrics["next_allowed_step"] = (
                    "Record the T1 text tradeoff before proceeding; do not claim text improvement unless a matched rerun beats EXP-T1-001."
                )
        else:
            metrics["text_loss_relative_delta_percent"] = None
            metrics["guardrail_status"] = "baseline_missing"
    if nan_or_inf:
        metrics["failure_modes_observed"].append("nan_or_inf")
    if loss_explosion:
        metrics["failure_modes_observed"].append("loss_explosion")

    metrics["registry_path"] = str(paths["registry"])
    metrics["metrics_path"] = str(paths["metrics"])
    metrics["report_path"] = str(paths["report"])
    _write_json({"samples": samples, "decoding_settings": config.get("generation", {})}, paths["samples"])
    _write_json(metrics, paths["metrics"])
    entry = make_t1_text_registry_entry(
        experiment_id=experiment_id,
        config_path=str(path),
        config=config,
        metrics=metrics,
        model_name=model_name,
        model_variant=variant,
        parameter_count=parameter_count,
        module_parameter_counts=module_counts,
        training_steps=steps,
        train_tokens=tokens_seen,
        checkpoint_path=str(checkpoint_path),
        working_tree_state="dirty",
    )
    write_registry_entry(entry, paths["registry"])
    _write_text(_report_markdown(metrics), paths["report"])
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Sprint T1 text/code baseline training")
    parser.add_argument("--config", required=True, help="Path to a T1 JSON config")
    args = parser.parse_args(argv)
    metrics = run_t1_text_training(args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())


__all__ = ["run_t1_text_training"]
