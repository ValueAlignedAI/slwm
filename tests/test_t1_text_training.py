import json
import shutil
from pathlib import Path

import numpy as np

from data import build_text_code_lm_datasets, build_text_tokenizer
from models import NumpySLWMCore, SLWMCoreConfig
from training.t1_text_baseline import run_t1_text_training
from utils.config import load_config


def _pilot_config(path: str, experiment_id: str, *, steps: int = 2) -> dict:
    config = load_config(path)
    config["training"] = dict(config["training"])
    config["training"]["steps"] = steps
    config["registry"] = {
        "experiment_id": experiment_id,
        "artifact_dir": f"experiments/text/t1/{experiment_id}",
        "path": f"experiments/text/t1/{experiment_id}/registry.json",
        "metrics_path": f"experiments/text/t1/{experiment_id}/metrics.json",
        "samples_path": f"experiments/text/t1/{experiment_id}/samples.json",
        "report_path": f"experiments/text/t1/{experiment_id}/report.md",
        "checkpoint_path": f"experiments/text/t1/{experiment_id}/checkpoint.npz",
        "config_copy_path": f"experiments/text/t1/{experiment_id}/config.json",
    }
    return config


def test_t1_slwm_text_lm_forward_backward_has_gradients() -> None:
    config = load_config("configs/t1/slwm_text_tiny_pilot.json")
    tokenizer = build_text_tokenizer(config["model"])
    bundle = build_text_code_lm_datasets(config, tokenizer)
    slwm_cfg = SLWMCoreConfig.from_mapping({**config, "model": {**config["model"], "text_vocab_size": tokenizer.vocab_size}})
    model = NumpySLWMCore(slwm_cfg)
    input_ids, target_ids = bundle.train.batch(batch_size=2, step=0)

    optimizer = model.make_optimizer(learning_rate=0.001)
    optimizer.zero_grad()
    loss, output = model.text_lm_loss_and_backward(input_ids, target_ids)
    grad_norm = sum(float(np.sum(np.abs(param.grad))) for param in model.parameters())

    assert np.isfinite(loss)
    assert output["text_logits"].shape == (2, config["training"]["sequence_length"], tokenizer.vocab_size)
    assert grad_norm > 0.0
    assert np.isfinite(optimizer.step())


def test_t1_runner_writes_registry_metrics_samples_and_checkpoint(tmp_path) -> None:
    experiment_id = "EXP-T1-901"
    config = _pilot_config("configs/t1/gpt2_text_tiny_pilot.json", experiment_id, steps=2)
    config_path = tmp_path / "t1_gpt2_test.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact_dir = Path(f"experiments/text/t1/{experiment_id}")

    try:
        metrics = run_t1_text_training(config_path)
        registry = json.loads(Path(metrics["registry_path"]).read_text(encoding="utf-8"))
        samples = json.loads(Path(metrics["samples_path"]).read_text(encoding="utf-8"))

        assert metrics["sprint"] == "T1"
        assert metrics["validation_loss"] > 0.0
        assert metrics["validation_perplexity"] > 1.0
        assert metrics["throughput_tokens_per_second"] > 0.0
        assert Path(metrics["checkpoint_path"]).exists()
        assert Path(metrics["metrics_path"]).exists()
        assert Path(metrics["report_path"]).exists()
        assert registry["sprint"]["id"] == "T1"
        assert registry["model"]["enabled_modalities"] == ["text_code"]
        assert registry["data"]["preprocessing"]["audio_codec_or_features"] is None
        assert registry["data"]["preprocessing"]["visual_codec_or_features"] is None
        assert samples["samples"][0]["decoding_settings"]["max_new_tokens"] == config["generation"]["max_new_tokens"]
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


def test_t1_runner_rejects_artifact_path_outside_allowlist(tmp_path) -> None:
    config = _pilot_config("configs/t1/gpt2_text_tiny_pilot.json", "EXP-T1-902", steps=1)
    config["registry"]["path"] = "experiments/baselines/not_allowed.json"
    config_path = tmp_path / "bad_t1_path.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        try:
            run_t1_text_training(config_path)
        except ValueError as exc:
            assert "experiments/text/t1" in str(exc)
        else:  # pragma: no cover - explicit guard
            raise AssertionError("expected bad T1 artifact path to be rejected")
    finally:
        shutil.rmtree(Path("experiments/text/t1/EXP-T1-902"), ignore_errors=True)
