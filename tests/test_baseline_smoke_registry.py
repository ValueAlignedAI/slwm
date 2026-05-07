from pathlib import Path
import json

import pytest

from utils.config import load_config
from training.baseline_smoke import run_baseline_smoke


def _with_test_registry(config: dict, experiment_id: str) -> dict:
    updated = dict(config)
    updated["training"] = dict(config["training"])
    updated["training"]["steps"] = 30
    updated["registry"] = {
        "experiment_id": experiment_id,
        "path": f"experiments/baselines/{experiment_id}.test.json",
        "metrics_path": f"experiments/baselines/{experiment_id}.test.metrics.json",
    }
    return updated


def _cleanup(paths: list[str]) -> None:
    for path in paths:
        Path(path).unlink(missing_ok=True)


def test_gpt2_smoke_overfits_and_writes_registry(tmp_path) -> None:
    config = load_config(Path("configs/baselines/gpt2_tiny_smoke.json"))
    config = _with_test_registry(config, "EXP-I1-901")
    config_path = tmp_path / "gpt2_smoke.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        metrics = run_baseline_smoke(config_path)

        assert metrics["final_loss"] < metrics["initial_loss"]
        assert metrics["loss_drop_percent"] > 90.0
        assert Path(metrics["registry_path"]).exists()
        assert Path(metrics["metrics_path"]).exists()
    finally:
        _cleanup(["experiments/baselines/EXP-I1-901.test.json", "experiments/baselines/EXP-I1-901.test.metrics.json"])


def test_multimodal_smoke_overfits_and_writes_registry(tmp_path) -> None:
    config = load_config(Path("configs/baselines/vanilla_multimodal_tiny_smoke.json"))
    config = _with_test_registry(config, "EXP-I1-902")
    config_path = tmp_path / "multimodal_smoke.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        metrics = run_baseline_smoke(config_path)

        assert metrics["final_loss"] < metrics["initial_loss"]
        assert metrics["loss_drop_percent"] > 90.0
        assert metrics["uniform_null_loss"] > metrics["final_loss"]
        assert Path(metrics["registry_path"]).exists()
        assert Path(metrics["metrics_path"]).exists()
    finally:
        _cleanup(["experiments/baselines/EXP-I1-902.test.json", "experiments/baselines/EXP-I1-902.test.metrics.json"])


def test_smoke_runner_rejects_artifact_path_outside_allowlist(tmp_path) -> None:
    config = load_config(Path("configs/baselines/gpt2_tiny_smoke.json"))
    config["training"] = dict(config["training"])
    config["training"]["steps"] = 1
    config["registry"] = {
        "experiment_id": "EXP-I1-903",
        "path": "../EXP-I1-903.json",
        "metrics_path": "experiments/baselines/EXP-I1-903.metrics.json",
    }
    config_path = tmp_path / "bad_path.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="experiments/baselines"):
        run_baseline_smoke(config_path)


def test_smoke_runner_rejects_full_scale_config_instantiation(tmp_path) -> None:
    config = load_config(Path("configs/baselines/gpt2_small_style.json"))
    config["training"] = dict(config["training"])
    config["training"].update({"steps": 1, "batch_size": 1, "sequence_length": 8})
    config["registry"] = {
        "experiment_id": "EXP-I1-904",
        "path": "experiments/baselines/EXP-I1-904.test.json",
        "metrics_path": "experiments/baselines/EXP-I1-904.test.metrics.json",
    }
    config_path = tmp_path / "too_large.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="vocab_size|parameter_count|context_length|n_layer|n_embd"):
        run_baseline_smoke(config_path)
