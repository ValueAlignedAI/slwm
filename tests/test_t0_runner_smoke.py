from pathlib import Path
import json
import shutil

import pytest

from training.t0_synthetic_pretrain import run_t0_synthetic_pretraining
import training.t0_synthetic_pretrain as t0_runner
from training.synthetic_signals import make_synthetic_signal_batch
from utils.config import load_config


def _tiny_test_config(experiment_id: str) -> dict:
    config = load_config(Path("configs/t0/synthetic_tiny.json"))
    config["data"] = dict(config["data"])
    config["data"]["tasks"] = ["sine_mixture", "noisy_periodic_denoising"]
    config["training"] = dict(config["training"])
    config["training"].update({"steps": 3, "batch_size": 1})
    config["registry"] = {
        "experiment_id": experiment_id,
        "artifact_dir": f"experiments/synthetic/t0/{experiment_id}",
        "path": f"experiments/synthetic/t0/{experiment_id}/registry.json",
        "metrics_path": f"experiments/synthetic/t0/{experiment_id}/metrics.json",
        "comparison_path": f"experiments/synthetic/t0/{experiment_id}/comparison_table.csv",
        "failure_report_path": f"experiments/synthetic/t0/{experiment_id}/failure_report.md",
        "config_copy_path": f"experiments/synthetic/t0/{experiment_id}/config.json",
    }
    return config


def test_t0_runner_writes_metrics_registry_and_comparison(tmp_path) -> None:
    experiment_id = "EXP-T0-901"
    config = _tiny_test_config(experiment_id)
    config_path = tmp_path / "t0_smoke.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact_dir = Path(f"experiments/synthetic/t0/{experiment_id}")

    try:
        metrics = run_t0_synthetic_pretraining(config_path)

        assert metrics["sprint"] == "T0"
        assert metrics["tasks"] == ["sine_mixture", "noisy_periodic_denoising"]
        assert Path(metrics["metrics_path"]).exists()
        assert Path(metrics["registry_path"]).exists()
        assert Path(metrics["comparison_path"]).exists()
        assert metrics["success_gate"]["spectral_ablation_measured"] is True
        assert metrics["success_gate"]["phase_frequency_metrics_reported"] is True
        assert "slwm" in metrics["model_parameter_counts"]
        assert metrics["model_parameter_counts"]["slwm_no_spectral"]["total"] < metrics["model_parameter_counts"]["slwm"]["total"]
        for task in metrics["tasks"]:
            assert Path(metrics["preview_artifacts"][task]["csv"]).exists()
            assert Path(metrics["preview_artifacts"][task]["svg"]).exists()
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


def test_t0_runner_rejects_artifact_path_outside_allowlist(tmp_path) -> None:
    config = _tiny_test_config("EXP-T0-902")
    config["registry"]["path"] = "experiments/baselines/not_allowed.json"
    config_path = tmp_path / "bad_t0_path.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="experiments/synthetic/t0"):
        run_t0_synthetic_pretraining(config_path)


def test_t0_runner_writes_failure_report_when_slwm_has_no_mse_win(tmp_path, monkeypatch) -> None:
    experiment_id = "EXP-T0-903"
    config = _tiny_test_config(experiment_id)
    config["data"] = dict(config["data"])
    config["data"]["tasks"] = ["sine_mixture"]
    config["training"] = dict(config["training"])
    config["training"]["steps"] = 1
    config_path = tmp_path / "t0_failure.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact_dir = Path(f"experiments/synthetic/t0/{experiment_id}")

    def fake_run_one_task(_config, *, task: str, task_index: int):
        batch = make_synthetic_signal_batch(task, batch_size=1, context_length=24, latent_dim=4, seed=99)
        zeros = batch.target_latents * 0.0
        predictions = {
            "slwm": zeros,
            "slwm_no_spectral": zeros,
            "vanilla_transformer": zeros,
            "random_signal": zeros,
            "noop_signal": zeros,
        }
        metrics = {
            "slwm": {
                "initial_metrics": {"mse": 3.0, "spectral_loss": 3.0, "phase_error": 3.0, "frequency_recovery_error": 3.0},
                "final_metrics": {"mse": 2.0, "spectral_loss": 2.0, "phase_error": 2.0, "frequency_recovery_error": 2.0},
                "training": {"loss_drop_percent": 1.0, "throughput_samples_per_second": 1.0, "stability": {"nan_or_inf": False, "loss_explosion": False}},
                "parameter_count": 1,
            },
            "slwm_no_spectral": {
                "initial_metrics": {"mse": 3.0, "spectral_loss": 3.0, "phase_error": 3.0, "frequency_recovery_error": 3.0},
                "final_metrics": {"mse": 1.5, "spectral_loss": 1.5, "phase_error": 1.5, "frequency_recovery_error": 1.5},
                "training": {"loss_drop_percent": 1.0, "throughput_samples_per_second": 1.0, "stability": {"nan_or_inf": False, "loss_explosion": False}},
                "parameter_count": 1,
            },
            "vanilla_transformer": {
                "initial_metrics": {"mse": 2.0, "spectral_loss": 2.0, "phase_error": 2.0, "frequency_recovery_error": 2.0},
                "final_metrics": {"mse": 1.0, "spectral_loss": 1.0, "phase_error": 1.0, "frequency_recovery_error": 1.0},
                "training": {"loss_drop_percent": 1.0, "throughput_samples_per_second": 1.0, "stability": {"nan_or_inf": False, "loss_explosion": False}},
                "parameter_count": 1,
            },
            "random_signal": {
                "initial_metrics": None,
                "final_metrics": {"mse": 3.0, "spectral_loss": 3.0, "phase_error": 3.0, "frequency_recovery_error": 3.0},
                "training": {"steps": 0, "throughput_samples_per_second": None, "stability": {"nan_or_inf": False, "loss_explosion": False}},
                "parameter_count": 0,
            },
            "noop_signal": {
                "initial_metrics": None,
                "final_metrics": {"mse": 3.0, "spectral_loss": 3.0, "phase_error": 3.0, "frequency_recovery_error": 3.0},
                "training": {"steps": 0, "throughput_samples_per_second": None, "stability": {"nan_or_inf": False, "loss_explosion": False}},
                "parameter_count": 0,
            },
        }
        metrics["comparison"] = t0_runner._comparison_for_task(metrics)
        metrics["batch_metadata"] = batch.metadata
        return metrics, predictions, batch

    monkeypatch.setattr(t0_runner, "_run_one_task", fake_run_one_task)

    try:
        metrics = run_t0_synthetic_pretraining(config_path)
        registry = json.loads(Path(metrics["registry_path"]).read_text(encoding="utf-8"))

        assert metrics["success_gate"]["slwm_beats_vanilla_on_any_task"] is False
        assert metrics["failure_report_path"] is not None
        assert Path(metrics["failure_report_path"]).exists()
        assert registry["status"] == "failed"
        assert "Stop" in registry["interpretation"]["next_allowed_step"]
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)
