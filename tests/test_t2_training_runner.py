import json
import shutil
from pathlib import Path

import pytest

from training.t2_prepare_latents import prepare_t2_audio_visual_latents


def _prepare_config(output_dir: str) -> dict:
    return {
        "runtime": {"seed": 1234, "deterministic": True},
        "data": {
            "modalities": ["audio", "visual_video"],
            "output_dir": output_dir,
            "sample_schema_version": "t2.runner_test_v0",
            "audio_length": 8,
            "visual_length": 8,
            "max_audio_length": 8,
            "max_visual_length": 8,
            "dataset_mix": {"text_code": None, "audio": 0.5, "visual_video": 0.5},
            "split_policy": {"train": 0.7, "validation": 0.2, "test": 0.1, "seed": 1234},
            "sources": [
                {
                    "type": "generated_fixture",
                    "name": "runner_test_fixture",
                    "dataset": "project_generated_t2_runner_fixture",
                    "license": "project-authored test fixture",
                    "sample_count": 24,
                    "audio_length": 8,
                    "visual_length": 8,
                    "audio_feature_dim": 4,
                    "visual_feature_dim": 5,
                    "label_count": 4,
                    "seed": 1234,
                }
            ],
        },
    }


def _train_config(output_dir: str, experiment_id: str) -> dict:
    return {
        "runtime": {"seed": 1234, "deterministic": True, "device": "cpu", "precision": "fp32"},
        "model": {
            "name": "tiny-t2-runner-test",
            "variant": "slwm_audio_visual_latent",
            "parameter_accounting_mode": "strict",
            "context_length": 16,
            "latent_length": 16,
            "latent_dim": 16,
            "n_layer": 1,
            "n_head": 2,
            "audio_feature_dim": 4,
            "visual_feature_dim": 5,
            "dropout": 0.0,
            "architecture_flags": {
                "use_local_temporal_mixer": True,
                "use_spectral_mixer": True,
                "use_long_conv": True,
                "use_attention_binding": True,
                "use_gated_mlp": True,
                "local_kernel_size": 3,
                "spectral_kernel_size": 7,
                "long_kernel_size": 7,
                "mlp_ratio": 2,
            },
        },
        "data": {
            "modalities": ["audio", "visual_video"],
            "prepared_corpus_dir": output_dir,
            "sample_schema_version": "t2.runner_test_v0",
            "dataset_mix": {"text_code": None, "audio": 0.5, "visual_video": 0.5},
        },
        "training": {
            "objective": [
                "audio_latent_continuation_mse",
                "visual_latent_continuation_mse",
                "missing_span_reconstruction_mse",
                "audio_video_contrastive_alignment",
            ],
            "optimizer": "adamw_torch",
            "learning_rate_schedule": "constant",
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "steps": 2,
            "validation_batches": 1,
            "context_fraction": 0.5,
            "missing_span_fraction": 0.125,
            "grad_clip_norm": 1.0,
            "loss_weights": {"audio_prediction": 1.0, "visual_prediction": 1.0, "alignment": 0.1},
        },
        "registry": {
            "experiment_id": experiment_id,
            "artifact_dir": f"experiments/multimodal/t2/{experiment_id}",
            "path": f"experiments/multimodal/t2/{experiment_id}/registry.json",
            "metrics_path": f"experiments/multimodal/t2/{experiment_id}/metrics.json",
            "report_path": f"experiments/multimodal/t2/{experiment_id}/report.md",
            "checkpoint_path": f"experiments/multimodal/t2/{experiment_id}/checkpoint.pt",
            "config_copy_path": f"experiments/multimodal/t2/{experiment_id}/config.json",
        },
    }


def test_t2_runner_writes_registered_smoke_artifacts(tmp_path) -> None:
    pytest.importorskip("torch")
    from training.t2_train_latents import run_t2_audio_visual_training

    output_dir = "artifacts/t2_audio_visual/test_runner_901"
    experiment_id = "EXP-T2-991"
    artifact_dir = Path(f"experiments/multimodal/t2/{experiment_id}")
    prepare_path = tmp_path / "prepare_t2.json"
    prepare_path.write_text(json.dumps(_prepare_config(output_dir)), encoding="utf-8")
    train_path = tmp_path / "train_t2.json"
    train_path.write_text(json.dumps(_train_config(output_dir, experiment_id)), encoding="utf-8")

    try:
        prepare_t2_audio_visual_latents(prepare_path)
        metrics = run_t2_audio_visual_training(train_path, max_steps=2, no_checkpoint=True)
        registry = json.loads(Path(metrics["registry_path"]).read_text(encoding="utf-8"))

        assert metrics["sprint"] == "T2"
        assert metrics["validation"]["audio_mse"] >= 0.0
        assert metrics["validation"]["visual_mse"] >= 0.0
        assert metrics["success_gate"]["shuffled_modality_baseline_included"] is True
        assert metrics["success_gate"]["cross_modal_alignment_metric_reported"] is True
        assert Path(metrics["metrics_path"]).exists()
        assert Path(metrics["report_path"]).exists()
        assert registry["sprint"]["id"] == "T2"
        assert registry["data"]["preprocessing"]["text_codec"] is None
        assert registry["evaluation"]["controls"]["shuffled_pairs"] is True
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)
        shutil.rmtree(artifact_dir, ignore_errors=True)


def test_t2_runner_describe_only_reports_large_config_estimate() -> None:
    pytest.importorskip("torch")
    from training.t2_train_latents import run_t2_audio_visual_training

    metrics = run_t2_audio_visual_training("configs/t2/slwm_700m_audio_visual_24gb_fitcheck.json", describe_only=True)
    assert metrics["describe_only"] is True
    assert metrics["estimated_parameter_count"]["total"] > 700_000_000


def test_t2_runner_rejects_artifact_path_outside_allowlist(tmp_path) -> None:
    pytest.importorskip("torch")
    from training.t2_train_latents import run_t2_audio_visual_training

    config = _train_config("artifacts/t2_audio_visual/missing", "EXP-T2-992")
    config["registry"]["path"] = "experiments/text/t1/not_allowed.json"
    config_path = tmp_path / "bad_t2_path.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="experiments/multimodal/t2"):
        run_t2_audio_visual_training(config_path, describe_only=True)


def test_t2_runner_describe_only_rejects_prepared_dir_outside_allowlist(tmp_path) -> None:
    pytest.importorskip("torch")
    from training.t2_train_latents import run_t2_audio_visual_training

    config = _train_config("artifacts/t1_text_code/not_allowed", "EXP-T2-993")
    config_path = tmp_path / "bad_t2_prepared_dir.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="artifacts/t2_audio_visual"):
        run_t2_audio_visual_training(config_path, describe_only=True)
