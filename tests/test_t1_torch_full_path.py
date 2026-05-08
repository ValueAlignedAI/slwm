import json
import shutil
from pathlib import Path

import pytest

from training.t1_prepare_text_code import prepare_t1_text_code_corpus


def _prepare_config(output_dir: str) -> dict:
    return {
        "runtime": {"seed": 7, "deterministic": True},
        "tokenizer": {"type": "byte_fallback", "vocab_size": 260},
        "data": {
            "modalities": ["text_code"],
            "output_dir": output_dir,
            "sample_schema_version": "t1.test_prepare_v0",
            "dataset_mix": {"text_code": 1.0, "audio": None, "visual_video": None},
            "split_policy": {"train": 0.6, "validation": 0.2, "test": 0.2, "seed": 7},
            "sources": [
                {
                    "type": "inline_records",
                    "name": "test_inline_text_code",
                    "category": "text_english",
                    "records": [
                        {
                            "sample_id": "doc-a",
                            "text": "The first document contains enough English text for multiple token windows. " * 4,
                            "dataset": "test_inline",
                        },
                        {
                            "sample_id": "doc-b",
                            "text": "def add(a, b):\n    return a + b\n# deterministic code sample\n" * 6,
                            "dataset": "test_inline",
                        },
                        {
                            "sample_id": "doc-c",
                            "text": "A validation style paragraph records split hashes and reproducible metadata. " * 4,
                            "dataset": "test_inline",
                        },
                    ],
                }
            ],
        },
    }


def test_t1_prepare_text_code_writes_token_files_and_manifest(tmp_path) -> None:
    output_dir = "artifacts/t1_text_code/test_prepare_901"
    config = _prepare_config(output_dir)
    config_path = tmp_path / "prepare.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        card = prepare_t1_text_code_corpus(config_path)
        assert card["status"] == "prepared"
        assert card["modalities"] == ["text_code"]
        assert Path(card["token_files"]["train"]["path"]).exists()
        assert Path(card["token_files"]["validation"]["path"]).exists()
        assert card["manifest_sha256"].startswith("sha256:")
        assert card["split_counts"]["validation"]["tokens"] > 0
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)


def test_t1_prepare_text_code_streaming_mode_writes_token_files(tmp_path) -> None:
    output_dir = "artifacts/t1_text_code/test_prepare_streaming_901"
    config = _prepare_config(output_dir)
    config["data"]["preparation_mode"] = "streaming"
    config["data"]["sources"][0]["records"] = [
        {
            "sample_id": f"stream-doc-{index:03d}",
            "text": f"Streaming preparation document {index} keeps token arrays off Python lists. " * 5,
            "dataset": "test_inline_streaming",
        }
        for index in range(100)
    ]
    config_path = tmp_path / "prepare_streaming.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        card = prepare_t1_text_code_corpus(config_path)
        assert card["status"] == "prepared"
        assert card["preparation_mode"] == "streaming"
        assert Path(card["token_files"]["train"]["path"]).exists()
        assert Path(card["token_files"]["validation"]["path"]).exists()
        assert card["split_counts"]["train"]["tokens"] > 0
        assert card["split_counts"]["validation"]["tokens"] > 0
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)


def test_t1_torch_runner_writes_registered_smoke_artifacts(tmp_path) -> None:
    pytest.importorskip("torch")
    from training.t1_torch_text import run_t1_torch_text_training

    output_dir = "artifacts/t1_text_code/test_prepare_902"
    experiment_id = "EXP-T1-990"
    prepare_config = _prepare_config(output_dir)
    prepare_path = tmp_path / "prepare.json"
    prepare_path.write_text(json.dumps(prepare_config), encoding="utf-8")
    artifact_dir = Path(f"experiments/text/t1/{experiment_id}")

    train_config = {
        "runtime": {"seed": 7, "deterministic": True, "device": "cpu", "precision": "fp32"},
        "tokenizer": {"type": "byte_fallback", "vocab_size": 260},
        "model": {
            "name": "tiny-torch-gpt2-t1-test",
            "variant": "gpt2_baseline",
            "parameter_accounting_mode": "strict",
            "context_length": 16,
            "n_layer": 1,
            "n_embd": 16,
            "n_head": 2,
            "dropout": 0.0,
            "architecture_flags": {"causal_decoder_only": True},
        },
        "data": {
            "modalities": ["text_code"],
            "prepared_corpus_dir": output_dir,
            "sample_schema_version": "t1.test_prepare_v0",
            "dataset_mix": {"text_code": 1.0, "audio": None, "visual_video": None},
        },
        "training": {
            "objective": ["next_token_cross_entropy"],
            "optimizer": "adamw_torch",
            "learning_rate_schedule": "constant",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "sequence_length": 16,
            "steps": 1,
            "validation_batches": 1,
            "grad_clip_norm": 1.0,
            "guardrail_tolerance_percent": 20.0,
        },
        "generation": {"prompts": ["def add"], "max_new_tokens": 2, "temperature": 0.0, "seed": 9, "stop_on_eos": True},
        "registry": {
            "experiment_id": experiment_id,
            "artifact_dir": f"experiments/text/t1/{experiment_id}",
            "path": f"experiments/text/t1/{experiment_id}/registry.json",
            "metrics_path": f"experiments/text/t1/{experiment_id}/metrics.json",
            "samples_path": f"experiments/text/t1/{experiment_id}/samples.json",
            "report_path": f"experiments/text/t1/{experiment_id}/report.md",
            "checkpoint_path": f"experiments/text/t1/{experiment_id}/checkpoint.pt",
            "config_copy_path": f"experiments/text/t1/{experiment_id}/config.json",
        },
    }
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps(train_config), encoding="utf-8")

    try:
        prepare_t1_text_code_corpus(prepare_path)
        metrics = run_t1_torch_text_training(train_path)
        registry = json.loads(Path(metrics["registry_path"]).read_text(encoding="utf-8"))
        assert metrics["tokenizer"]["effective_type"] == "byte_fallback"
        assert metrics["validation_loss"] > 0.0
        assert metrics["throughput_tokens_per_second"] > 0.0
        assert Path(metrics["checkpoint_path"]).exists()
        assert registry["sprint"]["id"] == "T1"
        assert registry["data"]["preprocessing"]["audio_codec_or_features"] is None
        assert registry["data"]["preprocessing"]["visual_codec_or_features"] is None
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)
        shutil.rmtree(artifact_dir, ignore_errors=True)
