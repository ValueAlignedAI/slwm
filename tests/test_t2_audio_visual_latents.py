import json
import shutil
from pathlib import Path

import pytest

from data.audio_visual_latents import T2PreparedLatentDataset, t2_batch_contract_summary, validate_t2_audio_visual_data_config
from training.t2_prepare_latents import prepare_t2_audio_visual_latents


def _prepare_config(output_dir: str) -> dict:
    return {
        "runtime": {"seed": 99, "deterministic": True},
        "data": {
            "modalities": ["audio", "visual_video"],
            "output_dir": output_dir,
            "sample_schema_version": "t2.test_v0",
            "audio_codec_or_features": "generated_logmel_like_latents_smoke_only",
            "visual_codec_or_features": "generated_patch_like_latents_smoke_only",
            "audio_length": 6,
            "visual_length": 6,
            "max_audio_length": 6,
            "max_visual_length": 6,
            "dataset_mix": {"text_code": None, "audio": 0.5, "visual_video": 0.5},
            "split_policy": {"train": 0.6, "validation": 0.2, "test": 0.2, "seed": 99},
            "sources": [
                {
                    "type": "generated_fixture",
                    "name": "test_t2_generated_fixture",
                    "dataset": "project_generated_t2_test_fixture",
                    "license": "project-authored test fixture",
                    "sample_count": 15,
                    "audio_length": 6,
                    "visual_length": 6,
                    "audio_feature_dim": 3,
                    "visual_feature_dim": 4,
                    "label_count": 3,
                    "seed": 99,
                }
            ],
        },
    }


def test_t2_data_config_rejects_text_code_scope() -> None:
    with pytest.raises(ValueError, match="audio and visual_video"):
        validate_t2_audio_visual_data_config({"modalities": ["audio", "visual_video", "text_code"]})


def test_t2_prepare_latents_writes_dataset_card_and_manifests(tmp_path) -> None:
    output_dir = "artifacts/t2_audio_visual/test_prepare_901"
    config = _prepare_config(output_dir)
    config_path = tmp_path / "prepare_t2.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        card = prepare_t2_audio_visual_latents(config_path)
        assert card["status"] == "prepared"
        assert card["modalities"] == ["audio", "visual_video"]
        assert card["manifest_sha256"].startswith("sha256:")
        assert card["split_counts"]["train"]["samples"] > 0
        assert Path(output_dir, "dataset_card.json").exists()
        assert Path(output_dir, "manifests", "train.jsonl").exists()
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)


def test_t2_prepared_dataset_batch_matches_contract(tmp_path) -> None:
    output_dir = "artifacts/t2_audio_visual/test_prepare_902"
    config = _prepare_config(output_dir)
    config_path = tmp_path / "prepare_t2.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        prepare_t2_audio_visual_latents(config_path)
        dataset = T2PreparedLatentDataset.load(output_dir, split="train")
        batch = dataset.batch(batch_size=2, seed=123, step=0, context_fraction=0.5, missing_span_fraction=0.1, sequential=True)
        summary = t2_batch_contract_summary(batch)

        assert summary["audio_shape"] == [2, 6, 3]
        assert summary["visual_shape"] == [2, 6, 4]
        assert summary["modality_ids"] == {"audio": 2, "visual_video": 3}
        assert summary["source_tags_valid"] is True
        assert batch["audio_loss_mask"].any()
        assert batch["visual_loss_mask"].any()
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)


def test_t2_prepared_dataset_rejects_tampered_latent_hash(tmp_path) -> None:
    output_dir = "artifacts/t2_audio_visual/test_prepare_903"
    config = _prepare_config(output_dir)
    config_path = tmp_path / "prepare_t2.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    try:
        prepare_t2_audio_visual_latents(config_path)
        manifest_path = Path(output_dir, "manifests", "train.jsonl")
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
        rows[0]["sha256"] = "sha256:" + "0" * 64
        manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

        with pytest.raises(ValueError, match="manifest hash mismatch"):
            T2PreparedLatentDataset.load(output_dir, split="train")
    finally:
        shutil.rmtree(Path(output_dir), ignore_errors=True)


def test_t2_prepare_rejects_output_path_outside_allowlist(tmp_path) -> None:
    config = _prepare_config("artifacts/t1_text_code/not_allowed")
    config_path = tmp_path / "bad_prepare_t2.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="artifacts/t2_audio_visual"):
        prepare_t2_audio_visual_latents(config_path)
