import pytest

torch = pytest.importorskip("torch")

from training import t2_train_latents as t2


def test_t2_alignment_rejects_single_item_microbatch() -> None:
    with pytest.raises(ValueError, match="batch_size >= 2"):
        t2._validate_t2_batching(  # noqa: SLF001 - intentional guard test
            {
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "loss_weights": {"alignment": 0.1},
            }
        )


def test_t2_eval_batch_size_clamps_to_dataset_size() -> None:
    assert t2._configured_eval_batch_size({"batch_size": 128, "eval_batch_size": 32}, 5) == 5  # noqa: SLF001


def test_t2_single_item_retrieval_metrics_are_not_reported() -> None:
    prediction = torch.zeros((1, 2, 1), dtype=torch.float32)
    target = torch.ones((1, 2, 1), dtype=torch.float32)
    mask = torch.ones((1, 2), dtype=torch.bool)
    details = {
        "output": {
            "audio_prediction": prediction,
            "visual_prediction": prediction,
        },
        "audio_mse": torch.tensor(1.0),
        "visual_mse": torch.tensor(1.0),
        "alignment_loss": torch.tensor(0.0),
        "alignment_logits": torch.zeros((1, 1), dtype=torch.float32),
    }
    batch = {
        "audio_targets": target,
        "visual_targets": target,
        "audio_loss_mask": mask,
        "visual_loss_mask": mask,
        "audio_features": prediction,
        "visual_features": prediction,
    }

    metrics = t2._metric_bundle(loss_value=1.0, details=details, batch=batch)  # noqa: SLF001

    assert metrics["audio_video_correspondence_accuracy"] is None
    assert metrics["retrieval_r1"] is None
    assert metrics["shuffled_retrieval_r1"] is None


def test_t2_resume_checkpoint_rejects_absolute_path() -> None:
    with pytest.raises(ValueError, match="repository-relative"):
        t2._safe_checkpoint_read_path("/tmp/not_allowed.pt")  # noqa: SLF001


def test_t2_resume_checkpoint_rejects_parent_escape() -> None:
    with pytest.raises(ValueError, match="under experiments/multimodal/t2"):
        t2._safe_checkpoint_read_path("experiments/multimodal/t2/../../escape.pt")  # noqa: SLF001
