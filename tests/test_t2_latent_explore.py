from __future__ import annotations

import numpy as np

from exploration.t2_latent_explore import _completed_latent, _latent_to_db, _save_visual_media


def test_completed_latent_fills_only_masked_positions() -> None:
    observed = np.zeros((3, 2), dtype=np.float32)
    prediction = np.arange(6, dtype=np.float32).reshape(3, 2)
    mask = np.array([False, True, False])

    completed = _completed_latent(observed, prediction, mask)

    np.testing.assert_allclose(completed[0], [0.0, 0.0])
    np.testing.assert_allclose(completed[1], prediction[1])
    np.testing.assert_allclose(completed[2], [0.0, 0.0])


def test_latent_to_db_is_finite_and_bounded() -> None:
    latent = np.linspace(-3.0, 3.0, 80, dtype=np.float32).reshape(10, 8)

    db = _latent_to_db(latent, reference=latent)

    assert db.shape == latent.shape
    assert np.isfinite(db).all()
    assert float(db.min()) >= -80.0
    assert float(db.max()) <= 0.0


def test_save_visual_media_writes_diagnostic_images(tmp_path) -> None:
    visual_input = np.zeros((1, 512, 256), dtype=np.float32)
    visual_prediction = np.ones((1, 512, 256), dtype=np.float32) * 0.25
    visual_target = np.ones((1, 512, 256), dtype=np.float32) * -0.25
    visual_loss_mask = np.zeros((1, 512), dtype=bool)
    visual_loss_mask[:, 128:256] = True

    artifacts = _save_visual_media(
        mode_dir=tmp_path,
        mode="masked_av",
        sample_index=0,
        visual_input=visual_input,
        visual_prediction=visual_prediction,
        visual_target=visual_target,
        visual_loss_mask=visual_loss_mask,
        fps=1,
        seconds=1.0,
    )

    assert (tmp_path / "completed_visual_frame.png").exists()
    assert (tmp_path / "diagnostic_visual_panels.png").exists()
    assert "completed_visual_frame" in artifacts
