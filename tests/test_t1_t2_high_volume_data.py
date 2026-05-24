import json
from pathlib import Path

import pytest

from training.t1_t2_high_volume_data import plan_t1_t2_high_volume_data


def _config(data_root: Path) -> dict:
    return {
        "active_stage": "quality",
        "storage": {
            "plan_id": "test_plan",
            "data_root": str(data_root),
            "max_total_bytes": 10_000_000,
            "dry_run": True,
            "allow_network": False,
            "allow_non_data_root": True,
            "bootstrap_directories": True,
        },
        "t1": {
            "enabled": True,
            "modalities": ["text_code"],
            "dataset_mix": {"text_english": 0.7, "code": 0.3, "audio": None, "visual_video": None},
            "sources": [
                {
                    "name": "tiny_text_source",
                    "dataset": "project_fixture_text",
                    "stage": "smoke",
                    "modality": "text_code",
                    "category": "text_english",
                    "estimated_bytes": 1024,
                    "download_kind": "local_fixture",
                    "license_notes": "project-authored fixture",
                }
            ],
        },
        "t2": {
            "enabled": True,
            "modalities": ["audio", "visual_video"],
            "dataset_mix": {"text_code": None, "audio": 0.5, "visual_video": 0.5},
            "sources": [
                {
                    "name": "tiny_av_source",
                    "dataset": "project_fixture_av",
                    "stage": "quality",
                    "modality": "audio_visual_pair",
                    "category": "audio_video_correspondence",
                    "estimated_bytes": 2048,
                    "download_kind": "local_fixture",
                    "license_notes": "project-authored fixture",
                    "caption_use": "metadata_only",
                },
                {
                    "name": "gated_future_source",
                    "dataset": "gated_future",
                    "stage": "high_volume",
                    "modality": "audio",
                    "estimated_bytes": 4096,
                    "requires_auth": True,
                    "requires_terms": True,
                    "accepted_terms": False,
                    "required_tools": ["example-tool"],
                },
            ],
        },
    }


def test_high_volume_data_plan_dry_run_writes_manifests(tmp_path) -> None:
    config = _config(tmp_path / "slwm_data")
    config_path = tmp_path / "plan.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    summary = plan_t1_t2_high_volume_data(config_path)

    assert summary["dry_run"] is True
    assert summary["active_stage"] == "quality"
    assert summary["estimated_bytes"] == 3072
    assert summary["selected_sources_by_sprint"] == {"T1": ["tiny_text_source"], "T2": ["tiny_av_source"]}
    assert Path(summary["plan_path"]).exists()
    assert Path(summary["sources_path"]).exists()
    assert Path(summary["directory_layout_path"]).exists()
    assert (tmp_path / "slwm_data" / "t1_text_code" / "tokenized_gpt2").exists()
    assert (tmp_path / "slwm_data" / "t2_signal_latents" / "audio_latents").exists()

    source_rows = [json.loads(line) for line in Path(summary["sources_path"]).read_text(encoding="utf-8").splitlines()]
    assert {row["name"] for row in source_rows} == {"tiny_text_source", "tiny_av_source"}
    assert summary["acceptance_gates"]["not_satisfied_by_this_script"]


def test_high_volume_data_plan_reports_high_volume_blockers_in_dry_run(tmp_path) -> None:
    config = _config(tmp_path / "slwm_data")
    config["active_stage"] = "high_volume"
    config_path = tmp_path / "plan.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    summary = plan_t1_t2_high_volume_data(config_path)

    blockers = {row["blocker"] for row in summary["blockers"] if row["source"] == "gated_future_source"}
    assert any(blocker.startswith("requires_auth") for blocker in blockers)
    assert "requires_terms_acceptance: set accepted_terms=true only after terms/license review" in blockers
    assert "requires_tool:example-tool" in blockers


def test_high_volume_data_plan_rejects_t1_audio_scope(tmp_path) -> None:
    config = _config(tmp_path / "slwm_data")
    config["t1"]["dataset_mix"]["audio"] = 0.1
    config_path = tmp_path / "bad_plan.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="must not include audio"):
        plan_t1_t2_high_volume_data(config_path)


def test_high_volume_data_plan_rejects_t2_text_generation_scope(tmp_path) -> None:
    config = _config(tmp_path / "slwm_data")
    config["t2"]["sources"][0]["caption_use"] = "text_generation_target"
    config_path = tmp_path / "bad_plan.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata_only"):
        plan_t1_t2_high_volume_data(config_path)


def test_high_volume_data_plan_non_dry_run_requires_confirmation(tmp_path) -> None:
    config = _config(tmp_path / "slwm_data")
    config["storage"]["dry_run"] = False
    config["storage"]["allow_network"] = True
    config_path = tmp_path / "bad_plan.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="confirm-download"):
        plan_t1_t2_high_volume_data(config_path)


def test_high_volume_data_plan_rejects_non_data_root_without_test_override(tmp_path) -> None:
    # Keep this root outside /data even when pytest's tmp_path is redirected to
    # /data-backed scratch space by the operator's TMPDIR.
    config = _config(Path("/tmp") / f"slwm_non_data_root_{tmp_path.name}")
    config["storage"]["allow_non_data_root"] = False
    config_path = tmp_path / "bad_root.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="under /data"):
        plan_t1_t2_high_volume_data(config_path)


def test_high_volume_data_plan_rejects_absolute_target_subdir(tmp_path) -> None:
    config = _config(tmp_path / "slwm_data")
    config["t1"]["sources"][0]["target_subdir"] = "/etc/slwm_escape"
    config_path = tmp_path / "bad_target.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="target_subdir must be relative"):
        plan_t1_t2_high_volume_data(config_path)


def test_high_volume_data_plan_rejects_parent_escape_target_subdir(tmp_path) -> None:
    config = _config(tmp_path / "slwm_data")
    config["t2"]["sources"][0]["target_subdir"] = "../escaped"
    config_path = tmp_path / "bad_escape.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes data_root"):
        plan_t1_t2_high_volume_data(config_path)


def test_high_volume_data_plan_rejects_absolute_plan_id(tmp_path) -> None:
    config = _config(tmp_path / "slwm_data")
    config["storage"]["plan_id"] = "/tmp/slwm_escape"
    config_path = tmp_path / "bad_plan_id.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="storage.plan_id"):
        plan_t1_t2_high_volume_data(config_path)


def test_high_volume_data_plan_rejects_parent_escape_plan_id(tmp_path) -> None:
    config = _config(tmp_path / "slwm_data")
    config["storage"]["plan_id"] = "../../escaped"
    config_path = tmp_path / "bad_plan_id_escape.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="storage.plan_id"):
        plan_t1_t2_high_volume_data(config_path)
