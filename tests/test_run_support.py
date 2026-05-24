import json
from pathlib import Path

import pytest

from training.run_support import CheckpointSpec, SafePathPolicy, append_jsonl, atomic_write_json, validate_checkpoint_payload


def test_safe_path_policy_rejects_absolute_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    policy = SafePathPolicy(Path("experiments"), "test artifact")

    with pytest.raises(ValueError, match="repository-relative"):
        policy.validate("/tmp/escape.pt")


def test_safe_path_policy_rejects_parent_escape(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    policy = SafePathPolicy(Path("experiments"), "test artifact")

    with pytest.raises(ValueError, match="under experiments"):
        policy.validate("experiments/../escape.pt")


def test_safe_path_policy_rejects_symlink(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "experiments"
    root.mkdir()
    target = root / "target.pt"
    target.write_text("checkpoint", encoding="utf-8")
    link = root / "latest.pt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are not available in this environment")
    policy = SafePathPolicy(Path("experiments"), "test artifact")

    with pytest.raises(ValueError, match="symlink"):
        policy.validate("experiments/latest.pt", must_exist=True)


def test_append_jsonl_and_atomic_json_write(tmp_path) -> None:
    json_path = tmp_path / "state.json"
    progress_path = tmp_path / "progress.jsonl"

    atomic_write_json({"step": 1}, json_path)
    append_jsonl(progress_path, {"event": "train_step", "loss": 1.25}, echo_stdout=False)

    assert json.loads(json_path.read_text(encoding="utf-8"))["step"] == 1
    assert json.loads(progress_path.read_text(encoding="utf-8").strip())["loss"] == 1.25


def test_checkpoint_payload_validation_rejects_schema_mismatch() -> None:
    spec = CheckpointSpec(
        schema_version="expected_v1",
        required_keys=frozenset({"schema_version", "config_hash"}),
        expected_values={"config_hash": "sha256:ok"},
    )

    with pytest.raises(ValueError, match="schema_version"):
        validate_checkpoint_payload({"schema_version": "wrong", "config_hash": "sha256:ok"}, spec)


def test_checkpoint_payload_validation_rejects_expected_mismatch() -> None:
    spec = CheckpointSpec(
        schema_version="expected_v1",
        required_keys=frozenset({"schema_version", "prepared_dataset_manifest_sha256"}),
        expected_values={"prepared_dataset_manifest_sha256": "sha256:manifest"},
    )

    with pytest.raises(ValueError, match="prepared_dataset_manifest_sha256"):
        validate_checkpoint_payload({"schema_version": "expected_v1", "prepared_dataset_manifest_sha256": "sha256:other"}, spec)
