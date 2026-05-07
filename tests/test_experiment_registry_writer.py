import json
from pathlib import Path

import pytest

from utils.config import load_config
from utils.experiment_registry import make_i0_registry_entry, validate_registry_entry, write_registry_entry


def test_i0_registry_entry_matches_minimal_schema() -> None:
    config = load_config(Path("configs/default_i0.json"))
    entry = make_i0_registry_entry(config=config, git_commit="abc123", working_tree_state="dirty")
    validate_registry_entry(entry)

    assert entry["experiment_id"] == "EXP-I0-001"
    assert entry["sprint"] == {"id": "I0", "name": "Repo skeleton and contracts", "owner_role": "implementation"}
    assert entry["model"]["total_trainable_parameters"] == 0
    assert entry["model"]["module_parameter_counts"] == {
        "adapters": 0,
        "processor": 0,
        "heads": 0,
        "policy": 0,
        "decoders": 0,
    }
    assert entry["interpretation"]["hypothesis_decision"] == "untested"


def test_registry_writer_emits_valid_json_yaml_subset(tmp_path) -> None:
    config = load_config(Path("configs/default_i0.json"))
    entry = make_i0_registry_entry(config=config)
    output_path = write_registry_entry(entry, tmp_path / "EXP-I0-001.json")
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == entry
    assert loaded["config"]["config_hash"].startswith("sha256:")
    assert loaded["evaluation"]["eval_script"] == "tests/test_dummy_end_to_end.py"


def test_registry_validator_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="missing required"):
        validate_registry_entry({})

    config = load_config(Path("configs/default_i0.json"))
    entry = make_i0_registry_entry(config=config)
    bad_id = dict(entry)
    bad_id["experiment_id"] = "I0-001"
    with pytest.raises(ValueError, match="start with EXP"):
        validate_registry_entry(bad_id)

    bad_sprint = dict(entry)
    bad_sprint["sprint"] = {}
    with pytest.raises(ValueError, match="sprint"):
        validate_registry_entry(bad_sprint)

    bad_config = dict(entry)
    bad_config["config"] = {}
    with pytest.raises(ValueError, match="config"):
        validate_registry_entry(bad_config)
