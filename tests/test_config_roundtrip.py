from pathlib import Path

import pytest

from utils.config import config_hash, load_config, write_config


def test_default_i0_config_loads_and_hashes() -> None:
    config = load_config(Path("configs/default_i0.json"))
    assert config["sprint"]["id"] == "I0"
    assert config["model"]["latent_length"] == 1024
    assert config_hash(config).startswith("sha256:")


def test_config_round_trip_is_lossless(tmp_path) -> None:
    config = load_config(Path("configs/default_i0.json"))
    output_path = write_config(config, tmp_path / "roundtrip.json")
    assert load_config(output_path) == config
    assert config_hash(load_config(output_path)) == config_hash(config)


def test_config_loader_rejects_non_object_json(tmp_path) -> None:
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Top-level config"):
        load_config(bad_path)
