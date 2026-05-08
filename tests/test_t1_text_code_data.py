import pytest

from data import build_text_code_lm_datasets, build_text_tokenizer
from data.contract import MODALITY_IDS
from data.text_code import validate_t1_text_only_data_config
from utils.config import load_config


def test_t1_byte_fallback_tokenizer_roundtrip_and_metadata() -> None:
    config = load_config("configs/t1/gpt2_text_tiny_pilot.json")
    tokenizer = build_text_tokenizer(config["model"])
    ids = tokenizer.encode("def f():\n    return 1", add_eos=True)

    assert tokenizer.vocab_size == 260
    assert ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(ids).startswith("def f()")
    assert tokenizer.metadata()["effective_type"] == "byte_fallback"
    assert tokenizer.metadata()["intended_tokenizer"] == "gpt2_bpe"


def test_t1_text_code_dataset_uses_only_text_modality_and_stable_splits() -> None:
    config = load_config("configs/t1/gpt2_text_tiny_pilot.json")
    tokenizer = build_text_tokenizer(config["model"])
    bundle = build_text_code_lm_datasets(config, tokenizer)

    assert MODALITY_IDS["text_code"] == 1
    assert bundle.train.windows.shape[1] == config["training"]["sequence_length"] + 1
    assert bundle.validation.windows.shape[1] == config["training"]["sequence_length"] + 1
    assert bundle.split_digests()["train"].startswith("sha256:")
    assert bundle.split_digests()["validation"].startswith("sha256:")
    assert {record.split for record in bundle.records} == {"train", "validation", "test"}
    input_ids, target_ids = bundle.train.batch(batch_size=2, step=0)
    assert input_ids.shape == target_ids.shape == (2, config["training"]["sequence_length"])


def test_t1_data_config_rejects_audio_or_visual_mix() -> None:
    config = load_config("configs/t1/gpt2_text_tiny_pilot.json")
    bad_data = dict(config["data"])
    bad_data["dataset_mix"] = dict(bad_data["dataset_mix"])
    bad_data["dataset_mix"]["audio"] = 0.1

    with pytest.raises(ValueError, match="must not include audio"):
        validate_t1_text_only_data_config(bad_data)
