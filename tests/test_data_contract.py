import pytest

from data import MODALITY_IDS, REQUIRED_MODALITIES, SOURCE_TAGS, SignalSample, SignalStreamRef, validate_sample_contract


def test_modality_ids_are_stable() -> None:
    assert MODALITY_IDS == {"noop": 0, "text_code": 1, "audio": 2, "visual_video": 3}
    assert REQUIRED_MODALITIES == ("text_code", "audio", "visual_video")


def test_source_tags_are_controlled_set() -> None:
    assert SOURCE_TAGS == (
        "observed",
        "reconstructed",
        "predicted",
        "inferred",
        "imagined",
        "unknown",
        "unsupported",
    )


def test_sample_contract_accepts_required_schema() -> None:
    sample = {
        "sample_id": "sample-1",
        "streams": {
            "text_code": {"data": "hello", "start": 0.0, "end": 1.0},
            "audio": {"path": "audio.wav", "start": 0.0, "end": 1.0},
            "visual_video": {"path": "frame.png", "start": 0.0, "end": 1.0},
        },
        "targets": {"future_text": None},
        "metadata": {"dataset": "dummy", "split": "train"},
    }
    validate_sample_contract(sample)


def test_sample_contract_rejects_unknown_modality() -> None:
    sample = {"sample_id": "bad", "streams": {"sensor": {}}, "targets": {}, "metadata": {}}
    with pytest.raises(ValueError, match="Unknown stream modality"):
        validate_sample_contract(sample)


def test_signal_stream_ref_validates_modality() -> None:
    stream = SignalStreamRef(modality="text_code", data="hello")
    sample = SignalSample(sample_id="s", streams={"text_code": stream})
    assert sample.streams["text_code"].data == "hello"

    with pytest.raises(ValueError, match="Unknown modality"):
        SignalStreamRef(modality="sensor")


def test_sample_contract_rejects_missing_keys_and_bad_streams() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        validate_sample_contract({"sample_id": "missing"})
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_sample_contract({"sample_id": "bad", "streams": [], "targets": {}, "metadata": {}})
