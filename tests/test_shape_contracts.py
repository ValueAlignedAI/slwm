import pytest

from data import MODALITY_IDS, SOURCE_TAGS
from models import LatentSignalField, TensorSpec, ensure_latent, ensure_mask
from models.adapters import AudioSignalAdapter, TextSignalAdapter, VisualSignalAdapter
from models.adapters.base import BaseModalityAdapter
from models.heads import (
    AudioDecoderHead,
    LatentPredictionHead,
    NoOpHead,
    ReconstructionHead,
    TextDecoderHead,
    UncertaintyHead,
    VisualDecoderHead,
)
from models.processor import SignalWorldProcessor
from models.types import dtype_name, make_latent_spec, make_mask_spec, shape_of


B, T, D = 2, 8, 16


def test_tensor_spec_validates_canonical_shapes() -> None:
    z = TensorSpec((B, T, D), "float32", "z")
    mask = TensorSpec((B, T), "bool", "mask")
    assert ensure_latent(z) == (B, T, D)
    assert ensure_mask(mask, (B, T)) == (B, T)
    assert z.ndim == 3
    assert z.as_dict() == {"name": "z", "shape": [B, T, D], "dtype": "float32"}


def test_tensor_spec_rejects_bad_rank_and_dtype() -> None:
    with pytest.raises(ValueError, match="rank 3"):
        ensure_latent(TensorSpec((B, T), "float32", "z_bad"))
    with pytest.raises(ValueError, match="floating"):
        ensure_latent(TensorSpec((B, T, D), "bool", "z_bad_dtype"))
    with pytest.raises(ValueError, match="bool"):
        ensure_mask(TensorSpec((B, T), "float32", "mask_bad_dtype"))
    with pytest.raises(ValueError, match="at least one dimension"):
        TensorSpec((), "float32", "empty")
    with pytest.raises(ValueError, match="positive"):
        TensorSpec((B, 0, D), "float32", "zero_dim")
    with pytest.raises(ValueError, match="does not match expected"):
        ensure_mask(TensorSpec((B, T + 1), "bool", "wrong_mask"), (B, T))
    with pytest.raises(TypeError, match="does not expose"):
        shape_of(object())


def test_tensor_like_shape_and_dtype_helpers_accept_plain_objects() -> None:
    class TensorLike:
        shape = [B, T, D]

    tensor_like = TensorLike()
    assert shape_of(tensor_like) == (B, T, D)
    assert dtype_name(tensor_like) == "unknown"
    assert ensure_latent(tensor_like) == (B, T, D)


def test_adapters_emit_canonical_packets_and_modality_ids() -> None:
    adapters = [
        TextSignalAdapter(latent_length=T, latent_dim=D),
        AudioSignalAdapter(latent_length=T, latent_dim=D),
        VisualSignalAdapter(latent_length=T, latent_dim=D),
    ]
    for adapter in adapters:
        packet = adapter({"batch_size": B})
        assert packet["z"].shape == (B, T, D)
        assert packet["z"].dtype == "float32"
        assert packet["mask"].shape == (B, T)
        assert packet["mask"].dtype == "bool"
        assert packet["metadata"]["modality_id"] == MODALITY_IDS[packet["metadata"]["modality"]]
        assert packet["metadata"]["observed"] is True


def test_adapter_accepts_explicit_specs_and_rejects_bad_inputs() -> None:
    adapter = BaseModalityAdapter(latent_length=T, latent_dim=D, modality="text_code")
    z = make_latent_spec(B, T, D)
    mask = make_mask_spec(B, T)
    packet = adapter({"z": z, "mask": mask, "metadata": {"sample_id": "s"}})
    assert packet["z"] is z
    assert packet["mask"] is mask
    assert packet["metadata"]["sample_id"] == "s"

    with pytest.raises(ValueError, match="Unknown modality"):
        BaseModalityAdapter(modality="sensor")
    with pytest.raises(ValueError, match="expected D"):
        adapter({"z": make_latent_spec(B, T, D + 1)})
    with pytest.raises(ValueError, match="does not match expected"):
        adapter({"z": z, "mask": make_mask_spec(B, T + 1)})


def test_latent_field_packs_adapter_outputs_to_context_shape() -> None:
    packets = [
        TextSignalAdapter(latent_length=T, latent_dim=D)({"batch_size": B}),
        AudioSignalAdapter(latent_length=T, latent_dim=D)({"batch_size": B}),
    ]
    field = LatentSignalField(latent_length=T, latent_dim=D).from_adapter_outputs(packets)
    assert field["z"].shape == (B, T, D)
    assert field["mask"].shape == (B, T)
    assert field["metadata"]["modalities"] == ["text_code", "audio"]


def test_latent_field_rejects_invalid_packets() -> None:
    field = LatentSignalField(latent_length=T, latent_dim=D)
    valid = TextSignalAdapter(latent_length=T, latent_dim=D)({"batch_size": B})

    with pytest.raises(ValueError, match="At least one"):
        field.from_adapter_outputs([])
    with pytest.raises(ValueError, match="must contain"):
        field.from_adapter_outputs([{"z": valid["z"]}])

    bad_dim = dict(valid)
    bad_dim["z"] = make_latent_spec(B, T, D + 1)
    with pytest.raises(ValueError, match="latent dim"):
        field.from_adapter_outputs([bad_dim])

    other_batch = TextSignalAdapter(latent_length=T, latent_dim=D)({"batch_size": B + 1})
    with pytest.raises(ValueError, match="must share"):
        field.from_adapter_outputs([valid, other_batch])


def test_processor_preserves_latent_shape() -> None:
    z = TensorSpec((B, T, D), "float32", "z_context")
    mask = TensorSpec((B, T), "bool", "context_mask")
    output = SignalWorldProcessor()(z, mask=mask)
    assert output["z_world"].shape == (B, T, D)
    assert output["aux"]["mask_shape"] == [B, T]


def test_output_heads_report_expected_shapes() -> None:
    z_world = TensorSpec((B, T, D), "float32", "z_world")

    assert LatentPredictionHead()(z_world)["latent_prediction"].shape == (B, T, D)
    assert ReconstructionHead()(z_world)["reconstruction"].shape == (B, T, D)

    uncertainty = UncertaintyHead()(z_world)
    assert uncertainty["uncertainty"].shape == (B, T, 1)
    assert uncertainty["source_logits"].shape == (B, T, len(SOURCE_TAGS))

    assert TextDecoderHead(vocab_size=99)(z_world)["text_logits"].shape == (B, T, 99)
    assert AudioDecoderHead(audio_dim=80)(z_world)["audio_latents"].shape == (B, T, 80)
    assert VisualDecoderHead(visual_dim=32)(z_world)["visual_latents"].shape == (B, T, 32)
    assert NoOpHead()(z_world)["proposal"]["channel"] == "none"
