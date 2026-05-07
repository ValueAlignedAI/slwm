from models import LatentSignalField
from models.adapters import AudioSignalAdapter, TextSignalAdapter, VisualSignalAdapter
from models.heads import NoOpHead, TextDecoderHead, UncertaintyHead
from models.policy import PolicyCommitGate
from models.processor import SignalWorldProcessor


def test_dummy_batch_passes_adapter_to_policy_default_noop() -> None:
    batch_size, latent_length, latent_dim = 2, 8, 16
    adapters = [
        TextSignalAdapter(latent_length=latent_length, latent_dim=latent_dim),
        AudioSignalAdapter(latent_length=latent_length, latent_dim=latent_dim),
        VisualSignalAdapter(latent_length=latent_length, latent_dim=latent_dim),
    ]
    packets = [adapter({"batch_size": batch_size}) for adapter in adapters]

    field = LatentSignalField(latent_length=latent_length, latent_dim=latent_dim).from_adapter_outputs(packets)
    processed = SignalWorldProcessor()(field["z"], mask=field["mask"])
    z_world = processed["z_world"]

    text_proposal = TextDecoderHead(vocab_size=128)(z_world)
    noop_proposal = NoOpHead()(z_world)
    uncertainty = UncertaintyHead()(z_world)
    policy = PolicyCommitGate()(z_world, [text_proposal, noop_proposal], uncertainty=uncertainty)

    assert z_world.shape == (batch_size, latent_length, latent_dim)
    assert text_proposal["text_logits"].shape == (batch_size, latent_length, 128)
    assert uncertainty["uncertainty"].shape == (batch_size, latent_length, 1)
    assert policy["noop_probability"] == 1.0
    assert len(policy["commitments"]) == 1
    assert policy["commitments"][0]["head"] == "noop"
    assert policy["commitments"][0]["channel"] == "none"
    assert policy["commitments"][0]["modality_id"] == 0
    assert policy["commitments"][0]["status"] == "committed"
    assert policy["commitments"][0]["reason"] == "no_external_commit_default_noop"
    assert [item["status"] for item in policy["suppressed"]] == ["suppressed"]


def test_policy_can_exercise_single_head_commit_path_when_requested() -> None:
    batch_size, latent_length, latent_dim = 1, 4, 6
    packet = TextSignalAdapter(latent_length=latent_length, latent_dim=latent_dim)({"batch_size": batch_size})
    field = LatentSignalField(latent_length=latent_length, latent_dim=latent_dim).from_adapter_outputs([packet])
    z_world = SignalWorldProcessor()(field["z"], mask=field["mask"])["z_world"]
    text_proposal = TextDecoderHead(vocab_size=32)(z_world)

    policy = PolicyCommitGate()(z_world, [text_proposal], goal={"commit_head": "text_decoder"})

    assert policy["noop_probability"] == 0.0
    assert policy["gates"]["text_decoder"] == 1.0
    assert policy["commitments"][0]["head"] == "text_decoder"
    assert policy["commitments"][0]["status"] == "committed"


def test_policy_accepts_mapping_and_raw_proposals_and_rejects_bad_items() -> None:
    batch_size, latent_length, latent_dim = 1, 4, 6
    packet = TextSignalAdapter(latent_length=latent_length, latent_dim=latent_dim)({"batch_size": batch_size})
    field = LatentSignalField(latent_length=latent_length, latent_dim=latent_dim).from_adapter_outputs([packet])
    z_world = SignalWorldProcessor()(field["z"], mask=field["mask"])["z_world"]
    text_proposal = TextDecoderHead(vocab_size=32)(z_world)

    mapping_policy = PolicyCommitGate()(z_world, {"text": text_proposal})
    assert mapping_policy["metadata"]["proposal_count"] == 1
    assert mapping_policy["commitments"][0]["head"] == "noop"

    raw_policy = PolicyCommitGate()(z_world, [text_proposal["proposal"]])
    assert raw_policy["metadata"]["proposal_count"] == 1
    assert raw_policy["commitments"][0]["head"] == "noop"

    try:
        PolicyCommitGate()(z_world, [object()])
    except TypeError as exc:
        assert "Unsupported proposal item" in str(exc)
    else:  # pragma: no cover - defensive guard
        raise AssertionError("PolicyCommitGate should reject non-mapping proposals")
