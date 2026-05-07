from data import MODALITY_IDS, SOURCE_TAGS
from models import NumpySLWMCore, SLWMCoreConfig, TensorSpec, make_i2_dummy_batch
from models.heads import AudioDecoderHead, LatentPredictionHead, NoOpHead, TextDecoderHead, VisualDecoderHead
from models.policy import FixedRulePolicyCommitGate, LearnedPolicyCommitGateStub, PolicyCommitGate


B, T, D = 2, 6, 8


def _z_world() -> TensorSpec:
    return TensorSpec((B, T, D), "float32", "z_world")


def _all_head_outputs(metadata=None) -> list[dict]:
    z_world = _z_world()
    return [
        TextDecoderHead(vocab_size=32)(z_world, metadata=metadata),
        AudioDecoderHead(audio_dim=12)(z_world, metadata=metadata),
        VisualDecoderHead(visual_dim=10)(z_world, metadata=metadata),
        NoOpHead()(z_world, metadata=metadata),
    ]


def test_i3_output_heads_emit_scored_modality_proposals() -> None:
    text, audio, visual, noop = _all_head_outputs()

    assert text["text_logits"].shape == (B, T, 32)
    assert audio["audio_latents"].shape == (B, T, 12)
    assert visual["visual_latents"].shape == (B, T, 10)

    expected = [
        (text, "text_decoder", "text", "text_code"),
        (audio, "audio_decoder", "audio", "audio"),
        (visual, "visual_decoder", "visual", "visual_video"),
        (noop, "noop", "none", "noop"),
    ]
    for output, head_name, channel, modality in expected:
        proposal = output["proposal"]
        assert proposal["head"] == head_name
        assert proposal["channel"] == channel
        assert proposal["modality"] == modality
        assert proposal["modality_id"] == MODALITY_IDS[modality]
        assert proposal["latent_shape"] == [B, T, D]
        assert isinstance(proposal["score"], float)
        assert proposal["source_tag"] in SOURCE_TAGS
        assert proposal["status"] == "proposal"
        assert proposal["committed"] is False


def test_i3_fixed_rule_policy_default_noop_suppresses_decode_all() -> None:
    policy = PolicyCommitGate()(_z_world(), _all_head_outputs())

    assert [item["head"] for item in policy["commitments"]] == ["noop"]
    assert policy["commitments"][0]["status"] == "committed"
    assert policy["commitments"][0]["reason"] == "no_external_commit_default_noop"
    assert {item["head"] for item in policy["suppressed"]} == {"text_decoder", "audio_decoder", "visual_decoder"}
    assert all(item["status"] == "suppressed" for item in policy["suppressed"])
    assert policy["diagnostic_only"] == []
    assert policy["gates"]["text_decoder"] == 0.0
    assert policy["gates"]["noop"] == 1.0
    assert policy["noop_probability"] == 1.0


def test_i3_fixed_rule_policy_single_and_multi_head_commitments() -> None:
    z_world = _z_world()
    proposals = _all_head_outputs()

    single = FixedRulePolicyCommitGate()(z_world, proposals, goal={"commit_head": "text_decoder"})
    assert [item["head"] for item in single["commitments"]] == ["text_decoder"]
    assert single["gates"]["text_decoder"] == 1.0
    assert single["noop_probability"] == 0.0
    assert {item["head"] for item in single["suppressed"]} == {"audio_decoder", "visual_decoder", "noop"}

    multi = FixedRulePolicyCommitGate()(z_world, proposals, goal={"commit_heads": ["text_decoder", "audio_decoder"]})
    assert [item["head"] for item in multi["commitments"]] == ["text_decoder", "audio_decoder"]
    assert multi["gates"]["text_decoder"] == 1.0
    assert multi["gates"]["audio_decoder"] == 1.0
    assert multi["gates"]["visual_decoder"] == 0.0
    assert multi["gates"]["noop"] == 0.0
    assert {item["head"] for item in multi["suppressed"]} == {"visual_decoder", "noop"}
    assert multi["metadata"]["external_commitment_count"] == 2


def test_i3_policy_can_select_zero_heads_without_synthetic_noop() -> None:
    policy = FixedRulePolicyCommitGate()(_z_world(), _all_head_outputs(), goal={"commit_heads": [], "commit_noop": False})

    assert policy["commitments"] == []
    assert {item["head"] for item in policy["suppressed"]} == {"text_decoder", "audio_decoder", "visual_decoder", "noop"}
    assert policy["metadata"]["committed_count"] == 0


def test_i3_diagnostic_probe_outputs_are_internal_only() -> None:
    z_world = _z_world()
    text_probe = TextDecoderHead(vocab_size=32)(z_world, metadata={"mode": "explore"})
    latent_probe = LatentPredictionHead()(z_world, metadata={"mode": "explore"})

    assert text_probe["proposal"]["status"] == "diagnostic-only"
    assert text_probe["proposal"]["diagnostic_only"] is True
    assert latent_probe["proposal"]["channel"] == "internal"
    assert latent_probe["proposal"]["status"] == "diagnostic-only"

    policy = PolicyCommitGate()(z_world, [text_probe, latent_probe], goal={"mode": "explore", "commit_noop": False})
    assert policy["commitments"] == []
    assert {item["head"] for item in policy["diagnostic_only"]} == {"text_decoder", "latent_prediction"}
    assert all(item["status"] == "diagnostic-only" for item in policy["diagnostic_only"])


def test_i3_learned_policy_stub_preserves_commit_api_without_training() -> None:
    z_world = _z_world()
    policy = LearnedPolicyCommitGateStub()(z_world, _all_head_outputs(), goal={"commit_head": "visual_decoder"})

    assert [item["head"] for item in policy["commitments"]] == ["visual_decoder"]
    assert policy["metadata"]["learned_policy_stub"] is True
    assert policy["metadata"]["training_enabled"] is False
    assert set(policy["metadata"]["policy_logits"]) == {"text_decoder", "audio_decoder", "visual_decoder", "noop"}


def test_i3_numpy_core_wires_output_heads_and_policy_paths() -> None:
    config = SLWMCoreConfig(context_length=9, latent_dim=8, n_layer=1, text_vocab_size=24, audio_feature_dim=5, visual_feature_dim=6)
    model = NumpySLWMCore(config)
    batch = make_i2_dummy_batch(config, batch_size=2, seed=11)

    output = model.forward(batch, policy_goal={"commit_heads": ["text_decoder", "audio_decoder"]})

    assert output["z_world"].shape == (2, 9, 8)
    assert set(output["output_heads"]) == {"text", "audio", "visual", "noop"}
    assert output["output_heads"]["text"]["text_logits"].shape == (2, 9, 24)
    assert output["output_heads"]["audio"]["audio_latents"].shape == (2, 9, 5)
    assert output["output_heads"]["visual"]["visual_latents"].shape == (2, 9, 6)
    assert [item["head"] for item in output["policy"]["commitments"]] == ["text_decoder", "audio_decoder"]
    assert {item["status"] for item in output["policy"]["decisions"]} <= {"committed", "suppressed", "diagnostic-only"}

    explore = model.forward(batch, output_metadata={"mode": "explore"}, policy_goal={"mode": "explore", "commit_noop": False})
    assert explore["policy"]["commitments"] == []
    assert {item["head"] for item in explore["policy"]["diagnostic_only"]} == {
        "text_decoder",
        "audio_decoder",
        "visual_decoder",
        "noop",
    }
