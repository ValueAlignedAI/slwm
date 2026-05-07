def test_i0_stub_imports() -> None:
    import models.baselines
    from data import MODALITY_IDS, SOURCE_TAGS, validate_sample_contract
    from models import LatentSignalField, TensorSpec
    from models.adapters import AudioSignalAdapter, TextSignalAdapter, VisualSignalAdapter
    from models.heads import (
        AudioDecoderHead,
        LatentPredictionHead,
        NoOpHead,
        ReconstructionHead,
        TextDecoderHead,
        UncertaintyHead,
        VisualDecoderHead,
    )
    from models.policy import FixedRulePolicyCommitGate, LearnedPolicyCommitGateStub, PolicyCommitGate
    from models.processor import SignalWorldProcessor
    from models.baselines import NumpyGPT2DecoderBaseline, NumpyVanillaMultimodalTransformerBaseline
    from utils import load_config, make_i0_registry_entry, make_i1_baseline_registry_entry, write_config, write_registry_entry

    assert MODALITY_IDS["text_code"] == 1
    assert "observed" in SOURCE_TAGS
    assert validate_sample_contract is not None
    assert TensorSpec((1, 2, 3)).shape == (1, 2, 3)
    assert LatentSignalField is not None
    assert TextSignalAdapter is not None
    assert AudioSignalAdapter is not None
    assert VisualSignalAdapter is not None
    assert SignalWorldProcessor is not None
    assert LatentPredictionHead is not None
    assert ReconstructionHead is not None
    assert UncertaintyHead is not None
    assert TextDecoderHead is not None
    assert AudioDecoderHead is not None
    assert VisualDecoderHead is not None
    assert NoOpHead is not None
    assert PolicyCommitGate is not None
    assert FixedRulePolicyCommitGate is not None
    assert LearnedPolicyCommitGateStub is not None
    assert NumpyGPT2DecoderBaseline is not None
    assert NumpyVanillaMultimodalTransformerBaseline is not None
    assert "NumpyGPT2DecoderBaseline" in models.baselines.__all__
    assert load_config is not None
    assert write_config is not None
    assert make_i0_registry_entry is not None
    assert make_i1_baseline_registry_entry is not None
    assert write_registry_entry is not None
