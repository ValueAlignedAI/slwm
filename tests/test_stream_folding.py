import copy
import pytest
torch=pytest.importorskip("torch")
from models.slwm_stream import StreamWorld
from models.slwm_stream_folded import fold_signal_branches


def test_parallel_signal_fold_preserves_fields_memory_and_all_heads():
    torch.set_num_threads(1); torch.manual_seed(31); model=StreamWorld(width=64,layers=3,heads=4,memory=4).eval(); torch.nn.init.normal_(model.audio_head[-1].weight,std=.02); torch.nn.init.normal_(model.video_head.weight,std=.02); b,t=2,128; valid=torch.ones(b,t,dtype=torch.bool); args=(torch.randn(b,t,400),torch.randn(b,t,64),torch.randint(0,41,(b,t)),torch.rand(b,t)>.3,torch.rand(b,t)>.5,torch.rand(b,t)>.7,valid,torch.randn(b,64)); folded=fold_signal_branches(copy.deepcopy(model)); assert folded.folding_metadata["deployed_parameters"]<folded.folding_metadata["original_trained_parameters"]
    with torch.inference_mode():
        a=model(*args); z=folded(*args)
        for key in a: torch.testing.assert_close(a[key],z[key],rtol=3e-5,atol=3e-5)
    assert all(not p.requires_grad for p in folded.parameters())


def test_full_frequency_parameterization_spans_same_fifteen_tap_filter_space():
    basis=torch.eye(16,dtype=torch.float64); kernels=torch.fft.irfft(torch.complex(basis[:,:8],basis[:,8:]),n=15,dim=-1); assert torch.linalg.matrix_rank(kernels).item()==15; assert torch.equal(kernels[8],torch.zeros(15,dtype=torch.float64))


def test_fold_does_not_accept_a_training_model_or_transformer():
    with pytest.raises(ValueError,match="inference"): fold_signal_branches(StreamWorld(width=32,heads=4,layers=1))
    with pytest.raises(ValueError,match="eligible"): fold_signal_branches(StreamWorld(variant="transformer",width=32,heads=4,layers=1).eval())
