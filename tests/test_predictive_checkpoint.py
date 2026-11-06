import torch
from training.predictive_train import atomic_save,atomic_link


def test_checkpoint_stream_roundtrip_and_immutable_weight_aliases(tmp_path):
    value={'model':{'weights':torch.arange(16.).reshape(4,4)},'position':12000}
    source=tmp_path/'position_12000.pt';endpoint=tmp_path/'endpoint.pt'
    atomic_save(source,value);atomic_link(source,endpoint)
    assert source.stat().st_ino==endpoint.stat().st_ino
    loaded=torch.load(endpoint,weights_only=False)
    assert torch.equal(loaded['model']['weights'],value['model']['weights'])
    source.unlink()
    assert torch.equal(torch.load(endpoint,weights_only=False)['model']['weights'],value['model']['weights'])
