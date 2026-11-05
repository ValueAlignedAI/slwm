import json
from pathlib import Path
import pytest
torch=pytest.importorskip("torch")
pytest.importorskip("vocos")
pytest.importorskip("cmudict")
from models.slwm_stream import StreamWorld
from training.broad_stream_train import losses,alignment_loss
from training.broad_batch import predict


def test_shared_objectives_backpropagate_without_target_inputs():
    torch.set_num_threads(1); torch.manual_seed(71); b,t=4,128; valid=torch.ones(b,t,dtype=torch.bool); audio_mask=valid.clone(); audio_mask[:,40:48]=False
    batch={"audio":torch.randn(b,t,400),"video":torch.randn(b,t,64),"wrong_video":torch.randn(b,t,64),"phones":torch.randint(0,41,(b,t)),"valid":valid,"a_mask":audio_mask,"v_mask":torch.zeros_like(valid),"t_mask":torch.zeros_like(valid),"holes":~audio_mask,"has_video":torch.tensor([False,False,True,True]),"appearance":torch.zeros(b,64),"texts":[[3,4,5,6]]*b,"ids":[str(i) for i in range(b)],"actors":[str(i) for i in range(b)],"modes":["repair"]*b}
    cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); model=StreamWorld(width=64,layers=6,heads=4,memory=4,chunk=32); out=predict(model,batch); loss,parts=losses(out,batch,cfg); loss=loss+.05*alignment_loss(model,batch); loss.backward()
    assert torch.isfinite(loss)
    assert model.audio_head[-1].weight.grad.norm()>0 and model.future_head.weight.grad.norm()>0
    assert model.memory_read.in_proj_weight.grad is not None
    model.zero_grad(set_to_none=True); out=predict(model,batch); value,_=losses(out,batch,cfg,no_content=True); value.backward(); assert model.character_head.weight.grad is None
