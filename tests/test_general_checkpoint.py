"""A slice must remain resumable after the coordinator releases model RAM."""
import copy
import json
import torch
from training.general_train import persist


def test_background_checkpoint_survives_released_training_item(tmp_path):
    data=tmp_path/"data";data.mkdir();(data/"dataset_contract.json").write_text("{}")
    model=torch.nn.Linear(3,2);optimizer=torch.optim.AdamW(model.parameters())
    model(torch.ones(2,3)).sum().backward();optimizer.step()
    expected=copy.deepcopy(model.state_dict())
    item={"folder":tmp_path,"state":{"position":1,"finished":True},"model":model,"teacher":copy.deepcopy(model),"optimizer":optimizer.state_dict(),"run":{"id":"fixture"}}
    save=persist(item,{"data_root":str(data),"updates":1},{},None,True,True)
    item.update(model=None,teacher=None,optimizer=None);save()
    resume=torch.load(tmp_path/"resume.pt",weights_only=False)
    restored=torch.nn.Linear(3,2);restored.load_state_dict(resume["model"])
    restored_optimizer=torch.optim.AdamW(restored.parameters());restored_optimizer.load_state_dict(resume["optimizer"])
    for current,opt in [(model,optimizer),(restored,restored_optimizer)]:
        opt.zero_grad();current(torch.ones(2,3)).square().sum().backward();opt.step()
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters()))
    endpoint=torch.load(tmp_path/"endpoint.pt",weights_only=False)
    assert all(torch.equal(endpoint["model"][k],v) for k,v in expected.items())
    assert json.loads((tmp_path/"complete.json").read_text())["target_reached"]
