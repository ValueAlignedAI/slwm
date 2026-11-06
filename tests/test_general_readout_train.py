import json
import numpy as np
import torch
from training.general_readout_train import fit_one,Readout


def test_frozen_readout_learns_and_exports_a_known_transfer_rule(tmp_path):
    rng=np.random.default_rng(3);rule=rng.normal(size=(4,7)).astype(np.float32)
    for role,n in [("train",128),("selection",48)]:
        folder=tmp_path/"features/droid/fixture"/role;folder.mkdir(parents=True)
        x=rng.normal(size=(n,4)).astype(np.float32)
        np.save(folder/"shared.npy",x);np.save(folder/"target_target.npy",x@rule)
        np.save(folder/"target_common.npy",np.zeros((n,17),np.float32))
        (folder/"cases.json").write_text(json.dumps({"groups":[str(i%3) for i in range(n)]}))
    recipe={"projection":8,"hidden":32,"batch":32,"updates":300,"validation_every":50,"patience":3,"learning_rate":.01,"weight_decay":.01,"target_scale_floor":.01}
    condition={"id":"fixture","run":"fixture","feature":"shared","baseline":False}
    result=fit_one({},recipe,tmp_path,"droid",condition,"actor",201,{})
    assert result["selection"]<.15
    path=tmp_path/"heads/droid/actor/fixture/201/selected.pt"
    saved=torch.load(path,weights_only=False)
    model=Readout(saved["inputs"],saved["outputs"],saved["hidden"]);model.load_state_dict(saved["model"])
    assert saved["task"]=="actor" and result["core_weights_unchanged"]
