"""Load one verified G3 parent and delta and read the joint field without updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.joint_transfer import JointWorld
from training.joint_data import read, tokenizer
from training.joint_sources import CONFIG,digest
from training.scale_text_train import initialize


def restore(delta_path,parent_path,device):
    card = read(delta_path.with_suffix(".json"))
    if digest(delta_path)!=card["delta_sha256"] or digest(parent_path)!=card["parent_sha256"]:
        raise ValueError("Parent/delta hashes do not match the endpoint card")
    delta = torch.load(delta_path,map_location="cpu",weights_only=True)
    saved = torch.load(parent_path,map_location="cpu",mmap=True,weights_only=False)
    profile = read(CONFIG)["profiles"][delta["profile"]]
    core = initialize(saved["config"],profile["core"],delta["seed"])
    core.load_state_dict(saved["model"])
    model = JointWorld(core,profile,delta["seed"])
    model.apply_delta(delta["delta"])
    return model.to(device).eval().requires_grad_(False),card


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--delta",type=Path,required=True)
    p.add_argument("--parent",type=Path,required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt")
    inputs.add_argument("--packets",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--tokens",type=int,default=24)
    a = p.parse_args()
    torch.set_num_threads(3)
    model,card = restore(a.delta,a.parent,a.device)
    with torch.inference_mode():
        if a.prompt is not None:
            enc = tokenizer()
            ids = enc.encode(a.prompt,disallowed_special=())
            if not ids or a.tokens<1:
                p.error("Provide a nonempty prompt and positive continuation length")
            seq = torch.tensor([ids],device=a.device)
            for _ in range(a.tokens):
                state,_,_ = model.field({"text":seq[:,-model.core.context_length:]})
                new = F.linear(state[:,-1],model.core.token_embedding.weight).argmax(-1,keepdim=True)
                seq = torch.cat([seq,new],1)
                if int(new[0,0])==enc.eot_token:
                    break
            a.output.write_text(json.dumps({"profile":card["profile"],"step":card["step"],"prompt":a.prompt,
                "continuation":enc.decode(seq[0,len(ids):].tolist()),"decoding":"greedy"},indent=2)+"\n")
        else:
            with np.load(a.packets,allow_pickle=False) as data:
                inputs = {k:torch.from_numpy(data[k]).to(a.device) for k in data.files if k in ("image","audio","numeric","present")}
            keys = [k for k in inputs if k!="present"]
            if not keys or ("numeric" in keys and "present" not in inputs):
                p.error("Provide image[B,T,192], audio[B,T,100] or numeric[B,T,64] with Boolean present[B,T,64]")
            state,_,_ = model.field(inputs)
            out = {"field":state.cpu().numpy()}
            for k in keys:
                out[k+"_next"] = model.future[k](state[:,-1]).float().cpu().numpy()
            with a.output.open("wb") as f:
                np.savez_compressed(f,**out)


if __name__=="__main__":
    main()
