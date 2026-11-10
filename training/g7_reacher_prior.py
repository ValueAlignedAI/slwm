"""A train-only rotationally equivariant endpoint prior with no goal information."""
import argparse
import json
from pathlib import Path
import numpy as np


def design(n,bank,xp):
    n=xp.asarray(n); last=n[...,-1,:]; prev=n[...,-2,:]; older=n[...,-3,:]
    def velocity(last,prev):
        s=last[...,[4,6]];c=last[...,[5,7]];ps=prev[...,[4,6]];pc=prev[...,[5,7]]
        return xp.arctan2(s*pc-c*ps,c*pc+s*ps)
    delta=velocity(last,prev);previous=velocity(prev,older)
    state=xp.concatenate([last[...,6:8],last[...,8:10],delta,previous,last[...,10:12]],axis=-1)
    state=xp.broadcast_to(state[...,None,:],(*bank.shape[:-1],state.shape[-1]))
    x=xp.concatenate([state,bank],-1)
    return xp.concatenate([x,xp.sin(x),xp.cos(x),xp.sin(x*3),xp.cos(x*3)],-1),last[...,4:6]


def rotate(v,sc,xp,inverse=False):
    s=sc[...,0,None];c=sc[...,1,None]
    if inverse:s=-s
    return xp.stack([c*v[...,0]-s*v[...,1],s*v[...,0]+c*v[...,1]],axis=-1)


def infer(net,n,bank):
    import jax
    import jax.numpy as jnp
    x,sc=design(n,bank,jnp);x=(x-net["mean"])/net["std"]
    for i in range(3): x=jax.nn.silu(x@net[f"w{i}"]+net[f"b{i}"])
    return rotate(x@net["w3"]+net["b3"],sc,jnp)


def fit(root,out,steps):
    import torch
    from torch import nn
    import hashlib
    from training.g7_reacher import H
    torch.manual_seed(811);torch.set_num_threads(4)
    def load(role):
        d=root/"reacher"/role
        n=np.load(d/"n.npy",mmap_mode="r");bank=np.load(d/"actions.npy");y=np.load(d/"end.npy")/.21
        x,sc=design(n,bank,np); target=rotate(y,sc,np,True)
        return x.astype(np.float32),target.astype(np.float32),sc,np.array(n),bank,y,np.load(d/"goals.npy")/.21
    x,y,*_=load("train"); mean=x.reshape(-1,x.shape[-1]).mean(0);std=np.maximum(.05,x.reshape(-1,x.shape[-1]).std(0))
    xin=torch.from_numpy(((x-mean)/std).reshape(-1,x.shape[-1])).cuda();yin=torch.from_numpy(y.reshape(-1,2)).cuda()
    model=nn.Sequential(nn.Linear(x.shape[-1],192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU(),nn.Linear(192,2)).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0005)
    for step in range(steps):
        ids=torch.randint(len(xin),(768,),device="cuda");estimate=model(xin[ids]);loss=(estimate-yin[ids]).square().mean()
        opt.zero_grad();loss.backward();opt.step()
    vx,_,sc,n,bank,truth,goals=load("dev")
    with torch.inference_mode():
        local=model(torch.from_numpy((vx-mean)/std).cuda()).cpu().numpy()
    pred=rotate(local,sc,np);cost=np.linalg.norm(truth[:,:,None]-goals[:,None,:,None],axis=-1)
    pick=np.linalg.norm(pred[:,:,None]-goals[:,None,:,None],axis=-1).argmin(-1)
    regret=(np.take_along_axis(cost,pick[...,None],axis=-1)[...,0]-cost.min(-1))*21
    out.mkdir(exist_ok=True,parents=True);payload={"mean":mean,"std":std}
    for i,layer in enumerate(model[::2]):
        payload[f"w{i}"]=layer.weight.detach().cpu().numpy().T;payload[f"b{i}"]=layer.bias.detach().cpu().numpy()
    with (out/"prior.part").open("wb") as f:np.savez(f,**payload)
    (out/"prior.part").replace(out/"prior.npz")
    torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"step":steps},out/"optimizer.pt")
    card={"training_only":True,"updates":steps,"states":len(x)*4,"branches":len(x)*4*9,
          "held_endpoint_cm":float(np.linalg.norm(pred-truth,axis=-1).mean()*21),"held_oracle_goal_regret_cm":float(regret.mean()),
          "held_oracle_goal_top1":float((pick==cost.argmin(-1)).mean())}
    (out/"card.json").write_text(json.dumps(card,sort_keys=True,indent=2)+"\n");print(card,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--steps",type=int,default=3600)
    a=p.parse_args();fit(a.root,a.out,a.steps)
