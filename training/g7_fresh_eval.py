"""Prospective evaluation on unseen recordings, written documents, real event rooms and simulator resets."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from models.g4_shared import Spec
from models.g5_shared import decoded
from training.g4_train import sha_file,write
from training.g5_eval import evaluate
from training.g5_batches import load_pack
from scripts.train.g7_scale_lock import datasets,verify


def signal_scores(run,root,out,role):
    card=json.loads((run/"config.json").read_bytes())
    spec=Spec(**card["spec"])
    with np.load(run/"inference.npz") as f:
        p={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
    @jax.jit
    def infer(parameters,b):
        pred=decoded(parameters,spec,b,b["source"])
        x=b["target_x"];lm=b["family"]==5
        target=jnp.where(lm[:,None],jnp.pad(x[:,1:],((0,0),(0,1))),x)
        prev=jnp.where(lm[:,None],x,jnp.pad(x[:,:-1],((0,0),(1,0)),constant_values=1))
        probs=jax.nn.log_softmax(pred[:,:,373:632]+parameters["bigram"][prev])
        nll=-jnp.take_along_axis(probs,target[:,:,None],-1)[...,0]
        mask=(target>0)*b["ok"]
        loss=(nll*mask).sum(-1);tokens=mask.sum(-1)
        aud=((pred[:,:,:100]-(b["target_a"].astype(jnp.float32)/48-7/3))**2).mean(-1)
        img=((pred[:,:,100:356]-(b["target_v"].astype(jnp.float32)/127.5-1))**2).mean(-1)
        return loss,tokens,(aud*b["ok"]).sum(1)/b["ok"].sum(1),(img*b["ok"]).sum(1)/b["ok"].sum(1)
    summary={};raw={}
    for family,f in (("crema",0),("libri",1),("text",5)):
        src,_=load_pack(root/"fresh"/family/role,True)
        ids=np.argsort(src["uid"])[:512]; group=src["group"][ids];raw[family+"_group"]=group
        conditions=((1,"video"),(3,"text"),(0,"audio")) if f==0 else ((0,"audio"),(3,"text")) if f==1 else ((3,"text"),)
        for mode,name in conditions:
            for erase in (False,True):
                key=family+"_"+name+("_erased" if erase else ""); parts=[]
                for j in range(0,len(ids),64):
                    ix=ids[j:j+64];length=len(ix);ix=np.pad(ix,(0,64-length),mode="edge")
                    n=src["length"][ix] if "length" in src else np.full(64,64)
                    ok=np.arange(64)[None]<n[:,None]
                    a=src["a"][ix].astype(np.float32) if "a" in src else np.zeros((64,64,100),np.float32)
                    v=src["v"][ix].astype(np.float32) if "v" in src else np.zeros((64,64,256),np.float32)
                    def motion(x):return np.clip(x+.65*(x-np.concatenate((x[:,:1],x[:,:-1]),1)),0,255)
                    x=src["x"][ix].astype(np.int32);sizes=np.maximum(1,(x>0).sum(1))
                    at=np.arange(64)[None]*(sizes[:,None]-1)//63
                    chosen=np.zeros((64,4),np.uint8);chosen[:,mode]=1
                    b={"ok":ok,"sense":ok&~np.full(ok.shape,erase),"source":chosen,"a":motion(a),"v":motion(v),
                       "n":np.zeros((64,64,17),np.float32),"u":np.zeros((64,64,2),np.float32),
                       "input_x":x if f==5 else np.take_along_axis(x,at,1),"target_x":x,"target_a":a,"target_v":v,"family":np.full(64,f)}
                    parts.append(np.stack([np.asarray(y)[:length] for y in infer(p,b)],axis=1))
                values=np.concatenate(parts);raw[key]=values
                summary[key]={"rows":len(ids),"groups":len(np.unique(group)),"bytes_scored":int(values[:,1].sum()),
                              "byte_nll":float(values[:,0].sum()/values[:,1].sum()) if mode!=3 or f==5 else None,
                              "audio_mse":float(values[:,2].mean()) if f!=5 and mode!=0 else None,
                              "video_mse":float(values[:,3].mean()) if f==0 and mode!=1 else None}
    with (out/(role+"-signals.tmp")).open("wb") as h:np.savez_compressed(h,**raw)
    (out/(role+"-signals.tmp")).replace(out/(role+"-signals.npz"))
    write(out/(role+"-signals.json"),{"checkpoint":sha_file(run/"inference.npz"),"metrics":summary,
          "text_scoring":"Conditional byte NLL with true previous byte for the bigram, also in erased controls."})


def run(a):
    datasets(a.root,a.old);out=a.out/a.run.parent.name/a.run.name;out.mkdir(exist_ok=True,parents=True)
    role=a.role
    if role=="test":
        ck=json.loads(a.lock.read_bytes());verify(ck,a.run,sha_file(a.run/"inference.npz"),a.root/"evaluation")
        for f,pin in ck["paired_new"].items():
            if pin!=sha_file(a.root/"fresh"/f/"test/card.json"):raise ValueError("Speech/text sources changed")
    if not (out/(role+"-metrics.json")).is_file():
        evaluate(a.run,a.root/"evaluation",out,role,"gpu",a.lock)
    if not (out/(role+"-signals.json")).is_file():
        signal_scores(a.run,a.root,out,role)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("root","old","run","out","lock"):parser.add_argument("--"+key,type=Path,required=key!="lock")
    parser.add_argument("--role",choices=("dev","test"),required=True);run(parser.parse_args())
