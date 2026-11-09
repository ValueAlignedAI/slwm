"""Continuous held-out AMI replay through the same trained field, with state-reset and lag controls."""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from models.g4_shared import encode
from training.g4_data import tokens
from training.g4_eval import check_lock, weights
from training.g4_train import backend,read,sha_file,write


PROMPTS=(
    "The room is quiet and a person sits beside a table.",
    "Someone begins speaking while looking toward the window.",
    "A small blue cup sits near the edge of the desk.",
    "A second person asks where the cup should go.",
    "The first speaker pauses and then answers slowly.",
    "A hand reaches across the table and lifts the cup.",
    "The cup moves to the left of a closed notebook.",
    "Someone explains what happened in a calm voice.",
    "The hand returns to the center and waits.",
    "Another person opens the notebook to the first page.",
    "A pencil touches the paper and makes a short line.",
    "The speaker describes the line and the empty space.",
    "The cup stays still as the conversation continues.",
    "A person points to the pencil and nods twice.",
    "The pencil rolls a little before coming to rest.",
    "Two people look down and consider the drawing.",
    "The first speaker reads a short sentence aloud.",
    "A hand turns the page and leaves the notebook open.",
    "Someone moves the cup back toward the middle.",
    "The other person listens without moving.",
    "A final sentence is spoken softly.",
    "The hand closes the notebook on the table.",
    "A person looks up and waits for an answer.",
    "The room grows quiet again.",
    "The cup and the pencil remain beside each other.")


def decode_bytes(base,bigram):
    def advance(prev,position):
        new=jnp.argmax(position+bigram[prev],axis=-1).astype(jnp.int32)
        return new,new
    _,seq=jax.lax.scan(advance,jnp.ones(base.shape[0],jnp.int32),base.swapaxes(0,1))
    return seq.T


def make_step(spec):
    @jax.jit
    def step(params,stats,batch,sources,erase,memory,offset):
        z,carried=encode(params,spec,batch,sources,stats,erase=erase,memory=memory,offset=offset)
        pred=(z@params["readout"].astype(z.dtype)).astype(jnp.float32)
        return carried,pred[:,:,:356],decode_bytes(pred[:,:,448:707],params["bigram"])
    return step


def text(seq):
    output=[]
    for i in seq:
        if i==2: break
        if i>=3: output.append(int(i)-3)
    return bytes(output).decode("utf-8",errors="replace")


def run_live(run,root,lock,device):
    backend(device)
    cfg,complete,spec,p,stats=weights(run)
    check_lock(lock,run,complete)
    if read(lock)["live_sha256"]!=sha_file(Path(__file__)): raise ValueError("Live protocol changed")
    with np.load(root/"test/live.npz",allow_pickle=False) as stream:
        a,v,dt,group=stream["a"],stream["v"],float(stream["dt"]),str(stream["group"])
    length=(len(a)//64)*64
    lag=8; delays=np.maximum(np.arange(length)-lag,0)
    modes=jnp.array([0,1,0,1,0,1,3],jnp.int32)
    erased=jnp.array([False,False,False,False,True,True,False])
    memory=None; result=[]; reset=[]; transcripts=[]; costs=[]; fn=make_step(spec)
    for start in range(0,length,64):
        sl=slice(start,start+64); ids=tokens(PROMPTS[(start//64)%len(PROMPTS)])
        batch={"a":np.zeros((7,64,100),np.uint8),"v":np.zeros((7,64,256),np.uint8),
               "n":np.zeros((7,64,17),np.float16),"x":np.zeros((7,64),np.uint16),
               "ok":np.ones((7,64),np.uint8),"family":np.full(7,6,np.uint8)}
        batch["a"][[0,4]]=a[sl]; batch["a"][2]=a[delays[sl]]
        batch["v"][[1,5]]=v[sl]; batch["v"][3]=v[delays[sl]]
        batch["x"][6,:len(ids)]=ids
        begin=time.monotonic()
        memory,y,chars=fn(p,stats,batch,modes,erased,memory,jnp.int32(start))
        y=np.asarray(y); chars=np.asarray(chars); costs.append(time.monotonic()-begin)
        _,baseline,_=fn(p,stats,batch,modes,erased,None,jnp.int32(start))
        result.append(y); reset.append(np.asarray(baseline))
        transcripts.append({"at_seconds":start*dt,"audio_to_text":text(chars[0]),"video_to_text":text(chars[1]),
                            "text_input":PROMPTS[(start//64)%len(PROMPTS)]})
    result=np.concatenate(result,axis=1); reset=np.concatenate(reset,axis=1)
    out=run/"live"; out.mkdir(exist_ok=True)
    measured_a=(a[:length].astype(np.float32)/16-7)/3
    measured_v=v[:length].astype(np.float32)/127.5-1
    metrics={}
    for name,idx,late,blank,cols,truth in (("audio_to_video",0,2,4,slice(100,356),measured_v),
                                           ("video_to_audio",1,3,5,slice(0,100),measured_a)):
        def score(arr): return float(np.mean((arr-truth)**2))
        pred=result[idx,:,cols]
        metrics[name]={"mse":score(pred),"past_341ms_mse":score(result[late,:,cols]),"erased_mse":score(result[blank,:,cols]),
                       "reset_every_64_mse":score(reset[idx,:,cols]),
                       "output_to_target_motion_rms":float(np.sqrt(np.mean(np.diff(pred,axis=0)**2)/max(np.mean(np.diff(truth,axis=0)**2),1e-12)))}
    def pixel(y): return np.rint((np.clip(y,-1,1)+1)*127.5).astype(np.uint8).reshape(length,16,16)
    np.savez_compressed(out/"generated.npz",audio_to_video=pixel(result[0,:,100:356]),
        video_to_logmel=np.clip(result[1,:,:100]*3-5,-12,4).astype(np.float16),
        text_to_video=pixel(result[6,:,100:356]),text_to_logmel=np.clip(result[6,:,:100]*3-5,-12,4).astype(np.float16),dt=dt)
    write(out/"live.json",{"run":cfg["name"],"ticks":length,"seconds":length*dt,"block_ticks":64,"single_recording_group":group,
         "warm_block_wall_s_median":float(np.median(costs[2:])),"conditions_batched":len(modes),
         "scores":metrics,"prompts_and_free_decodes":transcripts,"generated_sha256":sha_file(out/"generated.npz")})
    print(json.dumps({"run":run.name,"seconds":length*dt,"scores":metrics}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","data","lock"): p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--backend",default="tpu")
    a=p.parse_args(); run_live(a.run,a.data,a.lock,a.backend)
