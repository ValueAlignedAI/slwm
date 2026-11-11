"""Independently check CPU and CUDA at batch sizes one and 128 on held speakers."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scripts.train.g13_speech import probs,metrics
from scripts.train.g13_optim import ROOT
from scripts.train.g9_storage import write
from models.g13_speech import predict

def compute(backend):
    sp=np.load(ROOT/"data/speech/directions.npz");ix=np.flatnonzero(sp["role"]=="test");sample=ix[::23]
    out={}
    for portable in (False,True):
        name="portable" if portable else "native";directory=ROOT/"work"/("speech-portable" if portable else "speech")
        with np.load(directory/"selected.npz") as z:p={k:jnp.asarray(z[k]) for k in z.files}
        out[name+"_batch"]=probs(p,sp["x"][ix],portable)
        single=jax.jit(lambda params,x:jax.nn.softmax(predict(params,x,portable)[1]))
        out[name+"_single"]=np.concatenate([np.asarray(single(p,jnp.asarray(x[None]))) for x in sp["x"][sample]])
    np.savez_compressed(ROOT/"work"/("speech-parity-"+backend+".npz"),**out)

def summarize():
    sp=np.load(ROOT/"data/speech/directions.npz");ix=np.flatnonzero(sp["role"]=="test");step=np.arange(0,len(ix),23)
    cp=np.load(ROOT/"work/speech-parity-cpu.npz");gp=np.load(ROOT/"work/speech-parity-gpu.npz")
    vals={}
    for key in ("native","portable"):
        a=cp[key+"_batch"];b=gp[key+"_batch"]
        vals[key]=dict(clips=len(ix),speaker_groups=len(np.unique(sp["speaker"][ix])),single_samples=len(step),
             cpu=metrics(a,sp["y"][ix],sp["speaker"][ix]),gpu=metrics(b,sp["y"][ix],sp["speaker"][ix]),
             max_probability_difference=float(np.max(abs(a-b))),same_top=int((a.argmax(1)==b.argmax(1)).sum()),
             cpu_single_delta=float(abs(cp[key+"_single"]-a[step]).max()),gpu_single_delta=float(abs(gp[key+"_single"]-b[step]).max()))
    if vals["portable"]["same_top"]!=len(ix) or vals["portable"]["max_probability_difference"]>.003:
        raise ValueError("Replacement directional codec still has a cross-backend discrepancy: "+str(vals["portable"]))
    write(ROOT/"release/metrics/speech-parity.json",vals);print(vals,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("backend",choices=("gpu","cpu","summary"));a=p.parse_args()
    summarize() if a.backend=="summary" else compute(a.backend)
