"""Run the released G10 controller or a selected cross-source decoder over an ongoing stream."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from data.g10_language import timeline
from evals.g9_live import initial_memory
from evals.g10_control import live_kernel
from models.g9_shared import encode,decode
from models.g10_alignment import encode_record
from scripts.infer.g9_world import checkpoint

SOURCES=("audio","vision","physics","words")
WIDTHS={"a":64,"v":768,"n":12,"w":8,"x":128}
HEADS={"audio":"a","vision":"v","physics":"n","event":"w","action":"policy","next_audio":"next_a","next_physics":"next_n"}


def read_input(path,sources,utterances=None):
    with np.load(path,allow_pickle=False) as f:
        selected=[k for k,src in zip(("a","v","n","w"),SOURCES) if src in sources and k in f]
        if "words" in sources and "x" in f:selected.append("x")
        if not selected and "u" not in f:raise ValueError("No selected source supplies the clock")
        length=len(f[selected[0]] if selected else f["u"])
        if not length:raise ValueError("Empty stream")
        rows={}
        for k,d in WIDTHS.items():
            src={"a":"audio","v":"vision","n":"physics","x":"words","w":"words"}[k]
            if src in sources and k in f:val=f[k]
            elif src in sources and k not in ("x","w"):raise ValueError("Missing selected sensor "+src)
            else:val=np.zeros((length,d),np.float32 if k=="n" else np.int8 if k=="x" else np.uint8)
            if val.shape!=(length,d) or not np.isfinite(val).all():raise ValueError("Invalid "+k+" clock or values")
            if k!="n" and not np.issubdtype(val.dtype,np.integer):raise ValueError("Expected an integer edge "+k)
            if k in ("a","v") and ((val<0).any() or (val>255).any()):raise ValueError("Quantized signal outside byte range")
            if k=="w" and ((val<0).any() or (val>=48).any()):raise ValueError("Unknown event token")
            if k=="x" and ((val< -127).any() or (val>127).any()):raise ValueError("Invalid natural-language feature")
            rows[k]=val
        if "words" in sources and "w" not in f and "x" not in f and utterances is None:raise ValueError("No selected word or transcript input")
        if utterances is not None:
            if "words" not in sources:raise ValueError("Select words to condition on speech transcripts")
            rows["x"]=timeline(json.loads(Path(utterances).read_text()),length)
        for k in ("u","plan"):
            v=f[k] if k in f else np.zeros(length,np.int32)
            if v.shape!=(length,) or not np.issubdtype(v.dtype,np.integer) or (v<0).any() or (v>=7).any():raise ValueError("Invalid discrete control "+k)
            rows[k]=v.astype(np.int32)
    return rows


def stream(p,s,rows,sources,emit):
    keep=jnp.asarray([[v in sources for v in SOURCES]])
    keep_state=initial_memory(s,1,512);outputs={k:[] for k in emit}
    @jax.jit
    def chunk(params,r,cache,t):
        z,cache=encode_record(params,s,r,keep,memory=cache,offset=t,with_memory=True) if "utterance_edge" in params else encode(params,s,r,keep,memory=cache,offset=t,window=512)
        pred=decode(params,z,r["plan"])
        if "instruction" in emit:
            if "utterance_read" not in params:raise ValueError("Natural words require a post-trained model")
            pred["instruction"]=(z@params["utterance_read"].astype(z.dtype)).astype(jnp.float32)
        return {k:z if k=="state" else pred[HEADS.get(k,k)] for k in emit},cache
    length=len(rows["u"])
    for i in range(0,length,32):
        ix=np.arange(i,min(i+32,length));r={k:jnp.asarray(v[None,ix]) for k,v in rows.items()}
        d=np.abs(rows["v"][ix].astype(np.int16)-rows["v"][np.maximum(0,ix-1)].astype(np.int16))
        r["m"]=jnp.asarray(np.clip(d.reshape(1,len(ix),256,3).mean(-1)/64,0,2),dtype=jnp.float32)
        result,keep_state=chunk(p,r,keep_state,jnp.int32(i))
        for k in emit:outputs[k].append(np.asarray(result[k][0],np.float32))
    return {k:np.concatenate(v) for k,v in outputs.items()}


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("weights",type=Path);ap.add_argument("--output",type=Path,required=True)
    main=ap.add_mutually_exclusive_group(required=True);main.add_argument("--input",type=Path);main.add_argument("--mission",action="store_true")
    ap.add_argument("--sources",nargs="+",choices=SOURCES,default=list(SOURCES));ap.add_argument("--emit",nargs="+",choices=[*HEADS,"instruction","state"],default=["state"])
    ap.add_argument("--utterances",type=Path);ap.add_argument("--seed",type=int,default=810537);ap.add_argument("--worlds",type=int,default=1);ap.add_argument("--changes",type=int,choices=(0,2),default=0)
    ap.add_argument("--instruction",choices=("continuous","brief","latched","blank"),default="continuous");args=ap.parse_args()
    spec,params,sha=checkpoint(args.weights)
    if args.mission:
        if not 1<=args.worlds<=256:raise ValueError("Expected 1 to 256 simultaneous worlds")
        a=np.asarray(live_kernel(spec,"full",args.changes,args.worlds,args.instruction)(params,jnp.int32(args.seed)))
        args.output.write_text(json.dumps(dict(parent_sha256=sha,source_seed=args.seed,hz=10,steps=512,issued_actions=a[:,:,5].T.tolist(),goal_occupancy=a[:,:,0].mean(0).tolist()),indent=2)+"\n")
    else:
        r=read_input(args.input,args.sources,args.utterances);out=stream(params,spec,r,args.sources,tuple(args.emit))
        np.savez_compressed(args.output,**out,parent_sha256=np.array(sha))
