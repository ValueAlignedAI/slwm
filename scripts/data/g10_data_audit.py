"""Audit public task splits, clocks, transcript exposure and actual source masks without loading held video."""
import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from models.g10_context import PATTERNS
from g10_holo import session_group


def main(root,star,dest):
    sets=json.loads((root/"banks.json").read_text());data={};groups=set()
    for role in ("train","dev","familiar","novel"):
        rows=sets[role]
        ids={session_group(r["name"]) for r in rows}
        if len(ids)!=len(rows) or ids&groups:raise ValueError("Recording group leakage")
        groups|=ids
        tasks=sorted({r["task"] for r in rows});novel=[any(k in task for k in ("coffee","computer")) for task in tasks]
        if any(v!=(role=="novel") for v in novel):raise ValueError("Held task family entered the wrong split")
        fields={k:sum(r[k] for r in rows) for k in ("ticks","completed_utterances","transcribed_words","text_present_ticks","stale_imu_ticks","action_examples","mistakes")}
        data[role]={**fields,"recordings":len(rows),"hours":fields["ticks"]/36000,"tasks":tasks,"sha256":json.loads((root/("holo-"+role+".json")).read_text())["sha256"],
                    "session_list":[{"name":r["name"],"task":r["task"],"ticks":r["ticks"],"sha256":r["sha256"]} for r in rows]}
    with np.load(root/"holo-train.npz") as b,np.load(star) as a:
        eligible=((b["x"]!=0).any(-1)&(b["n"][:,6]>0));length=b["lengths"].astype(np.int32);off=np.cumsum(np.r_[0,length[:-1]]).astype(np.int32);sl=a["lengths"]
    exposure={}
    for seed in (1001,1002,1003):
        def draw(step):
            rng=jax.random.fold_in(jax.random.PRNGKey(seed+431),step);choice,temporal,mask,_=jax.random.split(rng,4)
            si=jax.random.randint(jax.random.fold_in(choice,0),(4,),0,len(sl));hi=jax.random.randint(jax.random.fold_in(choice,1),(4,),0,len(length))
            lens=jnp.r_[jnp.asarray(sl)[si],jnp.asarray(length)[hi]]
            span=lens[:,None]//8;jitter=(jax.random.uniform(temporal,(8,8))*jnp.maximum(span-33,1)).astype(jnp.int32)
            starts=jnp.asarray(off)[hi,None]+jnp.arange(8)[None,:]*span[4:]+jitter[4:]
            keep=PATTERNS[jax.random.randint(mask,(64,),0,len(PATTERNS))]
            return starts.reshape(-1),keep
        starts,keep=(np.asarray(v) for v in jax.jit(jax.vmap(draw))(jnp.arange(1536)))
        ix=starts[:,:,None]+np.arange(32);hk=keep[:,32:]
        exposure[str(seed)]=dict(sim_worlds=8*1536,sim_positions=8*512*1536,real_positions=64*32*1536,holo_positions=32*32*1536,
            distinct_holo_window_starts=len(np.unique(starts)),distinct_holo_clock_positions=len(np.unique(ix)),
            four_real_sources_assigned=float(keep.all(-1).mean()),four_holo_sources_assigned=float(hk.all(-1).mean()),
            four_holo_sources_and_present_utterance_and_fresh_imu=float((hk.all(-1)[:,:,None]&eligible[ix]).mean()),
            per_source_real_fraction_audio_camera_physics_words=keep.mean((0,1)).tolist())
    pins={k:json.loads((root/(k+"-index.json")).read_text())["etag"] for k in ("video_compress","video_pitch_shifted","imu")}
    obj=dict(study="G10",holo=data,matched_posttraining_exposure=exposure,source="https://holoassist.github.io/",source_etags=pins,
             labels_sha256=hashlib.sha256((root/"labels.json").read_bytes()).hexdigest(),splits_sha256=hashlib.sha256((root/"split.zip").read_bytes()).hexdigest(),
             source_notes="The compressed video member had no audio. Main AV features use the original pitch-shifted container. One recorded microphone is repeated across four spectral slots. Six IMU axes and their validity are supplied. Natural utterances are hashed after their annotated end with an eight-second expiry.")
    dest.parent.mkdir(exist_ok=True,parents=True);dest.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
    print([(k,v["recordings"],round(v["hours"],2)) for k,v in data.items()])

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("holo",type=Path);p.add_argument("star",type=Path);p.add_argument("dest",type=Path);a=p.parse_args();main(a.holo,a.star,a.dest)
