"""Post-hoc frozen-field linear decoders: diagnose representation vs head loss.

The core stays frozen. Fit streaming least-squares heads on training frames
only, with fixed ridge=10. They are exploratory supervised diagnostic heads,
not changes to the preregistered model/gates. Original outputs are retained.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from models.slwm_motion import MotionSLWM
from training.t2_motion_train import load_data,crop,masked_inputs,basic_errors
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json
from exploration.motion_outputs import inverse_appearance,inverse_motion,invert_mel,save_video,save_wave


def solve_stats(stats: dict,regularization: float=10.) -> dict:
    """Streaming frame sums -> coefficient [D,288] and intercept [288]."""
    n=stats["n"]; xm=stats["x"]/n; ym=stats["y"]/n
    gram=stats["xx"]-np.outer(stats["x"],stats["x"])/n
    gram.flat[::len(gram)+1]+=regularization
    cross=stats["xy"]-np.outer(stats["x"],stats["y"])/n
    coef=np.linalg.solve(gram,cross)
    return {"coefficient":coef.astype(np.float32),"intercept":(ym-xm@coef).astype(np.float32)}


def fit(model,data,window):
    d=model.width
    stats={mode:{"n":0,"x":np.zeros(d),"y":np.zeros(288),"xx":np.zeros((d,d)),"xy":np.zeros((d,288))} for mode in ("audio","video")}
    with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        for begin in range(0,len(data["rows"]),32):
            ids=np.arange(begin,min(begin+32,len(data["rows"]))); starts=(data["lengths"][ids]-window)//2
            a,v=crop(data,ids,starts,window); y=torch.cat([a,v],-1).reshape(-1,288).double().cpu().numpy()
            for mode in stats:
                am,vm=masked_inputs(a,v,mode); z,_=model.encode(a,v,am,vm,positions=True)
                x=z.reshape(-1,d).double().cpu().numpy(); s=stats[mode]
                s["n"]+=len(x); s["x"]+=x.sum(0); s["y"]+=y.sum(0); s["xx"]+=x.T@x; s["xy"]+=x.T@y
    return {mode:solve_stats(s) for mode,s in stats.items()}


def predict(model,a,v,mode,weights):
    """Only the supplied stream's [B,T,D] frozen field -> two edge outputs."""
    am,vm=masked_inputs(a,v,mode); z,_=model.encode(a,v,am,vm,positions=True)
    y=z.float()@weights["coefficient"]+weights["intercept"]
    return {"audio":y[...,:128],"video":y[...,128:]}


def evaluate(model,data,window,weights):
    errors={}
    with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        for begin in range(0,len(data["rows"]),32):
            ids=np.arange(begin,min(begin+32,len(data["rows"]))); starts=(data["lengths"][ids]-window)//2
            a,v=crop(data,ids,starts,window)
            for mode in weights:
                e=basic_errors(predict(model,a,v,mode,weights[mode]),a,v)
                for key,value in e.items(): errors.setdefault(mode+"_input_"+key,[]).append(value.cpu().numpy())
    return {key:float(np.concatenate(value).mean()) for key,value in errors.items()}


def render(model,a,v,mode,weights,window):
    length=len(a); starts=list(range(0,length-window+1,window//2))
    if starts[-1]!=length-window: starts.append(length-window)
    out=np.zeros((length,288),np.float32); count=np.zeros((length,1),np.float32)
    with torch.inference_mode(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        for start in starts:
            p=predict(model,a[None,start:start+window],v[None,start:start+window],mode,weights)
            out[start:start+window]+=torch.cat([p["audio"],p["video"]],-1)[0].float().cpu().numpy(); count[start:start+window]+=1
    return out/count


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text()); root=Path(cfg["artifacts"])
    registry=json.loads((root/"exploration/registry.json").read_text()); gallery=json.loads((root/"exploration/gallery.json").read_text())["examples"]
    data={s:load_data(Path(cfg["corpus"]),s,torch.device("cuda")) for s in ("train","validation","test")}
    with np.load(Path(cfg["corpus"])/"codec.npz",allow_pickle=False) as archive: codec={k:archive[k] for k in archive.files}
    torch.set_num_threads(4); results={}
    for identity in registry["selected_models"][:2]:
        folder=root/identity; checkpoint=torch.load(folder/"best.pt",map_location="cuda",weights_only=False)
        model=MotionSLWM(checkpoint["config"]["run"]["model"]).cuda().eval(); model.load_state_dict(checkpoint["model"])
        raw=fit(model,data["train"],cfg["training"]["window"])
        weights={mode:{key:torch.from_numpy(value).cuda() for key,value in w.items()} for mode,w in raw.items()}
        out=root/"exploration"/identity
        for mode,w in raw.items(): np.savez_compressed(out/(mode+"_linear_decoder.npz"),**w)
        results[identity]={"scope":"post-hoc frozen-core supervised decoder, fixed ridge 10; original heads and frozen metrics unchanged",
            "fitted_parameters":sum(w["coefficient"].size+w["intercept"].size for w in raw.values()),
            "validation":evaluate(model,data["validation"],cfg["training"]["window"],weights),
            "test":evaluate(model,data["test"],cfg["training"]["window"],weights),
            "checkpoint_sha256":sha256(folder/"best.pt"),"source_sha256":sha256(Path(__file__))}
        example=next(e for e in gallery if e["experiment_id"]==identity and e["selection"]=="median_error")
        index=next(i for i,r in enumerate(data["validation"]["rows"]) if r["sample_id"]==example["sample_id"])
        length=int(data["validation"]["lengths"][index]); a=data["validation"]["audio"][index,:length]; v=data["validation"]["video"][index,:length]
        case=Path(example["folder"])
        for mode in raw:
            p=render(model,a,v,mode,weights[mode],cfg["training"]["window"])
            np.savez_compressed(case/(mode+"_linear_decoder_predictions.npz"),features=p)
            save_video(case/(mode+"_linear_mouth.gif"),inverse_appearance(p[:,128:224],codec),f"FROZEN PROBE: {mode} -> mouth (linear head)")
            save_wave(case/(mode+"_linear_audio.wav"),invert_mel(p[:,:64],codec))
        print(json.dumps({"linear_decoder":identity,"results":results[identity]}),flush=True)
        del model,checkpoint; torch.cuda.empty_cache()
    write_json(root/"exploration/linear_decoder_metrics.json",results)
    destination=root/"exploration/source_snapshot/exploration/motion_linear_decoders.py"; destination.parent.mkdir(exist_ok=True,parents=True); shutil.copy2(Path(__file__),destination)


if __name__=="__main__": main()
