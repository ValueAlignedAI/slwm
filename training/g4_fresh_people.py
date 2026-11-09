"""A new speaker split for the decoded-state intervention, with all eight compared weights bound first."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from training.g4_data import Writer,grid,read


def prepare(repo,data):
    os.chdir(repo)
    dest=data/"bridge"
    if (dest/"card.json").exists(): return
    root=repo/"artifacts/t2_audio_visual/grid_content_v1"
    rows=read(root/"calibration.json"); actors=sorted({r["actor"] for r in rows})
    if actors!=["GRID-22","GRID-23","GRID-24"]: raise ValueError("Unexpected independent speaker split")
    for role in ("train","validation","test"):
        if set(actors)&{r["actor"] for r in read(root/(role+".json"))}: raise ValueError("Speaker leakage")
    with tempfile.TemporaryDirectory(dir=data) as tmp:
        part=Path(tmp)/"bridge"
        w=Writer(part,len(rows)); grid(w,root,"bridge",len(rows),source_split="calibration"); w.finish("bridge")
        part.rename(dest)
    print("prepared independent GRID speakers",actors,len(rows),flush=True)


def score(root,data,device):
    from training.g4_train import Data,sha_file,write
    from training.g4_eval import evaluate,weights
    from training.g4_transfer import grid_views
    expected=[f"g4-v5e-{tag}-medium-{core}-{policy}-{seed}"
              for seed in (81,82) for core in ("gpt2","signal") for tag,policy in (("fair-all","cross"),("tied","tied"))]
    record={"endpoints":{n:read(root/n/"complete.json")["weight_file_sha256"] for n in expected},
            "source_sha256":sha_file(Path(__file__)),"transfer_sha256":sha_file(Path("training/g4_transfer.py")),
            "evaluation_sha256":sha_file(Path("training/g4_eval.py")),"fresh_data_sha256":sha_file(data/"bridge/card.json")}
    lock=root/"new-speaker-lock.json"
    if lock.exists() and read(lock)!=record: raise ValueError("New speaker confirmation identity changed")
    write(lock,record)
    fresh=Data(data,"bridge")
    rows={}
    for n in expected:
        r=root/n
        if not (r/"new-speaker-transfer.json").exists():
            cfg,complete,spec,p,stats=weights(r)
            outcome=grid_views(data,"bridge",spec,p,stats,64)
            write(r/"new-speaker-transfer.json",{"run":n,"weight_sha256":complete["weight_file_sha256"],"grid":outcome})
        if not (r/"bridge.json").exists():
            evaluate(r,data,"bridge",64,device,data_cache=fresh)
        metric=read(r/"new-speaker-transfer.json")["grid"]["command_balanced_accuracy"]
        rows[n]={"audio_to_visual":metric["audio-to-visual"],"visual_to_audio":metric["visual-to-audio"],
                 "mean_swapped":.5*(metric["audio-to-visual"]+metric["visual-to-audio"])}
        print(n,rows[n],flush=True)
    write(root/"new-speaker-summary.json",{"lock":sha_file(lock),"rows":rows,"time":time.time()})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("phase",choices=["prepare","score"]); p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--out",type=Path); p.add_argument("--backend",default="cpu")
    a=p.parse_args()
    if a.phase=="prepare": prepare(a.repo,a.data)
    else: score(a.out,a.data,a.backend)
