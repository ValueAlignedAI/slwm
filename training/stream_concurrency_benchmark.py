"""Disposable train-only GPU concurrency study; scientific weights are never saved."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from training.broad_stream_train import construct,losses,alignment_loss
from training.broad_batch import StreamCorpus,predict
from training.t2_real_av import write_json

ROOT=Path("experiments/multimodal/t2/concurrency-benchmark-20261105")
MPS_ENV={"CUDA_MPS_PIPE_DIRECTORY":"/tmp/slwm_mps_20261105","CUDA_MPS_LOG_DIRECTORY":"/tmp/slwm_mps_log_20261105"}


def mps_start():
    for folder in MPS_ENV.values(): Path(folder).mkdir(exist_ok=True)
    subprocess.run(["nvidia-cuda-mps-control","-d"],env={**os.environ,**MPS_ENV},check=True)


def mps_stop():
    subprocess.run(["nvidia-cuda-mps-control"],input=b"quit\n",env={**os.environ,**MPS_ENV},check=True,capture_output=True)


def gradient_sketch(model):
    rng=np.random.default_rng(271); samples=[]; norm=0.
    for p in model.parameters():
        if p.grad is None: continue
        g=p.grad.detach().float().flatten(); norm+=float(g.double().square().sum()); ids=torch.as_tensor(rng.integers(g.numel(),size=min(32,g.numel())),device=g.device); samples.extend(g[ids].cpu().tolist())
    return {"l2_norm":norm**.5,"fixed_coordinate_sketch":samples}


def worker(name,index,barrier):
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=True; cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); run=cfg["runs"][index]; data=StreamCorpus(cfg,"train"); model=construct(cfg,run).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,betas=(.9,.95),weight_decay=.01,fused=True); rng=np.random.default_rng(718); losses_seen=[]; sketch=None
    def step():
        nonlocal sketch
        batch=data.batch(data.sample(rng,128)); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out=predict(model,batch); loss,_=losses(out,batch,cfg); loss=loss+.05*alignment_loss(model,batch)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); return float(loss.detach())
    for _ in range(3): step()
    torch.cuda.synchronize(); ready=ROOT/(name+".ready"); ready.touch()
    if barrier:
        while not Path(barrier).exists(): time.sleep(.02)
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    for _ in range(16): losses_seen.append(step())
    torch.cuda.synchronize(); seconds=time.perf_counter()-start; sketch=gradient_sketch(model); write_json(ROOT/(name+".json"),{"name":name,"variant":run["variant"],"seed":run["seed"],"timed_updates":16,"batch":128,"seconds":seconds,"windows_per_second":16*128/seconds,"losses":losses_seen,"gradient":sketch,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),"scientific_weights_saved":False})


def launch(name,index,barrier=None):
    command=[sys.executable,"-m","training.stream_concurrency_benchmark","--worker",name,"--index",str(index)]
    if barrier: command.extend(("--barrier",str(barrier)))
    log=(ROOT/(name+".log")).open("w"); process=subprocess.Popen(command,env={**os.environ,**MPS_ENV},stdout=log,stderr=subprocess.STDOUT); return process,log


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX); mps_start()
        try:
            for index in range(3):
                name="single"+str(index); process,log=launch(name,index); assert process.wait()==0,name; log.close()
            results={}
            for label,indices in (("mixed",(0,1)),("hybrid_pair",(0,2))):
                barrier=ROOT/(label+".start"); barrier.unlink(missing_ok=True); names=[label+str(i) for i in indices]; children=[launch(n,i,barrier) for n,i in zip(names,indices)]; started=time.monotonic()
                while not all((ROOT/(n+".ready")).exists() for n in names):
                    if any(p.poll() not in (None,0) for p,_ in children): raise RuntimeError("Concurrency worker failed before barrier")
                    if time.monotonic()-started>180: raise TimeoutError("Concurrency worker setup")
                    time.sleep(.1)
                start=time.perf_counter(); barrier.touch()
                for process,log in children: assert process.wait()==0,label; log.close()
                wall=time.perf_counter()-start; rows=[json.loads((ROOT/(n+".json")).read_text()) for n in names]; single=[json.loads((ROOT/("single"+str(i)+".json")).read_text()) for i in indices]; agreement=[]
                for a,b in zip(single,rows):
                    x=np.asarray(a["gradient"]["fixed_coordinate_sketch"]); y=np.asarray(b["gradient"]["fixed_coordinate_sketch"]); agreement.append({"loss_max_difference":float(np.max(np.abs(np.array(a["losses"])-b["losses"]))),"gradient_sketch_cosine":float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),"gradient_sketch_relative_error":float(np.linalg.norm(x-y)/np.linalg.norm(x))})
                timed=max(r["seconds"] for r in rows); speedup=sum(r["seconds"] for r in single)/timed; results[label]={"timed_group_seconds":timed,"full_group_seconds_including_exit":wall,"sequential_seconds":sum(r["seconds"] for r in single),"speedup":speedup,"rows":rows,"agreement":agreement,"passes_numerical_gate":all(v["gradient_sketch_cosine"]>.999 and v["gradient_sketch_relative_error"]<.02 and v["loss_max_difference"]<.005 for v in agreement)}; write_json(ROOT/"summary.json",results); print(json.dumps({"concurrency":label,"speedup":speedup,"agreement":agreement}),flush=True)
        finally: mps_stop()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--worker"); p.add_argument("--index",type=int); p.add_argument("--barrier"); a=p.parse_args()
    if a.worker: worker(a.worker,a.index,a.barrier)
    else: run()


if __name__=="__main__": main()
