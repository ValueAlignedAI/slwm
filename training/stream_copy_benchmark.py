"""Single-context full-batch alias benchmark with matched independent optimizers."""
import copy
import fcntl
import json
from pathlib import Path
import time
import numpy as np
import torch
from training.broad_stream_train import construct,losses,alignment_loss,subset
from training.broad_batch import StreamCorpus,predict
from training.t2_real_av import write_json


def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True; cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); root=Path("experiments/multimodal/t2/copy-benchmark-20261105"); root.mkdir(parents=True,exist_ok=True); data=StreamCorpus(cfg,"train"); results={}
    with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for run in cfg["runs"][:2]:
            base=construct(cfg,run).cuda(); alias=copy.deepcopy(base); models=(base,alias); optimizers=[torch.optim.AdamW(m.parameters(),lr=.0003,betas=(.9,.95),weight_decay=.01,fused=True) for m in models]; rng=np.random.default_rng(718); times=[[],[]]; scores=[[],[]]; first_grads=[]
            for step in range(19):
                batch=data.batch(data.sample(rng,128)); ids=torch.arange(128,device="cuda")
                for k,(model,optimizer) in enumerate(zip(models,optimizers)):
                    torch.cuda.synchronize(); started=time.perf_counter(); part=subset(batch,ids) if k==0 else batch; optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        out=predict(model,part); loss,_=losses(out,part,cfg); loss=loss+.05*alignment_loss(model,part)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); torch.cuda.synchronize(); elapsed=time.perf_counter()-started
                    if step>=3: times[k].append(elapsed); scores[k].append(float(loss.detach()))
                    if step==0: first_grads.append(torch.cat([p.grad.detach().flatten().float() for p in model.parameters() if p.grad is not None]))
                    del out,part,loss
            a,b=first_grads; grad_relative=float((a-b).norm()/a.norm()); difference=sum(float((a-b).double().square().sum()) for a,b in zip(base.parameters(),alias.parameters()))**.5; norm=sum(float(a.double().square().sum()) for a in base.parameters())**.5; lossdiff=float(np.max(np.abs(np.asarray(scores[0])-scores[1]))); speed=float(sum(times[0])/sum(times[1])); results[run["variant"]]={"baseline_seconds":sum(times[0]),"alias_seconds":sum(times[1]),"speedup":speed,"first_gradient_relative_error":grad_relative,"final_parameter_relative_error":difference/norm,"trajectory_loss_max_difference":lossdiff,"passes_numerical_gate":grad_relative<1e-5 and difference/norm<1e-5 and lossdiff<.005,"scientific_weights_saved":False,"scope":"Model/optimizer steps only; common CPU batch construction excluded"}; write_json(root/"summary.json",results); print(json.dumps(results[run["variant"]]),flush=True); del base,alias,models,optimizers,first_grads,a,b,batch; torch.cuda.empty_cache()


if __name__=="__main__": main()
