"""Bounded train-only compilation study; never edits scientific checkpoints."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from training.broad_batch import StreamCorpus,predict
from training.broad_stream_train import construct,losses,alignment_loss
from training.research_sources import file_sha
from training.t2_real_av import write_json

ROOT=Path("experiments/multimodal/t2/compile-benchmark-20261105")


def gradient(model):
    return torch.cat([p.grad.detach().float().flatten() for p in model.parameters() if p.grad is not None])


def backward(model,batch,cfg):
    with torch.autocast("cuda",dtype=torch.bfloat16):
        out=predict(model,batch); loss,_=losses(out,batch,cfg)
        loss=loss+cfg["training"]["alignment_weight"]*alignment_loss(model,batch)
    loss.backward()
    return float(loss.detach())


def agreement(a,b):
    return {"gradient_relative_error":float((a-b).norm()/a.norm()),
            "gradient_cosine":float(F.cosine_similarity(a,b,dim=0))}


def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); ROOT.mkdir(parents=True,exist_ok=True)
    assert not(ROOT/"summary.json").exists(),"Keep earlier benchmark output immutable"
    write_json(ROOT/"registered.json",{"source_sha256":file_sha(Path(__file__)),
        "torch":torch.__version__,"cuda":torch.version.cuda,"batch":128,"mode":"default",
        "compile_scope":"StreamWorld.process, shared across chunks and alignment views; fixed shapes 128 and 8",
        "warmup_updates":3,"timed_updates":16,"data_seed":718,
        "first_loss_absolute_tolerance":.001,"first_gradient_relative_tolerance":.02,"first_gradient_cosine_minimum":.999,
        "scope":"Future runtime candidate only; current registered B1 execution remains unchanged, no weights saved"})
    data=StreamCorpus(cfg,"train"); results={}
    for run in cfg["runs"][:2]:
        name=run["variant"]; eager=construct(cfg,run).cuda(); compiled=copy.deepcopy(eager)
        compiled.process=torch.compile(compiled.process,fullgraph=True,dynamic=False,mode="default")
        rng=np.random.default_rng(718); batch=data.batch(data.sample(rng,128))
        ref_loss=backward(eager,batch,cfg); first=gradient(eager).clone(); eager.zero_grad(set_to_none=True)
        repeated_loss=backward(eager,batch,cfg); repeat=agreement(first,gradient(eager)); repeat["loss_difference"]=abs(ref_loss-repeated_loss)
        eager.zero_grad(set_to_none=True); torch.cuda.synchronize(); cold=time.perf_counter()
        actual_loss=backward(compiled,batch,cfg); torch.cuda.synchronize(); cold=time.perf_counter()-cold
        check=agreement(first,gradient(compiled)); check["loss_difference"]=abs(ref_loss-actual_loss)
        compiled.zero_grad(set_to_none=True); del first,batch
        opts=[torch.optim.AdamW(m.parameters(),lr=.0003,betas=(.9,.95),weight_decay=.01,fused=True) for m in (eager,compiled)]
        times=[[],[]]; observed=[[],[]]; rng=np.random.default_rng(718)
        torch.cuda.reset_peak_memory_stats()
        for step in range(19):
            batch=data.batch(data.sample(rng,128))
            for index,(model,opt) in enumerate(zip((eager,compiled),opts)):
                torch.cuda.synchronize(); start=time.perf_counter(); opt.zero_grad(set_to_none=True)
                value=backward(model,batch,cfg); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step(); torch.cuda.synchronize(); elapsed=time.perf_counter()-start
                if step>=3: times[index].append(elapsed); observed[index].append(value)
        result={"variant":name,"first_agreement":check,"eager_repeat_agreement":repeat,
            "initial_compiled_forward_backward_seconds":cold,
            "eager_seconds":sum(times[0]),"compiled_seconds":sum(times[1]),
            "steady_step_speedup":sum(times[0])/sum(times[1]),
            "eager_step_seconds":times[0],"compiled_step_seconds":times[1],
            "trajectory_losses":{"eager":observed[0],"compiled":observed[1]},
            "trajectory_loss_max_difference":float(np.max(np.abs(np.array(observed[0])-observed[1]))),
            "passes_local_numerical_gate":check["loss_difference"]<.001 and check["gradient_relative_error"]<.02 and check["gradient_cosine"]>.999,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"weights_saved":False,
            "timing_scope":"Common CPU batch construction excluded; cold compile and steady update times separate; no long-run learning-quality equivalence claim"}
        results[name]=result; write_json(ROOT/"summary.json",results); print(json.dumps(result),flush=True)
        del eager,compiled,opts,batch; torch.cuda.empty_cache(); torch._dynamo.reset()


if __name__=="__main__": main()
