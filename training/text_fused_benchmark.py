"""Numerical agreement and disposable fused-head throughput before adopting it."""
import gc
import json
from pathlib import Path
import time
import fcntl
import torch
import torch.nn.functional as F
from training.scale_text_train import initialize,UniqueWindows
from training.text_fused_loss import fused_loss
from training.t2_real_av import write_json


def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    cfg=json.loads(Path("configs/t1/fineweb_scale.json").read_text()); rows=[]; numerical=[]
    with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for variant in ("gpt2","slwm"):
            model=initialize(cfg,variant,81).cuda(); stream=UniqueWindows(Path(cfg["corpus"])/"train.tokens.bin",512,81); x,y=stream.batch(0,2,"cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16): original=F.cross_entropy(model(x).float().reshape(-1,50257),y.reshape(-1))
            original.backward(); expected=model.token_embedding.weight.grad.detach().clone(); model.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16): fused=fused_loss(model,x,y)
            fused.backward(); actual=model.token_embedding.weight.grad.detach(); cosine=F.cosine_similarity(expected.flatten(),actual.flatten(),dim=0).item(); delta=abs(float(original)-float(fused)); relative=float((expected-actual).norm()/expected.norm())
            result={"variant":variant,"original_loss":float(original),"fused_loss":float(fused),"absolute_loss_difference":delta,"embedding_gradient_cosine":cosine,"relative_gradient_error":relative}; numerical.append(result); print(json.dumps(result),flush=True)
            assert delta<.01 and cosine>.999 and relative<.025
            del model,x,y,original,fused,expected,actual; gc.collect(); torch.cuda.empty_cache()
            for micro in (16,32):
                model=initialize(cfg,variant,81).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),fused=True); x,y=stream.batch(0,32,"cuda"); times=[]
                try:
                    for rep in range(5):
                        optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
                        for begin in range(0,32,micro):
                            with torch.autocast("cuda",dtype=torch.bfloat16): loss=fused_loss(model,x[begin:begin+micro],y[begin:begin+micro])
                            (loss/(32//micro)).backward()
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); torch.cuda.synchronize()
                        if rep>=2: times.append(time.perf_counter()-start)
                    row={"variant":variant,"micro_batch":micro,"effective_batch":32,"tokens_per_second":16384/(sum(times)/len(times)),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"finite_gradient":bool(torch.isfinite(norm)),"weights_retained":False}
                except torch.cuda.OutOfMemoryError: row={"variant":variant,"micro_batch":micro,"out_of_memory":True}
                rows.append(row); print(json.dumps(row),flush=True); del model,optimizer,x,y; gc.collect(); torch.cuda.empty_cache()
    write_json(Path("experiments/text/t1/batch-benchmark-20261105/fused_results.json"),{"numerical_agreement":numerical,"rows":rows,"scope":"Disposable runtime experiment; no scientific weight updates retained"})


if __name__=="__main__": main()
