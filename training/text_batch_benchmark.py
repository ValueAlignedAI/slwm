"""Disposable 3090 batching measurements; no retained weights or quality claims."""
import gc
import json
from pathlib import Path
import time
import fcntl
import torch
import torch.nn.functional as F
from training.scale_text_train import initialize,UniqueWindows
from training.t2_real_av import write_json


def main():
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    cfg=json.loads(Path("configs/t1/fineweb_scale.json").read_text()); rows=[]
    with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for variant in ("gpt2","slwm"):
            for micro in (8,16,32):
                model=initialize(cfg,variant,81).cuda(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),fused=True)
                stream=UniqueWindows(Path(cfg["corpus"])/"train.tokens.bin",512,81); x,y=stream.batch(0,32,"cuda"); times=[]; failed=None
                try:
                    for repetition in range(5):
                        optimizer.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter()
                        for begin in range(0,32,micro):
                            with torch.autocast("cuda",dtype=torch.bfloat16): loss=F.cross_entropy(model(x[begin:begin+micro]).float().reshape(-1,50257),y[begin:begin+micro].reshape(-1))
                            (loss/(32//micro)).backward()
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); torch.cuda.synchronize(); seconds=time.perf_counter()-start
                        if repetition>=2: times.append(seconds)
                    row={"variant":variant,"micro_batch":micro,"effective_batch":32,"sequence_length":512,"median_step_seconds":sorted(times)[len(times)//2],"tokens_per_second":16384/(sum(times)/len(times)),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),"finite_gradient":bool(torch.isfinite(norm)),"weights_retained":False}
                except torch.cuda.OutOfMemoryError:
                    row={"variant":variant,"micro_batch":micro,"out_of_memory":True,"weights_retained":False}
                rows.append(row); print(json.dumps(row),flush=True); del model,optimizer,x,y; gc.collect(); torch.cuda.empty_cache()
    root=Path("experiments/text/t1/batch-benchmark-20261105"); root.mkdir(parents=True,exist_ok=True); write_json(root/"results.json",{"gpu":torch.cuda.get_device_name(),"rows":rows,"scope":"Disposable synthetic optimization on already-used token windows; no retained scientific weights"})


if __name__=="__main__": main()
