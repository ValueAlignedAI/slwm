"""Disposable random-data GPU fit/speed check; no source examples or saved weights."""
import fcntl
import json
from pathlib import Path
import time
import torch
from models.slwm_general import GeneralWorld
from training.general_sources import write
from training.research_sources import file_sha


def main():
    cfg=json.loads(Path("configs/t2/general_transfer.json").read_text()); results=[]
    torch.set_num_threads(4)
    with open("/tmp/slwm_research_gpu.lock","w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for variant in ("transformer","slwm"):
            for batch in (32,64):
                try:
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    model=GeneralWorld(variant=variant,phone_count=79).cuda().train(); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
                    obs={"time":torch.arange(32,device="cuda")[None].float().expand(batch,-1)*.04,"time_unit":torch.zeros(batch,device="cuda",dtype=torch.long),"valid":torch.ones(batch,32,device="cuda",dtype=torch.bool),"images":torch.randn(batch,32,3,3,64,64,device="cuda"),"image_present":torch.ones(batch,32,3,device="cuda",dtype=torch.bool),"numeric":torch.randn(batch,32,64,device="cuda"),"numeric_present":torch.ones(batch,32,64,device="cuda",dtype=torch.bool),"text":torch.randint(0,50257,(batch,64),device="cuda"),"text_present":torch.ones(batch,64,device="cuda",dtype=torch.bool)}
                    durations=[]
                    for step in range(4):
                        torch.cuda.synchronize(); start=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda",dtype=torch.bfloat16):
                            state=model(obs); query=model.query(state,torch.arange(8,device="cuda")[None].expand(batch,-1).float()*.04+1.28,obs["time_unit"],2,torch.randn(batch,8,7,device="cuda"))
                            prediction=model.propose(query,"numeric"); loss=prediction.square().mean()+state["embedding"].square().mean()
                        assert torch.isfinite(loss); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); torch.cuda.synchronize(); durations.append(time.perf_counter()-start)
                    result={"variant":variant,"batch":batch,"parameters":sum(p.numel() for p in model.parameters()),"peak_bytes":torch.cuda.max_memory_allocated(),"step_seconds":durations,"warm_mean_seconds":sum(durations[1:])/3,"finite":True,"real_data_used":False,"weights_discarded":True}
                    del loss,prediction,query,state,optimizer,model,obs
                except torch.cuda.OutOfMemoryError as exc:
                    result={"variant":variant,"batch":batch,"oom":True,"error":str(exc)}
                results.append(result); print(json.dumps(result),flush=True)
        write(Path(cfg["root"])/"random_capacity_preflight.json",{"checks":results,"source_sha256":file_sha(__file__),"config_sha256":file_sha("configs/t2/general_transfer.json"),"no_scientific_model_fitted":True})


if __name__=="__main__": main()
