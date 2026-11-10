"""Equal-data causal GPT-2-scale runs on immutable broad text, RTX3090."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from training.t1_torch_text import TorchGPT2LM,TorchSLWMTextLM
from training.research_sources import file_sha
from training.t2_real_av import stable_hash,write_json


class UniqueWindows:
    """Immutable uint16 token stream -> disjoint LongTensor[B,T] input/targets."""
    def __init__(self,path,length,seed):
        self.tokens=np.memmap(path,dtype="<u2",mode="r"); self.length=length; self.blocks=(len(self.tokens)-1)//length
        self.offset=seed%self.blocks; self.stride=(seed*7919+104729)%self.blocks
        while math.gcd(self.stride,self.blocks)!=1: self.stride+=1
    def batch(self,index,batch,device):
        if index+batch>self.blocks: raise ValueError("Training token windows would repeat")
        ids=(self.offset+self.stride*np.arange(index,index+batch,dtype=np.int64))%self.blocks
        rows=np.stack([self.tokens[i*self.length:i*self.length+self.length+1] for i in ids]).astype(np.int64)
        values=torch.as_tensor(rows,device=device); return values[:,:-1].contiguous(),values[:,1:].contiguous()


def initialize(cfg,variant,seed):
    torch.manual_seed(seed); common=TorchGPT2LM(**cfg["model"])
    if variant=="gpt2": return common
    torch.manual_seed(seed+1000); model=TorchSLWMTextLM(**cfg["model"],flags={})
    for name in ("token_embedding","position_embedding","ln_f"): getattr(model,name).load_state_dict(getattr(common,name).state_dict())
    for a,b in zip(model.blocks,common.blocks):
        for name in ("attn","mlp"): getattr(a,name).load_state_dict(getattr(b,name).state_dict())
        a.norm_attn.load_state_dict(b.ln_1.state_dict()); a.norm_mlp.load_state_dict(b.ln_2.state_dict())
    return model


def evaluate(model,stream,cfg):
    tc=cfg["training"]; values=[]; model.eval()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for i in range(tc["eval_batches"]):
            x,y=stream.batch(i*tc["micro_batch"],tc["micro_batch"],"cuda"); loss=F.cross_entropy(model(x).float().reshape(-1,cfg["model"]["vocab_size"]),y.reshape(-1)); values.append(float(loss))
    return {"nll":float(np.mean(values)),"perplexity":float(np.exp(np.mean(values))),"tokens":tc["eval_batches"]*tc["micro_batch"]*tc["sequence_length"]}


def generate(model,cfg,seed,sample):
    import tiktoken
    enc=tiktoken.get_encoding("gpt2"); rows=[]; model.eval(); torch.manual_seed(seed+9000)
    for prompt in cfg["prompts"]:
        ids=torch.tensor([enc.encode(prompt)],device="cuda")
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for _ in range(64):
                logits=model(ids[:,-cfg["training"]["sequence_length"]:])[:,-1].float()
                if sample:
                    logits=logits/.8; threshold=logits.topk(50).values[:,-1:]; logits=logits.masked_fill(logits<threshold,-float("inf")); next_id=torch.multinomial(logits.softmax(-1),1)
                else: next_id=logits.argmax(-1,keepdim=True)
                ids=torch.cat([ids,next_id],1)
                if int(next_id)==enc.eot_token: break
        rows.append({"prompt":prompt,"generated_text":enc.decode(ids[0].tolist()),"mode":"top50_temperature.8" if sample else "greedy","seed":seed+9000})
    return rows


def train(cfg,run,card,sources):
    root=Path(cfg["artifacts"]); out=root/run["id"]; out.mkdir(exist_ok=True)
    if (out/"complete.json").exists(): return
    tc=cfg["training"]; model=initialize(cfg,run["variant"],run["seed"]).cuda(); torch.cuda.reset_peak_memory_stats()
    optimizer=torch.optim.AdamW(model.parameters(),lr=tc["lr"],betas=(.9,.95),weight_decay=tc["weight_decay"],fused=True)
    trainset=UniqueWindows(Path(cfg["corpus"])/"train.tokens.bin",tc["sequence_length"],run["seed"]); valset=UniqueWindows(Path(cfg["corpus"])/"validation.tokens.bin",tc["sequence_length"],901)
    n=sum(p.numel() for p in model.parameters()); provenance={"source_hashes":sources,"config_hash":stable_hash(cfg),"dataset_card_sha256":file_sha(Path(cfg["corpus"])/"dataset_card.json"),"parameters":n,"gpu":torch.cuda.get_device_name(),"warmstart":"None; all random; paired common module initializers","training_context":512}
    write_json(out/"config.json",{"matrix":cfg,"run":run,"provenance":provenance}); best=float("inf"); curves=[]; losses=[]; started=time.perf_counter(); consecutive=0; stopped=False; tokens_per_step=tc["micro_batch"]*tc["gradient_accumulation"]*tc["sequence_length"]
    for step in range(tc["steps"]+1):
        if step%tc["eval_every"]==0 or step==tc["steps"]:
            val=evaluate(model,valset,cfg); curves.append({"step":step,"tokens":step*tokens_per_step,"validation":val,"training_nll":float(np.mean(losses)) if losses else None,"seconds":time.perf_counter()-started}); losses=[]
            previous=best; improved=val["nll"]<best
            if improved:
                best=val["nll"]; torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"run":run,"config":cfg,"provenance":provenance,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},out/"best.pt")
            consecutive=consecutive+1 if val["nll"]>previous*(1+tc["overfit_relative_rise"]) else 0
            write_json(out/"curves.json",curves); print(json.dumps({"run":run["id"],"step":step,"tokens":step*tokens_per_step,"validation_nll":val["nll"],"seconds":time.perf_counter()-started}),flush=True)
            if consecutive>=tc["overfit_patience"]: stopped=True; break
        if step==tc["steps"]: break
        model.train(); optimizer.zero_grad(set_to_none=True); lr=tc["lr"]*min(1.,(step+1)/tc["warmup"])*(tc["minimum_lr_fraction"]+(1-tc["minimum_lr_fraction"])*.5*(1+math.cos(math.pi*step/tc["steps"])))
        for g in optimizer.param_groups: g["lr"]=lr
        for micro in range(tc["gradient_accumulation"]):
            index=(step*tc["gradient_accumulation"]+micro)*tc["micro_batch"]; x,y=trainset.batch(index,tc["micro_batch"],"cuda")
            with torch.autocast("cuda",dtype=torch.bfloat16): loss=F.cross_entropy(model(x).float().reshape(-1,cfg["model"]["vocab_size"]),y.reshape(-1))
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite scale loss")
            (loss/tc["gradient_accumulation"]).backward(); losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),tc["gradient_clip"]); optimizer.step()
    torch.save({"model":model.state_dict(),"step":step,"run":run,"config":cfg,"provenance":provenance},out/"last.pt")
    chosen=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
    write_json(out/"complete.json",{"run":run,"parameters":n,"selected_step":chosen["step"],"selected_validation_nll":best,"updates":step,"tokens_seen":step*tokens_per_step,"available_train_tokens":card["tokens"]["train"],"overfit_stop":stopped,"wall_seconds":time.perf_counter()-started,"peak_gpu_bytes":torch.cuda.max_memory_allocated(),"checkpoint_sha256":file_sha(out/"best.pt"),"last_sha256":file_sha(out/"last.pt")})
    del model,optimizer,chosen; torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/t1/fineweb_scale.json"); p.add_argument("--stage",choices=("train","evaluate"),required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"]); root.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    card=json.loads((Path(cfg["corpus"])/"dataset_card.json").read_text())
    if card["tokens"]["train"]<2_500_000_000: raise ValueError("Scale corpus too small")
    for name,digest in card["files"].items():
        if file_sha(Path(cfg["corpus"])/name)!=digest: raise ValueError("Immutable text corpus changed")
    if a.stage=="train":
        names=(a.config,cfg["protocol"],"training/scale_text_train.py","training/prepare_scale_text.py","training/research_sources.py","training/t1_torch_text.py"); hashes={}
        for name in names:
            dest=root/"source_snapshot"/name; dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and file_sha(dest)!=file_sha(name): raise ValueError("Registered scale source changed")
            if not dest.exists(): shutil.copyfile(name,dest)
            hashes[name]=file_sha(name)
        write_json(root/"source_hashes.json",hashes)
        for run in cfg["runs"]: train(cfg,run,card,hashes)
        return
    choices={r["id"]:json.loads((root/r["id"]/"complete.json").read_text()) for r in cfg["runs"]}; write_json(root/"selection.json",choices)
    test=UniqueWindows(Path(cfg["corpus"])/"test.tokens.bin",cfg["training"]["sequence_length"],902); results={}
    for run in cfg["runs"]:
        out=root/run["id"]; ck=torch.load(out/"best.pt",map_location="cpu",weights_only=False); model=initialize(cfg,run["variant"],run["seed"]); model.load_state_dict(ck["model"],strict=True); model.cuda().eval()
        value=evaluate(model,test,cfg); results[run["id"]]=value; write_json(out/"test.json",value); write_json(out/"samples.json",generate(model,cfg,run["seed"],False)+generate(model,cfg,run["seed"],True)); del model,ck; torch.cuda.empty_cache()
    guard={}
    for seed in (81,82):
        g=next(r["id"] for r in cfg["runs"] if r["seed"]==seed and r["variant"]=="gpt2"); s=next(r["id"] for r in cfg["runs"] if r["seed"]==seed and r["variant"]=="slwm")
        ratio=choices[s]["selected_validation_nll"]/choices[g]["selected_validation_nll"]; guard[str(seed)]={"slwm_to_gpt2_validation_nll_ratio":ratio,"passes_early_guardrail":ratio<=1.25 and not choices[s]["overfit_stop"] and not choices[g]["overfit_stop"]}
    write_json(root/"summary.json",{"scope":"Early causal text learning only;16.8M tokens/run, not converged124M or jointly trained multimodal quality","runs":choices,"test":results,"early_guardrail":guard,"training_minutes":sum(v["wall_seconds"] for v in choices.values())/60,"source_hashes_sha256":file_sha(root/"source_hashes.json")}); print(json.dumps({"scale_evaluation_complete":True,"guardrail":guard}),flush=True)


if __name__=="__main__": main()
