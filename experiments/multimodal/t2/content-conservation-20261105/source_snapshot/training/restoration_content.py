"""Controlled R2 auxiliary-objective training; parent R1 source remains frozen."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from models.slwm_content import ContentRestorationSLWM,content_loss
from models.slwm_restoration import restoration_loss
from training.restoration_train import load_data,load_codec,crop,holes,evaluate,synchronization,output
from training.restoration_other import expanded_config
from training.t2_motion_train import alignment_loss,shuffled_indices
from training.t2_real_av import write_json,stable_hash
from training.t2_prepare_cremad import sha256


def load_other(cfg: dict,device="cuda") -> dict:
    root=Path(cfg["other_corpus"]); path=root/"other_test_manifest.json"; card=json.loads(path.read_text())
    for file,digest in card["files"].items():
        if sha256(root/file)!=digest: raise ValueError("New holdout changed")
    rows=json.loads((root/"other_test.json").read_text())
    return {"mel100":torch.from_numpy(np.load(root/"other_test_mel100.npy")).to(device),"video":torch.from_numpy(np.load(root/"other_test_video.npy")).to(device),
        "lengths":np.load(root/"other_test_lengths.npy"),"rows":rows,"video_present":np.zeros(len(rows),bool),"wrong":shuffled_indices(rows),"manifest_sha256":sha256(path),"split":"other_test"}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--evaluate",action="store_true"); args=parser.parse_args()
    cfg=expanded_config(args.config); root=Path(cfg["artifacts"]); root.mkdir(parents=True,exist_ok=True); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    codec=load_codec(cfg)
    if args.evaluate:
        for split in ("crema_test","libri_test","ravdess_test","other_test"):
            data=load_other(cfg) if split=="other_test" else load_data(cfg,split)
            for run in cfg["runs"]:
                checkpoint=torch.load(root/run["id"]/"best.pt",map_location="cpu",weights_only=False); model=ContentRestorationSLWM(cfg,run).cuda().eval(); model.load_state_dict(checkpoint["model"])
                result=evaluate(model,codec,data,True)
                if split in {"crema_test","ravdess_test"}: result["synchronization"]=synchronization(model,codec,data)
                result.update({"run":run,"split":split,"selected_step":checkpoint["step"],"checkpoint_sha256":sha256(root/run["id"]/"best.pt"),"manifest_sha256":data["manifest_sha256"]})
                write_json(root/run["id"]/(split+"_metrics.json"),result); del model
            del data; torch.cuda.empty_cache()
        return
    names=[args.config,cfg["parent_config"],cfg["protocol"],"models/slwm_content.py","training/restoration_content.py","training/restoration_other.py"]
    parent=json.loads(Path(cfg["parent_config"]).read_text()); names+=list(json.loads((Path(parent["artifacts"])/"source_hashes.json").read_text()))
    source={}
    for name in sorted(set(names)):
        p=Path(name); q=root/"source_snapshot"/p; q.parent.mkdir(parents=True,exist_ok=True)
        if q.exists() and sha256(q)!=sha256(p): raise ValueError("R2 training source changed")
        if not q.exists(): shutil.copyfile(p,q)
        source[name]=sha256(p)
    write_json(root/"source_hashes.json",source)
    data={s:load_data(cfg,s) for s in ("crema_train","libri_train","crema_validation","libri_validation")}; c,l=data["crema_train"],data["libri_train"]; tc=cfg["training"]; batch=tc["batch_size"]; nc=tc["crema_batch"]
    for run in cfg["runs"]:
        out=root/run["id"]; out.mkdir(exist_ok=True)
        if (out/"training_complete.json").exists(): continue
        torch.manual_seed(run["seed"]); torch.cuda.manual_seed_all(run["seed"]); model=ContentRestorationSLWM(cfg,run)
        old=cfg["checkpoints"]["signal"]
        if sha256(Path(old["path"]))!=old["sha256"]: raise ValueError("Warmstart changed")
        model.initialize_world(torch.load(old["path"],map_location="cpu",weights_only=False)["model"]); model=model.cuda(); rng=np.random.default_rng(run["seed"])
        world=[p for n,p in model.named_parameters() if n.startswith("world.")]; heads=[p for n,p in model.named_parameters() if not n.startswith("world.")]
        optimizer=torch.optim.AdamW([{"params":world,"lr":tc["core_lr"],"base_lr":tc["core_lr"]},{"params":heads,"lr":tc["head_lr"],"base_lr":tc["head_lr"]}],weight_decay=.01)
        provenance={"config_hash":stable_hash(cfg),"source_hashes":source,"counts":{"total":sum(p.numel() for p in model.parameters()),"auxiliary":sum(p.numel() for p in model.content_head.parameters()),"decoder":sum(p.numel() for p in model.head.parameters()),"frozen":0},
            "data_manifests":{s:d["manifest_sha256"] for s,d in data.items()},"pretraining_sha256":old["sha256"],"extra_observations":False}
        write_json(out/"config.json",{"matrix":cfg,"run":run,"provenance":provenance}); curves=[]; best=float("inf"); started=time.perf_counter(); losses=[]; auxes=[]
        for step in range(tc["steps"]+1):
            if step%tc["eval_every"]==0:
                val={s:evaluate(model,codec,data[s]) for s in ("crema_validation","libri_validation")}; score=float(np.mean([v["selection_score"] for v in val.values()]))
                curves.append({"step":step,"validation":val,"selection_score":score,"seconds":time.perf_counter()-started,"loss":float(np.mean(losses)) if losses else None,"auxiliary_mse":float(np.mean(auxes)) if auxes else None}); losses=[]; auxes=[]
                write_json(out/"curves.json",curves)
                if score<best:
                    best=score; torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"config":cfg,"run":run,"provenance":provenance,"torch_rng":torch.get_rng_state(),"numpy_rng":rng.bit_generator.state},out/"best.pt")
                print(json.dumps({"run":run["id"],"step":step,"selection":score,"seconds":time.perf_counter()-started}),flush=True)
            if step==tc["steps"]: break
            ci=rng.integers(0,len(c["rows"]),nc); li=rng.integers(0,len(l["rows"]),batch-nc)
            cs=np.array([rng.integers(0,int(c["lengths"][i])-31) for i in ci]); ls=np.array([rng.integers(0,int(l["lengths"][i])-31) for i in li])
            first=rng.integers(2,12,batch); second=rng.integers(18,27,batch); gain=rng.uniform(-12,12,batch).astype(np.float32); eq=rng.uniform(-6,6,(batch,4)).astype(np.float32)
            keep=rng.random(batch)<.5; shifts=rng.choice([-9,-6,-3,3,6,9],nc); use_wrong=rng.random(nc)<.5
            cy,cv=crop(c,ci,cs); ly,lv=crop(l,li,ls); y=torch.cat([cy,ly]); video=torch.cat([cv,lv]); g=torch.from_numpy(gain).cuda(); q=torch.from_numpy(eq).cuda()
            y=codec.augment(y,g,q); visible=holes(first,second,y.device); obs=codec.observe(y,visible)
            vm=torch.as_tensor(np.concatenate([c["video_present"][ci],l["video_present"][li]])&keep,device=y.device)[:,None].expand_as(visible)
            model.train(); optimizer.zero_grad(set_to_none=True); factor=min(1.,(step+1)/tc["warmup_steps"])*(.1+.9*.5*(1+math.cos(math.pi*step/tc["steps"])))
            for group in optimizer.param_groups: group["lr"]=group["base_lr"]*factor
            with torch.autocast("cuda",dtype=torch.bfloat16):
                pred=output(model,obs,video,vm); auxiliary=content_loss(model.content_prediction,obs["target"],visible); loss=restoration_loss(pred,obs["target"],visible)+run["aux_weight"]*auxiliary
                if step%tc["alignment_every"]==0:
                    eligible=np.where(c["lengths"][ci]>=50)[0]
                    if len(eligible)>=2:
                        ai=ci[eligible]; center=(c["lengths"][ai]-32)//2; ay,av=crop(c,ai,center); ay=codec.augment(ay,g[eligible],q[eligible])
                        a=codec.observe(ay,torch.ones(ay.shape[:2],device=y.device,dtype=torch.bool))["audio"]
                        _,shifted=crop(c,ai,center+shifts[eligible]); wi=c["wrong"][ai]; _,wrong=crop(c,wi,(c["lengths"][wi]-32)//2)
                        negative=torch.where(torch.as_tensor(use_wrong[eligible],device=y.device)[:,None,None],wrong,shifted)
                        align,_=alignment_loss(model.world,a,av,negative,[c["rows"][i]["sentence"] for i in ai]); loss=loss+tc["alignment_weight"]*align
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite content objective")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); losses.append(float(loss.detach())); auxes.append(float(auxiliary.detach()))
        checkpoint=torch.load(out/"best.pt",map_location="cpu",weights_only=False)
        torch.save({"model":model.state_dict(),"step":tc["steps"],"config":cfg,"run":run,"provenance":provenance},out/"last.pt")
        write_json(out/"training_complete.json",{"run":run,"selected_step":checkpoint["step"],"best_validation_score":best,"wall_seconds":time.perf_counter()-started,
            "checkpoint_sha256":sha256(out/"best.pt"),"last_checkpoint_sha256":sha256(out/"last.pt"),"provenance":provenance,"windows_seen":batch*tc["steps"]})
        del model,optimizer; torch.cuda.empty_cache()


if __name__=="__main__": main()
