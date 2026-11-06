"""Paced ongoing observation/proposal cycle with frozen weights and stale input rejection."""
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import tiktoken
import torch

from training.general_batch import tensor_tree
from training.general_data import order
from training.general_evaluation_lock import register
from training.general_frozen_features import supplied_text
from training.general_readout_data import ReadoutData
from training.general_readout_train import Readout
from training.general_sources import write
from training.general_train import initialize
from training.research_sources import file_sha


def state_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        array=value.detach().cpu().contiguous().numpy();digest.update(name.encode());digest.update(str(array.dtype).encode());digest.update(array.tobytes())
    return digest.hexdigest()


class DecisionReadout:
    """Frozen last shared state -> trained physical command proposal; no optimizer."""
    def __init__(self,cfg,run):
        self.cfg=cfg;root=Path(cfg["root"]);self.encoder=tiktoken.get_encoding("gpt2")
        ck=torch.load(root/run["id"]/"endpoint.pt",map_location="cpu",weights_only=False,mmap=True)
        self.model=initialize(cfg,run).cuda().eval().requires_grad_(False);self.model.load_state_dict(ck["model"])
        self.heads=[]
        for seed in (201,202):
            folder=root/"readouts/heads/droid/actor"/run["id"]/str(seed)
            saved=torch.load(folder/"selected.pt",map_location="cpu",weights_only=False)
            head=Readout(saved["inputs"],saved["outputs"],saved["hidden"]).cuda().eval().requires_grad_(False);head.load_state_dict(saved["model"])
            arrays={name:torch.from_numpy(value).cuda().float() for name,value in dict(np.load(folder/"projection.npz")).items()}
            self.heads.append((head,arrays))

    def propose(self,batch):
        obs={**batch["obs"],**supplied_text(batch["captions"],self.encoder)};obs=tensor_tree(obs,"cuda")
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):state=self.model(obs)
        # Feature caches use float16 storage; preserve the same quantization at
        # runtime. Small projections/readouts run in float32 as during fitting.
        values=state["last"].half().float();common=obs["numeric"][:,-1,33:50];predictions=[]
        with torch.inference_mode():
            for head,p in self.heads:
                feature=((values-p["mean"])/p["scale"])@p["basis"]
                extra=(common-p["common_mean"])/p["common_scale"]
                value=head(torch.cat([feature,extra],1))*p["target_scale"]+p["target_mean"]
                predictions.append(value)
            result=torch.stack(predictions).mean(0).cpu().numpy()
        return result


def run_cycle(cfg,run,root):
    destination=root/"operational"/run["id"];destination.mkdir(parents=True,exist_ok=True)
    if (destination/"summary.json").exists():return
    steps=401;period=.1;data=ReadoutData(cfg,"droid","test",True)
    eligible=[r for r in data.rows.values() if r["steps"]>=steps+cfg["context"]+cfg["horizon"]]
    if not eligible:
        write(destination/"summary.json",{"skipped":"No held-out source episode supports the fixed long cycle","no_substitution_by_performance":True});return
    row=min(eligible,key=lambda r:order(r["id"]+"operational40s"))
    data.cases=[{"id":row["id"],"group":row["group"],"start":i} for i in range(steps)]
    records=[]
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);policy=DecisionReadout(cfg,run)
        before=state_digest(policy.model);head_before=[state_digest(h) for h,_ in policy.heads]
        cold_start=time.perf_counter();policy.propose(data.batch([0]));cold=time.perf_counter()-cold_start
        started=time.monotonic();tick=0
        while tick<steps:
            target=started+tick*period;delay=target-time.monotonic()
            if delay>0:time.sleep(delay)
            latest=int((time.monotonic()-started)/period);skipped=max(0,latest-tick);tick=max(tick,latest)
            if tick>=steps:break
            begin=time.perf_counter();batch=data.batch([tick]);current=tick+cfg["context"]-1
            source_steps=np.arange(tick,tick+cfg["context"])
            stale=(source_steps%50>=20)&(source_steps%50<25)
            # Simulate packets arriving five consumer ticks late. Their stale
            # content is erased before the model sees a claimed current frame.
            batch["obs"]["image_present"][:,stale]=False;batch["obs"]["images"][:,stale]=0
            proposal=policy.propose(batch)[0];emitted=time.monotonic();latency=time.perf_counter()-begin
            records.append({"tick":tick,"source_ordinal_step":current,"seconds_since_cycle_start":emitted-started,"latency_seconds":latency,
                            "deadline_missed":emitted>started+(tick+1)*period,"skipped_policy_ticks":skipped,
                            "latest_image_rejected":bool(stale[-1]),"injected_image_age_consumer_seconds":.5 if stale[-1] else 0.,
                            "translation_body_m":proposal[:3].tolist(),"rotation_vector_body_rad":proposal[3:6].tolist(),"gripper":float(proposal[6]),
                            "proposal_finite":bool(np.isfinite(proposal).all()),"gripper_in_unit_interval":bool(0<=proposal[6]<=1),
                            "reference_command_scoring_only":batch["target"][0].tolist(),"status":"internal_proposal_no_actuator"})
            tick+=1
        elapsed=time.monotonic()-started;after=state_digest(policy.model);head_after=[state_digest(h) for h,_ in policy.heads]
        assert before==after and head_before==head_after,"Inference changed trained weights"
        write(destination/"trace.json",records)
        times=np.array([r["latency_seconds"] for r in records])
        write(destination/"summary.json",{"run":run["id"],"source_id":row["id"],"source_group":row["group"],"cycle_seconds":elapsed,"consumer_period_seconds":period,
              "policy_ticks":len(records),"cold_inference_seconds":cold,"warm_latency_median_seconds":float(np.median(times)),"warm_latency_p95_seconds":float(np.quantile(times,.95)),
              "missed_deadlines":sum(r["deadline_missed"] for r in records),"skipped_policy_ticks":sum(r["skipped_policy_ticks"] for r in records),
              "stale_latest_images_rejected":sum(r["latest_image_rejected"] for r in records),"core_hash_before":before,"core_hash_after":after,"head_hashes_before":head_before,"head_hashes_after":head_after,
              "inference_weight_updates":0,"source_clock":"Ordinal RLDS steps;10Hz is a chosen receiver cadence, not verified recording frequency",
              "feedback":"Recorded observations and previous demonstrator commands; proposals do not alter future inputs",
              "scope":"Over40s ongoing paced replay with a rolling observation buffer; no physical closed-loop control or raw-camera transport measurement"})
        del policy;gc.collect();torch.cuda.empty_cache()


def main():
    torch.set_num_threads(4);cfg=json.loads(Path("configs/t2/general_transfer.json").read_text());root=Path(cfg["root"]);register(cfg);assert (root/"evaluation_lock.json").exists()
    for run in cfg["runs"]:
        if run["id"] in {"G1-101","G1-102","G1-103","G1-104"}:run_cycle(cfg,run,root)


if __name__=="__main__":main()
