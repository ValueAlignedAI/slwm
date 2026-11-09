"""Supervise a bounded accelerator study and persist progress while child kernels compile or run."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import threading
import time

from training.g4_train import CFG, Data, archive, parser as train_parser, read, run as train, sha_file, write
from training.g4_eval import evaluate


@contextmanager
def monitor(status,progress,description,deadline):
    end=threading.Event()
    def watch():
        while not end.is_set():
            current=read(progress) if progress and progress.exists() else None
            write(status,{"stage":description,"progress":current,"checked_unix":time.time(),"training_deadline_unix":deadline})
            end.wait(15)
    thread=threading.Thread(target=watch,daemon=True); thread.start()
    failure=None
    try:
        yield
    except BaseException as e:
        failure=str(e)[:400]
        raise
    finally:
        end.set(); thread.join()
        if failure: write(status,{"stage":description,"failed":failure,"time":time.time()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--bench",type=Path,required=True); p.add_argument("--backend",choices=["gpu","tpu"],required=True)
    p.add_argument("--label",required=True); p.add_argument("--spool",type=Path,required=True)
    p.add_argument("--deadline",type=float,required=True); p.add_argument("--stage",choices=["pilot","medium","anchor"],default="pilot")
    p.add_argument("--one-pair",action="store_true")
    a=p.parse_args(); cfg=read(CFG); a.out.mkdir(parents=True,exist_ok=True)
    status=a.out/f"stages-{a.label}-{a.stage}.json"
    while not a.bench.exists() or "selected_micro" not in read(a.bench):
        if time.time()+120>=a.deadline:
            write(status,{"stage":"capacity_window_ended_before_benchmark","time":time.time()}); return
        write(status,{"stage":"batch_sweep","checked_unix":time.time()})
        time.sleep(8)
    bench=read(a.bench)
    if bench["platform"]!=a.backend or bench["effective_batch"]!=cfg["effective_batch"]:
        raise ValueError("Batch selection has the wrong platform or exposure")
    micro=bench["selected_micro"]
    steps=cfg[{"pilot":"screen_updates","medium":"medium_updates","anchor":"anchor_updates"}[a.stage]]
    if a.stage!="pilot":
        micro=min(micro,128 if a.stage=="medium" else 64)
    seeds=cfg["seeds"] if not a.one_pair else cfg["seeds"][:1]
    order=[(s,m) for s in seeds for m in (["gpt2-cross","signal-cross"] if s%2 else ["signal-cross","gpt2-cross"])]
    if not a.one_pair:
        for m in ("signal-self","gpt2-self","sparse-cross","gpt2-matched"):
            order.extend((s,m) for s in seeds)
    rate=next(r["examples_s"] for r in bench["rows"] if r["micro"]==bench["selected_micro"])
    estimate=steps*cfg["effective_batch"]/rate
    with monitor(status,None,"verify_and_load_device_arrays",a.deadline):
        train_data=Data(a.data,"train")
        dev_data=Data(a.data,"dev")
        stats=train_data.stats()
    done=[]
    for seed,profile in order:
        tag=f"g4-{a.label}-{a.stage}-{profile}-{seed}"
        run=a.out/tag; run.mkdir(exist_ok=True)
        if not (run/"complete.json").exists():
            if time.time()+max(90,1.4*estimate+40)>a.deadline:
                write(status,{"stage":"reserved_time_for_uploads","pending":tag,"completed":done,"time":time.time()})
                return
            start=time.monotonic()
            cmd=["--data",str(a.data),"--output",str(a.out),
                 "--scale",a.stage,"--profile",profile,"--label",a.label,"--seed",str(seed),"--backend",a.backend,
                 "--steps",str(steps),"--batch",str(cfg["effective_batch"]),"--micro",str(micro),
                 "--scan",str(cfg["scan_steps"]),"--save-every",str(min(256,steps)),"--spool",str(a.spool),"--checked-data"]
            with monitor(status,run/"progress.json",tag+"/train",a.deadline):
                train(train_parser().parse_args(cmd),train_data,stats)
            estimate=max(20,time.monotonic()-start)
        if not (run/"dev.json").exists():
            with monitor(status,None,tag+"/dev",a.deadline):
                evaluate(run,a.data,"dev",min(micro,128),a.backend,data_cache=dev_data)
        archive(tag+"-evidence",steps,{f:run/f for f in ("complete.json","dev.json","dev-paired.npz")},a.spool)
        done.append(tag)
    write(status,{"stage":"complete","completed":done,"finished_unix":time.time(),"selected_micro":micro,
                  "benchmark_sha256":sha_file(a.bench)})


if __name__=="__main__": main()
