"""Combine checked, non-media G4 evidence from accelerator and Fedora exports for publication."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def put(path,record):
    path.write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+"\n")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); source=a.input; cloud=source/"cloud"; confirmed=source/"confirm"; a.out.mkdir(parents=True,exist_ok=True)
    data=read(source/"training-data-card.json")
    benches={"v5e_pilot":read(cloud/"batch-tpu.json"),"v5e_anchor":read(cloud/"batch-anchor.json"),
             "3090_pilot":read(source/"batch-3090.json")}
    if (source/"batch-3090-anchor.json").exists(): benches["3090_anchor"]=read(source/"batch-3090-anchor.json")
    put(a.out/"throughput.json",benches)
    rows={}
    for file in sorted(cloud.glob("g4-*/complete.json")):
        run=file.parent; tag=run.name
        cfg=read(run/"config.json"); done=read(file); dev=read(run/"dev.json")
        assert cfg["name"]==tag and done["weight_file_sha256"]==dev["weight_sha256"] and done["step"]==cfg["steps"]
        row={"config":cfg,"weight_sha256":done["weight_file_sha256"],"timing":done["timing"],
             "training_examples":cfg["steps"]*cfg["batch"],"validation_examples_per_second":dev["warm_conditioned_examples_s"],
             "development":dev["scores"]}
        for root in (cloud,confirmed):
            loc=root/tag
            for name,key in (("test.json","confirmation"),("transfer-test.json","frozen_transfer")):
                if not (loc/name).exists():continue
                x=read(loc/name); assert x["weight_sha256"]==row["weight_sha256"]
                row[key]=x["scores"] if name=="test.json" else x["tasks"]
            if (loc/"live/live.json").exists():
                doc=read(loc/"live/live.json")
                row["live"]={k:doc[k] for k in ("ticks","seconds","block_ticks","warm_block_wall_s_median","conditions_batched","scores","generated_sha256")}
                if (loc/"live/media.json").exists(): row["live"]["rendered"]=read(loc/"live/media.json")
            if (loc/"action-history-audit.json").exists(): row["action_history_audit"]=read(loc/"action-history-audit.json")
        rows[tag]=row
    if len(rows)!=40 or sum("confirmation" in v for v in rows.values())!=16:
        raise ValueError("Expected forty trained endpoints and sixteen registered broad confirmations")
    put(a.out/"data_and_endpoints.json",{"data":data,"runs":{tag:{k:v for k,v in val.items() if k not in ("development","confirmation","frozen_transfer","live","action_history_audit")} for tag,val in rows.items()}})
    put(a.out/"development.json",{tag:v["development"] for tag,v in rows.items()})
    put(a.out/"confirmation.json",{tag:{k:v[k] for k in ("confirmation","frozen_transfer","live","action_history_audit") if k in v} for tag,v in rows.items() if "confirmation" in v})
    for name in ("exposure-counts.json","new-speaker-summary.json","motion-confirmation-summary.json","speech-content.json","final-artifact-audit.json"):
        if (source/name).exists(): put(a.out/name,read(source/name))
    print("validated",len(rows),"endpoints",flush=True)


if __name__=="__main__": main()
