"""Assemble registered outcomes and artifacts after the complete prospective evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.g8_models import read,CONFIG,G8_DATA,sealed
from training.g7_units import put
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha

EXPECTED={"models":20,"pushing_results":42,"reacher_results":23,"live_controllers":48,"retention":20,"initial_speech_panels":10,"rate_speech_panels":10}


def progress():
    a,b=roots("g8");c=a/"confirmation";p=a/"public";audit=read(a/"archive-audit.json") if (a/"archive-audit.json").exists() else {}
    return {"models":len(read(a/"confirmation-lock.json")["models"]) if (a/"confirmation-lock.json").is_file() else 0,
            "pushing_results":sum(len(list((c/task).glob("*/metrics.json"))) for task in ("pusht","puck_barrier")),
            "reacher_results":len(list((c/"reacher").glob("*/metrics.json"))),"live_controllers":len(list((a/"live").glob("*/*/metrics.json"))),
            "retention":len(list((a/"retention").glob("*/metrics.json"))),"initial_speech_panels":len(list((a/"speech-audit").glob("*/*/scores.json"))),
            "rate_speech_panels":len(list((a/"rate-audit").glob("*/*/scores.json"))),"archive_complete":audit.get("complete",False),
            "durable_rungs":audit.get("verified_rungs",0),"analysis":(p/"factorial.json").is_file(),"replays":(p/"rollout-examples.json").is_file()}


def development_curves(doc):
    from scripts.train.g8_analysis import CONTRASTS,effect,estimate
    results={}
    for core in ("hybrid","gpt2"):
        for seed in (511,512):
            visits={}
            for step in (10240,12288,16384):
                cells={};ids=None
                for arm in ("replay","sim","other","fresh"):
                    rec=doc["models"][f"{core}-{seed}-{arm}"]
                    stage=Path(rec["root"]).parent/f"step-{step:06d}"
                    proof=read(stage/"verified.json");file=stage/"dev-predictions.npz"
                    if sha(file)!=proof["sha256"]["dev-predictions.npz"]:raise ValueError("Changed development predictions")
                    with np.load(file) as f:
                        if ids is not None:np.testing.assert_array_equal(ids,f["sim_group"])
                        ids=np.array(f["sim_group"]);cells[arm]=np.array(f["push_joint"],np.float64)
                draw=np.random.default_rng(810251).integers(len(ids),size=(4000,len(ids)),dtype=np.int32)
                visits[str(step)]={"original_development_starts":len(ids),"endpoint_px":{k:float(x.mean()) for k,x in cells.items()},
                                   "effects_px":{k:estimate(effect(cells,CONTRASTS[k]),draw) for k in ("other_main","sim_main","interaction")}}
            results[f"{core}-{seed}"]=visits
    return results

def assemble():
    a,b=roots("g8");pub=a/"public";m=progress()
    if any(m[k]!=n for k,n in EXPECTED.items()) or not all(m[k] for k in ("archive_complete","analysis","replays")):raise ValueError(f"Pending G8 evidence: {m}")
    doc=sealed();archive=read(a/"archive-audit.json")
    if archive!=read(b/"archive-audit.json"):raise ValueError("Independent readback differs")
    payload=read(pub/"factorial.json")
    if payload["lock_sha256"]!=sha(a/"confirmation-lock.json") or payload["row_file_sha256"]!=sha(pub/"paired-scenes.npz"):raise ValueError("Statistical result hash mismatch")
    tim={};raw_seconds=0.
    for tag,r in archive["rungs"].items():
        stage=a/tag;cfg=read(stage/"config.json");p=read(stage/"progress.json");wall=read(stage/"wall.json")
        if r["inference_sha256"]!=read(stage/"verified.json")["sha256"]["inference.npz"] or cfg["batch"]!=256 or cfg["params"]!=116502025:raise ValueError("Wrong registered generation")
        tim[tag]={"update_seconds_cumulative":p["continuation_update_seconds"],"examples_per_second":p["warm_examples_per_second"],
              "training_invocation_s":wall["training_invocation_s"],"original_development_eval_s":wall["evaluation_invocation_s"],"params":cfg["params"],
              "parent":cfg["parent"],"parent_optimizer":cfg["parent_optimizer"],"data_identity":cfg["data"],"families":cfg["fresh_families"],
              "original_dev":read(stage/"dev-metrics.json")["push"]["joint"]}
        raw_seconds+=wall["training_invocation_s"]
    heads={}
    for name,h in doc["readouts"].items():
        folder=a/"reacher-heads"/name;dest=b/"reacher-heads"/name;meta=read(folder/"complete.json")
        if sha(dest/"weights.npz")!=h or sha(dest/"optimizer.npz")!=meta["optimizer"]:raise ValueError("Unmirrored transfer artifact")
        heads[name]=meta
    source={"study":"G8","complete":True,"status":m,"factorial_gate":payload["primary_gate"],"hardware":"v5e-1","models":doc["models"],
            "live_control_input":"16 observed ticks and an exogenous goal-directed three-proposal bank per decision",
            "readout_heads":heads,"confirmation_lock":sha(a/"confirmation-lock.json"),"held_panels":doc["panels"],"training_audit":doc["training_audit"],
            "rungs":archive["rungs"],"inference_and_optimizer_bytes_per_disk":archive["optimizer_and_inference_bytes_per_disk"],
            "registered_development_learning_curves":development_curves(doc),"registered_training_seconds_excluding_bridges_and_transfer":raw_seconds,"per_rung_timing":tim,"development_smoke":read(G8_DATA/"development-smoke/verified.json")}
    put(pub/"artifact-inventory.json",source)
    for name in ("factorial.json","raw-and-live.json","retention.json","paired-scenes.npz","artifact-inventory.json","rollout-examples.json","decision-effects.png"):
        if not (pub/name).is_file():raise ValueError("Missing public evidence: "+name)
    add=["examples/pusht.webm","examples/puck_barrier.webm"]
    paths=["factorial.json","raw-and-live.json","retention.json","paired-scenes.npz","artifact-inventory.json","rollout-examples.json","decision-effects.png",*add]
    receipt={"complete":True,"gates":m,"public_sha256":{k:sha(pub/k) for k in paths},"inference_example_seconds":read(pub/"rollout-examples.json")["puck_barrier"]["duration_seconds"]}
    put(pub/"release-ready.json",receipt);print(json.dumps(m),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=("status","assemble"));a=p.parse_args()
    if a.action=="status":print(json.dumps(progress()))
    else:assemble()
