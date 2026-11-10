"""Resolve only registered factorial cells and verify their immutable parents, controls and data."""
import json
from pathlib import Path
import numpy as np
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha,lock

CONFIG=Path("configs/t8/factorial.json")
MANIFEST=Path("experiments/multimodal/t7/continuation-alignment-control-20261110/metrics/checkpoint-inventory.json")
G7_DATA=Path("/mnt/slwm-data/SLWM/g7")
G8_DATA=Path("/mnt/slwm-data/SLWM/g8")
ARMS=("parent","replay","sim","other","fresh")


def read(file):return json.loads(file.read_bytes())


def cases():
    return [(core,seed,arm) for core in ("hybrid","gpt2") for seed in (511,512) for arm in ARMS]


def key(core,seed,arm):return f"{core}-{seed}-{arm}"


def resolve(core,seed,arm):
    dh=roots("g8")
    if arm=="parent":study,n="g6",8192
    elif arm in ("sim","other"):study,n="g8",16384
    else:
        proof=dh[0]/"bridges"/f"{core}-{seed}.json"
        if not proof.is_file():return None
        b=read(proof)
        if (b["core"],b["seed"])!=(core,seed):raise ValueError("Incompatible bridge identity")
        identical=b["observed"]["resume.npz"]==b["expected_optimizer"] and b["observed"]["inference.npz"]==b["expected_weights"]
        if identical!=b["exact"]:raise ValueError("Unverified bridge declaration")
        expected=read(MANIFEST)["rungs"][f"g7-v5e-anchor-{core}-{seed}-{b['arm']}/step-010240"]
        if (expected["inference_sha256"],expected["optimizer_sha256"])!=(b["expected_weights"],b["expected_optimizer"]):raise ValueError("Different G7 bridge reference")
        study,n=("g7" if identical else "g8"),16384
    tag=f"{study}-v5e-anchor-{core}-{seed}"+("" if arm=="parent" else "-"+arm)
    location=Path(tag)/f"step-{n:06d}";da,db=roots(study)
    if not all((d/location/"verified.json").is_file() for d in (da,db)):return None
    a,b=(read(d/location/"verified.json") for d in (da,db))
    if a["run"]!=tag or a["step"]!=n or a["sha256"]!=b["sha256"] or len(set(a["disk_uuids"]))!=2:raise ValueError("Unmatched inference receipts")
    if study=="g7":
        pin=read(MANIFEST)["rungs"][str(location)]
        if pin["inference_sha256"]!=a["sha256"]["inference.npz"] or pin["optimizer_sha256"]!=a["sha256"]["resume.npz"]:raise ValueError("Wrong pre-existing control")
    cfg=read(da/location/"config.json")
    if cfg["params"]!=116502025 or cfg["seed"]!=seed or cfg["spec"]["core"]!=core:raise ValueError("Unmatched architecture or parent")
    return {"model":key(core,seed,arm),"run":tag,"study":study,"step":n,"root":str(da/location),
            "weights":a["sha256"]["inference.npz"],"optimizer":a["sha256"]["resume.npz"],"config":a["sha256"]["config.json"],"params":cfg["params"]}


def load_model(rec):
    import jax
    from models.g4_shared import Spec
    stage=Path(rec["root"]);file=stage/"inference.npz"
    if sha(file)!=rec["weights"]:raise ValueError("Inference bytes differ from registered receipt")
    with np.load(file,allow_pickle=False) as f:
        p={k:jax.device_put(f[k]) for k in f if k!="stats" and not k.startswith("prior_")}
        cal={k[6:]:jax.device_put(f[k]) for k in f if k.startswith("prior_")};stats=np.array(f["stats"])
    return Spec(**read(stage/"config.json")["spec"]),p,cal,stats


def measurement_files():
    return ["configs/t8/factorial.json","training/g8_scenes.py","training/g8_models.py","training/g8_pushing.py","training/g8_reacher.py",
        "training/g8_live.py","training/g8_score.py","scripts/train/g8_analysis.py"]


def freeze():
    left,right=roots("g8");records={}
    for case in cases():
        rec=resolve(*case)
        if rec is None:return None
        records[rec["model"]]=rec
    panels=G8_DATA/"panels";data={}
    for task in ("pusht","puck_barrier","reacher_test"):
        folder=panels/task;card=read(folder/"card.json")
        if task=="reacher_test":
            for f,h in card["sha256"].items():
                if sha(folder/(f+".npy"))!=h:raise ValueError("Unseen Reacher file changed")
        elif sha(folder/"scenes.npz")!=card["sha256"]:raise ValueError("Pushing scene archive changed")
        data[task]=sha(folder/"card.json")
    heads={};head_receipts={}
    for name in [*records,"raw-511","raw-512","physical"]:
        f=left/"reacher-heads"/name
        if not (f/"complete.json").is_file():return None
        d=read(f/"complete.json")
        if name in records and d["parent"]!=records[name]["weights"]:raise ValueError("Readout points at different core")
        if sha(f/"weights.npz")!=d["weights"] or sha(f/"optimizer.npz")!=d["optimizer"]:raise ValueError("Readout artifact changed")
        other=right/"reacher-heads"/name
        if not (other/"complete.json").is_file():return None
        if sha(other/"weights.npz")!=d["weights"] or sha(other/"optimizer.npz")!=d["optimizer"] or read(other/"complete.json")!=d:
            raise ValueError("Fitted readout copies disagree")
        if d.get("normalization") and any(sha(folder/"normalization.npz")!=d["normalization"] for folder in (f,other)):
            raise ValueError("Physical transfer normalization changed")
        heads[name]=d["weights"];head_receipts[name]=sha(f/"complete.json")
    audit=read(G8_DATA/"batch-audit.json")
    if not audit["factorization_exact"]:raise ValueError("Data manipulation failed")
    doc={"study":"G8","models":records,"readouts":heads,"readout_receipts":head_receipts,"panels":data,"training_audit":sha(G8_DATA/"batch-audit.json"),
         "previous_panels":{f:sha(G7_DATA/"fresh"/f/"test/card.json") for f in ("crema","libri","text","events","sim")},
         "code":{f:sha(Path(f)) for f in measurement_files()}}
    for disk in (left,right):lock(disk/"confirmation-lock.json",doc)
    return doc


def sealed():
    a,b=roots("g8");doc=read(a/"confirmation-lock.json")
    if doc!=read(b/"confirmation-lock.json") or set(doc["models"])!={key(*c) for c in cases()}:raise ValueError("Incomplete confirmation lock")
    for file,h in doc["code"].items():
        if sha(Path(file))!=h:raise ValueError("Locked scoring code changed: "+file)
    return doc
