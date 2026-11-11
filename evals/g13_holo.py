"""Measure natural speech transfer against raw, earlier-utterance, silent and frequency controls."""
import json
from pathlib import Path
import numpy as np
from evals.g13_stats import Groups
from scripts.train.g9_storage import write
from scripts.train.g13_holo import CORES,SEEDS,nll

def main():
    root=Path("/mnt/slwm-data/SLWM/g13");work=root/"work/holo";out=root/"release/metrics";out.mkdir(exist_ok=True,parents=True)
    doc={"data":json.loads((work/"complete.json").read_bytes()),"runs":{},"controls":{}}
    compact={}
    with np.load(work/"predictions.npz") as z:
        for fold in "AB":
            y=z[fold+"_y"];groups=z[fold+"_groups"];fam=z[fold+"_families"];bs=Groups(groups,fam)
            def metric(p):
                hits=p.argmax(-1)==y
                return {"ordered":hits.mean(1),"exact":hits.all(1).astype(float),"nll":nll(p,y)}
            def gains(a,b):return {k:bs.score((b[k]-v) if k=="nll" else (v-b[k])) for k,v in a.items()}
            p0=metric(z[fold+"_prior"]);doc["controls"][fold]={k:bs.score(v) for k,v in p0.items()}
            compact[fold+"_y"]=y;compact[fold+"_recordings"]=groups;compact[fold+"_families"]=fam
            for seed in SEEDS:
                for c in CORES:
                    tag=f"{fold}-{c}-{seed}";entry={}
                    for arm in ("parent","shrink","small","expanded"):
                        m={}
                        for mode in ("correct","wrong","none"):
                            pr=z[tag+"_"+arm+"_"+mode];m[mode]=metric(pr)
                            compact[tag+"_"+arm+"_"+mode+"_class"]=pr.argmax(-1).astype("i2")
                            compact[tag+"_"+arm+"_"+mode+"_nll"]=m[mode]["nll"].astype("f4")
                        entry[arm]=dict(absolute={k:bs.score(v) for k,v in m["correct"].items()},vs_wrong=gains(m["correct"],m["wrong"]),
                            vs_silence=gains(m["correct"],m["none"]),vs_prior=gains(m["correct"],p0))
                        for old in ("parent","small"):
                            if old!=arm:entry[arm]["vs_"+old]=gains(m["correct"],metric(z[tag+"_"+old+"_correct"]))
                    doc["runs"][tag]=entry
    np.savez_compressed(out/"holo-predictions.npz",**compact);write(out/"holo.json",doc)
    for f in "AB":
        cells=[v for k,v in doc["runs"].items() if k.startswith(f)]
        print(f,doc["controls"][f]["ordered"]["value"],{a:round(np.mean([x[a]["absolute"]["ordered"]["value"] for x in cells]),3) for a in ("parent","shrink","small","expanded")},
              "better NLL than parent",sum(x["expanded"]["vs_parent"]["nll"]["lo"]>0 for x in cells),"of",len(cells),flush=True)


if __name__=="__main__":main()
