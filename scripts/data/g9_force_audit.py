"""Quantify the ±20 N range used by the contact bank and restore uncapped observed force targets."""
import csv
import io
import json
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from g9_contact import RAW,fetch

def main(folder):
    meta=json.loads((folder/"roma.json").read_text());doc={}
    for split in ("cal","held"):
        with np.load(folder/f"roma-{split}.npz") as db:
            names=list(map(str,db["names"]));lengths=db["lengths"];stored=db["n"][:,:3]*10
        def trial(name):
            text=fetch(RAW+urllib.parse.quote(name)+"/force/data.csv").decode()
            rows=list(csv.DictReader(io.StringIO(text)))
            ts=np.asarray([float(x["time"]) for x in rows]);xyz=np.asarray([[float(x[k]) for k in ("fx","fy","fz")] for x in rows],np.float32)
            note=meta["trials"][name];tick=note["clock_start"]+.1*np.arange(note["ticks"])
            return xyz[np.searchsorted(ts,tick,side="right")-1]
        with ThreadPoolExecutor(4) as pool:values=np.concatenate(list(pool.map(trial,names)))
        np.testing.assert_allclose(np.clip(values,-20,20),stored,atol=3e-5)
        clipped=np.abs(values)>20
        doc[split]={"axes_clipped_fraction":float(clipped.mean()),"ticks_with_any_clipping":float(clipped.any(1).mean()),
              "p99_absolute_newtons":float(np.quantile(np.abs(values),.99)),"max_absolute_newtons":float(np.abs(values).max()),
              "ticks":len(values),"trials":len(names)}
        np.savez_compressed(folder/f"force-{split}-uncapped.npz",names=np.array(names),lengths=lengths,force=values)
    (folder/"force-range.json").write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n")


if __name__=="__main__":main(Path(sys.argv[1]))
