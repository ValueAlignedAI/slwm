"""Independently read every finished model and optimizer back from both physical drives as transfers finish."""
from concurrent.futures import ThreadPoolExecutor
import json
import time
from scripts.train.g6_archive_audit import check_disk
from scripts.train.g6_artifacts import roots,UUIDS
from scripts.train.g7_scale_lock import cases


def main():
    a,b=roots("g7");path=a/"archive-audit.json"
    items=json.loads(path.read_bytes())["rungs"] if path.exists() else {}
    expected=[f"{tag}/step-{step:06d}" for tag,terminal in cases() if terminal==16384 for step in (10240,12288,16384)]
    until=time.time()+12*3600
    while len(items)<len(expected) and time.time()<until:
        pending=[p for p in expected if p not in items and all((d/p/"verified.json").is_file() for d in (a,b))]
        if not pending:
            time.sleep(24);continue
        with ThreadPoolExecutor(max_workers=2) as pool:
            left,right=[f.result() for f in [pool.submit(check_disk,d,pending) for d in (a,b)]]
        if left!=right:raise ValueError("Independent physical readback differs")
        items.update(left["milestones"])
        if len(set(UUIDS))!=2:raise ValueError("The two disk identifiers coincide")
        doc={"rungs":items,"complete":len(items)==len(expected),"expected":48,"physical_disk_uuids":list(UUIDS),
             "optimizer_and_inference_bytes_per_disk":sum(v["optimizer_bytes"]+v["inference_bytes"] for v in items.values()),
             "verified_rungs":len(items),"checked_unix":time.time()}
        for dest in (a,b):
            tmp=dest/"archive-audit.partial";tmp.write_text(json.dumps(doc,sort_keys=True,indent=2)+"\n");tmp.replace(dest/"archive-audit.json")
        print("Verified independent disk reads",len(items),"of 48",flush=True)


if __name__=="__main__":main()
