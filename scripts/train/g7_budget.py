"""Meter this study's named native Colab endpoints, independently of other account activity."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from g5_colab_budget import usage


def parse_listing(text):
    return {m[2]:(m[1],m[3].upper()) for m in re.finditer(r"^\[([^\]]+)\]\s+(\S+)\s+\|\s+Hardware:\s*([A-Za-z0-9_]+)",text,re.M)}


def save(path,doc):
    tmp=path.with_suffix(".own-new");tmp.write_text(json.dumps(doc,sort_keys=True)+"\n");tmp.replace(path)


def update(ledger,now,observed,own,prices,last):
    for endpoint,entry in ledger.items():
        if entry["active"]:
            seconds=max(0.,now-last)
            entry["seconds"]+=seconds
            entry["units"]+=entry["rate"]*seconds/3600
        if observed is not None:
            entry["active"]=endpoint in observed
            if entry["active"]: entry["rate"]=prices[entry["hardware"]]
            elif not entry.get("ended"): entry["ended"]=now
    if observed is not None:
        for endpoint,(name,kind) in observed.items():
            if name in own and endpoint not in ledger:
                if kind not in prices: raise ValueError("Unexpected study hardware "+kind)
                seconds=max(0.,now-last)
                ledger[endpoint]={"name":name,"hardware":kind,"started_bound":last,"active":True,"ended":None,"seconds":seconds,
                                  "units":seconds*prices[kind]/3600,"rate":prices[kind]}
    return sum(v["units"] for v in ledger.values())


def main(folder):
    cfg=json.loads((folder/"bounds.json").read_bytes())
    oldfile=folder/"budget.json"; prior=json.loads(oldfile.read_bytes()) if oldfile.is_file() else {}
    names=set(cfg["sessions"]); prices=prior.get("generation_rates",{"V5E1":3.09,"V6E1":4.5})
    ledger=prior.get("project_endpoints",{}); previous=prior.get("observed_account_endpoints",{})
    oldrate=prior.get("rate",0.); failures=0
    last=prior.get("project_updated_unix",cfg["started"]);balance=prior.get("balance",cfg["starting_balance"])
    stop=prior.get("shutdown_reason"); cli=[str(Path.home()/".local/bin/colab"),"--auth","adc"]
    env=dict(os.environ,CLOUDSDK_CONFIG=str(Path.home()/".config/slwm-colab/gcloud"),COLAB_CLI_REQUEST_TIMEOUT="19")
    def api(*args):
        return subprocess.check_output([*cli,*args],env=env,timeout=32,text=True,stderr=subprocess.DEVNULL)
    while True:
        ok=False; seen=None; now=time.time()
        try:
            seen=parse_listing(api("sessions")); account,rate,reported=usage(api("usage"))
            if reported!=len(seen): raise ValueError("Meter and assignment list settling")
            if seen and len(set(t for _,t in seen.values()))==1:
                kind=next(iter(seen.values()))[1]
                if kind in prices and rate>0: prices[kind]=rate/len(seen)
            elif previous:
                add=set(seen)-set(previous); gone=set(previous)-set(seen)
                if len(add)==1 and not gone and rate>oldrate:
                    kind=seen[next(iter(add))][1]
                    if kind in prices: prices[kind]=rate-oldrate
            previous=seen;oldrate=rate;balance=account;failures=0;ok=True
        except (subprocess.SubprocessError,OSError,ValueError):
            failures+=1;seen=None
        now=time.time(); used=update(ledger,now,seen,names,prices,last);last=now
        active=[v for v in ledger.values() if v["active"]]
        if not stop:
            if used>=cfg["limit"]-cfg["reserve"]:stop="project credit reserve"
            elif balance<=0:stop="provider reports exhausted balance"
            elif (folder/"release").exists():stop="explicit release"
            elif now>=cfg["deadline"]-90:stop="wave wall limit"
            elif any(now-v["started_bound"]>=10.95*3600 for v in active):stop="session time limit"
            # A failed poll continues to accrue the last known owned rates. The project and wall bounds still apply.
        if stop:
            for item in active:
                try: api("stop","-s",item["name"])
                except (subprocess.SubprocessError,OSError):pass
        done=ok and not active and ((folder/"requests-done").exists() or bool(stop))
        save(oldfile,{"phase":"complete" if done else "releasing" if stop else "watching" if ok else "accounting_retry",
            "scope":"project endpoint duration times per-generation rate","project_updated_unix":now,"checked_unix":now,
            "generation_rates":prices,"project_endpoints":ledger,"project_spend":used,"credit_limit":cfg["limit"],
            "reserve":cfg["reserve"],"balance":balance,"starting_balance":cfg["starting_balance"],
            "account_spend":max(0,cfg["starting_balance"]-balance),"rate":oldrate,"observed_account_endpoints":previous,
            "active":sorted(v["name"] for v in active),"project_rate":sum(v["rate"] for v in active),
            "deadline_unix":cfg["deadline"],"shutdown_reason":stop,"unrelated_active_count":max(0,len(previous)-len(active))})
        if done:return
        time.sleep(12 if stop else 20)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("folder",type=Path);main(p.parse_args().folder)
