"""Manage one project credit budget across named Colab assignments. No assignment is created here."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def usage(text):
    def value(label):
        match=re.search(rf"^{label}:\s*([\d,.]+)",text,re.M)
        if not match: raise ValueError(f"Missing Colab accounting field {label}")
        return float(match.group(1).replace(",",""))
    return value("Current balance"),value("Usage rate"),int(value("Active assignments"))


def named(text,names):
    return {m[1]:m[2] for m in re.finditer(r"^\[([^\]]+)\]\s+(\S+)",text,re.M) if m[1] in names}


def guard(a):
    cfg=Path.home()/".config/slwm-colab"
    env=dict(os.environ,CLOUDSDK_CONFIG=str(cfg/"gcloud"),COLAB_CLI_REQUEST_TIMEOUT="18")
    cli=[str(Path.home()/".local/bin/colab"),"--auth","adc"]
    def call(*args):
        return subprocess.check_output([*cli,*args],env=env,stderr=subprocess.DEVNULL,text=True,timeout=32)
    def save(row):
        a.state.parent.mkdir(parents=True,exist_ok=True)
        tmp=a.state.with_suffix(".new"); tmp.write_text(json.dumps(row,sort_keys=True)+"\n"); tmp.replace(a.state)
    prior=json.loads(a.state.read_bytes()) if a.state.exists() else {}
    seen=dict(prior.get("seen",{})); reserved=set(prior.get("released",[])); wall=a.started+a.hours*3600
    last_good=float(prior.get("accounted_unix",a.started)); old_rate=float(prior.get("rate",0))
    balance=float(prior.get("balance",a.balance)); failures=0
    while True:
        now=time.time(); stop=now>=wall-120 or (a.state.parent/"release").exists()
        active={}; unknown=False
        try:
            active=named(call("sessions"),a.sessions)
            for key,endpoint in active.items():
                if key in seen and seen[key]!=endpoint: raise RuntimeError(f"Assignment identity changed for {key}")
                seen[key]=endpoint
            current,rate,count=usage(call("usage"))
            balance,old_rate,last_good,failures=current,rate,time.time(),0
            reserved.update(set(seen)-set(active))
            stop |= a.balance-current+max(a.reserve,rate*.2)>=a.credits
        except (ValueError,subprocess.CalledProcessError,subprocess.TimeoutExpired):
            failures+=1; unknown=True
            stop |= failures>=3 or now-last_good>160
        # The second bound covers a temporarily stale displayed balance at the observed rate.
        stop |= a.balance-balance+(max(0,now-last_good)/3600)*max(old_rate,4.08*len(seen))+a.reserve>=a.credits
        if stop:
            for key in (sorted(set(seen)-reserved) if unknown else active):
                try: call("stop","-s",key)
                except (subprocess.SubprocessError,OSError): pass
        done=(not unknown and not active and ((a.state.parent/"requests-done").exists() or now>wall))
        phase="complete" if done else ("releasing" if stop else ("accounting_retry" if unknown else "watching"))
        save({"phase":phase,"seen":seen,"active":sorted(active),"released":sorted(reserved),"checked_unix":time.time(),
              "accounted_unix":last_good,"starting_balance":a.balance,"balance":balance,"rate":old_rate,
              "account_spend":max(0,a.balance-balance),"credit_limit":a.credits,"reserve":a.reserve,"deadline_unix":wall})
        if done: return
        time.sleep(12 if stop else 25)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state",type=Path,required=True); p.add_argument("--sessions",nargs="+",required=True)
    p.add_argument("--balance",type=float,required=True); p.add_argument("--credits",type=float,required=True)
    p.add_argument("--started",type=float,required=True); p.add_argument("--hours",type=float,default=10)
    p.add_argument("--reserve",type=float,default=4)
    a=p.parse_args()
    if len(set(a.sessions))!=len(a.sessions) or not 0<a.hours<=12 or not 0<a.reserve<a.credits<=a.balance:
        p.error("Invalid assignment or credit bounds")
    guard(a)
