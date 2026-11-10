"""Audit independent Drive evidence for portable models and full optimizer checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def listing(path):
    raw=subprocess.check_output(["rclone","lsjson","--recursive","--fast-list","--files-only","--hash",
                                 "--hash-type","SHA-256",path])
    return {row["Path"]:{"size":row["Size"],"sha":next((v for k,v in row.get("Hashes",{}).items()
             if k.lower().replace("-","")=="sha256"),None)} for row in json.loads(raw)}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--catalog",type=Path,required=True)
    p.add_argument("--handoff",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--drive",default="slwm-drive:SLWM"); a=p.parse_args()
    expected=json.loads(a.catalog.read_bytes())
    archives=listing(a.drive+"/g4/weights"); objects=listing(a.drive+"/v1/objects"); full=listing(a.drive+"/g4/optimizer")
    groups={}
    for name,info in archives.items():
        if not name.endswith("-inference.json"): continue
        raw=subprocess.check_output(["rclone","cat",a.drive+"/g4/weights/"+name])
        if hashlib.sha256(raw).hexdigest()!=info["sha"]: raise ValueError("Group marker checksum mismatch")
        card=json.loads(raw)
        file=card["object"].rsplit("/",1)[-1]
        if archives.get(file)!={"size":card["bytes"],"sha":card["sha256"]}: raise ValueError("Group archive checksum mismatch")
        for run in card["runs"]: groups[run]=card["group"]
    rows={}
    for name,digest in expected.items():
        where=[]
        if objects.get(digest,{}).get("sha")==digest: where.append("drive-vault")
        if full.get(name+"/resume.npz",{}).get("sha")==digest: where.append("drive-optimizer")
        local=a.handoff/name/"resume.npz"
        if local.is_file():
            with local.open("rb") as f:
                if hashlib.file_digest(f,"sha256").hexdigest()==digest: where.append("fedora-hdd")
        rows[name]={"immutable_optimizer_sha256":digest,"portable_group":groups.get(name),"full_optimizer_locations":where}
    result={"runs":rows,"endpoints":len(rows),"verified_inference":sum(bool(x["portable_group"]) for x in rows.values()),
            "verified_full_optimizer":sum(bool(x["full_optimizer_locations"]) for x in rows.values())}
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print({k:v for k,v in result.items() if k!="runs"},flush=True)


if __name__=="__main__": main()
