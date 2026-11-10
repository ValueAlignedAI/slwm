"""Freeze all compared endpoints and evaluation implementations before confirmation."""
import argparse
import json
from pathlib import Path
from training.g4_train import CFG,read,sha_file,write


def freeze(root,label,stage,seeds,profiles,filename):
    expected=[f"g4-{label}-{stage}-{profile}-{seed}" for seed in seeds for profile in profiles]
    markers={}
    for key in expected:
        r=root/key
        train,dev=read(r/"complete.json"),read(r/"dev.json")
        if train["weight_file_sha256"]!=dev["weight_sha256"]:
            raise ValueError("Development endpoint mismatch")
        markers[key]=train["weight_file_sha256"]
    record={"endpoints":markers,"evaluation_sha256":sha_file(Path("training/g4_eval.py")),
            "transfer_sha256":sha_file(Path("training/g4_transfer.py")),"live_sha256":sha_file(Path("exploration/g4_live.py")),
            "data_config_sha256":sha_file(CFG)}
    if filename.exists() and read(filename)!=record: raise ValueError("Existing confirmation lock differs")
    write(filename,record)
    print("frozen",len(markers),"endpoints",sha_file(filename),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--label",required=True)
    p.add_argument("--stage",default="pilot"); p.add_argument("--seeds",type=int,nargs="+",default=[81,82])
    p.add_argument("--profiles",nargs="+",default=read(CFG)["screen_profiles"]); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); freeze(a.root,a.label,a.stage,a.seeds,a.profiles,a.out)
