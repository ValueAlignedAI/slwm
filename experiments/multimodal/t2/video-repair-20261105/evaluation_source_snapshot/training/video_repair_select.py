"""Freeze validation-only audio comparisons before any final test evaluation."""
import argparse
import json
from pathlib import Path
from training.t2_real_av import write_json,stable_hash
from training.t2_prepare_cremad import sha256


def select(cfg):
    root=Path(cfg["artifacts"]); out=root/"selection.json"
    scores={r["id"]:json.loads((root/r["id"]/"training_complete.json").read_text())["best_validation"] for r in cfg["anchors"]+cfg["runs"]}
    value={"config_hash":stable_hash(cfg),"rule":"Minimum equal-corpus mean validation gap320ms MSE; no test metrics read",
        "primary_video":{"61":"EXP-T2-614","62":"EXP-T2-615"},"anchor":{"61":"EXP-T2-601","62":"EXP-T2-602"},
        "audio":{"61":min(("EXP-T2-611","EXP-T2-619"),key=lambda k:scores[k]),"62":"EXP-T2-612"},"validation_scores":scores,
        "checkpoint_hashes":{r["id"]:sha256(root/r["id"]/"best.pt") for r in cfg["anchors"]+cfg["runs"]}}
    if out.exists():
        if json.loads(out.read_text())!=value: raise ValueError("Previously frozen choices changed")
        return value
    if any(root.glob("EXP-T2-*/crema_test_gap*.json")) or any(root.glob("EXP-T2-*/rav_test_gap*.json")): raise ValueError("Test evaluated before selection freeze")
    write_json(out,value); return value


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); print(json.dumps(select(json.loads(Path(a.config).read_text()))))
