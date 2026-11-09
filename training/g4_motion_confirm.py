"""Confirmation of the preregistered temporal loss on GRID's held-out people and sentence combinations."""
import argparse
from pathlib import Path

import numpy as np

from training.g4_eval import check_lock,evaluate
from training.g4_lock import freeze
from training.g4_train import Data,read,write


def main():
    q=argparse.ArgumentParser(); q.add_argument("--out",type=Path,required=True); q.add_argument("--data",type=Path,required=True)
    q.add_argument("--backend",default="cpu"); q.add_argument("--batch",type=int,default=64)
    a=q.parse_args(); lock=a.out/"motion-confirmation.json"
    freeze(a.out,"v5e-fair-all","medium",[81,82],["gpt2-cross","gpt2-motion","signal-cross","signal-motion"],lock)
    data=Data(a.data,"test")
    data.pools=[data.pools[0]]+[np.array([],np.int32) for _ in range(6)]
    rows={}
    for tag in sorted(read(lock)["endpoints"]):
        path=a.out/tag
        check_lock(lock,path,read(path/"complete.json"))
        if not (path/"motion-test.json").exists(): evaluate(path,a.data,"motion-test",a.batch,a.backend,data_cache=data)
        scores=read(path/"motion-test.json")["scores"]
        rows[tag]={key:scores["grid/"+key] for key in ("visual-to-audio","audio-to-visual")}
        print(tag,rows[tag]["visual-to-audio"]["correct"],flush=True)
    write(a.out/"motion-confirmation-summary.json",rows)


if __name__=="__main__": main()
