"""Count actual single-source views, skipped observations and unique records at fixed training budgets."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from training.g4_train import CFG,Sampler,read,write


class Index:
    def __init__(self,root):
        self.host={k:np.load(root/f"{k}.npy",mmap_mode="r") for k in ("family","has")}
        self.uid=np.load(root/"uid.npy",mmap_mode="r")
        self.pools=[np.flatnonzero(self.host["family"]==i) for i in range(7)]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); data=Index(a.data/"train"); cfg=read(CFG); result={}
    for fraction in (1.,.25):
        sam=Sampler(data,256,cfg["seed"],cfg["mixture_16"],fraction)
        counts=np.zeros(4,np.int64); observed=np.zeros(4,np.int64); ids=set()
        for step in range(1024):
            samples,views=sam.sample(step); ids.update(map(int,samples))
            source=views[:,:2]; visible=~views[:,2:].astype(bool)
            counts+=np.bincount(source.flatten(),minlength=4)
            observed+=np.bincount(source[visible],minlength=4)
            if step+1 in (256,512,1024):
                result[f"{fraction}-{step+1}"]={"unique_packed_examples":len(ids),"presentations":(step+1)*256,
                    "source_order":["audio","visual","numeric","text"],"selected_views":counts.tolist(),
                    "visible_views":observed.tolist(),"visible_share":(observed/observed.sum()).tolist()}
    write(a.out,result)


if __name__=="__main__": main()
