"""Check the full tensor stream for all four cells at preregistered training positions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from training.g4_data import digest
from training.g5_batches import JointData,load_pack
from training.g7_batches import ContinuationData,FAMILIES
from training.g8_factorial import FactorialData
from training.g7_units import put


def main(old,fresh,out):
    cfg=json.loads(Path("configs/t8/factorial.json").read_bytes())
    base=JointData(old,"train",verify=True)
    for family in FAMILIES.values():load_pack(fresh/family/"train",True)
    full=ContinuationData(base,fresh)
    mixed={arm:FactorialData(base,fresh,arm,cfg) for arm in ("sim","other")}
    steps=np.unique(np.r_[np.linspace(8192,16383,128,dtype=int),8192,10239,10240,12287,12288,16383])
    hashes={k:hashlib.sha256() for k in ("replay","sim","other","fresh")};counts=np.zeros(9,int)
    for step in steps:
        before=base.requests(int(step),256);after=full.requests(int(step),256)
        a,b=(d.requests(int(step),256) for d in mixed.values())
        families=before["family"];is_sim=families==8;changed=np.zeros(256,bool)
        for k in sorted(before):
            for cell,expected,other in ((a,after,before),(b,before,after)):
                np.testing.assert_array_equal(cell[k][is_sim],expected[k][is_sim])
                np.testing.assert_array_equal(cell[k][~is_sim],other[k][~is_sim])
            different=(before[k]!=after[k]).reshape(256,-1).any(1)
            if k in ("family","has","source","ok","sense") and different.any():raise ValueError("Different family, source, target availability or valid ticks")
            changed|=different
            for tag,x in (("replay",before),("sim",a),("other",b),("fresh",after)):
                hashes[tag].update(k.encode()+b"\0"+np.ascontiguousarray(x[k]).tobytes())
        counts+=np.bincount(families[changed],minlength=9)
    assert all(counts[k]>0 for k in FAMILIES)
    put(out,{"steps":steps.tolist(),"sampled_rows":int(len(steps)*256),"all_tensor_fields_verified":sorted(before),"changed_by_family":counts.tolist(),
        "factorization_exact":True,"stream_hashes":{k:v.hexdigest() for k,v in hashes.items()},"replay_identity":base.identity,"g7_all_fresh_identity":full.identity,
        "submitted_partial_identities":{k:v.identity for k,v in mixed.items()},"original_sampler_sha256":digest(Path("training/g7_batches.py"))})
    print("Exact four-way tensor audit",len(steps)*256,counts.tolist(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("old","fresh","out"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();main(a.old,a.fresh,a.out)
