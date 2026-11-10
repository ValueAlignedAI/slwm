"""Build independent simulator and text examples. Source grouping precedes window generation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from training.g4_data import number,tokens,digest
from training.g5_physical_data import simulate

CFG=Path("configs/t7/study.json")


def save_pack(folder,arrays,meta):
    folder.parent.mkdir(exist_ok=True,parents=True)
    if (folder/"card.json").is_file():return
    with tempfile.TemporaryDirectory(dir=folder.parent,prefix=".build-") as t:
        new=Path(t)/"pack";new.mkdir()
        for k,v in arrays.items():np.save(new/(k+".npy"),v,allow_pickle=False)
        card=dict(meta,rows=len(next(iter(arrays.values()))),sha256={k:digest(new/(k+".npy")) for k in arrays})
        (new/"card.json").write_text(json.dumps(card,sort_keys=True,indent=2)+"\n")
        new.rename(folder)
    print(folder,card["rows"],flush=True)


def physics(root):
    cfg=json.loads(CFG.read_bytes())
    for role,(lo,hi) in cfg["new_pusht"].items():
        dest=root/"fresh"/"sim"/role
        if (dest/"card.json").is_file():continue
        pieces=root/"simulation-records"/role;pieces.mkdir(exist_ok=True,parents=True)
        with ProcessPoolExecutor(max_workers=5) as pool:
            for j,_ in enumerate(pool.map(simulate,((k,pieces) for k in range(lo,hi)),chunksize=4),1):
                if j%2048==0:print(role,j,hi-lo,flush=True)
        arrays={k:[] for k in ("n","v","plan","outcome","group")}
        for k in range(lo,hi):
            with np.load(pieces/f"{k}.npz") as f:
                if int(f["group"][0])!=k:raise ValueError("Wrong fresh simulator index")
                for name in arrays:arrays[name].append(np.array(f[name]))
        save_pack(dest,{k:np.concatenate(v) for k,v in arrays.items()},{"role":role,"family":"pusht","unique_resets":hi-lo,"plans":3})


def text_pool(root,original,old):
    import tiktoken
    enc=tiktoken.get_encoding("gpt2")
    forbidden=set(np.load(old/"paired/train/uid.npy",mmap_mode="r").tolist())
    old_x=np.load(old/"paired/train/x.npy",mmap_mode="r")
    fam=np.load(old/"paired/train/family.npy",mmap_mode="r")
    content={v.tobytes() for v in old_x[fam==5]}
    for split,role,count in (("train","train",131072),("validation","dev",512),("test","test",512)):
        dest=root/"fresh"/"text"/role
        if (dest/"card.json").exists():continue
        corpus=np.memmap(original/(split+".tokens.bin"),dtype="<u2",mode="r")
        old_indices=np.random.default_rng(number("g4-text-"+role)).choice(len(corpus)//64,{"train":131072,"dev":384,"test":512}[role],replace=False)
        # Recover the document grouping from the exact source token offsets. G4 rows represented 64-token blocks.
        ends=[]; docids=[]; current=0
        with (original/"documents.tsv").open() as f:
            header=f.readline().strip().split("\t")
            for line in f:
                row=dict(zip(header,line.rstrip("\n").split("\t")))
                if row["split"]!=split:continue
                current+=int(row["tokens"]); ends.append(current);docids.append(row["id"])
        if current!=len(corpus):raise ValueError(f"Document offsets differ from corpus ({current}, {len(corpus)})")
        ends=np.array(ends); old_docs=set(np.searchsorted(ends,old_indices*64,side="right").tolist())|set(np.searchsorted(ends,old_indices*64+63,side="right").tolist())
        rng=np.random.default_rng(number("g7-document-"+role)); order=rng.permutation(len(ends))
        xs=[];uids=[];groups=[]
        for doc in order:
            if doc in old_docs:continue
            begin=int(ends[doc-1]) if doc else 0; stop=int(ends[doc])
            if stop-begin<66:continue
            start=begin+int(rng.integers(0,stop-begin-64))
            body=enc.decode(corpus[start:start+64].tolist()).strip()
            if not body:continue
            x=np.zeros(64,np.uint16); ids=tokens(body);x[:len(ids)]=ids
            uid=number(f"g7-text/{split}/{docids[doc]}")
            if uid in forbidden or x.tobytes() in content:continue
            forbidden.add(uid);content.add(x.tobytes());xs.append(x);uids.append(uid);groups.append(number(docids[doc]))
            if len(xs)==count:break
        if len(xs)!=count:raise ValueError(f"Not enough unused documents: {role} {len(xs)}")
        save_pack(dest,{"x":np.stack(xs),"uid":np.array(uids,np.uint64),"group":np.array(groups,np.uint64)},{"role":role,"family":"text","disjoint_source_documents":len(xs)})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=("sim","text"));p.add_argument("--root",type=Path,required=True);p.add_argument("--old",type=Path,default=Path("/mnt/slwm-data/SLWM/g5"));p.add_argument("--text",type=Path)
    a=p.parse_args()
    if a.phase=="sim":physics(a.root)
    else:text_pool(a.root,a.text,a.old)
