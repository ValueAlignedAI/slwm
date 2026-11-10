"""Pinned broad English corpus with document/domain isolation and no token fitting."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlsplit
import numpy as np
import publicsuffix2
import pyarrow.parquet as pq
import tiktoken
from training.research_sources import file_sha


def document_keys(text,url):
    normalized=re.sub(r"\s+"," ",unicodedata.normalize("NFKC",text).casefold()).strip()
    digest=hashlib.sha256(normalized.encode()).hexdigest()
    parts=urlsplit(url); host=(parts.hostname or "missing.invalid").lower().removeprefix("www.")
    domain=publicsuffix2.get_sld(host,strict=True) or host
    canonical=host+parts.path.rstrip("/")
    # A long identical normalized prefix also excludes copied/truncated pages.
    prefix=hashlib.sha256(" ".join(normalized.split()[:256]).encode()).hexdigest()
    bucket=int(hashlib.sha256(domain.encode()).hexdigest()[:12],16)%1000
    split="test" if bucket>=995 else "validation" if bucket>=990 else "train"
    return split,domain,canonical,digest,prefix


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="artifacts/text/fineweb_edu_scale_v1"); a=p.parse_args(); root=Path(a.root)
    if (root/"dataset_card.json").exists(): return
    pinned=json.loads((root/"pinned_sources.json").read_text()); enc=tiktoken.get_encoding("gpt2"); seen=set(); urls=set(); prefixes=set(); domains={s:set() for s in ("train","validation","test")}
    counts=Counter(); token_counts=Counter(); doc_counts=Counter(); caps={"train":2_600_000_000,"validation":20_000_000,"test":20_000_000}; started=time.time(); handles={s:(root/(s+".tokens.bin.part")).open("wb") for s in domains}; sources=[]
    index=(root/"documents.tsv.part").open("w"); index.write("split\tid\tdomain\ttokens\tcontent_sha256\n")
    try:
        for item in pinned["files"]:
            path=root/"sources"/Path(item["path"]).name
            while not path.exists(): time.sleep(5)
            if file_sha(path)!=item["lfs"]["oid"]: raise ValueError("Pinned parquet hash changed")
            source={"path":str(path),"sha256":item["lfs"]["oid"],"documents_scanned":0}; sources.append(source)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=["text","url","id"]):
                rows=batch.to_pylist(); keep=[]
                for row in rows:
                    counts["scanned"]+=1; source["documents_scanned"]+=1
                    split,domain,url,digest,prefix=document_keys(row["text"],row["url"])
                    if digest in seen or prefix in prefixes or url in urls: counts["duplicate"]+=1; continue
                    seen.add(digest); prefixes.add(prefix); urls.add(url)
                    if token_counts[split]>=caps[split]: continue
                    keep.append((row,split,domain,digest))
                encoded=enc.encode_batch([r[0]["text"] for r in keep],allowed_special=set(),disallowed_special=(),num_threads=6)
                for (row,split,domain,digest),tokens in zip(keep,encoded):
                    if token_counts[split]>=caps[split]: continue
                    if len(tokens)<128: counts["short"]+=1; continue
                    tokens=tokens+[enc.eot_token]; tokens=tokens[:caps[split]-token_counts[split]]
                    handles[split].write(np.asarray(tokens,dtype="<u2").tobytes()); token_counts[split]+=len(tokens); doc_counts[split]+=1; domains[split].add(domain)
                    index.write(f"{split}\t{row['id']}\t{domain}\t{len(tokens)}\t{digest}\n")
                if counts["scanned"]%8192<256:
                    progress={"documents":dict(counts),"tokens":dict(token_counts),"seconds":time.time()-started}; (root/"prepare_progress.json").write_text(json.dumps(progress)+"\n"); print(json.dumps(progress),flush=True)
                if token_counts["train"]>=caps["train"]: break
            if token_counts["train"]>=caps["train"]: break
    finally:
        for f in handles.values(): f.close()
        index.close()
    if token_counts["train"]<2_500_000_000 or min(token_counts["validation"],token_counts["test"])<2_000_000: raise ValueError("Insufficient unique broad text for registered scale profile")
    for s in domains:
        (root/(s+".tokens.bin.part")).replace(root/(s+".tokens.bin"))
    (root/"documents.tsv.part").replace(root/"documents.tsv")
    assert not(domains["train"]&domains["validation"] or domains["train"]&domains["test"] or domains["validation"]&domains["test"])
    tokenizer_hash=hashlib.sha256(b"".join(token+b":"+str(rank).encode()+b"\n" for token,rank in sorted(enc._mergeable_ranks.items(),key=lambda p:p[1]))).hexdigest()
    card={"dataset":pinned["dataset"],"revision":pinned["revision"],"license":pinned["license"],"processed_sources":sources,"tokens":dict(token_counts),"documents":dict(doc_counts),"filters":dict(counts),
        "tokenizer":{"name":"gpt2","vocab":enc.n_vocab,"eos":enc.eot_token,"tiktoken_version":tiktoken.__version__,"merge_ranks_sha256":tokenizer_hash,"trained_here":False},
        "split":"Registrable-domain SHA256 buckets; train<990, validation990..994, test995..999",
        "dedup":"Global NFKC/casefold whitespace-normalized exact text, first256-word prefix and canonical host/path; source dedup remains additional, near duplicates not guaranteed absent",
        "no_cross_split_domains":True,"domain_counts":{s:len(v) for s,v in domains.items()},"tokens_per_124m_parameter":token_counts["train"]/124_439_808,"scope":"English web text only; no code-quality or joint multimodal claim",
        "files":{name:file_sha(root/name) for name in [s+".tokens.bin" for s in domains]+["documents.tsv"]},"seconds":time.time()-started,"source_sha256":file_sha(Path(__file__))}
    (root/"domains.json").write_text(json.dumps({s:sorted(v) for s,v in domains.items()})+"\n"); card["files"]["domains.json"]=file_sha(root/"domains.json")
    (root/"dataset_card.json").write_text(json.dumps(card,indent=2,sort_keys=True)+"\n"); print(json.dumps({"prepared":dict(token_counts),"domains":card["domain_counts"],"seconds":card["seconds"]}),flush=True)


if __name__=="__main__": main()
