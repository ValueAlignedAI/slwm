"""Verify completed scale artifacts without reading any new evaluation examples."""
import hashlib
import json
from pathlib import Path
import numpy as np
import tiktoken
import torch
from training.research_sources import file_sha
from training.t2_real_av import write_json,stable_hash


def main():
    cfg=json.loads(Path("configs/t1/fineweb_scale.json").read_text()); root=Path(cfg["artifacts"]); corpus=Path(cfg["corpus"]); card=json.loads((corpus/"dataset_card.json").read_text()); torch.set_num_threads(4)
    for name,digest in card["files"].items(): assert file_sha(corpus/name)==digest
    sources=json.loads((root/"source_hashes.json").read_text())
    for name,digest in sources.items(): assert file_sha(name)==digest==file_sha(root/"source_snapshot"/name)
    for entry in card["processed_sources"]: assert file_sha(entry["path"])==entry["sha256"]
    domains={s:set(v) for s,v in json.loads((corpus/"domains.json").read_text()).items()}
    assert not(domains["train"]&domains["validation"] or domains["train"]&domains["test"] or domains["validation"]&domains["test"])
    tokens={}
    for split,count in card["tokens"].items():
        data=np.memmap(corpus/(split+".tokens.bin"),dtype="<u2",mode="r"); assert len(data)==count and int(data.max())<card["tokenizer"]["vocab"]; tokens[split]=len(data)
    enc=tiktoken.get_encoding("gpt2"); ranks=hashlib.sha256(b"".join(t+b":"+str(r).encode()+b"\n" for t,r in sorted(enc._mergeable_ranks.items(),key=lambda p:p[1]))).hexdigest(); assert ranks==card["tokenizer"]["merge_ranks_sha256"]
    selection=json.loads((root/"selection.json").read_text()); runs={}
    for name,entry in selection.items():
        folder=root/name; assert file_sha(folder/"best.pt")==entry["checkpoint_sha256"] and file_sha(folder/"last.pt")==entry["last_sha256"]
        ck=torch.load(folder/"best.pt",map_location="cpu",weights_only=False); assert "optimizer" in ck and "torch_rng" in ck and "cuda_rng" in ck
        assert ck["step"]==entry["selected_step"] and ck["provenance"]["dataset_card_sha256"]==file_sha(corpus/"dataset_card.json") and ck["provenance"]["config_hash"]==stable_hash(cfg)
        assert ck["provenance"]["source_hashes"]==sources and sum(v.numel() for v in ck["model"].values())==entry["parameters"]
        assert entry["tokens_seen"]==entry["updates"]*32*512
        runs[name]={"checkpoint_and_final_hashes_verified":True,"optimizer_and_rng_retained":True,"selected_step":ck["step"],"tokens_seen":entry["tokens_seen"]}; del ck
    write_json(root/"audit.json",{"passed":True,"all_corpus_and_processed_source_hashes_verified":True,"source_snapshot_verified":True,"registrable_domains_disjoint":True,"tokens":tokens,"tokenizer_verified":True,"runs":runs,"source_sha256":file_sha(Path(__file__))}); print(json.dumps({"text_audit_passed":True,"runs":len(runs),"train_tokens":tokens["train"]}))


if __name__=="__main__": main()
