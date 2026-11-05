"""Freeze and verify G1/G2 source, corpora, checkpoints and actual media.

`--verify-only` uses the standard library and verifies a copied artifact bundle.
`--raw` additionally checks every original file against its recorded SHA/CRC.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import wave
import zlib


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--verify-only",action="store_true"); parser.add_argument("--raw",action="store_true"); args=parser.parse_args()
    configs=[json.loads(Path(p).read_text()) for p in ("configs/t2/conditional_generation.json","configs/t2/conditional_generation_residual.json")]
    corpus=Path(configs[0]["corpus"]); corpus_hashes={}; counts={}
    for split in ("train","validation","test","external"):
        path=corpus/(split+"_manifest.json"); manifest=json.loads(path.read_text()); corpus_hashes[path.name]=digest(path)
        for name,expected in manifest["files"].items():
            if digest(corpus/name)!=expected: raise ValueError("Corpus integrity failure: "+name)
        counts[split]=manifest["samples"]
    if args.verify_only:
        totals={}
        for cfg in configs:
            root=Path(cfg["artifacts"]); audit=json.loads((root/"artifact_audit.json").read_text())
            if audit["corpus_manifests"]!=corpus_hashes: raise ValueError("Copied corpus manifest mismatch")
            for name,expected in audit["artifact_hashes"].items():
                if digest(root/name)!=expected: raise ValueError("Copied artifact changed: "+name)
            for name,expected in audit["final_source_hashes"].items():
                if digest(Path(name))!=expected: raise ValueError("Copied source changed: "+name)
            totals[cfg["sprint"]]=len(audit["artifact_hashes"])
        print(json.dumps({"status":"verified","corpus_counts":counts,"artifact_files":totals,"source_parity":True})); return
    import numpy as np
    import torch
    import torchaudio
    from models.slwm_generation_residual import SignalResidualAudioHead
    from training.t2_generation_train import make_model
    from training.t2_generation_residual import codecs
    old,new=codecs(configs[1]); people={}
    for split in counts:
        rows=json.loads((corpus/(split+".json")).read_text()); lengths=np.load(corpus/f"{split}_lengths.npy")
        if len(rows)!=counts[split] or not ((lengths>=32)&(lengths<=128)).all(): raise ValueError("Invalid lengths")
        people[split]=set(r["actor"] for r in rows)
    for a,b in (("train","validation"),("train","test"),("validation","test")):
        if people[a]&people[b]: raise ValueError("People leakage")
    raw_count=0
    if args.raw:
        source=Path(configs[0]["source_corpus"])
        for split in ("train","validation","test"):
            for row in json.loads((source/(split+".json")).read_text()):
                if digest(Path(row["raw_path"]))!=row["raw_sha256"]: raise ValueError("Raw CREMA changed")
                raw_count+=1
        for row in json.loads((Path(configs[0]["external_corpus"])/"dataset_card.json").read_text())["records"]:
            path=Path(row["raw_path"])
            if digest(path)!=row["sha256"] or zlib.crc32(path.read_bytes())!=row["zip_member_crc32"]: raise ValueError("Raw RAVDESS changed")
            raw_count+=1
    late=("training/t2_generation_report.py","training/t2_generation_residual_report.py","exploration/generation_outputs.py",
        "exploration/generation_residual_outputs.py","exploration/generation_infer.py","evals/generation_intelligibility.py",
        "evals/generation_controls.py","evals/generation_shuffled_audit.py","evals/generation_artifact_audit.py","training/generation_external.py",
        "tests/test_conditional_generation.py","scripts/train/run_conditional_generation_fedora.sh","scripts/train/run_conditional_residual_fedora.sh",
        "docs/research/findings.md","docs/training/conditional_generation.md")
    for cfg in configs:
        root=Path(cfg["artifacts"]); initial=json.loads((root/"source_hashes.json").read_text())
        for name,expected in initial.items():
            if digest(root/"source_snapshot"/name)!=expected or digest(Path(name))!=expected: raise ValueError("Frozen training source changed: "+name)
        final={}
        for name in sorted(set(late)|set(initial)):
            path=Path(name); dest=root/"final_source_snapshot"/path; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,dest); final[name]=digest(path)
        checkpoints=[]
        for run in cfg["runs"]:
            path=root/run["id"]/"best.pt"; complete=json.loads((path.parent/"training_complete.json").read_text())
            if digest(path)!=complete["checkpoint_sha256"]: raise ValueError("Checkpoint integrity failure")
            checkpoint=torch.load(path,map_location="cpu",weights_only=False)
            model=make_model(run) if cfg["sprint"]=="T2-G1" else SignalResidualAudioHead(old,new)
            model.load_state_dict(checkpoint["model"],strict=True)
            count=sum(p.numel() for p in model.parameters()); p=complete["provenance"]
            expected=p["parameter_accounting"]["decoder_trainable"] if cfg["sprint"]=="T2-G1" else p["decoder_trainable"]
            if count!=expected: raise ValueError("Parameter accounting mismatch")
            checkpoints.append({"id":run["id"],"sha256":digest(path),"parameters":count,"strict_load":True}); del model,checkpoint
        for path in root.rglob("*.wav"):
            with wave.open(str(path),"rb") as audio:
                if (audio.getframerate(),audio.getnchannels(),audio.getsampwidth())!=(16000,1,2) or audio.getnframes()==0: raise ValueError("Malformed generated audio")
        if cfg["sprint"]=="T2-G1":
            originals=json.loads((root/"exploration/artifact_hashes.json").read_text())
            for name,expected in originals.items():
                if digest(root/"exploration"/name)!=expected: raise ValueError("Original G1 exploration changed: "+name)
            archive=root/"evaluation_correction/original_derived"
            for name,expected in json.loads((archive/"original_hashes.json").read_text()).items():
                if digest(archive/name)!=expected: raise ValueError("Original derived report archive changed")
        files={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="artifact_audit.json"}
        audit={"status":"verified","sprint":cfg["sprint"],"corpus_counts":counts,"corpus_manifests":corpus_hashes,"people_counts":{k:len(v) for k,v in people.items()},
            "raw_files_sha_verified":raw_count,"external_raw_crc_verified":bool(args.raw),"frozen_source_parity":True,"final_source_hashes":final,
            "checkpoint_loads":checkpoints,"artifact_hashes":files,"runtime":{"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
                "torchaudio":torchaudio.__version__,"numpy":np.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name()},
            "gpu_status":subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,driver_version","--format=csv,noheader"],capture_output=True,text=True,check=True).stdout.strip()}
        (root/"artifact_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"sprint":cfg["sprint"],"status":"verified","checkpoints":len(checkpoints),"files":len(files),"raw_files":raw_count}),flush=True)


if __name__=="__main__": main()
