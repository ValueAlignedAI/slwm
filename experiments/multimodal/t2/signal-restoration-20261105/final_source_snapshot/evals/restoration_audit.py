"""Audit immutable R1/R2 models, data, raw sources and actual output artifacts.

The copied-bundle verifier requires only Python's standard library.
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


def read(p): return json.loads(Path(p).read_text())


def digest(path,kind="sha256"):
    h=hashlib.new(kind)
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--raw",action="store_true"); parser.add_argument("--verify-only",action="store_true"); args=parser.parse_args()
    c1=read("configs/t2/signal_restoration.json"); c2={**c1,**read("configs/t2/content_conservation.json")}; configs=(c1,c2)
    datafiles={}; selected={}
    specs=[(Path(c1["generation_corpus"]),s) for s in ("train","validation","test")]+[(Path(c1["new_corpus"]),s) for s in ("libri_train","libri_validation","libri_test","ravdess_test")]+[(Path(c2["other_corpus"]),"other_test")]
    for root,split in specs:
        manifest=root/(split+"_manifest.json"); card=read(manifest); datafiles[str(manifest)]=digest(manifest)
        for name,expected in card["files"].items():
            path=root/name
            if digest(path)!=expected: raise ValueError("Dataset changed: "+str(path))
            datafiles[str(path)]=expected
        selected[split]=read(root/(split+".json"))
    old=Path(c1["old_corpus"]); oldcard=read(old/"dataset_card.json")
    if digest(old/"codec.npz")!=oldcard["files"]["codec.npz"]: raise ValueError("Original codec changed")
    datafiles[str(old/"codec.npz")]=digest(old/"codec.npz")
    for source in c1["checkpoints"].values():
        if digest(source["path"])!=source["sha256"]: raise ValueError("M1 initialization changed")
        datafiles[source["path"]]=source["sha256"]
    if args.verify_only:
        result={}
        for cfg in configs:
            root=Path(cfg["artifacts"]); audit=read(root/"artifact_audit.json")
            if datafiles!=audit["data_files"]: raise ValueError("Copied data index changed")
            for name,expected in audit["artifact_files"].items():
                if digest(root/name)!=expected: raise ValueError("Copied artifact changed: "+name)
            for name,expected in audit["final_source_hashes"].items():
                if digest(name)!=expected: raise ValueError("Copied current source changed: "+name)
            result[cfg["sprint"]]=len(audit["artifact_files"])
        print(json.dumps({"status":"verified","artifact_files":result,"data_files":len(datafiles),"model_source_parity":True})); return
    import numpy as np
    import torch
    import torchaudio
    from models.slwm_restoration import RestorationSLWM
    from models.slwm_content import ContentRestorationSLWM
    people={k:{r["actor"] for r in rows} for k,rows in selected.items()}
    for a,b in (("train","validation"),("train","test"),("validation","test"),("libri_train","libri_validation"),("libri_train","libri_test"),("libri_validation","libri_test"),("other_test","libri_train"),("other_test","libri_validation"),("other_test","libri_test")):
        if people[a]&people[b]: raise ValueError("Speaker leakage")
    old_rav=read("artifacts/t2_audio_visual/ravdess_external_v1/dataset_card.json")
    if people["ravdess_test"]&{r["actor"] for r in old_rav["records"]}: raise ValueError("New RAVDESS overlaps earlier external actors")
    raw_count=0; archive_records=[]
    if args.raw:
        for split,rows in selected.items():
            for r in rows:
                path=Path(r["raw_path"])
                if digest(path)!=r["raw_sha256"]: raise ValueError("Original media changed: "+str(path))
                if "crc32" in r and zlib.crc32(path.read_bytes())!=r["crc32"]: raise ValueError("RAVDESS CRC changed")
                raw_count+=1
        for root,subset,md5 in ((Path(c1["new_corpus"]),"dev-clean","42e2234ba48799c1f50f24a7926300a1"),(Path(c1["new_corpus"]),"test-clean","32fa31d27d2e1cad72775fee3f4849a9"),(Path(c2["other_corpus"]),"test-other","fb5a50374b501bb3bac4815ee91d3135")):
            path=root/(subset+".tar.gz")
            if digest(path,"md5")!=md5: raise ValueError("Official archive mismatch")
            archive_records.append({"source":str(path),"md5":md5,"sha256":digest(path)})
    late=("evals/restoration_content_probe.py","evals/content_conservation_probe.py","evals/restoration_silence_control.py","evals/restoration_inference_checks.py","evals/restoration_audit.py",
        "exploration/restoration_outputs.py","exploration/restoration_infer.py","exploration/content_conservation_outputs.py","exploration/generation_outputs.py","exploration/motion_outputs.py",
        "evals/motion_audio_intelligibility.py","training/restoration_report.py","tests/test_signal_restoration.py","scripts/train/run_signal_restoration_fedora.sh","scripts/train/run_content_conservation_fedora.sh",
        "docs/research/findings.md","docs/research/restoration_probe_addendum_20261105.md","docs/training/signal_restoration.md")
    for cfg in configs:
        root=Path(cfg["artifacts"]); initial=read(root/"source_hashes.json"); final={}
        for name,expected in initial.items():
            if digest(name)!=expected or digest(root/"source_snapshot"/name)!=expected: raise ValueError("Frozen training source changed: "+name)
        for name in sorted(set(late)|set(initial)):
            dest=root/"final_source_snapshot"/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(name,dest); final[name]=digest(name)
        loads=[]
        for run in cfg["runs"]:
            folder=root/run["id"]; complete=read(folder/"training_complete.json")
            for filename,key in (("best.pt","checkpoint_sha256"),("last.pt","last_checkpoint_sha256")):
                if digest(folder/filename)!=complete[key]: raise ValueError("Checkpoint changed")
            checkpoint=torch.load(folder/"best.pt",map_location="cpu",weights_only=False)
            model=(ContentRestorationSLWM if cfg["sprint"]=="T2-R2" else RestorationSLWM)(cfg,run); model.load_state_dict(checkpoint["model"],strict=True)
            count=sum(p.numel() for p in model.parameters())
            if count!=complete["provenance"]["counts"]["total"]: raise ValueError("Parameter count mismatch")
            if run["id"]=="EXP-T2-509":
                warm=torch.load(c1["checkpoints"]["signal"]["path"],map_location="cpu",weights_only=False)["model"]
                for name,value in model.world.state_dict().items():
                    if not torch.equal(value,warm[name]): raise ValueError("Frozen core updated")
            probe=folder/"content_probe.json"
            if probe.exists():
                p=read(probe)
                for split,ids in p["training_clips"].items():
                    corpus_split=split.removeprefix("crema_")
                    if not set(ids)<={r["sample_id"] for r in selected[corpus_split]}: raise ValueError("Readout fitted outside training")
            loads.append({"id":run["id"],"selected_step":complete["selected_step"],"parameters":count,"sha256":complete["checkpoint_sha256"],"strict_load":True})
            del checkpoint,model
        for path in root.rglob("*.wav"):
            with wave.open(str(path),"rb") as audio:
                if (audio.getframerate(),audio.getnchannels(),audio.getsampwidth())!=(16000,1,2) or audio.getnframes()<=0: raise ValueError("Malformed waveform")
        for path in root.rglob("predicted_mels.npz"):
            with np.load(path,allow_pickle=False) as arrays:
                if any(not np.isfinite(arrays[k]).all() for k in arrays.files): raise ValueError("Nonfinite generated mel")
        files={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="artifact_audit.json"}
        audit={"status":"verified","sprint":cfg["sprint"],"data_files":datafiles,"split_counts":{s:len(rows) for s,rows in selected.items()},
            "people_counts":{s:len(a) for s,a in people.items()},"raw_files_verified":raw_count,"official_archives":archive_records,"checkpoint_loads":loads,
            "initial_source_parity":True,"final_source_hashes":final,"artifact_files":files,"runtime":{"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,
                "torchaudio":torchaudio.__version__,"numpy":np.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name()},
            "gpu_status":subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,driver_version","--format=csv,noheader"],capture_output=True,text=True,check=True).stdout.strip()}
        (root/"artifact_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"sprint":cfg["sprint"],"status":"verified","models":len(loads),"artifact_files":len(files),"raw_files":raw_count}),flush=True)


if __name__=="__main__": main()
