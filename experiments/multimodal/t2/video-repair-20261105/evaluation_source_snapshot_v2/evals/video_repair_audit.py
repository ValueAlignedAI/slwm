"""Integrity, frozen-weight and actual inference-contract audit for T2-V1."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import zlib
import numpy as np
import torch
from exploration.video_repair_audio import decode_pcm,constrained_audio
from exploration.video_repair_infer import load_codecs,predict_mel,decode_mouth,observed_features,support_masks
from training.video_repair_train import load_model
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json,stable_hash


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"]); corpus=Path(cfg["corpus"]); torch.set_num_threads(4)
    manifests={}; people={}; raw_count=0
    for split in ("crema_train","crema_validation","crema_test","rav_train","rav_validation","rav_test"):
        path=corpus/(split+"_manifest.json"); card=json.loads(path.read_text()); assert card["config_hash"]==stable_hash(cfg)
        assert sha256(corpus/(split+".json"))==card["metadata_sha256"]
        for item in card["arrays"].values(): assert sha256(Path(item["path"]))==item["sha256"]
        rows=json.loads((corpus/(split+".json")).read_text()); people[split]=set(r["actor"] for r in rows)
        for r in rows:
            assert sha256(Path(r["raw_path"]))==r["raw_sha256"]
            if "crc32" in r: assert zlib.crc32(Path(r["raw_path"]).read_bytes())==r["crc32"]
            raw_count+=1
        manifests[split]={"samples":len(rows),"people":len(people[split]),"manifest_sha256":sha256(path),"all_array_and_raw_hashes_verified":True}
    for prefix in ("crema_","rav_"):
        train,val,test=(people[prefix+s] for s in ("train","validation","test")); assert not(train&val or train&test or val&test)
    sources=json.loads((root/"source_hashes.json").read_text())
    for name,digest in sources.items():
        assert sha256(Path(name))==digest and sha256(root/"source_snapshot"/name)==digest
    selection=json.loads((root/"selection.json").read_text()); checkpoint_audit={}
    experts={}; anchors={}
    for seed in ("61","62"):
        assert sha256(Path(cfg["experts"][seed]["path"]))==cfg["experts"][seed]["sha256"]
        experts[seed]=torch.load(cfg["experts"][seed]["path"],map_location="cpu",weights_only=False)["model"]
        anchors[seed]=torch.load(root/selection["anchor"][seed]/"best.pt",map_location="cpu",weights_only=False)["model"]
    for run in cfg["anchors"]+cfg["runs"]:
        folder=root/run["id"]; complete=json.loads((folder/"training_complete.json").read_text()); digest=sha256(folder/"best.pt")
        assert digest==complete["checkpoint_sha256"]==selection["checkpoint_hashes"][run["id"]]
        assert sha256(folder/"last.pt")==complete["last_checkpoint_sha256"]
        ck=torch.load(folder/"best.pt",map_location="cpu",weights_only=False); state=ck["model"]; seed=str(run["seed"])
        for key,value in experts[seed].items(): assert torch.equal(state["expert."+key],value)
        if run in cfg["runs"]:
            for key,value in anchors[seed].items(): assert torch.equal(state[key],value)
        checkpoint_audit[run["id"]]={"checkpoint_sha256":digest,"expert_unchanged":True,"anchor_unchanged":not ck["anchor_only"],"selected_step":ck["step"]}
    # A real trained CNN, independent CPU inference, validation example only.
    model,ck=load_model(root/"EXP-T2-614/best.pt","cpu"); codec,old,new=load_codecs(cfg,"cpu",root/"bundle")
    row=json.loads((corpus/"crema_validation.json").read_text())[0]; sound=decode_pcm(Path(row["raw_path"])).astype(np.float32)/32768.
    pixels=decode_mouth(row["raw_path"],cfg); length=min(len(pixels),len(sound)//640); start=(length-32)//2+12; hidden=np.zeros(len(sound),bool); hidden[start*640:(start+8)*640]=True
    changed=sound.copy(); changed[hidden]=np.random.default_rng(82).normal(0,100,hidden.sum())
    prediction,meta=predict_mel(model,codec,new,sound,hidden,pixels,device="cpu"); alternate,_=predict_mel(model,codec,new,changed,hidden,pixels,device="cpu")
    np.testing.assert_array_equal(prediction,alternate)
    wa=constrained_audio(prediction,sound,hidden,new,iterations=8); wb=constrained_audio(alternate,changed,hidden,new,iterations=8); np.testing.assert_array_equal(wa,wb)
    # The observation dictionary contains a target for losses, but the model
    # must ignore it. This checks the trained artifact, not just source intent.
    y=observed_features(sound,hidden,new); packets=min(128,len(y)//4,len(pixels)); seen,unknown=support_masks(hidden,packets); s=meta["windows"][0]
    obs=codec.observe(torch.tensor(y[s*4:(s+32)*4].reshape(1,32,256)),torch.tensor(seen[None,s:s+32]))
    args=(torch.tensor(pixels[None,s:s+32]),torch.zeros(1,32,160),torch.tensor(unknown[None,s*4:(s+32)*4]),torch.ones(1,dtype=torch.bool))
    with torch.inference_mode():
        first=model(obs,*args)["prediction"]; obs["target"]=torch.randn_like(obs["target"])*100; second=model(obs,*args)["prediction"]
        torch.testing.assert_close(first,second,atol=0,rtol=0)
        absent=model(obs,*args[:-1],torch.zeros(1,dtype=torch.bool)); torch.testing.assert_close(absent["prediction"],absent["anchor"],atol=0,rtol=0)
    # Container audio is never a video input. Decode the same video stream from
    # a lossless canonical container with and without its separate soundtrack.
    controls=root/"inference_checks"; controls.mkdir(exist_ok=True); silent=controls/"recorded_video_silent.mkv"; voiced=controls/"recorded_video_with_audio.mkv"
    subprocess.run(["ffmpeg","-v","error","-y","-i",row["raw_path"],"-map","0:v:0","-an","-c:v","ffv1",str(silent)],check=True,timeout=60)
    subprocess.run(["ffmpeg","-v","error","-y","-i",str(silent),"-i",row["raw_path"],"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a","pcm_s16le",str(voiced)],check=True,timeout=60)
    v0,v1=decode_mouth(silent,cfg),decode_mouth(voiced,cfg); np.testing.assert_array_equal(v0,v1)
    p0,_=predict_mel(model,codec,new,sound,hidden,v0,device="cpu"); p1,_=predict_mel(model,codec,new,sound,hidden,v1,device="cpu"); np.testing.assert_array_equal(p0,p1)
    subprocess.run(["/home/db/venvs/vllm/bin/python","-m","exploration.video_repair_infer","--checkpoint",str(root/"EXP-T2-614/best.pt"),"--codec-dir",str(root/"bundle"),
        "--audio",row["raw_path"],"--video",str(voiced),"--gaps",f"{start*.04}:{(start+8)*.04}","--device","cpu","--output",str(controls/"cpu_file_inference")],check=True,timeout=120)
    cases=json.loads((root/"exploration/audio_cases.json").read_text()); waves=0
    for case in cases:
        folder=Path(case["folder"]); missing=np.load(folder/"missing_samples.npy"); reference=decode_pcm(folder/"reference.wav")
        for name in case["conditions"]:
            path=folder/(name+".wav"); assert sha256(path)==case["wave_hashes"][name]
            np.testing.assert_array_equal(decode_pcm(path)[~missing],reference[~missing]); waves+=1
    value={"passed":True,"data":manifests,"raw_records_verified":raw_count,"checkpoints":checkpoint_audit,"frozen_expert_and_anchor_exact":True,
        "training_source_snapshot_exact":True,"people_disjoint":True,"trained_hidden_waveform_invariance_exact":True,"trained_target_dictionary_invariance_exact":True,
        "trained_no_video_equals_anchor_exact":True,"container_soundtrack_invariance_exact":True,"cpu_file_inference_passed":True,
        "waveforms_verified":waves,"all_observed_pcm_exact":True,"audit_source_sha256":sha256(Path(__file__))}
    write_json(root/"audit.json",value); print(json.dumps({"passed":True,"raw_records":raw_count,"waveforms":waves,"checkpoints":len(checkpoint_audit)}))


if __name__=="__main__": main()
