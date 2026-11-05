"""Source/split integrity and actual trained content/rejection contracts."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import zlib
from training.research_sources import file_sha
from training.t2_real_av import write_json
from training.content_sync_train import load_model
from training.grid_content_data import decode_record
from exploration.video_repair_infer import load_codecs,observed_features,support_masks
from exploration.content_sync_infer import predict_sequence
from exploration.video_repair_audio import decode_pcm,constrained_audio


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/t2/content_sync.json"); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"]); corpus=Path(cfg["corpus"]); torch.set_num_threads(4)
    for name,digest in json.loads((root/"source_hashes.json").read_text()).items(): assert file_sha(name)==digest==file_sha(root/"source_snapshot"/name)
    data_card=json.loads((corpus/"dataset_card.json").read_text()); vocabulary=json.loads((corpus/"vocabulary.json").read_text()); assert vocabulary==data_card["vocabulary"]; roles={}; sentences={}; raw_count=0
    for split in ("train","validation","calibration","test","test_seen","train_person_holdout"):
        card=json.loads((corpus/(split+"_manifest.json")).read_text()); assert file_sha(corpus/(split+".json"))==card["metadata_sha256"]
        for item in card["arrays"].values(): assert file_sha(item["path"])==item["sha256"]
        rows=json.loads((corpus/(split+".json")).read_text()); roles[split]={r["actor"] for r in rows}; sentences[split]={r["sentence"] for r in rows}
        for row in rows:
            raw=Path(row["raw_path"]); assert file_sha(raw)==row["raw_sha256"] and zlib.crc32(raw.read_bytes())==row["zip_member_crc32"]; raw_count+=1
    for one,two in (("train","validation"),("train","calibration"),("train","test"),("validation","test"),("calibration","test")):
        assert not roles[one]&roles[two] and not sentences[one]&sentences[two]
    assert not roles["validation"]&roles["calibration"]
    excluded=json.loads((corpus/"exclusions.json").read_text()); assert len(excluded)==90 and all(r["actor"]=="GRID-08" for r in excluded); assert file_sha(corpus/"exclusions.json")==data_card["exclusions_sha256"]
    frozen={}; selection=json.loads((root/"selection.json").read_text())
    for run in cfg["candidate_runs"]:
        state=torch.load(root/run["id"]/"best.pt",map_location="cpu",weights_only=False)["model"]; base=torch.load(root/cfg["base_by_seed"][str(run["seed"])]/"best.pt",map_location="cpu",weights_only=False)["model"]
        for key,value in base.items(): assert torch.equal(state["base."+key],value)
        if run["arm"]!="audio":
            audio=torch.load(root/cfg["audio_by_seed"][str(run["seed"])]/"best.pt",map_location="cpu",weights_only=False)["model"]
            for key,value in audio.items():
                if key.startswith("branch."): assert torch.equal(state["audio_candidate."+key.removeprefix("branch.")],value)
        frozen[run["id"]]={"base_unchanged":True,"strong_audio_fallback_unchanged":run["arm"]!="audio"}
    for run in cfg["base_runs"]+cfg["candidate_runs"]:
        d=selection[run["id"]]; assert file_sha(root/run["id"]/"best.pt")==d["checkpoint_sha256"] and file_sha(root/run["id"]/"last.pt")==d["last_sha256"]
    calibration=json.loads((root/"calibration.json").read_text()); model,ck=load_model(root/"EXP-T2-713/best.pt","cpu"); audio,_=load_model(root/"EXP-T2-711/best.pt","cpu"); codec,old,new=load_codecs(cfg,"cpu",root/"bundle")
    row=json.loads((corpus/"validation.json").read_text())[0]
    with np.load(row["decoded_path"],allow_pickle=False) as d: pcm=d["pcm"].copy(); mouth=d["mouth"].copy()
    original_mouth,original_pcm,_=decode_record(row["raw_path"],json.loads(Path(cfg["data_config"]).read_text())); np.testing.assert_array_equal(mouth,original_mouth[:len(mouth)]); np.testing.assert_array_equal(pcm,original_pcm)
    sound=pcm.astype(np.float32)/32768.; start=(len(mouth)-32)//2+12; missing=np.zeros(len(sound),bool); missing[start*640:(start+8)*640]=True; changed=sound.copy(); changed[missing]=np.random.default_rng(22).normal(0,100,missing.sum()); threshold=calibration["EXP-T2-713"]["threshold"]
    pred,meta=predict_sequence(model,"candidate",codec,new,sound,missing,mouth,threshold,True,device="cpu"); altered,ameta=predict_sequence(model,"candidate",codec,new,changed,missing,mouth,threshold,True,device="cpu"); np.testing.assert_array_equal(pred,altered); assert meta["windows"]==ameta["windows"]
    np.testing.assert_array_equal(constrained_audio(pred,sound,missing,new,iterations=8),constrained_audio(altered,changed,missing,new,iterations=8))
    absent,_=predict_sequence(model,"candidate",codec,new,sound,missing,mouth,threshold,False,device="cpu"); expected,_=predict_sequence(audio,"candidate",codec,new,sound,missing,mouth,present=True,device="cpu"); np.testing.assert_array_equal(absent,expected)
    features=observed_features(sound,missing,new); t=min(len(mouth),(len(features)+3)//4); seen,unknown=support_masks(missing,t); offset=meta["windows"][0]["start_packet"]; obs=codec.observe(torch.tensor(features[offset*4:(offset+32)*4].reshape(1,32,256)),torch.tensor(seen[None,offset:offset+32])); pixels=torch.tensor(mouth[None,offset:offset+32]); unknown=torch.tensor(unknown[None,offset*4:(offset+32)*4])
    with torch.inference_mode():
        first=model(obs,pixels,unknown,torch.ones(1,dtype=torch.bool),threshold); obs["target"]=torch.randn_like(obs["target"])*100; obs["word_labels"]=torch.randint(0,len(vocabulary),(1,32)); second=model(obs,pixels,unknown,torch.ones(1,dtype=torch.bool),threshold)
        for key in ("raw_prediction","prediction","unit_logits","quality"): torch.testing.assert_close(first[key],second[key],atol=0,rtol=0)
    cases=json.loads((root/"exploration/audio_cases.json").read_text()); count=0
    for case in cases:
        folder=Path(case["folder"]); ref=decode_pcm(folder/"reference.wav"); mask=np.load(folder/"missing_samples.npy")
        for name in case["conditions"]:
            path=folder/(name+".wav"); assert file_sha(path)==case["wave_hashes"][name]; np.testing.assert_array_equal(decode_pcm(path)[~mask],ref[~mask]); count+=1
        np.testing.assert_array_equal(decode_pcm(folder/"missing_video71.wav"),decode_pcm(folder/"audio71.wav"))
    value={"passed":True,"raw_records_verified":raw_count,"held_out_people_and_compositions_disjoint":True,"all_arrays_and_metadata_verified":True,"reviewed_training_source_exclusions":len(excluded),"neural_checkpoints":10,"frozen_components":frozen,
        "trained_hidden_waveform_invariance":True,"trained_target_and_label_dictionary_invariance":True,"trained_missing_video_equals_strong_audio_exact":True,"video_pts_aligned_pcm_redecoded_exact":True,"waveforms_verified":count,"observed_pcm_exact":True,"source_sha256":file_sha(Path(__file__))}
    write_json(root/"audit.json",value); print(json.dumps({"audit_passed":True,"raw_records":raw_count,"waveforms":count}))


if __name__=="__main__": main()
