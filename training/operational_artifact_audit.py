"""Selected-state identity, actual-model causality and persisted media integrity."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from models.slwm_stream_folded import fold_signal_branches
from training.broad_stream_train import load_model
from training.research_sources import file_sha
from training.t2_real_av import write_json


def tensor_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()): h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def same_state(a,b):
    assert set(a)==set(b)
    for key in a: assert torch.equal(a[key].cpu(),b[key].cpu()),key


def main():
    torch.set_num_threads(4); torch.manual_seed(410); cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); root=Path(cfg["artifacts"]); textcfg=json.loads(Path("configs/t1/fineweb_extend_v2.json").read_text()); textroot=Path(textcfg["artifacts"]); policy=Path("experiments/policy/continuous-sensing-20261105"); assert(root/"export_complete.json").exists(); result={"sources":{},"worlds":{},"language":{},"actors":{},"media":{}}
    for folder in (root,textroot,policy,root/"evaluation"):
        hashes=json.loads((folder/"source_hashes.json").read_text())
        for name,digest in hashes.items(): assert file_sha(name)==digest and file_sha(folder/"source_snapshot"/name)==digest,name
        result["sources"][str(folder)]={"files":len(hashes),"live_and_snapshot_match":True}
    selection=json.loads((root/"selection.json").read_text()); bundle=root/"inference_bundle"; manifest=json.loads((bundle/"manifest.json").read_text())
    for run in cfg["runs"]:
        name=run["id"]; path=root/name/"best.pt"; assert file_sha(path)==selection[name]["checkpoint_sha256"]; model,ck=load_model(path,"cpu"); model.requires_grad_(False); exported=torch.load(bundle/manifest["worlds"][name]["file"],map_location="cpu",weights_only=False); same_state(ck["model"],exported["model"]); assert sum(p.numel() for p in model.parameters())==ck["parameters"]; before=tensor_digest(model.state_dict()); n=64; valid=torch.ones(1,n,dtype=torch.bool); observed=valid.clone(); observed[:,20:28]=False; no=torch.zeros_like(valid); audio=torch.randn(1,n,400); video=torch.randn(1,n,64); phones=torch.randint(0,41,(1,n)); style=torch.zeros(1,64); args=(audio,video,phones,observed,no,no,valid,style)
        with torch.inference_mode():
            expected=model(*args); changed_audio=audio.clone(); changed_audio[:,20:28]=float("nan"); hidden=model(changed_audio,video+1000,phones+100000,observed,no,no,valid,style)
            for key in expected: torch.testing.assert_close(expected[key],hidden[key],rtol=0,atol=0)
            later=audio.clone(); later[:,32:]+=999; future_changed=model(later,video,phones,observed,no,no,valid,style); torch.testing.assert_close(expected["field"][:,:32],future_changed["field"][:,:32],rtol=0,atol=0)
            first=model(*(x[:,:32] for x in args[:7]),style); second=model(*(x[:,32:] for x in args[:7]),style,first["memory"])
            for key in ("mel","video","field","phone_logits","character_logits","activity","risk"): torch.testing.assert_close(expected[key],torch.cat((first[key],second[key]),1),rtol=2e-5,atol=2e-5)
            folding=None
            if run["variant"]=="slwm":
                folded=fold_signal_branches(copy.deepcopy(model)); actual=folded(*args); maximum=max(float((expected[k]-actual[k]).abs().max()) for k in expected)
                for key in expected: torch.testing.assert_close(expected[key],actual[key],rtol=1e-4,atol=2e-4)
                folding={**folded.folding_metadata,"all_head_state_max_error":maximum}; del folded,actual
        assert tensor_digest(model.state_dict())==before; result["worlds"][name]={"selected_step":ck["step"],"parameters":ck["parameters"],"neural_state_sha256":before,"export_identical":True,"hidden_audio_video_text_invariant":True,"future_chunk_isolation":True,"chunk_state_matches_full":True,"weights_unchanged":True,"folding":folding}; del model,ck,exported,expected,hidden,future_changed,first,second; print(json.dumps({"audited_world":name}),flush=True)
    text_manifest=json.loads((textroot/"inference_exports/manifest.json").read_text()); text_selection=json.loads((textroot/"selection.json").read_text())
    for run in textcfg["runs"]:
        name=run["id"]; path=textroot/name/"best.pt"; assert file_sha(path)==text_selection[name]["checkpoint_sha256"]; ck=torch.load(path,map_location="cpu",weights_only=False); exported=torch.load(textroot/"inference_exports"/text_manifest[name]["file"],map_location="cpu",weights_only=False); same_state(ck["model"],exported["model"]); result["language"][name]={"export_identical":True,"neural_state_sha256":tensor_digest(ck["model"]),"actual_optimizer_step":ck["step"],"selected_training_tokens":ck["total_token_equivalent_step"]*16384}; del ck,exported
    actors=json.loads((policy/"selection.json").read_text())
    for world,chosen in actors.items():
        for mode,entry in (("field",chosen),("raw",chosen["raw_control"])):
            ck=torch.load(policy/world/("raw" if mode=="raw" else "")/"actor.pt",map_location="cpu",weights_only=False); exported=torch.load(bundle/manifest["actors"][entry["id"]]["file"],map_location="cpu",weights_only=False); same_state(ck["model"],exported["model"]); assert np.array_equal(ck["mean"],exported["mean"]) and np.array_equal(ck["scale"],exported["scale"]); assert ck["thresholds"]==exported["thresholds"]; assert exported["world_checkpoint_sha256"]==manifest["worlds"][world]["sha256"]; result["actors"][entry["id"]]={"world":world,"mode":mode,"export_identical":True,"thresholds_identical":True}; del ck,exported
    repaired=0; observed=0
    for case in json.loads((root/"evaluation/speech_cases.json").read_text()):
        folder=Path(case["folder"]); reference,sr=sf.read(folder/"reference.wav",dtype="int16"); assert sr==24000; missing=np.load(folder/"missing_samples.npy")
        for name in case["conditions"]:
            wave,rate=sf.read(folder/(name+".wav"),dtype="int16"); assert rate==sr and wave.shape==reference.shape; assert np.array_equal(wave[~missing],reference[~missing]); repaired+=1; observed+=int((~missing).sum())
    for case in json.loads((root/"evaluation/av_repair_cases.json").read_text()):
        reference,sr=sf.read(root/"evaluation/long_inputs"/case["id"]/"input.wav",dtype="int16"); wave,rate=sf.read(Path(case["folder"])/"output.wav",dtype="int16"); missing=np.zeros(len(reference),bool); a,z=case["gap_interval"]; missing[round(a*sr):round(z*sr)]=True; assert sr==rate==24000 and wave.shape==reference.shape and np.array_equal(wave[~missing],reference[~missing]); repaired+=1; observed+=int((~missing).sum())
    result["media"].update(repaired_wavs_checked=repaired,observed_samples_checked=observed,all_observed_pcm_exact=True)
    long_cases=json.loads((root/"evaluation/long_cases.json").read_text())+json.loads((root/"evaluation/text_cases.json").read_text()); durations=[]
    for case in long_cases:
        folder=Path(case["folder"]); info=sf.info(folder/"output.wav"); assert info.samplerate==24000 and info.frames==case["waveform"]["samples"]; assert info.duration>30; assert file_sha(folder/"output.mp4")==case["video"]["sha256"]; assert case["video"]["frames"]*1024==info.frames; durations.append(info.duration)
    result["media"].update(long_media_cases=len(long_cases),minimum_long_seconds=min(durations),maximum_long_seconds=max(durations),frame_sample_clocks_match=True); paced=json.loads((policy/"EXP-T2-801/paced_cpu.json").read_text()); assert paced["audit"]["weights_unchanged"] and len(paced["rows"])==30 and paced["audit"]["wall_seconds"]>=40.95; assert all(r["observation_read_wall_seconds"]+1e-6>=r["observation_available_after_bootstrap_seconds"] for r in paced["rows"]); result["paced_cpu"]={"arrival_order_respected":True,"automatic_decisions":30,"wall_seconds":paced["audit"]["wall_seconds"],"missed_deadlines":sum(r["deadline_missed"] for r in paced["rows"]),"weights_unchanged":True}; write_json(root/"artifact_audit.json",result); print(json.dumps({"artifact_audit_passed":True,"worlds":len(result["worlds"]),"text":len(result["language"]),"actors":len(result["actors"]),"media":result["media"]}),flush=True)


if __name__=="__main__": main()
