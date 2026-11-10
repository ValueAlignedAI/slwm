"""All registered fresh-speaker conditions, after core/readout/calibration freeze."""
import fcntl
import gc
import json
from pathlib import Path

import numpy as np
import torch

from exploration.stream_media import write_audio
from exploration.broad_stream_evaluate import edit_distance
from training.general_batch import CACHE
from training.general_evaluation_lock import register
from training.general_sources import write
from training.general_speech_calibrate import recipes,case_rows,case_name
from training.general_speech_media import SpeechRenderer
from training.general_speech_score import Recognizer,aggregate
from training.general_train import initialize
from training.research_sources import file_sha


def paired_wer(candidate,baseline):
    by_id={r["id"]:r for r in baseline};pairs=[(r,by_id[r["id"]]) for r in candidate if r["reference_words"]>0]
    groups=sorted({r["cluster"] for r,_ in pairs});values=[]
    for group in groups:
        current=[(a,b) for a,b in pairs if a["cluster"]==group]
        values.append([sum(a["edits"] for a,b in current),sum(b["edits"] for a,b in current),sum(a["reference_words"] for a,b in current)])
    if not values:return {"groups":0,"gain":None,"gain_95ci":[None,None],"passes_half_point_gain":False}
    values=np.asarray(values,np.float64);total=values.sum(0);rng=np.random.default_rng(7391)
    sampled=values[rng.integers(0,len(values),(10000,len(values)))].sum(1)
    effect=(sampled[:,1]-sampled[:,0])/sampled[:,2].clip(1)
    gain=(total[1]-total[0])/max(1,total[2]);low,high=np.quantile(effect,[.025,.975])
    return {"groups":len(groups),"gain":float(gain),"gain_95ci":[float(low),float(high)],"candidate_wer":float(total[0]/max(1,total[2])),"baseline_wer":float(total[1]/max(1,total[2])),"passes_half_point_gain":bool(gain>=.005 and low>0),"resampling":"10000 paired cluster bootstrap draws, seed7391"}


def baseline_files(renderer,prepared,family,destination):
    marker=destination/"complete.json"
    if marker.exists():return json.loads(marker.read_text())
    row=prepared["row"];t=renderer.cfg["context"];start=prepared["source_start_packet"]+prepared["window_start_packet"]
    mel=np.asarray(np.load(row["mel_file"],mmap_mode="r")[row["offset"]+start:row["offset"]+start+t],np.float32).reshape(-1,100)
    interpolation=prepared["baseline"].reshape(-1,100)*renderer.stats["audio_scale"]+renderer.stats["audio_mean"]
    candidates={"interpolation":interpolation,"oracle_mel_gap":mel}
    if family=="speech":
        phone=np.asarray(np.load(row["phone_file"],mmap_mode="r")[row["offset"]+start:row["offset"]+start+t],np.int64).reshape(1,-1)
        with torch.inference_mode():
            physical=renderer.phone.render_physical(torch.from_numpy(phone).cuda(),torch.from_numpy(prepared["physical"][None]).cuda(),torch.from_numpy(prepared["present"][None]).cuda())[0].cpu().numpy()
        physical[prepared["present"]]=prepared["physical"][prepared["present"]];candidates["oracle_phone_gap"]=physical
    result=renderer.save_many(prepared,candidates,destination)
    for name,sound in [("source",prepared["sound"]),("silenced",np.where(prepared["missing"],0,prepared["sound"]))]:
        path=destination/(name+".flac");result[name]={"path":str(path),**write_audio(path,sound),"oracle":False}
    write(marker,result);return result


def main():
    torch.set_num_threads(4);cfg,recipe,root=recipes();register(cfg)
    lock=json.loads((Path(cfg["root"])/"evaluation_lock.json").read_text())
    assert lock["speech_calibration_sha256"]==file_sha(root/"calibration.json")
    calibration=json.loads((root/"calibration.json").read_text())["runs"];all_rows=[]
    for run in (r for r in cfg["runs"] if r["id"] in recipe["runs"]):
        folder=root/"evaluation"/run["id"];folder.mkdir(parents=True,exist_ok=True)
        if (folder/"complete.json").exists():all_rows+=json.loads((folder/"recognized.json").read_text());continue
        chosen=calibration[run["id"]];checkpoint=Path(cfg["root"])/run["id"]/"endpoint.pt"
        assert file_sha(checkpoint)==chosen["core_checkpoint_sha256"]
        jobs=[];content=[];traces=[]
        with open("/tmp/slwm_research_gpu.lock","a") as gpu_lock:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX)
            ck=torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=True);model=initialize(cfg,run).cuda().eval().requires_grad_(False);model.load_state_dict(ck["model"]);del ck
            renderer=SpeechRenderer(cfg)
            for family in ("speech","ami"):
                for row,case in case_rows(family,"test",secondary=True):
                    name=case_name(row,case);prepared=renderer.case(row,case);stratum=case.get("stratum","read_speech")
                    metadata={"id":family+":"+name,"person":row["actor"],"cluster":row["group"],"family":family,"stratum":stratum,"gap_seconds":case.get("gap_seconds",.16),"reference":prepared["reference"],"run":run["id"]}
                    marker=folder/family/name/"case.json"
                    if marker.exists():
                        saved=json.loads(marker.read_text());jobs+=saved["jobs"];traces+=saved["traces"]
                        if saved.get("content"):content.append(saved["content"])
                        continue
                    base=baseline_files(renderer,prepared,family,root/"evaluation/baselines"/family/name)
                    jobs.extend({**metadata,"condition":condition,"baseline":True,**media} for condition,media in base.items())
                    controls=["audio"] if family=="speech" else ["audio","aligned","static",*["shift:"+str(n) for n in recipe["primary_shifts_packets"]+recipe["diagnostic_shifts_packets"]]]
                    candidates={};case_traces=[]
                    for condition in controls:
                        proposal=renderer.infer(model,prepared,condition,chosen["timing_threshold"])
                        alpha=chosen["alpha"];key=condition.replace(":","_")
                        candidates[key]=(1-alpha)*proposal["direct"]+alpha*proposal["phonetic"]
                        if condition in ("audio","aligned"):
                            candidates[key+"_direct"]=proposal["direct"];candidates[key+"_phonetic"]=proposal["phonetic"]
                        item={**metadata,"condition":condition,"matching_probability":proposal["matching_probability"],"video_accepted":proposal["video_accepted"],"field_local_text":proposal["ctc_local"],"field_text_scope":"Received1.365s context only"}
                        if family=="speech":
                            a=row["offset"]+prepared["window_start_packet"];truth=np.asarray(np.load(row["phone_file"],mmap_mode="r")[a:a+cfg["context"]],np.int64).reshape(-1)
                            valid=(truth>0)&~prepared["present"];item.update(hidden_nonsil_phone_errors=int(((proposal["phones"]!=truth)&valid).sum()),hidden_nonsil_phone_frames=int(valid.sum()))
                        case_traces.append(item)
                    files=renderer.save_many(prepared,candidates,folder/family/name)
                    own_jobs=[{**metadata,"condition":condition,"baseline":False,**media} for condition,media in files.items()]
                    jobs+=own_jobs;traces+=case_traces
                    own_content=None
                    if family=="speech":own_content={**metadata,**renderer.content_rollout(model,prepared)};content.append(own_content)
                    marker.parent.mkdir(parents=True,exist_ok=True)
                    write(marker,{"jobs":[*[{**metadata,"condition":condition,"baseline":True,**media} for condition,media in base.items()],*own_jobs],"traces":case_traces,"content":own_content})
            recognizer=Recognizer(recipe,root/"recognizer_cache.json");recognized=recognizer.score(jobs)
            for row in content:
                errors,count=edit_distance(recognizer.processor.tokenizer.normalize(row["reference"]),recognizer.processor.tokenizer.normalize(row["text"]));row.update(ctc_edits=errors,reference_words=count)
            write(folder/"recognized.json",recognized);write(folder/"traces.json",traces);write(folder/"whole_content.json",content)
            write(folder/"complete.json",{"run":run["id"],"conditions":len(recognized),"core_checkpoint_sha256":file_sha(checkpoint),"calibration":chosen,"new_fit_updates":0})
            all_rows+=recognized;del model,renderer,recognizer;gc.collect();torch.cuda.empty_cache()
        print(json.dumps({"fresh_speech_evaluated":run["id"],"waveforms":len(recognized)}),flush=True)
    summary={};gates={}
    for run in recipe["runs"]:
        subset=[r for r in all_rows if r["run"]==run]
        def selected(family,condition,stratum=None,gap=.16):return [r for r in subset if r["family"]==family and r["condition"]==condition and r["gap_seconds"]==gap and (stratum is None or r["stratum"]==stratum)]
        speech=paired_wer(selected("speech","audio"),selected("speech","interpolation"))
        av=paired_wer(selected("ami","aligned","own_nonoverlap"),selected("ami","audio","own_nonoverlap"))
        shifted={str(n):paired_wer(selected("ami","shift_"+str(n),"own_nonoverlap"),selected("ami","audio","own_nonoverlap")) for n in recipe["primary_shifts_packets"]}
        harm=max([-v["gain"] for v in shifted.values() if v["gain"] is not None],default=0.)
        strata={}
        for family in ("speech","ami"):
            for category in sorted({r["stratum"] for r in subset if r["family"]==family}):
                strata[family+"/"+category]={condition:aggregate(selected(family,condition,category)) for condition in sorted({r["condition"] for r in subset if r["family"]==family})}
        secondary={str(gap):paired_wer(selected("speech","audio",gap=gap),selected("speech","interpolation",gap=gap)) for gap in recipe["secondary_gap_seconds"]}
        summary[run]={"speech_gain_over_interpolation":speech,"secondary_gap_effects":secondary,"correct_video_gain_over_audio":av,"mistimed_video_effects":shifted,"worst_mistimed_video_wer_harm":harm,"stratified_waveform_scores":strata}
        gates[run]={"speech":speech["passes_half_point_gain"],"video_adds_content":av["passes_half_point_gain"],"mistimed_harm_point_cap":harm<=.0025}
    write(root/"summary.json",{"runs":summary,"gates":gates,"both_slwm_seeds_speech":all(gates[r]["speech"] for r in ("G1-101","G1-103")),"both_slwm_seeds_video":all(gates[r]["video_adds_content"] and gates[r]["mistimed_harm_point_cap"] for r in ("G1-101","G1-103")),"independent_recognizer_parameters":37760256,"scope":"Waveform intelligibility proxy and frozen-rule timing rejection; group intervals remain important"})


if __name__=="__main__":main()
