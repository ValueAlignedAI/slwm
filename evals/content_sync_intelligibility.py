"""Independent pinned recognizer; fixed digit normalization, no grammar correction."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import torch
from evals.motion_audio_intelligibility import read_wave,word_error
from training.research_sources import file_sha
from training.t2_real_av import write_json,cluster_interval


def normalize(text):
    text=re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])"," ",text)
    names=("zero","one","two","three","four","five","six","seven","eight","nine")
    return re.sub(r"\b[0-9]\b",lambda m:names[int(m[0])],text)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/t2/content_sync.json"); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"])/"exploration"; cases=json.loads((root/"audio_cases.json").read_text()); torch.set_num_threads(4)
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    name="openai/whisper-tiny.en"; revision="87c7102498dcde7456f24cfd30239ca606ed9063"; processor=WhisperProcessor.from_pretrained(name,revision=revision,local_files_only=True); model=WhisperForConditionalGeneration.from_pretrained(name,revision=revision,local_files_only=True).cuda().half().eval(); results=[]
    for case in cases:
        record={k:case[k] for k in ("sample_id","actor","transcript")}; record["conditions"]={}
        for begin in range(0,len(case["conditions"]),4):
            names=case["conditions"][begin:begin+4]; inputs=processor([read_wave(Path(case["folder"])/(c+".wav")) for c in names],sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
            with torch.inference_mode(): tokens=model.generate(inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),max_length=64,num_beams=1,do_sample=False)
            texts=processor.batch_decode(tokens,skip_special_tokens=True)
            for condition,text in zip(names,texts): record["conditions"][condition]={"recognizer_text":text.strip(),"wer":word_error(normalize(case["transcript"]),normalize(text)),"raw_wer":word_error(case["transcript"],text),"wave_sha256":file_sha(Path(case["folder"])/(condition+".wav"))}
        results.append(record); write_json(root/"intelligibility_progress.json",results); print(json.dumps({"recognized":len(results),"sample_id":case["sample_id"],"audio71":record["conditions"]["audio71"]["wer"],"video71":record["conditions"]["video71"]["wer"]}),flush=True)
    means={c:float(np.mean([r["conditions"][c]["wer"] for r in results])) for c in cases[0]["conditions"]}; comparisons={}
    pairs=[(f"video{s}",c) for s in ("71","72") for c in (f"audio{s}","silence")]+[(f"content{s}",f"plain{s}") for s in ("71","72")]+[("video71",c) for c in ("raw_video71","wrong71","shift71")]
    for new,old in pairs:
        delta=np.array([r["conditions"][old]["wer"]-r["conditions"][new]["wer"] for r in results]); interval=cluster_interval(delta,[r["actor"] for r in results]); relative=float(delta.mean()/max(means[old],1e-9)); comparisons[new+"_vs_"+old]={"absolute_wer_reduction":float(delta.mean()),"relative_wer_reduction":relative,"person_cluster_95_interval":interval,"passes_5percent_positive_interval":bool(relative>=.05 and interval[0]>0)}
    passed=all(comparisons[f"video{s}_vs_{c}"]["passes_5percent_positive_interval"] for s in ("71","72") for c in (f"audio{s}","silence"))
    write_json(root/"intelligibility.json",{"recognizer":name,"revision":revision,"external_evaluation_parameters":sum(p.numel() for p in model.parameters()),"evaluation_only":True,"scope":"Fallible independent recognizer on all32 fixed fresh-person composition-held-out clips; not a human intelligibility study",
        "normalization":"Case/punctuation per word_error, separate adjacent letters/digits, convert single digits0..9 to English words in reference and hypothesis; no grammar or homophone correction","mean_clip_wer":means,"comparisons":comparisons,"practical_gate_passes":passed,"examples":results,"source_sha256":file_sha(Path(__file__))})


if __name__=="__main__": main()
