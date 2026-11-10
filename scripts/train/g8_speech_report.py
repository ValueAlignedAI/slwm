"""Verify and summarize the two independently locked waveform panels and the released 48-second codec example."""
import json
import hashlib
from pathlib import Path
import subprocess
import wave
from training.g8_models import read,G8_DATA
from scripts.train.g6_artifacts import roots
from scripts.train.g7_lock_heads import sha
from training.g7_units import put


def compact(metric):return {k:v for k,v in metric.items() if k not in ("hypotheses","per_clip_edits")}


def collect():
    a,_=roots("g8");out=a/"public";out.mkdir(exist_ok=True);reports={}
    for label,dataset,kind in (("product","speech","speech-audit"),("allocated","speech-rate-confirm","rate-audit")):
        root=a/kind;lock=read(root/"lock.json");all_cases={}
        for family in ("libri","crema"):
            folder=root/family;oracle=read(folder/"oracles/scores.json")
            source=read(G8_DATA/dataset/"confirm"/family/"card.json");ids=oracle["ids"]
            if len(ids)!=source["eligible"] or len(set(ids))!=len(ids):raise ValueError("Lost or duplicated waveform cases")
            controls={k:compact(v) for k,v in oracle["conditions"].items()}
            variants={}
            for seed in (711,712):
                for core in ("hybrid","gpt2"):
                    tag=f"{core}-{seed}";d=read(folder/tag/"scores.json");expected=lock.get("heads",lock.get("predictors",{}))[tag]
                    if d["head"]!=expected or d["ids"]!=ids:raise ValueError("Unpaired waveform comparison")
                    variants[tag]={"weights":d["head"],"conditions":{k:compact(v) for k,v in d["conditions"].items()},"contrasts":d["paired_wer_effects"]}
            modes=("text","masked-audio","repair")+(("video",) if family=="crema" else ())
            gates={}
            for core in ("hybrid","gpt2"):
                for mode in modes:
                    vals=[variants[f"{core}-{seed}"]["contrasts"][mode+"-minus-"+control]["speaker_ci95"][1] for seed in (711,712) for control in ("wrong","erased")]
                    gates[core+"-"+mode]=all(x<0 for x in vals)
            all_cases[family]={"clips":source["eligible"],"speakers":source["people"],"excluded":source["excluded"],"raw_source_selection":source["selection"],
                "controls":controls,"variants":variants,"correct_reliably_better_than_wrong_and_erased_at_both_seeds":gates,
                "rate_contrasts":oracle.get("allocated_minus_equal",oracle.get("acoustic_minus_semantic_wer"))}
        reports[label]={"condition_lock_sha256":sha(root/"lock.json"),"pinned_weights":lock,"cases":all_cases}
    reports["product_codec"]=read(a/"acoustic-codec/complete.json");reports["allocated_codec"]=read(a/"rate-codec/complete.json")
    put(out/"speech.json",reports)
    example=a/"rate-example";source=read(example/"source.json");items={}
    for b in (48,192):
        packet=example/f"chapter-{b}.slwma"
        with packet.open("rb") as f:
            if f.readline()!=b"SLWMAC1\n":raise ValueError("Missing codec packet")
            meta=json.loads(f.readline());payload=f.read()
        if meta["weights_sha256"]!=reports["allocated_codec"]["weights"] or hashlib.sha256(payload).hexdigest()!=meta["payload_sha256"]:raise ValueError("Example changed")
        with wave.open(str(example/f"chapter-{b}.wav"),"rb") as wf:
            if wf.getframerate()!=24000 or wf.getnframes()!=meta["samples"] or wf.getnchannels()!=1:raise ValueError("Incorrect round-trip physical duration")
        ogg=example/f"chapter-{b}.ogg"
        duration=float(subprocess.check_output(["ffprobe","-v","error","-show_entries","format=duration","-of","csv=p=0",str(ogg)]))
        if abs(duration-source["seconds"])>.1 or meta["frame_pairs"]*b!=len(payload)*8:raise ValueError("Wrong example payload or duration")
        items[str(b)]={"duration_seconds":duration,"payload_bytes":len(payload),"stream_bytes_with_header":packet.stat().st_size,"packet_sha256":sha(packet),"decoded_pcm_file_sha256":sha(example/f"chapter-{b}.wav"),"preview_sha256":sha(ogg)}
    for family,view in (("libri","text"),("crema","video")):
        f=a/"rate-audit"/family/"hybrid-711"/(view+"-correct.wav");dest=example/(view+"-conditioned.ogg")
        subprocess.run(["ffmpeg","-y","-v","error","-i",str(f),"-c:a","libopus","-b:a","48k",str(dest)],check=True)
        scores=read(a/"rate-audit"/family/"hybrid-711/scores.json");source_id=scores["ids"][0]
        oracle=read(a/"rate-audit"/family/"oracles/scores.json")
        items[view+"-conditioned"]={"source_id":source_id,"reference":oracle["references"][0],"frozen_recognizer":scores["conditions"][view+"-correct"]["hypotheses"][0],
            "preview_sha256":sha(dest),"original_prediction_wav_sha256":sha(f),"core":"hybrid-711, G6 parent seed 511"}
    doc={"source":source,"examples":items,"weights":reports["allocated_codec"]["weights"],"known_waveform_samples_preserved_only_by_gap_repair":True}
    put(example/"verified.json",doc);put(out/"speech-examples.json",doc);print("All 20 scored waveform panels and long file round trips verified",flush=True)


if __name__=="__main__":collect()
