"""Usable content/sync repair from ordinary media; no hidden audio or text assumed."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import numpy as np
import torch
from exploration.video_repair_audio import decode_pcm,write_pcm,constrained_audio
from exploration.video_repair_infer import observed_features,pad_subframes,support_masks,repair_windows,load_codecs
from training.content_sync_train import load_model
from training.grid_content_data import decode_record
from training.research_sources import file_sha
from training.t2_real_av import write_json


def predict_sequence(model,kind,codec,new,sound,missing,mouth,threshold=None,present=True,text_ids=None,device="cpu"):
    """Globally erased waveform -> normalized spectrum and selected word/gate probes.

    True masks are applied before the first transform and shared by every
    window. Supplied timed text is explicit and supported only by a base model.
    """
    if text_ids is not None and kind!="base": raise ValueError("Use a base checkpoint for explicit timed-text conditioning")
    # Match preparation/calibration: fixed CPU FFT and stored float16 feature
    # precision, before interpolation. Reconstruction itself stays float32.
    features=observed_features(sound,missing,new,"cpu").astype(np.float16).astype(np.float32)
    packets=min(128,(len(features)+3)//4,len(mouth)); features=pad_subframes(features,packets)
    if packets<32: raise ValueError("At least1.28s of context is required")
    if missing.any() and np.flatnonzero(missing)[-1]>=packets*640: raise ValueError("Recorded video/context does not cover the gap")
    visible,unknown=support_masks(missing,packets); starts=repair_windows(unknown,packets); totals=np.zeros((packets*4,64),np.float32); weight=np.zeros(packets*4,np.float32); records=[]
    amp=torch.autocast("cuda",dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()
    with torch.inference_mode(),amp:
        for start in starts:
            stop=start+32; obs=codec.observe(torch.as_tensor(features[start*4:stop*4].reshape(1,32,256),device=device),torch.as_tensor(visible[None,start:stop],device=device))
            pixels=torch.as_tensor(mouth[None,start:stop],device=device); mask=torch.as_tensor(unknown[None,start*4:stop*4],device=device)
            if kind=="base": result=model(obs,torch.as_tensor(text_ids[None,start:stop],device=device) if text_ids is not None else None)
            else: result=model(obs,pixels,mask,torch.tensor([present],device=device),threshold)
            totals[start*4:stop*4]+=result["prediction"][0].float().cpu().numpy(); weight[start*4:stop*4]+=1
            record={"start_packet":start,"unit_logits":result["unit_logits"][0].float().cpu().numpy().tolist()}
            if "quality" in result: record.update({"quality":float(result["quality"][0]),"predicted_offset":int(result["predicted_offset"][0]),"accepted":bool(result.get("accepted",torch.tensor([present],device=device))[0])})
            records.append(record)
    pred=features.copy(); use=unknown&(weight>0); pred[:packets*4][use]=totals[use]/weight[use,None]
    if (unknown&(weight==0)).any(): raise ValueError("Unsupported long/edge gap")
    return pred,{"windows":records,"visible":visible,"unknown":unknown}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--media",help="Original GRID media; uses shared audio/video PTS")
    p.add_argument("--audio",help="Already synchronized waveform; audio-only if no media supplied"); p.add_argument("--gaps",required=True); p.add_argument("--codec-dir"); p.add_argument("--calibration")
    p.add_argument("--timed-text",help="Explicit JSON list of {start,end,word}; base checkpoints only"); p.add_argument("--probe-words",action="store_true"); p.add_argument("--device",choices=("cpu","cuda"),default="cpu"); p.add_argument("--output",required=True); a=p.parse_args()
    if bool(a.media)==bool(a.audio): p.error("Choose either original GRID media or an audio waveform")
    torch.set_num_threads(4); model,ck=load_model(a.checkpoint,a.device); cfg=ck["config"]; dc=json.loads(Path(cfg["data_config"]).read_text()); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    codec,old,new=load_codecs(cfg,a.device,a.codec_dir)
    if a.media:
        mouth,pcm,timing=decode_record(Path(a.media),dc)
    else:
        pcm=decode_pcm(Path(a.audio)); mouth=np.zeros((min(128,(len(pcm)+639)//640),64,96),np.uint8); timing={"assumption":"Waveform time zero supplied by caller; no video"}
    sound=pcm.astype(np.float32)/32768.; missing=np.zeros(len(sound),bool)
    for span in filter(None,a.gaps.split(",")):
        begin,end=map(float,span.split(":"))
        if begin<0 or end<=begin or end-begin>.481 or end*16000>len(sound): p.error("Each gap must fit the recording and be at most480ms")
        missing[int(round(begin*16000)):int(round(end*16000))]=True
    threshold=None
    if ck["kind"]!="base" and ck["run"]["arm"]!="audio":
        if not a.calibration: p.error("A video checkpoint needs its frozen calibration JSON")
        cal=json.loads(Path(a.calibration).read_text()); cal=cal[ck["run"]["id"]] if "threshold" not in cal else cal
        if file_sha(a.checkpoint)!=cal["checkpoint_sha256"]: p.error("Calibration belongs to a different checkpoint")
        threshold=cal["threshold"]
    text=None
    if a.timed_text:
        if ck["kind"]!="base": p.error("Timed text is an explicitly separate base-model mode")
        vocab=json.loads((Path(cfg["corpus"])/"vocabulary.json").read_text()) if a.codec_dir is None else json.loads((Path(a.codec_dir)/"vocabulary.json").read_text()); text=np.full(len(mouth),-100,np.int64)
        for span in json.loads(Path(a.timed_text).read_text()):
            if span["word"].lower() not in vocab: p.error("Word outside the trained closed vocabulary: "+span["word"])
            text[(np.arange(len(mouth))*.04>=span["start"])&(np.arange(len(mouth))*.04<span["end"])]=vocab[span["word"].lower()]
    predicted,audit=predict_sequence(model,ck["kind"],codec,new,sound,missing,mouth,threshold,bool(a.media),text,a.device)
    repaired=constrained_audio(predicted,sound,missing,new,device=a.device); write_pcm(out/"repaired.wav",repaired); np.save(out/"missing_samples.npy",missing); np.save(out/"predicted_mel.npy",predicted)
    actual=decode_pcm(out/"repaired.wav"); exact=bool(np.array_equal(actual[~missing],pcm[~missing])); assert exact
    if a.probe_words: write_json(out/"word_and_gate_probe.json",audit["windows"])
    write_json(out/"inference.json",{"run":ck["run"],"kind":ck["kind"],"checkpoint_sha256":file_sha(a.checkpoint),"input_sha256":file_sha(a.media or a.audio),"threshold":threshold,"text_supplied":bool(a.timed_text),"timing":timing,
        "observed_pcm_exact":exact,"hidden_samples":int(missing.sum()),"windows":[{k:v for k,v in r.items() if k!="unit_logits"} for r in audit["windows"]],"hidden_audio_used":False,"recognizer_used":False,"scope":"Offline true short-gap repair; closed GRID text codes, no free-form language or policy claim"})
    print(json.dumps({"output":str(out/"repaired.wav"),"observed_pcm_exact":exact}))


if __name__=="__main__": main()
