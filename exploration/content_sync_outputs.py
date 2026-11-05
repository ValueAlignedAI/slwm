"""Actual repaired PCM and word probes on all32 pre-fixed fresh-speaker clips."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from exploration.content_sync_infer import predict_sequence
from exploration.video_repair_infer import observed_features,pad_subframes,support_masks,repair_windows,load_codecs
from exploration.video_repair_audio import write_pcm,decode_pcm,constrained_audio,mel100
from training.content_sync_train import load_model
from training.research_sources import file_sha
from training.t2_real_av import write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/t2/content_sync.json"); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"]); out=root/"exploration"; out.mkdir(exist_ok=True); torch.set_num_threads(4)
    calibration=json.loads((root/"calibration.json").read_text()); ids=json.loads((root/"listening_ids.json").read_text()); rows=json.loads((Path(cfg["corpus"])/"test.json").read_text()); byid={r["sample_id"]:r for r in rows}; vocab=json.loads((Path(cfg["corpus"])/"vocabulary.json").read_text()); inverse={v:k for k,v in vocab.items()}
    models={}; checkpoints={}
    for run in cfg["base_runs"]+cfg["candidate_runs"][:4]:
        model,ck=load_model(root/run["id"]/"best.pt"); models[run["id"]]=model; checkpoints[run["id"]]=ck
    codec,old,new=load_codecs(cfg,"cuda"); cases=[]
    for sample_id in ids:
        row=byid[sample_id]; folder=out/sample_id; folder.mkdir(exist_ok=True)
        with np.load(row["decoded_path"],allow_pickle=False) as d: pcm=d["pcm"].copy(); mouth=d["mouth"].copy(); labels=d["labels"].copy()
        sound=pcm.astype(np.float32)/32768.; length=len(mouth); start=(length-32)//2+12; missing=np.zeros(len(sound),bool); missing[start*640:(start+8)*640]=True; visible,unknown=support_masks(missing,length); context=repair_windows(unknown,length)[0]
        write_pcm(folder/"reference.wav",sound); erased=sound.copy(); erased[missing]=0; write_pcm(folder/"silence.wav",erased); np.save(folder/"missing_samples.npy",missing)
        observed=pad_subframes(observed_features(sound,missing,new,"cuda"),length); y=torch.as_tensor(observed[:length*4].reshape(1,length,256),device="cuda"); interpolated=observed.copy(); base=codec.observe(y,torch.as_tensor(visible[None],device="cuda"))["base"][0].cpu().numpy(); interpolated[:length*4][unknown]=base[unknown]
        write_pcm(folder/"interpolation.wav",constrained_audio(interpolated,sound,missing,new,device="cuda")); oracle=(mel100(torch.as_tensor(sound,device="cuda")).cpu().numpy()-new["mean"])/new["scale"]; write_pcm(folder/"oracle_spectrum.wav",constrained_audio(oracle,sound,missing,new,device="cuda"))
        pool=[r for r in rows if r["actor"]==row["actor"] and r["sentence"]!=row["sentence"]]; import hashlib
        other=pool[int(hashlib.sha256(sample_id.encode()).hexdigest()[:8],16)%len(pool)]
        with np.load(other["decoded_path"],allow_pickle=False) as d: other_mouth=d["mouth"].copy()
        wrong=mouth.copy(); os=(len(other_mouth)-32)//2; wrong[context:context+32]=other_mouth[os:os+32]
        shifted=mouth[np.clip(np.arange(length)+6,0,length-1)]; static=np.repeat(mouth[start-1:start],length,axis=0)
        specs={"plain71":("EXP-T2-701",mouth,None,True,None),"content71":("EXP-T2-702",mouth,None,True,None),"plain72":("EXP-T2-703",mouth,None,True,None),"content72":("EXP-T2-704",mouth,None,True,None),
            "timed_text71":("EXP-T2-702",mouth,None,True,labels),"audio71":("EXP-T2-711",mouth,None,True,None),"audio72":("EXP-T2-712",mouth,None,True,None),
            "raw_video71":("EXP-T2-713",mouth,None,True,None),"video71":("EXP-T2-713",mouth,calibration["EXP-T2-713"]["threshold"],True,None),"video72":("EXP-T2-714",mouth,calibration["EXP-T2-714"]["threshold"],True,None),
            "wrong71":("EXP-T2-713",wrong,calibration["EXP-T2-713"]["threshold"],True,None),"raw_wrong71":("EXP-T2-713",wrong,None,True,None),"shift71":("EXP-T2-713",shifted,calibration["EXP-T2-713"]["threshold"],True,None),"raw_shift71":("EXP-T2-713",shifted,None,True,None),
            "static71":("EXP-T2-713",static,calibration["EXP-T2-713"]["threshold"],True,None),"missing_video71":("EXP-T2-713",mouth,calibration["EXP-T2-713"]["threshold"],False,None)}
        predictions={"interpolation":interpolated}; audit={}; probes={}
        for name,(run,pixels,threshold,present,text) in specs.items():
            predicted,meta=predict_sequence(models[run],checkpoints[run]["kind"],codec,new,sound,missing,pixels,threshold,present,text,"cuda"); wave=constrained_audio(predicted,sound,missing,new,device="cuda"); write_pcm(folder/(name+".wav"),wave); predictions[name]=predicted
            actual=decode_pcm(folder/(name+".wav")); exact=bool(np.array_equal(actual[~missing],pcm[~missing])); assert exact
            audit[name]={"observed_pcm_exact":exact,"windows":[{k:v for k,v in m.items() if k!="unit_logits"} for m in meta["windows"]]}
            units=np.asarray(meta["windows"][0]["unit_logits"]).argmax(-1); words=[]
            for u in units:
                word=inverse[int(u)]
                if word not in {"sil","sp"} and (not words or word!=words[-1]): words.append(word)
            probes[name]={"predicted_context_words":" ".join(words),"scope":"Forced diagnostic closed-word readout in1.28s context; model predictions, not external ASR and not a committed policy output"}
        np.testing.assert_array_equal(predictions["missing_video71"],predictions["audio71"])
        np.savez_compressed(folder/"predictions.npz",**predictions); write_json(folder/"word_probes.json",probes)
        from PIL import Image
        Image.fromarray(np.concatenate([mouth[t] for t in range(start-2,start+10)],1)).save(folder/"recorded_video.png")
        conditions=["reference","silence","interpolation","oracle_spectrum"]+list(specs)
        case={"sample_id":sample_id,"actor":row["actor"],"transcript":row["transcript"],"folder":str(folder),"gap_seconds":[start*.04,(start+8)*.04],"conditions":conditions,"audit":audit,"raw_sha256":row["raw_sha256"],"wave_hashes":{name:file_sha(folder/(name+".wav")) for name in conditions},"wrong_source_id":other["sample_id"],"timing":row["timing"],"no_video_equals_strong_audio":True,
            "oracle_scope":"True spectrum is supplied only to an explicitly labeled reconstruction diagnostic; timed text is explicit oracle conditioning; neither enters practical audio/video inference"}
        cases.append(case); write_json(out/"audio_cases.json",cases); print(json.dumps({"rendered":len(cases),"sample_id":sample_id}),flush=True)
    bundle=root/"bundle"; bundle.mkdir(exist_ok=True); parent=json.loads(Path(cfg["restoration_config"]).read_text())
    for path,name in ((Path(parent["old_corpus"])/"codec.npz","codec.npz"),(Path(parent["generation_corpus"])/"codec100.npz","codec100.npz"),(Path(cfg["corpus"])/"vocabulary.json","vocabulary.json")):
        shutil.copyfile(path,bundle/name)
    write_json(out/"render_manifest.json",{"cases":len(cases),"waveforms":sum(len(c["conditions"]) for c in cases),"fixed_ids_sha256":file_sha(root/"listening_ids.json"),"calibration_sha256":file_sha(root/"calibration.json"),"all_observed_pcm_exact":True,"all_missing_video_equal_strong_audio":True,
        "reference_clock":"Video PTS aligned decoded PCM; initial source audio trim/pad is recorded per clip","source_hashes":{name:file_sha(name) for name in ("exploration/content_sync_outputs.py","exploration/content_sync_infer.py","exploration/video_repair_audio.py","exploration/video_repair_infer.py")}})


if __name__=="__main__": main()
