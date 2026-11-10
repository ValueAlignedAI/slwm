"""Preregistered, unselected listening cases and actual PCM speech repairs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from exploration.video_repair_audio import decode_pcm,write_pcm,constrained_audio
from exploration.video_repair_infer import load_codecs,predict_mel,observed_features,support_masks,repair_windows
from models.slwm_restoration import fill_observed
from training.video_repair_train import load_model
from training.t2_prepare_cremad import sha256
from training.t2_real_av import write_json


def listening_cases(cfg):
    corpus=Path(cfg["corpus"]); cases=[]
    for split in ("crema_test","rav_test"):
        rows=json.loads((corpus/(split+".json")).read_text()); lengths=np.load(corpus/(split+"_lengths.npy"))
        for person in sorted({r["actor"] for r in rows}):
            pool=sorted([i for i,r in enumerate(rows) if r["actor"]==person and lengths[i]>=50],key=lambda i:rows[i]["sample_id"])
            if split=="rav_test": pool=[i for i in pool if rows[i]["sample_id"].split("-")[2]=="01"]
            if len(pool)<2: raise ValueError("Preregistered listening subset unavailable")
            picked=pool if split=="rav_test" else [pool[0],pool[-1]]
            if len(picked)!=2: raise ValueError("Expected two neutral statements")
            cases.extend({**rows[i],"split":split,"index":int(i),"length":int(lengths[i])} for i in picked)
    if len(cases)!=32: raise ValueError("Expected fixed 32-case gallery")
    return cases


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"]); out=root/"exploration"; out.mkdir(exist_ok=True)
    torch.set_num_threads(4); selection=json.loads((root/"selection.json").read_text()); codec,old,new=load_codecs(cfg,"cuda")
    # All 32 IDs are fixed by speaker/ID, before reading any signal or WER score.
    cases=listening_cases(cfg); write_json(out/"fixed_cases.json",cases)
    models={}
    for seed in ("61","62"):
        for kind in ("anchor","audio","primary_video"):
            models[kind+seed]=load_model(root/selection[kind][seed]/"best.pt")[0]
    arrays={s:np.load(Path(cfg["corpus"])/(s+"_mouth.npy"),mmap_mode="r") for s in ("crema_test","rav_test")}
    metadata={s:json.loads((Path(cfg["corpus"])/(s+".json")).read_text()) for s in arrays}
    records=[]
    for case in cases:
        folder=out/case["sample_id"]; folder.mkdir(exist_ok=True); pcm=decode_pcm(Path(case["raw_path"])); sound=pcm.astype(np.float32)/32768.; length=case["length"]
        start=(length-32)//2+12; missing=np.zeros(len(sound),bool); missing[start*640:(start+8)*640]=True
        mouth=np.asarray(arrays[case["split"]][case["index"],:length]).copy(); visible,unknown=support_masks(missing,length); windows=repair_windows(unknown,length)
        if len(windows)!=1: raise ValueError("Gallery requires exactly one fixed context")
        context=windows[0]; local_unknown=unknown[context*4:(context+32)*4]; corrupted=sound.copy(); corrupted[missing]=0
        write_pcm(folder/"reference.wav",sound); write_pcm(folder/"silence.wav",corrupted); np.save(folder/"missing_samples.npy",missing)
        observed=observed_features(sound,missing,new,"cuda"); y=torch.as_tensor(observed[:length*4].reshape(1,length,256),device="cuda")
        base=codec.observe(y,torch.as_tensor(visible[None],device="cuda"))["base"][0].cpu().numpy(); interp=observed.copy(); interp[:length*4][unknown]=base[unknown]
        write_pcm(folder/"interpolation.wav",constrained_audio(interp,sound,missing,new,device="cuda")); predictions={"interpolation":interp}
        # Evaluation-only ceiling. No clean spectrum is supplied to any model.
        from exploration.video_repair_audio import mel100
        oracle=(mel100(torch.as_tensor(sound,device="cuda")).cpu().numpy()-new["mean"])/new["scale"]
        write_pcm(folder/"oracle_spectrum.wav",constrained_audio(oracle,sound,missing,new,device="cuda"))
        # Wrong video: same person, a different utterance; fixed hash choice,
        # with its own central32 frames. No retiming to the target utterance.
        import hashlib
        rows=metadata[case["split"]]; pool=[i for i,r in enumerate(rows) if r["actor"]==case["actor"] and r["sentence"]!=case["sentence"]]
        wrong_id=pool[int(hashlib.sha256(case["sample_id"].encode()).hexdigest()[:8],16)%len(pool)]
        wrong_row=rows[wrong_id]; wrong_length=int(wrong_row["frames"]); wrong_start=(wrong_length-32)//2
        wrong=mouth.copy(); wrong[context:context+32]=arrays[case["split"]][wrong_id,wrong_start:wrong_start+32]
        shifted=mouth[np.clip(np.arange(length)+6,0,length-1)]
        static=np.repeat(mouth[start-1:start],length,axis=0)
        conditions={"anchor61":("anchor61",mouth,True),"anchor62":("anchor62",mouth,True),"audio61":("audio61",mouth,True),"audio62":("audio62",mouth,True),
            "video61":("primary_video61",mouth,True),"video62":("primary_video62",mouth,True),"wrong61":("primary_video61",wrong,True),
            "shift61":("primary_video61",shifted,True),"static61":("primary_video61",static,True),"no_video61":("primary_video61",mouth,False)}
        audits={}
        for name,(key,pixels,present) in conditions.items():
            predicted,audit=predict_mel(models[key],codec,new,sound,missing,pixels,present=present,device="cuda")
            wave=constrained_audio(predicted,sound,missing,new,device="cuda"); write_pcm(folder/(name+".wav"),wave); predictions[name]=predicted
            decoded=decode_pcm(folder/(name+".wav")); exact=bool(np.array_equal(decoded[~missing],pcm[~missing]))
            if not exact: raise ValueError("Observed samples changed: "+name)
            audits[name]={"observed_pcm_exact":exact,"gates":audit["gates"],"windows":audit["windows"]}
        np.testing.assert_array_equal(predictions["no_video61"],predictions["anchor61"])
        np.savez_compressed(folder/"predictions.npz",**predictions); np.save(folder/"visible_packets.npy",visible)
        from PIL import Image
        # Real recorded frames only, with an outline identifying the true gap.
        strip=np.concatenate([mouth[t] for t in range(start-2,start+10)],axis=1); Image.fromarray(strip).save(folder/"recorded_mouth.png")
        record={k:case[k] for k in ("sample_id","actor","split","transcript","raw_sha256")}; record.update({"folder":str(folder),"gap_seconds":[start*.04,(start+8)*.04],
            "conditions":["reference","silence","interpolation","oracle_spectrum"]+list(conditions),"wrong_video_sample_id":wrong_row["sample_id"],"conditions_audit":audits,
            "wave_hashes":{name:sha256(folder/(name+".wav")) for name in ["reference","silence","interpolation","oracle_spectrum"]+list(conditions)},"no_video_equals_anchor_exact":True,
            "phase":"64 iterations, observed-only phase and sample constraints; practical features erase hidden samples before transforms; oracle spectrum deliberately uses reference magnitudes for evaluation only"})
        records.append(record); write_json(out/"audio_cases.json",records); print(json.dumps({"rendered":len(records),"sample_id":case["sample_id"]}),flush=True)
    bundle=root/"bundle"; bundle.mkdir(exist_ok=True); parent=json.loads(Path(cfg["restoration_config"]).read_text())
    for source,name in ((Path(parent["old_corpus"])/"codec.npz","codec.npz"),(Path(parent["generation_corpus"])/"codec100.npz","codec100.npz")):
        shutil.copyfile(source,bundle/name)
    write_json(out/"render_manifest.json",{"cases":len(records),"selection_sha256":sha256(root/"selection.json"),"cases_sha256":sha256(out/"fixed_cases.json"),
        "source_hashes":{f:sha256(Path(f)) for f in ("exploration/video_repair_outputs.py","exploration/video_repair_audio.py","exploration/video_repair_infer.py")},
        "all_observed_pcm_exact":all(r["conditions_audit"][c]["observed_pcm_exact"] for r in records for c in r["conditions_audit"]),"all_no_video_equal_anchor":True})


if __name__=="__main__": main()
