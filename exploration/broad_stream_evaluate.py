"""Registered broad speech, independent shared fields and >30s generation probes."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torchaudio
from training.broad_stream_train import load_model
from training.broad_batch import StreamCorpus,predict
from training.stream_codec import PACKET_SECONDS,estimated_text,observed_mel,load_vocoder
from training.research_sources import file_sha
from training.t2_real_av import write_json
from exploration.stream_media import read_audio,write_audio,rollout,synthesize,restore_known,decode_video,encode_video

CONFIG="configs/t2/broad_stream.json"
MAIN=("EXP-T2-801","EXP-T2-802","EXP-T2-803","EXP-T2-804")


def gap_interval(row,seconds,width):
    eligible=[s for s in row["spans"] if s[0]>=.2*seconds and s[1]<=.8*seconds]
    word=min(eligible,key=lambda s:(-(s[1]-s[0]),s[0])) if eligible else None; center=(word[0]+word[1])/2 if word else seconds/2
    return max(0.,center-width/2),min(seconds,center+width/2)


def vocabulary_bundle(cfg):
    root=Path(cfg["artifacts"])/"inference_bundle"; root.mkdir(exist_ok=True); corpus=Path(cfg["corpus"])
    for name in ("codec.npz","duration_model.json","vocabulary.json","dataset_card.json"):
        shutil.copyfile(corpus/name,root/name)
    row=json.loads((corpus/"ami_train/rows.json").read_text())[0]; index=int(round(100/PACKET_SECONDS)); pixels=np.load(corpus/"ami_train"/(row["id"]+"_pixels.npy"),mmap_mode="r")[index]; code=np.load(corpus/"ami_train"/(row["id"]+"_video.npy"),mmap_mode="r")[index]; np.save(root/"appearance_pixels.npy",pixels); np.save(root/"appearance_code.npy",code); write_json(root/"appearance.json",{"training_id":row["id"],"source_packet":index,"source_seconds":index*PACKET_SECONDS,"provided_static_appearance":True,"heldout_data_used":False}); return root


def freeze(cfg):
    root=Path(cfg["artifacts"])/"evaluation"; root.mkdir(exist_ok=True); assert (Path(cfg["artifacts"])/"selection.json").exists(); assert Path("experiments/policy/continuous-sensing-20261105/selection.json").exists()
    names=(CONFIG,"docs/research/broad_stream_evaluation_20261105.md","exploration/broad_stream_evaluate.py","exploration/stream_media.py","exploration/broad_stream_infer.py","tests/test_stream_media.py")
    for name in names:
        target=root/"source_snapshot"/name; target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and file_sha(target)!=file_sha(name): raise ValueError("Registered evaluation source changed: "+name)
        if not target.exists(): shutil.copyfile(name,target)
    write_json(root/"source_hashes.json",{n:file_sha(n) for n in names}); return root


def speech(cfg,root):
    codec=dict(np.load(Path(cfg["corpus"])/"codec.npz")); mean=np.tile(codec["audio_mean"],4); scale=np.tile(codec["audio_scale"],4); vocoder=load_vocoder("cuda"); models={r["id"]:load_model(Path(cfg["artifacts"])/r["id"]/"best.pt")[0].requires_grad_(False) for r in cfg["runs"]}; all_records=[]
    for split in ("test-clean","test-other"):
        rows=json.loads((Path(cfg["corpus"])/("libri_"+split)/"rows.json").read_text())
        for row in rows:
            sound=read_audio(row["raw_path"]); packets=(1+len(sound)//256)//4; true,_,_=observed_mel(sound,np.zeros(len(sound),bool),packets); target=(true-mean)/scale
            for width in (cfg["primary_gap_seconds"],cfg["secondary_gap_seconds"]):
                folder=root/"speech"/split/(row["id"]+"_"+str(round(width*1000))); folder.mkdir(parents=True,exist_ok=True)
                if (folder/"case.json").exists(): all_records.append(json.loads((folder/"case.json").read_text())); continue
                begin,end=gap_interval(row,len(sound)/24000,width); missing=np.zeros(len(sound),bool); missing[round(begin*24000):round(end*24000)]=True; physical,fine,visible=observed_mel(sound,missing,packets); normalized=(physical-mean)/scale; mel={"interpolated_mel":normalized,"oracle_mel":target}; waves={"reference":sound,"silence":np.where(missing,0.,sound),"linear_wave":restore_known(np.zeros_like(sound),sound,missing,fade_samples=0)}
                # Pure linear waveform baseline is independent of the learned head.
                waves["linear_wave"][missing]=np.interp(np.flatnonzero(missing),np.flatnonzero(~missing),sound[~missing]); field_text={}; timing={}
                for name,model in models.items():
                    proposal=rollout(model,normalized,np.zeros((packets,64),np.float32),np.zeros(packets,np.int64),np.zeros(64,np.float32),"repair",visible=visible,device="cuda",reset_memory=name=="EXP-T2-808"); mel[name]=proposal["mel"]; field_text[name]=proposal["text"]; timing[name]=sum(t["compute_seconds"] for t in proposal["trace"])
                metrics={}; mask=((np.arange(packets*4)*256/24000)>=begin)&((np.arange(packets*4)*256/24000)<end)
                for name,features in mel.items():
                    generated,renderer=synthesize(features,codec,vocoder,"cuda"); waves[name]=restore_known(generated,sound,missing); metrics[name]={"true_gap_mel_mae":float(np.abs(features.reshape(-1,100)[mask]-target.reshape(-1,100)[mask]).mean()),"renderer":renderer}
                for name,wave in waves.items():
                    assert np.array_equal(np.rint(wave[~missing]*32768).astype("<i2"),np.rint(sound[~missing]*32768).astype("<i2")); write_audio(folder/(name+".wav"),wave)
                np.save(folder/"missing_samples.npy",missing); np.savez_compressed(folder/"mels.npz",**{k:v.astype(np.float16) for k,v in mel.items()}); record={"id":row["id"],"actor":row["actor"],"split":split,"gap_seconds":width,"gap_interval":[begin,end],"transcript":row["transcript"],"folder":str(folder),"conditions":list(waves),"field_text":field_text,"metrics":metrics,"compute_seconds":timing,"observed_pcm_exact":True,"hidden_samples_erased_before_fft":True,"source_phase_used":False,"raw_sha256":row["raw_sha256"]}; write_json(folder/"case.json",record); all_records.append(record); write_json(root/"speech_cases.json",all_records); print(json.dumps({"speech_case":len(all_records),"id":row["id"],"gap":width}),flush=True)
    write_json(root/"speech_cases.json",all_records)


def fields(cfg,root):
    data=StreamCorpus(cfg,"test_ami"); name=next(iter(data.rows)); plans=[]
    for i,row in enumerate(data.rows[name]):
        for j in range(16):
            start=int(round((40+8*j)/PACKET_SECONDS)); begin=start*PACKET_SECONDS; end=(start+128)*PACKET_SECONDS; count=sum(a>=begin and z<=end for a,z,_ in row["spans"])
            if count>=4: plans.append({"name":name,"index":i,"start":start,"wrong":start+int(round(8/PACKET_SECONDS)),"mode":"audio","video_in_repair":False,"gap_start":48,"gap_length":4,"drop_chunks":[False]*4,"include_style":False})
    write_json(root/"field_plans.json",plans)
    for run in cfg["runs"]:
        target=root/(run["id"]+"_fields.json")
        if target.exists(): continue
        model,ck=load_model(Path(cfg["artifacts"])/run["id"]/"best.pt"); model.requires_grad_(False); embeddings={k:[] for k in ("audio","video","text")}; entries=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for at in range(0,len(plans),32):
                part=plans[at:at+32]; batch=data.batch(part)
                for mode in embeddings:
                    out=predict(model,batch,run.get("reset_memory",False),view=mode); embeddings[mode].append(out["embedding"].cpu().numpy())
                    if mode=="audio":
                        both={**batch,"a_mask":batch["valid"],"v_mask":batch["valid"],"t_mask":torch.zeros_like(batch["valid"]),"appearance":torch.zeros_like(batch["appearance"])}; forecast=predict(model,both,run.get("reset_memory",False))
                        for i,p in enumerate(part):
                            row=data.rows[p["name"]][p["index"]]; errors=[]; base=[]
                            for k in range(3):
                                begin=(k+1)*32+2; pred=forecast["future"][i,k]; truth=batch["audio"][i,begin:begin+8]; vision=batch["video"][i,begin:begin+8]; errors.append([float((pred[:,:400]-truth).abs().mean()),float((pred[:,400:]-vision).square().mean())]); base.append([float((batch["audio"][i,(k+1)*32-1]-truth).abs().mean()),float((batch["video"][i,(k+1)*32-1]-vision).square().mean())])
                            valid=batch["phones"][i]>1; correct=(out["phone_logits"][i].argmax(-1)==batch["phones"][i]); entries.append({"id":row["id"],"actor":row["actor"],"start_packet":p["start"],"phone_accuracy_active":float(correct[valid].float().mean()) if valid.any() else None,"future_errors":np.mean(errors,0).tolist(),"persistence_errors":np.mean(base,0).tolist()})
        arrays={k:np.concatenate(v) for k,v in embeddings.items()}; np.savez_compressed(root/(run["id"]+"_embeddings.npz"),**arrays); retrieval=[]
        for actor in sorted({r["actor"] for r in entries}):
            ids=np.asarray([i for i,e in enumerate(entries) if e["actor"]==actor]); n=len(ids)
            for a,b in (("audio","video"),("audio","text"),("video","text")):
                score=arrays[a][ids]@arrays[b][ids].T; retrieval.append({"actor":actor,"direction":a+"_to_"+b,"windows":n,"chance":1/n,"accuracy":float(np.mean(score.argmax(1)==np.arange(n))),"correct_similarity":float(score.diagonal().mean()),"same_person_wrong_similarity":float(score[~np.eye(n,dtype=bool)].mean()) if n>1 else None})
        write_json(target,{"parameters":ck["parameters"],"windows":entries,"retrieval":retrieval}); print(json.dumps({"field_evaluated":run["id"],"windows":len(entries)}),flush=True); del model; torch.cuda.empty_cache()


def long_media(cfg,root):
    corpus=Path(cfg["corpus"]); codec=dict(np.load(corpus/"codec.npz")); bundle=vocabulary_bundle(cfg); appearance=np.load(bundle/"appearance_code.npy"); picture=np.load(bundle/"appearance_pixels.npy"); vocoder=load_vocoder("cuda"); duration=json.loads((corpus/"duration_model.json").read_text()); rows=json.loads((corpus/"ami_test/rows.json").read_text()); mean=np.tile(codec["audio_mean"],4); scale=np.tile(codec["audio_scale"],4); records=[]; packets=round(cfg["long_probe_seconds"]/PACKET_SECONDS); first=round(cfg["long_probe_start"]/PACKET_SECONDS)
    for row in rows:
        folder=root/"long_inputs"/row["id"]; folder.mkdir(parents=True,exist_ok=True); sound=read_audio(row["raw_path"],cfg["long_probe_start"],cfg["long_probe_seconds"]); sound=np.pad(sound,(0,max(0,packets*1024-len(sound))))[:packets*1024]; physical,_,_=observed_mel(sound,np.zeros(len(sound),bool),packets); audio=(physical-mean)/scale; video=np.asarray(np.load(corpus/"ami_test"/(row["id"]+"_video.npy"),mmap_mode="r")[first:first+packets],np.float32); pixels=np.asarray(np.load(corpus/"ami_test"/(row["id"]+"_pixels.npy"),mmap_mode="r")[first:first+packets]); reference=" ".join(w for a,z,w in row["spans"] if a>=first*PACKET_SECONDS and z<=(first+packets)*PACKET_SECONDS); write_audio(folder/"input.wav",sound)
        if not(folder/"input.mp4").exists(): encode_video(folder/"input.mp4",pixels,folder/"input.wav")
        manifest={"id":row["id"],"actor":row["actor"],"original_audio":row["raw_path"],"original_video":row["video_path"],"original_audio_seconds":row["audio_duration"],"original_video_seconds":row["duration"],"raw_sha256":row["raw_sha256"],"video_sha256":row["video_decode"]["raw_video_sha256"],"segment_start_seconds":first*PACKET_SECONDS,"seconds":packets*PACKET_SECONDS,"transcript":reference,"appearance_provided":True}; write_json(folder/"input.json",manifest)
        for runid in MAIN:
            model,_=load_model(Path(cfg["artifacts"])/runid/"best.pt"); model.requires_grad_(False)
            for mode in ("audio","video"):
                controls=("normal","reset","prefix","wrong_time") if runid in MAIN[:2] else ("normal",)
                for control in controls:
                    outdir=root/"long"/row["id"]/runid/(mode+"_"+control); outdir.mkdir(parents=True,exist_ok=True)
                    if (outdir/"case.json").exists(): records.append(json.loads((outdir/"case.json").read_text())); continue
                    aa=audio if mode=="audio" else np.zeros_like(audio); vv=video if mode=="video" else np.zeros_like(video)
                    if control=="wrong_time":
                        shift=round(80/PACKET_SECONDS)
                        if mode=="audio": mel=np.load(corpus/"ami_test/mel.npy",mmap_mode="r"); aa=(np.asarray(mel[row["offset"]+first+shift:row["offset"]+first+shift+packets],np.float32)-mean)/scale
                        else: vv=np.asarray(np.load(corpus/"ami_test"/(row["id"]+"_video.npy"),mmap_mode="r")[first+shift:first+shift+packets],np.float32)
                    proposal=rollout(model,aa,vv,np.zeros(packets,np.int64),appearance if mode=="audio" else np.zeros(64,np.float32),mode,prefix_chunks=8 if control=="prefix" else None,reset_memory=control=="reset",device="cuda"); np.savez_compressed(outdir/"proposals.npz",mel=proposal["mel"].astype(np.float16),video=proposal["video"].astype(np.float16)); write_json(outdir/"trace.json",proposal["trace"]); (outdir/"field_text.txt").write_text(proposal["text"]+"\n")
                    if mode=="audio" and control!="prefix": wave=sound; renderer={"soundtrack":"Original input audio; only video motion generated"}
                    else: wave,renderer=synthesize(proposal["mel"],codec,vocoder,"cuda")
                    wav=write_audio(outdir/"output.wav",wave); frames=pixels if mode=="video" else decode_video(proposal["video"],codec,picture,appearance); meta=encode_video(outdir/"output.mp4",frames,outdir/"output.wav"); case={**manifest,"run":runid,"mode":mode,"control":control,"folder":str(outdir),"waveform":wav,"video":meta,"renderer":renderer,"source_audio_decoded_by_model":mode=="audio","source_phase_used":False,"reference_text_supplied":False,"field_text":proposal["text"],"normalized_mel_mae":float(np.abs(proposal["mel"]-audio).mean()),"video_code_mse":float(np.square(proposal["video"]-video).mean()),"video_persistence_mse":float(np.square(appearance-video).mean()),"predicted_motion_mse":float(np.square(np.diff(proposal["video"],axis=0)).mean()),"compute_seconds":sum(r["compute_seconds"] for r in proposal["trace"])}; write_json(outdir/"case.json",case); records.append(case); write_json(root/"long_cases.json",records); print(json.dumps({"long_case":len(records),"id":row["id"],"run":runid,"mode":mode,"control":control}),flush=True)
            del model; torch.cuda.empty_cache()
        constant=folder/"constant_appearance.mp4"
        if not constant.exists(): encode_video(constant,np.broadcast_to(picture,(packets,64,64)),folder/"input.wav")
        silence=folder/"silence.wav"; write_audio(silence,np.zeros_like(sound)); meanwave,_=synthesize(np.zeros((packets,400),np.float32),codec,vocoder,"cuda"); write_audio(folder/"mean_spectrum.wav",meanwave); oracle,_=synthesize(audio,codec,vocoder,"cuda"); write_audio(folder/"oracle_vocoder.wav",oracle)
    write_json(root/"long_cases.json",records); text_records=[]
    for runid in MAIN:
        model,_=load_model(Path(cfg["artifacts"])/runid/"best.pt"); model.requires_grad_(False)
        for i,text in enumerate(cfg["text_prompts"]):
            phones,spans,length=estimated_text(text,duration); outdir=root/"text"/("prompt"+str(i+1))/runid; outdir.mkdir(parents=True,exist_ok=True)
            if (outdir/"case.json").exists(): text_records.append(json.loads((outdir/"case.json").read_text())); continue
            if length*PACKET_SECONDS<=30: raise ValueError("Fixed text prompt must genuinely exceed30s")
            proposal=rollout(model,np.zeros((length,400),np.float32),np.zeros((length,64),np.float32),phones,appearance,"text",device="cuda"); wave,renderer=synthesize(proposal["mel"],codec,vocoder,"cuda"); wav=write_audio(outdir/"output.wav",wave); frames=decode_video(proposal["video"],codec,picture,appearance); meta=encode_video(outdir/"output.mp4",frames,outdir/"output.wav"); np.savez_compressed(outdir/"proposals.npz",mel=proposal["mel"].astype(np.float16),video=proposal["video"].astype(np.float16)); write_json(outdir/"trace.json",proposal["trace"]); case={"run":runid,"prompt":i+1,"transcript":text,"estimated_spans":spans,"reference_timings_used":False,"mode":"text","folder":str(outdir),"waveform":wav,"video":meta,"renderer":renderer,"source_phase_used":False,"field_text":proposal["text"]}; write_json(outdir/"case.json",case); text_records.append(case); print(json.dumps({"text_case":runid,"prompt":i+1,"seconds":wav["seconds"]}),flush=True)
        del model; torch.cuda.empty_cache()
    write_json(root/"text_cases.json",text_records)


def edit_distance(reference,hypothesis):
    a=reference.split(); b=hypothesis.split(); row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1): new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1],len(a)


def recognize(cfg,root):
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    revision="87c7102498dcde7456f24cfd30239ca606ed9063"; processor=WhisperProcessor.from_pretrained("openai/whisper-tiny.en",revision=revision,local_files_only=True); recognizer=WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en",revision=revision,local_files_only=True).cuda().half().eval(); normalizer=processor.tokenizer.normalize; jobs=[]
    for record in json.loads((root/"speech_cases.json").read_text()):
        for condition in record["conditions"]: jobs.append({"group":"speech","id":record["id"],"actor":record["actor"],"split":record["split"],"gap_seconds":record["gap_seconds"],"condition":condition,"path":str(Path(record["folder"])/(condition+".wav")),"reference":record["transcript"]})
    for group,name in (("long","long_cases.json"),("text","text_cases.json")):
        for record in json.loads((root/name).read_text()):
            if group=="long" and record["mode"]=="audio" and record["control"]!="prefix": continue
            jobs.append({"group":group,"id":record.get("id","prompt"+str(record.get("prompt"))),"actor":record.get("actor"),"condition":record["run"]+"_"+record["mode"]+"_"+record.get("control","normal"),"path":str(Path(record["folder"])/"output.wav"),"reference":record["transcript"]})
    for folder in sorted((root/"long_inputs").iterdir()):
        rec=json.loads((folder/"input.json").read_text())
        for name in ("input","silence","mean_spectrum","oracle_vocoder"): jobs.append({"group":"long_control","id":rec["id"],"actor":rec["actor"],"condition":name,"path":str(folder/(name+".wav")),"reference":rec["transcript"]})
    target=root/"recognizer_results.json"; done=json.loads(target.read_text()) if target.exists() else []; seen={r["path"] for r in done}; cache={}
    for job in jobs:
        if job["path"] in seen: continue
        digest=file_sha(job["path"])
        if digest in cache: chunks=cache[digest]
        else:
            sound=read_audio(job["path"]); sound=torchaudio.functional.resample(torch.from_numpy(sound)[None],24000,16000)[0].numpy(); chunks=[]
            for start in range(0,len(sound),20*16000):
                segment=sound[start:start+20*16000]; inputs=processor(segment,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
                with torch.inference_mode(): ids=recognizer.generate(inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),max_new_tokens=256,num_beams=1,do_sample=False)
                text=processor.batch_decode(ids,skip_special_tokens=True)[0].strip(); chunks.append({"start_seconds":start/16000,"end_seconds":(start+len(segment))/16000,"text":text,"generated_token_count":int(ids.shape[-1]),"max_new_tokens":256})
            cache[digest]=chunks
        text=" ".join(c["text"] for c in chunks); error,words=edit_distance(normalizer(job["reference"]),normalizer(text)); raw_error,raw_words=edit_distance(job["reference"].lower(),text.lower()); done.append({**job,"wave_sha256":digest,"recognizer_text":text,"chunks":chunks,"normalized_edit_count":error,"normalized_reference_words":words,"wer":error/max(1,words),"raw_wer":raw_error/max(1,raw_words)}); seen.add(job["path"])
        if len(done)%10==0: write_json(target,done); print(json.dumps({"recognized":len(done),"total":len(jobs)}),flush=True)
    write_json(target,done); write_json(root/"recognizer.json",{"name":"openai/whisper-tiny.en","revision":revision,"external_evaluation_parameters":sum(p.numel() for p in recognizer.parameters()),"normalization":"Whisper English standard, both reference and hypothesis","long_chunk_seconds":20,"complete_output_coverage":True,"test_used_for_selection":False})


def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage",required=True,choices=("speech","fields","long","score")); a=p.parse_args(); torch.set_num_threads(4); cfg=json.loads(Path(CONFIG).read_text()); root=freeze(cfg)
    with Path("/tmp/slwm_research_gpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        {"speech":speech,"fields":fields,"long":long_media,"score":recognize}[a.stage](cfg,root)


if __name__=="__main__": main()
