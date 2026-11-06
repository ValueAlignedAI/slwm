"""R1 old-selection codec ceiling, never a fresh-speaker confirmation test."""
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
from transformers import WhisperProcessor,WhisperForConditionalGeneration
from models.phonetic_renderer import PhoneticRenderer
from training.temporal_masks import mask_support
from training.general_data import order
from training.general_sources import write
from training.research_sources import file_sha
from training.stream_codec import load_vocoder,observed_mel
from exploration.stream_media import read_audio,write_audio,synthesize,restore_known
from exploration.broad_stream_evaluate import edit_distance


def main():
    torch.set_num_threads(4);cfg=json.loads(Path("configs/t2/phonetic_renderer.json").read_text());root=Path(cfg["root"]);out=root/"old_selection_diagnostic";out.mkdir(exist_ok=True)
    rows=json.loads(Path(cfg["selection_rows"]).read_text());eligible=[r for r in rows if 8<=r["duration"]<=18 and r["ctc_alignment_usable"]]
    chosen=[];people=set()
    for row in sorted(eligible,key=lambda r:order(r["id"])):
        if row["actor"] in people:continue
        chosen.append(row);people.add(row["actor"])
        if len(chosen)==8:break
    write(out/"cases.json",{"scope":"Previously exposed old selection people only; oracle phone ceiling, no G1 model and no new final people","ids":[r["id"] for r in chosen],"checkpoint_sha256":file_sha(root/"best.pt"),"source_sha256":file_sha(__file__)})
    jobs=[]
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);ck=torch.load(root/"best.pt",map_location="cpu",weights_only=False);state=ck["model"];model=PhoneticRenderer(ck["phone_count"],state["mean"],state["scale"]).cuda().eval();model.load_state_dict(state);vocoder=load_vocoder("cuda")
        for row in chosen:
            dest=out/row["id"];dest.mkdir(exist_ok=True);sound=read_audio(row["raw_path"]);packets=row["length"];n=packets*1024;sound=sound[:n];missing=np.zeros(len(sound),bool);center=len(sound)//2;missing[max(0,center-1920):center+1920]=True
            observed,present,_=observed_mel(sound,missing,packets);physical=observed.reshape(-1,100);a=row["offset"];phones=np.asarray(np.load(row["phone_file"],mmap_mode="r")[a:a+packets],np.int64).reshape(1,-1)
            source=np.asarray(np.load(row["mel_file"],mmap_mode="r")[a:a+packets],np.float32)
            with torch.inference_mode():prediction=model.render_physical(torch.from_numpy(phones).cuda(),torch.from_numpy(physical[None]).cuda(),torch.from_numpy(present[None]).cuda())[0].cpu().numpy()
            codec={"audio_mean":np.zeros(100,np.float32),"audio_scale":np.ones(100,np.float32)}
            waves={"source":sound,"silenced":np.where(missing,0,sound)}
            for name,mel in [("interpolation",physical),("oracle_phone_full",prediction),("source_mel_roundtrip",source.reshape(-1,100))]:
                wave,renderer=synthesize(mel,codec,vocoder,"cuda");waves[name]=wave[:len(sound)]
            waves["oracle_phone_repair"]=restore_known(waves["oracle_phone_full"],sound,missing)
            waves["interpolation"]=restore_known(waves["interpolation"],sound,missing)
            assert np.array_equal(waves["oracle_phone_repair"][~missing],sound[~missing])
            for name,wave in waves.items():
                p=dest/(name+".wav");write_audio(p,wave);jobs.append({"id":row["id"],"person":row["actor"],"reference":row["transcript"],"condition":name,"path":str(p),"sha256":file_sha(p),"oracle_phone_input":name.startswith("oracle_phone"),"new_final_person":False})
        del model,vocoder;torch.cuda.empty_cache()
    revision="87c7102498dcde7456f24cfd30239ca606ed9063";processor=WhisperProcessor.from_pretrained("openai/whisper-tiny.en",revision=revision,local_files_only=True)
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);recognizer=WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en",revision=revision,local_files_only=True).cuda().half().eval()
        for start in range(0,len(jobs),8):
            current=jobs[start:start+8];signals=[torchaudio.functional.resample(torch.from_numpy(read_audio(j["path"])),24000,16000).numpy() for j in current];inputs=processor(signals,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
            with torch.inference_mode():tokens=recognizer.generate(inputs.input_features.cuda().half(),attention_mask=inputs.attention_mask.cuda(),max_new_tokens=256,num_beams=1,do_sample=False)
            for job,text,ids in zip(current,processor.batch_decode(tokens,skip_special_tokens=True),tokens.cpu().tolist()):
                edits,count=edit_distance(processor.tokenizer.normalize(job["reference"]),processor.tokenizer.normalize(text));job.update(recognizer_text=text,edits=edits,reference_words=count,token_limit_reached=recognizer.generation_config.eos_token_id not in ids)
            write(out/"recognized.json",jobs);print(json.dumps({"old_selection_waves_scored":min(start+8,len(jobs)),"total":len(jobs)}),flush=True)
    summary={name:{"edits":sum(j["edits"] for j in jobs if j["condition"]==name),"words":sum(j["reference_words"] for j in jobs if j["condition"]==name)} for name in sorted({j["condition"] for j in jobs})}
    for value in summary.values():value["wer"]=value["edits"]/max(1,value["words"])
    write(out/"summary.json",{"scope":"Eight previously exposed old-selection people, exploratory codec ceiling only; never fresh generalization evidence","conditions":summary,"renderer_parameters":289860,"vocos_parameters":13531650,"whisper_parameters":37760256,"recognizer_revision":revision,"new_final_people":0,"g1_shared_models_used":0,"oracle_phone_full_uses_reference_phone_intervals":True})


if __name__=="__main__":main()
