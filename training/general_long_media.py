"""Unselected40.96s cross-modal outputs and decoder ceilings; native16px RGB."""
import fcntl
import gc
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from exploration.stream_media import read_audio,write_audio,synthesize,ctc_text,video_encoder
from training.general_batch import Sources,STEP,base_observation,tensor_tree
from training.general_data import CACHE,order
from training.general_evaluation_lock import register
from training.general_sources import write
from training.general_speech_media import SpeechRenderer
from training.general_speech_score import Recognizer
from training.general_train import initialize

PROMPTS=[
 "A person walks through a quiet room and opens a window. Birds call outside while a car passes along the street. The person places a cup on a wooden table, turns toward the door, and takes several slow steps. Water begins to run in the kitchen. A dog barks twice in the distance. The person closes a drawer, lifts the cup, and places it beside a small red bowl. Another car moves past the house. Leaves move in the wind and footsteps continue across the floor. The person stops, listens, and returns to the table. The room becomes quiet for a moment before the window closes and the person walks toward the kitchen again, carrying the cup carefully with one hand.",
 "The robot observes a table with a bottle, a cup, and a small box. It moves its hand toward the bottle, pauses before contact, and closes the gripper. The hand lifts slowly and carries the bottle toward the empty part of the table. It lowers the bottle, opens the gripper, and moves away. The robot then approaches the cup from the side. It changes direction when the box blocks the path, keeps the hand above the table, and reaches toward the cup again. It pauses, closes the gripper, and lifts the cup. The hand moves toward the bottle and places the cup beside it. Finally the robot opens the gripper and returns its hand to the original position above the table."
]


def encode_video(path,frames,audio):
    encoder,options=video_encoder();path.parent.mkdir(parents=True,exist_ok=True)
    values=np.rint((frames.clip(-1,1)+1)*127.5).astype(np.uint8).transpose(0,2,3,1)
    command=["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","16x16","-framerate","375/16","-i","pipe:0","-i",str(audio),"-c:v",encoder,*options,"-pix_fmt","yuv420p","-c:a","aac","-b:a","96k","-shortest","-movflags","+faststart",str(path)]
    done=subprocess.run(command,input=values.tobytes(),capture_output=True)
    if done.returncode:raise RuntimeError(done.stderr.decode()[-2000:])
    info=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames,width,height","-of","json",str(path)]))["streams"][0]
    assert int(info["nb_read_frames"])==len(values) and info["width"]==info["height"]==16
    return {"frames":len(values),"decoded_frames":int(info["nb_read_frames"]),"seconds":len(values)*STEP,"native_width":16,"native_height":16,"encoder":encoder}


def rollout(model,source,renderer,mode,*,mel=None,pixels=None,text=None,prefix=False):
    length=960;t=source.cfg["context"];pictures=[];phones=[];characters=[];trace=[]
    for start in range(0,length,t):
        n=min(t,length-start);obs=base_observation(1,n,STEP);obs["time"]+=start*STEP
        if mode=="audio" and (not prefix or start==0):
            values=(mel[start:start+n].reshape(n,4,100)-source.stats["audio_mean"])/source.stats["audio_scale"]
            obs.update(audio=values[None],audio_present=np.ones((1,n,4),bool))
        elif mode=="video" and (not prefix or start==0):
            image=np.asarray(pixels[start:start+n],np.float32)/127.5-1
            obs.update(images=np.repeat(image[None,:,None,None],3,3),image_present=np.ones((1,n,1),bool))
        elif mode=="text":
            words=text.split();a=int(start*STEP*3);b=int((start+n)*STEP*3)
            supplied=source.text_observation([" ".join(words[a:b])]);obs.update(text=supplied["text"],text_present=supplied["text_present"])
        obs=tensor_tree(obs,"cuda")
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            state=model(obs);query=model.query(state,obs["time"],obs["time_unit"],0)
            phones.extend(model.propose(query,"phones")[0].argmax(-1).reshape(-1).cpu().tolist())
            characters.extend(model.propose(query,"characters")[0].argmax(-1).cpu().tolist())
            visual=model.query(state,obs["time"],obs["time_unit"],1,view=0);pictures.append(2*model.propose(visual,"image")[0].tanh().cpu().numpy())
        trace.append({"first_packet":start,"received_packets":n,"input_mode":mode,"prefix_only":prefix,"observed_after_prefix":not prefix or start==0})
    phone=np.array(phones,np.int64)[None];physical=renderer.phone.mean[None,None].expand(1,len(phones),-1).clone();present=torch.zeros((1,len(phones)),dtype=torch.bool,device="cuda")
    with torch.inference_mode():predicted=renderer.phone.render_physical(torch.from_numpy(phone).cuda(),physical,present)[0].cpu().numpy()
    wave,meta=synthesize(predicted,{"audio_mean":np.zeros(100,np.float32),"audio_scale":np.ones(100,np.float32)},renderer.vocoder,"cuda")
    return {"frames":np.concatenate(pictures),"wave":wave,"phone_ids":phones,"field_text":ctc_text(characters),"trace":trace,"renderer":meta}


def main():
    torch.set_num_threads(4);cfg=json.loads(Path("configs/t2/general_transfer.json").read_text());root=Path(cfg["root"]);register(cfg);assert (root/"evaluation_lock.json").exists()
    out=root/"long_media";out.mkdir(exist_ok=True);source=Sources(cfg,"test");rows=source.rows["ami"];chosen=[]
    for group in sorted({r["group"] for r in rows}):
        eligible=[r for r in rows if r["group"]==group and r["duration"]>=121]
        if eligible:chosen.append(min(eligible,key=lambda r:order(r["id"]+"long40s")))
    cases=[];inputs={};jobs=[]
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        renderer=SpeechRenderer(cfg)
        for row in chosen:
            first=1875;length=960;folder=out/"inputs"/row["id"];folder.mkdir(parents=True,exist_ok=True)
            sound=read_audio(row["raw_path"],80.,length*STEP);write_audio(folder/"source.flac",sound)
            mel=np.asarray(source.load(row["mel_file"])[row["offset"]+first:row["offset"]+first+length],np.float32)
            pixels=np.asarray(source.load(row["pixel_file"])[first:first+length]);reference=" ".join(w for a,b,w in row["spans"] if a>=80 and b<=80+length*STEP)
            frames=np.repeat((pixels.reshape(length,16,4,16,4).mean((2,4))/127.5-1)[:,None],3,1)
            encode_video(folder/"source.mp4",frames,folder/"source.flac")
            oracle,meta=synthesize(mel,{"audio_mean":np.zeros(100,np.float32),"audio_scale":np.ones(100,np.float32)},renderer.vocoder,"cuda");write_audio(folder/"oracle_mel.flac",oracle)
            inputs[row["id"]]={"row":row,"mel":mel,"pixels":pixels,"reference_frames":frames,"reference":reference,"folder":folder}
            for condition in ("source","oracle_mel"):jobs.append({"id":row["id"],"condition":condition,"path":str(folder/(condition+".flac")),"reference":reference,"cluster":row["group"]})
        del renderer;gc.collect();torch.cuda.empty_cache()
    for run in (r for r in cfg["runs"] if r["id"] in {"G1-101","G1-102","G1-103","G1-104"}):
        marker=out/run["id"]/"complete.json"
        if marker.exists():
            saved=json.loads(marker.read_text());cases+=saved["cases"];jobs+=saved["jobs"];continue
        current=[];own_jobs=[]
        with open("/tmp/slwm_research_gpu.lock","a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);ck=torch.load(root/run["id"]/"endpoint.pt",map_location="cpu",weights_only=False,mmap=True);model=initialize(cfg,run).cuda().eval().requires_grad_(False);model.load_state_dict(ck["model"]);del ck;renderer=SpeechRenderer(cfg)
            for key,item in inputs.items():
                for mode,control in [("audio","aligned"),("audio","wrong_time"),("audio","prefix"),("video","aligned")]:
                    mel=item["mel"] if control!="wrong_time" else np.roll(item["mel"],480,axis=0)
                    proposal=rollout(model,source,renderer,mode,mel=mel,pixels=item["pixels"],prefix=control=="prefix")
                    folder=out/run["id"]/key/(mode+"_"+control);folder.mkdir(parents=True,exist_ok=True)
                    audio=item["folder"]/"source.flac" if mode=="audio" else folder/"output.flac"
                    if mode=="video":write_audio(audio,proposal["wave"]);own_jobs.append({"id":key,"run":run["id"],"condition":mode+"_"+control,"path":str(audio),"reference":item["reference"],"cluster":item["row"]["group"]})
                    frames=proposal["frames"] if mode=="audio" else item["reference_frames"]
                    media=encode_video(folder/"output.mp4",frames,audio);write(folder/"trace.json",proposal["trace"])
                    reference=item["reference_frames"];predicted=proposal["frames"]
                    record={"id":key,"run":run["id"],"mode":mode,"control":control,"folder":str(folder),"media":media,"field_text":proposal["field_text"],"reference":item["reference"],"source_audio_supplied_to_model":mode=="audio","generated_audio_uses_training_mean_voice":mode=="video",
                            "frame_mse":float(((predicted-reference)**2).mean()),"motion_mse":float(((np.diff(predicted,axis=0)-np.diff(reference,axis=0))**2).mean()),"zero_motion_error":float((np.diff(reference,axis=0)**2).mean()),
                            "audio_conditioning_shift_packets":480 if control=="wrong_time" else 0,"appearance_provided":False,
                            "visual_decoder_training":"COCO masked images and DROID cross-view/future images; no dense AMI face reconstruction training",
                            "scope":"Audio-to-image is an untrained direction; video mode supplies a privileged visual-decoder reconstruction ceiling. Displayed input video/soundtrack remains fixed across controls."}
                    write(folder/"case.json",record);current.append(record)
            for index,prompt in enumerate(PROMPTS):
                proposal=rollout(model,source,renderer,"text",text=prompt);folder=out/run["id"]/("text_"+str(index+1));folder.mkdir(parents=True,exist_ok=True);audio=folder/"output.flac";write_audio(audio,proposal["wave"]);media=encode_video(folder/"output.mp4",proposal["frames"],audio)
                consumed=" ".join(prompt.split()[:int(960*STEP*3)]);record={"id":"text"+str(index+1),"run":run["id"],"mode":"text","control":"provided_text","folder":str(folder),"media":media,"provided_text":prompt,"consumed_text":consumed,"reference":consumed,"field_text":proposal["field_text"],"estimated_words_per_second":3,"reference_timings_used":False,"source_audio_or_video_supplied":False,"scope":"Out-of-distribution text-only generation probe; no arbitrary-scene decoder or TTS capability assumed"};write(folder/"case.json",record);current.append(record);own_jobs.append({"id":record["id"],"run":run["id"],"condition":"provided_text","path":str(audio),"reference":consumed,"cluster":record["id"]})
            marker.parent.mkdir(parents=True,exist_ok=True);write(marker,{"cases":current,"jobs":own_jobs});cases+=current;jobs+=own_jobs;del model,renderer;gc.collect();torch.cuda.empty_cache()
    recipe=json.loads(Path("configs/t2/general_speech_eval.json").read_text())
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);recognizer=Recognizer(recipe,root/"speech/recognizer_cache.json");recognized=recognizer.score_long(jobs)
    write(out/"cases.json",cases);write(out/"recognized.json",recognized)
    write(out/"summary.json",{"cases":len(cases),"native_rgb_pixels":[16,16],"seconds_per_output":40.96,"fresh_meeting_components":len(chosen),"scope":"Raw generation probes and visual/audio decoder ceilings; no reference-free video-quality or physical-control claim","waveform_scores_file":"recognized.json"})


if __name__=="__main__":main()
