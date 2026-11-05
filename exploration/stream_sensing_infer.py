"""Autonomous sensing decisions over ordinary files; no transcripts/future labels."""
from contextlib import nullcontext
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from training.broad_data import face_video
from training.broad_stream_train import load_model
from training.stream_codec import PACKET_SECONDS,observed_mel
from training.stream_sensing import RiskActor,BITS,features,raw_features,choose,goal_error,observe,weight_digest
from training.research_sources import file_sha
from training.t2_real_av import write_json
from exploration.stream_media import read_audio,check_video_cache


def main():
    p=argparse.ArgumentParser(); p.add_argument("--world",required=True); p.add_argument("--actor",required=True); p.add_argument("--codec-dir",required=True); p.add_argument("--audio",required=True); p.add_argument("--video",required=True); p.add_argument("--start",type=float,default=0.); p.add_argument("--seconds",type=float,default=31*32*PACKET_SECONDS); p.add_argument("--goal",choices=("audio","video","both"),default="both"); p.add_argument("--device",choices=("cpu","cuda"),default="cpu"); p.add_argument("--pace",action="store_true"); p.add_argument("--output",required=True); a=p.parse_args(); torch.set_num_threads(4); root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    if a.start<0 or a.seconds<=0: p.error("Start must be nonnegative and duration positive")
    ck=torch.load(a.actor,map_location="cpu",weights_only=False); assert file_sha(a.world)==ck["world_checkpoint_sha256"],"Actor belongs to another world checkpoint"; model,_=load_model(a.world,a.device); model.requires_grad_(False); actor=RiskActor(ck["features"],ck["hidden"],ck["config"]["risk_dropout"]); actor.load_state_dict(ck["model"],strict=True); actor.to(a.device).eval().requires_grad_(False); codec=dict(np.load(Path(a.codec_dir)/"codec.npz")); sound=read_audio(a.audio,a.start,a.seconds); packets=((1+len(sound)//256)//4)//32*32
    if packets<64: p.error("At least two complete received chunks are required")
    physical,_,_=observed_mel(sound,np.zeros(len(sound),bool),packets); audio=(physical-np.tile(codec["audio_mean"],4))/np.tile(codec["audio_scale"],4); first=round(a.start/PACKET_SECONDS); cache=root/"video_cache"; cache.mkdir(exist_ok=True); check_video_cache(cache,a.video,first+packets); face_video({"id":"input","video_path":a.video},cache,first+packets); pixels=np.asarray(np.load(cache/"input_pixels.npy",mmap_mode="r")[first:first+packets]); video=((pixels.reshape(packets,-1).astype(np.float32)/255.-codec["pixel_mean"])@codec["video_basis"].T)/codec["video_scale"]; aa=torch.as_tensor(audio[None],device=a.device); vv=torch.as_tensor(video[None],device=a.device); norm=(torch.as_tensor(ck["mean"],device=a.device),torch.as_tensor(ck["scale"],device=a.device)); before=weight_digest(model); actor_before=weight_digest(actor); records=[]; forecasts=[]; amp=torch.autocast("cuda",dtype=torch.bfloat16) if a.device=="cuda" else nullcontext()
    with torch.inference_mode(),amp:
        bits=torch.ones(1,2,device=a.device); current=observe(model,aa,vv,0,bits); beginning=time.perf_counter()
        for tick,at in enumerate(range(32,packets,32)):
            begin=time.perf_counter(); state=(raw_features if ck.get("feature_mode")=="raw" else features)(current,bits); pred=actor((state-norm[0])/norm[1]).float().clamp(-12,8).exp(); decision=choose(pred,a.goal,ck["config"]["sensor_cost"]); index=int(decision.item()); bits=torch.as_tensor(BITS[index:index+1],device=a.device); risk=float(goal_error(pred[:,index],a.goal)); current=observe(model,aa,vv,at,bits,current["memory"],current["raw_memory"]); forecasts.append(current["future"][0,0].float().cpu().numpy()); elapsed=time.perf_counter()-begin; record={"tick":tick,"source_received_until_seconds":a.start+(at+32)*PACKET_SECONDS,"forecast_starts_seconds":a.start+(at+34)*PACKET_SECONDS,"action":ck["config"]["actions"][index],"committed":risk<=ck["thresholds"][a.goal],"predicted_goal_error":risk,"memory_norm":float(current["memory"].float().norm()),"compute_seconds":elapsed,"deadline_missed":elapsed>32*PACKET_SECONDS}; records.append(record); print(json.dumps(record),flush=True); write_json(root/"decisions.json",records)
            if a.pace:
                delay=(tick+1)*32*PACKET_SECONDS-(time.perf_counter()-beginning)
                if delay>0: time.sleep(delay)
    assert weight_digest(model)==before and weight_digest(actor)==actor_before; np.save(root/"forecasts.npy",np.stack(forecasts)); write_json(root/"inference.json",{"world_run":ck["world_run"],"actor_id":ck["id"],"world_sha256":file_sha(a.world),"actor_sha256":file_sha(a.actor),"audio_sha256":file_sha(a.audio),"video_sha256":file_sha(a.video),"automatic_decisions":len(records),"observation_seconds":packets*PACKET_SECONDS,"paced":a.pace,"wall_seconds_after_bootstrap":time.perf_counter()-beginning,"weights_unchanged":True,"future_ground_truth_used":False,"transcript_used":False,"physical_world_control":False,"sensing_cost_scope":"Hypothetical recorded-sensor access; file features prepared before replay","algorithmic_chunk_seconds":32*PACKET_SECONDS,"goal":a.goal})


if __name__=="__main__": main()
