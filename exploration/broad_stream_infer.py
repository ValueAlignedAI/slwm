"""Ordinary-file continuous audio/video/text proposals from the trained shared field."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.broad_stream_train import load_model
from training.broad_data import face_video
from training.stream_codec import PACKET_SECONDS,estimated_text,timed_phones,observed_mel,load_vocoder
from training.research_sources import file_sha
from training.t2_real_av import write_json
from exploration.stream_media import read_audio,write_audio,rollout,synthesize,restore_known,decode_video,encode_video,load_renderer,check_video_cache,preserve_mel_context


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--codec-dir"); p.add_argument("--mode",choices=("audio-to-video","video-to-audio","text-to-av","repair"),required=True); p.add_argument("--audio"); p.add_argument("--video"); p.add_argument("--text"); p.add_argument("--start",type=float,default=0.); p.add_argument("--seconds",type=float,default=40.96); p.add_argument("--min-text-seconds",type=float,default=0.); p.add_argument("--gaps",default=""); p.add_argument("--preserve-mel-context",action="store_true"); p.add_argument("--prefix-chunks",type=int); p.add_argument("--reset-memory",action="store_true"); p.add_argument("--device",choices=("cpu","cuda"),default="cpu"); p.add_argument("--output",required=True); a=p.parse_args(); torch.set_num_threads(4)
    if a.preserve_mel_context and a.mode!="repair": p.error("Mel context conservation requires repair mode")
    outdir=Path(a.output); outdir.mkdir(parents=True,exist_ok=True); model,ck=load_model(a.checkpoint,a.device); model.requires_grad_(False); bundle=Path(a.codec_dir or ck["config"]["corpus"]); codec=dict(np.load(bundle/"codec.npz",allow_pickle=False)); appearance_pixels=np.load(bundle/"appearance_pixels.npy"); appearance_code=np.load(bundle/"appearance_code.npy"); source_hash=None; sound=None; missing=None; visible=None
    if a.mode in ("audio-to-video","repair"):
        if not a.audio: p.error("This mode requires --audio")
        sound=read_audio(a.audio,a.start,a.seconds); packets=(1+len(sound)//256)//4
        if packets<32: p.error("At least1.365s of input is required")
        missing=np.zeros(len(sound),bool)
        if a.mode=="repair":
            for span in filter(None,a.gaps.split(",")):
                begin,end=map(float,span.split(":"))
                if begin<0 or end<=begin or end>len(sound)/24000: p.error("Gaps must fit the selected waveform interval")
                missing[int(round(begin*24000)):int(round(end*24000))]=True
            if not missing.any(): p.error("Repair mode requires actual --gaps")
        physical,fine,visible=observed_mel(sound,missing,packets); audio=(physical-np.tile(codec["audio_mean"],4))/np.tile(codec["audio_scale"],4); video=np.zeros((packets,64),np.float32); phones=np.zeros(packets,np.int64); mode="repair" if a.mode=="repair" else "audio"; appearance=np.zeros(64,np.float32) if mode=="repair" else appearance_code; source_hash=file_sha(a.audio)
    elif a.mode=="video-to-audio":
        if not a.video: p.error("This mode requires --video; source audio is never decoded")
        first=int(round(a.start/PACKET_SECONDS)); packets=int(round(a.seconds/PACKET_SECONDS)); cache=outdir/"input_video_cache"; cache.mkdir(exist_ok=True); check_video_cache(cache,a.video,first+packets); row={"id":"input","video_path":a.video}; face_video(row,cache,first+packets); pixels=np.asarray(np.load(cache/"input_pixels.npy",mmap_mode="r")[first:first+packets]); video=((pixels.reshape(packets,-1).astype(np.float32)/255.-codec["pixel_mean"])@codec["video_basis"].T)/codec["video_scale"]; audio=np.zeros((packets,400),np.float32); phones=np.zeros(packets,np.int64); mode="video"; appearance=np.zeros(64,np.float32); source_hash=file_sha(a.video)
    else:
        if not a.text: p.error("This mode requires --text")
        duration_model=json.loads((bundle/"duration_model.json").read_text()); phones,spans,packets=estimated_text(a.text,duration_model); native_end=spans[-1][1] if spans else 0.
        if native_end<a.min_text_seconds and native_end>0:
            scale=a.min_text_seconds/native_end; spans=[(x*scale,y*scale,w) for x,y,w in spans]; packets=int(np.ceil(a.min_text_seconds/PACKET_SECONDS)); phones=timed_phones(spans,packets)
        write_json(outdir/"provided_text.json",{"text":a.text,"estimated_spans":spans,"original_estimated_seconds":native_end,"reference_timings_used":False}); audio=np.zeros((packets,400),np.float32); video=np.zeros((packets,64),np.float32); mode="text"; appearance=appearance_code
    if a.mode=="repair" and a.video:
        first=round(a.start/PACKET_SECONDS); cache=outdir/"input_video_cache"; cache.mkdir(exist_ok=True); check_video_cache(cache,a.video,first+packets); face_video({"id":"input","video_path":a.video},cache,first+packets); pixels=np.asarray(np.load(cache/"input_pixels.npy",mmap_mode="r")[first:first+packets]); video=((pixels.reshape(packets,-1).astype(np.float32)/255.-codec["pixel_mean"])@codec["video_basis"].T)/codec["video_scale"]; appearance=video[0].copy(); mode="repair_av"
    proposals=rollout(model,audio,video,phones,appearance,mode,visible=visible,prefix_chunks=a.prefix_chunks,reset_memory=a.reset_memory,device=a.device)
    if a.preserve_mel_context:
        np.save(outdir/"raw_predicted_mel.npy",proposals["mel"]); proposals["mel"]=preserve_mel_context(proposals["mel"],audio,fine)
    np.save(outdir/"predicted_mel.npy",proposals["mel"]); np.save(outdir/"predicted_video_codes.npy",proposals["video"]); (outdir/"field_text.txt").write_text(proposals["text"]+"\n"); write_json(outdir/"trace.json",proposals["trace"])
    preserve=False
    if a.mode=="audio-to-video" and a.prefix_chunks is None:
        wave=np.pad(sound,(0,max(0,packets*1024-len(sound))))[:packets*1024]; renderer={"soundtrack":"Original supplied audio; only video motion is generated"}
    else:
        vocoder=load_renderer(bundle/"vocos",a.device); wave,renderer=synthesize(proposals["mel"],codec,vocoder,a.device)
        if a.mode=="repair":
            wave=restore_known(wave,sound,missing); preserve=bool(np.array_equal(np.rint(wave[~missing]*32768).astype("<i2"),np.rint(sound[~missing]*32768).astype("<i2"))); assert preserve; np.save(outdir/"missing_samples.npy",missing)
    wave_meta=write_audio(outdir/"output.wav",wave)
    video_meta=None
    if a.mode!="repair":
        frames=pixels if a.mode=="video-to-audio" else decode_video(proposals["video"],codec,appearance_pixels,appearance_code); video_meta=encode_video(outdir/"output.mp4",frames,outdir/"output.wav")
    write_json(outdir/"inference.json",{"run":ck["run"],"checkpoint_sha256":file_sha(a.checkpoint),"input_sha256":source_hash,"mode":a.mode,"device":a.device,"video_assisted_repair":mode=="repair_av","observed_fine_mel_context_preserved":a.preserve_mel_context,"video_input_sha256":file_sha(a.video) if mode=="repair_av" else None,"prefix_observation_chunks":a.prefix_chunks,"model_parameters":sum(p.numel() for p in model.parameters()),"waveform":wave_meta,"video":video_meta,"renderer":renderer,"observed_pcm_exact":preserve if a.mode=="repair" else None,"source_audio_decoded":a.mode in ("audio-to-video","repair"),"text_supplied":a.mode=="text-to-av","source_phase_used":False,"appearance":"Declared fixed training image; predicted motion,64x64 native" if a.mode in ("audio-to-video","text-to-av") else "Input video" if a.mode in ("video-to-audio",) or mode=="repair_av" else "None","scope":"Continuous research proposals, no arbitrary-scene or motor-control claim"}); print(json.dumps({"output":str(outdir),"seconds":wave_meta["seconds"],"observed_pcm_exact":preserve}),flush=True)


if __name__=="__main__": main()
