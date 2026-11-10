"""Observation-only G1 speech repair and counted waveform rendering."""
from pathlib import Path
import json

import numpy as np
import torch

from models.phonetic_renderer import PhoneticRenderer
from training.general_batch import STEP, base_observation, interpolate, tensor_tree
from training.general_losses import without
from training.stream_codec import observed_mel, load_vocoder
from exploration.stream_media import read_audio, restore_known, synthesize, write_audio, ctc_text


class SpeechRenderer:
    def __init__(self,cfg,device="cuda"):
        self.cfg,self.device=cfg,device
        self.stats=np.load(Path(cfg["data_root"])/"normalizers.npz")
        root=Path("experiments/multimodal/t2/phonetic-renderer-20261106")
        ck=torch.load(root/"best.pt",map_location="cpu",weights_only=False)
        weights=ck["model"];self.phone=PhoneticRenderer(ck["phone_count"],weights["mean"],weights["scale"]).to(device).eval().requires_grad_(False);self.phone.load_state_dict(weights)
        self.vocoder=load_vocoder(device)

    def case(self,row,case):
        start=case.get("start_seconds",0.);duration=case.get("duration_seconds",row.get("duration"))
        sound=read_audio(row["raw_path"],start,duration)
        packets=len(sound)//1024;sound=sound[:packets*1024]
        center=len(sound)//2;width=round(case.get("gap_seconds",.16)*24000)
        missing=np.zeros(len(sound),bool);a=max(0,center-width//2);missing[a:a+width]=True
        physical,present,_=observed_mel(sound,missing,packets)
        # A complete received32-packet window surrounds the gap. No hidden FFT
        # support frame can enter its edge, interpolation or voice statistics.
        window=int(np.clip(center//1024-self.cfg["context"]//2,0,packets-self.cfg["context"]))
        t=self.cfg["context"];known=present.reshape(packets,4)[window:window+t]
        mel=physical[window:window+t].reshape(t,4,100)
        normalized=(mel-self.stats["audio_mean"])/self.stats["audio_scale"]
        obs=base_observation(1,t,STEP);obs.update(audio=np.where(known[None,...,None],normalized[None],0).astype(np.float32),audio_present=known[None])
        baseline=interpolate(normalized.reshape(-1,100),known.reshape(-1)).reshape(t,400)
        if "spans" in row:
            reference=" ".join(w for lo,hi,w in row["spans"] if lo>=start and hi<=start+len(sound)/24000)
        else:reference=row["transcript"]
        # Libri's official full utterance string is primary, not an accidental
        # clipped word list inferred from forced-alignment boundaries.
        if row.get("actor","").startswith("LIBRI-"):reference=row["transcript"]
        return {"obs":obs,"baseline":baseline,"physical":mel.reshape(-1,100),"present":known.reshape(-1),"full_physical":physical,"full_present":present,"sound":sound,"missing":missing,"window_start_sample":window*1024,"window_start_packet":window,"source_start_packet":round(start/STEP),"reference":reference,"row":row,"case":case}

    def content_rollout(self,model,prepared):
        characters=[];phones=[];t=self.cfg["context"]
        for start in range(0,len(prepared["full_physical"]),t):
            physical=prepared["full_physical"][start:start+t].reshape(-1,4,100);n=len(physical)
            present=prepared["full_present"].reshape(-1,4)[start:start+n]
            audio=(physical-self.stats["audio_mean"])/self.stats["audio_scale"]
            obs=base_observation(1,n,STEP);obs.update(audio=np.where(present[None,...,None],audio[None],0).astype(np.float32),audio_present=present[None])
            obs=tensor_tree(obs,self.device)
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=str(self.device).startswith("cuda")):
                state=model(obs);query=model.query(state,obs["time"],obs["time_unit"],0)
                characters.extend(model.propose(query,"characters")[0].argmax(-1).cpu().tolist())
                phones.extend(model.propose(query,"phones")[0].argmax(-1).reshape(-1).cpu().tolist())
        return {"text":ctc_text(characters),"character_ids":characters,"phone_ids":phones,"scope":"Whole recording in consecutive received32-packet contexts; no transcript/phone labels supplied"}

    def visual(self,prepared,condition):
        row=prepared["row"]
        if "pixel_file" not in row or condition=="audio":return None
        pixels=np.load(row["pixel_file"],mmap_mode="r");t=self.cfg["context"]
        first=prepared["source_start_packet"]+prepared["window_start_packet"]
        offset=0
        if condition.startswith("shift"):
            offset=int(condition.split(":",1)[1])
        indices=np.arange(first+offset,first+offset+t)
        assert indices.min()>=0 and indices.max()<len(pixels),"Registered center lacks a valid timing-control window"
        picture=np.asarray(pixels[indices],np.float32)/127.5-1
        if condition=="static":picture=np.repeat(picture[:1],t,0)
        image=np.repeat(picture[None,:,None,None],3,3)
        return {"images":image,"image_present":np.ones((1,t,1),bool)}

    def infer(self,model,prepared,condition="audio",threshold=0.):
        obs=tensor_tree(prepared["obs"],self.device);visual=self.visual(prepared,condition)
        score=None
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=str(self.device).startswith("cuda")):
            if visual is not None:
                video={**without(obs,"audio"),**tensor_tree(visual,self.device)}
                a=model(obs);v=model(video)
                score=float(model.correspondence_logits(a["sequence_embedding"],v["sequence_embedding"]).mean().sigmoid())
                if score>=threshold:
                    obs.update(tensor_tree(visual,self.device))
                    obs["image_weight"]=torch.full_like(obs["image_present"],score,dtype=torch.float32)
            state=model(obs);query=model.query(state,obs["time"],obs["time_unit"],0)
            residual=model.propose(query,"audio")[0].cpu().numpy()
            probability=model.propose(query,"phones")[0].softmax(-1).cpu().numpy()
            characters=model.propose(query,"characters")[0].argmax(-1).cpu().numpy()
        direct=(prepared["baseline"]+4*np.tanh(residual)).reshape(-1,100)*self.stats["audio_scale"]+self.stats["audio_mean"]
        phones=probability.argmax(-1).reshape(1,-1)
        with torch.inference_mode():
            phonetic=self.phone.render_physical(torch.from_numpy(phones).to(self.device),torch.from_numpy(prepared["physical"][None]).to(self.device),torch.from_numpy(prepared["present"][None]).to(self.device))[0].cpu().numpy()
        for prediction in (direct,phonetic):prediction[prepared["present"]]=prepared["physical"][prepared["present"]]
        return {"direct":direct,"phonetic":phonetic,"phones":phones[0],"phone_probability":probability,"ctc_local":ctc_text(characters),"matching_probability":score,"video_accepted":visual is not None and score>=threshold}

    def waveform(self,prepared,mel):
        codec={"audio_mean":np.zeros(100,np.float32),"audio_scale":np.ones(100,np.float32)}
        wave,meta=synthesize(mel,codec,self.vocoder,self.device)
        generated=np.zeros_like(prepared["sound"]);start=prepared["window_start_sample"];n=min(len(wave),len(generated)-start);generated[start:start+n]=wave[:n]
        restored=restore_known(generated,prepared["sound"],prepared["missing"])
        assert np.array_equal(restored[~prepared["missing"]],prepared["sound"][~prepared["missing"]])
        return restored,meta

    def save(self,prepared,mel,path):
        wave,meta=self.waveform(prepared,mel);waveform=write_audio(path,wave)
        return {**meta,**waveform,"observed_pcm_exact":True,"source_phase_used":False}

    def save_many(self,prepared,candidates,directory):
        """Batch the identical bounded-window Vocos operation across candidates."""
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        names=list(candidates);physical=np.stack([candidates[name] for name in names])
        clipped=(physical< -16.12)|(physical>8.)
        physical=np.clip(physical,-16.12,8.).astype(np.float32)
        physical=np.concatenate([physical,physical[:,-1:]],1)
        with torch.inference_mode():
            waves=self.vocoder.decode(torch.from_numpy(physical.transpose(0,2,1).copy()).to(self.device)).float().cpu().numpy()
        result={}
        for i,name in enumerate(names):
            generated=np.zeros_like(prepared["sound"]);start=prepared["window_start_sample"]
            n=min(len(waves[i]),len(generated)-start);generated[start:start+n]=waves[i,:n]
            restored=restore_known(generated,prepared["sound"],prepared["missing"])
            assert np.array_equal(restored[~prepared["missing"]],prepared["sound"][~prepared["missing"]])
            path=directory/(name+".flac");waveform=write_audio(path,restored)
            result[name]={"path":str(path),**waveform,"observed_pcm_exact":True,"source_phase_used":False,
                          "external_vocoder_parameters":sum(p.numel() for p in self.vocoder.parameters()),
                          "physical_mel_clipped_fraction":float(clipped[i].mean()),"lossless_pcm16":True}
        return result
