"""Immutable corpus views and architecture-independent shared sampling plans."""
import json
from pathlib import Path
import numpy as np
import torch
from training.stream_codec import PACKET_SECONDS,character_ids


class StreamCorpus:
    """Memory-mapped physical mel/phones/video -> batches of aligned coarse packets."""
    def __init__(self,cfg,split):
        self.root=Path(cfg["corpus"]); self.window=cfg["training"]["window"]; self.cfg=cfg; self.split=split; self.rows={}; self.arrays={}; self.videos={}
        names={"train":("libri_train-clean-100","ami_train"),"validation":("libri_dev-other","ami_validation"),"calibration":("ami_calibration",),"test_clean":("libri_test-clean",),"test_other":("libri_test-other",),"test_ami":("ami_test",)}[split]
        for name in names:
            folder=self.root/name; rows=json.loads((folder/"rows.json").read_text()); self.rows[name]=rows; self.arrays[name]=(np.load(folder/"mel.npy",mmap_mode="r"),np.load(folder/"phones.npy",mmap_mode="r"))
            if name.startswith("ami_"):
                for row in rows: self.videos[row["id"]]=np.load(folder/(row["id"]+"_video.npy"),mmap_mode="r")
        c=np.load(self.root/"codec.npz",allow_pickle=False); self.mean=np.tile(c["audio_mean"],4); self.scale=np.tile(c["audio_scale"],4)

    def sample(self,rng,batch,force_mode=None,ami_only=False,libri_only=False):
        plan=[]
        for i in range(batch):
            choose_ami=False if libri_only else ami_only or i>=batch*3//4
            name=next((n for n in self.rows if n.startswith("ami_" if choose_ami else "libri_")),next(iter(self.rows)))
            rows=self.rows[name]; index=int(rng.integers(len(rows))); row=rows[index]; maximum=max(0,row["length"]-self.window); start=int(rng.integers(maximum+1))
            # Same-person wrong time, sampled unconditionally for every variant.
            wrong=int(rng.integers(maximum+1)); wrong=(wrong+self.window*2)%(maximum+1) if abs(wrong-start)<self.window else wrong
            draw=float(rng.random()); mode="repair" if draw<.5 else "audio" if draw<.7 else "video" if draw<.85 and choose_ami else "text"
            if draw>.985: mode="null"
            if force_mode is not None: mode=force_mode
            plan.append({"name":name,"index":index,"start":start,"wrong":wrong,"mode":mode,"video_in_repair":bool(rng.random()<.5),"gap_start":int(rng.integers(24,104)),"gap_length":int(rng.choice([2,4,8])),"drop_chunks":(rng.random(4)<.1).tolist(),"include_style":bool(rng.random()<.5)})
        return plan

    def batch(self,plan,device="cuda",shuffled=False):
        b=len(plan); t=self.window; audio=np.zeros((b,t,400),np.float32); video=np.zeros((b,t,64),np.float32); wrong_video=video.copy(); phones=np.zeros((b,t),np.int64); valid=np.zeros((b,t),bool); has_video=np.zeros(b,bool); a_mask=np.zeros((b,t),bool); v_mask=a_mask.copy(); t_mask=a_mask.copy(); holes=a_mask.copy(); texts=[]; ids=[]; actors=[]; appearance=np.zeros((b,64),np.float32); phone_valid=np.ones(b,bool); ctc_valid=np.ones(b,bool); actual_modes=[]
        for i,p in enumerate(plan):
            row=self.rows[p["name"]][p["index"]]; n=min(t,row["length"]-p["start"]); offset=row["offset"]+p["start"]; mel,units=self.arrays[p["name"]]; audio[i,:n]=(np.asarray(mel[offset:offset+n],np.float32)-self.mean)/self.scale; phones[i,:n]=units[offset:offset+n]; valid[i,:n]=True; ids.append(row["id"]); actors.append(row["actor"])
            phone_valid[i]=row.get("phone_supervision_usable",True); ctc_valid[i]=row.get("ctc_alignment_usable",True)
            begin=p["start"]*PACKET_SECONDS; end=(p["start"]+n)*PACKET_SECONDS; texts.append(character_ids(" ".join(w for a,z,w in row["spans"] if a>=begin and z<=end)))
            if row["source"]=="ami":
                values=self.videos[row["id"]]; has_video[i]=True; at=p["wrong"] if shuffled else p["start"]; video[i,:n]=values[at:at+n]; wrong_at=min(p["wrong"],len(values)-n); wrong_video[i,:n]=values[wrong_at:wrong_at+n]
            mode=p["mode"]
            if mode=="text" and not row.get("text_generation_usable",True): mode="audio"
            actual_modes.append(mode)
            if mode in ("audio","repair"):
                a_mask[i,:n]=True
                if mode=="repair":
                    a=min(p["gap_start"],max(0,n-p["gap_length"]-2)); z=min(n,a+p["gap_length"]); holes[i,a:z]=True; a_mask[i,max(0,a-1):min(n,z+1)]=False
                    if has_video[i] and p["video_in_repair"]: v_mask[i,:n]=True
            if mode=="video" and has_video[i]: v_mask[i,:n]=True
            if mode=="text": t_mask[i,:n]=True
            if has_video[i] and p["include_style"] and (mode in ("audio","video","text") or v_mask[i].any()): appearance[i]=video[i,0]
            for chunk,dropped in enumerate(p["drop_chunks"]):
                if dropped: a_mask[i,chunk*32:(chunk+1)*32]=False; v_mask[i,chunk*32:(chunk+1)*32]=False; t_mask[i,chunk*32:(chunk+1)*32]=False
        # Target values remain separate; the model masks every absent input and
        # computes its own observed-only interpolation inside each received chunk.
        tensors={"audio":audio,"video":video,"wrong_video":wrong_video,"phones":phones,"valid":valid,"a_mask":a_mask,"v_mask":v_mask,"t_mask":t_mask,"holes":holes,"has_video":has_video,"appearance":appearance,"phone_valid":phone_valid,"ctc_valid":ctc_valid}
        out={k:torch.as_tensor(v,device=device) for k,v in tensors.items()}; out.update(texts=texts,ids=ids,actors=actors,modes=actual_modes); return out


def predict(model,batch,reset_memory=False,view=None):
    valid=batch["valid"]; a,v,t=batch["a_mask"],batch["v_mask"],batch["t_mask"]; picture=batch["video"]; appearance=batch["appearance"]
    if view is not None:
        a=valid if view=="audio" else torch.zeros_like(valid); v=valid&batch["has_video"][:,None] if view in ("video","wrong_video") else torch.zeros_like(valid); t=valid if view=="text" else torch.zeros_like(valid); appearance=torch.zeros_like(appearance)
        if view=="wrong_video": picture=batch["wrong_video"]
    args=(batch["audio"],picture,batch["phones"],a,v,t,valid,appearance)
    if not reset_memory: return model(*args)
    results=[]
    for start in range(0,valid.shape[1],model.chunk):
        end=start+model.chunk; results.append(model(*(x[:,start:end] for x in args[:7]),args[7]))
    result={}
    for key in ("mel","video","field","phone_logits","character_logits","activity","risk"): result[key]=torch.cat([r[key] for r in results],1)
    result["future"]=torch.cat([r["future"] for r in results],1); result["embedding"]=torch.nn.functional.normalize(torch.stack([r["embedding"] for r in results],1).mean(1),dim=-1); result["memory"]=results[-1]["memory"]; return result
