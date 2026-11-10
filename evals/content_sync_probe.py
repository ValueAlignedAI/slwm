"""Train-only fitted field-word readouts; no raw bypass or label input to the core."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from training.content_sync_common import load_data,crop,get_codec,make_masks
from training.content_sync_train import load_model
from training.t2_real_av import write_json,cluster_interval
from training.research_sources import file_sha


def clock_features(starts,device):
    t=torch.as_tensor(starts,device=device)[:,None]/75+torch.arange(32,device=device)[None]/75
    return torch.stack((t,t*t,t*t*t,torch.sin(t*2*torch.pi),torch.cos(t*2*torch.pi),torch.sin(t*4*torch.pi),torch.cos(t*4*torch.pi)),-1)


def collect(model,codec,data,ids,device="cuda"):
    fields=[]; clocks=[]; labels=[]; masks=[]; records=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for begin in range(0,len(ids),32):
            ii=ids[begin:begin+32]; starts=(data["lengths"][ii]-32)//2; batch=crop(data,ii,starts); mask=make_masks(len(ii),8,device); obs=codec.observe(batch["mel100"],mask["visible"])
            field=model(obs)["field"].float(); fields.append(field.cpu()); clocks.append(clock_features(starts,device).cpu()); labels.append(batch["labels"].cpu()); masks.append(mask["target"].reshape(len(ii),32,4)[:,:,0].cpu()); records.extend(data["rows"][i] for i in ii)
    return torch.cat(fields),torch.cat(clocks),torch.cat(labels),torch.cat(masks),records


def fit(features,labels,classes):
    use=labels>0; x=features[use].double().cuda(); y=labels[use].cuda(); mean=x.mean(0); scale=x.std(0).clamp_min(.05); z=(x-mean)/scale; z=torch.cat((z,torch.ones(len(z),1,device=z.device,dtype=z.dtype)),1)
    targets=torch.nn.functional.one_hot(y,classes).double(); regularizer=torch.eye(z.shape[-1],device=z.device,dtype=z.dtype)*.01; regularizer[-1,-1]=0
    weights=torch.linalg.solve(z.T@z/len(z)+regularizer,z.T@targets/len(z))
    return {"mean":mean.cpu().numpy(),"scale":scale.cpu().numpy(),"weights":weights.cpu().numpy(),"fit_frames":len(z),"fitted_values":int(mean.numel()+scale.numel()+weights.numel())}


def score(probe,features,labels,mask,records,vocabulary):
    x=(features.double().numpy()-probe["mean"])/probe["scale"]; x=np.concatenate((x,np.ones((*x.shape[:-1],1))),-1); prediction=(x@probe["weights"]).argmax(-1); target=labels.numpy(); valid=(target>0)&mask.numpy()
    digit_names={"zero","one","two","three","four","five","six","seven","eight","nine"}; letter=[i for w,i in vocabulary.items() if len(w)==1]; digit=[i for w,i in vocabulary.items() if w in digit_names]
    rows=[]
    for i,r in enumerate(records):
        row={"sample_id":r["sample_id"],"actor":r["actor"]}
        for name,restriction in (("all",np.ones_like(target[i],bool)),("letters",np.isin(target[i],letter)),("digits",np.isin(target[i],digit))):
            use=valid[i]&restriction; count=int(use.sum()); correct=int(((prediction[i]==target[i])&use).sum()); row[name+"_count"]=count; row[name+"_correct"]=correct; row[name+"_accuracy"]=correct/count if count else None
        rows.append(row)
    summary={}
    for name in ("all","letters","digits"):
        use=[r for r in rows if r[name+"_count"]]; summary[name]={"scored_clips":len(use),"mean_clip_accuracy":float(np.mean([r[name+"_accuracy"] for r in use])) if use else None,"frame_accuracy":sum(r[name+"_correct"] for r in use)/max(1,sum(r[name+"_count"] for r in use))}
    return {"summary":summary,"per_clip":rows}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/t2/content_sync.json"); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); root=Path(cfg["artifacts"]); out=root/"field_content_probe"; out.mkdir(exist_ok=True); torch.set_num_threads(4)
    codec=get_codec(cfg); train=load_data(cfg,"train"); test=load_data(cfg,"test"); vocab=json.loads((Path(cfg["corpus"])/"vocabulary.json").read_text()); fit_ids=[]
    for actor in sorted({r["actor"] for r in train["rows"]}): fit_ids.extend([i for i,r in enumerate(train["rows"]) if r["actor"]==actor][:128])
    results={}; time_probe=None
    for run in cfg["base_runs"]:
        model,ck=load_model(root/run["id"]/"best.pt"); f,t,y,m,records=collect(model,codec,train,np.array(fit_ids)); probe=fit(f,y,len(vocab)); np.savez_compressed(out/(run["id"]+"_ridge.npz"),**probe)
        tf,tt,ty,tm,trows=collect(model,codec,test,np.arange(len(test["rows"]))); result=score(probe,tf,ty,tm,trows,vocab); result.update({"run":run,"training_clips":len(fit_ids),"fit_frames":probe["fit_frames"],"fitted_values":probe["fitted_values"],"checkpoint_sha256":file_sha(root/run["id"]/"best.pt")}); write_json(out/(run["id"]+".json"),result); results[run["id"]]=result
        if time_probe is None:
            time_probe=fit(t,y,len(vocab)); write_json(out/"time_only.json",score(time_probe,tt,ty,tm,trows,vocab)); np.savez_compressed(out/"time_only_ridge.npz",**time_probe)
        del model; torch.cuda.empty_cache(); print(json.dumps({"probed":run["id"],"accuracy":result["summary"]}),flush=True)
    comparisons={}
    for before,after in (("EXP-T2-701","EXP-T2-702"),("EXP-T2-703","EXP-T2-704")):
        old,new=results[before]["per_clip"],results[after]["per_clip"]; assert [r["sample_id"] for r in old]==[r["sample_id"] for r in new]
        changes={}
        for key in ("all","letters","digits"):
            pair=[(a,b) for a,b in zip(old,new) if a[key+"_count"] and b[key+"_count"]]
            if not pair:
                changes[key]={"absolute_accuracy_gain":None,"person_cluster_95_interval":None,"passes_5pp_positive_interval":False}; continue
            delta=np.array([b[key+"_accuracy"]-a[key+"_accuracy"] for a,b in pair]); interval=cluster_interval(delta,[b["actor"] for _,b in pair]); changes[key]={"absolute_accuracy_gain":float(delta.mean()),"person_cluster_95_interval":interval,"passes_5pp_positive_interval":bool(delta.mean()>=.05 and interval[0]>0)}
        comparisons[after+"_vs_"+before]=changes
    write_json(out/"summary.json",{"scope":"Fixed-ridge word readout from audio-only processed fields, trained only on training-person training-sentence examples; primary missing-region mean clip accuracy","comparisons":comparisons,"fit_ids":[train["rows"][i]["sample_id"] for i in fit_ids],"ridge_lambda":.01,"training_manifest_sha256":train["manifest_sha256"],"test_manifest_sha256":test["manifest_sha256"],"source_sha256":file_sha(Path(__file__))})


if __name__=="__main__": main()
