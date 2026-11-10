"""Revision2: score lexical words only, excluding both sil and sp pause codes.

No fitting, model choice, threshold, spectrum or waveform changes. Original
ridge coefficients and primary results remain intact for an explicit audit.
"""
import json
from pathlib import Path
import numpy as np
import torch
from evals.content_sync_probe import collect,score
from training.content_sync_common import load_data,crop,get_codec,make_masks,hole_mse
from training.content_sync_train import load_model,predict
from training.t2_real_av import write_json,cluster_interval
from training.research_sources import file_sha


def main():
    cfg=json.loads(Path("configs/t2/content_sync.json").read_text()); root=Path(cfg["artifacts"]); dest=root/"word_evaluation_v2"; dest.mkdir(exist_ok=True); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=True
    vocab=json.loads((Path(cfg["corpus"])/"vocabulary.json").read_text()); pause=vocab["sp"]; codec=get_codec(cfg); data=load_data(cfg,"test"); results={}; original_labels=data["labels"].clone(); head_results={}; calibration=json.loads((root/"calibration.json").read_text())
    for run in cfg["base_runs"]:
        model,_=load_model(root/run["id"]/"best.pt"); f,t,y,m,rows=collect(model,codec,data,np.arange(len(data["rows"]))); mask=m&(y!=pause)
        with np.load(root/"field_content_probe"/(run["id"]+"_ridge.npz"),allow_pickle=False) as d: probe={k:d[k] for k in d.files}
        result=score(probe,f,y,mask,rows,vocab); results[run["id"]]=result; write_json(dest/(run["id"]+"_probe.json"),result); del model; torch.cuda.empty_cache()
    with np.load(root/"field_content_probe/time_only_ridge.npz",allow_pickle=False) as d: time_probe={k:d[k] for k in d.files}
    write_json(dest/"time_only.json",score(time_probe,t,y,mask,rows,vocab)); comparisons={}
    for before,after in (("EXP-T2-701","EXP-T2-702"),("EXP-T2-703","EXP-T2-704")):
        old,new=results[before]["per_clip"],results[after]["per_clip"]; changes={}
        for key in ("all","letters","digits"):
            pairs=[(a,b) for a,b in zip(old,new) if a[key+"_count"] and b[key+"_count"]]
            if not pairs: changes[key]={"absolute_accuracy_gain":None,"person_cluster_95_interval":None,"passes_5pp_positive_interval":False}; continue
            values=np.array([b[key+"_accuracy"]-a[key+"_accuracy"] for a,b in pairs]); ci=cluster_interval(values,[b["actor"] for _,b in pairs]); changes[key]={"absolute_accuracy_gain":float(values.mean()),"person_cluster_95_interval":ci,"passes_5pp_positive_interval":bool(values.mean()>=.05 and ci[0]>0)}
        comparisons[after+"_vs_"+before]=changes
    original=json.loads((root/"field_content_probe/summary.json").read_text()); write_json(dest/"probe_summary.json",{**original,"comparisons":comparisons,"evaluation_revision":2,"correction":"Exclude both sil and sp from lexical-word scoring; original train-fitted probe coefficients reused exactly","correction_source_sha256":file_sha(Path(__file__))})
    # Re-score only the word heads. Labels are not model inputs. Also verify
    # the spectrum predictions are numerically unchanged at the same batch size.
    data["labels"]=original_labels.masked_fill(original_labels==pause,-100)
    for run in cfg["base_runs"]+cfg["candidate_runs"]:
        model,ck=load_model(root/run["id"]/"best.pt"); kind=ck["kind"]; threshold=calibration[run["id"]]["threshold"] if run["id"] in calibration else None; records=[]; errors=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for begin in range(0,len(data["rows"]),16):
                ids=np.arange(begin,min(begin+16,len(data["rows"]))); starts=(data["lengths"][ids]-32)//2; batch=crop(data,ids,starts); masks=make_masks(len(ids),8,"cuda"); obs=codec.observe(batch["mel100"],masks["visible"]); out=predict(model,kind,obs,batch["mouth"],masks,threshold,torch.ones(len(ids),device="cuda",dtype=torch.bool)); unit=out["unit_logits"].argmax(-1); valid=(batch["labels"]>0)&masks["target"].reshape(len(ids),32,4)[:,:,0]
                correct=((unit==batch["labels"])&valid).sum(1).cpu().numpy(); count=valid.sum(1).cpu().numpy(); errors.extend(hole_mse(out["prediction"],obs["target"],masks["target"]).cpu().numpy().tolist())
                records.extend({"sample_id":data["rows"][i]["sample_id"],"actor":data["rows"][i]["actor"],"correct":int(c),"frames":int(n)} for i,c,n in zip(ids,correct,count))
        old=json.loads((root/run["id"]/"test_gap8.json").read_text()); assert abs(np.mean(errors)-old["metrics"]["correct_hole_mse"])<2e-7
        head_results[run["id"]]={"correct_word_correct":float(np.mean([r["correct"] for r in records])),"correct_word_frames":float(np.mean([r["frames"] for r in records])),"correct_word_accuracy":sum(r["correct"] for r in records)/sum(r["frames"] for r in records)}; write_json(dest/(run["id"]+"_head.json"),{"metrics":head_results[run["id"]],"per_clip":records,"unchanged_hole_mse":float(np.mean(errors))}); del model; torch.cuda.empty_cache()
    write_json(dest/"head_metrics.json",head_results); write_json(root/"word_metric_correction.json",{"revision":2,"reason":"The GRID vocabulary contains a short-pause sp code; original labels>0 counted19 pause frames in the primary gap as words","excluded_primary_pause_frames":19,"neural_weights_changed":False,"thresholds_changed":False,"ridge_coefficients_refitted":False,"spectral_scores_unchanged":True,"waveforms_unchanged":True,"original_results_retained":True,"source_sha256":file_sha(Path(__file__))}); print(json.dumps({"lexical_word_correction_complete":True,"comparisons":comparisons}))


if __name__=="__main__": main()
