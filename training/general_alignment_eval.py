"""Unseen modality-combination retrieval with an explicit specialist text bridge."""
import fcntl
import gc
import json
from pathlib import Path

import numpy as np
import torch

from training.general_batch import Sources,base_observation,tensor_tree
from training.general_data import CACHE
from training.general_evaluation_lock import register
from training.general_sources import write
from training.general_train import initialize
from training.research_sources import file_sha


def normalize(values):return values/np.linalg.norm(values,axis=-1,keepdims=True).clip(1e-8)


def observations(source,family,rows):
    if family=="coco":
        images=np.stack([source.load(str(CACHE/"coco"/r["array"]))[r["index"]] for r in rows]).astype(np.float32).transpose(0,3,1,2)/127.5-1
        obs=base_observation(len(rows),1,1.);obs.update(images=images[:,None,None],image_present=np.ones((len(rows),1,1),bool));return obs
    t=source.cfg["context"];values=[]
    for row in rows:
        mel=np.asarray(source.load(str(CACHE/"clotho"/row["file"])),np.float32).reshape(-1,100)
        edges=np.linspace(0,len(mel),t*4+1).astype(int)
        coarse=np.stack([mel[edges[i]:max(edges[i]+1,edges[i+1])].mean(0) for i in range(t*4)]).reshape(t,4,100)
        values.append((coarse-source.stats["audio_mean"])/source.stats["audio_scale"])
    obs=base_observation(len(rows),t,1.);obs["time"]=np.stack([np.linspace(0,r["duration_seconds"],t,endpoint=False) for r in rows]).astype(np.float32)
    obs.update(audio=np.stack(values),audio_present=np.ones((len(rows),t,4),bool));return obs


def encode(model,source,family,rows):
    output=[]
    for start in range(0,len(rows),64):
        obs=observations(source,family,rows[start:start+64])
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):output.append(model(tensor_tree(obs,"cuda"))["embedding"].cpu().numpy())
    return np.concatenate(output)


def encode_text(model,source,strings):
    output=[]
    for start in range(0,len(strings),128):
        obs=source.text_observation(strings[start:start+128])
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):output.append(model(tensor_tree(obs,"cuda"))["embedding"].cpu().numpy())
    return np.concatenate(output)


def paired_caption_retrieval(values,captions):
    # Caption order is five per record; all five captions are positives.
    scores=values@captions.T
    order=np.argsort(-scores,axis=1,kind="stable")
    truth=np.arange(len(values))[:,None]
    return {"image_or_sound_to_text_recall1":float((order[:,:1]//5==truth).any(1).mean()),
            "image_or_sound_to_text_recall5":float((order[:,:5]//5==truth).any(1).mean()),
            "text_to_image_or_sound_recall1":float((scores.argmax(0)==np.arange(len(captions))//5).mean()),
            "records":len(values),"captions":len(captions),"independent_encoding":True}


def semantic_rows(scores,queries,images):
    result=[];classes=sorted({c for r in queries for c in r["labels"]})
    for query,value in zip(queries,scores):
        positive=np.array([bool(set(query["labels"])&set(image["labels"])) for image in images])
        order=np.argsort(-value,kind="stable");ranked=positive[order];count=int(positive.sum())
        precision=np.cumsum(ranked)/(np.arange(len(ranked))+1)
        chance={str(k):1-float(np.prod([(len(images)-count-j)/(len(images)-j) for j in range(k)])) for k in (1,5)}
        result.append({"id":query["id"],"group":query["group"],"labels":query["labels"],"hit1":float(ranked[:1].any()),"hit5":float(ranked[:5].any()),
                       "class_hit1":{label:float(label in images[int(order[0])]["labels"]) for label in query["labels"]},
                       "class_hit5":{label:float(any(label in images[int(i)]["labels"] for i in order[:5])) for label in query["labels"]},
                       "average_precision":float((precision*ranked).sum()/max(1,count)),"positive_images":count,"chance_hit":chance,"first_image":images[int(order[0])]["id"]})
    return result


def group_gain(candidate,baseline):
    by_id={r["id"]:r for r in baseline};groups=sorted({r["group"] for r in candidate})
    values=np.array([np.mean([r["hit1"]-by_id[r["id"]]["hit1"] for r in candidate if r["group"]==g]) for g in groups])
    rng=np.random.default_rng(7391);draws=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
    return {"uploader_groups":len(groups),"group_mean_hit1_gain":float(values.mean()),"gain_95ci":np.quantile(draws,[.025,.975]).tolist()}


def main():
    torch.set_num_threads(4);cfg=json.loads(Path("configs/t2/general_transfer.json").read_text());root=Path(cfg["root"]);register(cfg);assert (root/"evaluation_lock.json").exists()
    source=Sources(cfg,"test");out=root/"alignment";out.mkdir(exist_ok=True)
    semantic=json.loads((CACHE/"evaluation_cases.json").read_text())["semantic"]
    classes=list(semantic["aliases"]);names=[r["id"] for r in cfg["runs"] if r["id"] in {"G1-101","G1-102","G1-103","G1-104","G1-105","G1-106","G1-107","G1-108","G1-109","G1-114","G1-115"}]
    embeddings={};retrieval={}
    for run in (r for r in cfg["runs"] if r["id"] in names):
        path=out/(run["id"]+".npz")
        if path.exists():embeddings[run["id"]]=dict(np.load(path));retrieval[run["id"]]=json.loads((out/(run["id"]+"_caption_retrieval.json")).read_text());continue
        with open("/tmp/slwm_research_gpu.lock","a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);ck=torch.load(root/run["id"]/"endpoint.pt",map_location="cpu",weights_only=False,mmap=True);model=initialize(cfg,run).cuda().eval().requires_grad_(False);model.load_state_dict(ck["model"]);del ck
            current={"prototypes":encode_text(model,source,classes)};scores={}
            for family in ("coco","clotho"):
                current[family]=encode(model,source,family,source.rows[family])
                captions=encode_text(model,source,[s for r in source.rows[family] for s in r["captions"]])
                scores[family]=paired_caption_retrieval(current[family],captions)
            np.savez_compressed(path,**current);write(out/(run["id"]+"_caption_retrieval.json"),scores);embeddings[run["id"]]=current;retrieval[run["id"]]=scores
            del model;gc.collect();torch.cuda.empty_cache()
        print(json.dumps({"alignment_encoded":run["id"]}),flush=True)
    image_labels={r["id"]:r["labels"] for r in semantic["coco"] if r["role"]=="test"}
    images=[{"id":r["id"],"labels":image_labels.get(r["id"],[])} for r in source.rows["coco"]]
    query_meta={r["id"]:r for r in semantic["clotho"] if r["role"]=="test"}
    query_indices=[i for i,r in enumerate(source.rows["clotho"]) if r["id"] in query_meta];queries=[query_meta[source.rows["clotho"][i]["id"]] for i in query_indices]
    def pivot(values,prototypes):
        value=values@prototypes.T;return normalize(value-value.mean(1,keepdims=True))
    audio=embeddings["G1-115"];vision=embeddings["G1-114"]
    bridge=pivot(audio["clotho"][query_indices],audio["prototypes"])@pivot(vision["coco"],vision["prototypes"]).T
    bridge_rows=semantic_rows(bridge,queries,images);all_rows={"specialist_text_bridge":bridge_rows};summaries={}
    for name,current in embeddings.items():
        rows=semantic_rows(current["clotho"][query_indices]@current["coco"].T,queries,images);all_rows[name]=rows
        summaries[name]={"hit1":float(np.mean([r["hit1"] for r in rows])),"hit5":float(np.mean([r["hit5"] for r in rows])),"mean_average_precision":float(np.mean([r["average_precision"] for r in rows])),"gain_over_specialist_bridge":group_gain(rows,bridge_rows)}
    summaries["specialist_text_bridge"]={"hit1":float(np.mean([r["hit1"] for r in bridge_rows])),"hit5":float(np.mean([r["hit5"] for r in bridge_rows]))}
    for name,rows in all_rows.items():
        per_class={label:{"queries":sum(label in r["labels"] for r in rows),"hit1":float(np.mean([r["class_hit1"][label] for r in rows if label in r["labels"]])),"hit5":float(np.mean([r["class_hit5"][label] for r in rows if label in r["labels"]]))} for label in classes if any(label in r["labels"] for r in rows)}
        summaries[name]["per_class"]=per_class;summaries[name]["macro_class_hit1"]=float(np.mean([v["hit1"] for v in per_class.values()]))
    write(out/"semantic_cases.json",all_rows)
    write(out/"summary.json",{"caption_retrieval":retrieval,"audio_to_image":summaries,"queries":len(queries),"images":len(images),"categories":classes,"frozen_class_counts":semantic["counts"],"scope":"Semantic category agreement without direct sound/image training pairs; not matched physical events or a control-state test","specialist_bridge":"Two independently trained cores, fixed text category prototypes, per-record centered cosine profiles; no final fitting"})


if __name__=="__main__":main()
