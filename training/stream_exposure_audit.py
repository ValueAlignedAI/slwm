"""Replay sampled target exposure and check identity separation without model scores."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from training.broad_batch import StreamCorpus
from training.broad_data import prior_people
from training.research_sources import file_sha
from training.stream_codec import PACKET_SECONDS, words
from training.t2_real_av import write_json


def normalized_text(row):
    return " ".join(words(row["transcript"]))


def covered_packets(intervals):
    end=0; total=0
    for start,stop in sorted(intervals):
        total+=max(0,stop-max(start,end)); end=max(end,stop)
    return total


def main():
    torch.set_num_threads(1)
    cfg=json.loads(Path("configs/t2/broad_stream.json").read_text())
    root=Path(cfg["artifacts"]); corpus=Path(cfg["corpus"])
    selection=json.loads((root/"selection.json").read_text())
    rows={p.parent.name:json.loads(p.read_text()) for p in corpus.glob("*/rows.json")}
    people={name:{r["actor"] for r in values} for name,values in rows.items()}
    hashes={name:{r["raw_sha256"] for r in values} for name,values in rows.items()}
    texts={name:{normalized_text(r) for r in values if normalized_text(r)} for name,values in rows.items()}
    pairs={}
    names=sorted(rows)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            overlap=people[left]&people[right]
            raw_overlap=hashes[left]&hashes[right]
            assert not overlap,(left,right,overlap)
            assert not raw_overlap,(left,right,raw_overlap)
            pairs[left+"|"+right]={"participant_id_overlap":0,"raw_audio_sha256_overlap":0,
                                      "exact_nonempty_transcript_overlap":len(texts[left]&texts[right])}
    exposed={"LIBRI-"+p for p in prior_people()}
    fresh={name:len(people[name]&exposed) for name in ("libri_test-clean","libri_test-other")}
    assert not any(fresh.values()),fresh
    train=StreamCorpus(cfg,"train"); histories={}
    for seed in sorted({r["seed"] for r in cfg["runs"]}):
        runs=[r for r in cfg["runs"] if r["seed"]==seed]
        checkpoints={selection[r["id"]]["updates"] for r in runs}
        rng=np.random.default_rng(seed+50000); packets=Counter(); windows=Counter()
        modes=defaultdict(Counter); unique=defaultdict(set); coverage=defaultdict(list)
        for step in range(1,max(checkpoints)+1):
            for plan in train.sample(rng,cfg["training"]["batch"]):
                row=train.rows[plan["name"]][plan["index"]]
                count=min(cfg["training"]["window"],row["length"]-plan["start"])
                name=plan["name"]; packets[name]+=count; windows[name]+=1
                mode=plan["mode"]
                if mode=="text" and not row.get("text_generation_usable",True): mode="audio"
                modes[name][mode]+=count; unique[name].add(row["id"])
                coverage[(name,row["id"])].append((plan["start"],plan["start"]+count))
            if step not in checkpoints: continue
            matching=[r for r in runs if selection[r["id"]]["updates"]==step]
            for run in matching:
                ck=torch.load(root/run["id"]/"resume.pt",map_location="cpu",weights_only=False)
                assert ck["step"]==step and ck["data_rng"]==rng.bit_generator.state,run["id"]
                del ck
            histories[str(seed)+"|"+str(step)]={"runs":[r["id"] for r in matching],
                "exact_resume_rng_matches":True,"steps":step,
                "nominal_window_hours":step*cfg["training"]["batch"]*cfg["training"]["window"]*PACKET_SECONDS/3600,
                "valid_target_packet_hours":sum(packets.values())*PACKET_SECONDS/3600,
                "by_source":{name:{"windows":windows[name],"valid_target_packet_hours":packets[name]*PACKET_SECONDS/3600,
                    "unique_source_records":len(unique[name]),"available_source_records":len(train.rows[name]),
                    "unique_temporal_target_packet_hours":sum(covered_packets(v) for (source,_),v in coverage.items() if source==name)*PACKET_SECONDS/3600,
                    "available_target_packet_hours":sum(r["length"] for r in train.rows[name])*PACKET_SECONDS/3600,
                    "mode_target_packet_hours":{m:n*PACKET_SECONDS/3600 for m,n in modes[name].items()}}
                    for name in packets}}
        print(json.dumps({"exposure_seed_audited":seed}),flush=True)
    result={"model_scores_read":False,"source_sha256":file_sha(Path(__file__)),
            "dataset_card_sha256":file_sha(corpus/"dataset_card.json"),
            "identity_scope":"Official corpus participant IDs and exact raw-audio hashes; no biometric identity claim",
            "split_pairs":pairs,"prior_exposure_fresh_test_overlap":fresh,
            "transcript_overlap_policy":"Report exact nonempty strings, including common short utterances; no post-fit filtering",
            "exposure_definition":"Valid target coarse packets counted with repetition, independent of observation masks; union of sampled intervals additionally reports unique temporal coverage per source record. Nominal padded windows are separate. Parallel person tracks are not independent meeting events.",
            "sampled_training_exposure":histories}
    write_json(root/"exposure_identity_audit.json",result)


if __name__=="__main__": main()
