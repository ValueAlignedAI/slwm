"""Train/selection-only supervision feasibility, independent of learned scores."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from training.broad_batch import StreamCorpus
from training.stream_codec import PACKET_SECONDS,character_ids
from training.research_sources import file_sha
from training.t2_real_av import write_json


def main():
    torch.set_num_threads(1); cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); result={"test_read":False,"model_scores_read":False,"ctc_frames_per_second":1/PACKET_SECONDS,"window_seconds":128*PACKET_SECONDS,"plans_seed":93217,"source_sha256":file_sha(Path(__file__)),"dataset_card_sha256":file_sha(Path(cfg["corpus"])/"dataset_card.json"),"splits":{}}
    for split,batches in (("train",100),("validation",8)):
        data=StreamCorpus(cfg,split); rng=np.random.default_rng(93217); groups=defaultdict(list)
        for _ in range(batches):
            for p in data.sample(rng,128):
                row=data.rows[p["name"]][p["index"]]; n=min(128,row["length"]-p["start"]); begin=p["start"]*PACKET_SECONDS; end=(p["start"]+n)*PACKET_SECONDS; labels=character_ids(" ".join(w for a,z,w in row["spans"] if a>=begin and z<=end)); minimum=len(labels)+sum(a==b for a,b in zip(labels,labels[1:])); _,units=data.arrays[p["name"]]; phones=np.asarray(units[row["offset"]+p["start"]:row["offset"]+p["start"]+n]); groups[p["name"]].append({"valid_packets":n,"minimum_ctc_steps":minimum,"empty_ctc":not labels,"impossible_ctc":minimum>n,"ctc_usable":row.get("ctc_alignment_usable",True),"unknown_phone_frames":int((phones==1).sum()),"non_silence_phone_frames":int((phones>0).sum())})
        result["splits"][split]={}
        for name,rows in groups.items():
            usable=[r for r in rows if r["ctc_usable"]]; nonempty=[r for r in usable if not r["empty_ctc"]]; result["splits"][split][name]={"windows":len(rows),"lexically_eligible_windows":len(usable),"empty_target_fraction":float(np.mean([r["empty_ctc"] for r in usable])),"impossible_ctc_fraction_of_eligible":float(np.mean([r["impossible_ctc"] for r in usable])),"impossible_ctc_fraction_of_nonempty":float(np.mean([r["impossible_ctc"] for r in nonempty])),"minimum_steps_to_available_ratio_quantiles":np.quantile([r["minimum_ctc_steps"]/r["valid_packets"] for r in nonempty],[.1,.5,.9,.99]).tolist(),"unknown_fraction_of_non_silence_phone_frames":sum(r["unknown_phone_frames"] for r in rows)/max(1,sum(r["non_silence_phone_frames"] for r in rows))}
    training=json.loads((Path(cfg["corpus"])/"ami_train/rows.json").read_text()); meetings=defaultdict(list)
    for row in training: meetings[row["id"].split("-")[0]].append(row["duration"])
    result["paired_training_diversity"]={"people":len({r["actor"] for r in training}),"meetings":len(meetings),"person_track_hours":sum(r["duration"] for r in training)/3600,"common_meeting_hours":sum(min(v) for v in meetings.values())/3600,"overlapping_tracks_are_not_independent_events":True}; write_json(Path(cfg["artifacts"])/"supervision_audit.json",result); print(json.dumps(result),flush=True)


if __name__=="__main__": main()
