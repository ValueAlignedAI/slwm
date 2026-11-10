"""Structured registry records, preserving preregistration and actual completion."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
from training.research_sources import file_sha
from training.t2_real_av import write_json


def read(path): return json.loads(Path(path).read_text()) if Path(path).exists() else None


def main():
    p=argparse.ArgumentParser(); p.add_argument("--complete",action="store_true"); p.add_argument("--text-only",action="store_true"); a=p.parse_args(); date=datetime.now(timezone.utc).date().isoformat(); commit=subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip(); source=file_sha(Path(__file__)); entries=[]
    for config in ("configs/t2/broad_stream.json","configs/t1/fineweb_extend_v2.json"):
        cfg=read(config); root=Path(cfg["artifacts"]); text=cfg["sprint"].startswith("T1"); corpus=Path(read(cfg["parent_config"])["corpus"] if text else cfg["corpus"]); card=corpus/"dataset_card.json"
        if a.text_only and not text: continue
        for run in cfg["runs"]:
            folder=root/run["id"]; folder.mkdir(exist_ok=True); complete=read(folder/"complete.json"); saved=read(folder/"config.json"); parameters=(complete or {}).get("parameters",(saved or {}).get("parameters",((saved or {}).get("provenance") or {}).get("parameters"))); state="completed" if a.complete and complete else "running" if saved else "planned"; entry={"experiment_id":run["id"],"created_at":"2026-11-05","updated_at":date,"status":state,"sprint":{"id":"T1" if text else "T2","name":cfg["sprint"],"owner_role":"training"},"claim_trace":{"hypothesis_ids":["G-R0-1","H-R0-3"] if text else ["H-R0-1","H-R0-2","H-R0-3"],"expected_decision":"untested"},"repository":{"git_commit_at_registry_write":commit,"working_tree_state":"source snapshots authoritative; unrelated edits may exist","source_snapshot":str(root/"source_snapshot"),"source_hashes":read(root/"source_hashes.json")},"config":{"config_path":config,"config_sha256":file_sha(config),"protocol":cfg["protocol"],"protocol_sha256":file_sha(cfg["protocol"]),"seed":run["seed"],"precision":"bf16 training; fp32 parameters/Adam states","deterministic_data_sampling":True,"bitwise_gpu_determinism_claimed":False},"model":{"variant":run["variant"],"neural_parameters":parameters,"external_renderer_parameters":0 if text else 13531650,"parameter_accounting_mode":"strict" if text else "approximate","scope":"Exact trained-neural counts at prototype size; fitted codecs and fixed dictionary separate" if not text else "All text weights, including tied embeddings; no pretrained LM initialization"},"data":{"corpus":str(corpus),"dataset_card_sha256":file_sha(card),"split_notes":"Domain-disjoint and deduplicated; fixed fresh test windows" if text else "Official participant IDs disjoint; fresh Libri people exclude prior SLWM exposure; separate world selection/head selection/calibration/test","test_selection_used_for_fitting":False},"training":{"registered_config":cfg,"actual":complete,"checkpoints":{"best":str(folder/"best.pt"),"resume":str(folder/"resume.pt"),"last":str(folder/"last.pt")}},"evaluation":{"script":"training/extend_text_train_v2.py" if text else "exploration/broad_stream_evaluate.py","metrics":read(folder/"test.json") if a.complete and text else str(root/"speech_summary.json") if a.complete else None,"all_cases_retained":True},"interpretation":{"hypothesis_decision":"untested" if not a.complete else "See frozen gates and complete reports; no generic world-model or convergence claim","known_limits":["Two-seed early language learning, not convergence"] if text else ["CMU phonetic text edge; separate125M language weights","64x64 face/PCA video and sparse own-speaker paired speech","Recorded sensing is not physical-action control"]},"registry_writer_sha256":source}; write_json(folder/"registry.json",entry); entries.append(entry["experiment_id"])
    cfg=read("configs/policy/stream_sensing.json"); root=Path(cfg["artifacts"]); root.mkdir(parents=True,exist_ok=True)
    for world in ([] if a.text_only else cfg["world_runs"]):
        folder=root/world; folder.mkdir(exist_ok=True); selected=read(folder/"frozen.json"); summary=read(root/"summary.json") if a.complete else None; write_json(folder/"registry.json",{"experiment_id":cfg["policy_ids"][world],"created_at":"2026-11-05","updated_at":date,"status":"completed" if a.complete and summary else "running" if selected else "planned","sprint":{"id":"P1","name":"P1-S1 exploratory sensing","owner_role":"training"},"claim_trace":{"hypothesis_ids":["H-R0-4"],"scope":"Recorded sensor utility, not hallucination reduction or physical control"},"repository":{"git_commit_at_registry_write":commit,"source_hashes":read(root/"source_hashes.json")},"config":{"path":"configs/policy/stream_sensing.json","sha256":file_sha("configs/policy/stream_sensing.json"),"seed":cfg["seed"]},"model":{"world":world,"actor_neural_parameters":(selected or {}).get("parameters"),"world_weights_frozen":True},"data":{"risk_head_fit_people":22,"risk_head_selection_people":6,"independent_calibration_people":4,"final_test_people":12},"training":{"protocol":cfg["protocol"],"protocol_sha256":file_sha(cfg["protocol"]),"actual":read(folder/"fit.json"),"best_optimizer":str(folder/"best_optimizer.pt"),"last_optimizer":str(folder/"last_optimizer.pt")},"evaluation":{"utility_counts_all_ticks":True,"metrics":summary[world] if summary else None},"interpretation":{"physical_world_control":False,"language_hallucination_claim":False,"decision":"See registered cost/coverage gate"},"registry_writer_sha256":source}); entries.append(cfg["policy_ids"][world])
        raw_folder=folder/"raw"; raw_folder.mkdir(exist_ok=True)
        raw_selected=read(raw_folder/"frozen.json")
        raw_entry=read(folder/"registry.json")
        raw_entry.update(experiment_id=cfg["raw_policy_ids"][world],status="completed" if a.complete and summary else "running" if raw_selected else "planned")
        raw_entry["claim_trace"]={"hypothesis_ids":["H-R0-4"],"scope":"Matched direct-sensor statistics control; same fitted counterfactual targets and people"}
        raw_entry["model"].update(actor_neural_parameters=(raw_selected or {}).get("parameters"),feature_mode="raw",learned_world_features_used=False,world_used_for_training_targets_and_scoring=True)
        raw_entry["training"].update(actual=read(raw_folder/"fit.json"),best_optimizer=str(raw_folder/"best_optimizer.pt"),last_optimizer=str(raw_folder/"last_optimizer.pt"),same_sample_trajectories_as_field_actor=True)
        raw_entry["evaluation"]["strategy"]="raw"
        raw_entry["interpretation"]["decision"]="Compare shared-field actor against this adaptive control as well as fixed strategies"
        write_json(raw_folder/"registry.json",raw_entry); entries.append(cfg["raw_policy_ids"][world])
    destination=Path("experiments/text/t1/extended-throughput-20261105" if a.text_only else "experiments/multimodal/t2/broad-stream-20261105")
    write_json(destination/("registry_complete.json" if a.complete else "registry_registered.json"),{"entries":entries,"date":date,"final_scores_read":a.complete,"git_commit":commit,"source_sha256":source,"text_only":a.text_only}); print(json.dumps({"registry_entries":len(entries),"completed_mode":a.complete,"text_only":a.text_only}),flush=True)


if __name__=="__main__": main()
