"""Portable runtime-pilot states, preserved failure lineage and bounded source proof."""
import json
from pathlib import Path

import numpy as np
import torch

from training.broad_stream_train import load_model
from training.research_sources import file_sha
from training.t2_real_av import write_json


def main():
    torch.set_num_threads(4)
    root=Path("experiments/multimodal/t2/compiled-learning-v2-20261105")
    sources=json.loads((root/"source_hashes.json").read_text())
    for name,digest in sources.items(): assert file_sha(name)==file_sha(root/"source_snapshot"/name)==digest,name
    selection=json.loads((root/"selection.json").read_text()); result={"sources_match":True,"models":{}}
    valid=torch.ones(1,32,dtype=torch.bool); no=torch.zeros_like(valid)
    for name,record in selection.items():
        path=root/name/"best.pt"; assert file_sha(path)==record["checkpoint_sha256"]
        model,ck=load_model(path,"cpu"); model.requires_grad_(False)
        assert sum(p.numel() for p in model.parameters())==record["parameters"]
        with torch.inference_mode():
            out=model(torch.zeros(1,32,400),torch.zeros(1,32,64),torch.zeros(1,32,dtype=torch.long),valid,no,no,valid,torch.zeros(1,64))
            assert all(torch.isfinite(value).all() for value in out.values())
        resume=torch.load(root/name/"resume.pt",map_location="cpu",weights_only=False)
        assert resume["step"]==400 and "optimizer" in resume and "data_rng" in resume
        result["models"][name]={"selected_sha256":file_sha(path),"parameters":record["parameters"],"selected_step":ck["step"],"actual_updates":400,"ordinary_eager_cpu_inference_passed":True,"compilation_required_for_inference":False}
        del model,ck,out,resume
    lineage=json.loads((root/"resume_lineage.json").read_text())
    for entry in lineage.values():
        assert file_sha(Path(entry["source"])/"resume.pt")==entry["resume_sha256"]
        assert file_sha(Path(entry["source"])/"best.pt")==entry["best_sha256"]
    result["preserved_resume_lineage"]=lineage
    old=Path("experiments/multimodal/t2/compiled-learning-20261105"); text=(old/"train.log").read_text(); assert "FailOnRecompileLimitHit" in text
    failure={"type":"Dynamo cache variant limit","log_sha256":file_sha(old/"train.log"),"original_sources_unchanged":True,"new_revision":str(root),"runs":{}}
    for folder in sorted(old.glob("EXP-T2-*")):
        curves=json.loads((folder/"curves.json").read_text()) if (folder/"curves.json").exists() else []
        steps=curves[-1]["step"] if curves else 0; entry=json.loads((folder/"registry.json").read_text()); entry.update(status="stopped_runtime_failure" if steps else "not_completed_after_runtime_failure",saved_updates=steps,continuation_root=str(root)); write_json(folder/"registry.json",entry); failure["runs"][folder.name]={"saved_updates":steps,"completed":False}
    write_json(old/"failure.json",failure); result["failed_original_pilot"]=failure
    probe=Path("experiments/policy/droid-source-probe-20261105"); raw=Path("artifacts/robotics/droid_debug_probe_v1"); report=json.loads((probe/"report.json").read_text()); review=json.loads((probe/"action_contract_review.json").read_text())
    for source in report["sources"]: assert file_sha(raw/source["file"])==source["sha256"]
    assert file_sha(probe/"report.json")==review["source_probe_report_sha256"]
    assert file_sha("training/action_data_probe.py")==json.loads((probe/"registered.json").read_text())["source_sha256"]
    assert file_sha("training/action_data_review.py")==review["source_sha256"]
    assert all(r["default_action_matches_commanded_cartesian_pose_max_error"]==r["default_gripper_matches_commanded_position_max_error"]==0 for r in review["episodes"])
    result["next_action_source"]={"model_fitted":False,"source_hashes_match":True,"episodes":len(review["episodes"]),"verified_frames":sum(r["verified_camera_frames"] for r in review["episodes"]),"typed_pose_gripper_contract_verified":True,"physical_timestamps_missing":True}
    write_json(root/"auxiliary_audit.json",result); print(json.dumps({"auxiliary_audit_passed":True,"models":len(result["models"]),"source":result["next_action_source"]}),flush=True)


if __name__=="__main__": main()
