"""Prospectively freeze evaluation implementation, then unlock final data once."""
import argparse
import json
from pathlib import Path

from training.general_sources import write
from training.research_sources import file_sha

SOURCES=("configs/t2/general_readouts.json","configs/t2/general_speech_eval.json",
         "docs/research/general_calibration_protocol_20261106.md","docs/research/general_aligned_state_probe_20261106.md",
         "docs/research/vocoder_exposure_audit_20261106.md",
         "training/general_readout_data.py","training/general_frozen_features.py","training/general_readout_train.py","training/general_readout_eval.py",
         "training/general_speech_media.py","training/general_speech_score.py","training/general_speech_calibrate.py","training/general_speech_eval.py",
         "training/general_consequences.py","training/general_alignment_eval.py","training/general_live_eval.py","training/general_long_media.py",
         "training/general_evaluation_lock.py","training/general_report.py","tests/test_general_evaluation_lock.py")


def register(cfg):
    root=Path(cfg["root"]);hashes={name:file_sha(name) for name in SOURCES}
    path=root/"evaluation_source_hashes.json"
    if path.exists():assert json.loads(path.read_text())==hashes,"Evaluation implementation changed after registration"
    else:
        for name in SOURCES:
            target=root/"evaluation_source_snapshot"/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(name).read_bytes())
        write(path,hashes)
    return hashes


def freeze(cfg):
    root=Path(cfg["root"]);hashes=register(cfg)
    trained=json.loads((root/"training_complete.json").read_text());assert trained["all_weights_frozen"]
    readouts=json.loads((root/"readouts/training_complete.json").read_text());assert readouts["all_readouts_frozen"]
    calibration=json.loads((root/"speech/calibration.json").read_text());assert calibration["all_rules_frozen"] and calibration["fresh_final_people_used"]==0
    weights={}
    for run in cfg["runs"]:
        folder=root/run["id"];complete=json.loads((folder/"complete.json").read_text());digest=file_sha(folder/"endpoint.pt")
        assert complete["endpoint_sha256"]==digest;weights[run["id"]]=digest
    result={"core_endpoints":weights,"readouts_sha256":file_sha(root/"readouts/training_complete.json"),"speech_calibration_sha256":file_sha(root/"speech/calibration.json"),
            "dataset_contract_sha256":file_sha(Path(cfg["data_root"])/"dataset_contract.json"),"cases_sha256":file_sha(Path(cfg["data_root"])/"evaluation_cases.json"),
            "evaluation_sources":hashes,"all_compared_weights_and_rules_frozen":True,"final_observations_may_open_after_this_lock":True}
    path=root/"evaluation_lock.json"
    if path.exists():assert json.loads(path.read_text())==result
    else:write(path,result)
    print(json.dumps({"final_evaluation_unlocked":True,"core_count":len(weights)}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=("register","freeze"),required=True);args=parser.parse_args();cfg=json.loads(Path("configs/t2/general_transfer.json").read_text())
    register(cfg) if args.stage=="register" else freeze(cfg)


if __name__=="__main__":main()
