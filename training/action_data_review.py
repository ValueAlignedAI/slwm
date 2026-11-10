"""Resolve the sampled action contract; preserve the original schema probe."""
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.action_data_probe import ROOT,OUTPUT,OBJECTS,example_type,records,array
from training.research_sources import file_sha
from training.t2_real_av import write_json


def main():
    Example=example_type(); results=[]
    for i,payload in enumerate(records(ROOT/OBJECTS[-1][0])):
        feature=Example.FromString(payload).features.feature
        n=len(feature["steps/is_first"].int64_list.value)
        action=array(feature["steps/action"],7,n)
        command=array(feature["steps/action_dict/cartesian_position"],6,n)
        observed=array(feature["steps/observation/cartesian_position"],6,n)
        grip=array(feature["steps/action_dict/gripper_position"],1,n)
        pose_error=float(np.abs(action[:,:6]-command).max()); grip_error=float(np.abs(action[:,6:]-grip).max())
        assert pose_error==grip_error==0
        angular=command[:,3:]-observed[:,3:]; wrapped=np.arctan2(np.sin(angular),np.cos(angular))
        decoded=0
        for camera in ("wrist_image_left","exterior_image_1_left","exterior_image_2_left"):
            frames=feature["steps/observation/"+camera].bytes_list.value
            assert len(frames)==n
            for raw in frames:
                with Image.open(io.BytesIO(raw)) as image:
                    image.load(); assert image.size==(320,180) and image.mode=="RGB"
                decoded+=1
        np.savez_compressed(OUTPUT/("episode"+str(i+1))/"explicit_action_contract.npz",commanded_cartesian_pose=command,commanded_gripper_position=grip,observed_cartesian_pose=observed)
        results.append({"episode":i+1,"steps":n,"verified_camera_frames":decoded,"default_action_matches_commanded_cartesian_pose_max_error":pose_error,"default_gripper_matches_commanded_position_max_error":grip_error,"command_vs_observation_position_coordinate_max_difference":float(np.abs(command[:,:3]-observed[:,:3]).max()),"command_vs_observation_euler_raw_coordinate_max_difference":float(np.abs(angular).max()),"wrapped_euler_coordinate_max_difference":float(np.abs(wrapped).max()),"physical_timestamp_present":any("timestamp" in key for key in feature),"camera_count":3})
    write_json(OUTPUT/"action_contract_review.json",{"source_sha256":file_sha(Path(__file__)),"source_probe_report_sha256":file_sha(OUTPUT/"report.json"),"episodes":results,"contract_scope":"Only the two verified records from debug1.0.0, not a universal claim about DROID versions or all action fields","rotation_note":"XYZ Euler convention documented by dataset authors; wrapped coordinate differences are diagnostic, not SO(3) geodesic error","author_conventions_source":"https://github.com/droid-dataset/droid/issues/9","model_training":False,"robot_command_sent":False,"next_requirement":"Typed action representation, verified transition timing and environment-disjoint episodes before fitting physical dynamics"})
    print(json.dumps({"action_contract_verified":True,"results":results}),flush=True)


if __name__=="__main__": main()
