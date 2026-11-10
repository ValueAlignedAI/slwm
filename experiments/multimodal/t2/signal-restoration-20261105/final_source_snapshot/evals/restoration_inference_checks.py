"""Real-file CPU checks and pixel-identical soundtrack counterfactuals."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from training.restoration_other import expanded_config
from training.t2_real_av import write_json


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--r1",required=True); parser.add_argument("--r2",required=True); args=parser.parse_args()
    c1=json.loads(Path(args.r1).read_text()); c2=expanded_config(args.r2); p1,p2=Path(c1["artifacts"]),Path(c2["artifacts"])
    other=json.loads((Path(c2["other_corpus"])/"other_test.json").read_text())[0]
    visible=p2/"exploration"/other["sample_id"]/"visible_mask.npy"
    def infer(checkpoint,audio,mask,out,video=None):
        command=[sys.executable,"-m","exploration.restoration_infer","--checkpoint",str(checkpoint),"--audio",str(audio),"--visibility",str(mask),"--output",str(out),"--device","cpu"]
        if video: command += ["--video",str(video),"--video-domain","ravdess"]
        subprocess.run(command,check=True); array=np.load(out/"predicted_mel.npy"); assert array.ndim==2 and array.shape[1]==64 and np.isfinite(array).all(); return array
    infer(p2/"EXP-T2-521/best.pt",other["raw_path"],visible,p2/"standalone_audio")
    infer(p1/"EXP-T2-507/best.pt",other["raw_path"],visible,p1/"standalone_audio")
    row=json.loads((Path(c1["new_corpus"])/"ravdess_test.json").read_text())[0]
    folder=p1/"exploration"/("ravdess_test_"+row["sample_id"]); audio=folder/"reference.wav"; mask=folder/"visible_mask.npy"
    predictions=[]; pixel_hashes=[]
    for label,volume in (("original_track","1"),("silent_track","0")):
        path=Path("/tmp")/("slwm_r2_"+label+".mkv")
        subprocess.run(["ffmpeg","-v","error","-y","-i",row["raw_path"],"-map","0:v:0","-map","0:a:0","-c:v","copy","-c:a","pcm_s16le","-af","volume="+volume,str(path)],check=True)
        frames=subprocess.run(["ffmpeg","-v","error","-i",str(path),"-an","-vf",c1["external_crop"],"-frames:v","128","-f","rawvideo","-pix_fmt","gray","-"],capture_output=True,check=True).stdout
        pixel_hashes.append(hashlib.sha256(frames).hexdigest())
        predictions.append(infer(p2/"EXP-T2-521/best.pt",audio,mask,p2/("standalone_av_"+label),path))
    assert pixel_hashes[0]==pixel_hashes[1]; np.testing.assert_array_equal(*predictions)
    old=torch.load(p1/"EXP-T2-501/best.pt",map_location="cpu",weights_only=False)["model"]
    zero=torch.load(p2/"EXP-T2-523/best.pt",map_location="cpu",weights_only=False)["model"]
    exact=all(torch.equal(value,zero[key]) for key,value in old.items())
    if not exact: raise ValueError("Zero-auxiliary control changed primary weights")
    write_json(p2/"inference_checks.json",{"actual_cpu_file_inference":["new auxiliary core","raw audio control","audio plus real video"],
        "soundtrack_counterfactual_exact":True,"identical_decoded_video_sha256":pixel_hashes[0],"zero_aux_control_primary_tensors_exact":exact,
        "fixed_audio_sample":other["sample_id"],"fixed_video_sample":row["sample_id"],"no_weights_updated":True})
    print(json.dumps({"real_file_checks":"passed","soundtrack_isolation":"exact","zero_aux_weight_control":"exact primary tensors"}),flush=True)


if __name__=="__main__": main()
