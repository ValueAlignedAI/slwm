"""A self-contained recorded-input operational demo; no training or quality gate."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from exploration.stream_media import read_audio,write_audio,encode_video
from training.research_sources import file_sha
from training.stream_codec import PACKET_SECONDS
from training.t2_real_av import write_json


def main():
    cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); root=Path(cfg["artifacts"])
    assert (root/"export_complete.json").exists()
    output=root/"live_demo"; output.mkdir(exist_ok=True)
    assert not(output/"complete.json").exists(),"Keep an existing demonstration immutable"
    row=json.loads((Path(cfg["corpus"])/"ami_test/rows.json").read_text())[0]
    n=31*32; first=round(80/PACKET_SECONDS); seconds=n*PACKET_SECONDS
    wave=read_audio(row["raw_path"],first*PACKET_SECONDS,seconds); assert len(wave)==n*1024
    frames=np.asarray(np.load(Path(cfg["corpus"])/"ami_test"/(row["id"]+"_pixels.npy"),mmap_mode="r")[first:first+n])
    sound=write_audio(output/"recorded_input.wav",wave)
    video=encode_video(output/"recorded_input.mp4",frames,output/"recorded_input.wav")
    bundle=root/"inference_bundle"
    command=[sys.executable,"-m","exploration.stream_sensing_infer","--world",str(bundle/"EXP-T2-801.pt"),
        "--actor",str(bundle/"EXP-P1-101.pt"),"--codec-dir",str(bundle),"--audio",str(output/"recorded_input.wav"),
        "--video",str(output/"recorded_input.mp4"),"--seconds",str(seconds),"--goal","both","--device","cpu","--pace",
        "--output",str(output/"replay")]
    write_json(output/"command.json",command)
    with (output/"replay.log").open("w") as log: subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    replay=json.loads((output/"replay/inference.json").read_text()); rows=json.loads((output/"replay/decisions.json").read_text())
    assert replay["automatic_decisions"]==30 and replay["weights_unchanged"] and replay["wall_seconds_after_bootstrap"]>=40.95
    assert all(r["observation_read_wall_seconds"]+1e-6>=r["observation_available_after_bootstrap_seconds"] for r in rows)
    write_json(output/"complete.json",{"portable_recorded_replay_passed":True,"source_participant":row["id"],"source_start_seconds":first*PACKET_SECONDS,
        "source_audio_sha256":row["raw_sha256"],"source_video_sha256":row["video_decode"]["raw_video_sha256"],
        "waveform":sound,"video":video,"source_sha256":file_sha(__file__),"recorded_inputs_are_generated":False,
        "automatic_decisions":len(rows),"missed_deadlines":sum(r["deadline_missed"] for r in rows),"weights_updated":False,
        "quality_gate":False,"scope":"Post-selection delivery fixture; same first fixed participant, independently decoded delivered files. File features prepared before paced replay; physical actions and live capture are not implemented."})
    (output/"README.md").write_text("# Recorded continuous-sensing demo\n\nThe two input files are42.325s of recorded AMI data, not generated media. This excerpt includes one bootstrap plus30 automatic listen/look/both/wait decisions. The delivered files were actually replayed on CPU using the exact exported world and actor; replay/ contains decisions, forecasts and invariant-weight/arrival-order evidence. Input features are prepared before replay. The separate P1 report finds no replicated utility advantage over the strongest fixed sensing strategy. This is an operational file replay, not physical robot control.\n\nRun the argv in command.json from the repository root, choosing a fresh output directory. The original corpus is not required for this excerpt. Portable weights and fitted codecs are in ../inference_bundle.\n")
    print(json.dumps({"portable_recorded_sensing_demo_passed":True,"decisions":30,"seconds":seconds}),flush=True)


if __name__=="__main__": main()
