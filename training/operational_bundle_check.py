"""Exercise portable exports through ordinary file CLIs without fitting weights."""
import json
from pathlib import Path
import subprocess
import sys
import time

import soundfile as sf

from training.research_sources import file_sha
from training.t2_real_av import write_json


def main():
    cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); root=Path(cfg["artifacts"])
    assert (root/"export_complete.json").exists()
    bundle=root/"inference_bundle"; output=root/"bundle_file_checks"; output.mkdir(exist_ok=True)
    assert not(output/"complete.json").exists(),"Preserve completed interface checks"
    world=bundle/"EXP-T2-801.pt"; actor=bundle/"EXP-P1-101.pt"
    row=json.loads((Path(cfg["corpus"])/"ami_test/rows.json").read_text())[0]
    speech=json.loads((Path(cfg["corpus"])/"libri_test-clean/rows.json").read_text())[0]
    commands=[]
    def run(label,module,arguments):
        command=[sys.executable,"-m",module,*map(str,arguments)]
        started=time.perf_counter()
        with (output/(label+".log")).open("w") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        commands.append({"label":label,"argv":command,"elapsed_seconds":time.perf_counter()-started})
        write_json(output/"commands.json",commands)
        print(json.dumps({"portable_cli_completed":label}),flush=True)
    shared=["--checkpoint",world,"--codec-dir",bundle,"--device","cpu"]
    for label,arguments in (
        ("audio_to_video",["--mode","audio-to-video","--audio",row["raw_path"],"--start",80,"--seconds",40.96]),
        ("video_to_audio",["--mode","video-to-audio","--video",row["video_path"],"--start",80,"--seconds",40.96]),
        ("text_to_av",["--mode","text-to-av","--text",cfg["text_prompts"][0]]),
        ("repair",["--mode","repair","--audio",speech["raw_path"],"--seconds",8,"--gaps","2:2.16","--preserve-mel-context"]),
    ):
        run(label,"exploration.broad_stream_infer",[*shared,*arguments,"--output",output/label])
        record=json.loads((output/label/"inference.json").read_text())
        assert record["checkpoint_sha256"]==file_sha(world) and record["device"]=="cpu"
        if label!="repair":
            info=sf.info(output/label/"output.wav"); assert info.duration>30
            assert record["video"]["frames"]*1024==info.frames
        else: assert record["observed_pcm_exact"] and record["observed_fine_mel_context_preserved"]
        if label=="video_to_audio": assert record["source_audio_decoded"] is False
    run("paced_actor","exploration.stream_sensing_infer",["--world",world,"--actor",actor,"--codec-dir",bundle,
        "--audio",row["raw_path"],"--video",row["video_path"],"--start",80,"--seconds",31*32*1024/24000,
        "--goal","both","--device","cpu","--pace","--output",output/"paced_actor"])
    paced=json.loads((output/"paced_actor/inference.json").read_text()); decisions=json.loads((output/"paced_actor/decisions.json").read_text())
    assert paced["automatic_decisions"]==30 and paced["wall_seconds_after_bootstrap"]>=40.95
    assert paced["weights_unchanged"] and not paced["future_ground_truth_used"] and not paced["transcript_used"]
    assert all(r["observation_read_wall_seconds"]+1e-6>=r["observation_available_after_bootstrap_seconds"] for r in decisions)
    textroot=Path("experiments/text/t1/extended-throughput-20261105/inference_exports")
    for name in ("EXP-T1-221","EXP-T1-222"):
        run(name,"exploration.scale_text_infer",["--checkpoint",textroot/(name+".pt"),"--prompt","The experiment shows",
            "--tokens",16,"--device","cpu","--output",output/(name+".json")])
        assert json.loads((output/(name+".json")).read_text())["training_tokens_at_checkpoint"]==150994944
    write_json(output/"complete.json",{"portable_cli_checks_passed":True,"commands":len(commands),"all_devices":"cpu",
        "world_export_sha256":file_sha(world),"actor_export_sha256":file_sha(actor),"source_sha256":file_sha(__file__),
        "source_participant":row["id"],"source_start_seconds":80,"fresh_test_used_for_weight_or_threshold_selection":False,
        "long_generation_modes":["audio_to_video","video_to_audio","text_to_av"],"paced_automatic_decisions":30,
        "missed_actor_deadlines":sum(r["deadline_missed"] for r in decisions),"weights_updated":False,
        "scope":"Interface and delivery checks; no additional quality gate or new trained candidate"})
    print(json.dumps({"portable_bundle_check_passed":True}),flush=True)


if __name__=="__main__": main()
