"""Disposable evaluation-path check on training observations, never final cases."""
import fcntl
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from models.slwm_general import GeneralWorld
from training.general_batch import Sources
from training.general_data import order
from training.general_long_media import encode_video
from training.general_sources import write
from training.general_speech_eval import baseline_files
from training.general_speech_media import SpeechRenderer


def main():
    torch.set_num_threads(4);cfg=json.loads(Path("configs/t2/general_transfer.json").read_text());root=Path(cfg["root"])/"evaluation_preflight";root.mkdir(exist_ok=True)
    source=Sources(cfg,"train");row=min([r for r in source.rows["speech"] if 8<=r["duration"]<=18 and r["ctc_alignment_usable"]],key=lambda r:order(r["id"]))
    with open("/tmp/slwm_research_gpu.lock","a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);model=GeneralWorld(variant="transformer",width=32,layers=1,heads=4,phone_count=79).cuda().eval().requires_grad_(False);renderer=SpeechRenderer(cfg)
        prepared=renderer.case(row,{"gap_seconds":.16});proposal=renderer.infer(model,prepared)
        files=renderer.save_many(prepared,{"direct":proposal["direct"],"predicted_phone":proposal["phonetic"]},root/"random_discarded_predictions")
        base=baseline_files(renderer,prepared,"speech",root/"training_only_codec_controls")
        for item in [*files.values(),*base.values()]:
            decoded,sr=sf.read(item["path"],dtype="float32");assert sr==24000
            if "silenced" not in item["path"] and "source" not in item["path"]:assert np.array_equal(decoded[~prepared["missing"]],prepared["sound"][~prepared["missing"]])
        # Independent encoder roundtrip checks native RGB dimensions and40.96s
        # duration without opening any held-out source or scoring an experiment.
        frames=np.zeros((960,3,16,16),np.float32);sound=np.zeros(960*1024,np.float32)
        from exploration.stream_media import write_audio
        write_audio(root/"synthetic_clock.flac",sound);video=encode_video(root/"synthetic_clock.mp4",frames,root/"synthetic_clock.flac")
        content=renderer.content_rollout(model,prepared)
    write(root/"complete.json",{"train_only_source_id":row["id"],"random_shared_model_discarded":True,"no_optimizer_updates":True,"no_final_cases_opened":True,"known_pcm_exact_after_flac_roundtrip":True,"video_clock":video,"whole_content_characters":len(content["character_ids"]),"r1_weights_were_frozen":True})


if __name__=="__main__":main()
