"""Load a trained T2 artifact and predict internal audio/video feature futures."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.t2_prepare_cremad import decode_clip, log_mel_sequence, sha256
from training.t2_real_av import batch_from_arrays, model_config
from training.t2_train_latents import T2AudioVisualSLWM


def predict(checkpoint_path: str, clip_path: str, output_path: str, mode: str = "paired", device_name: str = "cuda") -> dict:
    from torchvision.models import resnet18, ResNet18_Weights
    from torchaudio.transforms import MelSpectrogram

    torch.set_num_threads(4)
    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=False)
    config, run = checkpoint["config"]["matrix"], checkpoint["config"]["run"]
    if not Path(clip_path).is_file():
        raise FileNotFoundError(f"Input clip does not exist: {clip_path}")
    corpus = Path(config["corpus_dir"])
    if sha256(corpus/"dataset_card.json") != checkpoint["provenance"]["dataset_card_sha256"]:
        raise ValueError("Checkpoint and preprocessing dataset card do not match")
    card = json.loads((corpus/"dataset_card.json").read_text())
    normalizer_path = Path(checkpoint["normalizer_path"])
    if sha256(normalizer_path) != card["files"]["normalizer.npz"]:
        raise ValueError("Checkpoint normalizer file has changed")
    with np.load(normalizer_path,allow_pickle=False) as archive:
        norm = {k:torch.from_numpy(archive[k]).to(device) for k in archive.files}
    context = config["training"]["context_steps"]
    frames, waveform, duration = decode_clip(Path(clip_path),config["model"]["context_length"]//2)
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    backbone.fc = torch.nn.Identity()
    mel = MelSpectrogram(sample_rate=16000,n_fft=400,hop_length=160,n_mels=64,center=False,power=2).to(device)
    mean = torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    with torch.inference_mode():
        # Only observed frames enter the frozen encoder; all future visual
        # slots are zeros. Audio context windows cannot see future waveform.
        x = torch.from_numpy(frames[:context]).to(device).permute(0,3,1,2).float()/255.
        x = F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False)
        v = ((backbone((x-mean)/std)-norm["visual_mean"]) @ norm["visual_pca"])/norm["visual_std"]
        v = F.pad(v,(0,0,0,len(frames)-context))[None]
        a = log_mel_sequence(torch.from_numpy(waveform).to(device),mel,len(frames))
        a = ((a-norm["audio_mean"])/norm["audio_std"])[None]
        model = T2AudioVisualSLWM(model_cfg=model_config(config,run["arm"]),audio_feature_dim=a.shape[-1],visual_feature_dim=v.shape[-1]).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        output = model(batch_from_arrays(a,v,context,mode))
    path = Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,
        observed_audio=a[0,:context].cpu().numpy(), observed_visual=v[0,:context].cpu().numpy(),
        predicted_audio=output["audio_prediction"][0,context:].float().cpu().numpy(),
        predicted_visual=output["visual_prediction"][0,context:].float().cpu().numpy())
    report = {"checkpoint":checkpoint_path,"checkpoint_sha256":sha256(Path(checkpoint_path)),
              "sample":clip_path,"source_sha256":sha256(Path(clip_path)),"duration_seconds":duration,
              "observed_mode":mode,"context_steps":context,"prediction_steps":len(frames)-context,
              "artifact":str(path),"output_tag":"predicted_internal_features_uncommitted",
              "limitations":"Feature predictions only; no raw audio/video decoder or general-world-model claim"}
    path.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--clip",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--mode",choices=("paired","audio_only","visual_only"),default="paired")
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint,args.clip,args.output,args.mode,args.device),indent=2))
