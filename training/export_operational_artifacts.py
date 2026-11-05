"""Portable inference exports, full-state originals retained, content-hashed delivery."""
import json
from pathlib import Path
import shutil
import torch
from exploration.broad_stream_evaluate import vocabulary_bundle
from training.research_sources import file_sha
from training.t2_real_av import write_json


def main():
    cfg=json.loads(Path("configs/t2/broad_stream.json").read_text()); root=Path(cfg["artifacts"]); choices=json.loads((root/"selection.json").read_text()); bundle=vocabulary_bundle(cfg); manifest={"worlds":{},"actors":{},"external_vocoder":{"name":"charactr/vocos-mel-24khz","revision":"0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21","parameters":13531650},"text_scope":"Separate125M language weights; world text input is fixed CMU pronunciation"}
    for run in cfg["runs"]:
        name=run["id"]; source=root/name/"best.pt"; assert file_sha(source)==choices[name]["checkpoint_sha256"]; ck=torch.load(source,map_location="cpu",weights_only=False); target=bundle/(name+".pt"); payload={k:ck[k] for k in ("model","config","run","step","source_hashes","parameters","dataset_card_sha256")}; payload["selected_full_state_sha256"]=file_sha(source); torch.save(payload,target); manifest["worlds"][name]={"file":target.name,"sha256":file_sha(target),"full_state_sha256":file_sha(source),"parameters":ck["parameters"],"selected_step":ck["step"]}; del ck,payload
    policy=Path("experiments/policy/continuous-sensing-20261105"); actors=json.loads((policy/"selection.json").read_text())
    for world,entry in actors.items():
        ck=torch.load(policy/world/"actor.pt",map_location="cpu",weights_only=False); ck["training_world_checkpoint_sha256"]=ck["world_checkpoint_sha256"]; ck["world_checkpoint_sha256"]=manifest["worlds"][world]["sha256"]; target=bundle/(entry["id"]+".pt"); torch.save(ck,target); manifest["actors"][entry["id"]]={"world":world,"file":target.name,"sha256":file_sha(target),"original_actor_sha256":entry["actor_sha256"],"world_weights_identical":True}
        raw=entry["raw_control"]; ck=torch.load(policy/world/"raw/actor.pt",map_location="cpu",weights_only=False); ck["training_world_checkpoint_sha256"]=ck["world_checkpoint_sha256"]; ck["world_checkpoint_sha256"]=manifest["worlds"][world]["sha256"]; target=bundle/(raw["id"]+".pt"); torch.save(ck,target); manifest["actors"][raw["id"]]={"world":world,"file":target.name,"sha256":file_sha(target),"original_actor_sha256":raw["actor_sha256"],"world_weights_identical":True,"feature_mode":"raw"}
    external=Path("artifacts/decoders/vocos-mel-24khz"); (bundle/"vocos").mkdir(exist_ok=True)
    for name in ("config.yaml","pytorch_model.bin","provenance.json","README.md"):
        if (external/name).exists(): shutil.copyfile(external/name,bundle/"vocos"/name)
    manifest["external_vocoder"].update(weights_sha256=file_sha(bundle/"vocos/pytorch_model.bin"),config_sha256=file_sha(bundle/"vocos/config.yaml")); manifest["codec_files"]={name:file_sha(bundle/name) for name in ("codec.npz","duration_model.json","vocabulary.json","appearance_pixels.npy","appearance_code.npy")}; write_json(bundle/"manifest.json",manifest)
    textcfg=json.loads(Path("configs/t1/fineweb_extend_v2.json").read_text()); textroot=Path(textcfg["artifacts"]); text_choices=json.loads((textroot/"selection.json").read_text()); exports=textroot/"inference_exports"; exports.mkdir(exist_ok=True); text_manifest={}
    for run in textcfg["runs"]:
        name=run["id"]; source=textroot/name/"best.pt"; assert file_sha(source)==text_choices[name]["checkpoint_sha256"]; ck=torch.load(source,map_location="cpu",weights_only=False); payload={k:ck[k] for k in ("model","config","run","step","total_token_equivalent_step","provenance")}; target=exports/(name+".pt"); torch.save(payload,target); text_manifest[name]={"file":target.name,"sha256":file_sha(target),"full_state_sha256":file_sha(source),"training_tokens":ck["total_token_equivalent_step"]*16384}; del ck,payload
    write_json(exports/"manifest.json",text_manifest); write_json(root/"export_complete.json",{"world_manifest_sha256":file_sha(bundle/"manifest.json"),"text_manifest_sha256":file_sha(exports/"manifest.json"),"world_exports":len(manifest["worlds"]),"actor_exports":len(manifest["actors"]),"text_exports":len(text_manifest)}); print(json.dumps({"world_exports":len(manifest["worlds"]),"actor_exports":len(manifest["actors"]),"text_exports":len(text_manifest)}),flush=True)


if __name__=="__main__": main()
