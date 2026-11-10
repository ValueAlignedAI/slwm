"""Post-hoc strong simple controls; not part of the frozen neural-model gates.

Fit temporal means and a fixed-regularization linear future predictor using
training clips only. Report all results even when these controls beat SLWM.
They help diagnose normalized-time/identity shortcuts in this speech corpus.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np


def arrays(root: Path, split: str, context: int) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    with np.load(root/f"{split}.npz",allow_pickle=False) as data:
        audio,visual = data["audio"].astype(np.float64),data["visual"].astype(np.float64)
    x = np.concatenate([audio[:,:context].reshape(len(audio),-1),visual[:,:context].reshape(len(visual),-1)],axis=1)
    y = np.concatenate([audio[:,context:].reshape(len(audio),-1),visual[:,context:].reshape(len(visual),-1)],axis=1)
    anchor = np.concatenate([np.repeat(audio[:,context-1:context],audio.shape[1]-context,axis=1).reshape(len(audio),-1),
                             np.repeat(visual[:,context-1:context],visual.shape[1]-context,axis=1).reshape(len(visual),-1)],axis=1)
    return x,y,anchor


def fit_linear(x: np.ndarray, y: np.ndarray, *, regularization: float = 100.) -> dict[str,np.ndarray]:
    """[N,I], [N,O] -> train-only ridge weights [I,O] plus means."""
    xm,ym = x.mean(0),y.mean(0)
    xc,yc = x-xm,y-ym
    gram = xc @ xc.T
    gram.flat[::len(gram)+1] += regularization
    coef = xc.T @ np.linalg.solve(gram,yc)
    return {"input_mean":xm.astype(np.float32),"target_mean":ym.astype(np.float32),"coefficient":coef.astype(np.float32)}


def predict_linear(weights: dict[str,np.ndarray], x: np.ndarray) -> np.ndarray:
    """Observed flattened features [B,I] -> predicted flattened residual [B,O]."""
    return (x-weights["input_mean"]) @ weights["coefficient"] + weights["target_mean"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root,out = Path(cfg["corpus_dir"]),Path(cfg["artifact_dir"])/"posthoc_simple_controls"
    out.mkdir(parents=True,exist_ok=True)
    context = cfg["training"]["context_steps"]
    train = arrays(root,"train",context)
    values = {s:arrays(root,s,context) for s in ("validation","test")}
    half = train[1].shape[1]//2
    results = {"status":"posthoc_exploratory_control; does not alter preregistered gates",
               "regularization":100.,"regularization_selected_without_test_tuning":True,"controls":{}}
    def errors(prediction,target):
        return {"audio_mse":float(np.mean((prediction[:,:half]-target[:,:half])**2)),
                "visual_mse":float(np.mean((prediction[:,half:]-target[:,half:])**2))}
    mean = train[1].mean(0)
    results["controls"]["train_time_profile"] = {s:errors(np.broadcast_to(mean,v[1].shape),v[1]) for s,v in values.items()}
    input_half = train[0].shape[1]//2
    for name,selection in (("linear_joint",slice(None)),("linear_audio_only",slice(None,input_half)),("linear_visual_only",slice(input_half,None))):
        start = time.perf_counter()
        weights = fit_linear(train[0][:,selection],train[1]-train[2],regularization=100.)
        np.savez_compressed(out/(name+".npz"),**weights)
        scores = {}
        for split,v in values.items():
            prediction = v[2]+predict_linear(weights,v[0][:,selection])
            scores[split] = errors(prediction,v[1])
        # Single-modality inputs still use each target's own persistence anchor;
        # names identify the residual regressor input, not an absent-modality test.
        results["controls"][name] = {"input_dim":weights["coefficient"].shape[0],"output_dim":weights["coefficient"].shape[1],
                                     "fitted_coefficient_count":int(weights["coefficient"].size+weights["target_mean"].size),
                                     "fit_seconds":time.perf_counter()-start,**scores}
    results["interpretation_limit"] = ("Each linear control retains both targets' own observed persistence anchors; "
        "audio_only/visual_only name the residual-regressor inputs. No modality-deletion claim. These controls "
        "were added after discovery and some follow-up test outputs were seen; treat as exploratory, not new confirmation.")
    (out/"metrics.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__ == "__main__":
    main()
