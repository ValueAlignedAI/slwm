"""Controlled small T2 runs on recorded, person-disjoint audiovisual data.

Uses the existing T2 model, fixed codecs, future-only feature targets and blind
unimodal contrastive encodes. Outputs are internal feature predictions, never
committed media, language, or behavior. All arms and gates are config controlled.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from training.t2_train_latents import T2AudioVisualSLWM, _module_parameter_counts
from training.t2_prepare_cremad import sha256


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    tmp.replace(path)


def stable_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_split(root: Path, split: str, device: torch.device) -> dict:
    """Load fixed audio/video ``FloatTensor[N,T,64]`` plus private metadata."""
    card = json.loads((root/"dataset_card.json").read_text())
    for suffix in (".npz", ".json"):
        path = root/(split+suffix)
        if sha256(path) != card["files"][path.name]:
            raise ValueError(f"Dataset hash mismatch: {path}")
    rows = json.loads((root/f"{split}.json").read_text())
    with np.load(root/f"{split}.npz", allow_pickle=False) as archive:
        audio = torch.from_numpy(archive["audio"].astype(np.float32)).to(device)
        visual = torch.from_numpy(archive["visual"].astype(np.float32)).to(device)
    if len(rows) != len(audio) or len(rows) != len(visual):
        raise ValueError("Feature/metadata sample counts differ")
    if not torch.isfinite(audio).all() or not torch.isfinite(visual).all():
        raise ValueError("Nonfinite corpus features")
    return {"audio": audio, "visual": visual, "rows": rows}


def batch_from_arrays(audio: torch.Tensor, visual: torch.Tensor, context: int, mode: str = "paired") -> dict:
    """[B,T,A/V] -> zero-hidden streams, [B,T] visibility/validity masks.

    Targets are separate detached tensors. No hidden values, labels or person
    identifiers can become input evidence through any processor path.
    """
    masks = {}
    result = {}
    for name, value in (("audio", audio), ("visual", visual)):
        visible = torch.zeros(value.shape[:2], dtype=torch.bool, device=value.device)
        if mode == "paired" or mode == name+"_only":
            visible[:, :context] = True
        valid = torch.ones_like(visible)
        loss_mask = valid.clone()
        loss_mask[:, :context] = False
        masks[name] = visible
        result.update({f"{name}_features": value.masked_fill(~visible.unsqueeze(-1), 0),
                       f"{name}_mask": visible, f"{name}_valid_mask": valid,
                       f"{name}_targets": value.detach(), f"{name}_loss_mask": loss_mask})
    return result


def model_config(cfg: dict, arm: str) -> dict:
    result = deepcopy(cfg["model"])
    result.update({"name": f"slwm-real-av-{arm}", "variant": arm})
    flags = result["architecture_flags"]
    if arm in {"vanilla", "no_spectral"}:
        flags["use_spectral_mixer"] = False
    if arm in {"vanilla", "no_longconv"}:
        flags["use_long_conv"] = False
    if arm == "vanilla":
        flags["use_local_temporal_mixer"] = False
    return result


def cluster_interval(values: np.ndarray, actors: list[str], *, samples: int = 2000, seed: int = 71) -> list[float]:
    """Person-cluster percentile 95% interval for the sample-weighted mean."""
    groups = [np.where(np.asarray(actors) == actor)[0] for actor in sorted(set(actors))]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(samples):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        means.append(float(np.mean(values[indices])))
    return [float(x) for x in np.quantile(means, [.025, .975])]


def within_person_permutation(rows: list[dict]) -> np.ndarray:
    """Derange each person's clips deterministically without changing people."""
    order = np.arange(len(rows))
    for actor in sorted({r["actor"] for r in rows}):
        indices = np.array([i for i,r in enumerate(rows) if r["actor"] == actor])
        if len(indices) < 2:
            raise ValueError("Within-person shuffle needs at least two clips per person")
        order[indices] = np.roll(indices, 1)
    return order


def retrieval_metrics(audio: np.ndarray, visual: np.ndarray, rows: list[dict]) -> dict:
    """Independent [N,D] encodes -> full-split bidirectional retrieval metrics."""
    audio = audio / np.linalg.norm(audio, axis=1, keepdims=True).clip(1e-8)
    visual = visual / np.linalg.norm(visual, axis=1, keepdims=True).clip(1e-8)
    similarities = audio @ visual.T
    n = len(rows)
    metrics = {"candidate_count": n, "chance_r1": 1/n, "encodes_are_independent": True}
    for name, scores in (("audio_to_video", similarities), ("video_to_audio", similarities.T)):
        ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)[np.arange(n), np.arange(n)]+1
        metrics[name] = {f"r{k}": float(np.mean(ranks <= k)) for k in (1,5,10)}
        metrics[name]["median_rank"] = float(np.median(ranks))
    order = within_person_permutation(rows)
    pos = similarities.diagonal()
    neg = similarities[np.arange(n), order]
    # Matched-choice accuracy and pooled binary AUC are separate statistics.
    metrics["within_person_choice_accuracy"] = float(np.mean((pos > neg)+.5*(pos == neg)))
    metrics["within_person_auc"] = float(np.mean((pos[:,None] > neg[None,:])+.5*(pos[:,None] == neg[None,:])))
    rng = np.random.default_rng(907)
    guesses = similarities.argmax(1)
    controls = [float(np.mean(guesses == rng.permutation(n))) for _ in range(200)]
    metrics["permuted_label_r1_mean"] = float(np.mean(controls))
    metrics["permuted_label_r1_95_interval"] = [float(v) for v in np.quantile(controls, [.025,.975])]
    return metrics


def _errors(output: dict, audio: torch.Tensor, visual: torch.Tensor, context: int) -> dict:
    return {"audio": ((output["audio_prediction"][:,context:].float()-audio[:,context:])**2).mean((1,2)),
            "visual": ((output["visual_prediction"][:,context:].float()-visual[:,context:])**2).mean((1,2))}


def evaluate(model: T2AudioVisualSLWM, data: dict, cfg: dict, *, detailed: bool = False) -> dict:
    model.eval()
    context = cfg["training"]["context_steps"]
    chunk = cfg["training"]["batch_size"]
    arm = model.variant
    observed_mode = arm if arm in {"audio_only", "visual_only"} else "paired"
    arrays = {key: [] for key in ("audio", "visual", "audio_absent_other", "visual_absent_other",
                                  "audio_shuffled_other", "visual_shuffled_other", "audio_zero", "visual_zero",
                                  "audio_persistence", "visual_persistence")}
    audio_reprs, visual_reprs = [], []
    order = torch.as_tensor(within_person_permutation(data["rows"]), device=data["audio"].device)
    amp = data["audio"].device.type == "cuda" and cfg["training"]["precision"] == "bf16"
    with torch.inference_mode(), torch.autocast(device_type=data["audio"].device.type, dtype=torch.bfloat16, enabled=amp):
        for begin in range(0, len(data["rows"]), chunk):
            end = begin+chunk
            a, v = data["audio"][begin:end], data["visual"][begin:end]
            batch = batch_from_arrays(a,v,context,observed_mode)
            errors = _errors(model(batch),a,v,context)
            for name in ("audio", "visual"):
                arrays[name].append(errors[name].float().cpu().numpy())
            if detailed:
                ar, vr = model.independent_representations(batch_from_arrays(a,v,context))
                audio_reprs.append(ar.float().cpu().numpy())
                visual_reprs.append(vr.float().cpu().numpy())
                for name, values in (("audio", a), ("visual", v)):
                    arrays[f"{name}_zero"].append((values[:,context:]**2).mean((1,2)).cpu().numpy())
                    arrays[f"{name}_persistence"].append(((values[:,context:]-values[:,context-1:context])**2).mean((1,2)).cpu().numpy())
                for target, other in (("audio", "visual"), ("visual", "audio")):
                    absent = batch_from_arrays(a,v,context,target+"_only")
                    errors = _errors(model(absent),a,v,context)
                    arrays[target+"_absent_other"].append(errors[target].float().cpu().numpy())
                    shuffled = batch_from_arrays(a,v,context,observed_mode)
                    indices = order[begin:end]
                    shuffled[other+"_features"] = data[other][indices].masked_fill(~shuffled[other+"_mask"].unsqueeze(-1), 0)
                    errors = _errors(model(shuffled),a,v,context)
                    arrays[target+"_shuffled_other"].append(errors[target].float().cpu().numpy())
    means = {key: float(np.concatenate(parts).mean()) for key,parts in arrays.items() if parts}
    result = {"errors": means, "prediction_score": .5*(means["audio"]+means["visual"]), "samples": len(data["rows"])}
    if detailed:
        actors = [r["actor"] for r in data["rows"]]
        result["retrieval"] = retrieval_metrics(np.concatenate(audio_reprs),np.concatenate(visual_reprs),data["rows"])
        contributions = {}
        for name in ("audio", "visual"):
            paired = np.concatenate(arrays[name])
            shuffled = np.concatenate(arrays[name+"_shuffled_other"])
            # Absolute paired differences are bootstrapped; relative gain is
            # scaled by the fixed aggregate control MSE for interpretability.
            improvement = (shuffled-paired)/max(float(shuffled.mean()),1e-8)
            contributions[name] = {"relative_improvement": float(improvement.mean()),
                                   "person_cluster_95_interval": cluster_interval(improvement, actors, samples=cfg["training"]["bootstrap_samples"])}
        result["cross_modal_contribution"] = contributions
        result["per_sample"] = [{"sample_id": row["sample_id"], "actor": row["actor"],
                                  **{k:float(np.concatenate(parts)[i]) for k,parts in arrays.items() if parts}}
                                 for i,row in enumerate(data["rows"])]
    return result


def train_run(cfg: dict, run: dict, datasets: dict, card: dict, root: Path) -> dict:
    seed, arm = run["seed"], run["arm"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    device = datasets["train"]["audio"].device
    model = T2AudioVisualSLWM(model_cfg=model_config(cfg,arm),
                             audio_feature_dim=datasets["train"]["audio"].shape[-1],
                             visual_feature_dim=datasets["train"]["visual"].shape[-1]).to(device)
    settings = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    path = root/run["experiment_id"]
    path.mkdir(parents=True, exist_ok=True)
    config = {"matrix": cfg, "run": run, "model": model_config(cfg,arm)}
    write_json(path/"config.json", config)
    provenance = {"source_commit": cfg["source_commit"], "working_tree_state": "research changes on recorded base",
                  "source_hashes": {str(p): sha256(p) for p in [Path(__file__), Path("training/t2_train_latents.py"),
                                                               Path("training/t2_prepare_cremad.py"), Path(cfg["protocol"])]},
                  "dataset_card_sha256": sha256(Path(cfg["corpus_dir"])/"dataset_card.json"),
                  "config_sha256": stable_hash(config)}
    registry = {"experiment_id": run["experiment_id"], "sprint": "T2", "status": "running", "arm": arm, "seed": seed,
                "parameter_accounting_mode": "all trainable adapters+processor+embeddings+heads; frozen encoder separate",
                "parameter_counts": _module_parameter_counts(model),
                "frozen_encoder_parameters": card["feature_spec"]["frozen_encoder_parameters"],
                "planned_steps": settings["steps"], "batch_size": settings["batch_size"],
                "dataset": str(Path(cfg["corpus_dir"])), "split_counts": card["split_counts"],
                "hardware": torch.cuda.get_device_name(0) if device.type=="cuda" else platform.platform(),
                "torch": torch.__version__, "precision": settings["precision"], **provenance}
    write_json(path/"registry.json", registry)
    initial = evaluate(model,datasets["validation"],cfg)
    curve = [{"step": 0, "validation": initial}]
    best_score, best_step = initial["prediction_score"], 0
    start = time.perf_counter()
    def save_checkpoint(step):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                    "step": step, "validation_score": best_score, "provenance": provenance,
                    "normalizer_path": str(Path(cfg["corpus_dir"])/"normalizer.npz"),
                    "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type=="cuda" else [],
                            "numpy": rng.bit_generator.state, "python": random.getstate()}}, path/"best.pt")
    save_checkpoint(0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    amp = device.type == "cuda" and settings["precision"] == "bf16"
    for step in range(1, settings["steps"]+1):
        model.train()
        indices = torch.as_tensor(rng.choice(len(datasets["train"]["rows"]),size=settings["batch_size"],replace=False), device=device)
        a, v = datasets["train"]["audio"][indices], datasets["train"]["visual"][indices]
        if arm == "shuffled":
            v = v.roll(1,0)
        context = settings["context_steps"]
        mode = arm if arm in {"audio_only", "visual_only"} else "paired"
        batch = batch_from_arrays(a,v,context,mode)
        if mode == "paired":
            dropped = torch.as_tensor(rng.random(len(a)), device=device)
            probability = settings["modality_dropout_probability"]
            for name, remove in (("audio", dropped < probability), ("visual", (dropped >= probability) & (dropped < 2*probability))):
                batch[name+"_mask"][remove] = False
                batch[name+"_features"][remove] = 0
        warmup = settings["warmup_steps"]
        rate = min(step/max(1,warmup),1.) * .5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,settings["steps"]-warmup)))
        for group in optimizer.param_groups:
            group["lr"] = settings["learning_rate"]*rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            output = model(batch)
            errors = _errors(output,a,v,context)
            loss = errors["audio"].mean()+errors["visual"].mean()
            alignment = loss.new_zeros(())
            if arm not in {"audio_only", "visual_only"}:
                ar, vr = model.independent_representations(batch_from_arrays(a,v,context))
                logits = F.normalize(ar.float(),dim=-1) @ F.normalize(vr.float(),dim=-1).T
                logits = logits * model.logit_scale.float().exp().clamp(max=100.)
                labels = torch.arange(len(a), device=device)
                alignment = .5*(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))
                loss = loss + settings["alignment_weight"]*alignment
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss in {run['experiment_id']} step {step}; stop")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(),settings["grad_clip"])
        if not torch.isfinite(grad):
            raise RuntimeError(f"Nonfinite gradient in {run['experiment_id']} step {step}; stop")
        optimizer.step()
        if step % 25 == 0:
            curve.append({"step":step, "loss":float(loss.detach()), "audio_mse":float(errors["audio"].mean().detach()),
                          "visual_mse":float(errors["visual"].mean().detach()), "alignment":float(alignment.detach()),
                          "gradient_norm":float(grad), "elapsed_seconds":time.perf_counter()-start})
        if step % settings["eval_every"] == 0 or step == settings["steps"]:
            validation = evaluate(model,datasets["validation"],cfg)
            curve.append({"step":step, "validation":validation})
            if validation["prediction_score"] < best_score:
                best_score, best_step = validation["prediction_score"], step
                save_checkpoint(step)
            write_json(path/"curves.json", {"curve":curve})
            print(json.dumps({"experiment_id":run["experiment_id"],"arm":arm,"step":step,"validation":validation,
                              "best_step":best_step,"elapsed_seconds":round(time.perf_counter()-start,2)}),flush=True)
    train_seconds = time.perf_counter()-start
    checkpoint = torch.load(path/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model"])
    validation = evaluate(model,datasets["validation"],cfg,detailed=True)
    test = evaluate(model,datasets["test"],cfg,detailed=True)
    gates = {}
    for name in ("audio", "visual"):
        e = test["errors"]
        control = min(e[name+"_zero"],e[name+"_persistence"])
        improvement = 1-e[name]/max(control,1e-8)
        contribution = test["cross_modal_contribution"][name]
        gates[name] = {"prediction_relative_improvement_over_stronger_null":improvement,
                       "prediction_pass": improvement >= cfg["gates"]["prediction_improvement"],
                       "cross_modal_pass": contribution["relative_improvement"] >= cfg["gates"]["cross_modal_improvement"]
                           and contribution["person_cluster_95_interval"][0] > 0}
    result = {"experiment_id":run["experiment_id"],"arm":arm,"seed":seed,"initial_validation":initial,
              "validation":validation,"test":test,"best_step":best_step,"completed_steps":settings["steps"],
              "samples_seen":settings["steps"]*settings["batch_size"],"training_seconds":train_seconds,
              "samples_per_second":settings["steps"]*settings["batch_size"]/train_seconds,
              "peak_gpu_memory_mb":torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else None,
              "parameter_counts":registry["parameter_counts"],"gates":gates,
              "checkpoint":str(path/"best.pt"),"checkpoint_sha256":sha256(path/"best.pt"),**provenance}
    write_json(path/"metrics.json",result)
    write_json(path/"curves.json",{"curve":curve})
    registry.update({"status":"completed","best_step":best_step,"completed_steps":settings["steps"],
                     "training_seconds":train_seconds,"checkpoint":result["checkpoint"],"checkpoint_sha256":result["checkpoint_sha256"]})
    write_json(path/"registry.json",registry)
    print(json.dumps({"completed":run["experiment_id"],"gates":gates,"test_errors":test["errors"],"retrieval":test["retrieval"]}),flush=True)
    del optimizer, model, checkpoint
    if device.type=="cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--only",help="Comma-separated experiment IDs; existing completed results are reused")
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["artifact_dir"])
    root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    corpus = Path(cfg["corpus_dir"])
    card = json.loads((corpus/"dataset_card.json").read_text())
    sets = [set(card["actor_splits"][s]) for s in ("train","validation","test")]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError("People overlap across splits")
    datasets = {s:load_split(corpus,s,device) for s in ("train","validation","test")}
    results = []
    wanted = set(args.only.split(",")) if args.only else None
    for run in cfg["runs"]:
        if wanted is not None and run["experiment_id"] not in wanted:
            continue
        metrics_path = root/run["experiment_id"]/"metrics.json"
        if metrics_path.exists():
            existing = json.loads(metrics_path.read_text())
            expected = stable_hash({"matrix":cfg,"run":run,"model":model_config(cfg,run["arm"])})
            if existing["config_sha256"] != expected:
                raise ValueError("Existing run config differs; choose a new cycle/ID")
            results.append(existing)
        else:
            results.append(train_run(cfg,run,datasets,card,root))
        write_json(root/"summary.json",{"config":cfg,"runs":results})


if __name__ == "__main__":
    main()
