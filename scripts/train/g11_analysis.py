"""Balance device families, pair the same recording decisions and preserve every seed and null result."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from g11_store import roots
from g9_storage import write

FOLDS = ("A", "B")
SEEDS = (1001, 1002, 1003)
CORES = ("hybrid", "gpt2", "gated")
TREAT = ("paired", "deranged", "absent")
SCORES = (
    "verb",
    "exact",
    "open_exact",
    "effect_mae",
    "magnitude_mm",
    "device_mae",
    "isolated_mae",
    "device_isolated_mae",
)


def read(p):
    return json.loads(p.read_bytes())


@lru_cache(maxsize=256)
def resamples(n, seed):
    return np.random.default_rng(seed).integers(0, n, (4000, n))


def estimate(v, groups, families, ci=True):
    x = np.asarray(v, np.float64)
    ok = np.isfinite(x)
    ids = np.unique(groups[ok])
    if not len(ids):
        return None
    den = np.bincount(groups[ok], minlength=len(families))
    sums = np.bincount(groups[ok], weights=x[ok], minlength=len(families))
    group_values = sums[ids] / den[ids]
    fam = families[ids]
    means = {}
    sample = []
    for label in sorted(set(fam)):
        values = group_values[fam == label]
        means[str(label)] = float(values.mean())
        if ci:
            seed = int(
                hashlib.sha256(
                    ("G11/record-bootstrap/" + str(label)).encode()
                ).hexdigest()[:8],
                16,
            )
            sample.append(values[resamples(len(values), seed)].mean(1))
    result = {
        "mean": float(np.mean(list(means.values()))),
        "events": int(ok.sum()),
        "recordings": len(ids),
        "families": means,
    }
    if ci:
        result["ci95"] = np.quantile(np.mean(sample, axis=0), [0.025, 0.975]).tolist()
    return result


def label_data(banks, fold, role, grades):
    meta = read(banks / f"{fold}-{role}.json")
    with np.load(banks / f"{fold}-{role}.npz") as file:
        labels = {
            k: file[k] for k in ("verb", "noun", "open_noun", "group", "conflict")
        }
    with np.load(banks / f"{fold}-{role}-effects.npz") as file:
        labels.update(
            {
                k: file[k]
                for k in (
                    "effect",
                    "valid",
                    "strict",
                    "pre_motion",
                    "neutral",
                    "background",
                )
            }
        )
    families = {r["recording"]: r["family"] for r in meta["rows"]}
    names = meta["records"]
    labels["families"] = np.array([families.get(n, "no_eligible_event") for n in names])
    labels["device"] = np.array(
        [grades.get((r["recording"], r["action_id"])) == "D" for r in meta["rows"]]
    )
    labels["strong_conflict"] = labels["conflict"].astype(bool)
    return labels


def metric(data, pred):
    y = data["verb"]
    n = data["noun"]
    v = (y >= 0) & (pred["verb"] == y)
    noun = (n >= 0) & (pred["noun"] == n)
    on = (data["open_noun"] >= 0) & (pred["open_noun"] == data["open_noun"])
    out = dict(
        verb=v.astype(float),
        noun=noun.astype(float),
        exact=(v & noun).astype(float),
        open_exact=(v & on).astype(float),
    )
    if "verb_nll" in pred:
        out.update(verb_nll=pred["verb_nll"], verb_top3=pred["verb_top3"].astype(float))
    out["conflict_verb"] = np.where(data["strong_conflict"], v, np.nan)
    if "effect" in pred:
        target = data["effect"]
        err = np.abs(pred["effect"][:, :2].sum(1) - target[:, :2].sum(1))
        valid = data["valid"]
        device = valid & data["device"]
        strict = valid & data["strict"]
        for key, mask in (
            ("effect_mae", valid),
            ("device_mae", device),
            ("isolated_mae", strict),
            ("device_isolated_mae", device & strict),
        ):
            out[key] = np.where(mask, err, np.nan)
        out["magnitude_mm"] = np.where(
            valid, 100 * np.abs(pred["effect"][:, 2] - target[:, 2]), np.nan
        )
        out["near_mae"] = np.where(
            valid, np.abs(pred["effect"][:, 0] - target[:, 0]), np.nan
        )
        out["far_mae"] = np.where(
            valid, np.abs(pred["effect"][:, 1] - target[:, 1]), np.nan
        )
    return out


def shift(left, right, dat, key):
    mult = -1 if "mae" in key or key in ("magnitude_mm", "verb_nll") else 1
    return estimate(mult * (left[key] - right[key]), dat["group"], dat["families"])


def subgroup(file, prefix):
    return {
        key[len(prefix) :]: file[key] for key in file.files if key.startswith(prefix)
    }


def custody(tag, step, scale=False):
    a, b = roots("scale" if scale else "runs")
    loc = Path(tag) / f"step-{step:05d}"
    x = read(a / loc / "verified.json")
    if x != read(b / loc / "verified.json"):
        raise ValueError("Mismatched dual-volume receipt")
    d = read(a / loc / "receipt.json")
    if x["inference_sha256"] != d["inference_sha256"]:
        raise ValueError("Generation changed")
    return {
        k: d[k]
        for k in (
            "tag",
            "seed",
            "step",
            "parent_sha256",
            "inference_sha256",
            "optimizer_sha256",
            "active_parent_parameters",
            "trainable_parameters",
            "backend",
            "wall_this_process_s",
        )
    }


def analyze(banks, evaldir, out, scale=False, partial=False):
    out.mkdir(exist_ok=True, parents=True)
    audit = read(evaldir / "device-audit.json")
    grades = {(v["recording"], v["id"]): v["grade"] for v in audit["rows"]}
    rows = {}
    comparisons = {}
    artifacts = {}
    baselines = {}
    truths = {}
    all_metrics = {}
    bmetrics = {}
    gates = {}
    public = {}
    for fold in FOLDS:
        bmetrics[fold] = {}
        with np.load(
            evaldir / f"baselines-{fold}-joint.npz", allow_pickle=False
        ) as file:
            for role in ("held", "familiar", "dev"):
                dat = label_data(banks, fold, role, grades)
                truths[fold, role] = dat
                methods = sorted(
                    {k.split("|")[1] for k in file.files if k.startswith(role + "|")}
                )
                bmetrics[fold][role] = {}
                for name in methods:
                    m = metric(dat, subgroup(file, role + "|" + name + "|"))
                    bmetrics[fold][role][name] = m
                    baselines[fold + "/" + role + "/" + name] = {
                        k: estimate(v, dat["group"], dat["families"], ci=False)
                        for k, v in m.items()
                    }

        def tag_for(c, t, seed, family):
            return (
                f"g11s-{fold}-{family}-{c}-paired-s{seed}"
                if scale
                else f"g11-{fold}-{c}-{t}-s{seed}"
            )

        for seed in (911, 912) if scale else SEEDS:
            for core in CORES:
                for family in ("small", "large") if scale else ("main",):
                    for train in ("paired",) if scale else TREAT:
                        tag = tag_for(core, train, seed, family)
                        file = (
                            evaldir
                            / ("scale-predictions" if scale else "predictions")
                            / (tag + "-1024.npz")
                        )
                        meta = file.with_suffix(".json")
                        if not file.exists() or not meta.exists():
                            if partial:
                                continue
                            raise ValueError("Missing predictions " + tag)
                        src = custody(tag, 1024, scale)
                        d = read(meta)
                        if (
                            d["delta_sha256"] != src["inference_sha256"]
                            or d["parent_sha256"] != src["parent_sha256"]
                            or hashlib.sha256(file.read_bytes()).hexdigest()
                            != d["output_sha256"]
                        ):
                            raise ValueError("Test and fitted artifacts disagree")
                        for rung in (256, 640, 1024):
                            artifacts[tag + "/" + str(rung)] = custody(tag, rung, scale)
                        entries = {}
                        all_metrics[tag] = {}
                        with np.load(file) as predictions:
                            for role in ("held", "familiar", "dev"):
                                dat = truths[fold, role]
                                all_metrics[tag][role] = {}
                                modes = sorted(
                                    {
                                        k.split("|")[1]
                                        for k in predictions.files
                                        if k.startswith(role + "|")
                                    }
                                )
                                for mode in modes:
                                    raw = subgroup(predictions, role + "|" + mode + "|")
                                    m = metric(dat, raw)
                                    all_metrics[tag][role][mode] = m
                                    entries[role + "/" + mode] = {
                                        k: estimate(
                                            v, dat["group"], dat["families"], ci=False
                                        )
                                        for k, v in m.items()
                                    }
                                    if role == "held":
                                        for key in (
                                            "verb",
                                            "noun",
                                            "open_noun",
                                            "effect",
                                        ):
                                            public[tag + "|" + mode + "|" + key] = raw[
                                                key
                                            ]
                        rows[tag] = dict(
                            values=entries,
                            parameters=src["trainable_parameters"],
                            parent_parameters=src["active_parent_parameters"],
                            hashes=[src["parent_sha256"], src["inference_sha256"]],
                        )
                    full = tag_for(core, "paired", seed, family)
                    if full not in all_metrics:
                        continue
                    delta = {}
                    for role in ("held", "familiar", "dev"):
                        dat = truths[fold, role]
                        ref = all_metrics[full][role]
                        a = ref["correct"]
                        options = {
                            "wrong": ref["wrong"],
                            "none": ref["none"],
                            "raw_history": bmetrics[fold][role]["raw_history_correct"],
                            "raw_joint": bmetrics[fold][role]["raw_joint_correct"],
                            "transition": bmetrics[fold][role]["action_transition"],
                            "repeat": bmetrics[fold][role]["repeat_last_action"],
                            "persistence": bmetrics[fold][role]["persistence"],
                            "mean": bmetrics[fold][role]["training_mean"],
                            "median": bmetrics[fold][role]["training_median"],
                        }
                        if not scale:
                            for t, mode in (
                                ("deranged", "correct"),
                                ("absent", "none"),
                            ):
                                name = tag_for(core, t, seed, family)
                                if name in all_metrics:
                                    options[t + "_trained"] = all_metrics[name][role][
                                        mode
                                    ]
                        result = {
                            name: {
                                k: shift(a, m, dat, k)
                                for k in (*SCORES, "conflict_verb")
                                if k in m and k in a
                            }
                            for name, m in options.items()
                        }
                        for better, other in (
                            ("small", "wrong_small"),
                            ("human", "wrong_human"),
                            ("human", "correct"),
                            ("acoustic", "asr_text"),
                        ):
                            if better in ref and other in ref:
                                result[better + "_minus_" + other] = {
                                    k: shift(ref[better], ref[other], dat, k)
                                    for k in SCORES
                                }
                        delta[role] = result
                    dat = truths[fold, "held"]
                    delta["intervention_response"] = {
                        "changed_verb_prediction": estimate(
                            public[full + "|correct|verb"]
                            != public[full + "|wrong|verb"],
                            dat["group"],
                            dat["families"],
                        ),
                        "absolute_change_in_predicted_changed_fraction": estimate(
                            np.where(
                                dat["valid"],
                                np.abs(
                                    public[full + "|correct|effect"][:, :2].sum(1)
                                    - public[full + "|wrong|effect"][:, :2].sum(1)
                                ),
                                np.nan,
                            ),
                            dat["group"],
                            dat["families"],
                        ),
                    }
                    comparisons[full] = delta
                    gates[full] = {}
                    for k in ("verb", "effect_mae"):
                        diffs = [delta["held"][alt].get(k) for alt in ("wrong", "none")]
                        gates[full][k] = bool(
                            all(x and x["ci95"][0] > 0 for x in diffs)
                        )
    size_comparisons = {}
    if scale:
        for fold in FOLDS:
            dat = truths[fold, "held"]
            for core in CORES:
                for seed in (911, 912):
                    names = [
                        f"g11s-{fold}-{family}-{core}-paired-s{seed}"
                        for family in ("small", "large")
                    ]
                    if not all(tag in all_metrics for tag in names):
                        continue
                    small, large = [all_metrics[tag]["held"] for tag in names]
                    effects = {
                        k: shift(large["correct"], small["correct"], dat, k)
                        for k in SCORES
                    }
                    sensitivity = {}
                    for k in ("verb", "effect_mae", "device_mae"):
                        sign = 1 if k == "verb" else -1
                        gain = sign * (
                            (large["correct"][k] - large["wrong"][k])
                            - (small["correct"][k] - small["wrong"][k])
                        )
                        sensitivity[k] = estimate(gain, dat["group"], dat["families"])
                    size_comparisons[f"{fold}-{core}-s{seed}"] = dict(
                        large_minus_small=effects,
                        change_in_correct_minus_wrong_benefit=sensitivity,
                    )
    for fold in FOLDS:
        d = truths[fold, "held"]
        for k in (
            "verb",
            "noun",
            "open_noun",
            "effect",
            "valid",
            "strict",
            "device",
            "group",
            "conflict",
            "families",
        ):
            public[fold + "|truth|" + k] = d[k]
    tag = "scale" if scale else "primary"
    result = dict(
        study="G11",
        kind=tag,
        fits=len(rows),
        rows=rows,
        baselines=baselines,
        comparisons=comparisons,
        gates=gates,
        size_comparisons=size_comparisons,
        unit="equal-weight families; equal-weight recordings within each family; 4000 paired recording resamples",
        positive_means="higher accuracy or lower error for the first condition in paired contrasts",
    )
    write(out / (tag + ".json"), result)
    write(out / (tag + "-artifacts.json"), artifacts)
    np.savez_compressed(out / (tag + "-predictions.npz"), **public)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("banks", type=Path)
    p.add_argument("eval", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--scale", action="store_true")
    p.add_argument("--partial", action="store_true")
    a = p.parse_args()
    print("scored", analyze(a.banks, a.eval, a.out, a.scale, a.partial)["fits"])
