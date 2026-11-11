"""Paired whole-recording inference for both endpoints and the registered architecture comparisons."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from evals.g12_baselines import baseline
from scripts.train.g12_fit import ROOT, read, SEEDS
from scripts.train.g9_storage import write
from models.g12_world import CORES

ROLES = ("confirm", "replica")
SOFT = ("ordered", "exact", "nll", "brier", "state_accuracy")
POSITIVE = {"ordered", "exact", "state_accuracy"}


class RecordingBootstrap:
    def __init__(self, rows):
        self.groups = np.asarray([r["recording"] for r in rows])
        self.family = np.asarray([r["family"] for r in rows])
        self.cache = {}

    def estimates(self, values):
        values = np.asarray(values, np.float64)
        mask = np.isfinite(values)
        key = mask.tobytes()
        if not mask.any():
            return None
        if key not in self.cache:
            names, gi, counts = np.unique(
                self.groups[mask], return_inverse=True, return_counts=True
            )
            lookup = {n: f for n, f in zip(self.groups, self.family)}
            fam = np.asarray([lookup[n] for n in names])
            fams = np.unique(fam)
            weights = np.zeros((4000, len(names)))
            mean = np.zeros(len(names))
            for f in fams:
                indices = np.flatnonzero(fam == f)
                size = len(indices)
                # The same family/recording sample is reused for every model on a given endpoint.
                seed = int.from_bytes(
                    hashlib.sha256(
                        ("G12/resample/" + f + "/" + "/".join(names[indices])).encode()
                    ).digest()[:8],
                    "big",
                )
                rng = np.random.default_rng(seed)
                weights[:, indices] = rng.multinomial(
                    size, np.full(size, 1.0 / size), size=4000
                ) / (size * len(fams))
                mean[indices] = 1.0 / (size * len(fams))
            self.cache[key] = (gi, counts, weights, mean, len(fams))
        gi, counts, w, mean, nf = self.cache[key]
        avg = np.bincount(gi, weights=values[mask]) / counts
        lo, hi = np.quantile(w @ avg, [0.025, 0.975])
        return dict(
            value=float(mean @ avg),
            lo=float(lo),
            hi=float(hi),
            n=int(mask.sum()),
            recordings=len(counts),
            families=nf,
        )

    def gain(self, a, b, metric):
        return self.estimates((a - b) if metric in POSITIVE else (b - a))


def scoring(prob, physical, data):
    y = data["y"]
    target = data["state"]
    valid = target >= 0
    p = np.take_along_axis(prob, np.maximum(y, 0)[:, :, None], 2)[:, :, 0]
    p = np.where(y >= 0, p, 1e-7)
    hits = prob.argmax(-1) == y
    out = dict(
        ordered=hits.mean(1),
        exact=hits.all(1).astype(float),
        nll=-np.log(np.maximum(p, 1e-7)).mean(1),
        pos1=hits[:, 0].astype(float),
        pos2=hits[:, 1].astype(float),
        pos3=hits[:, 2].astype(float),
    )
    if physical is not None:
        out.update(
            brier=np.where(valid, (target - physical) ** 2, np.nan),
            state_accuracy=np.where(valid, (physical >= 0.5) == target, np.nan),
        )
    return out


def control_summary(b, meta, data, resample):
    result = {}
    metrics = {}
    for k, p in b.items():
        if k == "repeat_correct":
            metrics["repeat"] = dict(ordered=p.mean(1), exact=p.all(1).astype(float))
        elif k in ("no_change", "state_prior"):
            valid = data["state"] >= 0
            metrics[k] = dict(
                brier=np.where(valid, (data["state"] - p) ** 2, np.nan),
                state_accuracy=np.where(valid, (p >= 0.5) == data["state"], np.nan),
            )
        else:
            metrics[k] = scoring(*p, data)
    for name, ms in metrics.items():
        result[name] = {k: resample.estimates(v) for k, v in ms.items()}
    return result, metrics


def public_case_table(root, fold, role, meta, labels, catalog):
    opened = {}
    result = []
    for r in meta["rows"]:
        n = r["recording"]
        rec = catalog[n]
        spoken = {s["id"]: s for s in rec["speech"]}
        t = (r["cut_tick"] + 1) / 10
        ins = spoken[r["cue"]]
        alt = spoken[r["wrong"]]
        if not (
            alt["end"] <= ins["start"] - 0.1 + 1e-6
            and ins["end"] <= t + 1e-6
            and r["target_onsets"][0] >= t + 0.2 - 1e-6
        ):
            raise ValueError("Causal clocks not separated")
        intervened = any(
            t + 1e-6 < c["end"] <= r["target_onsets"][2] for c in rec["speech"]
        )
        label = labels.get((n, r["id"]), -1)
        result.append(
            dict(
                **r,
                fold=fold,
                role=role,
                instruction=ins["text"],
                foil=alt["text"],
                speech_span=[ins["start"], ins["end"]],
                foil_span=[alt["start"], alt["end"]],
                intervened=intervened,
                state=label,
                actual_lex=bool(set(r["verbs"]) & set(ins["explicit"])),
                wrong_lex=bool(set(r["verbs"]) & set(alt["explicit"])),
            )
        )
    return result


def analyze(out):
    out.mkdir(exist_ok=True, parents=True)
    root = ROOT / "data"
    state = json.loads((root / "state-labels.json").read_bytes())
    labels = {(r["recording"], r["id"]): r["label"] for r in state["items"]}
    catalog = json.loads(
        Path("/mnt/slwm-data/SLWM/g11/data/annotations.json").read_bytes()
    )
    catalog.update(json.loads((root / "annotations.json").read_bytes()))
    groups = json.loads((root / "groups.json").read_bytes())
    research = dict(
        sets={}, runs={}, contrasts={}, baselines={}, gates={}, direct={}, groups=groups
    )
    cases = []
    public = {}
    for fold in ("A", "B"):
        all_data = {
            r: read(root, fold, r, labels)
            for r in ("train", "dev", "confirm", "replica")
        }
        train = all_data["train"][1]
        b, recipe = baseline(
            all_data["train"],
            all_data["dev"],
            {r: all_data[r] for r in ROLES},
            all_data["train"][0]["vocabulary"],
        )
        recipe["groundable_cued"] = int((train["cued"] & train["groundable"]).sum())
        recipe["train_cued"] = int(train["cued"].sum())
        research["baselines"][fold] = dict(recipe=recipe)
        roles = {}
        for role in ROLES:
            meta, data = all_data[role]
            bs = RecordingBootstrap(meta["rows"])
            by, metric = control_summary(b[role], meta, data, bs)
            research["baselines"][fold][role] = by
            tab = public_case_table(root, fold, role, meta, labels, catalog)
            cases.extend(tab)
            inv = np.array([r["intervened"] for r in tab])
            lex = np.array(
                [float(r["actual_lex"]) - float(r["wrong_lex"]) for r in tab]
            )
            targets = data["state"]
            valid = targets >= 0
            rset = research["sets"][fold + "-" + role] = dict(
                decisions=len(tab),
                recordings=len(set(r["recording"] for r in tab)),
                state_cases=int(valid.sum()),
                changed=int((targets == 1).sum()),
                stable=int((targets == 0).sum()),
                eligible_per_family={
                    f: dict(Counter(str(r["state"]) for r in tab if r["family"] == f))
                    for f in sorted({r["family"] for r in tab})
                },
                intervened=int(inv.sum()),
                lexical_delta=bs.estimates(lex),
                known_verb_fraction=float((data["y"] >= 0).mean()),
            )
            roles[role] = (bs, metric, inv)
            prefix = fold + "_" + role
            public[prefix + "_y"] = data["y"]
            public[prefix + "_state"] = targets
            public[prefix + "_group"] = data["group"]
            for name, value in b[role].items():
                if isinstance(value, tuple):
                    prob, st = value
                    public[prefix + "_" + name + "_top"] = prob.argmax(-1).astype("i2")
                    if st is not None:
                        public[prefix + "_" + name + "_state"] = st.astype("f4")
                else:
                    public[prefix + "_" + name] = np.asarray(value)
        tags = sorted(p for p in (ROOT / "eval").glob(fold + "-*") if p.is_dir())
        if len(tags) != 24:
            raise ValueError(f"Incomplete {fold} comparison: {len(tags)}")
        for folder in tags:
            tag = folder.name
            _, core, treat, seed = tag.split("-")
            receipt = json.loads((folder / "receipt.json").read_bytes())
            run = dict(
                selected=receipt["selected"],
                thresholds=receipt["thresholds"],
                parameters=receipt["config"]["parameters"],
                vocab=len(receipt["config"]["vocabulary"]),
                curve=receipt["curve"],
                sha256=receipt["weights_sha256"],
                by_role={},
            )
            research["runs"][tag] = run
            for role in ROLES:
                bs, controls, inv = roles[role]
                meta, data = all_data[role]
                with np.load(folder / (role + ".npz")) as file:
                    available = [x[:-5] for x in file.files if x.endswith("_verb")]
                    modes = {
                        k: scoring(file[k + "_verb"], file[k + "_state"], data)
                        for k in available
                    }
                    actual = modes["correct"]
                    summaries = {
                        k: {m: bs.estimates(v) for m, v in scores.items() if m in SOFT}
                        for k, scores in modes.items()
                    }
                    result = dict(
                        input=summaries,
                        position=[
                            bs.estimates(actual["pos" + str(k)]) for k in (1, 2, 3)
                        ],
                        speech={
                            k: {m: bs.gain(actual[m], modes[k][m], m) for m in SOFT}
                            for k in ("wrong", "none")
                        },
                    )
                    result["against_controls"] = {
                        k: {m: bs.gain(actual[m], row[m], m) for m in SOFT if m in row}
                        for k, row in controls.items()
                        if k
                        in (
                            "repeat",
                            "frequency",
                            "transition",
                            "no_change",
                            "state_prior",
                            "raw_correct",
                            "lexical_correct",
                        )
                    }
                    result["conflict_free_of_later_instruction"] = {
                        m: bs.gain(
                            np.where((~inv) & data["conflict"], actual[m], np.nan),
                            np.where(
                                (~inv) & data["conflict"], modes["wrong"][m], np.nan
                            ),
                            m,
                        )
                        for m in ("ordered", "brier")
                    }
                    result["per_family"] = {}
                    for fam in sorted(set(r["family"] for r in meta["rows"])):
                        mask = np.array([r["family"] == fam for r in meta["rows"]])
                        result["per_family"][fam] = {
                            m: bs.gain(
                                np.where(mask, actual[m], np.nan),
                                np.where(mask, modes["wrong"][m], np.nan),
                                m,
                            )
                            for m in ("ordered", "brier")
                        }
                    thresholds = np.asarray(receipt["thresholds"])
                    commit = file["correct_confidence"] >= thresholds
                    result["policy"] = dict(
                        action_coverage=bs.estimates(commit[:, 0]),
                        state_coverage=bs.estimates(commit[:, 1]),
                        internal_only=bs.estimates(~commit.any(1)),
                        action_conditional=bs.estimates(
                            np.where(commit[:, 0], actual["ordered"], np.nan)
                        ),
                        state_conditional=bs.estimates(
                            np.where(commit[:, 1], actual["state_accuracy"], np.nan)
                        ),
                    )
                    for mode in ("correct", "wrong", "none"):
                        prob = file[mode + "_verb"]
                        prefix = tag + "_" + role + "_" + mode
                        public[prefix + "_top"] = prob.argmax(-1).astype("i2")
                        public[prefix + "_py"] = np.where(
                            data["y"] >= 0,
                            np.take_along_axis(
                                prob, np.maximum(data["y"], 0)[:, :, None], 2
                            )[:, :, 0],
                            1e-7,
                        ).astype("f4")
                        public[prefix + "_state"] = file[mode + "_state"]
                        public[prefix + "_confidence"] = file[mode + "_confidence"]
                run["by_role"][role] = result
            _, tr = all_data["train"]
            ix = receipt["train_indices"]
            with np.load(folder / "train.npz") as z:
                sub = {k: tr[k][ix] for k in ("y", "state")}
                run["training_slice"] = {
                    m: float(scoring(z[m + "_verb"], None, sub)["ordered"].mean())
                    for m in ("correct", "wrong", "none")
                }
        for role in ROLES:
            bs, _, _ = roles[role]
            meta, data = all_data[role]
            for path in tags:
                tag = path.name
                _, core, treat, seed = tag.split("-")
                rivals = [
                    ("dense", f"{fold}-gpt2-{treat}-{seed}"),
                    ("gated", f"{fold}-gated-{treat}-{seed}"),
                ]
                if treat != "paired":
                    rivals.append(("paired_training", f"{fold}-{core}-paired-{seed}"))
                for label, other in rivals:
                    if other == tag:
                        continue
                    with (
                        np.load(path / (role + ".npz")) as v,
                        np.load(ROOT / "eval" / other / (role + ".npz")) as z,
                    ):
                        a = scoring(v["correct_verb"], v["correct_state"], data)
                        b0 = scoring(z["correct_verb"], z["correct_state"], data)
                        research["contrasts"].setdefault(tag, {}).setdefault(role, {})[
                            label
                        ] = {m: bs.gain(a[m], b0[m], m) for m in SOFT}
    for role in ROLES:
        for core in CORES:
            for m in ("ordered", "brier"):
                gate = {}
                for fold in ("A", "B"):
                    vals = [
                        research["runs"][f"{fold}-{core}-paired-{s}"]["by_role"][role]
                        for s in SEEDS
                    ]
                    gate[fold] = sum(
                        all(v["speech"][k][m]["lo"] > 0 for k in ("wrong", "none"))
                        for v in vals
                    )
                gate["replicated"] = all(gate[f] >= 2 for f in ("A", "B"))
                research["gates"][f"{role}-{core}-{m}-speech"] = gate
                if core != "gpt2":
                    per = {
                        f: sum(
                            research["contrasts"][f"{f}-{core}-paired-{s}"][role][
                                "dense"
                            ][m]["lo"]
                            > 0
                            for s in SEEDS
                        )
                        for f in ("A", "B")
                    }
                    per["replicated"] = all(per[f] >= 2 for f in ("A", "B"))
                    research["gates"][f"{role}-{core}-{m}-versus_dense"] = per
    with gzip.open(out / "decisions.jsonl.gz", "wt") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n")
    np.savez_compressed(out / "paired-predictions.npz", **public)
    write(out / "comparisons.json", research)
    write(out / "visual-labels.json", state)
    print(
        "Fully matched",
        len(research["runs"]),
        "fits and",
        len(cases),
        "held decisions.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    analyze(p.parse_args().out)
