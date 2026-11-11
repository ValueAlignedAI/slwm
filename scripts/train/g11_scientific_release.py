"""Assemble paired statistical controls and verify every newly trained generation and source lineage."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from g11_analysis import analyze, read, CORES
from g11_store import roots
from g9_storage import write


def summarize(root, out):
    banks = root / "data/banks"
    ev = root / "eval"
    out.mkdir(exist_ok=True, parents=True)
    (out / "device-audit.json").write_bytes((ev / "device-audit.json").read_bytes())
    p = analyze(banks, ev, out)
    s = analyze(banks, ev, out, scale=True)
    groups = read(root / "data/groups.json")
    catalog = read(root / "data/annotations.json")
    if (p["fits"], s["fits"], len(p["gates"]), len(s["gates"])) != (54, 24, 18, 24):
        raise ValueError("Missing matched stage")
    pa = read(out / "primary-artifacts.json")
    sa = read(out / "scale-artifacts.json")
    parents = {}
    reuse = {}
    parents_a, parents_b = roots("scale-parents")
    for size in ("confirm", "exposure"):
        for c in CORES:
            for seed in (911, 912):
                tag = f"{size}-{c}-z128-gelu-s{seed}-fresh"
                for step in (1024, 2048, 3072, 4096) if c == "gated" else (4096,):
                    rel = Path(tag) / f"step-{step:05d}"
                    if c == "gated":
                        a, b = parents_a, parents_b
                    else:
                        a, b = (
                            r.parent.parent / "g9/runs" for r in (parents_a, parents_b)
                        )
                    r = read(a / rel / "verified.json")
                    if r != read(b / rel / "verified.json"):
                        raise ValueError("Pretraining copies differ")
                    source = read(a / rel / "receipt.json")
                    record = {
                        k: source[k]
                        for k in (
                            "tag",
                            "seed",
                            "step",
                            "inference_sha256",
                            "optimizer_sha256",
                            "trainable_parameters",
                            "backend",
                            "wall_this_process_s",
                        )
                    }
                    if (r["inference_sha256"], r["optimizer_sha256"]) != (
                        record["inference_sha256"],
                        record["optimizer_sha256"],
                    ):
                        raise ValueError("Unverified scale parent")
                    (parents if c == "gated" else reuse)[tag + "/" + str(step)] = record
    for k, row in sa.items():
        if not k.endswith("/1024"):
            continue
        _, fold, size, c, _, sd = row["tag"].split("-")
        kind = "exposure" if size == "small" else "confirm"
        parent = f"{kind}-{c}-z128-gelu-{sd}-fresh/4096"
        if row["parent_sha256"] != {**parents, **reuse}[parent]["inference_sha256"]:
            raise ValueError("Scale parent not found in lineage ledger")
    write(out / "scale-parents.json", dict(new_gated=parents, reused_g9=reuse))
    supplementary = {}
    for part, folder, count in (
        ("cue_content", "cue-probes", 18),
        ("optimization_gap", "training-gap", 18),
        ("development_selection", "development-selected", 54),
    ):
        d = {file.stem: read(file) for file in sorted((ev / folder).glob("*.json"))}
        if len(d) != count:
            raise ValueError("Incomplete supplemental " + part)
        supplementary[part] = d
    write(out / "diagnostics.json", supplementary)
    coverage = {}
    index = []
    bank_hashes = {}
    for fold, roles in groups["folds"].items():
        ab = read(banks / (fold + "-vocabulary.json"))
        record = {}
        for role, names in roles.items():
            base = read(banks / f"{fold}-{role}.json")
            extra = read(banks / f"{fold}-{role}-effects-receipt.json")
            if base["sha256"] != extra["base_sha256"]:
                raise ValueError("Bank lineage changed")
            bank_hashes[fold + "-" + role] = dict(
                base=base["sha256"], physical=extra["sha256"]
            )
            with np.load(banks / f"{fold}-{role}.npz") as db:
                record[role] = dict(
                    recordings=len(names),
                    decisions=int(db["cued"].sum()),
                    background=int((~db["cued"]).sum()),
                    known_verbs=int(((db["verb"] >= 0) & db["cued"]).sum()),
                    known_nouns=int(((db["noun"] >= 0) & db["cued"]).sum()),
                    public_ontology_nouns=int(
                        ((db["open_noun"] >= 0) & db["cued"]).sum()
                    ),
                    measured=extra["valid"],
                    isolated_hand_adjacent=extra["strict"],
                )
            if role == "held":
                for v in base["rows"]:
                    src = catalog[v["recording"]]
                    event = next(e for e in src["events"] if e["id"] == v["action_id"])
                    look = {i["id"]: i for i in src["speech"]}
                    correct = look[event["cue"]]
                    other = look[event["wrong"]]
                    cut = (v["cut_tick"] + 1) / 10
                    if (
                        event["start"] + 1e-8 < cut + 0.2
                        or other["end"] > correct["start"] - 0.1 + 1e-6
                        or correct["end"] > cut + 1e-7
                    ):
                        raise ValueError("Noncausal public decision")
                    index.append(
                        dict(
                            fold=fold,
                            **v,
                            cut_end_s=cut,
                            action_start_s=event["start"],
                            action_end_s=event["end"],
                            cue_interval=[correct["start"], correct["end"]],
                            other_interval=[other["start"], other["end"]],
                            true_speech_sha256=hashlib.sha256(
                                correct["text"].encode()
                            ).hexdigest(),
                            other_speech_sha256=hashlib.sha256(
                                other["text"].encode()
                            ).hexdigest(),
                        )
                    )
        coverage[fold] = dict(
            roles=record,
            training_verbs=len(ab["verb"]),
            training_nouns=len(ab["noun"]),
            open_nouns=len(ab["open_noun"]),
        )
    with gzip.open(out / "held-decisions.jsonl.gz", "wt") as fd:
        for row in index:
            fd.write(json.dumps(row, sort_keys=True) + "\n")
    write(
        out / "data.json",
        dict(
            groups=groups,
            coverage=coverage,
            banks=bank_hashes,
            total_catalog_recordings=len(catalog),
            primary_new_decisions=len(index),
            original_labels_sha256="cc7898b49958a62fe021ae2ffa53a709c3fd6f45fd3f893960b8aac6d13dfe9c",
            original_split_sha256="e10674e7ac32957386d5e88afa60acf3335251b92ba2f8cfb97a736dc85e1621",
        ),
    )
    root_a, root_b = roots("runs")
    inputs = read(root_a.parent / "inputs/verified.json")
    if inputs != read(root_b.parent / "inputs/verified.json") or len(inputs) < 35:
        raise ValueError("Prepared inputs not mirrored")
    if read(root_a.parent / "inference/verified.json") != read(
        root_b.parent / "inference/verified.json"
    ):
        raise ValueError("Portable copies differ")
    pack = read(root / "release/inference/portable.json")
    if (
        len(pack["adapters"]) != 6
        or len(pack["backbones"]) != 3
        or not (root / "release/cli-verified.json").exists()
    ):
        raise ValueError("Published inference verification missing")
    count = len(pa) + len(sa) + len(parents)
    if count != 250:
        raise ValueError(f"Unverified generation count {count}")
    write(
        out / "scientific-verdict.json",
        dict(
            total_fits=82,
            primary_fits=54,
            scale_adapter_fits=24,
            new_recurrent_parents=4,
            verified_generations=count,
            primary_held_decisions=len(index),
            fresh_recordings=sum(len(a) for a in groups["fresh"].values()),
            physical_events=sum(
                v["roles"]["held"]["measured"] for v in coverage.values()
            ),
            prepared_input_files=len(inputs),
            package_backbones=3,
            adapter_packages=6,
            passed_waveform_inference=True,
        ),
    )
    print(
        "Verified 82 fits, 250 generations and",
        len(index),
        "held decisions",
        flush=True,
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("root", type=Path)
    a.add_argument("out", type=Path)
    p = a.parse_args()
    summarize(p.root, p.out)
