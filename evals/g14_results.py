"""Recompute realized cue swaps, baseline comparisons and paired intervals from frozen simulator branches."""

from collections import Counter
import gzip
import json
from pathlib import Path
import numpy as np
from evals.g13_stats import Groups
from evals.g14_trials import ROOT, METRICS, read, summary
from scripts.train.g9_storage import sha, write
from scripts.train.g14_execute import tag

CONDITIONS = ("actual", "wrong_color", "wrong_address", "both_wrong")


def results():
    lock = json.loads((ROOT / "work/final-control-lock.json").read_bytes())
    ps = lock["policies"]
    stage = lock["stage"]
    route = lock["routed"]
    guard = lock["contact_only"]
    dest = ROOT / "release/metrics"
    dest.mkdir(parents=True, exist_ok=True)
    combined = {}
    for cohort, n in (("held", 256), ("fresh", 64)):
        folder = ROOT / "work/executed/final" / cohort
        if json.loads((folder / "complete.json").read_bytes())["cases"] != n:
            raise ValueError("Incomplete physical execution")
        rows = read(folder)
        if len(rows) != n:
            raise ValueError("Missing held scene")
        keys = [tag(p) for p in ps]
        expected = set(keys)
        total = 0
        for row in rows:
            if not row["isolated_branch_process"]:
                raise ValueError("Held branch was not isolated")
            if set(row["results"]) != expected:
                raise ValueError("Missing held policy")
            if (
                len(
                    {
                        v["image"]
                        for branches in row["results"].values()
                        for v in branches.values()
                    }
                )
                != 1
            ):
                raise ValueError("Unmatched camera image")
            for p, k in zip(ps, keys):
                branches = row["results"][k]
                if set(branches) != (
                    {"0"} if p["core"] == "noop" else {"0", "1", "2", "3"}
                ):
                    raise ValueError("Missing counterfactual branch")
                total += len(branches)
        trials, statistics = summary(rows, stratify=True)
        groups = np.array([r["seed"] for r in rows])
        people = np.array([r["speaker"] for r in rows])
        shift = np.array([r["shift"] for r in rows])
        families = [f"{p}/{int(s)}" for p, s in zip(people, shift)]
        bs = Groups(groups, families)

        def field(core, seed, **kw):
            return tag(
                dict(
                    stage=stage,
                    core=core,
                    arm="protected",
                    seed=seed,
                    routed=route,
                    contact_only=guard,
                    **kw,
                )
            )

        def measure(k, m, c="actual"):
            return np.asarray([x["conditions"][c][m] for x in trials[k]])

        def pooled(keys, m, c="actual"):
            return np.mean([measure(k, m, c) for k in keys], axis=0)

        def effect(a, b, m):
            return bs.score((a - b) * (-1 if m in ("final_mm", "other_mm") else 1))

        models = {
            c: [field(c, s) for s in (1411, 1412, 1413)] for c in ("gpt2", "transport")
        }
        geom = tag(dict(core="geometry", routed=route))
        raw = tag(dict(core="raw", stage=stage, routed=route, contact_only=guard))
        fast = tag(dict(core="contact_speed"))
        display = {
            **models,
            "geometry": [geom],
            "raw": [raw],
            "contact_speed": [fast],
            "straight": [tag({"core": "straight"})],
            "none": [tag({"core": "noop"})],
            "geometry_original_cue": [
                tag(dict(core="geometry", routed=route, original_edge=True))
            ],
            "geometry_open_gate": [
                tag(dict(core="geometry", routed=route, open_gate=True))
            ],
            "geometry_oracle_cue": [
                tag(dict(core="geometry", routed=route, oracle_cue=True))
            ],
            "geometry_oracle_pose_cue": [
                tag(
                    dict(
                        core="geometry", routed=route, oracle_cue=True, oracle_pose=True
                    )
                )
            ],
            "gated": [
                tag(
                    dict(
                        core="gated",
                        arm="protected",
                        stage="initial",
                        seed=1411,
                        routed=route,
                    )
                )
            ],
            "dense_spatial": [
                tag(
                    dict(
                        core="dense_spatial",
                        arm="protected",
                        stage="initial",
                        seed=1411,
                        routed=route,
                    )
                )
            ],
        }
        readout = {
            k: {m: bs.score(pooled(vals, m)) for m in METRICS}
            for k, vals in display.items()
        }
        pair = {}
        per_seed = {}
        for c, vals in models.items():
            for control, ks in (
                ("geometry", [geom]),
                ("raw", [raw]),
                ("contact_speed", [fast]),
                ("no_cue", display["none"]),
            ):
                pair[c + "_over_" + control] = {
                    m: effect(pooled(vals, m), pooled(ks, m), m) for m in METRICS
                }
            pair[c + "_full_over_mean_field"] = {
                m: effect(
                    measure(field(c, 1411), m),
                    measure(field(c, 1411, mean_field=True), m),
                    m,
                )
                for m in METRICS
            }
            for wrong in CONDITIONS[1:]:
                pair[c + "_over_" + wrong] = {
                    m: effect(pooled(vals, m), pooled(vals, m, wrong), m)
                    for m in METRICS
                }
        pair["transport_over_gpt2"] = {
            m: effect(pooled(models["transport"], m), pooled(models["gpt2"], m), m)
            for m in METRICS
        }
        pair["retained_over_reset"] = {
            m: effect(
                measure(field("transport", 1411), m),
                measure(field("transport", 1411, reset_memory=True), m),
                m,
            )
            for m in METRICS
        }
        for name, other in (
            ("new_over_original_speech", "geometry_original_cue"),
            ("confidence_gate_over_open", "geometry_open_gate"),
        ):
            pair[name] = {
                m: effect(measure(geom, m), pooled(display[other], m), m)
                for m in METRICS
            }
        for seed in (1411, 1412, 1413):
            per_seed[seed] = {
                m: effect(
                    measure(field("transport", seed), m),
                    measure(field("gpt2", seed), m),
                    m,
                )
                for m in METRICS
            }
        values = np.asarray(
            [
                [
                    [[v["conditions"][c][m] for m in METRICS] for c in CONDITIONS]
                    for v in trials[k]
                ]
                for k in keys
            ],
            dtype="f4",
        )
        preds = np.array(
            [
                [
                    sum(z["model_calls"] for z in r["results"][k].values())
                    / (24 * len(r["results"][k]))
                    for r in rows
                ]
                for k in keys
            ]
        )
        np.savez_compressed(
            dest / (cohort + "-scene-results.npz"),
            values=values,
            models=keys,
            conditions=CONDITIONS,
            metrics=METRICS,
            seed=groups,
            speaker=people,
            shift=shift,
            model_call_fraction=preds,
            start=[r["start"] for r in rows],
            goals=np.asarray(
                [
                    [[d["true_color"], d["true_station"]] for d in r["commands"]]
                    for r in rows
                ]
            ),
        )
        with (
            (dest / (cohort + "-branch-records.jsonl.gz")).open("wb") as rawfile,
            gzip.GzipFile(fileobj=rawfile, mode="wb", mtime=0) as z,
        ):
            for r in rows:
                z.write(
                    (
                        json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                )
        shifted = {
            str(bool(shift0)): {
                label: float(np.mean(pooled(ks, "success")[shift == shift0]))
                for label, ks in display.items()
            }
            for shift0 in (False, True)
        }
        speakers = {
            str(i): {
                label: float(np.mean(pooled(ks, "success")[people == i]))
                for label, ks in display.items()
            }
            for i in np.unique(people)
        }
        write(dest / (cohort + "-policies.json"), statistics)
        combined[cohort] = dict(
            starts=n,
            recorded_voices=sorted(map(int, np.unique(people))),
            branches=total,
            commands=n * 4,
            macros=total * 24,
            native_ticks=total * 72,
            by_policy_family=readout,
            paired=pair,
            spatial_over_dense_by_seed=per_seed,
            selective_by_shift=shifted,
            selective_by_speaker=speakers,
            neural_call_fraction={
                k: bs.score(preds[i])["value"] for i, k in enumerate(keys)
            },
            array_sha256=sha(dest / (cohort + "-scene-results.npz")),
        )
    passed = {
        k: sum(v["success"]["lo"] > 0 for v in d["spatial_over_dense_by_seed"].values())
        for k, d in combined.items()
    }
    doc = dict(
        cohorts=combined,
        preregistered_positive_seeds=passed,
        shared_processor_superiority=all(n >= 2 for n in passed.values()),
        routed=route,
        contact_only=guard,
        stage=stage,
        unit="Paired independent simulator state with all four human utterances. The 4,000 bootstrap draws resample starts within speaker and contact stratum.",
    )
    write(dest / "physical-results.json", doc)
    print(
        "Physical results",
        passed,
        {
            co: {
                k: round(v["success"]["value"], 4)
                for k, v in d["by_policy_family"].items()
                if k in ("geometry", "transport", "gpt2", "raw", "contact_speed")
            }
            for co, d in combined.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    results()
