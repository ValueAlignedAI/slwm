"""Publish acquisition lineage, complete checked counts, and compact cross-study summaries."""

import gzip
import json
import numpy as np
from g13_custody import ROOT, bases, COUNTS
from g9_storage import sha, write


def read(p):
    return json.loads(p.read_bytes())


def mean_ci(diff, group):
    from evals.g13_stats import Groups

    bs = Groups(group)
    stats = bs.score(diff)
    _, inv, n = np.unique(group, return_inverse=True, return_counts=True)
    v = np.bincount(inv, weights=diff) / n
    rng = np.random.default_rng(87313)
    est = rng.multinomial(len(v), np.full(len(v), 1 / len(v)), size=10000) @ v / len(v)
    stats["lo99"], stats["hi99"] = map(float, np.quantile(est, [0.005, 0.995]))
    return stats


def main():
    metric = ROOT / "release/metrics"
    sources = {}
    lineage = []
    accum = set()
    for name in ("robot", "robot-extra", "robot-prospective", "robot-readout"):
        r = ROOT / "data" / name
        data = read(r / "banks/data.json")
        files = read(r / "raw/selection.json")
        note = read(r / "raw/sources.json")
        assert len(files) == len(note)
        lookup = {n["url"]: n for n in note}
        receipts = []
        for f in files:
            url = (
                "https://storage.googleapis.com/gresearch/"
                + f["name"]
                + "?generation="
                + f["generation"]
            )
            xx = lookup[url]
            p = r / "raw" / f["name"].split("/")[-1]
            if xx["bytes"] != p.stat().st_size:
                raise ValueError("Changed archive")
            receipts.append(
                dict(
                    name=f["name"],
                    generation=f["generation"],
                    bytes=xx["bytes"],
                    md5=f["md5Hash"],
                    sha256=xx["sha256"],
                    url=url,
                )
            )
        episodes = read(r / "banks/episodes.json")
        g = {x["group"] for x in episodes}
        if accum & g:
            raise ValueError("An external parent survived the source exclusion")
        accum.update(g)
        for i, x in enumerate(episodes):
            lineage.append(
                dict(
                    acquisition=name,
                    index=i,
                    source_id=x["episode"],
                    parent_id=x["parent"],
                    group=x["group"],
                    role=x["role"],
                    shard=x["shard"],
                    byte_offset=x["offset"],
                    frames=x["length"],
                    held_color_pair=x["novel"],
                )
            )
        sources[name] = dict(
            shards=receipts,
            retained=data["counts"],
            excluded=data["excluded"],
            overlap_exclusions=[] if name == "robot" else data["excluded_groups"],
            fingerprint_merged_parents=data["merged_parent_ids"],
            unique_source_frame_digests=data["unique_frames"],
            corrected=read(r / "corrected/data.json"),
        )
    with gzip.open(metric / "robot-source-index.jsonl.gz", "wt") as fh:
        for x in lineage:
            fh.write(json.dumps(x, separators=(",", ":")) + "\n")
    holo = {}
    for name in ("holo", "components"):
        r = ROOT / "data" / name
        groups = read(r / "groups.json")
        holo[name] = {}
        for fold, names in groups.items():
            holo[name][fold] = {
                n: dict(
                    video_sha256=sha(r / "fresh/media" / (n + "-av.mp4")),
                    causal_feature_sha256=read(r / "fresh/shards" / (n + ".json"))[
                        "sha256"
                    ],
                )
                for n in names
            }
    component = read(ROOT / "data/components/component-banks/inventory.json")
    refs = read(ROOT / "data/components/minilm/source.json")
    write(
        metric / "data.json",
        dict(
            robot=sources,
            natural=holo,
            speech_commands=read(ROOT / "data/speech/directions.json"),
            semantic_edge=dict(
                parameters=component["edge_parameters"],
                revision=refs["revision"],
                file_hashes={k: v["sha256"] for k, v in refs["files"].items()},
            ),
        ),
    )
    with np.load(metric / "direct-actions.npz") as d:
        target = d["truth"]
        group = d["group"]
        raw = np.linalg.norm(d["raw"] - target, axis=1) * 1000
        readouts = {}
        for arm in ("native", "field", "protected"):
            scores = {
                c: np.mean(
                    [
                        1000
                        * np.linalg.norm(
                            d[
                                f"{c}-linear-{s}"
                                + ("_" if arm == "native" else "-")
                                + arm
                            ]
                            - target,
                            axis=1,
                        )
                        for s in (1301, 1302, 1303, 1311, 1312)
                    ],
                    axis=0,
                )
                for c in ("gpt2", "transport", "associative", "dense_spatial")
            }
            readouts[arm] = dict(
                raw_gain=mean_ci(raw - scores["transport"], group),
                over_dense=mean_ci(scores["gpt2"] - scores["transport"], group),
                error={k: mean_ci(v, group) for k, v in scores.items()},
            )
    component_results = {}
    from evals.g13_stats import Groups

    with np.load(metric / "component-predictions.npz") as z:
        for f in "AB":
            y = z[f + "_y"]
            bs = Groups(z[f + "_groups"], z[f + "_families"])
            component_results[f] = {}
            for c in ("gpt2", "gated", "oscillator", "associative"):
                effects = []
                for seed in (1301, 1302, 1303):
                    tags = [
                        f"{f}-{c}-{a}-{seed}_correct_top"
                        for a in ("learned", "semantic")
                    ]
                    direct, anchored = [
                        (z[k][:, 0, None] == y).any(1).astype(float) for k in tags
                    ]
                    effects.append(anchored - direct)
                component_results[f][c] = dict(
                    mean_over_learned=bs.score(np.mean(effects, axis=0)),
                    seed_intervals=[bs.score(v) for v in effects],
                )
    write(
        metric / "process-comparisons.json",
        dict(shared_readout=readouts, natural_components=component_results),
    )
    p = read(metric / "prospective.json")
    d = read(metric / "direct-action.json")
    qc = read(metric / "speech-parity.json")
    native = read(metric / "natural-waveforms.json")
    ck = read(metric / "checkpoints.json")
    count = {
        k: len(list((bases()[0] / k).glob("*/step-*/verified.json"))) for k in COUNTS
    }
    all_fits = sum(n for n, _ in COUNTS.values())
    pkg = read(ROOT / "release/inference/manifest.json")
    cost = read(metric / "compute.json")
    if not (
        isinstance(p["passed"], bool)
        and np.isfinite(d["five_seed_over_raw"]["lo99"])
        and len(ck) == 594
        and all_fits == 182
        and qc["portable"]["same_top"] == 1639
        and len(native) == 4
        and cost["all_owned_released"]
    ):
        raise ValueError("Incomplete or inconsistent G13 results")
    voice = read(metric / "voice-waveforms.json")
    natural = read(metric / "component-inference.json")
    world = read(metric / "world-parity.json")
    if (len(voice), len(natural), len(world)) != (32, 48, 10):
        raise ValueError("Missing portable checks")
    storage = read(ROOT / "work/storage.json")
    drive = read(bases()[0] / "drive-state.json")
    write(
        metric / "release.json",
        dict(
            complete=True,
            neural_fits=all_fits,
            training_counts={k: n for k, (n, _) in COUNTS.items()},
            generations=len(ck),
            generation_counts=count,
            verified_hdds=2,
            storage=storage,
            portable_models=len(pkg["worlds"]) + len(pkg["components"]) + 1,
            total_robot_acquisitions=len(sources),
            whole_robot_shards=sum(len(row["shards"]) for row in sources.values()),
            source_rows=len(lineage),
            named_sessions=["slwm-g13-speech-v6"],
            owned_units=cost["owned_units"],
            cumulative_owned_units=cost["cumulative_owned_units"],
            remaining_units=cost["remaining_units"],
            all_owned_released=cost["all_owned_released"],
            drive_at_release={k: v for k, v in drive.items() if k != "retry_unix"},
            next_experiment="Use the matched direct-action correction and physical system identification as baselines in a goal-intervention tabletop simulator, then validate executed outcomes with independent physical telemetry. Keep dense and spatial fields matched and audit the noun codec separately.",
        ),
    )
    print(
        "G13 complete",
        all_fits,
        len(ck),
        len(pkg["worlds"]) + len(pkg["components"]) + 1,
        flush=True,
    )


if __name__ == "__main__":
    main()
