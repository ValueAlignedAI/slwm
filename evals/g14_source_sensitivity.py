"""Secondary uncertainty weighting both physics starts and reused original human recordings."""

from pathlib import Path
import json
import numpy as np
from scripts.train.g14_execute import tag
from scripts.train.g14_stages import await_files
from scripts.train.g9_storage import write

ROOT = Path("/mnt/slwm-data/SLWM/g14")


def estimate(metric, world, clip, stratum):
    _, w = np.unique(world, return_inverse=True)
    _, c = np.unique(clip, return_inverse=True)
    strata = [np.flatnonzero(stratum == key) for key in np.unique(stratum)]
    rng = np.random.default_rng(1410629)
    sims = []
    for i in range(40):
        weights = (
            rng.exponential(size=(100, w.max() + 1))[:, w]
            * rng.exponential(size=(100, c.max() + 1))[:, c]
        )
        sims.append(
            np.mean(
                [
                    (weights[:, idx] @ metric[idx])
                    / weights[:, idx].sum(1, keepdims=True)
                    for idx in strata
                ],
                axis=0,
            )
        )
    samples = np.concatenate(sims)
    v = np.mean([metric[ix].mean(0) for ix in strata], axis=0)
    lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
    return [
        dict(value=float(a), lo=float(b), hi=float(g)) for a, b, g in zip(v, lo, hi)
    ]


def main():
    await_files([ROOT / "release/metrics/physical-results.json"])
    panel = json.loads((ROOT / "work/final-control-lock.json").read_bytes())
    out = {}
    keys = (
        "spatial_minus_dense",
        "spatial_minus_geometry",
        "spatial_full_minus_constant",
        "dense_full_minus_constant",
        "pretrained_minus_original_speech",
    )
    for role in ("held", "fresh"):
        world = []
        clip = []
        stratum = []
        data = []
        folder = ROOT / "work/executed/final" / role
        for file in sorted(folder.glob("[0-9]*.json")):
            case = json.loads(file.read_bytes())
            r = case["results"]

            def pick(core, seed=1411, **change):
                return tag(
                    dict(
                        stage=panel["stage"],
                        core=core,
                        arm="protected",
                        seed=seed,
                        routed=panel["routed"],
                        contact_only=panel["contact_only"],
                        **change,
                    )
                )

            for j, cmd in enumerate(case["commands"]):
                metric = lambda key: r[key][str(j)]["scores"][j]["success"]
                std = np.mean(
                    [metric(pick("transport", s)) for s in (1411, 1412, 1413)]
                )
                dense = np.mean([metric(pick("gpt2", s)) for s in (1411, 1412, 1413)])
                base = metric(tag(dict(core="geometry", routed=panel["routed"])))
                data.append(
                    [
                        std - dense,
                        std - base,
                        metric(pick("transport"))
                        - metric(pick("transport", mean_field=True)),
                        metric(pick("gpt2")) - metric(pick("gpt2", mean_field=True)),
                        base
                        - metric(
                            tag(
                                dict(
                                    core="geometry",
                                    routed=panel["routed"],
                                    original_edge=True,
                                )
                            )
                        ),
                    ]
                )
                world.append(case["seed"])
                clip.append(f"{case['speaker']}/{cmd['key']}")
                stratum.append(f"{case['speaker']}/{case['shift']}")
        out[role] = dict(
            starts=len(set(world)),
            recordings=len(set(clip)),
            assignments=len(world),
            selective=dict(
                zip(keys, estimate(np.asarray(data), world, clip, np.asarray(stratum)))
            ),
        )
    write(
        ROOT / "work/source-sensitivity.json",
        dict(
            roles=out,
            draws=4000,
            conditioning="voice and contact stratum",
            weights="Independent mean-one exponential factors per physical start and original recording; product weights normalized within stratum. Secondary source-uncertainty analysis.",
        ),
    )
    print("Completed crossed scene and recording sensitivity", flush=True)


if __name__ == "__main__":
    main()
