"""Export numerically faithful small and wide inference fields and score their causal operational replays."""

import json
import numpy as np
from models.g12_world import CORES, Spec
from scripts.infer.g12_live import load, run
from scripts.train.g12_fit import ROOT, read, read_params, predict, run_paths
from scripts.train.g9_storage import sha, write
from g12_custody import copy_checked, vaults


def main():
    data = ROOT / "data"
    out = ROOT / "release/inference"
    out.mkdir(parents=True, exist_ok=True)
    lex_path = ROOT / "release" / "lexicon.json"
    lex = json.loads(lex_path.read_bytes())
    public = dict(
        models={},
        lexicon=lex,
        storage="Lossless compressed original float32 training parameters.",
    )
    checks = {}
    ops = []
    for kind in ("primary", "wide", "early"):
        scale = kind == "wide"
        archives = run_paths(scale, False)
        folder = ROOT / (
            "early-eval" if kind == "early" else "scale-eval" if scale else "eval"
        )
        if kind == "early":
            archives = tuple(p.parent / "early" for p in archives)
        for fold in ("A", "B"):
            meta, batch = read(data, fold, "confirm", {})
            for core in CORES:
                tag = f"{fold}-{core}-paired-1201"
                alias = f"{fold}-{core}" + ("-w256" if scale else "")
                if kind == "early":
                    candidates = [
                        f"{fold}-{core}-{arm}-1201" for arm in ("full", "dropout")
                    ]

                    def dev_score(name):
                        r = json.loads((folder / name / "receipt.json").read_bytes())
                        return next(
                            x["selection"]
                            for x in r["curve"]
                            if x["step"] == r["selected"]
                        )

                    tag = min(candidates, key=dev_score)
                    alias = f"{fold}-{core}-early"
                receipt = json.loads((folder / tag / "receipt.json").read_bytes())
                base = archives[0] / tag / f"step-{receipt['selected']:05d}"
                spec = Spec(**receipt["config"]["spec"])
                p = read_params(base / "weights.npz")
                packed = {k: np.asarray(x, dtype=np.float32) for k, x in p.items()}
                file = out / (alias + ".npz")
                temp = file.with_suffix(".new.npz")
                np.savez_compressed(temp, **packed)
                temp.replace(file)
                entry = dict(
                    file=file.name,
                    sha256=sha(file),
                    source_sha256=receipt["weights_sha256"],
                    fold=fold,
                    core=core,
                    recipe=kind,
                    training_arm=receipt["config"]["treatment"],
                    width=spec.width,
                    spec=receipt["config"]["spec"],
                    trained_parameters=receipt["config"]["parameters"],
                    selected_update=receipt["selected"],
                    thresholds=receipt["thresholds"],
                    vocabulary=meta["vocabulary"],
                )
                public["models"][alias] = entry
                write(out / "manifest.json", public)
                _, _, packed_p = load(
                    out,
                    fold,
                    core,
                    spec.width,
                    "early" if kind == "early" else "primary",
                )
                scores = []
                with np.load(folder / tag / "confirm.npz") as known:
                    for mode in ("correct", "wrong", "none"):
                        prob, st, _ = predict(spec, core, packed_p, batch, mode)
                        verb_diff = float(np.max(np.abs(prob - known[mode + "_verb"])))
                        state_diff = float(np.max(np.abs(st - known[mode + "_state"])))
                        same = bool(
                            np.array_equal(
                                prob.argmax(-1), known[mode + "_verb"].argmax(-1)
                            )
                        )
                        if not same or max(verb_diff, state_diff) > 2e-5:
                            raise ValueError(
                                f"{alias} {mode}: export probability errors {verb_diff} and {state_diff}; argmax parity {same}"
                            )
                        scores.append(
                            dict(
                                mode=mode,
                                max_probability_error=verb_diff,
                                max_state_error=state_diff,
                                all_argmax_identical=same,
                            )
                        )
                checks[alias] = scores
                print("Checked portable inference", alias, flush=True)
    demos = ROOT / "release/examples"
    for example in json.loads((demos / "streams.json").read_bytes()):
        fam = example["family"]
        fold = example["fold"]
        labels = json.loads((demos / (fam + "-observations.json")).read_bytes())
        quiet = np.asarray([r["quiet_next_eight_seconds"] for r in labels])
        for core in CORES:
            for mode in ("correct", "wrong", "none"):
                actions = list(
                    run(
                        demos / (fam + "-stream.npz"),
                        out,
                        fold,
                        core,
                        mode,
                        batch_size=32,
                    )
                )
                if len(actions) != len(labels):
                    raise ValueError("Live clocks mismatch")
                act = np.array(["actions" in r for r in actions])
                effect = np.array(["device_change" in r for r in actions])
                noop = ~(act | effect)
                ops.append(
                    dict(
                        family=fam,
                        core=core,
                        mode=mode,
                        seconds=example["seconds"],
                        cycles=len(act),
                        action_commits=int(act.sum()),
                        state_commits=int(effect.sum()),
                        internal_only=int(noop.sum()),
                        quiet_clocks=int(quiet.sum()),
                        action_on_quiet=int((act & quiet).sum()),
                    )
                )
                if core == "associative" and mode == "correct":
                    (demos / (fam + "-decisions.jsonl")).write_text(
                        "".join(
                            json.dumps(r, separators=(",", ":")) + "\n" for r in actions
                        )
                    )
    write(ROOT / "release/metrics/inference-checks.json", checks)
    write(ROOT / "release/metrics/live.json", ops)
    inventory = {
        p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in out.iterdir()
        if p.is_file()
    }
    for d in vaults():
        for n, r in inventory.items():
            copy_checked(out / n, d / "inference" / n, r["sha256"])
        write(d / "inference/verified.json", inventory)
    print(
        "Verified",
        len(public["models"]),
        "packages and",
        len(ops),
        "operational replays",
        flush=True,
    )


if __name__ == "__main__":
    main()
