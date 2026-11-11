"""Publish both device folds for three fixed-seed models and verify compact inference against full weights."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_portable import pack, load
from scripts.train.g9_fit import numpy_weights, save_json
from scripts.train.g11_fit import parent, read_bank
from scripts.infer.g11_next import SIZES
from evals.g11_predict import infer
from scripts.train.g11_store import roots
from g9_storage import sha, write
import hashlib


def main(root, runs, parents, out):
    dest = out / "inference"
    examples = out / "examples"
    dest.mkdir(parents=True, exist_ok=True)
    examples.mkdir(exist_ok=True)
    banks = root / "banks"
    meta = {"backbones": {}, "adapters": {}, "checks": {}}
    grades = {
        (v["recording"], v["id"]): v["grade"]
        for v in json.loads((root.parent / "eval/device-audit.json").read_text())[
            "rows"
        ]
    }
    cat = json.loads((root / "annotations.json").read_text())
    examples_doc = {}
    for core in ("hybrid", "gpt2", "gated"):
        source, p, s = parent(parents, core, 1001)
        file = dest / (core + "-context-s1001.npz")
        pack(
            file,
            {k: np.asarray(v) for k, v in p.items()},
            source["spec"],
            source["inference_sha256"],
        )
        _, qu = load(file)
        compact = jax.tree.map(jnp.asarray, qu)
        meta["backbones"][core] = dict(
            file=file.name,
            bytes=file.stat().st_size,
            sha256=sha(file),
            source_sha256=source["inference_sha256"],
        )
        for fold in ("A", "B"):
            shutil.copyfile(
                banks / (fold + "-vocabulary.json"), dest / (fold + "-vocabulary.json")
            )
            voc = json.loads((dest / (fold + "-vocabulary.json")).read_text())
            rows = json.loads((banks / (fold + "-held.json")).read_text())["rows"]
            tag = f"g11-{fold}-{core}-paired-s1001"
            stage = runs / tag / "step-01024"
            d = json.loads((stage / "verified.json").read_text())
            full = numpy_weights(stage / "weights.npz")
            if d["parent_sha256"] != source["inference_sha256"] or d[
                "inference_sha256"
            ] != sha(stage / "weights.npz"):
                raise ValueError("Export source mismatch")
            adapter = dest / (tag + "-adapter.npz")
            np.savez_compressed(
                adapter, **{k: np.asarray(v, np.float16) for k, v in full.items()}
            )
            small = jax.tree.map(
                lambda v: jnp.asarray(v.astype(np.float16), jnp.float32), full
            )
            meta["adapters"][fold + "-" + core] = dict(
                file=adapter.name,
                bytes=adapter.stat().st_size,
                sha256=sha(adapter),
                source_sha256=d["inference_sha256"],
                parent_sha256=d["parent_sha256"],
            )
            bank, sem = read_bank(banks, fold, "held", True)
            ix = np.flatnonzero(bank["cued"])
            check = {}
            for mode in ("correct", "wrong"):
                a = infer(
                    p,
                    jax.tree.map(jnp.asarray, full),
                    s,
                    bank,
                    sem,
                    len(voc["verb"]),
                    len(voc["noun"]),
                    mode,
                    ix,
                )
                b = infer(
                    compact,
                    small,
                    s,
                    bank,
                    sem,
                    len(voc["verb"]),
                    len(voc["noun"]),
                    mode,
                    ix,
                )
                valid = bank["valid"][ix]
                truth = bank["effect"][ix, :2].sum(1)
                labels = bank["verb"][ix]
                check[mode] = dict(
                    events=len(ix),
                    device_effects=int(valid.sum()),
                    same_next_verb=float(np.mean(a["verb"] == b["verb"])),
                    original_micro_verb_accuracy=float(np.mean(a["verb"] == labels)),
                    portable_micro_verb_accuracy=float(np.mean(b["verb"] == labels)),
                    mean_absolute_effect_forecast_difference=float(
                        np.mean(
                            np.abs(
                                a["effect"][valid, :2].sum(1)
                                - b["effect"][valid, :2].sum(1)
                            )
                        )
                    ),
                    original_micro_effect_mae=float(
                        np.mean(np.abs(a["effect"][valid, :2].sum(1) - truth[valid]))
                    ),
                    portable_micro_effect_mae=float(
                        np.mean(np.abs(b["effect"][valid, :2].sum(1) - truth[valid]))
                    ),
                )
            meta["checks"][fold + "-" + core] = check
            if core == "hybrid":
                for family in sorted({r["family"] for r in rows}):
                    eligible = [
                        i
                        for i, r in enumerate(rows)
                        if r["family"] == family
                        and grades.get((r["recording"], r["action_id"])) == "D"
                    ]
                    pick = min(
                        eligible,
                        key=lambda i: hashlib.sha256(
                            (
                                "G11/example/"
                                + rows[i]["recording"]
                                + "/"
                                + str(rows[i]["action_id"])
                            ).encode()
                        ).hexdigest(),
                    )
                    r = rows[pick]
                    name = r["recording"]
                    case = examples / (family + ".npz")
                    np.savez_compressed(case, **{k: bank[k][pick] for k in SIZES})
                    source_audio = {}
                    for m, idkey in (("correct", "cue"), ("wrong", "wrong")):
                        utt = next(
                            v for v in cat[name]["speech"] if v["id"] == r[idkey]
                        )
                        wav = examples / (family + "-" + m + ".wav")
                        subprocess.run(
                            [
                                "ffmpeg",
                                "-nostdin",
                                "-v",
                                "error",
                                "-y",
                                "-ss",
                                str(utt["start"]),
                                "-i",
                                str(root / "fresh/media" / (name + "-av.mp4")),
                                "-t",
                                str(utt["end"] - utt["start"]),
                                "-map",
                                "0:a:0",
                                "-ac",
                                "1",
                                "-ar",
                                "16000",
                                "-c:a",
                                "pcm_s16le",
                                str(wav),
                            ],
                            check=True,
                        )
                        source_audio[m] = dict(
                            start=utt["start"],
                            end=utt["end"],
                            human_transcript=utt["text"],
                            asr=json.loads(
                                (root / "speech" / (name + ".json")).read_text()
                            )["utterances"][str(utt["id"])]["hypothesis"],
                        )
                    examples_doc[family] = dict(
                        fold=fold,
                        recording=name,
                        action_id=r["action_id"],
                        cut_tick=r["cut_tick"],
                        following_verb=r["verb"],
                        following_noun=r["noun"],
                        measured_changed_fraction=float(bank["effect"][pick, :2].sum()),
                        measured_mean_abs_mm=float(bank["effect"][pick, 2] * 100),
                        speech=source_audio,
                        sha256=sha(case),
                    )
            print(tag, check["correct"]["same_next_verb"], flush=True)
    save_json(dest / "portable.json", meta)
    save_json(examples / "outcomes.json", examples_doc)
    for disk in roots("runs"):
        folder = disk.parent / "inference"
        folder.mkdir(exist_ok=True)
        receipt = {}
        for f in dest.iterdir():
            if not f.is_file():
                continue
            to = folder / f.name
            shutil.copyfile(f, to)
            if sha(to) != sha(f):
                raise ValueError("Portable readback mismatch")
            receipt[f.name] = {"sha256": sha(to), "bytes": to.stat().st_size}
        write(folder / "verified.json", receipt)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("data", type=Path)
    a.add_argument("runs", type=Path)
    a.add_argument("parents", type=Path)
    a.add_argument("out", type=Path)
    a.add_argument("--cache")
    p = a.parse_args()
    if p.cache:
        jax.config.update("jax_compilation_cache_dir", p.cache)
    main(p.data, p.runs, p.parents, p.out)
