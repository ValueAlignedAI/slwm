"""Release lossless model/edge parameters and deterministic intervention fixtures without Mac-side training media."""

import hashlib
import json
import shutil
import tarfile
import numpy as np
from scripts.train.g13_optim import ROOT, roots, save_npz
from scripts.train.g9_storage import sha, write
from evals.g13_predict import bank
from scripts.train.g13_component_fit import count_prior

PUBLIC = ROOT / "release"


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def select_key(value):
    return hashlib.sha256(("G13/public-fixture/" + str(value)).encode()).hexdigest()


def main():
    pkg = PUBLIC / "inference"
    examples = PUBLIC / "examples"
    pkg.mkdir(parents=True, exist_ok=True)
    examples.mkdir(parents=True, exist_ok=True)
    manifest = dict(worlds={}, components={}, files={}, nouns=[])

    def cp(file, name, expected=None):
        h = sha(file)
        if expected and h != expected:
            raise ValueError("Unexpected chosen source weights")
        dest = pkg / name
        shutil.copyfile(file, dest)
        if sha(dest) != h:
            raise ValueError("Published artifact copy differed")
        manifest["files"][name] = dict(sha256=h, bytes=dest.stat().st_size)

    def save(name, items):
        save_npz(pkg / name, items)
        manifest["files"][name] = dict(
            sha256=sha(pkg / name), bytes=(pkg / name).stat().st_size
        )

    chosen = json.loads((ROOT / "work/score/model-lock.json").read_bytes())
    folder = roots("robot-corrected")[0]
    for recipe, cores in (
        ("linear", ("gpt2", "transport", "dense_spatial", "associative")),
        (
            "anchored",
            (
                "gpt2",
                "transport",
                "gated",
                "oscillator",
                "associative",
                "dense_spatial",
            ),
        ),
    ):
        for core in cores:
            tag = f"{core}-{recipe}-1301"
            rec = chosen["repaired/" + tag]
            name = f"world-{core}-{recipe}.npz"
            cp(
                folder / tag / f"step-{rec['step']:05d}" / "weights.npz",
                name,
                rec["sha256"],
            )
            manifest["worlds"][f"{core}-{recipe}"] = dict(**rec, file=name)
    r = ROOT / "work/repaired"
    cp(r / "system-id.npz", "system-id.npz")
    cp(r / "adaptive.npz", "adaptive.npz")
    manifest["adaptive_kind"] = json.loads((r / "adaptive-lock.json").read_bytes())[
        "best"
    ]["kind"]
    direct = json.loads((ROOT / "work/direct/lock.json").read_bytes())
    cp(ROOT / "work/direct/probes/raw.npz", "direct-raw.npz", direct["raw"]["sha256"])
    for core in ("gpt2", "transport", "dense_spatial", "associative"):
        k = f"{core}-linear-1301-protected"
        entry = direct[k]
        f = ROOT / "work/direct/probes" / (k + ".npz")
        if sha(f) != entry["sha256"]:
            raise ValueError("Direct head differs")
        save("direct-" + core + ".npz", dict(**read(f), alpha=np.array(entry["alpha"])))
    sp = json.loads((ROOT / "work/speech-portable/results.json").read_bytes())
    manifest["speech"] = sp["selected"]
    manifest["speech_operator"] = "explicit_patches"
    cp(ROOT / "work/speech-portable/selected.npz", "direction-codec.npz")
    vlock = json.loads((ROOT / "work/voice-portable/lock.json").read_bytes())
    for core in ("gpt2", "transport"):
        for target in ("effector", "object"):
            for arm in ("field", "protected"):
                k = f"{core}-anchored-1301_{target}_{arm}.npz"
                cp(
                    ROOT / "work/voice-portable/probes" / k,
                    f"voice-{core}-{target}-{arm}.npz",
                    vlock["probes"][k]["sha256"],
                )
    croot = ROOT / "data/components/component-banks"
    ci = json.loads((croot / "inventory.json").read_bytes())
    manifest["nouns"] = ci["nouns"]
    manifest["semantic_edge"] = dict(
        parameters=ci["edge_parameters"], revision=ci["revision"]
    )
    cp(croot / "prototypes.npz", "noun-prototypes.npz")
    priors = {}
    cl = json.loads((ROOT / "work/components/lock.json").read_bytes())
    cf = roots("components")[0]
    for fold in "AB":
        priors[fold] = count_prior(
            read(croot / (fold + "-train.npz")), len(ci["nouns"])
        )
        for core in ("gpt2", "gated", "oscillator", "associative"):
            for arm in ("learned", "semantic"):
                k = f"{fold}-{core}-{arm}-1301"
                entry = cl[k]
                rel = f"step-{entry['best']['step']:05d}"
                name = f"component-{k}.npz"
                cp(cf / k / rel / "weights.npz", name, entry["sha"])
                manifest["components"][f"{fold}-{core}-{arm}"] = dict(
                    file=name,
                    threshold=entry["best"]["threshold"],
                    step=entry["best"]["step"],
                    parameters=entry["parameters"],
                )
    save("component-priors.npz", priors)
    write(pkg / "manifest.json", manifest)
    # Direction-wave examples are selected by source ID, independent of scores.
    voice = read(PUBLIC / "metrics/voice-portable-cases.npz")
    robot = bank("repaired", "test", True)
    ep = json.loads((ROOT / "data/robot-extra/banks/episodes.json").read_bytes())
    case = []
    wanted = {}
    for direction, word in enumerate(("left", "right", "up", "down")):
        pool = np.flatnonzero(
            (voice["fresh_direction"] == direction) & (voice["fresh_object"] >= 0)
        )
        i = min(
            pool, key=lambda v: select_key(ep[voice["fresh_episode"][v]]["episode"])
        )
        j = voice["fresh_rows"][i]
        rec = ep[voice["fresh_episode"][i]]
        target = examples / ("voice-" + word + ".npz")
        save_npz(
            target,
            {k: robot[k][j : j + 1] for k in ("px", "geo", "kin", "plan", "tick")},
        )
        for key, suf in (("fresh_actual_clip", "true"), ("fresh_wrong_clip", "foil")):
            source = str(voice[key][i])
            wanted.setdefault(source, []).append(examples / f"voice-{word}-{suf}.wav")
        case.append(
            dict(
                word=word,
                bank_row=int(j),
                voice_index=int(i),
                episode=rec["episode"],
                goal=rec["goal"],
                group=rec["group"],
                object_direction=int(voice["fresh_object"][i]),
                effector_direction=int(voice["fresh_effector"][i]),
                spoken_source=str(voice["fresh_actual_clip"][i]),
                foil_source=str(voice["fresh_wrong_clip"][i]),
            )
        )
    with tarfile.open(ROOT / "data/speech/speech_commands_v0.02.tar.gz", "r|gz") as tar:
        for info in tar:
            name = info.name.removeprefix("./")
            if name in wanted:
                data = tar.extractfile(info).read()
                for path in wanted[name]:
                    path.write_bytes(data)
            if name in ("LICENSE", "README.md"):
                (
                    examples
                    / (
                        "Speech-Commands-"
                        + ("LICENSE.txt" if name == "LICENSE" else "README.txt")
                    )
                ).write_bytes(tar.extractfile(info).read())
    write(examples / "voice-examples.json", case)
    from scripts.data.g12_live import audio

    source = ROOT / "data/components"
    ann = json.loads((source / "annotations.json").read_bytes())
    natural = []
    for fold in "AB":
        b = read(croot / (fold + "-test.npz"))
        m = json.loads((croot / (fold + "-test.json")).read_bytes())
        i = min(
            range(len(m["rows"])),
            key=lambda v: select_key(
                m["rows"][v]["recording"] + "/" + str(m["rows"][v]["id"])
            ),
        )
        row = m["rows"][i]
        n = row["recording"]
        by = {v["id"]: v for v in ann[n]["speech"]}
        save_npz(
            examples / f"component-{fold}.npz",
            {
                k: b[k][i : i + 1]
                for k in ("scene", "ac", "wc", "text", "alt", "oracle", "alt_oracle")
            },
        )
        for nm, key in (("true", "cue"), ("foil", "wrong")):
            audio(
                source / "fresh/media" / (n + "-av.mp4"),
                by[row[key]],
                examples / f"component-{fold}-{nm}.wav",
            )
        natural.append(
            dict(
                fold=fold,
                index=i,
                **row,
                actual_transcript=by[row["cue"]]["text"],
                foil_transcript=by[row["wrong"]]["text"],
            )
        )
    write(examples / "component-examples.json", natural)
    print(
        "Released",
        len(manifest["worlds"]),
        "worlds,",
        len(manifest["components"]),
        "noun models and signed interventions",
        flush=True,
    )


if __name__ == "__main__":
    main()
