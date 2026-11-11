"""Ground the released noun inventory against native acoustic, visual and automatically recognized speech."""

import argparse
from collections import defaultdict
import json
import numpy as np
from g13_acquire import ROOT, get, write, file_hash

REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
G10 = ROOT.parent.parent / "g10/data/holo"
G11 = ROOT.parent.parent / "g11/data"
G12 = ROOT.parent.parent / "g12/data"
DIR = ROOT / "components"
FOLDS = {"A": {"coffee", "switch"}, "B": {"printer", "camera"}}


def download():
    folder = DIR / "minilm"
    folder.mkdir(parents=True, exist_ok=True)
    notes = {}
    for file in (
        "config.json",
        "model.safetensors",
        "vocab.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        notes[file] = get(
            "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/"
            + REV
            + "/"
            + file,
            folder / file,
        )
    write(folder / "source.json", dict(revision=REV, files=notes))
    print("Pinned MiniLM files", len(notes), flush=True)


def vectorize(strings, folder):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = (
        AutoModel.from_pretrained(folder, local_files_only=True).cuda().eval().half()
    )
    out = []
    with torch.inference_mode():
        for i in range(0, len(strings), 128):
            batch = tok(
                strings[i : i + 128],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            ).to("cuda")
            h = model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].float()[:, :, None]
            z = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            z = torch.nn.functional.normalize(z, dim=-1)
            out.append(z.cpu().numpy())
    return np.concatenate(out), sum(v.numel() for v in model.parameters())


def ontology():
    return sorted(
        {
            s.strip().split(maxsplit=1)[1].split("-", 1)[1].casefold()
            for s in (G11 / "fine_grained_actions_map.txt").read_text().splitlines()
            if "-" in s and len(s.split(maxsplit=1)) == 2
        }
    )


def main():
    dest = DIR / "component-banks"
    dest.mkdir(exist_ok=True)
    raw = json.loads((G10 / "labels.json").read_bytes())
    noun = {}
    for rec in raw:
        acts = sorted(
            (
                a
                for a in rec["events"]
                if a["label"] == "Fine grained action"
                and str(a["attributes"].get("Verb", "")).strip().casefold()
                not in ("", "none")
                and 0 <= a["start"] < a["end"]
            ),
            key=lambda a: (a["start"], a["id"]),
        )
        noun[rec["video_name"]] = {
            int(e["id"]): [
                (str(x["attributes"].get("Noun", "")).strip().casefold(), x["start"])
                for x in acts[i : i + 3]
            ]
            for i, e in enumerate(acts)
        }
    candidates = ontology()
    idx = {n: i for i, n in enumerate(candidates)}
    groups = json.loads((G12 / "groups.json").read_bytes())
    meta = {}
    refs = defaultdict(dict)
    strings = set(n.replace("_", " ") for n in candidates)
    for role in ("train", "dev", "test"):
        pool = {}
        for f in "AB":
            root = DIR / "banks" if role == "test" else G12 / "banks"
            p = root / f"{f}-{role}.npz"
            d = json.loads(p.with_suffix(".json").read_bytes())
            for i, row in enumerate(d["rows"]):
                if row["cue"] >= 0 or role == "train":
                    pool[(row["recording"], row["id"])] = (str(p), i, row)
        for f, held in FOLDS.items():
            if role == "test":
                entries = [e for e in pool.values() if e[2]["family"] in held]
            else:
                eligible = sorted(
                    {
                        e[2]["recording"]
                        for e in pool.values()
                        if e[2]["family"] not in held
                    }
                )
                names = set(eligible[:256] if role == "train" else eligible)
                entries = [e for e in pool.values() if e[2]["recording"] in names]
            entries.sort(
                key=lambda q: (q[2]["recording"], q[2]["cut_tick"], q[2]["id"])
            )
            meta[(f, role)] = entries
            for _, _, row in entries:
                n = row["recording"]
                if n in refs or row["cue"] < 0:
                    continue
                path = (DIR if role == "test" else G11) / "speech-small" / (n + ".json")
                if not path.is_file():
                    path = G12 / "speech-small" / (n + ".json")
                ss = json.loads(path.read_bytes())["utterances"]
                refs[n] = {int(k): v["hypothesis"] for k, v in ss.items()}
                strings.update(refs[n].values())
    catalog = json.loads((G11 / "annotations.json").read_bytes())
    catalog.update(json.loads((G12 / "annotations.json").read_bytes()))
    catalog.update(json.loads((DIR / "annotations.json").read_bytes()))
    human = {
        n: {s["id"]: s["text"] for s in cat["speech"]} for n, cat in catalog.items()
    }
    for n in refs:
        strings.update(human[n].values())
    strings.add("")
    ordered = sorted(strings)
    emb, params = vectorize(ordered, DIR / "minilm")
    lookup = {s: i for i, s in enumerate(ordered)}
    label = emb[[lookup[n.replace("_", " ")] for n in candidates]]
    # Public noun prototypes alone determine the fixed 128-dimensional semantic axes.
    _, _, basis = np.linalg.svd(label.astype("f8"), full_matrices=False)
    basis = basis[:128].T.astype("f4")
    semantic = label @ basis
    semantic /= np.maximum(np.linalg.norm(semantic, axis=1, keepdims=True), 1e-8)
    np.savez_compressed(
        dest / "prototypes.npz", semantic=semantic, full=label, basis=basis
    )
    cache = {}
    for (fold, role), entries in meta.items():
        rows = []
        examples = []
        for name, i, r in entries:
            if name not in cache:
                with np.load(name) as data:
                    cache[name] = {k: data[k] for k in ("scene", "ac", "wc")}
            rec = r["recording"]
            timeline = noun[rec][r["id"]]
            if len(timeline) != 3 or not np.allclose(
                [v[1] for v in timeline], r["target_onsets"]
            ):
                raise ValueError("Outcome clocks differ")
            target = [idx.get(v[0], -1) for v in timeline]
            cue = r["cue"]
            other = r["wrong"]
            modes = (
                (
                    refs[rec].get(cue, ""),
                    refs[rec].get(other, ""),
                    human[rec].get(cue, ""),
                    human[rec].get(other, ""),
                )
                if cue >= 0
                else ("", "", "", "")
            )
            row = {k: arr[i] for k, arr in cache[name].items()}
            row.update(
                {
                    k: emb[lookup[s]].astype("f2") if s.strip() else np.zeros(384, "f2")
                    for k, s in zip(("text", "alt", "oracle", "alt_oracle"), modes)
                }
            )
            row.update(y=np.array(target, np.int32), cued=cue >= 0)
            rows.append(row)
            examples.append(
                dict(
                    recording=rec,
                    id=r["id"],
                    family=r["family"],
                    cut=r["cut_tick"],
                    spoken=modes[0],
                    foil=modes[1],
                    cue=cue,
                    wrong=other,
                    nouns=[v[0] for v in timeline],
                    onsets=r["target_onsets"],
                )
            )
        names = sorted({r["recording"] for r in examples})
        groups = np.array([names.index(r["recording"]) for r in examples], np.int32)
        block = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        block["group"] = groups
        path = dest / f"{fold}-{role}.npz"
        np.savez_compressed(path, **block)
        write(
            path.with_suffix(".json"),
            dict(names=names, rows=examples, sha256=file_hash(path)),
        )
        print(
            "Component rows",
            fold,
            role,
            len(rows),
            "cued",
            int(block["cued"].sum()),
            "groups",
            len(names),
            flush=True,
        )
    write(
        dest / "inventory.json",
        dict(
            nouns=candidates, edge_parameters=params, revision=REV, strings=len(ordered)
        ),
    )
    (dest / "ready").touch()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("download", "build"))
    a = p.parse_args()
    download() if a.phase == "download" else main()
