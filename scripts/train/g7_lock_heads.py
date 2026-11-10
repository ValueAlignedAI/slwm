"""Freeze supervised choices and alignment head identities before final evaluation."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def lock(path, record):
    if path.exists():
        if json.loads(path.read_bytes()) != record:
            raise ValueError(f"Locked model set changed in {path}")
        return
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def freeze(root):
    selected = {}
    heads = {}
    prior = sha(root / "reacher-prior/prior.npz")
    for seed in (711, 712):
        for core in ("hybrid", "gpt2"):
            tag = f"{core}-{seed}"
            eligible = []
            for mode in ("frozen", "adapter"):
                folder = root / "policy-prior" / f"{core}-{mode}-{seed}"
                dev = json.loads((folder / "dev.json").read_bytes())
                weight = sha(folder / "weights.npz")
                if dev["weights_sha256"] != weight or dev["prior_sha256"] != prior:
                    raise ValueError(f"Changed supervised artifacts at {folder}")
                res = dev["results"]
                if res["gate"]["passed"]:
                    eligible.append((res["correct"]["regret_cm"], mode, weight))
            if not eligible:
                selected[tag] = {"gate": False}
            else:
                regret, mode, weight = min(eligible)
                selected[tag] = {"gate": True, "mode": mode, "sha256": weight, "dev_regret_cm": regret}
            for variant in ("reconstruct", "units-sync"):
                folder = root / "alignment" / tag / variant
                dev = json.loads((folder / "dev.json").read_bytes())
                weight = sha(folder / "weights.pt")
                if weight != dev["weights_sha256"] or dev["updates"] != 2048:
                    raise ValueError(f"Incomplete alignment model at {folder}")
                heads[f"{tag}/{variant}"] = {"sha256": weight, "parent": dev["core_sha256"], "codebook": dev["codebook_sha256"]}
    lock(root / "policy-prior/sft-lock.json", {"models": selected, "prior": prior})
    lock(root / "alignment/test-lock.json", {"models": heads, "renderer": sha(root / "unit-renderer/renderer.pt"),
         "waveform_sample": "Seed 7719, round robin across held-out people, 64 utterances per corpus; all 8 fixed heads.",
         "speech_recognizer": "openai/whisper-tiny.en@87c7102498dcde7456f24cfd30239ca606ed9063"})
    print(json.dumps(selected, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    freeze(parser.parse_args().root)
