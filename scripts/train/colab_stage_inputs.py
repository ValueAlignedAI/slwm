"""Stage the four frozen G3 parents, source supplements, and prepared FineWeb corpus."""
import argparse
import json
from pathlib import Path

from colab_artifacts import capture, sha256


def checked(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Frozen digest mismatch: {path}")
    return path


def stage(repo, parents, spool):
    config = repo / "configs/t4/joint_transfer.json"
    cfg = json.loads(config.read_text())
    for parent, expected in cfg["parents"].values():
        file = checked(parents / parent / "best.pt", expected)
        capture(spool, f"g3-parent-{parent}", 0, {"best.pt": file, "joint_transfer.json": config})
    source = repo / cfg["data"] / "sources"
    manifest = source / "manifest.json"
    names = {"cifar": "cifar.tar.gz", "fsdd": "fsdd.zip"}
    files = {"manifest.json": manifest}
    for key, expected in json.loads(manifest.read_text())["sha256"].items():
        name = names.get(key, key)
        files[name] = checked(source / name, expected)
    for name in ("cifar_images.npy", "pusht.zip"):
        files[name] = source / name
    capture(spool, "g3-data-supplements", 0, files)
    corpus = repo / "artifacts/text/fineweb_edu_scale_v1"
    card = corpus / "dataset_card.json"
    files = {key: checked(corpus / key, value) for key, value in json.loads(card.read_text())["files"].items()}
    files.update({"dataset_card.json": card, "pinned_sources.json": corpus / "pinned_sources.json"})
    capture(spool, "g3-fineweb-edu", 0, files)


def stage_derived(repo, spool):
    config = json.loads((repo/"configs/t4/joint_transfer.json").read_text())
    for group, folder in (("depth6","depth_parents"), ("structured","structured_parents")):
        cfg = repo/config["output"]/group/"config.json"
        parents = json.loads(cfg.read_text())["parents"]
        for name, expected in parents.values():
            run = f"g3-parent-{group}-{name}"
            if any((spool/"receipts"/run).glob("000000000000-*.json")):
                continue
            # Use the physical mount, since Linux forbids hard links across bind aliases.
            src = (repo/config["data"]/folder/name).resolve()
            model = checked(src/"best.pt",expected)
            capture(spool,run,0,{"best.pt":model,"best.json":src/"best.json","config.json":cfg})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("repo", "spool"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--parents",type=Path)
    parser.add_argument("--derived-only",action="store_true")
    args = parser.parse_args()
    if args.derived_only:
        stage_derived(args.repo,args.spool)
    elif args.parents is not None:
        stage(args.repo, args.parents, args.spool)
    else:
        parser.error("Provide --parents or select --derived-only")
