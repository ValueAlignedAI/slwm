"""Download versioned G3 inputs. Training opens only CIFAR's official train member."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile

import numpy as np

CONFIG = Path("configs/t4/joint_transfer.json")


def download(url, path):
    if not path.exists():
        temporary = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=90) as source, temporary.open("wb") as out:
            while block := source.read(1 << 20):
                out.write(block)
        temporary.replace(path)
    return path


def digest(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def main():
    cfg = json.loads(CONFIG.read_text())
    root = Path(cfg["data"]) / "sources"
    root.mkdir(exist_ok=True, parents=True)
    versions = cfg["source_revisions"]
    paths = {}
    cifar = download("https://cave.cs.toronto.edu/kriz/cifar-100-binary.tar.gz", root / "cifar.tar.gz")
    if digest(cifar, "md5") != versions["cifar_binary_md5"]:
        raise ValueError("CIFAR source checksum mismatch")
    with tarfile.open(cifar) as archive:
        with archive.extractfile("cifar-100-binary/train.bin") as stream:
            raw = np.frombuffer(stream.read(), np.uint8).reshape(50000, 3074)
        np.save(root / "cifar_images.npy", raw[:, 2:].reshape(-1, 3, 32, 32))
    paths["cifar"] = digest(cifar)
    for split in ("train", "t10k"):
        for kind, code in (("images", 3), ("labels", 1)):
            name = f"{split}-{kind}-idx{code}-ubyte.gz"
            url = f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/{versions['fashion']}/data/fashion/{name}"
            path = download(url, root / name)
            with gzip.open(path) as stream:
                if int.from_bytes(stream.read(4), "big") != 2048 + code:
                    raise ValueError(name)
            paths[name] = digest(path)
    for name in ("train.csv", "test.csv"):
        url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{versions['banking']}/banking_data/{name}"
        paths["banking_" + name] = digest(download(url, root / ("banking_" + name)))
    zip_path = download(f"https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/{versions['fsdd']}", root / "fsdd.zip")
    with zipfile.ZipFile(zip_path) as archive:
        if len([n for n in archive.namelist() if "/recordings/" in n and n.endswith(".wav")]) != 3000:
            raise ValueError("FSDD release changed")
    paths["fsdd"] = digest(zip_path)
    (root / "manifest.json").write_text(json.dumps({"revisions": versions, "sha256": paths, "cifar_train_images": 50000}, indent=2) + "\n")


if __name__ == "__main__":
    main()
