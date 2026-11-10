"""A tiny CPU optimizer/RNG replay drill for a pair of independent runtimes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

import torch

from colab_artifacts import capture, restore, sync


CONFIG = {"width": 16, "seed": 81, "lr": 0.003, "dropout": 0.25}


def setup(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    width = CONFIG["width"]
    model = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.Dropout(CONFIG["dropout"]),
                                torch.nn.Linear(width, 2))
    return model, torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])


def update(model, opt, step):
    x = torch.randn(8, CONFIG["width"]) + random.random()
    target = x[:, step % CONFIG["width"]] > 0
    loss = torch.nn.functional.cross_entropy(model(x), target.long())
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def save(model, opt, step, work):
    path = work / "resume.pt"
    temp = work / "resume.tmp"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                "config": CONFIG, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()}, temp)
    temp.replace(path)
    return path


def produce(work, remote, run):
    work.mkdir(parents=True, exist_ok=True)
    model, opt = setup(CONFIG["seed"])
    with ThreadPoolExecutor(max_workers=1) as worker:
        futures = []
        for step in range(1, 5):
            update(model, opt, step)
            if step in (2, 4):
                file = save(model, opt, step, work)
                capture(work / "spool", run, step, {"resume.pt": file})
                futures.append(worker.submit(sync, work / "spool", remote))
        for future in futures:
            future.result()


def same(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def verify(work, remote, run):
    restored = work / "restored"
    manifest = restore(remote, run, restored)
    saved = torch.load(restored / "resume.pt", map_location="cpu", weights_only=True)
    if saved["config"] != CONFIG or manifest["step"] != saved["step"]:
        raise ValueError("Recovery fixture identity changed")
    step = saved["step"]
    reference, reference_opt = setup(CONFIG["seed"])
    expected = [update(reference, reference_opt, i) for i in range(1, 9)]
    expected_rng = (torch.get_rng_state(), random.getstate())
    model, opt = setup(999)
    model.load_state_dict(saved["model"])
    opt.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng"])
    random.setstate(saved["python_rng"])
    actual = [update(model, opt, i) for i in range(step + 1, 9)]
    checks = {"next_losses": same(actual, expected[step:]), "weights": same(model.state_dict(), reference.state_dict()),
              "optimizer": same(opt.state_dict(), reference_opt.state_dict()),
              "rng": same((torch.get_rng_state(), random.getstate()), expected_rng)}
    if not all(checks.values()):
        raise RuntimeError(f"Resumed training diverged: {checks}")
    print(json.dumps({"restored_step": step, "last_step": 8, "bit_exact": checks}))
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["produce", "verify"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--work", type=Path, default=Path("/content/slwm-drill"))
    parser.add_argument("--remote", default="slwm-drive:SLWM/v1")
    args = parser.parse_args()
    (produce if args.phase == "produce" else verify)(args.work, args.remote, args.run)
