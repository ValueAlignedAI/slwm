"""Check the released compact models and decision readouts on actual held error targets."""

import argparse
import json
from pathlib import Path
import shutil
import jax
import jax.numpy as jnp
import numpy as np
from models.g9_portable import load
from models.g9_shared import Spec
from evals.g10_real import Bank, outcomes, predictor, predict_head
from g10_real_analysis import weighted_ranking
from g9_storage import sha, write


def main(holo, models, scores):
    bank = Bank(holo / "holo-novel.npz", True)
    starts, y, g, _ = outcomes(bank, holo, "novel")
    idx = np.searchsorted(np.unique(g), g)
    weights = np.ones((1, len(np.unique(g))), int)
    manifest = {}
    results = {}
    out = models / "risk-heads"
    out.mkdir(exist_ok=True)
    for core in ("hybrid", "gpt2"):
        tag = f"align-{core}-paired-s1003"
        meta, raw = load(models / (tag + ".npz"))
        p = jax.tree.map(jnp.asarray, raw)
        s = Spec(**meta["spec"])
        file = scores / (tag + "-risk-heads.npz")
        target = out / file.name
        shutil.copyfile(file, target)
        rec = json.loads((scores / (tag + ".json")).read_text())
        if (
            sha(target) != rec["risk"]["head_sha256"]
            or meta["parent_sha256"] != rec["sha256"]
        ):
            raise ValueError("Mismatched portable risk parent")
        manifest[file.name] = dict(
            sha256=sha(file), parent_sha256=rec["sha256"], bytes=file.stat().st_size
        )
        with np.load(file) as fitted:
            h = {
                k: fitted["all_" + k] for k in ("mean", "scale", "weight", "intercept")
            }
        zz = []
        for i in range(0, len(starts), 128):
            rr = bank.samples(starts[i : i + 128], g[i : i + 128])
            zz.append(np.asarray(predictor(s, True)(p, rr, np.ones(4, bool))))
        pred = predict_head(h, np.concatenate(zz))
        with np.load(scores / (tag + "-novel-risk.npz")) as db:
            f32 = db["all"]
        vals = {}
        for name, x in (("portable", pred), ("original", f32)):
            vals[name] = {
                k: float(v[0]) for k, v in weighted_ranking(y, x, idx, weights).items()
            }
        results[tag] = dict(
            held_recordings=len(np.unique(g)),
            events=len(y),
            mean_absolute_probability_change=float(np.abs(pred - f32).mean()),
            metrics=vals,
        )
    write(out / "receipts.json", manifest)
    write(out / "packaging-check.json", results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("holo", type=Path)
    ap.add_argument("models", type=Path)
    ap.add_argument("scores", type=Path)
    a = ap.parse_args()
    main(a.holo, a.models, a.scores)
