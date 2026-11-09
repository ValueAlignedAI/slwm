"""Train-only linear probes of G2 states, scored on old selection groups."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(root, run, family):
    target = "target_previous_command" if family == "droid" else "target_last_physical"
    splits = []
    for role in ("train", "selection"):
        path = root / "readouts/features" / family / run / role
        xs = np.load(path / "shared.npy").astype(np.float64)
        ys = np.load(path / (target + ".npy")).astype(np.float64)
        if family == "har":
            ys = ys[:, :6]
        groups = np.array(json.loads((path / "cases.json").read_text())["groups"])
        splits.append((xs, ys, groups))
    (x, y, _), (v, w, groups) = splits
    mx, scale = x.mean(0), x.std(0).clip(.001)
    x, v = (x-mx)/scale, (v-mx)/scale
    my, sy = y.mean(0), y.std(0).clip(1e-6)
    y, w = (y-my)/sy, (w-my)/sy
    cov = x.T @ x / len(x)
    eig, vec = np.linalg.eigh(cov)
    order = np.argsort(eig)[::-1]
    scores = {}
    for k in (128, x.shape[1]):
        basis = vec[:, order[:k]]
        coef = (basis.T @ x.T @ y / len(x)) / (eig[order[:k], None] + .01)
        pred = (v @ basis) @ coef
        per_group = {}
        for name in sorted(set(groups)):
            i = groups == name
            per_group[name] = (1 - ((pred[i]-w[i])**2).mean(0) / (w[i]**2).mean(0).clip(1e-9)).tolist()
        scores[str(k)] = {"mean_group_coordinate_r2_vs_train_mean": float(np.mean(list(per_group.values()))),
                          "by_group_and_coordinate": per_group}
    return {"run": run, "family": family, "target": target, "rows_train":len(x), "rows_selection":len(v),
            "selection_groups": len(set(groups)), "standardized_participation_ratio":float(eig.sum()**2/(eig@eig)),
            "variance_in_first_128":float(eig[order[:128]].sum()/eig.sum()), "scores":scores}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("g2", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    results = [audit(args.g2, run, family) for family in ("har","droid") for run in ("G2-201","G2-202","G2-203","G2-204","RND-201","RND-202")]
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text(json.dumps({"split":"G2 train to selection only", "ridge":.01, "results":results},indent=2)+"\n")
