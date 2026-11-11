"""Exploratory development-only recalibration of visible-state probabilities after inspecting primary Brier."""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize
from evals.g12_compare import RecordingBootstrap
from scripts.train.g12_fit import ROOT, read
from scripts.train.g9_storage import write


def fit(p, target, group, prior):
    keep = target >= 0
    y = target[keep]
    x = logit(np.clip(p[keep], 1e-6, 1 - 1e-6))
    ids, ig, c = np.unique(group[keep], return_inverse=True, return_counts=True)
    w = 1.0 / (len(ids) * c[ig])
    base = logit(np.clip(prior, 0.02, 0.98))

    def solve(null):
        def objective(v):
            a, b = v
            eta = a + b * x
            penalty = 0.05 * (a - base) ** 2 + 0.05 * b * b
            loss = np.sum(w * (np.logaddexp(0, eta) - y * eta)) + penalty
            resid = w * (expit(eta) - y)
            return loss, np.array(
                [resid.sum() + 0.10 * (a - base), np.dot(resid, x) + 0.10 * b]
            )

        optimum = minimize(
            objective,
            [base, 0],
            jac=True,
            bounds=[(-8, 8), (0, 0 if null else 4)],
            method="L-BFGS-B",
        )
        if not optimum.success:
            raise ValueError("Could not fit development calibration")
        return optimum.x

    return solve(False), solve(True)


def weighted_auc(rows, labels, scores):
    groups = np.array([r["recording"] for r in rows])
    family = np.array([r["family"] for r in rows])
    pieces = []
    for fam in sorted(set(family)):
        mask = (family == fam) & (labels >= 0)
        y = labels[mask]
        p = scores[mask]
        if len(np.unique(y)) != 2:
            continue
        _, idx, counts = np.unique(
            groups[mask], return_inverse=True, return_counts=True
        )
        w = 1.0 / counts[idx]
        diff = p[y == 1, None] - p[y == 0][None, :]
        v = (diff > 0) + 0.5 * (diff == 0)
        weights = w[y == 1, None] * w[y == 0][None, :]
        pieces.append(float((v * weights).sum() / weights.sum()))
    return float(np.mean(pieces)) if pieces else None


def main(out):
    root = ROOT / "data"
    labels = {
        (r["recording"], r["id"]): r["label"]
        for r in json.loads((root / "state-labels.json").read_bytes())["items"]
    }
    research = json.loads((out / "comparisons.json").read_bytes())
    res = {}
    for fold in ("A", "B"):
        prior = research["baselines"][fold]["recipe"]["train_state_prior"]
        _, dev = read(root, fold, "dev", labels)
        tests = {
            role: read(root, fold, role, labels) for role in ("confirm", "replica")
        }
        for kind, source in (
            ("primary", "eval"),
            ("wide", "scale-eval"),
            ("early", "early-eval"),
        ):
            for path in sorted((ROOT / source).glob(fold + "-*")):
                with np.load(path / "dev.npz") as z:
                    par, null = fit(
                        z["correct_state"], dev["state"], dev["group"], prior
                    )
                result = dict(
                    calibration=par.tolist(),
                    development_constant=float(expit(null[0])),
                    roles={},
                )
                for role, (meta, rows) in tests.items():
                    target = rows["state"]
                    bs = RecordingBootstrap(meta["rows"])
                    ok = target >= 0
                    with np.load(path / (role + ".npz")) as z:
                        p = z["correct_state"]
                        pred = expit(
                            par[0] + par[1] * logit(np.clip(p, 1e-6, 1 - 1e-6))
                        )

                    def loss(x):
                        return np.where(ok, (x - target) ** 2, np.nan)

                    result["roles"][role] = dict(
                        raw_brier=bs.estimates(loss(p)),
                        calibrated_brier=bs.estimates(loss(pred)),
                        gain_over_development_constant=bs.gain(
                            loss(pred), loss(expit(null[0])), "brier"
                        ),
                        gain_over_training_constant=bs.gain(
                            loss(pred), loss(prior), "brier"
                        ),
                        within_family_auc=weighted_auc(meta["rows"], target, p),
                    )
                res[kind + "/" + path.name] = result
    write(
        out / "calibration.json",
        dict(
            post_hoc=True,
            fit_split="development recording-balanced, positive slope, intercept-only control",
            models=res,
        ),
    )
    print("Development calibrations", len(res), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    main(p.parse_args().out)
