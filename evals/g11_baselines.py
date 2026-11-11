"""Raw sensor, speech-only, Markov, no-change, constant and oracle-action references."""

import argparse
import json
from pathlib import Path
import numpy as np
from scripts.train.g11_fit import read_bank


def summary(x):
    return np.concatenate([x[:, :8].mean(1), x[:, -8:].mean(1), x.std(1)], 1)


def inputs(r, mode, with_geo=True):
    b = len(r["cued"])
    image = (
        r["v"]
        .astype(np.float32)
        .reshape(b, 32, 4, 4, 4, 4, 3)
        .mean((3, 5))
        .reshape(b, 32, 48)
        / 127.5
        - 1
    )
    sensor = summary(
        np.concatenate(
            [r["a"].astype(np.float32) / 127.5 - 1, image, r["n"].astype(np.float32)],
            -1,
        )
    )
    if with_geo:
        sensor = np.c_[sensor, r["geometry"]]
    a = r["wc"] if mode in ("wrong", "wrong_human", "wrong_small") else r["ac"]
    lookup = {
        "correct": "aw",
        "wrong": "ww",
        "none": "aw",
        "asr_text": "aw",
        "acoustic": "aw",
        "human": "ah",
        "wrong_human": "wh",
        "small": "small_words",
        "wrong_small": "wrong_small",
    }
    w = r[lookup.get(mode, "aw")].astype(np.float32) / 127
    a = a.astype(np.float32).reshape(b, 4, 4, 128).mean(2).reshape(b, -1)
    if mode in ("none", "asr_text"):
        a = np.zeros_like(a)
    if mode in ("none", "acoustic"):
        w = np.zeros_like(w)
    return sensor, a, w


def fit(x, y, valid, classification=True):
    x = x[valid].astype(np.float64)
    y = y[valid].astype(np.float64)
    mu = x.mean(0)
    std = np.maximum(x.std(0), 0.06)
    x = (x - mu) / std
    center = y.mean(0)
    scale = 0.025 * len(x)
    weights = np.linalg.solve(x.T @ x + scale * np.eye(x.shape[1]), x.T @ (y - center))
    return mu, std, center, weights


def use(h, x):
    m, sd, base, w = h
    return (x - m) / sd @ w + base


def main(banks, out, fold, full):
    out.mkdir(parents=True, exist_ok=True)
    voc = json.loads((banks / (fold + "-vocabulary.json")).read_text())
    train, sem = read_bank(banks, fold, "train", full)
    nv, nn = len(voc["verb"]), len(voc["noun"])
    keep = train["cued"]
    ids = np.flatnonzero(keep)

    def feats(data, mode):
        # Chunk conversion avoids a dense float copy of an entire room corpus.
        pieces = []
        for start in range(0, len(data["cued"]), 128):
            pieces.append(
                inputs(
                    {k: v[start : start + 128] for k, v in data.items()},
                    mode,
                    with_geo=full,
                )
            )
        return tuple(np.concatenate(g) for g in zip(*pieces))

    tr = feats(train, "correct")

    def select(t, model):
        if model == "raw_joint":
            return np.concatenate(t, 1)
        if model == "raw_history":
            return t[0]
        if model == "speech_only":
            return np.c_[t[1], t[2]]
        return t[2]

    yv = np.eye(nv, dtype=np.float32)[train["verb"]]
    yn = np.eye(nn, dtype=np.float32)[train["noun"]]
    st = sem[np.maximum(train["open_noun"], 0)]
    labels = np.c_[yv, yn, st]
    raw = {}
    for model in ("raw_joint", "raw_history", "speech_only", "lexical"):
        x = select(tr, model)
        raw[model] = [fit(x, labels, keep)]
        if full:
            raw[model].append(fit(x, train["effect"], train["valid"], False))
    # A previous annotation is available to this stronger sequential reference only.
    ma = np.full((nv + 1, nv), 0.2, dtype=np.float64)
    np.add.at(ma, (train["prev_verb"][ids] + 1, train["verb"][ids]), 1)
    mn = np.full((nn + 1, nn), 0.2, dtype=np.float64)
    np.add.at(mn, (train["prev_noun"][ids] + 1, train["noun"][ids]), 1)
    freq = int(yv[ids].sum(0).argmax())
    frn = int(yn[ids].sum(0).argmax())
    norm = sem / np.maximum(1e-7, np.linalg.norm(sem, axis=1, keepdims=True))
    arrays = {}

    def store(role, name, verb, noun, open_noun, eff=None):
        for key, val in (("verb", verb), ("noun", noun), ("open_noun", open_noun)):
            arrays[role + "|" + name + "|" + key] = np.asarray(val, np.int16)
        if eff is not None:
            arrays[role + "|" + name + "|effect"] = np.asarray(eff, np.float32)

    for role in ("dev", "held", "familiar"):
        d, _ = read_bank(banks, fold, role, full)
        N = len(d["cued"])
        pv = d["prev_verb"]
        pn = d["prev_noun"]
        if full:
            active = train["effect"][train["valid"]]
            mean = active.mean(0)
            med = np.median(active, axis=0)
            changed_median = np.median(active[:, :2].sum(1))
            ratio = mean[0] / max(1e-9, mean[:2].sum())
            med[:2] = [changed_median * ratio, changed_median * (1 - ratio)]
            byverb = np.stack(
                [
                    train["effect"][(train["verb"] == i) & train["valid"]].mean(0)
                    if np.any((train["verb"] == i) & train["valid"])
                    else mean
                    for i in range(nv)
                ]
            )
            for key, v in (
                ("persistence", np.zeros((N, 3))),
                ("training_mean", np.tile(mean, (N, 1))),
                ("training_median", np.tile(med, (N, 1))),
                (
                    "oracle_verb",
                    np.where(
                        (d["verb"] >= 0)[:, None],
                        byverb[np.maximum(0, d["verb"])],
                        mean[None, :],
                    ),
                ),
            ):
                store(role, key, np.full(N, freq), np.full(N, frn), np.full(N, -1), v)
        store(
            role, "action_frequency", np.full(N, freq), np.full(N, frn), np.full(N, -1)
        )
        store(role, "repeat_last_action", pv, pn, np.full(N, -1))
        store(
            role,
            "action_transition",
            ma[pv + 1].argmax(-1),
            mn[pn + 1].argmax(-1),
            np.full(N, -1),
        )
        for mode in (
            (
                "correct",
                "wrong",
                "none",
                "acoustic",
                "asr_text",
                "human",
                "wrong_human",
                "small",
                "wrong_small",
            )
            if full
            else ("correct", "wrong", "none", "human")
        ):
            xx = feats(d, mode)
            for model, heads in raw.items():
                if model == "raw_history" and mode != "correct":
                    continue
                pred = use(heads[0], select(xx, model))
                vp = pred[:, :nv].argmax(-1)
                npred = pred[:, nv : nv + nn].argmax(-1)
                op = (pred[:, nv + nn :] @ norm.T).argmax(-1)
                ef = np.clip(use(heads[1], select(xx, model)), 0, 1) if full else None
                if ef is not None:
                    ef[:, :2] /= np.maximum(1.0, ef[:, :2].sum(1, keepdims=True))
                store(role, model + "_" + mode, vp, npred, op, ef)
    file = out / f"baselines-{fold}-{'joint' if full else 'action'}.npz"
    np.savez_compressed(file, **arrays)
    print(file, len(ids), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("banks", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("stage", choices=("action", "joint"))
    a = p.parse_args()
    for f in ("A", "B"):
        main(a.banks, a.out, f, a.stage == "joint")
