"""Isolate real spoken direction evidence on held real displacements and source/speaker-disjoint inputs."""

import argparse
from functools import lru_cache
import hashlib
import json
import re
import numpy as np
import jax
import jax.numpy as jnp
from data.g10_language import text_features
from models.g13_world import Spec, propose
from evals.g13_predict import bank, records, parameters
from evals.g13_stats import Groups
from scripts.train.g13_optim import ROOT, save_npz
from scripts.train.g9_storage import sha, write
from training.g13_physics import group_mean

WORDS = ("left", "right", "up", "down")
DIRECTION = re.compile(
    r"\b(leftwards?|rightwards?|left|right|upwards?|downwards?|up|down|upper|lower|top|bottom)\b",
    re.I,
)
MODES = ("correct", "wrong", "none", "oracle")
ARMS = ("field", "protected", "audio")


def key(s):
    return int.from_bytes(
        hashlib.sha256(("G13/voices/" + s).encode()).digest()[:8], "big"
    )


def facing(v):
    return np.stack((-v[:, 0], v[:, 0], -v[:, 1], v[:, 1]), 1).argmax(1)


def calibration():
    r = bank("repaired", "train")
    ok = r["first"] & (r["mover"] > 0)
    flow = r["flow"][ok, r["mover"][ok] - 1]
    delta = r["endpoint"][ok]
    # Map axis orientation from training movements only. No language label participates.
    valid = (np.linalg.norm(flow, axis=1) > 6) & (np.linalg.norm(delta, axis=1) > 0.02)
    a = np.linalg.solve(
        delta[valid].T @ delta[valid] + 0.002 * np.eye(2), delta[valid].T @ flow[valid]
    )
    return a


def collection(role, external, sp, post, axis):
    r = bank("repaired", role, external)
    src = ROOT / "data" / ("robot-extra" if external else "robot")
    ep = json.loads((src / "banks/episodes.json").read_bytes())
    ix = np.flatnonzero(r["first"] & (r["direction"] >= 0))
    s = {k: v[ix] for k, v in r.items()}
    speakers = []
    choices = []
    clips = np.flatnonzero(sp["role"] == role)
    lookup = {}
    for i in clips:
        lookup.setdefault((str(sp["speaker"][i]), int(sp["y"][i])), []).append(int(i))
    a = []
    w = []
    o = []
    clean = []
    for row, ei, d in zip(ix, s["episode"], s["direction"]):
        rec = ep[ei]
        op = int(d) ^ 1
        pool = sorted(v for (v, k) in lookup if k == d and (v, op) in lookup)
        if not pool:
            raise ValueError("No same-speaker opposite")
        h = key(rec["episode"])
        name = pool[h % len(pool)]
        li = lookup[name, int(d)]
        wi = lookup[name, op]
        i = li[(h // len(pool)) % len(li)]
        j = wi[(h // 17) % len(wi)]
        phrase = " ".join(DIRECTION.sub(" ", rec["goal"]).split())
        if DIRECTION.search(phrase):
            raise ValueError("Retained a directional word")
        features = np.stack(
            [text_features(phrase + " " + word) for word in WORDS]
        ).astype("f4")
        a.append(post[i] @ features)
        w.append(post[j] @ features)
        o.append(features[d])
        clean.append(text_features(phrase))
        speakers.append(name)
        choices.append((i, j))
    pi = np.asarray(choices)
    text = dict(
        correct=np.asarray(a, "f4"),
        wrong=np.asarray(w, "f4"),
        none=np.asarray(clean, "f4"),
        oracle=np.asarray(o, "f4"),
    )
    prob = dict(
        correct=post[pi[:, 0]],
        wrong=post[pi[:, 1]],
        none=np.full((len(pi), 4), 0.25, "f4"),
        oracle=np.eye(4, dtype="f4")[s["direction"]],
    )
    srcidx = s["bind"][:, :, 0].argmax(1)
    has = s["bind"][:, :, 0].sum(1) > 0
    flow = s["flow"][np.arange(len(ix)), srcidx]
    known = (
        has
        & s["visible"][np.arange(len(ix)), srcidx]
        & (np.linalg.norm(flow, axis=1) >= 6)
    )
    obj = np.where(known, facing(flow), -1)
    eff = np.where(
        np.linalg.norm(s["endpoint"], axis=1) >= 0.01, facing(s["endpoint"] @ axis), -1
    )
    return (
        s,
        text,
        prob,
        dict(object=obj, effector=eff),
        dict(
            groups=s["group"],
            speaker=np.asarray(speakers),
            rows=ix,
            episode=s["episode"],
            actual_clip=sp["names"][pi[:, 0]],
            wrong_clip=sp["names"][pi[:, 1]],
            direction=s["direction"],
            strict=known & (s["mover"] == srcidx + 1),
        ),
    )


@lru_cache(maxsize=8)
def emit(core):
    @jax.jit
    def pred(p, r):
        z = propose(p, Spec(), core, "anchored", r)["field"]
        source = r["bind"][:, :, 0]
        active = source.sum(1, keepdims=True) > 0
        local = (z[:, 1:] * source[:, :, None]).sum(1)
        return 0.5 * (z[:, 0] + jnp.where(active, local, z[:, 1:].mean(1)))

    return pred


def fields(d, collection):
    r, text, prob, targets, meta = collection
    p = parameters(d, "repaired")
    result = {}
    from scripts.train.g13_robot_fit import feature_keys

    for mode in MODES:
        xs = []
        for i in range(0, len(meta["rows"]), 256):
            ix = np.minimum(np.arange(i, i + 256), len(meta["rows"]) - 1)
            inputs = {k: jnp.asarray(r[k][ix]) for k in feature_keys("anchored")}
            inputs["goal"] = jnp.asarray(text[mode][ix])
            xs.append(
                np.asarray(emit(d["core"])(p, inputs))[
                    : min(256, len(meta["rows"]) - i)
                ]
            )
        z = np.concatenate(xs)
        result[mode] = {
            "field": z,
            "protected": np.concatenate((z[:, :124], prob[mode]), 1),
            "audio": prob[mode],
        }
    return result


def probabilities(x, fit):
    score = ((x - fit["mean"]) / fit["std"]) @ fit["beta"] + fit["base"]
    z = score * float(fit["temperature"])
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def readout(train, dev, y, v, gr, gd):
    ok = y >= 0
    tr = train[ok].astype("f8")
    g = gr[ok]
    _, inv, n = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / n[inv]
    w /= w.sum()
    mu = (tr * w[:, None]).sum(0)
    std = np.maximum(np.sqrt(((tr - mu) ** 2 * w[:, None]).sum(0)), 0.06)
    x = (tr - mu) / std
    target = np.eye(4)[y[ok]]
    base = (target * w[:, None]).sum(0)
    S = x.T @ (x * w[:, None])
    cross = x.T @ ((target - base) * w[:, None])
    choices = []
    for reg in (0.02, 0.1, 0.5, 2.0):
        beta = np.linalg.solve(S + reg * np.eye(len(mu)), cross)
        for temp in (1.0, 2.0, 4.0, 8.0):
            f = dict(
                mean=mu,
                std=std,
                beta=beta,
                base=base,
                temperature=np.asarray(temp),
                reg=np.asarray(reg),
            )
            p = probabilities(dev, f)
            val = np.where(
                v >= 0,
                -np.log(np.maximum(p[np.arange(len(v)), np.maximum(v, 0)], 1e-8)),
                np.nan,
            )
            choices.append((group_mean(val, gd), reg, temp, f))
    score, reg, temp, fit = min(choices, key=lambda x: x[:3])
    return fit, dict(dev_nll=score, reg=reg, temperature=temp)


def filetag(tag, target, arm):
    return tag + "_" + target + "_" + arm + ".npz"


def main(phase, portable=False):
    speech = ROOT / "work" / ("speech-portable" if portable else "speech")
    folder = ROOT / "work" / ("voice-portable" if portable else "voice")
    folder.mkdir(exist_ok=True)
    (folder / "probes").mkdir(exist_ok=True)
    sp = np.load(ROOT / "data/speech/directions.npz")
    post = np.load(speech / "posterior.npz")["p"]
    axis = calibration()
    docs = [d for d in records("repaired") if d["recipe"] == "anchored"]
    specs = {}
    results = {}
    predictions = {}
    if phase == "develop":
        col = {r: collection(r, False, sp, post, axis) for r in ("train", "dev")}
        for d in docs:
            ft = {r: fields(d, v) for r, v in col.items()}
            tag = d["tag"]
            for y in ("object", "effector"):
                for arm in ARMS:
                    if arm == "audio" and tag != docs[0]["tag"]:
                        continue
                    fit, meta = readout(
                        ft["train"]["correct"][arm],
                        ft["dev"]["correct"][arm],
                        col["train"][3][y],
                        col["dev"][3][y],
                        col["train"][4]["groups"],
                        col["dev"][4]["groups"],
                    )
                    name = filetag("audio-only" if arm == "audio" else tag, y, arm)
                    path = folder / "probes" / name
                    save_npz(path, fit)
                    specs[name] = dict(**meta, sha256=sha(path))
        write(
            folder / "lock.json",
            dict(
                axis=axis.tolist(),
                probes=specs,
                selected_speech=json.loads((speech / "results.json").read_bytes())[
                    "selected"
                ],
            ),
        )
    else:
        lock = json.loads((folder / "lock.json").read_bytes())
        if not np.allclose(lock["axis"], axis):
            raise ValueError("Training-only camera orientation changed")
        for cohort, external in (("hold", False), ("fresh", True)):
            col = collection("test", external, sp, post, axis)
            meta = col[4]
            bs = Groups(meta["groups"])
            voice = Groups(meta["speaker"])
            for k, v in meta.items():
                predictions[cohort + "_" + k] = v
            for k, v in col[3].items():
                predictions[cohort + "_" + k] = v
            for d in docs:
                ft = fields(d, col)
                tag = d["tag"]
                for target in ("object", "effector"):
                    y = col[3][target]
                    good = y >= 0
                    for arm in ARMS:
                        if arm == "audio" and tag != docs[0]["tag"]:
                            continue
                        alias = "audio-only" if arm == "audio" else tag
                        name = filetag(alias, target, arm)
                        file = folder / "probes" / name
                        if sha(file) != lock["probes"][name]["sha256"]:
                            raise ValueError("Direction readout modified")
                        with np.load(file) as z:
                            fit = {k: z[k] for k in z.files}
                        hit = {}
                        nll = {}
                        key0 = "/".join((cohort, alias, target, arm))
                        for m in MODES:
                            prob = probabilities(ft[m][arm], fit)
                            hit[m] = np.where(good, prob.argmax(1) == y, np.nan)
                            nll[m] = np.where(
                                good,
                                -np.log(
                                    np.maximum(
                                        prob[np.arange(len(y)), np.maximum(y, 0)], 1e-8
                                    )
                                ),
                                np.nan,
                            )
                            predictions[key0.replace("/", "_") + "_" + m] = prob.astype(
                                "f4"
                            )
                        results[key0] = dict(
                            modes={
                                m: dict(accuracy=bs.score(hit[m]), nll=bs.score(nll[m]))
                                for m in MODES
                            },
                            actual_wrong=bs.score(hit["correct"] - hit["wrong"]),
                            actual_silent=bs.score(hit["correct"] - hit["none"]),
                            actual_wrong_by_speaker=voice.score(
                                hit["correct"] - hit["wrong"]
                            ),
                            strict_actual_wrong=(
                                bs.score(
                                    np.where(
                                        meta["strict"],
                                        hit["correct"] - hit["wrong"],
                                        np.nan,
                                    )
                                )
                                if target == "object"
                                else None
                            ),
                        )
        out = ROOT / "release/metrics"
        write(
            out
            / ("voice-portable-transfer.json" if portable else "voice-transfer.json"),
            results,
        )
        np.savez_compressed(
            out / ("voice-portable-cases.npz" if portable else "voice-cases.npz"),
            **predictions,
        )
    print("Completed controlled-voice phase", phase, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("develop", "confirm"))
    p.add_argument("--portable", action="store_true")
    a = p.parse_args()
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "cache"))
    main(a.phase, a.portable)
