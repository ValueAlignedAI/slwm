"""Non-neural recording-balanced, persistence, transition, cue-only and raw-signal controls."""

from collections import defaultdict
import numpy as np


def design(data, mode):
    scene = data["scene"].astype(np.float32)
    ac = data["wc" if mode == "wrong" else "ac"].astype(np.float32)
    words = data["wrong" if mode == "wrong" else "words"].astype(np.float32)
    if mode == "none":
        ac = np.zeros_like(ac)
        words = np.zeros_like(words)
    return np.concatenate(
        (scene.mean(1), scene[:, -1], ac.mean(1), words.mean(1)), axis=1
    )


def learn_ridge(x, y, group, penalty):
    _, inv, ct = np.unique(group, return_counts=True, return_inverse=True)
    w = 1.0 / ct[inv]
    w /= w.sum()
    mean = (x * w[:, None]).sum(0)
    center = x - mean
    scale = np.maximum(np.sqrt((center**2 * w[:, None]).sum(0)), 0.10)
    xx = (center / scale * np.sqrt(w[:, None])).astype(np.float64)
    yy = y.astype(np.float64)
    target_mean = (yy * w[:, None]).sum(0)
    yc = (yy - target_mean) * np.sqrt(w[:, None])
    if len(x) < xx.shape[1]:
        beta = xx.T @ np.linalg.solve(xx @ xx.T + np.eye(len(xx)) * penalty, yc)
    else:
        beta = np.linalg.solve(xx.T @ xx + np.eye(xx.shape[1]) * penalty, xx.T @ yc)
    return lambda features: (((features - mean) / scale) @ beta + target_mean)


def softmax(x, scale=1.0):
    x = x * scale
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def ce(prob, y):
    p = np.take_along_axis(prob, np.maximum(y, 0)[:, :, None], axis=2)[:, :, 0]
    return -np.log(np.maximum(np.where(y >= 0, p, 1e-7), 1e-7)).mean()


def baseline(train, development, targets, vocab):
    tm, tr = train
    _, dv = development
    nc = len(vocab)
    cu = np.flatnonzero(tr["cued"])
    y = tr["y"][cu]
    group = tr["group"][cu]
    _, gi, ct = np.unique(group, return_counts=True, return_inverse=True)
    weight = 1.0 / ct[gi]
    freq = np.full((3, nc), 0.001)
    edges = defaultdict(lambda: np.full(nc, 0.001))
    for i, ix in enumerate(cu):
        row = tm["rows"][ix]
        prior = row["previous"]
        for pos, v in enumerate(y[i]):
            if v >= 0:
                freq[pos, v] += weight[i]
                edges[prior][v] += weight[i]
            prior = row["verbs"][pos]
    freq /= freq.sum(1, keepdims=True)
    onehot = np.eye(nc)[y].reshape(len(cu), -1)
    raw = design(tr, "correct")[cu]
    raw_action = learn_ridge(raw, onehot, group, 0.2)
    lab = tr["state"][cu]
    ok = lab >= 0
    raw_state = learn_ridge(raw[ok], lab[ok, None], group[ok], 0.35)
    prior = float(
        np.mean([lab[(group == g) & ok].mean() for g in np.unique(group[ok])])
    )
    temps = (0.5, 1, 2, 4, 8, 16)
    dscore = raw_action(design(dv, "correct")).reshape(-1, 3, nc)
    temperature = min(temps, key=lambda t: ce(softmax(dscore, t), dv["y"]))
    # The list is a fixed lexical codec shared by all architectures, never fit on device test recordings.
    lex = (
        "grab",
        "place",
        "insert",
        "withdraw",
        "screw",
        "unscrew",
        "rotate",
        "open",
        "close",
        "press",
        "lift",
        "slide",
        "pull",
        "push",
        "align",
        "flip",
        "lock",
        "unlock",
        "pour",
    )

    def cue_distribution(data, mode, boost):
        source = data["wrong" if mode == "wrong" else "words"][:, :, 128:].max(1)
        assert source.shape[1] == len(lex)
        hits = np.zeros((len(source), nc))
        if mode != "none":
            for j, k in enumerate(lex):
                if k in vocab:
                    hits[:, vocab.index(k)] = source[:, j]
        return softmax(np.log(freq)[None, :, :] + boost * hits[:, None, :])

    boost = min(
        (0.0, 0.5, 1.0, 2.0, 4.0),
        key=lambda k: ce(cue_distribution(dv, "correct", k), dv["y"]),
    )
    results = {}
    for role, (meta, data) in targets.items():
        n = len(data["y"])
        out = {}
        probs = np.broadcast_to(freq, (n, 3, nc)).copy()
        out["frequency"] = (probs, None)
        trans = np.zeros_like(probs)
        last = np.zeros((n, 3), int)
        repeat_ok = []
        for i, r in enumerate(meta["rows"]):
            prev = r["previous"]
            last[i] = vocab.index(prev) if prev in vocab else -1
            repeat_ok.append([v == prev and bool(prev) for v in r["verbs"]])
            for pos in range(3):
                v = edges.get(prev)
                p = freq[pos] if v is None else (v + 3.0 * freq[pos]) / (v.sum() + 3.0)
                trans[i, pos] = p
                prev = vocab[int(np.argmax(p))]
        out["transition"] = (trans, None)
        out["repeat_correct"] = np.asarray(repeat_ok, dtype=bool)
        out["no_change"] = np.zeros(n)
        out["state_prior"] = np.full(n, prior)
        for mode in ("correct", "wrong", "none"):
            x = design(data, mode)
            out["raw_" + mode] = (
                softmax(raw_action(x).reshape(-1, 3, nc), temperature),
                np.clip(raw_state(x)[:, 0], 0, 1),
            )
            out["lexical_" + mode] = (cue_distribution(data, mode, boost), None)
        results[role] = out
    return results, dict(
        train_state_prior=prior, raw_calibration_scale=temperature, lexical_boost=boost
    )
