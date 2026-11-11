"""Route open-vocabulary component forecasts from saved sensor features or a new spoken directive."""

import argparse
import json
from pathlib import Path
import jax.numpy as jnp
import jax
import numpy as np
from models.g13_component import forward
from scripts.infer.g13_world import BASE, artifact, read


def semantic(sentence, snapshot):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    net = (
        AutoModel.from_pretrained(str(snapshot), local_files_only=True)
        .to(device)
        .eval()
    )
    if device == "cuda":
        net = net.half()
    b = tok(
        [sentence], return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to(device)
    with torch.inference_mode():
        h = net(**b).last_hidden_state.float()
        m = b.attention_mask.float()[:, :, None]
        out = torch.nn.functional.normalize(
            (h * m).sum(1) / m.sum(1).clamp(min=1), dim=-1
        )
    return out.cpu().numpy().astype("f2")


def infer(
    scene,
    pkg,
    fold,
    core,
    arm,
    mode,
    wav=None,
    whisper=None,
    minilm=None,
    inspect=False,
):
    index = json.loads((pkg / "manifest.json").read_bytes())
    name = f"{fold}-{core}-{arm}"
    entry = index["components"][name]
    p = {k: jnp.asarray(v) for k, v in artifact(pkg, index, entry["file"]).items()}
    noun = artifact(pkg, index, "noun-prototypes.npz")
    prior = artifact(pkg, index, "component-priors.npz")[fold]
    r = {
        k: np.asarray(scene[k])
        for k in ("scene", "ac", "wc", "text", "alt", "oracle", "alt_oracle")
    }
    sentence = None
    if wav is not None:
        from scripts.infer.g12_live import raw_speech

        if len(r["ac"]) != 1 or whisper is None or minilm is None or mode != "correct":
            raise ValueError("Select one event and provide both local frozen codecs")
        ac, _, sentence = raw_speech(wav, whisper, [])
        r["ac"] = ac[None]
        r["text"] = semantic(sentence, minilm)
    logits, gate = forward(
        p,
        core,
        arm,
        {k: jnp.asarray(v) for k, v in r.items()},
        jnp.asarray(noun["semantic"]),
        jnp.asarray(noun["basis"]),
        jnp.asarray(prior),
        mode,
    )
    prob = np.asarray(jax.nn.softmax(logits))
    certainty = np.asarray(jax.nn.sigmoid(gate))
    rows = []
    for prediction, c in zip(prob, certainty):
        best = int(prediction.argmax())
        confidence = float(0.5 * (c + prediction[best]))
        emit = confidence >= entry["threshold"]
        d = dict(route="component" if emit else "internal_only", confidence=confidence)
        if emit:
            d["component"] = index["nouns"][best]
        if inspect:
            d["proposals"] = [
                dict(noun=index["nouns"][j], probability=float(prediction[j]))
                for j in np.argsort(-prediction)[:5]
            ]
        if sentence is not None:
            d["recognized"] = sentence
        rows.append(d)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path)
    p.add_argument("--package", type=Path, default=BASE)
    p.add_argument("--fold", choices=("A", "B"), required=True)
    p.add_argument("--core", default="gpt2")
    p.add_argument("--arm", choices=("semantic", "learned"), default="semantic")
    p.add_argument(
        "--mode", choices=("correct", "wrong", "none", "human"), default="correct"
    )
    p.add_argument("--wav", type=Path)
    p.add_argument("--whisper", type=Path)
    p.add_argument("--minilm", type=Path)
    p.add_argument("--inspect", action="store_true")
    a = p.parse_args()
    print(
        json.dumps(
            infer(
                read(a.scene),
                a.package,
                a.fold,
                a.core,
                a.arm,
                a.mode,
                a.wav,
                a.whisper,
                a.minilm,
                a.inspect,
            ),
            indent=2,
        )
    )
