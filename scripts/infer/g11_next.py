"""Predict the next device manipulation and visible surface change from a causal G11 case."""

import argparse
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from data.g10_language import text_features
from models.g9_portable import load
from models.g9_shared import Spec
from models.g11_instruction import predict

MODES = (
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
SIZES = {
    "a": (32, 64),
    "v": (32, 768),
    "n": (32, 12),
    "ac": (16, 128),
    "wc": (16, 128),
    "aw": (128,),
    "ww": (128,),
    "ah": (128,),
    "wh": (128,),
    "small_words": (128,),
    "wrong_small": (128,),
    "geometry": (201,),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def real_audio(path, model_dir):
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = gcd(int(sr), 16000)
        x = resample_poly(x, 16000 // g, int(sr) // g)
    if not 3200 <= len(x) <= 28 * 16000:
        raise ValueError("Use one completed instruction of 0.2 to 28 seconds")
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            "openai/whisper-tiny.en",
            revision="87c7102498dcde7456f24cfd30239ca606ed9063",
            local_files_only=True,
        )
    proc = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = (
        WhisperForConditionalGeneration.from_pretrained(
            model_dir,
            local_files_only=True,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        .to(device)
        .eval()
    )
    inp = proc.feature_extractor(
        [x],
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
        device=device,
    )
    mel = inp.input_features.to(device, dtype=next(net.parameters()).dtype)
    with torch.inference_mode():
        enc = net.model.encoder(mel, return_dict=True)
        h = (
            enc.last_hidden_state[0, : max(1, min(1500, int(np.ceil(len(x) / 320))))]
            .float()
            .cpu()
            .numpy()
        )
        seq = net.generate(
            input_features=mel,
            encoder_outputs=enc,
            attention_mask=inp.attention_mask.to(device),
            max_new_tokens=96,
            num_beams=1,
            do_sample=False,
        )
        words = proc.batch_decode(seq, skip_special_tokens=True)[0].strip()
    edges = np.linspace(0, len(h), 17).astype(int)
    pooled = np.stack(
        [
            h[min(a, len(h) - 1) : max(min(a, len(h) - 1) + 1, b)].mean(0)
            for a, b in zip(edges[:-1], edges[1:])
        ]
    )
    proj = np.linalg.qr(np.random.default_rng(4173).standard_normal((384, 128)))[
        0
    ].astype(np.float32)
    return np.clip(pooled @ proj, -20, 20).astype("f2"), text_features(words), words


def run(folder, case, fold, core, mode, audio=None, asr_dir=None, transcript=None):
    manifest = json.loads((folder / "portable.json").read_text())
    base = manifest["backbones"][core]
    adapt = manifest["adapters"][fold + "-" + core]
    if base["source_sha256"] != adapt["parent_sha256"]:
        raise ValueError("Mismatched public parent")
    for info in (base, adapt):
        if digest(folder / info["file"]) != info["sha256"]:
            raise ValueError("Package checksum mismatch")
    meta, p = load(folder / base["file"])
    s = Spec(**meta["spec"])
    if meta["parent_sha256"] != base["source_sha256"]:
        raise ValueError("Wrong backbone lineage")
    with np.load(folder / adapt["file"], allow_pickle=False) as f:
        d = {k: jnp.asarray(f[k], jnp.float32) for k in f.files}
    with np.load(case, allow_pickle=False) as f:
        r = {}
        for k, shape in SIZES.items():
            optional = k not in ("a", "v", "n")
            if k not in f and not optional:
                raise ValueError("Missing observed sensor " + k)
            x = f[k] if k in f else np.zeros(shape, "f4")
            if x.shape != shape or not np.isfinite(x).all():
                raise ValueError("Wrong or nonfinite signal " + k)
            r[k] = x
    text = None
    if audio is not None:
        if mode in ("human", "wrong_human") and transcript is None:
            raise ValueError(
                "Provide the completed human transcript for a new waveform in this condition"
            )
        if mode in ("small", "wrong_small"):
            raise ValueError(
                "The stronger-recognizer condition requires its prepared feature"
            )
        ac, aw, text = real_audio(audio, asr_dir)
        wrong = mode.startswith("wrong")
        r["wc" if wrong else "ac"] = ac
        r["ww" if wrong else "aw"] = aw
        if transcript is not None:
            r["wh" if wrong else "ah"] = text_features(transcript)
    r = {k: jnp.asarray(v[None]) for k, v in r.items()}
    out = predict(jax.tree.map(jnp.asarray, p), d, s, r, mode)
    voc = json.loads((folder / (fold + "-vocabulary.json")).read_text())
    logits = np.asarray(out["verb"])[0, : len(voc["verb"])]
    pr = np.exp(logits - logits.max())
    pr /= pr.sum()
    ix = pr.argsort()[::-1][:5]
    effect = np.asarray(out["effect"])[0]
    noun = np.asarray(out["noun"])[0, : len(voc["noun"])].argmax()
    sem = (
        np.stack([text_features(w.replace("_", " ")) for w in voc["open_noun"]]).astype(
            np.float32
        )
        / 127
    )
    sem /= np.maximum(1e-7, np.linalg.norm(sem, axis=1, keepdims=True))
    return dict(
        fold=fold,
        core=core,
        instruction=mode,
        recognized_new_audio=text,
        source_parent=base["source_sha256"],
        source_adapter=adapt["source_sha256"],
        ranked_next_verbs=[
            dict(verb=voc["verb"][i], probability=float(pr[i])) for i in ix
        ],
        training_vocabulary_noun=voc["noun"][noun],
        semantic_noun_candidate=voc["open_noun"][
            int((sem @ np.asarray(out["semantic"])[0]).argmax())
        ],
        forecast=dict(
            local_changed_fraction=float(effect[:2].sum()),
            nearer_fraction=float(effect[0]),
            farther_fraction=float(effect[1]),
            mean_abs_change_mm=float(effect[2] * 100),
        ),
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("bundle", type=Path)
    a.add_argument("case", type=Path)
    a.add_argument("--fold", choices=("A", "B"), required=True)
    a.add_argument("--core", choices=("hybrid", "gpt2", "gated"), default="hybrid")
    a.add_argument("--instruction", choices=MODES, default="correct")
    a.add_argument("--speech-wav", type=Path)
    a.add_argument("--asr-dir", type=Path)
    a.add_argument("--transcript")
    a.add_argument("--output", type=Path)
    args = a.parse_args()
    doc = run(
        args.bundle,
        args.case,
        args.fold,
        args.core,
        args.instruction,
        args.speech_wav,
        args.asr_dir,
        args.transcript,
    )
    txt = json.dumps(doc, indent=2) + "\n"
    if args.output:
        args.output.write_text(txt)
    else:
        print(txt, end="")
